import os
import math
import torch
import trimesh
import numpy as np
import o_voxel
from PIL import Image
import trellis2.modules.sparse as sp
from trellis2.representations import MeshWithVoxel

MAX_PREPROCESS_TEX_SIZE = 1024
EXPORT_TEXTURE_SIZE = 2048
EARLY_SIMPLIFY_TARGET_FACES = 120000
EARLY_SIMPLIFY_AGGRESSION = 2
EARLY_SIMPLIFY_ENABLED = True

def prepare_fast_simplifier():
    try:
        import fast_simplification
        return fast_simplification
    except ImportError:
        return None

def fast_simplify_mesh(input_path, output_path, target_reduction=0.7):
    """
    Simplify mesh using fast-simplification library.
    target_reduction: 0.0 to 1.0 (e.g., 0.7 means remove 70% of faces).
    """
    fast_simplification = prepare_fast_simplifier()
    if fast_simplification is None:
        print("[SegviGen] Warning: fast-simplification not found. Falling back to original mesh.")
        import shutil
        shutil.copy(input_path, output_path)
        return output_path

    asset = trimesh.load(input_path, force='scene', process=False)
    
    # Process each geometry in the scene
    for name, geom in asset.geometry.items():
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) < 100:
            continue
            
        print(f"[SegviGen] Simplifying {name}: {len(geom.faces)} faces...")
        try:
            # fast_simplification.simplify(v, f, target_reduction)
            new_v, new_f = fast_simplification.simplify(geom.vertices, geom.faces, target_reduction)
            
            # Maintain visual if possible (though simplification usually breaks it)
            # For pre-simplification, we mostly care about geometry.
            asset.geometry[name] = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
        except Exception as e:
            print(f"[SegviGen] Error simplifying {name}: {e}")
            
    asset.export(output_path)
    return output_path

def get_work_glb_path(item):
    base, _ = os.path.splitext(item["input_vxz"])
    return f"{base}_work.glb"

def _scene_to_single_mesh(asset):
    if isinstance(asset, trimesh.Scene):
        mesh = asset.to_mesh()
    elif isinstance(asset, trimesh.Trimesh):
        mesh = asset
    else:
        raise TypeError(f"Unsupported asset type: {type(asset)}")
    if mesh is None or (hasattr(mesh, 'faces') and len(mesh.faces) == 0):
        return mesh
    return mesh

def make_texture_square_pow2(img: Image.Image, target_size=None, max_size=MAX_PREPROCESS_TEX_SIZE):
    w, h = img.size
    max_side = max(w, h)
    pow2 = 1
    while pow2 < max_side:
        pow2 *= 2
    if target_size is not None:
        pow2 = target_size
    pow2 = min(pow2, max_size)
    return img.resize((pow2, pow2), Image.BILINEAR)

def preprocess_scene_textures(asset, max_texture_size=MAX_PREPROCESS_TEX_SIZE):
    if not isinstance(asset, trimesh.Scene):
        return asset
    tex_keys = ["baseColorTexture", "normalTexture", "metallicRoughnessTexture", "emissiveTexture", "occlusionTexture"]
    for geom in asset.geometry.values():
        visual = getattr(geom, "visual", None)
        mat = getattr(visual, "material", None)
        if mat is None:
            continue
        for key in tex_keys:
            if not hasattr(mat, key):
                continue
            tex = getattr(mat, key)
            if tex is None:
                continue
            if isinstance(tex, Image.Image):
                setattr(mat, key, make_texture_square_pow2(tex, max_size=max_texture_size))
            elif hasattr(tex, "image") and tex.image is not None:
                img = tex.image
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                tex.image = make_texture_square_pow2(img, max_size=max_texture_size)
        if hasattr(mat, "image") and mat.image is not None:
            img = mat.image
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            mat.image = make_texture_square_pow2(img, max_size=max_texture_size)
    return asset

def _apply_neutral_visual(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if mesh.visual is not None:
        return mesh
    face_colors = np.tile(
        np.array([[200, 200, 200, 255]], dtype=np.uint8),
        (len(mesh.faces), 1),
    )
    mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh, face_colors=face_colors)
    return mesh

def build_simplified_work_glb(
    input_glb_path,
    output_glb_path,
    target_faces=EARLY_SIMPLIFY_TARGET_FACES,
    aggression=EARLY_SIMPLIFY_AGGRESSION,
):
    os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
    asset = trimesh.load(input_glb_path, force="scene", process=False)
    mesh = _scene_to_single_mesh(asset)
    src_faces = int(len(mesh.faces))

    if src_faces <= target_faces:
        mesh = _apply_neutral_visual(mesh.copy())
        mesh.export(output_glb_path)
        return output_glb_path, src_faces, src_faces

    try:
        simplified = mesh.simplify_quadric_decimation(
            face_count=target_faces,
            aggression=aggression,
        )
        if simplified is None or len(simplified.faces) == 0:
            raise RuntimeError("simplify_quadric_decimation returned empty mesh")
        simplified = _apply_neutral_visual(simplified)
        simplified.export(output_glb_path)
        dst_faces = int(len(simplified.faces))
        return output_glb_path, src_faces, dst_faces
    except Exception as e:
        mesh = _apply_neutral_visual(mesh.copy())
        mesh.export(output_glb_path)
        return output_glb_path, src_faces, src_faces

def _colorvisuals_to_texturevisuals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if mesh.visual is None:
        return mesh
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        mat = getattr(mesh.visual, "material", None)
        if isinstance(mat, trimesh.visual.material.SimpleMaterial):
            mesh = mesh.copy()
            try:
                mesh.visual.material = mat.to_pbr()
            except Exception:
                mesh.visual.material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=mat.image
                )
        return mesh
    if not isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        return mesh

    F = int(len(mesh.faces))
    if F <= 0:
        return mesh

    face_rgba = None
    if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
        fc = np.asarray(mesh.visual.face_colors)
        if fc.ndim == 2 and fc.shape[0] == F:
            face_rgba = fc[:, :4].astype(np.uint8)

    if face_rgba is None and hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == len(mesh.vertices):
            tri = mesh.faces
            vcol = vc[tri]
            face_rgba = np.rint(vcol.mean(axis=1)).astype(np.uint8)

    if face_rgba is None:
        face_rgba = np.tile(np.array([[255, 255, 255, 255]], dtype=np.uint8), (F, 1))

    grid = int(math.ceil(math.sqrt(F)))
    img = np.zeros((grid, grid, 4), dtype=np.uint8)
    for i in range(F):
        x = i % grid
        y = i // grid
        if y >= grid:
            break
        img[y, x, :] = face_rgba[i]

    pil_img = Image.fromarray(img, mode="RGBA")
    v_new = mesh.vertices[mesh.faces].reshape(-1, 3)
    f_new = np.arange(F * 3, dtype=np.int64).reshape(F, 3)
    uv_new = np.zeros((F * 3, 2), dtype=np.float32)
    for i in range(F):
        x = i % grid
        y = i // grid
        u = (x + 0.5) / float(grid)
        v = (y + 0.5) / float(grid)
        uv_new[i * 3 : i * 3 + 3, 0] = u
        uv_new[i * 3 : i * 3 + 3, 1] = v

    pbr = trimesh.visual.material.PBRMaterial(
        baseColorTexture=pil_img,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
        alphaMode="BLEND",
    )
    visual = trimesh.visual.texture.TextureVisuals(uv=uv_new, material=pbr)
    return trimesh.Trimesh(vertices=v_new, faces=f_new, visual=visual, process=False)

def ensure_texture_visuals(asset):
    if isinstance(asset, trimesh.Scene):
        for geom_name, g in list(asset.geometry.items()):
            if isinstance(g, trimesh.Trimesh):
                asset.geometry[geom_name] = _colorvisuals_to_texturevisuals(g)
        return asset
    if isinstance(asset, trimesh.Trimesh):
        return _colorvisuals_to_texturevisuals(asset)
    return asset

def process_glb_to_vxz(glb_path, vxz_path, shape_glb_path=None):
    tex_asset = trimesh.load(glb_path, force='scene')
    tex_asset = ensure_texture_visuals(tex_asset)
    tex_asset = preprocess_scene_textures(tex_asset, max_texture_size=MAX_PREPROCESS_TEX_SIZE)

    if shape_glb_path is None:
        shape_asset = trimesh.load(glb_path, force='scene')
    else:
        shape_asset = trimesh.load(shape_glb_path, force='scene')

    aabb = tex_asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    max_side = (aabb[1] - aabb[0]).max()
    scale = 0.99999 / max_side
    print(f"[SegviGen] Vxz Calibration: AABB_MIN={aabb[0].tolist()}, AABB_MAX={aabb[1].tolist()}, Center={center.tolist()}, Scale={scale:.12f}, MaxSide={max_side:.12f}")
    
    tex_asset.apply_translation(-center)
    tex_asset.apply_scale(scale)
    shape_asset.apply_translation(-center)
    shape_asset.apply_scale(scale)

    shape_mesh = _scene_to_single_mesh(shape_asset)
    vertices = torch.from_numpy(shape_mesh.vertices).float()
    faces = torch.from_numpy(shape_mesh.faces).long()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False,
    )
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]

    voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
        tex_asset, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], timing=False
    )
    vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
    mapping_mat = torch.argsort(vid_mat)
    attributes = {k: v[mapping_mat] for k, v in attributes.items()}

    dual_vertices = dual_vertices * 512 - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

    attributes.update({'dual_vertices': dual_vertices, 'intersected': intersected})
    os.makedirs(os.path.dirname(vxz_path), exist_ok=True)
    o_voxel.io.write(vxz_path, voxel_indices, attributes)

def vxz_to_latent_slat(shape_encoder, shape_decoder, tex_encoder, vxz_path):
    device = next(shape_encoder.parameters()).device
    coords, data = o_voxel.io.read(vxz_path)
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1).to(device)
    vertices = (data['dual_vertices'].to(device) / 255)
    intersected = torch.cat([data['intersected'] % 2, data['intersected'] // 2 % 2, data['intersected'] // 4 % 2], dim=-1).bool().to(device)
    
    with torch.no_grad():
        shape_slat = shape_encoder(sp.SparseTensor(vertices, coords), sp.SparseTensor(intersected.float(), coords))
        shape_slat = sp.SparseTensor(shape_slat.feats.to(device), shape_slat.coords.to(device))
        shape_decoder.set_resolution(512)
        meshes, subs = shape_decoder(shape_slat, return_subs=True)

    attr = torch.cat([data['base_color'] / 255, data['metallic'] / 255, data['roughness'] / 255, data['alpha'] / 255], dim=-1).float().to(device) * 2 - 1
    with torch.no_grad():
        tex_slat = tex_encoder(sp.SparseTensor(attr, coords))
    return shape_slat, meshes, subs, tex_slat

def slat_to_glb(meshes, tex_voxels, resolution=512):
    pbr_attr_layout = {'base_color': slice(0, 3), 'metallic': slice(3, 4), 'roughness': slice(4, 5), 'alpha': slice(5, 6)}
    out_mesh = []
    for m, v in zip(meshes, tex_voxels):
        out_mesh.append(
            MeshWithVoxel(
                m.vertices, m.faces, origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / resolution, coords=v.coords[:, 1:], attrs=v.feats,
                voxel_shape=torch.Size([*v.shape, *v.spatial_shape]), layout=pbr_attr_layout,
            )
        )
    mesh = out_mesh[0]
    try:
        mesh.simplify(200000)
    except Exception:
        pass

    return o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs.cuda(),
        coords=mesh.coords.cuda(), attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=50000,
        texture_size=EXPORT_TEXTURE_SIZE, remesh=True, remesh_band=1, remesh_project=0, verbose=False,
    )

def bake_to_mesh(glb_path, tex_voxels, output_path, resolution=512, texture_size=2048, generate_uv=False):
    """
    Bake texture from voxels onto an existing GLB mesh using the same logic as to_glb.
    Handles scene graph transforms for accurate alignment.
    """
    print(f"[SegviGen] Bake: baking onto {glb_path} -> {output_path}")
    asset = trimesh.load(glb_path, force='scene')
    device = torch.device("cuda")
    
    # Prepare the attribute volume (sparse) as to_glb expects
    if isinstance(tex_voxels, list):
        attr_volume = torch.cat([vox.feats for vox in tex_voxels]).to(device)
        attr_coords = torch.cat([vox.coords for vox in tex_voxels]).to(device)[:, -3:]
    else:
        attr_volume = tex_voxels.feats.to(device)
        attr_coords = tex_voxels.coords.to(device)[:, -3:]
    
    pbr_attr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }

    # Calculate normalization (must match process_glb_to_vxz)
    full_aabb = asset.bounding_box.bounds
    center = (full_aabb[0] + full_aabb[1]) / 2
    max_side = (full_aabb[1] - full_aabb[0]).max()
    scale = 0.99999 / max_side

    # Iterate over nodes to handle scene graph transforms
    for node_name in asset.graph.nodes_geometry:
        transform, geom_name = asset.graph[node_name]
        geom = asset.geometry[geom_name]
        
        has_uv = hasattr(geom.visual, 'uv') and geom.visual.uv is not None
        if not generate_uv and not has_uv:
            print(f"[SegviGen] Bake Warning: Skipping {node_name} - No UVs found.")
            continue
            
        print(f"[SegviGen] Bake: sampling {node_name}...")
        
        # 1. Transform local vertices to world space
        world_vertices = trimesh.transformations.transform_points(geom.vertices, transform)
        
        # 2. Normalize world vertices to fit AI unit cube [-0.5, 0.5]
        norm_vertices = (world_vertices - center) * scale
        
        # Use to_glb with sparse input
        baked_mesh = o_voxel.postprocess.to_glb(
            vertices=torch.from_numpy(norm_vertices).float().to(device).contiguous(),
            faces=torch.from_numpy(geom.faces).int().to(device).contiguous(),
            attr_volume=attr_volume,
            coords=attr_coords, 
            attr_layout=pbr_attr_layout,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            voxel_size=1.0 / resolution,
            texture_size=texture_size,
            remesh=False, 
            verbose=False
        )
        
        if isinstance(baked_mesh, trimesh.Scene):
            baked_mesh = list(baked_mesh.geometry.values())[0]

        # 3. Reverse the axis swap/invert from to_glb (Y->Z, Z->-Y)
        y_glb = baked_mesh.vertices[:, 1].copy()
        z_glb = baked_mesh.vertices[:, 2].copy()
        baked_mesh.vertices[:, 1] = -z_glb # restore original Y
        baked_mesh.vertices[:, 2] = y_glb  # restore original Z

        # 4. Denormalize output vertices to match original GLB coordinates (Back to World Space)
        world_baked_vertices = (baked_mesh.vertices / scale) + center
        
        # 5. Transform back to LOCAL space for replacement in the original node
        inv_transform = np.linalg.inv(transform)
        baked_mesh.vertices = trimesh.transformations.transform_points(world_baked_vertices, inv_transform)

        # Replace geometry in the scene with the baked version
        asset.geometry[geom_name] = baked_mesh

    asset.export(output_path)
    print(f"[SegviGen] Bake: saved to {output_path}")
    return output_path
