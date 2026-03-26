import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "flash_attn_3"

import json
import math
import torch
import trimesh
import o_voxel
import numpy as np
import torch.nn as nn
import trellis2.modules.sparse as sp

from PIL import Image
from tqdm import tqdm
from trellis2 import models
from collections import OrderedDict
from transformers import Mistral3ForConditionalGeneration
from trellis2.pipelines.rembg import BiRefNet
from trellis2.representations import MeshWithVoxel
from data_toolkit.bpy_render import render_from_transforms
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

try:
    import cv2
    import nvdiffrast.torch as nr
    from flex_gemm.ops.grid_sample import grid_sample_3d
    BAKE_ENABLED = True
except ImportError as e:
    print(f"[SegviGen] Warning: Texture baking dependencies NOT found ({e}). Skipping bake mode.")
    BAKE_ENABLED = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRELLIS_PIPELINE_JSON = os.path.join(BASE_DIR, "data_toolkit/texturing_pipeline.json")
TRELLIS_TEX_FLOW = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16"
TRELLIS_SHAPE_ENC = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
TRELLIS_TEX_ENC = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
TRELLIS_SHAPE_DEC = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
TRELLIS_TEX_DEC = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"
DINO_PATH = "Aero-Ex/Dinov3"
import folder_paths
COMFY_MODELS_DIR = folder_paths.models_dir


EARLY_SIMPLIFY_ENABLED = True
EARLY_SIMPLIFY_TARGET_FACES = 120000
EARLY_SIMPLIFY_AGGRESSION = 2

MAX_PREPROCESS_TEX_SIZE = 1024
EXPORT_TEXTURE_SIZE = 2048


def _scene_to_single_mesh(asset):
    if isinstance(asset, trimesh.Scene):
        mesh = asset.to_mesh()
    elif isinstance(asset, trimesh.Trimesh):
        mesh = asset
    else:
        raise TypeError(f"Unsupported asset type: {type(asset)}")
    if mesh is None or (hasattr(mesh, 'faces') and len(mesh.faces) == 0):
        # Fallback for empty geometries
        return mesh
    return mesh


def make_texture_square_pow2(img: Image.Image, target_size=None, max_size=1024):
    w, h = img.size
    max_side = max(w, h)
    pow2 = 1
    while pow2 < max_side:
        pow2 *= 2
    if target_size is not None:
        pow2 = target_size
    pow2 = min(pow2, max_size)
    return img.resize((pow2, pow2), Image.BILINEAR)


def preprocess_scene_textures(asset, max_texture_size=1024):
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


def get_work_glb_path(item):
    base, _ = os.path.splitext(item["input_vxz"])
    return f"{base}_work.glb"


def build_simplified_work_glb(
    input_glb_path,
    output_glb_path,
    target_faces=EARLY_SIMPLIFY_TARGET_FACES,
    aggression=EARLY_SIMPLIFY_AGGRESSION,
):
    """
    Build a simplified geometry-only GLB for early rendering and shape processing.
    This keeps the original GLB untouched for texture/attribute extraction.
    """
    os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)

    asset = trimesh.load(input_glb_path, force="scene", process=False)
    mesh = _scene_to_single_mesh(asset)
    src_faces = int(len(mesh.faces))

    if src_faces <= target_faces:
        mesh = _apply_neutral_visual(mesh.copy())
        mesh.export(output_glb_path)
        print(f"[Simplify] Skip: faces={src_faces} <= target={target_faces}")
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
        print(f"[Simplify] faces: {src_faces} -> {dst_faces}")
        return output_glb_path, src_faces, dst_faces
    except Exception as e:
        print(f"[Simplify] Failed, fallback to original mesh: {e}")
        mesh = _apply_neutral_visual(mesh.copy())

def generate_2d_map_from_glb(glb_path, transforms_path, out_img_path, render_img_path=None):
    """
    Render the GLB first, then generate a 2D segmentation map with FLUX2.
    """
    PIPE.load_all_models()

    if render_img_path is None:
        base, _ = os.path.splitext(out_img_path)
        render_img_path = f"{base}_render.png"

    render_from_transforms(glb_path, transforms_path, render_img_path)

    prompt = "Apply distinct colors to different regions of this image"

    render_img = Image.open(render_img_path).convert("RGB")
    if max(render_img.size) > 768:
        scale = 768 / max(render_img.size)
        render_img = render_img.resize(
            (int(render_img.width * scale), int(render_img.height * scale)),
            Image.Resampling.LANCZOS,
        )

    torch.cuda.empty_cache()

    # Flux generation removed due to missing pipeline
    # image = PIPE.flux2(
    #     prompt=prompt,
    #     image=render_img,
    #     num_inference_steps=4,
    # ).images[0]
    
    # image.save(out_img_path)
    print("[Warning] 2D map generation with FLUX is currently disabled.")
    return None 


def _colorvisuals_to_texturevisuals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Convert ColorVisuals to TextureVisuals by baking per-face colors into a tiny atlas
    and generating per-face UVs. Ensure the resulting material is PBRMaterial to satisfy
    downstream GLTF/PBR-only pipelines.
    """
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

    out = trimesh.Trimesh(vertices=v_new, faces=f_new, visual=visual, process=False)
    return out


def ensure_texture_visuals(asset):
    """
    Ensure all geometries in a Scene (or a single Trimesh) use TextureVisuals.
    For ColorVisuals, we bake them into a synthetic atlas.
    """
    if isinstance(asset, trimesh.Scene):
        for geom_name, g in list(asset.geometry.items()):
            if isinstance(g, trimesh.Trimesh):
                asset.geometry[geom_name] = _colorvisuals_to_texturevisuals(g)
        return asset

    if isinstance(asset, trimesh.Trimesh):
        return _colorvisuals_to_texturevisuals(asset)

    return asset


class Sampler:
    def _inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, cond):
        t = torch.tensor([t * 1000] * x_t.shape[0], dtype=torch.float32).cuda()
        return model(x_t, tex_slat, shape_slat, t, cond, coords_len_list)

    def guidance_inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, guidance_strength, guidance_rescale=0.0):
        if guidance_strength == 1:
            return self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['cond'])
        elif guidance_strength == 0:
            return self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['neg_cond'])
        else:
            pred_pos = self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['cond'])
            pred_neg = self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['neg_cond'])
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)
            return pred

    def interval_inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, sampler_params):
        guidance_strength = sampler_params['guidance_strength']
        guidance_interval = sampler_params['guidance_interval']
        guidance_rescale = sampler_params['guidance_rescale']
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, guidance_strength, guidance_rescale)
        else:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, 1, guidance_rescale)

    @torch.no_grad()
    def sample_once(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, t_prev, cond_dict, sampler_params):
        pred_v = self.interval_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, sampler_params)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return pred_x_prev

    @torch.no_grad()
    def sample(self, model, noise, tex_slat, shape_slat, coords_len_list, cond_dict, sampler_params):
        sample = noise
        steps = sampler_params['steps']
        rescale_t = sampler_params['rescale_t']
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for t, t_prev in tqdm(t_pairs, desc="Sampling"):
            sample = self.sample_once(model, sample, tex_slat, shape_slat, coords_len_list, t, t_prev, cond_dict, sampler_params)
        return sample


class Gen3DSeg(nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x_t, tex_slats, shape_slats, t, cond, coords_len_list):
        input_tex_feats_list = []
        input_tex_coords_list = []
        shape_feats_list = []
        shape_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            input_tex_feats_list.append(x_t.feats[begin:end])
            input_tex_feats_list.append(tex_slats.feats[begin:end])
            input_tex_coords_list.append(x_t.coords[begin:end])
            input_tex_coords_list.append(tex_slats.coords[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            begin = end
        x_t = sp.SparseTensor(torch.cat(input_tex_feats_list), torch.cat(input_tex_coords_list))
        shape_slats = sp.SparseTensor(torch.cat(shape_feats_list), torch.cat(shape_coords_list))

        output_tex_slats = self.flow_model(x_t, t, cond, shape_slats)

        output_tex_feats_list = []
        output_tex_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            output_tex_feats_list.append(output_tex_slats.feats[begin:end])
            output_tex_coords_list.append(output_tex_slats.coords[begin:end])
            begin = begin + 2 * coords_len
        output_tex_slat = sp.SparseTensor(torch.cat(output_tex_feats_list), torch.cat(output_tex_coords_list))
        return output_tex_slat


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


def process_glb_to_vxz(glb_path, vxz_path, shape_glb_path=None):
    """
    Use the original GLB for texture/material attributes,
    and an optional simplified GLB for the geometry-heavy shape branch.
    """
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
        vertices,
        faces,
        grid_size=512,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False,
    )
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]

    # Material Voxelization (using full scene asset for correct attribute mapping)
    voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
        tex_asset, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], timing=False
    )
    vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
    mapping_mat = torch.argsort(vid_mat)
    attributes = {k: v[mapping_mat] for k, v in attributes.items()}

    # Quantize dual_vertices and intersection flags
    dual_vertices = dual_vertices * 512 - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

    attributes.update({'dual_vertices': dual_vertices})
    attributes.update({'intersected': intersected})
    
    os.makedirs(os.path.dirname(vxz_path), exist_ok=True)
    o_voxel.io.write(vxz_path, voxel_indices, attributes)
    print(f"[SegviGen] Vxz: done")


def vxz_to_latent_slat(shape_encoder, shape_decoder, tex_encoder, vxz_path):
    device = next(shape_encoder.parameters()).device
    coords, data = o_voxel.io.read(vxz_path)
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1).to(device)
    vertices = (data['dual_vertices'].to(device) / 255)
    intersected = torch.cat([data['intersected'] % 2, data['intersected'] // 2 % 2, data['intersected'] // 4 % 2], dim=-1).bool().to(device)
    vertices_sparse = sp.SparseTensor(vertices, coords)
    intersected_sparse = sp.SparseTensor(intersected.float(), coords)
    with torch.no_grad():
        shape_slat = shape_encoder(vertices_sparse, intersected_sparse)
        shape_slat = sp.SparseTensor(shape_slat.feats.to(device), shape_slat.coords.to(device))
        shape_decoder.set_resolution(512)
        meshes, subs = shape_decoder(shape_slat, return_subs=True)
        try:
            for i, m in enumerate(meshes):
                v_min = m.vertices.min(dim=0).values.tolist()
                v_max = m.vertices.max(dim=0).values.tolist()
                print(f"[Log - Decoder Output] Component {i} bounds: {v_min} to {v_max}")
        except Exception:
            pass

    base_color = (data['base_color'] / 255)
    metallic = (data['metallic'] / 255)
    roughness = (data['roughness'] / 255)
    alpha = (data['alpha'] / 255)
    attr = torch.cat([base_color, metallic, roughness, alpha], dim=-1).float().to(device) * 2 - 1
    with torch.no_grad():
        tex_slat = tex_encoder(sp.SparseTensor(attr, coords))
    return shape_slat, meshes, subs, tex_slat


def preprocess_image(rembg_model, input):
    if input.mode != "RGB":
        bg = Image.new("RGB", input.size, (255, 255, 255))
        bg.paste(input, mask=input.split()[3])
        input = bg
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        output = rembg_model(input)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output


def get_cond(image_cond_model, image):
    image_cond_model.image_size = 512
    cond = image_cond_model(image)
    neg_cond = torch.zeros_like(cond)
    return {'cond': cond, 'neg_cond': neg_cond}


def tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, input_tex_slat, cond_dict, steps=None, cfg_scale=None, seed=None):
    device = shape_slat.feats.device
    shape_std = torch.tensor(pipeline_args['shape_slat_normalization']['std'])[None].to(device)
    shape_mean = torch.tensor(pipeline_args['shape_slat_normalization']['mean'])[None].to(device)
    tex_std = torch.tensor(pipeline_args['tex_slat_normalization']['std'])[None].to(device)
    tex_mean = torch.tensor(pipeline_args['tex_slat_normalization']['mean'])[None].to(device)
    shape_slat = ((shape_slat - shape_mean) / shape_std)
    input_tex_slat = ((input_tex_slat - tex_mean) / tex_std)
    coords_len_list = [shape_slat.coords.shape[0]]
    
    # Handle seed with generator
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise_feats = torch.randn(input_tex_slat.feats.shape, generator=generator, device=device)
    else:
        noise_feats = torch.randn_like(input_tex_slat.feats)
    
    noise = sp.SparseTensor(noise_feats, shape_slat.coords)
    
    # Update sampler params if provided
    params = pipeline_args['tex_slat_sampler']['params'].copy()
    if steps is not None:
        params['steps'] = steps
    if cfg_scale is not None:
        params['cfg_scale'] = cfg_scale
    # Note: the sampler's .sample method might not take a 'seed' key in its internal params, 
    # but the noise already handles it. We pass it anyway if it is used internally.
    if seed is not None:
        params['seed'] = seed
        
    output_tex_slat = sampler.sample(gen3dseg, noise, input_tex_slat, shape_slat, coords_len_list, cond_dict, params)
    output_tex_slat = output_tex_slat * tex_std + tex_mean
    return output_tex_slat


def make_texture_square_pow2(img: Image.Image, target_size=None, max_size=1024):
    w, h = img.size
    max_side = max(w, h)
    pow2 = 1
    while pow2 < max_side:
        pow2 *= 2
    if target_size is not None:
        pow2 = target_size
    pow2 = min(pow2, max_size)
    return img.resize((pow2, pow2), Image.BILINEAR)


def bake_to_mesh(glb_path, tex_voxels, output_path, resolution=512, texture_size=2048, generate_uv=False):
    """
    Bake texture from voxels onto an existing GLB mesh using the same logic as to_glb.
    """
    if not BAKE_ENABLED:
        raise RuntimeError("Texture baking is not enabled (missing dependencies).")

    print(f"[SegviGen] Bake: baking onto {glb_path} -> {output_path}")
    asset = trimesh.load(glb_path, force='scene')
    device = torch.device("cuda")
    
    # Prepare the attribute volume (sparse) as to_glb expects
    if isinstance(tex_voxels, list):
        attr_volume = torch.cat([vox.feats for vox in tex_voxels]).to(device)
        attr_coords = torch.cat([vox.coords for vox in tex_voxels]).to(device)[:, -3:] # <--- Take only (Z, Y, X)
    else:
        attr_volume = tex_voxels.feats.to(device)
        attr_coords = tex_voxels.coords.to(device)[:, -3:]
    
    print(f"[SegviGen] Bake Debug: attr_volume shape={attr_volume.shape}, coords bounds={attr_coords.min(dim=0)[0].tolist()} to {attr_coords.max(dim=0)[0].tolist()}")
    
    pbr_attr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }

    # [AI Voxel Data Logging]
    num_voxels = attr_volume.shape[0]
    print(f"\n[AI Voxel Data] Found {num_voxels} voxels generated by AI.")
    
    # Log a sample of 10 voxels (Position -> Color)
    print("[AI Voxel Data] Sample Voxels (Grid Pos -> RGB):")
    sample_indices = torch.linspace(0, num_voxels - 1, 10).long()
    for idx in sample_indices:
        pos = attr_coords[idx].tolist()
        # Extract RGB from base_color slice (multiplied by 255)
        color = (attr_volume[idx, pbr_attr_layout['base_color']] * 255).int().tolist()
        print(f"  - Pos {pos} -> RGB {color}")
    print("-" * 50)

    # Scale voxels to match the slat to glb Unit Cube AABB
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    # Calculate normalization (must match process_glb_to_vxz)
    full_aabb = asset.bounding_box.bounds
    center = (full_aabb[0] + full_aabb[1]) / 2
    max_side = (full_aabb[1] - full_aabb[0]).max()
    scale = 0.99999 / max_side

    # Iterate over nodes to handle scene graph transforms
    for node_name in asset.graph.nodes_geometry:
        transform, geom_name = asset.graph[node_name]
        geom = asset.geometry[geom_name]
        
        # Robust UV check
        has_uv = False
        if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
            has_uv = True
        elif isinstance(geom.visual, trimesh.visual.color.ColorVisuals):
            # If it's color visuals, it definitely has no UVs. 
            # We must generate them or skip.
            pass
        elif hasattr(geom, 'texture_visuals') or hasattr(geom.visual, 'material'):
            # Sometimes UVs are hidden in material or other attributes in some trimesh versions
            try:
                geom.visual = geom.visual.to_texture()
                has_uv = hasattr(geom.visual, 'uv') and geom.visual.uv is not None
            except Exception:
                pass

        if not generate_uv and not has_uv:
            print(f"[SegviGen] Bake Warning: Skipping {node_name} - No UVs found (Visual type: {type(geom.visual)}).")
            continue
            
        print(f"[SegviGen] Bake: sampling {node_name}...")
        
        # Apply normalization to vertices for sampling (must use World Space)
        # 1. Transform local vertices to world space
        world_vertices = trimesh.transformations.transform_points(geom.vertices, transform)
        
        # 2. Normalize world vertices to fit AI unit cube [-0.5, 0.5]
        norm_vertices = (world_vertices - center) * scale
        

        # LOGS FOR USER:
        v_orig_min = world_vertices.min(axis=0).tolist()
        v_orig_max = world_vertices.max(axis=0).tolist()
        v_norm_min = norm_vertices.min(axis=0).tolist()
        v_norm_max = norm_vertices.max(axis=0).tolist()
        print(f"[Log - Bake Input] {node_name}: Original Bounds={v_orig_min} to {v_orig_max}")
        print(f"[Log - Bake Input] {node_name}: Normalized Bounds={v_norm_min} to {v_norm_max} (Target: [-0.5, 0.5])")
        
        # Use to_glb with sparse input
        baked_mesh = o_voxel.postprocess.to_glb(
            vertices=torch.from_numpy(norm_vertices).float().to(device).contiguous(),
            faces=torch.from_numpy(geom.faces).int().to(device).contiguous(),
            attr_volume=attr_volume,
            coords=attr_coords, 
            attr_layout=pbr_attr_layout,
            aabb=aabb,
            voxel_size=1.0 / resolution,
            texture_size=texture_size,
            remesh=False, 
            verbose=False
        )
        
        if isinstance(baked_mesh, trimesh.Scene):
            baked_mesh = list(baked_mesh.geometry.values())[0]

        # 1. Reverse the axis swap/invert from to_glb (Y->Z, Z->-Y)
        # Internal to_glb: v[:, 1], v[:, 2] = v[:, 2], -v[:, 1]
        y_glb = baked_mesh.vertices[:, 1].copy()
        z_glb = baked_mesh.vertices[:, 2].copy()
        baked_mesh.vertices[:, 1] = -z_glb # restore original Y
        baked_mesh.vertices[:, 2] = y_glb  # restore original Z

        # 2. Denormalize output vertices to match original GLB coordinates (Back to World Space)
        world_baked_vertices = (baked_mesh.vertices / scale) + center
        
        # 3. Transform back to LOCAL space for replacement in the original node
        inv_transform = np.linalg.inv(transform)
        baked_mesh.vertices = trimesh.transformations.transform_points(world_baked_vertices, inv_transform)

        # [Comparison: AI Voxel vs Mesh Vertex]
        print(f"[Comparison] Picking 3 sample vertices from {node_name} to verify alignment:")
        v_indices_sample = torch.linspace(0, len(geom.vertices) - 1, 3).long()
        v_tensor_sample = torch.from_numpy(norm_vertices).float().to(device)
        
        # --- Manual Trilinear Sampling (Exact Match) ---
        # 1. Prepare original coords (Z, Y, X) for axis detection
        orig_attr_coords_3 = torch.cat([vox.coords for vox in tex_voxels]).to(device)[:, -3:]
        
        # 2. Detect axis mapping by comparing ranges
        mesh_ranges = norm_vertices.max(axis=0) - norm_vertices.min(axis=0)
        vox_ranges = (orig_attr_coords_3.max(dim=0)[0] - orig_attr_coords_3.min(dim=0)[0]).float().cpu().numpy()
        m_sort = np.argsort(mesh_ranges)
        v_sort = np.argsort(vox_ranges)
        axis_map = {m_idx: v_idx for m_idx, v_idx in zip(m_sort, v_sort)}

        for v_idx_sample in v_indices_sample:
            v_idx_sample = v_idx_sample.item()
            v_pos_sample = v_tensor_sample[v_idx_sample].cpu().numpy() # [3] (X, Y, Z normalized)
            
            # Map mesh units to float grid positions
            grid_pos = np.zeros(3)
            for m_i, v_i in axis_map.items():
                grid_pos[v_i] = (v_pos_sample[m_i] + 0.5) * (resolution - 1)
            
            # Manual Trilinear Interpolation
            # Get 8 neighbors (floor/ceil)
            p0 = np.floor(grid_pos).astype(int)
            p1 = np.clip(p0 + 1, 0, resolution - 1)
            d = grid_pos - p0 # fractional parts
            
            # Sparse Lookup for 8 corners
            def get_vox_color(pos):
                pos_t = torch.from_numpy(pos).to(device)
                mask = torch.all(torch.eq(orig_attr_coords_3, pos_t), dim=1)
                if not torch.any(mask): return torch.zeros(3).to(device)
                idx = torch.where(mask)[0][0]
                return attr_volume[idx, pbr_attr_layout['base_color']]

            # Sample 8 corners [z, y, x]
            c000 = get_vox_color(p0)
            c001 = get_vox_color(np.array([p0[0], p0[1], p1[2]]))
            c010 = get_vox_color(np.array([p0[0], p1[1], p0[2]]))
            c011 = get_vox_color(np.array([p0[0], p1[1], p1[2]]))
            c100 = get_vox_color(np.array([p1[0], p0[1], p0[2]]))
            c101 = get_vox_color(np.array([p1[0], p0[1], p1[2]]))
            c110 = get_vox_color(np.array([p1[0], p1[1], p0[2]]))
            c111 = get_vox_color(p1)

            # Lerp across 3 axes
            # x-axis
            c00 = c000 * (1 - d[2]) + c001 * d[2]
            c01 = c010 * (1 - d[2]) + c011 * d[2]
            c10 = c100 * (1 - d[2]) + c101 * d[2]
            c11 = c110 * (1 - d[2]) + c111 * d[2]
            # y-axis
            c0 = c00 * (1 - d[1]) + c01 * d[1]
            c1 = c10 * (1 - d[1]) + c11 * d[1]
            # z-axis
            final_c = c0 * (1 - d[0]) + c1 * d[0]
            
            near_color = (final_c * 255).int().tolist()
            
            # --- Mesh Side: Find nearest vertex on baked mesh (Topology-Aware) ---
            v_orig = world_vertices[v_idx_sample] # Original world pos
            # baked_mesh.vertices is LOCAL here, so use world_baked_vertices for comparison
            dist_baked_all = np.linalg.norm(world_baked_vertices - v_orig, axis=1)
            v_idx_baked = np.argmin(dist_baked_all)
            match_dist = dist_baked_all[v_idx_baked]
            
            # Sample the Baked Color from the finished mesh
            baked_color = "N/A"
            try:
                img = None
                vis = baked_mesh.visual
                def find_image(obj, depth=0):
                    if depth > 2: return None
                    if hasattr(obj, 'image') and obj.image is not None: return obj.image
                    if hasattr(obj, 'diffuse') and obj.diffuse is not None: return obj.diffuse
                    for attr in ['baseColorTexture', 'base_color_texture']:
                        if hasattr(obj, attr):
                            sub = getattr(obj, attr)
                            if hasattr(sub, 'image'): return sub.image
                            if isinstance(sub, Image.Image): return sub
                    return None

                img = find_image(vis)
                if img is None and hasattr(vis, 'material'):
                    img = find_image(vis.material)
                
                if img is not None:
                    u, v_coord = baked_mesh.visual.uv[v_idx_baked]
                    w, h = img.size
                    px, py = int(u * (w-1)), int((1.0 - v_coord) * (h-1))
                    baked_color = list(img.getpixel((px, py))[:3])
            except Exception:
                pass
            
            print(f"  - Vertex {v_idx_sample} (Match Dist: {match_dist:.4f})")
            print(f"    -> AI Expected Color: RGB {near_color}")
            print(f"    -> Baked Color on Mesh: RGB {baked_color}")
        print("-" * 50)
        
        v_final_max = world_baked_vertices.max(axis=0).tolist()
        v_final_min = world_baked_vertices.min(axis=0).tolist()
        print(f"[Log - Bake Output] {node_name}: Final Denormalized (World) Bounds={v_final_min} to {v_final_max}")
        
        # Create a unique geometry for this node if we want to preserve instances with different bakes
        # However, for simple replacement:
        asset.geometry[geom_name] = baked_mesh

    asset.export(output_path)
    print(f"[SegviGen] Bake: saved to {output_path}")
    return output_path


def slat_to_glb(meshes, tex_voxels, resolution=512):
    pbr_attr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }
    out_mesh = []
    for i, (m, v) in enumerate(zip(meshes, tex_voxels)):
        try:
            v_min = m.vertices.min(dim=0).values.tolist()
            v_max = m.vertices.max(dim=0).values.tolist()
            print(f"[Log - Slat Input] Component {i}: vertices={len(m.vertices)}, bounds={v_min} to {v_max}")
        except Exception:
            pass
        out_mesh.append(
            MeshWithVoxel(
                m.vertices,
                m.faces,
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / resolution,
                coords=v.coords[:, 1:],
                attrs=v.feats,
                voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                layout=pbr_attr_layout,
            )
        )
    mesh = out_mesh[0]

    try:
        mesh.simplify(200000)
    except Exception as e:
        print(f"[Export] mesh.simplify skipped: {e}")

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs.cuda(),
        coords=mesh.coords.cuda(),
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=50000,
        texture_size=EXPORT_TEXTURE_SIZE,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    return glb


class _LoadedPipeline:
    def __init__(self):
        self.loaded = False
        self.current_ckpt = None

        self.pipeline_args = None
        self.tex_slat_flow_model = None
        self.gen3dseg = None
        self.sampler = None

        self.shape_encoder = None
        self.tex_encoder = None
        self.shape_decoder = None
        self.tex_decoder = None

        self.rembg_model = None
        self.image_cond_model = None

    def load_all_models(self):
        if self.loaded:
            return

        print("-" * 100)
        print("[Init] Loading pipeline config .....................")
        with open(TRELLIS_PIPELINE_JSON, "r") as f:
            pipeline_config = json.load(f)
        self.pipeline_args = pipeline_config['args']
        self.sampler = Sampler()
        self.loaded = True
        print("[Init] Config loaded. Models will be loaded on-demand and unloaded to save RAM.")

    def _download_if_missing(self, repo_id, local_root, subfolder=None):
        # Target dir is where we expect the files to actually exist
        target_dir = os.path.join(local_root, subfolder) if subfolder else local_root
        
        # Check if target dir exists and is not empty
        if os.path.exists(target_dir) and os.listdir(target_dir):
            # Already exists locally, skip download
            return target_dir

        from huggingface_hub import snapshot_download
        parts = repo_id.split("/")
        actual_repo_id = "/".join(parts[:2])
        
        # If repo_id had more than 2 parts, the rest is the subfolder pattern
        if subfolder is None and len(parts) > 2:
            subfolder = "/".join(parts[2:])

        print(f"[SegviGen] Downloading {actual_repo_id} (subfolder: {subfolder}) to {local_root}...")
        os.makedirs(local_root, exist_ok=True)
        
        snapshot_download(
            repo_id=actual_repo_id, 
            local_dir=local_root, 
            allow_patterns=[f"{subfolder}/*"] if subfolder else None
        )
        return target_dir

    def get_rembg(self):
        if self.rembg_model is None:
            print(f"[OnDemand] Loading RMBG-2.0")
            target_path = os.path.join(COMFY_MODELS_DIR, "Aero-Ex", "RMBG-2.0")
            self._download_if_missing("Aero-Ex/RMBG-2.0", target_path)
            self.rembg_model = BiRefNet(model_name=target_path)
            self.rembg_model.eval()
        return self.rembg_model

    def get_cond_model(self):
        if self.image_cond_model is None:
            print(f"[OnDemand] Loading DinoV3 conditioners")
            # repo Aero-Ex/Dinov3 has subfolder facebook/dinov3-vitl16-pretrain-lvd1689m
            base_dir = os.path.join(COMFY_MODELS_DIR, "facebook", "dinov3-vitl16-pretrain-lvd1689m")
            sub = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            self._download_if_missing(DINO_PATH, base_dir, subfolder=sub)
            
            # The files will be in base_dir/sub
            actual_local_dir = os.path.join(base_dir, sub)
            self.image_cond_model = DinoV3FeatureExtractor(DINO_PATH, local_dir=actual_local_dir, subfolder=sub)
            self.image_cond_model.eval()
        return self.image_cond_model

    def get_encoders_decoder(self):
        # Load encoders and shape decoder together as they are used in vxz_to_latent_slat
        if self.shape_encoder is None:
            print(f"[OnDemand] Loading Encoders and Shape Decoder")
            
            # Download to COMFY_MODELS_DIR root; snapshot_download keeps subfolder structure
            enc_path = self._download_if_missing(TRELLIS_SHAPE_ENC, COMFY_MODELS_DIR)
            tex_enc_path = self._download_if_missing(TRELLIS_TEX_ENC, COMFY_MODELS_DIR)
            dec_path = self._download_if_missing(TRELLIS_SHAPE_DEC, COMFY_MODELS_DIR)
            
            self.shape_encoder = models.from_pretrained(TRELLIS_SHAPE_ENC, local_dir=enc_path).eval()
            self.tex_encoder = models.from_pretrained(TRELLIS_TEX_ENC, local_dir=tex_enc_path).eval()
            self.shape_decoder = models.from_pretrained(TRELLIS_SHAPE_DEC, local_dir=dec_path).eval()
        return self.shape_encoder, self.tex_encoder, self.shape_decoder

    def get_gen3dseg(self):
        if self.gen3dseg is None:
            print(f"[OnDemand] Loading Backbone (Gen3DSeg)")
            flow_path = self._download_if_missing(TRELLIS_TEX_FLOW, COMFY_MODELS_DIR)
            
            self.tex_slat_flow_model = models.from_pretrained(TRELLIS_TEX_FLOW, local_dir=flow_path)
            self.gen3dseg = Gen3DSeg(self.tex_slat_flow_model)
            
            local_ckpt = self.current_ckpt
            if local_ckpt is not None:
                print(f"[OnDemand] Applying deferred checkpoint: {local_ckpt}")
                if local_ckpt.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(local_ckpt)
                else:
                    # Use weights_only=False for legacy .ckpt files if on newer torch
                    try:
                        state_dict = torch.load(local_ckpt, weights_only=False)
                    except TypeError:
                        state_dict = torch.load(local_ckpt)
                    
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                
                state_dict = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in state_dict.items()])
                self.gen3dseg.load_state_dict(state_dict)
            
            self.gen3dseg.eval()
        return self.gen3dseg

    def get_tex_decoder(self):
        if self.tex_decoder is None:
            print(f"[OnDemand] Loading Texture Decoder")
            dec_path = self._download_if_missing(TRELLIS_TEX_DEC, COMFY_MODELS_DIR)
            self.tex_decoder = models.from_pretrained(TRELLIS_TEX_DEC, local_dir=dec_path).eval()
        return self.tex_decoder

    def unload(self, *attr_names):
        import gc
        for name in attr_names:
            val = getattr(self, name, None)
            if val is not None:
                if hasattr(val, 'cpu'): val.cpu()
                setattr(self, name, None)
        gc.collect()
        torch.cuda.empty_cache()

    def load_ckpt_if_needed(self, ckpt_path: str):
        if self.current_ckpt == ckpt_path:
            return
        self.current_ckpt = ckpt_path
        print(f"[OnDemand] Checkpoint set to: {ckpt_path}. Will be applied when model is loaded.")


PIPE = _LoadedPipeline()


def inference_with_loaded_models(ckpt_path, item):
    PIPE.load_all_models()
    PIPE.load_ckpt_if_needed(ckpt_path)

    work_glb = item["glb"]
    if EARLY_SIMPLIFY_ENABLED:
        work_glb = get_work_glb_path(item)
        build_simplified_work_glb(
            input_glb_path=item["glb"],
            output_glb_path=work_glb,
            target_faces=EARLY_SIMPLIFY_TARGET_FACES,
            aggression=EARLY_SIMPLIFY_AGGRESSION,
        )

    if not item["2d_map"]:
        generate_2d_map_from_glb(
            glb_path=work_glb,
            transforms_path=item["transforms"],
            out_img_path=item["img"],
        )

    process_glb_to_vxz(
        glb_path=item["glb"],
        vxz_path=item["input_vxz"],
    )

    image = Image.open(item["img"])
    
    # 1. Background removal
    rembg_model = PIPE.get_rembg()
    rembg_model.cuda()
    image = preprocess_image(rembg_model, image)
    PIPE.unload('rembg_model')

    # 2. Condition generation
    cond_model = PIPE.get_cond_model()
    cond_model.cuda()
    cond = get_cond(cond_model, [image])
    # Offload cond to CPU
    cond = {k: v.cpu() for k, v in cond.items()}
    PIPE.unload('image_cond_model')

    # 3. VXZ to Latent SLat (Encoders + Shape Decoder)
    shape_enc, tex_enc, shape_dec = PIPE.get_encoders_decoder()
    shape_enc.cuda()
    tex_enc.cuda()
    shape_dec.cuda()
    
    shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(
        shape_enc,
        shape_dec,
        tex_enc,
        item["input_vxz"],
    )
    # Offload products to CPU
    shape_slat = sp.SparseTensor(shape_slat.feats.cpu(), shape_slat.coords.cpu())
    tex_slat = sp.SparseTensor(tex_slat.feats.cpu(), tex_slat.coords.cpu())
    # subs are usually small indices or similar, keep them as is
    
    PIPE.unload('shape_encoder', 'tex_encoder', 'shape_decoder')

    # 4. Sampling (Backbone)
    gen3dseg = PIPE.get_gen3dseg()
    gen3dseg.cuda()
    # Move inputs to GPU for sampling
    shape_slat_gpu = sp.SparseTensor(shape_slat.feats.cuda(), shape_slat.coords.cuda())
    tex_slat_gpu = sp.SparseTensor(tex_slat.feats.cuda(), tex_slat.coords.cuda())
    cond_gpu = {k: v.cuda() for k, v in cond.items()}
    
    output_tex_slat = tex_slat_sample_single(
        gen3dseg, PIPE.sampler, PIPE.pipeline_args, shape_slat_gpu, tex_slat_gpu, cond_gpu
    )
    
    # Offload result to CPU
    output_tex_slat_cpu = sp.SparseTensor(output_tex_slat.feats.cpu(), output_tex_slat.coords.cpu())
    
    # Cleanup GPU inputs
    del shape_slat_gpu, tex_slat_gpu, cond_gpu, output_tex_slat
    PIPE.unload('gen3dseg', 'tex_slat_flow_model')

    # 5. Texture Decoding
    tex_decoder = PIPE.get_tex_decoder()
    tex_decoder.cuda()
    output_tex_slat_gpu = sp.SparseTensor(output_tex_slat_cpu.feats.cuda(), output_tex_slat_cpu.coords.cuda())
    with torch.no_grad():
        subs_gpu = [s.cuda() if isinstance(s, torch.Tensor) else s for s in subs]
        tex_voxels = tex_decoder(output_tex_slat_gpu, guide_subs=subs_gpu) * 0.5 + 0.5
        # Move result to CPU immediately
        tex_voxels = [v.cpu() for v in tex_voxels]
    
    PIPE.unload('tex_decoder')

    if item.get("bake", False):
        bake_to_mesh(
            item["glb"], 
            tex_voxels, 
            item["export_glb"], 
            resolution=512, 
            texture_size=2048,
            generate_uv=item.get("generate_uv", False)
        )
    else:
        glb = slat_to_glb(meshes, tex_voxels)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )

        if hasattr(glb, "apply_transform") and callable(getattr(glb, "apply_transform")):
            glb.apply_transform(T)
            glb.export(item["export_glb"])
        else:
            glb.export(item["export_glb"])
            scene_or_mesh = trimesh.load(item["export_glb"], force="scene")
            scene_or_mesh.apply_transform(T)
            scene_or_mesh.export(item["export_glb"])