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
from diffusers import Flux2Pipeline, AutoModel, Flux2KleinKVPipeline
from transformers import Mistral3ForConditionalGeneration
from trellis2.pipelines.rembg import BiRefNet
from trellis2.representations import MeshWithVoxel
from data_toolkit.bpy_render import render_from_transforms
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor


TRELLIS_PIPELINE_JSON = "data_toolkit/texturing_pipeline.json"
TRELLIS_TEX_FLOW = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16"
TRELLIS_SHAPE_ENC = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
TRELLIS_TEX_ENC = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
TRELLIS_SHAPE_DEC = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
TRELLIS_TEX_DEC = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"
DINO_PATH = "fenghora/dinov3"


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
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError("Empty mesh after loading.")
    return mesh


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
        mesh.export(output_glb_path)
        return output_glb_path, src_faces, src_faces


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

    image = PIPE.flux2(
        prompt=prompt,
        image=render_img,
        num_inference_steps=4,
    ).images[0]

    image.save(out_img_path)
    return out_img_path


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
    scale = 0.99999 / (aabb[1] - aabb[0]).max()

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

    voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
        tex_asset,
        grid_size=512,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        timing=False,
    )
    vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
    mapping_mat = torch.argsort(vid_mat)
    attributes = {k: v[mapping_mat] for k, v in attributes.items()}

    dual_vertices = dual_vertices * 512 - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

    attributes['dual_vertices'] = dual_vertices
    attributes['intersected'] = intersected
    o_voxel.io.write(vxz_path, voxel_indices, attributes)


def vxz_to_latent_slat(shape_encoder, shape_decoder, tex_encoder, vxz_path):
    coords, data = o_voxel.io.read(vxz_path)
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1).cuda()
    vertices = (data['dual_vertices'].cuda() / 255)
    intersected = torch.cat([data['intersected'] % 2, data['intersected'] // 2 % 2, data['intersected'] // 4 % 2], dim=-1).bool().cuda()
    vertices_sparse = sp.SparseTensor(vertices, coords)
    intersected_sparse = sp.SparseTensor(intersected.float(), coords)
    with torch.no_grad():
        shape_slat = shape_encoder(vertices_sparse, intersected_sparse)
        shape_slat = sp.SparseTensor(shape_slat.feats.cuda(), shape_slat.coords.cuda())
        shape_decoder.set_resolution(512)
        meshes, subs = shape_decoder(shape_slat, return_subs=True)

    base_color = (data['base_color'] / 255)
    metallic = (data['metallic'] / 255)
    roughness = (data['roughness'] / 255)
    alpha = (data['alpha'] / 255)
    attr = torch.cat([base_color, metallic, roughness, alpha], dim=-1).float().cuda() * 2 - 1
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


def tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, input_tex_slat, cond_dict):
    device = shape_slat.feats.device
    shape_std = torch.tensor(pipeline_args['shape_slat_normalization']['std'])[None].to(device)
    shape_mean = torch.tensor(pipeline_args['shape_slat_normalization']['mean'])[None].to(device)
    tex_std = torch.tensor(pipeline_args['tex_slat_normalization']['std'])[None].to(device)
    tex_mean = torch.tensor(pipeline_args['tex_slat_normalization']['mean'])[None].to(device)
    shape_slat = ((shape_slat - shape_mean) / shape_std)
    input_tex_slat = ((input_tex_slat - tex_mean) / tex_std)
    coords_len_list = [shape_slat.coords.shape[0]]
    noise = sp.SparseTensor(torch.randn_like(input_tex_slat.feats), shape_slat.coords)
    output_tex_slat = sampler.sample(gen3dseg, noise, input_tex_slat, shape_slat, coords_len_list, cond_dict, pipeline_args['tex_slat_sampler']['params'])
    output_tex_slat = output_tex_slat * tex_std + tex_mean
    return output_tex_slat


def slat_to_glb(meshes, tex_voxels, resolution=512):
    pbr_attr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }
    out_mesh = []
    for m, v in zip(meshes, tex_voxels):
        m.fill_holes()
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
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=50000,
        texture_size=EXPORT_TEXTURE_SIZE,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
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
        print("[Init] Loading pipeline config ............")
        with open(TRELLIS_PIPELINE_JSON, "r") as f:
            pipeline_config = json.load(f)
        self.pipeline_args = pipeline_config['args']

        print("-" * 100)
        print("[Init] Loading TRELLIS backbone ............")
        self.tex_slat_flow_model = models.from_pretrained(TRELLIS_TEX_FLOW)

        self.gen3dseg = Gen3DSeg(self.tex_slat_flow_model)
        self.gen3dseg.eval()
        self.gen3dseg.cuda()

        self.sampler = Sampler()

        self.shape_encoder = models.from_pretrained(TRELLIS_SHAPE_ENC).cuda().eval()
        self.tex_encoder = models.from_pretrained(TRELLIS_TEX_ENC).cuda().eval()
        self.shape_decoder = models.from_pretrained(TRELLIS_SHAPE_DEC).cuda().eval()
        self.tex_decoder = models.from_pretrained(TRELLIS_TEX_DEC).cuda().eval()

        print("-" * 100)
        print("[Init] Loading conditioners ............")

        self.rembg_model = BiRefNet(model_name="briaai/RMBG-2.0")
        self.rembg_model.cuda()

        self.image_cond_model = DinoV3FeatureExtractor(DINO_PATH)
        self.image_cond_model.cuda()

        repo_id = "black-forest-labs/FLUX.2-klein-9b-kv"
        flux2 = Flux2KleinKVPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16)
        flux2.enable_model_cpu_offload()
        self.flux2 = flux2

        self.loaded = True
        print("[Init] Done.")

    def load_ckpt_if_needed(self, ckpt_path: str):
        if self.current_ckpt == ckpt_path:
            return

        print("-" * 100)
        print(f"[CKPT] Loading ckpt: {ckpt_path}")
        state_dict = torch.load(ckpt_path)['state_dict']
        state_dict = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in state_dict.items()])
        self.gen3dseg.load_state_dict(state_dict)
        self.gen3dseg.eval()
        self.gen3dseg.cuda()
        self.current_ckpt = ckpt_path


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

    if PIPE.rembg_model is None:
        raise RuntimeError("PIPE.rembg_model is None. Check BiRefNet loading and .cuda() usage.")
    if PIPE.image_cond_model is None:
        raise RuntimeError("PIPE.image_cond_model is None. Check DinoV3FeatureExtractor loading and .cuda() usage.")

    process_glb_to_vxz(
        glb_path=item["glb"],
        vxz_path=item["input_vxz"],
    )

    image = Image.open(item["img"])
    image = preprocess_image(PIPE.rembg_model, image)
    cond = get_cond(PIPE.image_cond_model, [image])

    shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(
        PIPE.shape_encoder,
        PIPE.shape_decoder,
        PIPE.tex_encoder,
        item["input_vxz"],
    )

    output_tex_slat = tex_slat_sample_single(
        PIPE.gen3dseg, PIPE.sampler, PIPE.pipeline_args, shape_slat, tex_slat, cond
    )
    with torch.no_grad():
        tex_voxels = PIPE.tex_decoder(output_tex_slat, guide_subs=subs) * 0.5 + 0.5

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