import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import random
from types import MethodType

# ComfyUI Model Management
import comfy.model_management
import comfy.model_patcher

# Local imports
from .trellis2 import models
from .trellis2.modules import sparse as sp
from .trellis2.pipelines.rembg import BiRefNet
from .trellis2.representations import MeshWithVoxel
from .trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
from .trellis2.modules.utils import manual_cast

# from .data_toolkit.bpy_render import render_from_transforms
import o_voxel
import nvdiffrast.torch as nr
from trimesh.visual.material import PBRMaterial


class Sampler:
    def _inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, cond):
        # Derive the device from the data (already on the correct device at this point)
        device = x_t.feats.device
        t = torch.tensor([t * 1000] * x_t.shape[0], dtype=torch.float32, device=device)
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


def tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, input_tex_slat, cond_dict):
    device = shape_slat.feats.device
    shape_std = torch.tensor(pipeline_args['shape_slat_normalization']['std'], device=device)[None]
    shape_mean = torch.tensor(pipeline_args['shape_slat_normalization']['mean'], device=device)[None]
    tex_std = torch.tensor(pipeline_args['tex_slat_normalization']['std'], device=device)[None]
    tex_mean = torch.tensor(pipeline_args['tex_slat_normalization']['mean'], device=device)[None]
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
        texture_size=2048,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    return glb


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


def flow_forward_interactive(self, x, t, cond, concat_cond, point_embeds, coords_len_list):
    x = sp.sparse_cat([x, concat_cond], dim=-1)
    if isinstance(cond, list):
        cond = sp.VarLenTensor.from_tensor_list(cond)
    h = self.input_layer(x)
    h = manual_cast(h, self.dtype)
    t_emb = self.t_embedder(t)
    t_emb = self.adaLN_modulation(t_emb)
    t_emb = manual_cast(t_emb, self.dtype)
    cond = manual_cast(cond, self.dtype)
    point_embeds = manual_cast(point_embeds, self.dtype)
    h_feats_list = []
    h_coords_list = []
    begin = 0
    for i, coords_len in enumerate(coords_len_list):
        end = begin + 2 * coords_len
        h_feats_list.append(h.feats[begin:end])
        h_coords_list.append(h.coords[begin:end])
        h_feats_list.append(point_embeds.feats[i*10:(i+1)*10])
        h_coords_list.append(point_embeds.coords[i*10:(i+1)*10])
        begin = end + 10
    h = sp.SparseTensor(torch.cat(h_feats_list), torch.cat(h_coords_list))
    for block in self.blocks:
        h = block(h, t_emb, cond)
    h_feats_list = []
    h_coords_list = []
    begin = 0
    for i, coords_len in enumerate(coords_len_list):
        end = begin + 2 * coords_len
        h_feats_list.append(h.feats[begin:end])
        h_coords_list.append(h.coords[begin:end])
        begin = end
    h = sp.SparseTensor(torch.cat(h_feats_list), torch.cat(h_coords_list))
    h = manual_cast(h, x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = self.out_layer(h)
    return h


class Gen3DSegInteractive(nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model
        self.seg_embeddings = nn.Embedding(1, 1536)
        
    def get_positional_encoding(self, input_points):
        if input_points is None:
            return None
        device = input_points['point_slats'].feats.device
        point_feats_embed = torch.zeros((10, 1536), dtype=torch.float32, device=device)
        labels = input_points['point_labels'].squeeze(-1)
        point_feats_embed[labels == 1] = self.seg_embeddings.weight
        return sp.SparseTensor(point_feats_embed, input_points['point_slats'].coords)

    def forward(self, x_t, tex_slats, shape_slats, t, cond, coords_len_list, input_points=None):
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

        point_embeds = self.get_positional_encoding(input_points)
        if point_embeds is not None:
            output_tex_slats = self.flow_model(x_t, t, cond, shape_slats, point_embeds, coords_len_list)
        else:
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

# --- Nodes ---


class SegviGenEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"repo_id": ("STRING", {"default": "microsoft/TRELLIS.2-4B"})}}
    RETURN_TYPES = ("TRELLIS_SHAPE_ENCODER", "TRELLIS_TEX_ENCODER")
    RETURN_NAMES = ("SHAPE_ENCODER", "TEX_ENCODER")
    FUNCTION = "load"
    CATEGORY = "SegviGen/Loaders"
    def load(self, repo_id):
        import folder_paths
        local_dir = os.path.join(folder_paths.models_dir, repo_id)
        os.makedirs(local_dir, exist_ok=True)
        
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        # 1. Shape Encoder
        print(f"[SegviGen] EncoderLoader: loading Shape Encoder from {repo_id}")
        shape_model = models.from_pretrained(f"{repo_id}/ckpts/shape_enc_next_dc_f16c32_fp16", local_dir=local_dir).eval()
        shape_patcher = comfy.model_patcher.ModelPatcher(shape_model, load_device=load_device, offload_device=offload_device)
        
        # 2. Tex Encoder
        print(f"[SegviGen] EncoderLoader: loading Tex Encoder from {repo_id}")
        tex_model = models.from_pretrained(f"{repo_id}/ckpts/tex_enc_next_dc_f16c32_fp16", local_dir=local_dir).eval()
        tex_patcher = comfy.model_patcher.ModelPatcher(tex_model, load_device=load_device, offload_device=offload_device)
        
        print(f"[SegviGen] EncoderLoader: done")
        return (shape_patcher, tex_patcher)

class SegviGenDecoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"repo_id": ("STRING", {"default": "microsoft/TRELLIS.2-4B"})}}
    RETURN_TYPES = ("TRELLIS_SHAPE_DECODER", "TRELLIS_TEX_DECODER")
    RETURN_NAMES = ("SHAPE_DECODER", "TEX_DECODER")
    FUNCTION = "load"
    CATEGORY = "SegviGen/Loaders"
    def load(self, repo_id):
        import folder_paths
        local_dir = os.path.join(folder_paths.models_dir, repo_id)
        os.makedirs(local_dir, exist_ok=True)
        
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        # 1. Shape Decoder
        print(f"[SegviGen] DecoderLoader: loading Shape Decoder from {repo_id}")
        shape_model = models.from_pretrained(f"{repo_id}/ckpts/shape_dec_next_dc_f16c32_fp16", local_dir=local_dir).eval()
        shape_patcher = comfy.model_patcher.ModelPatcher(shape_model, load_device=load_device, offload_device=offload_device)
        
        # 2. Tex Decoder
        print(f"[SegviGen] DecoderLoader: loading Tex Decoder from {repo_id}")
        tex_model = models.from_pretrained(f"{repo_id}/ckpts/tex_dec_next_dc_f16c32_fp16", local_dir=local_dir).eval()
        tex_patcher = comfy.model_patcher.ModelPatcher(tex_model, load_device=load_device, offload_device=offload_device)
        
        print(f"[SegviGen] DecoderLoader: done")
        return (shape_patcher, tex_patcher)

class SegviGenTrellisConfigLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"repo_id": ("STRING", {"default": "microsoft/TRELLIS.2-4B"})}}
    RETURN_TYPES = ("TRELLIS_CONFIG",)
    FUNCTION = "load"
    CATEGORY = "SegviGen/Loaders"
    def load(self, repo_id):
        import folder_paths
        from huggingface_hub import hf_hub_download
        vendor = repo_id.split('/')[0]
        local_dir = os.path.join(folder_paths.models_dir, vendor)
        os.makedirs(local_dir, exist_ok=True)
        print(f"Loading TRELLIS Config from {repo_id}...")
        local_path = os.path.join(local_dir, "TRELLIS.2-4B/pipeline.json")
        if os.path.exists(local_path):
            config_path = local_path
            print(f"Found local config at: {config_path}")
        else:
            config_path = hf_hub_download(repo_id=repo_id, filename="pipeline.json", local_dir=local_dir)
        
        with open(config_path, "r") as f:
            pipeline_config = json.load(f)
        return (pipeline_config['args'],)

class SegviGenCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["full", "full_w_2d_map", "interactive"], {"tooltip": "Select 'full' for standard texturing, 'full_w_2d_map' for improved 2D-guided texturing, or 'interactive' for segmentation support."}),
            }
        }
    RETURN_TYPES = ("SEG_FLOW_MODEL",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "SegviGen"

    _cached_models = {}

    def load_checkpoint(self, mode):
        print(f"[SegviGen] CheckpointLoader: mode={mode}")
        repo_id = "microsoft/TRELLIS.2-4B"
        seg_repo = "Aero-Ex/SegviGen"
        
        if mode == "full":
            filename = "full_seg.safetensors"
        elif mode == "full_w_2d_map":
            filename = "full_seg_w_2d_map.safetensors"
        else:
            filename = "interactive_seg.safetensors"
        
        from huggingface_hub import hf_hub_download
        import folder_paths
        from safetensors.torch import load_file
        
        vendor = "checkpoints" 
        local_dir = os.path.join(folder_paths.models_dir, vendor)
        os.makedirs(local_dir, exist_ok=True)
        
        print(f"Downloading SegviGen checkpoint: {filename} from {seg_repo} to {local_dir}...")
        local_ckpt_path = os.path.join(local_dir, filename)
        if os.path.exists(local_ckpt_path):
            ckpt_path = local_ckpt_path
            print(f"Found local SegviGen checkpoint at: {ckpt_path}")
        else:
            ckpt_path = hf_hub_download(repo_id=seg_repo, filename=filename, local_dir=local_dir)
        
        print(f"Loading {mode} model from {ckpt_path}...")
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
            state_dict = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in state_dict.items()])
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            # Handle the 'gen3dseg.' prefix if present in old ckpts
            state_dict = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in state_dict.items()])

        base_vendor = repo_id.split('/')[0]
        base_local_dir = os.path.join(folder_paths.models_dir, base_vendor)

        # Step 1: Load the backbone model architecture (skip weights to save RAM)
        tex_slat_flow_model_raw = models.from_pretrained(
            f"{repo_id}/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
            local_dir=base_local_dir,
            load_weights=False,
        ).eval()

        # Step 2: Override forward for interactive mode
        if mode == "interactive":
            tex_slat_flow_model_raw.forward = MethodType(flow_forward_interactive, tex_slat_flow_model_raw)

        # Step 3: Build Gen3DSeg wrapper around the RAW model (matching inference_full.py exactly)
        if mode == "interactive":
            gen3dseg = Gen3DSegInteractive(tex_slat_flow_model_raw).eval()
        else:
            gen3dseg = Gen3DSeg(tex_slat_flow_model_raw).eval()

        # Step 4: Load checkpoint into gen3dseg BEFORE wrapping in ModelPatcher
        gen3dseg.load_state_dict(state_dict, strict=False)
        print(f"[SegviGen] Loaded {mode} checkpoint, {len(state_dict)} keys.")

        # Step 5: Wrap the gen3dseg nn.Module in ModelPatcher for ComfyUI memory management
        # gen3dseg is a UNet-like model, so it uses unet_offload_device
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        gen3dseg_patcher = comfy.model_patcher.ModelPatcher(gen3dseg, load_device=load_device, offload_device=offload_device)

        print(f"[SegviGen] CheckpointLoader: ready, mode={mode}")
        return (gen3dseg_patcher,)

class SegviGenGLBToVXZ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "Path to the input GLB model."}),
                "vxz_output_path": ("STRING", {"default": "output.vxz", "tooltip": "Where to save the intermediate voxel (VXZ) file."}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "SegviGen"

    def process(self, glb_path, vxz_output_path):
        print(f"[SegviGen] GLBToVXZ: converting {glb_path} → {vxz_output_path}")
        import folder_paths
        glb_path = folder_paths.get_annotated_filepath(glb_path)
        asset = trimesh.load(glb_path, force='scene')
        asset = preprocess_scene_textures(asset)
        aabb = asset.bounding_box.bounds
        center = (aabb[0] + aabb[1]) / 2
        scale = 0.99999 / (aabb[1] - aabb[0]).max()
        asset.apply_translation(-center)
        asset.apply_scale(scale)
        mesh = asset.to_mesh()
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()

        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices, faces, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False
        )
        vid = o_voxel.serialize.encode_seq(voxel_indices)
        mapping = torch.argsort(vid)
        voxel_indices = voxel_indices[mapping]
        dual_vertices = dual_vertices[mapping]
        intersected = intersected[mapping]

        voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
            asset, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], timing=False
        )
        vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
        mapping_mat = torch.argsort(vid_mat)
        attributes = {k: v[mapping_mat] for k, v in attributes.items()}

        dual_vertices = dual_vertices * 512 - voxel_indices
        dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
        intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

        attributes.update({'dual_vertices': dual_vertices})
        attributes.update({'intersected': intersected})
        o_voxel.io.write(vxz_output_path, voxel_indices, attributes)
        print(f"[SegviGen] GLBToVXZ: done → {vxz_output_path}")
        return (vxz_output_path,)

class SegviGenVXZToSlat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vxz_path": ("STRING", {"default": "", "tooltip": "Path to the voxel (VXZ) file."}),
                "shape_encoder": ("TRELLIS_SHAPE_ENCODER",),
                "tex_encoder": ("TRELLIS_TEX_ENCODER",),
                "shape_decoder": ("TRELLIS_SHAPE_DECODER",),
            }
        }
    RETURN_TYPES = ("SHAPE_SLAT", "TEX_SLAT", "MESHES", "SUBS")
    FUNCTION = "process"
    CATEGORY = "SegviGen"

    def process(self, vxz_path, shape_encoder, tex_encoder, shape_decoder):
        print(f"[SegviGen] VXZToSlat: processing {vxz_path}")
        # Load tensors to the model's load device via patcher
        load_device = shape_encoder.load_device
        
        coords, data = o_voxel.io.read(vxz_path)
        coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1).to(load_device)
        vertices = (data['dual_vertices'].to(load_device) / 255)
        intersected = torch.cat([data['intersected'] % 2, data['intersected'] // 2 % 2, data['intersected'] // 4 % 2], dim=-1).bool().to(load_device)
        vertices_sparse = sp.SparseTensor(vertices, coords)
        intersected_sparse = sp.SparseTensor(intersected.float(), coords)
        
        comfy.model_management.load_models_gpu([shape_encoder, shape_decoder])
        
        with torch.no_grad():
            shape_slat = shape_encoder.model(vertices_sparse, intersected_sparse)
            shape_slat = sp.SparseTensor(shape_slat.feats.to(load_device), shape_slat.coords.to(load_device))
            shape_decoder.model.set_resolution(512)
            meshes, subs = shape_decoder.model(shape_slat, return_subs=True)
        
        base_color = (data['base_color'] / 255).to(load_device)
        metallic = (data['metallic'] / 255).to(load_device)
        roughness = (data['roughness'] / 255).to(load_device)
        alpha = (data['alpha'] / 255).to(load_device)
        attr = torch.cat([base_color, metallic, roughness, alpha], dim=-1).float() * 2 - 1
        
        tex_load_device = tex_encoder.load_device
        if tex_load_device != load_device:
            attr = attr.to(tex_load_device)
            coords = coords.to(tex_load_device)
        comfy.model_management.load_models_gpu([tex_encoder])
        with torch.no_grad():
            tex_slat = tex_encoder.model(sp.SparseTensor(attr, coords))
            
        # Stash intermediate slats on intermediate_device() (CPU normally, GPU on --gpu-only)
        offload_to = comfy.model_management.intermediate_device()
        shape_slat = shape_slat.to(offload_to)
        tex_slat = tex_slat.to(offload_to)
        comfy.model_management.soft_empty_cache()
        print(f"[SegviGen] VXZToSlat: done, shape_slat={shape_slat.coords.shape[0]} voxels, tex_slat={tex_slat.coords.shape[0]} voxels, meshes={len(meshes)}")
        return (shape_slat, tex_slat, meshes, subs)

class SegviGenRMBGLoader:
    """Loads the RMBG-2.0 background removal model as a ComfyUI ModelPatcher."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("RMBG_MODEL",)
    FUNCTION = "load"
    CATEGORY = "SegviGen"

    def load(self):
        import folder_paths
        repo_id = "briaai/RMBG-2.0"
        vendor = repo_id.split('/')[0]
        local_dir = os.path.join(folder_paths.models_dir, vendor)
        os.makedirs(local_dir, exist_ok=True)

        from huggingface_hub import snapshot_download
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            print(f"[SegviGen] RMBGLoader: Downloading RMBG-2.0 to {local_dir}...")
            snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
        else:
            print(f"[SegviGen] RMBGLoader: Loading RMBG-2.0 from local path...")

        rembg_model_raw = BiRefNet(model_name_or_path=local_dir).eval()
        load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        patcher = comfy.model_patcher.ModelPatcher(rembg_model_raw, load_device=load_device, offload_device=offload_device)
        print("[SegviGen] RMBGLoader: done")
        return (patcher,)


class SegviGenDinoLoader:
    """Loads the DinoV3 vision encoder as a ComfyUI ModelPatcher."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("DINO_MODEL",)
    FUNCTION = "load"
    CATEGORY = "SegviGen"

    def load(self):
        import folder_paths
        repo_id = "Aero-Ex/Dinov3"
        subfolder = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        local_dir = folder_paths.models_dir

        from huggingface_hub import hf_hub_download
        local_config = os.path.join(local_dir, f"{subfolder}/config.json")
        local_model = os.path.join(local_dir, f"{subfolder}/model.safetensors")
        if os.path.exists(local_config) and os.path.exists(local_model):
            print(f"[SegviGen] DinoLoader: Loading DinoV3 from local path...")
        else:
            print(f"[SegviGen] DinoLoader: Downloading DinoV3 to {local_dir}...")
            hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/config.json", local_dir=local_dir)
            hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/model.safetensors", local_dir=local_dir)

        cond_model_raw = DinoV3FeatureExtractor(model_name=repo_id, local_dir=local_dir, subfolder=subfolder).eval()
        load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        patcher = comfy.model_patcher.ModelPatcher(cond_model_raw, load_device=load_device, offload_device=offload_device)
        print("[SegviGen] DinoLoader: done")
        return (patcher,)


class SegviGenImagePreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The guidance image."}),
                "rmbg_model": ("RMBG_MODEL", {"tooltip": "RMBG-2.0 background removal model."}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"
    CATEGORY = "SegviGen"

    def preprocess(self, image, rmbg_model):
        print(f"[SegviGen] ImagePreprocessor: removing background...")
        comfy.model_management.load_models_gpu([rmbg_model])

        i = 255. * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        processed_image = rmbg_model.model(pil_image)

        pil_image = processed_image # The model returns a PIL Image with alpha
        max_size = float(max(pil_image.size))
        scale = float(min(1.0, 1024.0 / max_size))
        if scale < 1.0:
            pil_image = pil_image.resize((int(pil_image.width * scale), int(pil_image.height * scale)), Image.Resampling.LANCZOS)

        alpha = np.array(pil_image.split()[-1]) # Get alpha channel
        bbox = np.argwhere(alpha > 0.8 * 255)
        if len(bbox) > 0:
            bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
            center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
            pil_image = pil_image.crop(bbox)

        output_np = np.array(pil_image).astype(np.float32) / 255.
        output_np = output_np[:, :, :3] * output_np[:, :, 3:4] # Apply alpha to RGB

        out_image = torch.from_numpy(output_np).unsqueeze(0)
        comfy.model_management.soft_empty_cache()
        return (out_image,)


class SegviGenImageToCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Processed guidance image."}),
                "dino_model": ("DINO_MODEL", {"tooltip": "DinoV3 vision encoder model."}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "get_cond_emb"
    CATEGORY = "SegviGen"

    def get_cond_emb(self, image, dino_model):
        comfy.model_management.load_models_gpu([dino_model])
        model_on_device = dino_model.model

        i = 255. * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        model_on_device.image_size = 512
        cond = model_on_device([pil_image])
        neg_cond = torch.zeros_like(cond)

        # Store on intermediate_device so they transfer back at inference time
        inter_dev = comfy.model_management.intermediate_device()
        cond = cond.to(inter_dev)
        neg_cond = neg_cond.to(inter_dev)

        comfy.model_management.soft_empty_cache()
        return ({"cond": cond, "neg_cond": neg_cond},)


class SegviGenPointPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vxz_points": ("STRING", {"default": "256 256 256", "tooltip": "Space-separated x y z voxel coordinates (0-511)."}),
                "tex_encoder": ("TRELLIS_TEX_ENCODER",),
            }
        }
    RETURN_TYPES = ("INPUT_POINTS",)
    FUNCTION = "generate"
    CATEGORY = "SegviGen"

    def generate(self, vxz_points, tex_encoder):
        # Use the encoder's actual load device (respects the patcher's device assignment)
        device = tex_encoder.load_device
        points = [int(x) for x in vxz_points.split()]
        if len(points) % 3 != 0:
            raise ValueError("vxz_points must be a multiple of 3 (x y z)")
        
        input_vxz_points_list = [points[i:i+3] for i in range(0, len(points), 3)]
        vxz_points_coords = torch.tensor(input_vxz_points_list, dtype=torch.int32, device=device)
        vxz_points_coords = torch.cat([torch.zeros((vxz_points_coords.shape[0], 1), dtype=torch.int32, device=device), vxz_points_coords], dim=1)
        
        comfy.model_management.load_models_gpu([tex_encoder])
        with torch.no_grad():
            input_points_coords = tex_encoder.model(sp.SparseTensor(torch.zeros((vxz_points_coords.shape[0], 6), dtype=torch.float32, device=device), vxz_points_coords)).coords
        
        input_points_coords = torch.unique(input_points_coords, dim=0)
        point_num = input_points_coords.shape[0]
        if point_num >= 10:
            input_points_coords = input_points_coords[:10]
            point_labels = torch.tensor([[1]]*10, dtype=torch.int32, device=device)
        else:
            input_points_coords = torch.cat([input_points_coords, torch.zeros((10 - point_num, 4), dtype=torch.int32, device=device)], dim=0)
            point_labels = torch.tensor([[1]]*point_num+[[0]]*(10-point_num), dtype=torch.int32, device=device)
        
        comfy.model_management.soft_empty_cache()
        return ({"point_slats": sp.SparseTensor(input_points_coords, input_points_coords), "point_labels": point_labels},)

class SegviGenSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flow_model": ("SEG_FLOW_MODEL",),
                "trellis_config": ("TRELLIS_CONFIG",),
                "shape_slat": ("SHAPE_SLAT",),
                "tex_slat": ("TEX_SLAT",),
                "conditioning": ("CONDITIONING",),
                "subs": ("SUBS",),
                "steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "guidance_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),

                "guidance_interval": ("STRING", {"default": "0.6 0.9", "tooltip": "Optional CFG guidance interval (start end), e.g. '0.6 0.9'. Use '0.0 1.0' for full guidance."}),
                "rescale_t": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "tooltip": "T-scaling factor. default is 3.0 for segmentation."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "input_points": ("INPUT_POINTS",),
            }
        }
    RETURN_TYPES = ("OUTPUT_TEX_SLAT",)
    FUNCTION = "sample"
    CATEGORY = "SegviGen/Process"
 
    def sample(self, flow_model, trellis_config, shape_slat, tex_slat, conditioning, subs, steps, guidance_strength, guidance_interval, rescale_t, seed, input_points=None):
        print(f"[SegviGen] Sampler: steps={steps}, guidance={guidance_strength}, rescale_t={rescale_t}, seed={seed}")
        torch.manual_seed(seed)
        pipeline_args = trellis_config
        
        # Parse guidance interval
        try:
            g_iv = [float(x) for x in guidance_interval.split()]
            if len(g_iv) != 2:
                g_iv = [0.6, 0.9]
        except:
            g_iv = [0.6, 0.9]

        # Override sampler params with user values
        pipeline_args = dict(pipeline_args)
        params = dict(pipeline_args['tex_slat_sampler']['params'])
        params['steps'] = steps
        params['guidance_strength'] = guidance_strength
        params['guidance_interval'] = g_iv
        params['rescale_t'] = rescale_t if rescale_t > 0 else 3.0
        pipeline_args['tex_slat_sampler'] = dict(pipeline_args['tex_slat_sampler'])
        pipeline_args['tex_slat_sampler']['params'] = params

        # Move slats to the flow model's load device
        load_device = flow_model.load_device
        shape_slat = shape_slat.to(load_device)
        tex_slat = tex_slat.to(load_device)
        conditioning = {k: v.to(load_device) if torch.is_tensor(v) else v for k, v in conditioning.items()}

        # Load the gen3dseg ModelPatcher to GPU (flow_model IS the ModelPatcher wrapping Gen3DSeg)
        comfy.model_management.load_models_gpu([flow_model])

        # Pass the underlying Gen3DSeg nn.Module to tex_slat_sample_single, exactly like inference_full.py:
        #   tex_slat_sample_single(PIPE.gen3dseg, PIPE.sampler, ...)
        gen3dseg = flow_model.model
        sampler = Sampler()
        output_tex_slat = tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, tex_slat, conditioning)

        # Stash output on intermediate_device() (CPU normally, GPU on --gpu-only)
        offload_to = comfy.model_management.intermediate_device()
        output_tex_slat = output_tex_slat.to(offload_to)
        comfy.model_management.soft_empty_cache()
        print(f"[SegviGen] Sampler: done, output_tex_slat={output_tex_slat.coords.shape[0]} voxels")
        return (output_tex_slat,)


class SegviGenSlatToVoxel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tex_slat": ("OUTPUT_TEX_SLAT", {"tooltip": "Generated texture slats."}),
                "tex_decoder": ("TRELLIS_TEX_DECODER",),
                "subs": ("SUBS", {"tooltip": "Voxel sub-resolution information."}),
            }
        }
    RETURN_TYPES = ("TEX_VOXELS",)
    FUNCTION = "decode"
    CATEGORY = "SegviGen"

    def decode(self, tex_slat, tex_decoder, subs):
        print(f"[SegviGen] SlatToVoxel: decoding tex_slat ({tex_slat.coords.shape[0]} voxels)...")
        load_device = tex_decoder.load_device
        comfy.model_management.load_models_gpu([tex_decoder])
        tex_slat = tex_slat.to(load_device)
        
        with torch.no_grad():
            tex_voxels = tex_decoder.model(tex_slat, guide_subs=subs) * 0.5 + 0.5
            
        # Move slat and output to intermediate_device (CPU normally, GPU on --gpu-only)
        offload_to = comfy.model_management.intermediate_device()
        tex_slat = tex_slat.to(offload_to)
        tex_voxels = tex_voxels.to(offload_to)
        
        comfy.model_management.soft_empty_cache()
        print(f"[SegviGen] SlatToVoxel: done, tex_voxels shape={tex_voxels.feats.shape}")
        return (tex_voxels,)

class SegviGenVoxelToGLB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "meshes": ("MESHES", {"tooltip": "The decoded meshes."}),
                "tex_voxels": ("TEX_VOXELS", {"tooltip": "Decoded texture voxels."}),
                "output_path": ("STRING", {"default": "output.glb", "tooltip": "Output GLB filename."}),
                "resolution": ("INT", {"default": 512, "tooltip": "Voxel grid resolution."})
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "SegviGen"

    def export(self, meshes, tex_voxels, output_path, resolution):
        print(f"[SegviGen] VoxelToGLB: exporting {len(meshes)} mesh(es) → {output_path} (res={resolution})")
        import cumesh.remeshing
        if hasattr(cumesh.remeshing, 'remesh_narrow_band_dc_quad'):
            cumesh.remeshing.remesh_narrow_band_dc = cumesh.remeshing.remesh_narrow_band_dc_quad

        # Split tex_voxels into per-mesh parts
        tex_parts = []
        begin = 0
        for m in meshes:
            count = m.coords.shape[0] if hasattr(m, 'coords') else len(m.vertices)
            end = begin + count
            tex_parts.append(sp.SparseTensor(tex_voxels.feats[begin:end], tex_voxels.coords[begin:end]))
            begin = end

        # Unload all models before the geometry export — cumesh.remeshing is a heavy
        # GPU kernel and needs the VRAM that loaded models are still holding.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        # Move all to compute device
        device = comfy.model_management.get_torch_device()
        meshes = [m.to(device) if hasattr(m, 'to') else m for m in meshes]
        tex_parts = [v.to(device) for v in tex_parts]

        # slat_to_glb function directly
        glb = slat_to_glb(meshes, tex_parts, resolution=resolution)

        # transformation matrix
        import numpy as np
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0],
        ], dtype=np.float64)
        if hasattr(glb, 'apply_transform') and callable(glb.apply_transform):
            glb.apply_transform(T)

        # Save to ComfyUI output directory
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        full_output_path = os.path.join(output_dir, output_path)
        glb.export(full_output_path)
        comfy.model_management.soft_empty_cache()
        print(f"[SegviGen] VoxelToGLB: done → {full_output_path}")
        return (full_output_path,)

class SegviGenLoadGLB:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith((".glb", ".gltf"))]
        return {
            "required": {
                "glb_file": (sorted(files), {"tooltip": "Select a GLB from the ComfyUI input directory."}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_glb"
    CATEGORY = "SegviGen"

    def load_glb(self, glb_file):
        import folder_paths
        path = folder_paths.get_annotated_filepath(glb_file)
        print(f"[SegviGen] LoadGLB: {path}")
        return (path,)

class SegviGenSplitColorGLB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "The colored GLB generated by Sampler + VoxelToGLB."}),
                "output_name": ("STRING", {"default": "segmented_parts.glb", "tooltip": "Output GLB filename."}),
                "min_faces_per_part": ("INT", {"default": 1, "min": 1, "max": 100000}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "split"
    CATEGORY = "SegviGen"

    def split(self, glb_path, output_name, min_faces_per_part):
        import folder_paths
        import importlib.util as _ilu
        _split_path = os.path.join(os.path.dirname(__file__), "split.py")
        _spec = _ilu.spec_from_file_location("split", _split_path)
        splitter = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(splitter)
        glb_path = folder_paths.get_annotated_filepath(glb_path)
        output_dir = folder_paths.get_output_directory()
        out_glb_path = os.path.join(output_dir, output_name)
        
        splitter.split_glb_by_texture_palette_rgb(
            in_glb_path=glb_path,
            out_glb_path=out_glb_path,
            min_faces_per_part=min_faces_per_part,
            bake_transforms=True,
            debug_print=True
        )
        return (out_glb_path,)

NODE_CLASS_MAPPINGS = {
    "SegviGenEncoderLoader": SegviGenEncoderLoader,
    "SegviGenDecoderLoader": SegviGenDecoderLoader,
    "SegviGenTrellisConfigLoader": SegviGenTrellisConfigLoader,
    "SegviGenCheckpointLoader": SegviGenCheckpointLoader,
    "SegviGenRMBGLoader": SegviGenRMBGLoader,
    "SegviGenDinoLoader": SegviGenDinoLoader,
    "SegviGenGLBToVXZ": SegviGenGLBToVXZ,
    "SegviGenVXZToSlat": SegviGenVXZToSlat,
    "SegviGenImagePreprocessor": SegviGenImagePreprocessor,
    "SegviGenImageToCond": SegviGenImageToCond,
    "SegviGenPointPrompt": SegviGenPointPrompt,
    "SegviGenSampler": SegviGenSampler,
    "SegviGenSlatToVoxel": SegviGenSlatToVoxel,
    "SegviGenVoxelToGLB": SegviGenVoxelToGLB,
    "SegviGenLoadGLB": SegviGenLoadGLB,
    "SegviGenSplitColorGLB": SegviGenSplitColorGLB,
}

