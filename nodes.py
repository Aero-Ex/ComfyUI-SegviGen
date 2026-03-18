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

from .data_toolkit.bpy_render import render_from_transforms
import o_voxel
import nvdiffrast.torch as nr
from trimesh.visual.material import PBRMaterial

# --- Interactive Flow Forward ---
def flow_forward_interactive(self, x, t, cond, concat_cond, point_embeds, coords_len_list):
    # x.feats: [N, 32]
    x = sp.sparse_cat([x, concat_cond], dim=-1)
    if isinstance(cond, list):
        cond = sp.VarLenTensor.from_tensor_list(cond)
    # x.feats: [N, 64]
    h = self.input_layer(x)
    # h.feats: [N, 1536]
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

class Sampler:
    def _inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond):
        t = torch.tensor([t*1000] * x_t.shape[0], dtype=torch.float32).cuda()
        if input_points is not None:
            return model(x_t, tex_slat, shape_slat, t, cond, input_points, coords_len_list)
        else:
            return model(x_t, tex_slat, shape_slat, t, cond, coords_len_list)

    def _pred_to_xstart(self, x_t, t, pred):
        return x_t - t * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return (x_t - x_0) / t

    def guidance_inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, guidance_strength, guidance_rescale=0.0):
        if guidance_strength == 1:
            return self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['cond'])
        elif guidance_strength == 0:
            return self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['neg_cond'])
        else:
            pred_pos = self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['cond'])
            pred_neg = self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['neg_cond'])
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

    def interval_inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, sampler_params):
        guidance_strength = sampler_params['guidance_strength']
        guidance_interval = sampler_params['guidance_interval']
        guidance_rescale = sampler_params['guidance_rescale']
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, guidance_strength, guidance_rescale)
        else:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, 1, guidance_rescale)

    @torch.no_grad()
    def sample_once(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, t_prev, cond_dict, sampler_params):
        pred_v = self.interval_inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, sampler_params)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return pred_x_prev

    @torch.no_grad()
    def sample(self, model, noise, tex_slat, shape_slat, input_points, coords_len_list, cond_dict, sampler_params):
        sample = noise
        steps = sampler_params['steps']
        rescale_t = sampler_params['rescale_t']
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for t, t_prev in tqdm(t_pairs, desc="Sampling"):
            sample = self.sample_once(model, sample, tex_slat, shape_slat, input_points, coords_len_list, t, t_prev, cond_dict, sampler_params)
        return sample

class Gen3DSegInteractive(nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model
        self.seg_embeddings = nn.Embedding(1, 1536)
        
    def get_positional_encoding(self, input_points):
        point_feats_embed = torch.zeros((10, 1536), dtype=torch.float32).to(input_points['point_slats'].feats.device)
        labels = input_points['point_labels'].squeeze(-1)
        point_feats_embed[labels == 1] = self.seg_embeddings.weight
        return sp.SparseTensor(point_feats_embed, input_points['point_slats'].coords)

    def forward(self, x_t, tex_slats, shape_slats, t, cond, input_points, coords_len_list):
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
        output_tex_slats = self.flow_model.model(x_t, t, cond, shape_slats, point_embeds, coords_len_list)
        
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

class Gen3DSegFull(nn.Module):
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

        # Access the underlying model from the ModelPatcher
        output_tex_slats = self.flow_model.model(x_t, t, cond, shape_slats)
        
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

# --- Preprocessing Functions ---

def make_texture_square_pow2(img: Image.Image, target_size=None):
    w, h = img.size
    max_side = max(w, h)
    pow2 = 1
    while pow2 < max_side:
        pow2 *= 2
    if target_size is not None:
        pow2 = target_size
    pow2 = min(pow2, 2048)
    return img.resize((pow2, pow2), Image.BILINEAR)

def preprocess_scene_textures(asset):
    if not isinstance(asset, trimesh.Scene):
        return asset
    TEX_KEYS = ["baseColorTexture", "normalTexture", "metallicRoughnessTexture", "emissiveTexture", "occlusionTexture"]
    for geom in asset.geometry.values():
        visual = getattr(geom, "visual", None)
        mat = getattr(visual, "material", None)
        if mat is None: continue
        for key in TEX_KEYS:
            if not hasattr(mat, key): continue
            tex = getattr(mat, key)
            if tex is None: continue
            if isinstance(tex, Image.Image):
                setattr(mat, key, make_texture_square_pow2(tex))
            elif hasattr(tex, "image") and tex.image is not None:
                img = tex.image
                if not isinstance(img, Image.Image): img = Image.fromarray(img)
                tex.image = make_texture_square_pow2(img)
    return asset

# --- Nodes ---

class SegviGenShapeEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"repo_id": ("STRING", {"default": "microsoft/TRELLIS.2-4B"})}}
    RETURN_TYPES = ("TRELLIS_SHAPE_ENCODER",)
    FUNCTION = "load"
    CATEGORY = "SegviGen/Loaders"
    def load(self, repo_id):
        import folder_paths
        vendor = repo_id.split('/')[0]
        local_dir = os.path.join(folder_paths.models_dir, vendor)
        os.makedirs(local_dir, exist_ok=True)
        print(f"Loading TRELLIS Shape Encoder from {repo_id}...")
        model = models.from_pretrained(f"{repo_id}/ckpts/shape_enc_next_dc_f16c32_fp16", local_dir=local_dir).eval()
        return (comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.intermediate_device()),)

class SegviGenTexEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"repo_id": ("STRING", {"default": "microsoft/TRELLIS.2-4B"})}}
    RETURN_TYPES = ("TRELLIS_TEX_ENCODER",)
    FUNCTION = "load"
    CATEGORY = "SegviGen/Loaders"
    def load(self, repo_id):
        import folder_paths
        vendor = repo_id.split('/')[0]
        local_dir = os.path.join(folder_paths.models_dir, vendor)
        os.makedirs(local_dir, exist_ok=True)
        print(f"Loading TRELLIS Tex Encoder from {repo_id}...")
        model = models.from_pretrained(f"{repo_id}/ckpts/tex_enc_next_dc_f16c32_fp16", local_dir=local_dir).eval()
        return (comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.intermediate_device()),)

class SegviGenShapeDecoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"repo_id": ("STRING", {"default": "microsoft/TRELLIS.2-4B"})}}
    RETURN_TYPES = ("TRELLIS_SHAPE_DECODER",)
    FUNCTION = "load"
    CATEGORY = "SegviGen/Loaders"
    def load(self, repo_id):
        import folder_paths
        vendor = repo_id.split('/')[0]
        local_dir = os.path.join(folder_paths.models_dir, vendor)
        os.makedirs(local_dir, exist_ok=True)
        print(f"Loading TRELLIS Shape Decoder from {repo_id}...")
        model = models.from_pretrained(f"{repo_id}/ckpts/shape_dec_next_dc_f16c32_fp16", local_dir=local_dir).eval()
        return (comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.intermediate_device()),)

class SegviGenTexDecoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"repo_id": ("STRING", {"default": "microsoft/TRELLIS.2-4B"})}}
    RETURN_TYPES = ("TRELLIS_TEX_DECODER",)
    FUNCTION = "load"
    CATEGORY = "SegviGen/Loaders"
    def load(self, repo_id):
        import folder_paths
        vendor = repo_id.split('/')[0]
        local_dir = os.path.join(folder_paths.models_dir, vendor)
        os.makedirs(local_dir, exist_ok=True)
        print(f"Loading TRELLIS Tex Decoder from {repo_id}...")
        model = models.from_pretrained(f"{repo_id}/ckpts/tex_dec_next_dc_f16c32_fp16", local_dir=local_dir).eval()
        return (comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.intermediate_device()),)

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
        local_path = os.path.join(local_dir, "pipeline.json")
        if os.path.exists(local_path):
            config_path = local_path
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
                "mode": (["full", "interactive"], {"tooltip": "Select 'full' for standard texturing or 'interactive' for segmentation support."}),
            }
        }
    RETURN_TYPES = ("SEG_FLOW_MODEL",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "SegviGen"

    _cached_models = {}

    def load_checkpoint(self, mode):
        repo_id = "microsoft/TRELLIS.2-4B"
        seg_repo = "fenghora/SegviGen"
        
        cache_key = mode
        if cache_key in self._cached_models:
            return (self._cached_models[cache_key],)

        filename = "full_seg.ckpt" if mode == "full" else "interactive_seg.ckpt"
        
        from huggingface_hub import hf_hub_download
        import folder_paths
        
        vendor = "microsoft" # Grouping with TRELLIS as per user request
        local_dir = os.path.join(folder_paths.models_dir, vendor)
        os.makedirs(local_dir, exist_ok=True)
        
        print(f"Downloading SegviGen checkpoint: {filename} to {local_dir}...")
        local_ckpt_path = os.path.join(local_dir, filename)
        if os.path.exists(local_ckpt_path):
            ckpt_path = local_ckpt_path
        else:
            ckpt_path = hf_hub_download(repo_id=seg_repo, filename=filename, local_dir=local_dir)
        
        print(f"Loading {mode} model...")
        base_vendor = repo_id.split('/')[0]
        base_local_dir = os.path.join(folder_paths.models_dir, base_vendor)
        tex_slat_flow_model_raw = models.from_pretrained(f"{repo_id}/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16", local_dir=base_local_dir).eval()
        
        if mode == "interactive":
            tex_slat_flow_model_raw.forward = MethodType(flow_forward_interactive, tex_slat_flow_model_raw)
            
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        tex_slat_flow_model_raw.load_state_dict(checkpoint["model"], strict=True)
        
        # Wrap in ModelPatcher
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.intermediate_device()
        tex_slat_flow_model = comfy.model_patcher.ModelPatcher(tex_slat_flow_model_raw, load_device=load_device, offload_device=offload_device)

        # Define wrappers
        if mode == "interactive":
            gen3dseg_raw = Gen3DSegInteractive(tex_slat_flow_model).eval()
        else:
            gen3dseg_raw = Gen3DSegFull(tex_slat_flow_model).eval()
            
        # The checkpoint contains the state_dict for the Gen3DSeg wrapper, not just the flow_model_raw
        state_dict = torch.load(ckpt_path)['state_dict']
        state_dict = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in state_dict.items()])
        gen3dseg_raw.load_state_dict(state_dict, strict=False) # strict=False because flow_model is a patcher inside
        
        self._cached_models[cache_key] = gen3dseg_raw
        return (gen3dseg_raw,)

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
        
        coords, data = o_voxel.io.read(vxz_path)
        coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1).cuda()
        vertices = (data['dual_vertices'].cuda() / 255)
        intersected = torch.cat([data['intersected'] % 2, data['intersected'] // 2 % 2, data['intersected'] // 4 % 2], dim=-1).bool().cuda()
        vertices_sparse = sp.SparseTensor(vertices, coords)
        intersected_sparse = sp.SparseTensor(intersected.float(), coords)
        
        comfy.model_management.load_models_gpu([shape_encoder, shape_decoder])
        
        with torch.no_grad():
            shape_slat = shape_encoder.model(vertices_sparse, intersected_sparse)
            shape_slat = sp.SparseTensor(shape_slat.feats.cuda(), shape_slat.coords.cuda())
            shape_decoder.model.set_resolution(512)
            meshes, subs = shape_decoder.model(shape_slat, return_subs=True)
        
        base_color = (data['base_color'] / 255)
        metallic = (data['metallic'] / 255)
        roughness = (data['roughness'] / 255)
        alpha = (data['alpha'] / 255)
        attr = torch.cat([base_color, metallic, roughness, alpha], dim=-1).float().cuda() * 2 - 1
        
        comfy.model_management.load_models_gpu([tex_encoder])
        with torch.no_grad():
            tex_slat = tex_encoder.model(sp.SparseTensor(attr, coords))
            
        comfy.model_management.soft_empty_cache()
        return (shape_slat, tex_slat, meshes, subs)

class SegviGenImagePreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The guidance image."}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"
    CATEGORY = "SegviGen"

    _rembg_model = None

    def preprocess(self, image):
        if self._rembg_model is None:
            import folder_paths
            repo_id = "briaai/RMBG-2.0"
            vendor = repo_id.split('/')[0]
            local_dir = os.path.join(folder_paths.models_dir, vendor)
            os.makedirs(local_dir, exist_ok=True)
            
            print(f"Downloading/Loading RMBG-2.0 (Local: {local_dir})...")
            from huggingface_hub import hf_hub_download
            
            local_model = os.path.join(local_dir, "model.safetensors")
            if os.path.exists(local_model):
                model_path = local_model
            else:
                model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=local_dir)
                
            rembg_model_raw = BiRefNet(model_name=repo_id, local_dir=local_dir).eval()
            
            # Wrap in ModelPatcher
            load_device = comfy.model_management.get_torch_device()
            offload_device = comfy.model_management.intermediate_device()
            self._rembg_model = comfy.model_patcher.ModelPatcher(rembg_model_raw, load_device=load_device, offload_device=offload_device)

        # Ensure model is on GPU
        comfy.model_management.load_models_gpu([self._rembg_model])
        model_on_device = self._rembg_model.model

        i = 255. * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Call the original __call__ with the model on device
        input_images = model_on_device.transform_image(pil_image).unsqueeze(0).to(self._rembg_model.load_device)
        with torch.no_grad():
            preds = model_on_device.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = Image.fromarray((pred.numpy() * 255).astype(np.uint8)).resize(pil_image.size)
        pil_image.putalpha(pred_pil)
        
        # The original code had a block here that seems to be for handling non-RGB input
        # and then cropping. This new block replaces the original processing logic.
        # I'll keep the cropping logic if it's still desired after the new rembg processing.
        
        # Original cropping logic, adapted to use pil_image after rembg processing
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
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "get_cond_emb"
    CATEGORY = "SegviGen"

    _cond_model = None

    def get_cond_emb(self, image):
        if self._cond_model is None:
            import folder_paths
            repo_id = "Aero-Ex/Dinov3"
            subfolder = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            vendor = "facebook" # Keep saving to models/facebook as requested
            local_dir = os.path.join(folder_paths.models_dir, vendor)
            os.makedirs(local_dir, exist_ok=True)
            
            print(f"Downloading/Loading DinoV3 (Mirror: {repo_id}/{subfolder}) to {local_dir}...")
            from huggingface_hub import hf_hub_download
            
            local_config = os.path.join(local_dir, f"{subfolder}/config.json")
            if os.path.exists(local_config):
                config_path = local_config
            else:
                config_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/config.json", local_dir=local_dir)
                
            local_model = os.path.join(local_dir, f"{subfolder}/model.safetensors")
            if os.path.exists(local_model):
                model_path = local_model
            else:
                model_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/model.safetensors", local_dir=local_dir)
                
            cond_model_raw = DinoV3FeatureExtractor(model_name=repo_id, local_dir=local_dir, subfolder=subfolder).eval()
            
            # Wrap in ModelPatcher
            load_device = comfy.model_management.get_torch_device()
            offload_device = comfy.model_management.intermediate_device()
            self._cond_model = comfy.model_patcher.ModelPatcher(cond_model_raw, load_device=load_device, offload_device=offload_device)

        # Ensure model is on GPU
        comfy.model_management.load_models_gpu([self._cond_model])
        model_on_device = self._cond_model.model

        i = 255. * image[0].cpu().numpy()
        pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Call with model on device
        model_on_device.image_size = 512 # Assuming this property exists on the raw model
        cond = model_on_device([pil_image])
        neg_cond = torch.zeros_like(cond)
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
        points = [int(x) for x in vxz_points.split()]
        if len(points) % 3 != 0:
            raise ValueError("vxz_points must be a multiple of 3 (x y z)")
        
        input_vxz_points_list = [points[i:i+3] for i in range(0, len(points), 3)]
        vxz_points_coords = torch.tensor(input_vxz_points_list, dtype=torch.int32).cuda()
        vxz_points_coords = torch.cat([torch.zeros((vxz_points_coords.shape[0], 1), dtype=torch.int32).cuda(), vxz_points_coords], dim=1)
        
        comfy.model_management.load_models_gpu([tex_encoder])
        with torch.no_grad():
            input_points_coords = tex_encoder.model(sp.SparseTensor(torch.zeros((vxz_points_coords.shape[0], 6), dtype=torch.float32).cuda(), vxz_points_coords)).coords
        
        input_points_coords = torch.unique(input_points_coords, dim=0)
        point_num = input_points_coords.shape[0]
        if point_num >= 10:
            input_points_coords = input_points_coords[:10]
            point_labels = torch.tensor(([[1]]*10), dtype=torch.int32).cuda()
        else:
            input_points_coords = torch.cat([input_points_coords, torch.zeros((10 - point_num, 4), dtype=torch.int32).cuda()], dim=0)
            point_labels = torch.tensor(([[1]]*point_num+[[0]]*(10-point_num)), dtype=torch.int32).cuda()
        
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
                "steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0}),
            },
            "optional": {
                "input_points": ("INPUT_POINTS",),
            }
        }
    RETURN_TYPES = ("OUTPUT_TEX_SLAT",)
    FUNCTION = "sample"
    CATEGORY = "SegviGen/Process"
 
    def sample(self, flow_model, trellis_config, shape_slat, tex_slat, conditioning, steps, guidance_strength, input_points=None):
        pipeline_args = trellis_config
        device = shape_slat.feats.device
        
        shape_std = torch.tensor(pipeline_args['shape_slat_normalization']['std'])[None].to(device)
        shape_mean = torch.tensor(pipeline_args['shape_slat_normalization']['mean'])[None].to(device)
        tex_std = torch.tensor(pipeline_args['tex_slat_normalization']['std'])[None].to(device)
        tex_mean = torch.tensor(pipeline_args['tex_slat_normalization']['mean'])[None].to(device)
        
        norm_shape_slat = ((shape_slat - shape_mean) / shape_std)
        norm_tex_slat = ((tex_slat - tex_mean) / tex_std)
        
        coords_len_list = [shape_slat.coords.shape[0]]
        noise = sp.SparseTensor(torch.randn_like(norm_tex_slat.feats), shape_slat.coords)
        
        sampler = Sampler()
        sampler_params = pipeline_args['tex_slat_sampler']['params'].copy()
        sampler_params['steps'] = steps
        sampler_params['guidance_strength'] = guidance_strength
        
        # Flow model acts as a wrapper around the actual flow patcher
        flow_model_patcher = flow_model.flow_model
        # Ensure the model is on GPU
        comfy.model_management.load_models_gpu([flow_model_patcher])
        
        output_tex_slat = sampler.sample(flow_model, noise, norm_tex_slat, norm_shape_slat, input_points, coords_len_list, conditioning, sampler_params)
        output_tex_slat = output_tex_slat * tex_std + tex_mean
        comfy.model_management.soft_empty_cache()
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
        comfy.model_management.load_models_gpu([tex_decoder])
        with torch.no_grad():
            tex_voxels = tex_decoder.model(tex_slat, guide_subs=subs) * 0.5 + 0.5
        comfy.model_management.soft_empty_cache()
        return (tex_voxels,)

class SegviGenVoxelToGLB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "meshes": ("MESHES", {"tooltip": "The decoded meshes."}),
                "tex_voxels": ("TEX_VOXELS", {"tooltip": "Decoded texture voxels."}),
                "output_path": ("STRING", {"default": "output.glb", "tooltip": "Output GLB filename."}),
                "resolution": ("INT", {"default": 512, "tooltip": "Voxel grid resolution."}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    CATEGORY = "SegviGen"

    def export(self, meshes, tex_voxels, output_path, resolution):
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
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=pbr_attr_layout
                )
            )
        mesh = out_mesh[0]
        mesh.simplify(10000000)
        glb = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   mesh.layout,
            voxel_size          =   mesh.voxel_size,
            aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   =   100000,
            texture_size        =   4096,
            remesh              =   True,
            remesh_band         =   1,
            remesh_project      =   0,
            verbose             =   True
        )
        glb.export(output_path)
        comfy.model_management.soft_empty_cache()
        return (output_path,)

class SegviGenLoadGLB:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        return {
            "required": {
                "glb_file": (folder_paths.get_filename_list("input"), {"tooltip": "Select a GLB from the ComfyUI input directory."}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_glb"
    CATEGORY = "SegviGen"

    def load_glb(self, glb_file):
        import folder_paths
        path = folder_paths.get_annotation_path(glb_file, "input")
        if path is None:
            path = folder_paths.get_full_path("input", glb_file)
        return (path,)

class SegviGenGLBRenderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "Absolute path to input GLB."}),
                "transforms_path": ("STRING", {"default": "", "tooltip": "Path to camera transforms JSON."}),
                "output_image_path": ("STRING", {"default": "rendered.png", "tooltip": "Where to save the rendered image."}),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": "Rendering resolution."}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "render"
    CATEGORY = "SegviGen"

    def render(self, glb_path, transforms_path, output_image_path, resolution):
        render_from_transforms(glb_path, transforms_path, output_image_path, resolution=resolution)
        return (output_image_path,)

class SegviGenGLBToParts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "GLB to split."}),
                "output_dir": ("STRING", {"default": "output/parts", "tooltip": "Directory to save part GLBs."}),
            }
        }
    RETURN_TYPES = ("STRING",) # Dir path
    FUNCTION = "split"
    CATEGORY = "SegviGen"

    def split(self, glb_path, output_dir):
        scene = trimesh.load(glb_path, force='scene')
        os.makedirs(output_dir, exist_ok=True)
        geometries = list(scene.geometry.values())
        for idx, geometry in enumerate(geometries):
            part_scene = trimesh.Scene()
            part_scene.add_geometry(geometry)
            output_path = os.path.join(output_dir, f"{idx}.glb")
            part_scene.export(output_path)
        comfy.model_management.soft_empty_cache()
        return (output_dir,)

def set_mesh_solid_pbr(mesh: trimesh.Trimesh, rgba_uint8=(255, 255, 255, 255), emissive=True):
    rgb = np.array(rgba_uint8[:3], dtype=np.float32) / 255.0
    a = float(rgba_uint8[3]) / 255.0
    colors = np.tile(np.array(rgba_uint8, dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    mat_kwargs = {
        "baseColorFactor": [float(rgb[0]), float(rgb[1]), float(rgb[2]), a],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0,
    }
    if emissive:
        mat_kwargs["emissiveFactor"] = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
    mesh.visual.material = PBRMaterial(**mat_kwargs)
    return mesh

def _load_as_single_mesh(part_path):
    obj = trimesh.load(part_path, force="scene")
    if isinstance(obj, trimesh.Scene):
        dumped = obj.dump()
        meshes = [m for m in dumped if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
        return trimesh.util.concatenate(meshes)
    if isinstance(obj, trimesh.Trimesh):
        return obj

class SegviGenColorGLB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parts_dir": ("STRING", {"default": "output/parts", "tooltip": "Directory containing part GLBs."}),
                "output_dir": ("STRING", {"default": "output/seg_viz", "tooltip": "Directory to save colored results."}),
                "mode": (["random", "interactive"], {"tooltip": "Assign random colors or highlight a single part."}),
                "part_index": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Index of the part to highlight in interactive mode."}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING") # GLB Path, Colors JSON Path
    FUNCTION = "color"
    CATEGORY = "SegviGen"

    def color(self, parts_dir, output_dir, mode, part_index):
        part_meshes = []
        for part_name in sorted(os.listdir(parts_dir)):
            part_path = os.path.join(parts_dir, part_name)
            part_meshes.append(_load_as_single_mesh(part_path))

        os.makedirs(output_dir, exist_ok=True)
        colors_path = os.path.join(output_dir, "colors.json")
        glb_output_path = os.path.join(output_dir, "output.glb")

        if mode == "interactive":
            colors = [(255, 255, 255, 255) if idx == part_index else (0, 0, 0, 255) for idx in range(len(part_meshes))]
            scene = trimesh.Scene()
            for idx, m in enumerate(part_meshes):
                mc = m.copy()
                set_mesh_solid_pbr(mc, rgba_uint8=colors[idx], emissive=True)
                scene.add_geometry(mc, node_name=f"part_{idx}", geom_name=f"geom_{idx}")
            scene.export(glb_output_path)
        else:
            colors = []
            for i in range(len(part_meshes)):
                while True:
                    rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
                    if rgb not in colors:
                        colors.append(rgb)
                        break
            with open(colors_path, "w") as f:
                json.dump([list(c) for c in colors], f)
            scene = trimesh.Scene()
            for i, m in enumerate(part_meshes):
                mc = m.copy()
                set_mesh_solid_pbr(mc, rgba_uint8=colors[i], emissive=True)
                scene.add_geometry(mc, node_name=f"part_{i}", geom_name=f"geom_{i}")
            scene.export(glb_output_path)
        
        comfy.model_management.soft_empty_cache()
        return (glb_output_path, colors_path)

class SegviGenColorImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parts_dir": ("STRING", {"default": "output/parts", "tooltip": "Directory containing part GLBs."}),
                "colors_path": ("STRING", {"default": "output/seg_viz/colors.json", "tooltip": "JSON file mapping parts to colors."}),
                "transforms_path": ("STRING", {"default": "transforms.json", "tooltip": "Camera perspective transforms."}),
                "output_image_path": ("STRING", {"default": "output/seg_viz/2d_map.png", "tooltip": "Where to save the segment map image."}),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING") # Image, Path
    FUNCTION = "render"
    CATEGORY = "SegviGen"

    def render(self, parts_dir, colors_path, transforms_path, output_image_path):
        from .data_toolkit.color_img import load_parts_from_directory, compute_bbox_center_and_scale_like_blender, build_projection_matrix
        
        per_part_vertices, per_part_faces, part_names, vertices_counts = load_parts_from_directory(parts_dir)
        V = np.concatenate(per_part_vertices, axis=0).astype(np.float32)
        F = np.concatenate(per_part_faces, axis=0).astype(np.int32)
        offset, scale = compute_bbox_center_and_scale_like_blender(V)
        V = (V * scale) + offset[None, :]

        with open(colors_path, "r") as f:
            external_colors = json.load(f)
        colors = []
        for idx, part_name in enumerate(part_names):
            rgb = external_colors[idx][:3]
            num_v = vertices_counts[idx]
            col = (np.array(rgb, dtype=np.float32) / 255.0)[None, :]
            colors.append(np.repeat(col, repeats=num_v, axis=0))
        C = np.concatenate(colors, axis=0).astype(np.float32)

        # Rendering logic using nvdiffrast
        glctx = nr.RasterizeCudaContext()
        width = 512
        height = 512
        fov = 40.0 * np.pi / 180.0
        
        V_t = torch.from_numpy(V).cuda()
        F_t = torch.from_numpy(F).cuda()
        C_t = torch.from_numpy(C).cuda()
        
        theta = np.pi / 2.0
        Gx = torch.tensor([
            [1.0, 0.0,             0.0,            0.0],
            [0.0, np.cos(theta),  -np.sin(theta),  0.0],
            [0.0, np.sin(theta),   np.cos(theta),  0.0],
            [0.0, 0.0,             0.0,            1.0],
        ], dtype=torch.float32).cuda()

        with open(transforms_path, "r") as f:
            transforms = json.load(f)
        cam_to_world = np.array(transforms[0]["transform_matrix"], dtype=np.float32)
        world_to_cam = np.linalg.inv(cam_to_world)
        P = build_projection_matrix(fov, width, height)

        V_mat = torch.from_numpy(world_to_cam).cuda()
        P_mat = torch.from_numpy(P).cuda()
        M_t = torch.eye(4, dtype=torch.float32).cuda()
        pos_h = torch.cat([V_t, torch.ones((V_t.shape[0], 1)).cuda()], dim=1)
        pos_clip = (P_mat @ V_mat @ M_t @ Gx) @ pos_h.t()
        pos_clip = pos_clip.t().contiguous().unsqueeze(0)

        rast, _ = nr.rasterize(glctx, pos_clip, F_t, resolution=[height, width])
        feat, _ = nr.interpolate(C_t.unsqueeze(0), rast, F_t)
        cov = rast[..., 3:4]
        img = feat.clamp(0.0, 1.0)
        bg = torch.ones_like(img)
        out = img * (cov > 0) + bg * (cov <= 0)
        
        out_np = out[0].cpu().numpy()
        out_np = out_np[::-1, :, :] # Flip Y
        
        # Save and return as IMAGE
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        Image.fromarray((out_np * 255.0).astype(np.uint8)).save(output_image_path)
        
        image_tensor = torch.from_numpy(out_np).unsqueeze(0)
        comfy.model_management.soft_empty_cache()
        return (image_tensor, output_image_path)

NODE_CLASS_MAPPINGS = {
    "SegviGenShapeEncoderLoader": SegviGenShapeEncoderLoader,
    "SegviGenTexEncoderLoader": SegviGenTexEncoderLoader,
    "SegviGenShapeDecoderLoader": SegviGenShapeDecoderLoader,
    "SegviGenTexDecoderLoader": SegviGenTexDecoderLoader,
    "SegviGenTrellisConfigLoader": SegviGenTrellisConfigLoader,
    "SegviGenCheckpointLoader": SegviGenCheckpointLoader,
    "SegviGenGLBToVXZ": SegviGenGLBToVXZ,
    "SegviGenVXZToSlat": SegviGenVXZToSlat,
    "SegviGenImagePreprocessor": SegviGenImagePreprocessor,
    "SegviGenImageToCond": SegviGenImageToCond,
    "SegviGenPointPrompt": SegviGenPointPrompt,
    "SegviGenSampler": SegviGenSampler,
    "SegviGenSlatToVoxel": SegviGenSlatToVoxel,
    "SegviGenVoxelToGLB": SegviGenVoxelToGLB,
    "SegviGenLoadGLB": SegviGenLoadGLB,
    "SegviGenGLBRenderer": SegviGenGLBRenderer,
    "SegviGenGLBToParts": SegviGenGLBToParts,
    "SegviGenColorGLB": SegviGenColorGLB,
    "SegviGenColorImage": SegviGenColorImage,
}
