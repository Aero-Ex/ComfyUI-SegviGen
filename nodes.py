import os
import torch
import numpy as np
from PIL import Image
import trimesh
import folder_paths
from . import inference_full as inf
from . import split as splitter

REMOTE_CHECKPOINTS = {
    "SegviGen/full_seg.safetensors": ("Aero-Ex/SegviGen", "full_seg.safetensors"),
    "SegviGen/full_seg_w_2d_map.safetensors": ("Aero-Ex/SegviGen", "full_seg_w_2d_map.safetensors"),
}

def resolve_full_path(path):
    if not path or not isinstance(path, str):
        return path
    if os.path.isabs(path):
        return path
    
    # Try output, input, and temp directories
    for folder in [folder_paths.get_output_directory(), folder_paths.get_input_directory(), folder_paths.get_temp_directory()]:
        if folder is None: continue
        full = os.path.abspath(os.path.join(folder, path))
        if os.path.exists(full):
            # print(f"[SegviGen Res] Found: {full}")
            return full
    
    print(f"[SegviGen Res] FAILED to resolve relative path: {path}")
    print(f"  Checked directories based on folder_paths: output={folder_paths.get_output_directory()}, input={folder_paths.get_input_directory()}, temp={folder_paths.get_temp_directory()}")
    # Fallback to check relative to this file
    local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if os.path.exists(local_path):
        return local_path

    return path

# SegviGen Model Loader
class SegviGenModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        ckpts = folder_paths.get_filename_list("checkpoints")
        # Add remote checkpoints to the list if not already present locally
        for remote_name in REMOTE_CHECKPOINTS.keys():
            if remote_name not in ckpts:
                ckpts.append(remote_name)
        return {
            "required": {
                "ckpt_name": (ckpts,),
            }
        }

    RETURN_TYPES = ("SEG_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "SegviGen"

    def load_model(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        if ckpt_path is None or not os.path.exists(ckpt_path):
            if ckpt_name in REMOTE_CHECKPOINTS:
                repo_id, filename = REMOTE_CHECKPOINTS[ckpt_name]
                # Try to find the best local path
                base_ckpt_dir = folder_paths.get_folder_paths("checkpoints")[0]
                
                # If name doesn't have prefix, we still prefer downloading to SegviGen/ subfolder
                if "/" not in ckpt_name:
                    target_path = os.path.join(base_ckpt_dir, "SegviGen", ckpt_name)
                else:
                    target_path = os.path.join(base_ckpt_dir, ckpt_name)
                
                if not os.path.exists(target_path):
                    from huggingface_hub import hf_hub_download
                    print(f"[SegviGen] Downloading remote checkpoint: {ckpt_name} from {repo_id}")
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    hf_hub_download(
                        repo_id=repo_id, 
                        filename=filename, 
                        local_dir=os.path.dirname(target_path)
                    )
                ckpt_path = target_path
            else:
                raise FileNotFoundError(f"Checkpoint {ckpt_name} not found and no remote mapping exists.")

        inf.PIPE.load_all_models()
        inf.PIPE.load_ckpt_if_needed(ckpt_path)
        return ({"ckpt_path": ckpt_path},)

# SegviGen Mesh Voxelizer
class SegviGenMeshVoxelizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("*",),
            }
        }

    RETURN_TYPES = ("VXZ_DATA",)
    FUNCTION = "voxelize"
    CATEGORY = "SegviGen"

    def voxelize(self, mesh):
        # mesh can be a path, a trimesh object, or a dict containing either
        if isinstance(mesh, list) and len(mesh) > 0:
            mesh = mesh[0]
            
        if isinstance(mesh, dict):
            mesh = mesh.get("mesh") or mesh.get("glb_path") or mesh

        if isinstance(mesh, str):
            glb_path = resolve_full_path(mesh)
        else:
            glb_path = None
            for attr in ["source", "path", "_path", "full_path", "filename", "abs_path"]:
                if hasattr(mesh, attr):
                    val = getattr(mesh, attr)
                    if isinstance(val, str):
                        glb_path = val
                        break
            
            if glb_path is None:
                if hasattr(mesh, "export"):
                    temp_dir = folder_paths.get_temp_directory()
                    glb_path = os.path.join(temp_dir, f"segvigen_input_{os.urandom(4).hex()}.glb")
                    mesh.export(glb_path)
                elif type(mesh).__name__ == "File3D":
                    # Fallback for File3D if attributes didn't match, try to parse source from repr
                    m_repr = str(mesh)
                    if "source='" in m_repr:
                        glb_path = m_repr.split("source='")[1].split("'")[0]
                    else:
                        raise ValueError(f"Unsupported File3D format: {m_repr}")
                else:
                    raise ValueError(f"Unsupported mesh type: {type(mesh)}. Attributes: {dir(mesh)}")

        vxz_path = glb_path.replace(".glb", ".vxz")
        inf.process_glb_to_vxz(glb_path, vxz_path)
        return ({"vxz_path": vxz_path, "glb_path": glb_path},)

# SegviGen Latent Encoder
class SegviGenLatentEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vxz_data": ("VXZ_DATA",),
            }
        }

    RETURN_TYPES = ("SHAPE_SLAT", "TEX_SLAT", "MESHES", "SUBS")
    FUNCTION = "encode"
    CATEGORY = "SegviGen"

    def encode(self, vxz_data):
        inf.PIPE.load_all_models()
        shape_enc, tex_enc, shape_dec = inf.PIPE.get_encoders_decoder()
        shape_enc.cuda()
        tex_enc.cuda()
        shape_dec.cuda()
        
        shape_slat, meshes, subs, tex_slat = inf.vxz_to_latent_slat(
            shape_enc, shape_dec, tex_enc, vxz_data["vxz_path"]
        )
        
        # Offload models and move latents to CPU to save VRAM
        inf.PIPE.unload('shape_encoder', 'tex_encoder', 'shape_decoder')
        
        return (
            {"feats": shape_slat.feats.cpu(), "coords": shape_slat.coords.cpu()},
            {"feats": tex_slat.feats.cpu(), "coords": tex_slat.coords.cpu()},
            meshes,
            subs
        )

# SegviGen Image Conditioner
class SegviGenImageConditioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITION",)
    FUNCTION = "condition"
    CATEGORY = "SegviGen"

    def condition(self, image):
        # ComfyUI [B, H, W, C] -> PIL
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        inf.PIPE.load_all_models()
        rembg_model = inf.PIPE.get_rembg()
        rembg_model.cuda()
        pil_img = inf.preprocess_image(rembg_model, pil_img)
        inf.PIPE.unload('rembg_model')
        
        cond_model = inf.PIPE.get_cond_model()
        cond_model.cuda()
        cond = inf.get_cond(cond_model, [pil_img])
        # Offload to CPU
        cond = {k: v.cpu() for k, v in cond.items()}
        inf.PIPE.unload('image_cond_model')
        
        return (cond,)

# SegviGen Sampler
class SegviGenSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seg_model": ("SEG_MODEL",),
                "shape_slat": ("SHAPE_SLAT",),
                "tex_slat": ("TEX_SLAT",),
                "condition": ("CONDITION",),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT_SLAT_OUT",)
    FUNCTION = "sample"
    CATEGORY = "SegviGen"

    def sample(self, seg_model, shape_slat, tex_slat, condition, steps, guidance_scale, seed):
        inf.PIPE.load_all_models()
        inf.PIPE.load_ckpt_if_needed(seg_model["ckpt_path"])
        gen3dseg = inf.PIPE.get_gen3dseg()
        gen3dseg.cuda()
        
        # We need to reconstruct SparseTensor from dict
        import trellis2.modules.sparse as sp
        shape_slat_sp = sp.SparseTensor(shape_slat["feats"].cuda(), shape_slat["coords"].cuda())
        tex_slat_sp = sp.SparseTensor(tex_slat["feats"].cuda(), tex_slat["coords"].cuda())
        cond_gpu = {k: v.cuda() for k, v in condition.items()}
        
        output_tex_slat = inf.tex_slat_sample_single(
            gen3dseg, inf.PIPE.sampler, inf.PIPE.pipeline_args, 
            shape_slat_sp, tex_slat_sp, cond_gpu,
            steps=steps, cfg_scale=guidance_scale, seed=seed
        )
        params = inf.PIPE.pipeline_args['tex_slat_sampler']['params'].copy()
        params['steps'] = steps
        params['guidance_strength'] = guidance_scale
        
        output_tex_slat = inf.tex_slat_sample_single(
            gen3dseg, inf.PIPE.sampler, inf.PIPE.pipeline_args, shape_slat_sp, tex_slat_sp, cond_gpu
        )
        
        res = {"feats": output_tex_slat.feats.cpu(), "coords": output_tex_slat.coords.cpu()}
        inf.PIPE.unload('gen3dseg', 'tex_slat_flow_model')
        return (res,)

# SegviGen Texture Decoder
class SegviGenTextureDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_slat": ("LATENT_SLAT_OUT",),
                "subs": ("SUBS",),
            }
        }

    RETURN_TYPES = ("TEXTURE_VOXELS",)
    FUNCTION = "decode"
    CATEGORY = "SegviGen"

    def decode(self, latent_slat, subs):
        import trellis2.modules.sparse as sp
        inf.PIPE.load_all_models()
        tex_decoder = inf.PIPE.get_tex_decoder()
        tex_decoder.cuda()
        
        latent_slat_sp = sp.SparseTensor(latent_slat["feats"].cuda(), latent_slat["coords"].cuda())
        subs_gpu = [s.cuda() if isinstance(s, torch.Tensor) else s for s in subs]
        
        with torch.no_grad():
            tex_voxels = tex_decoder(latent_slat_sp, guide_subs=subs_gpu) * 0.5 + 0.5
            tex_voxels = [v.cpu() for v in tex_voxels]
            
        inf.PIPE.unload('tex_decoder')
        return (tex_voxels,)

# SegviGen Mesh Baker
class SegviGenMeshBaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("*",),
                "texture_voxels": ("TEXTURE_VOXELS",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 1024}),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 4096}),
                "generate_uv": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("GLB_PATH",)
    FUNCTION = "bake"
    CATEGORY = "SegviGen"
    OUTPUT_NODE = True

    def bake(self, mesh, texture_voxels, resolution, texture_size, generate_uv):
        # Extract mesh from various wrappers
        if isinstance(mesh, list) and len(mesh) > 0:
            mesh = mesh[0]
            
        if isinstance(mesh, dict):
            glb_path = mesh.get("glb_path") or mesh.get("mesh")
            if not isinstance(glb_path, str):
                mesh = glb_path
            else:
                mesh = glb_path

        if isinstance(mesh, str):
            glb_path = resolve_full_path(mesh)
        else:
            glb_path = None
            for attr in ["source", "path", "_path", "full_path", "filename", "abs_path"]:
                if hasattr(mesh, attr):
                    val = getattr(mesh, attr)
                    if isinstance(val, str):
                        glb_path = val
                        break
            
            if glb_path is None:
                if hasattr(mesh, "export"):
                    # If it's a trimesh object without original path, we must save it to bake onto it
                    temp_dir = folder_paths.get_temp_directory()
                    glb_path = os.path.join(temp_dir, f"segvigen_bake_src_{os.urandom(4).hex()}.glb")
                    mesh.export(glb_path)
                elif type(mesh).__name__ == "File3D":
                    m_repr = str(mesh)
                    if "source='" in m_repr:
                        glb_path = m_repr.split("source='")[1].split("'")[0]
                else:
                    raise ValueError(f"Unsupported mesh type for baking: {type(mesh)}")
        

        output_path = os.path.join(folder_paths.get_output_directory(), f"segvigen_baked_{os.urandom(4).hex()}.glb")
        
        inf.bake_to_mesh(
            glb_path, 
            texture_voxels, 
            output_path, 
            resolution=resolution, 
            texture_size=texture_size,
            generate_uv=generate_uv
        )
        return (output_path,)

# SegviGen Mesh Exporter
class SegviGenMeshExporter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "meshes": ("MESHES",),
                "texture_voxels": ("TEXTURE_VOXELS",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 1024}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("GLB_PATH",)
    FUNCTION = "export_mesh"
    CATEGORY = "SegviGen"
    OUTPUT_NODE = True

    def export_mesh(self, meshes, texture_voxels, resolution):
        glb = inf.slat_to_glb(meshes, texture_voxels, resolution=resolution)
        
        # Apply the same Y-up fix as in original app.py
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        
        output_path = os.path.join(folder_paths.get_output_directory(), f"segvigen_exported_{os.urandom(4).hex()}.glb")
        
        if hasattr(glb, "apply_transform"):
            glb.apply_transform(T)
            glb.export(output_path)
        else:
            glb.export(output_path)
            scene_or_mesh = trimesh.load(output_path, force="scene")
            scene_or_mesh.apply_transform(T)
            scene_or_mesh.export(output_path)
            
        return (output_path,)

# SegviGen Split Refine
class SegviGenSplitRefine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING",),
                "min_faces_per_part": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "bake_transforms": ("BOOLEAN", {"default": True}),
                "color_quant_step": ("INT", {"default": 16, "min": 1, "max": 64}),
                "palette_sample_pixels": ("INT", {"default": 2000000}),
                "palette_min_pixels": ("INT", {"default": 500}),
                "palette_max_colors": ("INT", {"default": 256}),
                "palette_merge_dist": ("INT", {"default": 32}),
                "samples_per_face": ([1, 4], {"default": 4}),
                "flip_v": ("BOOLEAN", {"default": True}),
                "uv_wrap_repeat": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PARTS_GLB_PATH",)
    FUNCTION = "split"
    CATEGORY = "SegviGen"
    OUTPUT_NODE = True

    def split(self, glb_path, **kwargs):
        glb_path = resolve_full_path(glb_path)
        out_parts_glb = os.path.join(folder_paths.get_output_directory(), f"segvigen_parts_{os.urandom(4).hex()}.glb")
        splitter.split_glb_by_texture_palette_rgb(
            in_glb_path=glb_path,
            out_glb_path=out_parts_glb,
            debug_print=True,
            **kwargs
        )
        return (out_parts_glb,)

# SegviGen Mesh Simplify
class SegviGenMeshSimplify:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING",),
                "target_faces": ("INT", {"default": 100000, "min": 1000, "max": 1000000}),
                "aggression": ("INT", {"default": 7, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("GLB_PATH",)
    FUNCTION = "simplify"
    CATEGORY = "SegviGen"
    OUTPUT_NODE = True

    def simplify(self, glb_path, target_faces, aggression):
        glb_path = resolve_full_path(glb_path)
        out_glb = os.path.join(folder_paths.get_output_directory(), f"segvigen_simplified_{os.urandom(4).hex()}.glb")
        inf.build_simplified_work_glb(glb_path, out_glb, target_faces=target_faces, aggression=aggression)
        return (out_glb,)

# SegviGen Monolithic Segmentation
class SegviGenMonolithicSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("*",),
                "seg_model": ("SEG_MODEL",),
                "bake_mode": ("BOOLEAN", {"default": False}),
                "generate_uv": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("GLB_PATH",)
    FUNCTION = "process"
    CATEGORY = "SegviGen"

    def process(self, mesh, seg_model, bake_mode, generate_uv, image=None):
        # 0. Robust Mesh Extract
        if isinstance(mesh, list) and len(mesh) > 0:
            mesh = mesh[0]
        if isinstance(mesh, dict):
            mesh = mesh.get("mesh") or mesh.get("glb_path") or mesh

        # 1. Mesh preparation
        if isinstance(mesh, str):
            glb_path = resolve_full_path(mesh)
        else:
            glb_path = None
            for attr in ["source", "path", "_path", "full_path", "filename", "abs_path"]:
                if hasattr(mesh, attr):
                    val = getattr(mesh, attr)
                    if isinstance(val, str):
                        glb_path = val
                        break
            
            if glb_path is None:
                if hasattr(mesh, "export"):
                    temp_dir = folder_paths.get_temp_directory()
                    glb_path = os.path.join(temp_dir, f"segvigen_mono_{os.urandom(4).hex()}.glb")
                    mesh.export(glb_path)
                elif type(mesh).__name__ == "File3D":
                    m_repr = str(mesh)
                    if "source='" in m_repr:
                        glb_path = m_repr.split("source='")[1].split("'")[0]
                else:
                    raise ValueError(f"Unsupported mesh type: {type(mesh)}")

        # 2. Temp files
        workdir = os.path.join(folder_paths.get_temp_directory(), f"seg_mono_{os.urandom(4).hex()}")
        os.makedirs(workdir, exist_ok=True)
        in_vxz = os.path.join(workdir, "input.vxz")
        export_glb = os.path.join(folder_paths.get_output_directory(), f"segvigen_mono_out_{os.urandom(4).hex()}.glb")

        item = {
            "glb": glb_path,
            "input_vxz": in_vxz,
            "export_glb": export_glb,
            "bake": bake_mode,
            "generate_uv": generate_uv,
        }

        if image is not None:
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            out_img = os.path.join(workdir, "input_map.png")
            pil_img.save(out_img)
            item["2d_map"] = True
            item["img"] = out_img
        else:
            item["2d_map"] = False
            item["transforms"] = os.path.join(os.path.dirname(__file__), "data_toolkit", "transforms.json")
            item["img"] = os.path.join(workdir, "render.png")

        inf.inference_with_loaded_models(seg_model["ckpt_path"], item)
        return (export_glb,)

NODE_CLASS_MAPPINGS = {
    "SegviGenModelLoader": SegviGenModelLoader,
    "SegviGenMeshVoxelizer": SegviGenMeshVoxelizer,
    "SegviGenLatentEncoder": SegviGenLatentEncoder,
    "SegviGenImageConditioner": SegviGenImageConditioner,
    "SegviGenSampler": SegviGenSampler,
    "SegviGenTextureDecoder": SegviGenTextureDecoder,
    "SegviGenMeshBaker": SegviGenMeshBaker,
    "SegviGenMeshExporter": SegviGenMeshExporter,
    "SegviGenSplitRefine": SegviGenSplitRefine,
    "SegviGenMeshSimplify": SegviGenMeshSimplify,
    "SegviGenMonolithicSegmentation": SegviGenMonolithicSegmentation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegviGenModelLoader": "SegviGen Model Loader",
    "SegviGenMeshVoxelizer": "SegviGen Mesh Voxelizer",
    "SegviGenLatentEncoder": "SegviGen Latent Encoder",
    "SegviGenImageConditioner": "SegviGen Image Conditioner",
    "SegviGenSampler": "SegviGen Sampler",
    "SegviGenTextureDecoder": "SegviGen Texture Decoder",
    "SegviGenMeshBaker": "SegviGen Mesh Baker",
    "SegviGenMeshExporter": "SegviGen Mesh Exporter",
    "SegviGenSplitRefine": "SegviGen Split & Refine",
    "SegviGenMeshSimplify": "SegviGen Mesh Simplify",
    "SegviGenMonolithicSegmentation": "SegviGen Monolithic Segmentation",
}
