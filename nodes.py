import os
import urllib.request
import torch
import numpy as np
from PIL import Image
import folder_paths
from .model_manager import PIPE
from .pipeline_inference import (
    run_segvigen_inference, Sampler, Gen3DSeg, 
    preprocess_image, get_cond, tex_slat_sample_single
)
from .utils_3d import (
    build_simplified_work_glb, get_work_glb_path, process_glb_to_vxz,
    vxz_to_latent_slat, bake_to_mesh, slat_to_glb, fast_simplify_mesh,
    EARLY_SIMPLIFY_ENABLED, EARLY_SIMPLIFY_TARGET_FACES, EARLY_SIMPLIFY_AGGRESSION
)
from . import split as splitter
import trellis2.modules.sparse as sp

# Get ComfyUI models directory
MODELS_DIR = folder_paths.models_dir

def load_checkpoint_helper(checkpoint):
    """Checks for checkpoint in ComfyUI checkpoints folder and downloads if missing."""
    if checkpoint.endswith(".ckpt"):
        checkpoint = checkpoint.replace(".ckpt", ".safetensors")
        
    ckpt_path = os.path.join(MODELS_DIR, "checkpoints", checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"SegviGen: Downloading {checkpoint} to {ckpt_path}...")
        # Always use safetensors extension for the download URL
        remote_filename = checkpoint
        if remote_filename.endswith(".ckpt"):
            remote_filename = remote_filename.replace(".ckpt", ".safetensors")
            
        url = f"https://huggingface.co/Aero-Ex/SegviGen/resolve/main/{remote_filename}"
        try:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            urllib.request.urlretrieve(url, ckpt_path)
            print(f"SegviGen: Download complete.")
        except Exception as e:
            raise RuntimeError(f"SegviGen: Failed to download {remote_filename} from {url}: {e}")

def resolve_3d_path(input_3d):
    """Helper to resolve paths from strings or File3D objects."""
    import re
    glb_path = None
    
    if isinstance(input_3d, (str, os.PathLike)):
        glb_path = str(input_3d)
    elif hasattr(input_3d, "source"):
        glb_path = str(input_3d.source)
    elif hasattr(input_3d, "path"):
        glb_path = str(input_3d.path)
    else:
        s = str(input_3d)
        # Fallback: parse source='...' from string representation
        match = re.search(r"source=['\"](.*?)['\"]", s)
        if match:
            glb_path = match.group(1)
        else:
            glb_path = s
    
    # Robust path resolution
    if not os.path.isabs(glb_path):
        for search_dir in [folder_paths.get_input_directory(), folder_paths.get_output_directory(), folder_paths.get_temp_directory()]:
            potential_path = os.path.join(search_dir, glb_path)
            if os.path.exists(potential_path):
                glb_path = potential_path
                break
    
    if not os.path.isfile(glb_path):
        base_dir = os.path.abspath(os.path.join(folder_paths.base_path))
        potential_path = os.path.join(base_dir, glb_path)
        if os.path.exists(potential_path):
            glb_path = potential_path
            
    return glb_path

class SegviGenModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (["full_seg.safetensors", "full_seg_w_2d_map.safetensors"],),
            }
        }
    
    RETURN_TYPES = ("SEG_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "SegviGen"

    def load_model(self, checkpoint):
        # Determine path in standard ComfyUI checkpoints
        ckpt_path = os.path.join(MODELS_DIR, "checkpoints", checkpoint)
        
        # Auto-download if missing
        load_checkpoint_helper(checkpoint)
        
        # Load the checkpoint through model manager
        PIPE.set_checkpoint(ckpt_path)
        
        return (checkpoint,)

class SegviGenRmbg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "PIL_IMAGE")
    RETURN_NAMES = ("image", "pil_image")
    FUNCTION = "process"
    CATEGORY = "SegviGen/Stages"

    def process(self, image):
        # Convert ComfyUI IMAGE tensor to PIL
        i = 255. * image[0].cpu().numpy()
        input_img = Image.fromarray(np.uint8(i))
        
        rembg_model = PIPE.get_rembg()
        rembg_model.cuda()
        processed_img = preprocess_image(rembg_model, input_img)
        PIPE.unload('rembg_model')
        
        # Convert back to IMAGE tensor for ComfyUI preview/output
        out_np = np.array(processed_img).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_np)[None,]
        
        return (out_tensor, processed_img)

class SegviGenConditioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pil_image": ("PIL_IMAGE",),
            }
        }
    
    RETURN_TYPES = ("SEG_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "generate"
    CATEGORY = "SegviGen/Stages"

    def generate(self, pil_image):
        cond_model = PIPE.get_cond_model()
        cond_model.cuda()
        # DinoV3 needs the image
        cond = get_cond(cond_model, [pil_image])
        # Move features to CPU to save VRAM
        cond = {k: v.cpu() for k, v in cond.items()}
        PIPE.unload('image_cond_model')
        return (cond,)

class SegviGenVoxelEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_3d": ("*",),
            }
        }
    
    RETURN_TYPES = ("SEG_SLAT", "SEG_MESHES", "SEG_SUBS", "SEG_SLAT")
    RETURN_NAMES = ("shape_slat", "meshes", "subs", "tex_slat")
    FUNCTION = "encode"
    CATEGORY = "SegviGen/Stages"

    def encode(self, input_3d):
        import trimesh
        import shutil
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = os.path.join(folder_paths.get_temp_directory(), f"segvigen_venc_{timestamp}")
        os.makedirs(workdir, exist_ok=True)
        
        in_glb = os.path.join(workdir, "input.glb")
        in_vxz = os.path.join(workdir, "input.vxz")

        # Resolve input_3d to GLB file
        if isinstance(input_3d, dict):
            vertices = input_3d.get('vertices')
            faces = input_3d.get('faces')
            if vertices is not None and faces is not None:
                if len(vertices.shape) == 3:
                    vertices = vertices[0]
                    faces = faces[0]
                t_mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
                t_mesh.export(in_glb)
            else:
                raise ValueError("Unsupported dict format for input_3d.")
        else:
            glb_path = resolve_3d_path(input_3d)
            if not glb_path or not os.path.isfile(glb_path):
                raise FileNotFoundError(f"Input GLB not found: {glb_path}")
            shutil.copy(glb_path, in_glb)

        # 3. VXZ to Latent SLat
        if EARLY_SIMPLIFY_ENABLED:
            work_glb = os.path.join(workdir, "work.glb")
            build_simplified_work_glb(in_glb, work_glb, EARLY_SIMPLIFY_TARGET_FACES, EARLY_SIMPLIFY_AGGRESSION)
        
        process_glb_to_vxz(in_glb, in_vxz)
        
        shape_enc, tex_enc, shape_dec = PIPE.get_encoders_decoder()
        for m in [shape_enc, tex_enc, shape_dec]: m.cuda()
        
        shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(shape_enc, shape_dec, tex_enc, in_vxz)
        
        # Move to CPU for intermediate storage
        shape_slat = sp.SparseTensor(shape_slat.feats.cpu(), shape_slat.coords.cpu())
        tex_slat = sp.SparseTensor(tex_slat.feats.cpu(), tex_slat.coords.cpu())
        
        PIPE.unload('shape_encoder', 'tex_encoder', 'shape_decoder')
        
        return (shape_slat, meshes, subs, tex_slat)

class SegviGenSamplerConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "rescale_t": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("SEG_SAMPLER_CONFIG",)
    RETURN_NAMES = ("sampler_config",)
    FUNCTION = "get_config"
    CATEGORY = "SegviGen/Stages"

    def get_config(self, steps, guidance_strength, rescale_t, seed):
        return ({"steps": steps, "guidance_strength": guidance_strength, "rescale_t": rescale_t, "seed": seed},)

class SegviGenSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (["full_seg.safetensors", "full_seg_w_2d_map.safetensors"],),
                "conditioning": ("SEG_CONDITIONING",),
                "shape_slat": ("SEG_SLAT",),
                "tex_slat": ("SEG_SLAT",),
            },
            "optional": {
                "sampler_settings": ("SEG_SAMPLER_CONFIG",),
            }
        }
    
    RETURN_TYPES = ("SEG_SLAT",)
    RETURN_NAMES = ("output_tex_slat",)
    FUNCTION = "sample"
    CATEGORY = "SegviGen/Stages"

    def sample(self, checkpoint, conditioning, shape_slat, tex_slat, sampler_settings=None):
        # Ensure checkpoint exists (trigger download if needed)
        load_checkpoint_helper(checkpoint)
        ckpt_path = os.path.join(MODELS_DIR, "checkpoints", checkpoint)
        
        PIPE.load_config()
        PIPE.set_checkpoint(ckpt_path)
        
        gen3dseg = PIPE.get_gen3dseg(Gen3DSeg)
        gen3dseg.cuda()
        
        # Merge settings
        sampler_params = PIPE.pipeline_args['tex_slat_sampler']['params'].copy()
        if sampler_settings:
            sampler_params.update(sampler_settings)
            torch.manual_seed(sampler_settings.get('seed', 0))
        
        # Ensure inputs are on CUDA for sampling
        cond_cuda = {k: v.cuda() for k, v in conditioning.items()}
        shape_slat_cuda = sp.SparseTensor(shape_slat.feats.cuda(), shape_slat.coords.cuda())
        tex_slat_cuda = sp.SparseTensor(tex_slat.feats.cuda(), tex_slat.coords.cuda())
        
        # Explicit normalization and sampling logic (robustly separated)
        device = shape_slat_cuda.feats.device
        pipeline_args = PIPE.pipeline_args
        shape_std = torch.tensor(pipeline_args['shape_slat_normalization']['std'])[None].to(device)
        shape_mean = torch.tensor(pipeline_args['shape_slat_normalization']['mean'])[None].to(device)
        tex_std = torch.tensor(pipeline_args['tex_slat_normalization']['std'])[None].to(device)
        tex_mean = torch.tensor(pipeline_args['tex_slat_normalization']['mean'])[None].to(device)
        
        norm_shape_slat = ((shape_slat_cuda - shape_mean) / shape_std)
        norm_tex_slat = ((tex_slat_cuda - tex_mean) / tex_std)
        
        noise = sp.SparseTensor(torch.randn_like(norm_tex_slat.feats), norm_shape_slat.coords)
        
        sampler_obj = Sampler()
        output_tex_slat = sampler_obj.sample(
            gen3dseg, noise, norm_tex_slat, norm_shape_slat, 
            [norm_shape_slat.coords.shape[0]], cond_cuda, sampler_params
        )
        
        output_tex_slat = output_tex_slat * tex_std + tex_mean
        
        # Move back to CPU
        output_tex_slat_cpu = sp.SparseTensor(output_tex_slat.feats.cpu(), output_tex_slat.coords.cpu())
        
        PIPE.unload('gen3dseg', 'tex_slat_flow_model')
        return (output_tex_slat_cpu,)

class SegviGenStage5Renderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_tex_slat": ("SEG_SLAT",),
                "meshes": ("SEG_MESHES",),
                "subs": ("SEG_SUBS",),
                "filename_prefix": ("STRING", {"default": "segvigen/seg"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    OUTPUT_NODE = True
    FUNCTION = "render"
    CATEGORY = "SegviGen/Stages"

    def render(self, output_tex_slat, meshes, subs, filename_prefix="segvigen/seg"):
        import datetime
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
        os.makedirs(full_output_folder, exist_ok=True)
        
        out_name = f"{filename}_{counter:05}_.glb"
        out_glb = os.path.join(full_output_folder, out_name)

        tex_decoder = PIPE.get_tex_decoder()
        tex_decoder.cuda()
        
        with torch.no_grad():
            tex_voxels = tex_decoder(
                sp.SparseTensor(output_tex_slat.feats.cuda(), output_tex_slat.coords.cuda()), 
                guide_subs=[s.cuda() if isinstance(s, torch.Tensor) else s for s in subs]
            ) * 0.5 + 0.5
            tex_voxels = [v.cpu() for v in tex_voxels]
        
        PIPE.unload('tex_decoder')
        
        glb = slat_to_glb(meshes, tex_voxels)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        glb.apply_transform(T)
        glb.export(out_glb)
        
        # Final cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return (out_glb,)

class SegviGenStage5Baker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_tex_slat": ("SEG_SLAT",),
                "subs": ("SEG_SUBS",),
                "original_3d": ("*",),
                "generate_uv": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "segvigen/bake"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    OUTPUT_NODE = True
    FUNCTION = "bake"
    CATEGORY = "SegviGen/Stages"

    def bake(self, output_tex_slat, subs, original_3d, generate_uv, filename_prefix="segvigen/bake"):
        import datetime
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
        os.makedirs(full_output_folder, exist_ok=True)
        
        out_name = f"{filename}_{counter:05}_.glb"
        out_glb = os.path.join(full_output_folder, out_name)

        tex_decoder = PIPE.get_tex_decoder()
        tex_decoder.cuda()
        
        with torch.no_grad():
            tex_voxels = tex_decoder(
                sp.SparseTensor(output_tex_slat.feats.cuda(), output_tex_slat.coords.cuda()), 
                guide_subs=[s.cuda() if isinstance(s, torch.Tensor) else s for s in subs]
            ) * 0.5 + 0.5
            tex_voxels = [v.cpu() for v in tex_voxels]
        
        PIPE.unload('tex_decoder')
        
        glb_path = resolve_3d_path(original_3d)
        if not glb_path or not os.path.isfile(glb_path):
            raise FileNotFoundError(f"Original GLB not found: {glb_path}")

        bake_to_mesh(glb_path, tex_voxels, out_glb, generate_uv=generate_uv)
        
        # Final cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return (out_glb,)

class SegviGenSegmenter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("SEG_MODEL",),
                "bake_mode": ("BOOLEAN", {"default": False}),
                "generate_uv": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "segvigen/seg"}),
            },
            "optional": {
                "input_3d": ("*",),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("glb_path", "preview_image")
    OUTPUT_NODE = True
    FUNCTION = "segment"
    CATEGORY = "SegviGen"
    def segment(self, model_name, bake_mode, generate_uv, filename_prefix="segvigen/seg", input_3d=None, image=None):
        import datetime
        import trimesh
        import shutil
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine output location
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
        os.makedirs(full_output_folder, exist_ok=True)
        
        # Temp workdir for vxz etc.
        workdir = os.path.join(folder_paths.get_temp_directory(), f"segvigen_{timestamp}")
        os.makedirs(workdir, exist_ok=True)
        
        in_glb = os.path.join(workdir, "input.glb")
        
        if input_3d is None:
            raise ValueError("No 3D input provided to 'input_3d'.")

        if isinstance(input_3d, dict):
            # Handle ComfyUI MESH type
            vertices = input_3d.get('vertices')
            faces = input_3d.get('faces')
            if vertices is not None and faces is not None:
                if len(vertices.shape) == 3: # Batched
                    vertices = vertices[0]
                    faces = faces[0]
                
                t_mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
                t_mesh.export(in_glb)
            else:
                raise ValueError("Unsupported dict format for input_3d. Expected MESH structure.")
        elif isinstance(input_3d, (str, os.PathLike)) or hasattr(input_3d, "path") or hasattr(input_3d, "source") or str(type(input_3d)).endswith("File3D'>"):
            # Handle string path or File3D object
            import re
            glb_path = None
            
            if hasattr(input_3d, "source"):
                glb_path = str(input_3d.source)
            elif hasattr(input_3d, "path"):
                glb_path = str(input_3d.path)
            else:
                s = str(input_3d)
                # Fallback: parse source='...' from string representation
                match = re.search(r"source=['\"](.*?)['\"]", s)
                if match:
                    glb_path = match.group(1)
                else:
                    glb_path = s
            
            # Robust path resolution
            if not os.path.isabs(glb_path):
                for search_dir in [folder_paths.get_input_directory(), folder_paths.get_output_directory(), folder_paths.get_temp_directory()]:
                    potential_path = os.path.join(search_dir, glb_path)
                    if os.path.exists(potential_path):
                        glb_path = potential_path
                        break
            
            if not os.path.isfile(glb_path):
                base_dir = os.path.abspath(os.path.join(folder_paths.base_path))
                potential_path = os.path.join(base_dir, glb_path)
                if os.path.exists(potential_path):
                    glb_path = potential_path

            if not os.path.isfile(glb_path):
                raise FileNotFoundError(f"Input GLB not found: {glb_path}")
            
            shutil.copy(glb_path, in_glb)
        else:
            raise ValueError(f"Unsupported type for input_3d: {type(input_3d)}")

        out_name = f"{filename}_{counter:05}_.glb"
        out_glb = os.path.join(full_output_folder, out_name)
        in_vxz = os.path.join(workdir, "input.vxz")

        effective_img_path = None
        if image is not None:
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.uint8(i))
            effective_img_path = os.path.join(workdir, "input_map.png")
            img.save(effective_img_path)

        # Ensure checkpoint is available
        load_checkpoint_helper(model_name)
        ckpt = os.path.join(MODELS_DIR, "checkpoints", model_name)

        # Prepare item for inference
        item = {
            "2d_map": bool(effective_img_path),
            "glb": in_glb,
            "input_vxz": in_vxz,
            "export_glb": out_glb,
            "bake": bake_mode,
            "generate_uv": generate_uv,
        }
        
        if effective_img_path:
            item["img"] = effective_img_path
        else:
            transforms_json = os.path.join(os.path.dirname(__file__), "data_toolkit/transforms.json")
            item["transforms"] = transforms_json
            item["img"] = os.path.join(workdir, "render.png")

        # Run inference
        run_segvigen_inference(ckpt, item)

        if not os.path.isfile(out_glb):
            raise RuntimeError("Segmentation failed: output GLB not found.")

        # Load preview image if generated
        preview_image = None
        preview_img_path = item["img"]
        if os.path.exists(preview_img_path):
            img = Image.open(preview_img_path).convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            preview_image = torch.from_numpy(img)[None,]

        return (out_glb, preview_image)

class SegviGenSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "segvigen/split"}),
                "color_quant_step": ("INT", {"default": 16, "min": 1, "max": 64}),
                "palette_sample_pixels": ("INT", {"default": 2000000}),
                "palette_min_pixels": ("INT", {"default": 500}),
                "palette_max_colors": ("INT", {"default": 256}),
                "palette_merge_dist": ("INT", {"default": 32}),
                "samples_per_face": ([1, 4], {"default": 4}),
                "flip_v": ("BOOLEAN", {"default": True}),
                "uv_wrap_repeat": ("BOOLEAN", {"default": True}),
                "min_faces_per_part": ("INT", {"default": 1}),
                "bake_transforms": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("part_glb_path",)
    OUTPUT_NODE = True
    FUNCTION = "split"
    CATEGORY = "SegviGen"

    def split(self, glb_path, filename_prefix="segvigen/split", **kwargs):
        if not os.path.isfile(glb_path):
            raise FileNotFoundError(f"Input GLB not found: {glb_path}")

        # Determine output location
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
        os.makedirs(full_output_folder, exist_ok=True)
        
        out_name = f"{filename}_{counter:05}_parts.glb"
        out_parts_glb = os.path.join(full_output_folder, out_name)

        splitter.split_glb_by_texture_palette_rgb(
            in_glb_path=glb_path,
            out_glb_path=out_parts_glb,
            **kwargs,
            debug_print=True
        )

        return (out_parts_glb,)

class SegviGenPreSimplification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_3d": ("*",),
                "target_reduction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "segvigen/simplified"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("simplified_glb_path",)
    OUTPUT_NODE = True 
    FUNCTION = "simplify"
    CATEGORY = "SegviGen"

    def simplify(self, input_3d, target_reduction, filename_prefix="segvigen/simplified"):
        glb_path = resolve_3d_path(input_3d)
        if not glb_path or not os.path.isfile(glb_path):
            raise FileNotFoundError(f"Input GLB not found: {glb_path}")

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
        os.makedirs(full_output_folder, exist_ok=True)
        
        out_name = f"{filename}_{counter:05}_.glb"
        out_glb = os.path.join(full_output_folder, out_name)

        fast_simplify_mesh(glb_path, out_glb, target_reduction=target_reduction)
        
        return {"ui": {"files": [{"filename": out_name, "subfolder": subfolder, "type": "output"}]}, "result": (out_glb,)}

NODE_CLASS_MAPPINGS = {
    "SegviGenModelLoader": SegviGenModelLoader,
    "SegviGenSegmenter": SegviGenSegmenter,
    "SegviGenSplitter": SegviGenSplitter,
    "SegviGenRmbg": SegviGenRmbg,
    "SegviGenConditioner": SegviGenConditioner,
    "SegviGenVoxelEncoder": SegviGenVoxelEncoder,
    "SegviGenSampler": SegviGenSampler,
    "SegviGenSamplerConfig": SegviGenSamplerConfig,
    "SegviGenStage5Renderer": SegviGenStage5Renderer,
    "SegviGenStage5Baker": SegviGenStage5Baker,
    "SegviGenPreSimplification": SegviGenPreSimplification
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegviGenModelLoader": "SegviGen Model Loader (Legacy)",
    "SegviGenSegmenter": "SegviGen Segmenter (Legacy)",
    "SegviGenSplitter": "SegviGen Splitter",
    "SegviGenRmbg": "SegviGen: Image Preprocessing",
    "SegviGenConditioner": "SegviGen: Conditioner",
    "SegviGenVoxelEncoder": "SegviGen:Voxel Encoder",
    "SegviGenSampler": "SegviGen: Sampling",
    "SegviGenSamplerConfig": "SegviGen Sampler Config",
    "SegviGenStage5Renderer": "SegviGen: Renderer",
    "SegviGenStage5Baker": "SegviGen: Baker",
    "SegviGenPreSimplification": "SegviGen: Pre-Simplification"
}
