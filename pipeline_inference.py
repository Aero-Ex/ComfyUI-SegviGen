import os
import torch
import torch.nn as nn
import numpy as np
import trellis2.modules.sparse as sp
from PIL import Image
from tqdm import tqdm
from .model_manager import PIPE
from .utils_3d import (
    build_simplified_work_glb, get_work_glb_path, process_glb_to_vxz,
    vxz_to_latent_slat, bake_to_mesh, slat_to_glb,
    EARLY_SIMPLIFY_ENABLED, EARLY_SIMPLIFY_TARGET_FACES, EARLY_SIMPLIFY_AGGRESSION
)

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
            return pred

    def interval_inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, sampler_params):
        guidance_strength = sampler_params['guidance_strength']
        guidance_interval = sampler_params['guidance_interval']
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, guidance_strength)
        else:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, 1)

    @torch.no_grad()
    def sample_once(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, t_prev, cond_dict, sampler_params):
        pred_v = self.interval_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, sampler_params)
        return x_t - (t - t_prev) * pred_v

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
        input_tex_feats_list, input_tex_coords_list = [], []
        shape_feats_list, shape_coords_list = [], []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            input_tex_feats_list.extend([x_t.feats[begin:end], tex_slats.feats[begin:end]])
            input_tex_coords_list.extend([x_t.coords[begin:end], tex_slats.coords[begin:end]])
            shape_feats_list.extend([shape_slats.feats[begin:end], shape_slats.feats[begin:end]])
            shape_coords_list.extend([shape_slats.coords[begin:end], shape_slats.coords[begin:end]])
            begin = end
        
        output_tex_slats = self.flow_model(sp.SparseTensor(torch.cat(input_tex_feats_list), torch.cat(input_tex_coords_list)), t, cond, sp.SparseTensor(torch.cat(shape_feats_list), torch.cat(shape_coords_list)))

        output_tex_feats_list, output_tex_coords_list = [], []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            output_tex_feats_list.append(output_tex_slats.feats[begin:end])
            output_tex_coords_list.append(output_tex_slats.coords[begin:end])
            begin = begin + 2 * coords_len
        return sp.SparseTensor(torch.cat(output_tex_feats_list), torch.cat(output_tex_coords_list))

def preprocess_image(rembg_model, input_img):
    if input_img.mode != "RGB":
        bg = Image.new("RGB", input_img.size, (255, 255, 255))
        bg.paste(input_img, mask=input_img.split()[3])
        input_img = bg
    
    max_size = max(input_img.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input_img = input_img.resize((int(input_img.width * scale), int(input_img.height * scale)), Image.Resampling.LANCZOS)
    
    output = rembg_model(input_img.convert('RGB'))
    alpha = np.array(output)[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output_np = np.array(output).astype(np.float32) / 255
    output_np = output_np[:, :, :3] * output_np[:, :, 3:4]
    return Image.fromarray((output_np * 255).astype(np.uint8))

def get_cond(image_cond_model, image):
    image_cond_model.image_size = 512
    cond = image_cond_model(image)
    return {'cond': cond, 'neg_cond': torch.zeros_like(cond)}

def tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, input_tex_slat, cond_dict):
    device = shape_slat.feats.device
    shape_std = torch.tensor(pipeline_args['shape_slat_normalization']['std'])[None].to(device)
    shape_mean = torch.tensor(pipeline_args['shape_slat_normalization']['mean'])[None].to(device)
    tex_std = torch.tensor(pipeline_args['tex_slat_normalization']['std'])[None].to(device)
    tex_mean = torch.tensor(pipeline_args['tex_slat_normalization']['mean'])[None].to(device)
    
    shape_slat = ((shape_slat - shape_mean) / shape_std)
    input_tex_slat = ((input_tex_slat - tex_mean) / tex_std)
    noise = sp.SparseTensor(torch.randn_like(input_tex_slat.feats), shape_slat.coords)
    output_tex_slat = sampler.sample(gen3dseg, noise, input_tex_slat, shape_slat, [shape_slat.coords.shape[0]], cond_dict, pipeline_args['tex_slat_sampler']['params'])
    return output_tex_slat * tex_std + tex_mean

def run_segvigen_inference(ckpt_path, item):
    PIPE.load_config()
    PIPE.set_checkpoint(ckpt_path)

    work_glb = item["glb"]
    if EARLY_SIMPLIFY_ENABLED:
        work_glb = os.path.join(os.path.dirname(item["input_vxz"]), "work.glb")
        build_simplified_work_glb(item["glb"], work_glb, EARLY_SIMPLIFY_TARGET_FACES, EARLY_SIMPLIFY_AGGRESSION)

    process_glb_to_vxz(item["glb"], item["input_vxz"])
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
    cond = {k: v.cpu() for k, v in cond.items()}
    PIPE.unload('image_cond_model')

    # 3. VXZ to Latent SLat
    shape_enc, tex_enc, shape_dec = PIPE.get_encoders_decoder()
    for m in [shape_enc, tex_enc, shape_dec]: m.cuda()
    shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(shape_enc, shape_dec, tex_enc, item["input_vxz"])
    shape_slat = sp.SparseTensor(shape_slat.feats.cpu(), shape_slat.coords.cpu())
    tex_slat = sp.SparseTensor(tex_slat.feats.cpu(), tex_slat.coords.cpu())
    PIPE.unload('shape_encoder', 'tex_encoder', 'shape_decoder')

    # 4. Sampling
    gen3dseg = PIPE.get_gen3dseg(Gen3DSeg)
    gen3dseg.cuda()
    output_tex_slat = tex_slat_sample_single(gen3dseg, Sampler(), PIPE.pipeline_args, 
                                           sp.SparseTensor(shape_slat.feats.cuda(), shape_slat.coords.cuda()), 
                                           sp.SparseTensor(tex_slat.feats.cuda(), tex_slat.coords.cuda()), 
                                           {k: v.cuda() for k, v in cond.items()})
    output_tex_slat_cpu = sp.SparseTensor(output_tex_slat.feats.cpu(), output_tex_slat.coords.cpu())
    PIPE.unload('gen3dseg', 'tex_slat_flow_model')

    # 5. Texture Decoding
    tex_decoder = PIPE.get_tex_decoder()
    tex_decoder.cuda()
    with torch.no_grad():
        tex_voxels = tex_decoder(sp.SparseTensor(output_tex_slat_cpu.feats.cuda(), output_tex_slat_cpu.coords.cuda()), 
                               guide_subs=[s.cuda() if isinstance(s, torch.Tensor) else s for s in subs]) * 0.5 + 0.5
        tex_voxels = [v.cpu() for v in tex_voxels]
    PIPE.unload('tex_decoder')

    if item.get("bake", False):
        bake_to_mesh(item["glb"], tex_voxels, item["export_glb"], generate_uv=item.get("generate_uv", False))
    else:
        glb = slat_to_glb(meshes, tex_voxels)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        glb.apply_transform(T)
        glb.export(item["export_glb"])
    
    # Final aggressive cleanup of any local large objects
    del shape_slat, tex_slat, meshes, subs, output_tex_slat, tex_voxels
    import gc
    gc.collect()
    torch.cuda.empty_cache()
