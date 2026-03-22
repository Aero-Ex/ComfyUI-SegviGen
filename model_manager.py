import os
import json
import torch
import gc
from collections import OrderedDict
from trellis2 import models
from trellis2.pipelines.rembg import BiRefNet
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRELLIS_PIPELINE_JSON = os.path.join(ROOT_DIR, "data_toolkit/texturing_pipeline.json")
TRELLIS_TEX_FLOW = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16"
TRELLIS_SHAPE_ENC = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
TRELLIS_TEX_ENC = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
TRELLIS_SHAPE_DEC = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
TRELLIS_TEX_DEC = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"
DINO_PATH = "Aero-Ex/Dinov3"

try:
    import folder_paths
    COMFY_MODELS_DIR = folder_paths.models_dir
except ImportError:
    COMFY_MODELS_DIR = os.path.join(os.path.dirname(ROOT_DIR), "models")

class ModelManager:
    def __init__(self):
        self.loaded = False
        self.current_ckpt = None
        self.pipeline_args = None
        self.tex_slat_flow_model = None
        self.gen3dseg = None
        self.shape_encoder = None
        self.tex_encoder = None
        self.shape_decoder = None
        self.tex_decoder = None
        self.rembg_model = None
        self.image_cond_model = None

    def load_config(self):
        if self.loaded:
            return
        with open(TRELLIS_PIPELINE_JSON, "r") as f:
            pipeline_config = json.load(f)
        self.pipeline_args = pipeline_config['args']
        self.loaded = True

    def get_rembg(self):
        if self.rembg_model is None:
            rmbg_local = os.path.join(COMFY_MODELS_DIR, "briaai/RMBG-2.0")
            if os.path.isdir(rmbg_local):
                self.rembg_model = BiRefNet(model_name=rmbg_local, cache_dir=COMFY_MODELS_DIR)
            else:
                self.rembg_model = BiRefNet(model_name="yuvraj108c/RMBG-2.0", cache_dir=COMFY_MODELS_DIR)
            self.rembg_model.eval()
        return self.rembg_model

    def get_cond_model(self):
        if self.image_cond_model is None:
            self.image_cond_model = DinoV3FeatureExtractor(DINO_PATH, local_dir=COMFY_MODELS_DIR, subfolder="facebook/dinov3-vitl16-pretrain-lvd1689m")
            self.image_cond_model.eval()
        return self.image_cond_model

    def get_encoders_decoder(self):
        if self.shape_encoder is None:
            self.shape_encoder = models.from_pretrained(TRELLIS_SHAPE_ENC, local_dir=COMFY_MODELS_DIR).eval()
            self.tex_encoder = models.from_pretrained(TRELLIS_TEX_ENC, local_dir=COMFY_MODELS_DIR).eval()
            self.shape_decoder = models.from_pretrained(TRELLIS_SHAPE_DEC, local_dir=COMFY_MODELS_DIR).eval()
        return self.shape_encoder, self.tex_encoder, self.shape_decoder

    def get_gen3dseg(self, Gen3DSegClass):
        if self.gen3dseg is None:
            self.tex_slat_flow_model = models.from_pretrained(TRELLIS_TEX_FLOW, local_dir=COMFY_MODELS_DIR)
            self.gen3dseg = Gen3DSegClass(self.tex_slat_flow_model)
            if self.current_ckpt:
                self.apply_checkpoint()
            self.gen3dseg.eval()
        return self.gen3dseg

    def apply_checkpoint(self):
        if not self.gen3dseg or not self.current_ckpt:
            return
        ckpt_path = self.current_ckpt
        filename = os.path.basename(ckpt_path).replace(".ckpt", ".safetensors")
        local_safetensors = os.path.join(COMFY_MODELS_DIR, "checkpoints", filename)
        if os.path.exists(local_safetensors):
            from safetensors.torch import load_file
            state_dict = load_file(local_safetensors)
        else:
            state_dict = torch.load(ckpt_path)['state_dict']
        
        state_dict = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in state_dict.items()])
        self.gen3dseg.load_state_dict(state_dict)

    def get_tex_decoder(self):
        if self.tex_decoder is None:
            self.tex_decoder = models.from_pretrained(TRELLIS_TEX_DEC, local_dir=COMFY_MODELS_DIR).eval()
        return self.tex_decoder

    def unload(self, *attr_names):
        import torch
        import gc
        for name in attr_names:
            val = getattr(self, name, None)
            if val is not None:
                if hasattr(val, 'to'): 
                    val.to("cpu")
                setattr(self, name, None)
                del val
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def set_checkpoint(self, ckpt_path: str):
        if self.current_ckpt == ckpt_path:
            return
        self.current_ckpt = ckpt_path
        if self.gen3dseg:
            self.apply_checkpoint()

# Singleton instance
PIPE = ModelManager()
