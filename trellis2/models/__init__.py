import importlib
from typing import Optional

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(model_path: str, local_dir: Optional[str] = None, load_weights: bool = True, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        model_path: Full Hugging Face path, e.g., 'microsoft/TRELLIS.2-4B/ckpts/shape_enc'
        local_dir: Base directory for models (e.g., '/home/aero/comfy/ComfyUI/models')
        load_weights: Whether to load the model weights. Set to False if you plan to load a custom checkpoint immediately after.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    import torch
    from huggingface_hub import hf_hub_download

    # Parse repo_id and the internal path
    # Example: 'microsoft/TRELLIS.2-4B/ckpts/shape_enc' -> repo_id='microsoft/TRELLIS.2-4B', subfolder='ckpts/shape_enc'
    parts = model_path.split('/')
    if len(parts) >= 3:
        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = "/".join(parts[2:])
        repo_name = parts[1]
    elif len(parts) == 2:
        repo_id = model_path
        subfolder = None
        repo_name = parts[1]
    else:
        repo_id = model_path
        subfolder = None
        repo_name = parts[0]

    config_file = None
    model_file = None

    # Determine local check paths
    if local_dir and subfolder:
        # 1. Try with repo_id (vendor/repo)
        path1 = os.path.join(local_dir, repo_id, subfolder)
        # 2. Try with repo_name (just repo)
        path2 = os.path.join(local_dir, repo_name, subfolder)
        # 3. Try directly (if local_dir already points to repo)
        path3 = os.path.join(local_dir, subfolder)
        
        for base in [path1, path2, path3]:
            potential_config = base + ".json"
            potential_model = base + ".safetensors"
            
            if os.path.exists(potential_config):
                config_file = potential_config
                print(f"Found local config at: {config_file}")
            
            if os.path.exists(potential_model) and load_weights:
                model_file = potential_model
                print(f"Found local model at: {model_file}")
            
            if config_file:
                break

    # Fallback to downloading if local files are missing
    if not config_file:
        filename = f"{subfolder}.json" if subfolder else "config.json"
        print(f"Downloading config for {repo_id}: {filename}...")
        config_file = hf_hub_download(repo_id, filename, local_dir=local_dir)

    if load_weights and not model_file:
        filename = f"{subfolder}.safetensors" if subfolder else "model.safetensors"
        print(f"Downloading model weights for {repo_id}: {filename}...")
        model_file = hf_hub_download(repo_id, filename, local_dir=local_dir)

    # Load and instantiate
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    
    if load_weights:
        if model_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(model_file, device="cpu")
        else:
            sd = torch.load(model_file, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
