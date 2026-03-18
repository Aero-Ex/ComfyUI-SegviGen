import importlib
import torch

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


def from_pretrained(path: str, local_dir: str = None, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        local_dir: Optional local directory to save/load the checkpoint.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from accelerate import init_empty_weights
    import comfy.utils
    
    config_file = None
    model_file = None
    is_local = False

    # Check if local_dir is provided and files exist there
    if local_dir:
        model_name = '/'.join(path.split('/')[2:])
        potential_config = os.path.join(local_dir, f"{model_name}.json")
        potential_model = os.path.join(local_dir, f"{model_name}.safetensors")
        if os.path.exists(potential_config) and os.path.exists(potential_model):
            config_file = potential_config
            model_file = potential_model
            is_local = True

    # If not found in local_dir, check the original path
    if not is_local:
        if os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors"):
            config_file = f"{path}.json"
            model_file = f"{path}.safetensors"
            is_local = True

    if not is_local:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json", local_dir=local_dir)
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors", local_dir=local_dir)

    with open(config_file, 'r') as f:
        config = json.load(f)
    
    with init_empty_weights():
        model = __getattr__(config['name'])(**config['args'], **kwargs)
    
    sd = comfy.utils.load_torch_file(model_file, device=torch.device("cpu"))
    model.load_state_dict(sd, strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
