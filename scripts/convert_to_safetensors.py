import torch
from safetensors.torch import save_file
import os
from collections import OrderedDict

def convert_segvigen_ckpt(ckpt_path, out_path):
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # SegviGen checkpoints typically have a 'state_dict' key
    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
    elif 'model' in checkpoint:
        sd = checkpoint['model']
    else:
        sd = checkpoint
        
    # Clean up keys (remove 'gen3dseg.' prefix if present)
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_key = k.replace("gen3dseg.", "")
        # Ensure tensors are in bf16 to match the size of existing safetensors (approx 2.6GB for 1.3B params)
        if isinstance(v, torch.Tensor):
            new_sd[new_key] = v.to(torch.bfloat16)
        else:
            new_sd[new_key] = v
            
    print(f"Saving {len(new_sd)} tensors to {out_path}...")
    save_file(new_sd, out_path)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert SegviGen .ckpt to .safetensors")
    parser.add_argument("input", help="Path to input .ckpt file")
    parser.add_argument("output", help="Path to output .safetensors file")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        convert_segvigen_ckpt(args.input, args.output)
    else:
        print(f"Error: {args.input} not found.")
