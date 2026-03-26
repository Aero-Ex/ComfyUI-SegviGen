import os
import shutil
from pathlib import Path

def setup_examples():
    print("SegviGen Startup: Checking for example assets...")
    # Base path of the custom node
    base_path = os.path.dirname(os.path.realpath(__file__))
    examples_path = os.path.join(base_path, "examples")
    
    # ComfyUI root: /home/aero/comfy/ComfyUI/custom_nodes/ComfyUI-SegviGen/ -> /home/aero/comfy/ComfyUI/
    # Go up two levels from the custom node directory
    comfy_root = os.path.dirname(os.path.dirname(base_path))
    input_path = os.path.join(comfy_root, "input")
    input_3d_path = os.path.join(input_path, "3d")
    
    if not os.path.exists(examples_path):
        print(f"SegviGen Startup: Examples path not found: {examples_path}")
        return

    # Ensure input/3d exists
    os.makedirs(input_3d_path, exist_ok=True)
    
    # Define file type destinations
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    mesh_extensions = {'.glb', '.obj', '.ply', '.off', '.stl'}
    
    moved_count = 0
    
    for filename in os.listdir(examples_path):
        src_file = os.path.join(examples_path, filename)
        
        # Skip directories
        if not os.path.isfile(src_file):
            continue
            
        ext = os.path.splitext(filename)[1].lower()
        
        target_dir = None
        if ext in image_extensions:
            target_dir = input_path
        elif ext in mesh_extensions:
            target_dir = input_3d_path
        
        if target_dir:
            dst_file = os.path.join(target_dir, filename)
            
            # Only copy if it doesn't already exist to avoid unnecessary work every startup
            if not os.path.exists(dst_file):
                try:
                    shutil.copy2(src_file, dst_file)
                    moved_count += 1
                except Exception as e:
                    print(f"SegviGen Startup: Error copying {filename}: {e}")

    # Count total files present
    total_images = len([f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in image_extensions])
    total_meshes = len([f for f in os.listdir(input_3d_path) if os.path.splitext(f)[1].lower() in mesh_extensions])
    
    if moved_count > 0:
        print(f"SegviGen Startup: Copied {moved_count} new example files to ComfyUI input folders. Total images: {total_images}, Total meshes: {total_meshes}.")
    else:
        print(f"SegviGen Startup: All example assets are already present in input folders. Total images: {total_images}, Total meshes: {total_meshes}.")

# Execute the setup
setup_examples()
