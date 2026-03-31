import os
import sys
import bpy

def split_mesh_by_texture(input_filepath, output_dir):
    print(f"[*] Loading {input_filepath}...")
    
    # 1. Clear background scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 2. Import
    ext = os.path.splitext(input_filepath)[1].lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=input_filepath)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=input_filepath)
    else:
        print(f"Error: Unsupported extension '{ext}'. Need .glb or .obj")
        sys.exit(1)

    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if not meshes:
        print("Error: No mesh geometry found inside the file.")
        sys.exit(1)
        
    obj = meshes[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    mesh = obj.data
    uv_layer = mesh.uv_layers.active
    
    if not uv_layer:
        print("Error: Model has no UV mapping.")
        sys.exit(1)
        
    mat = obj.active_material
    if not mat or not mat.use_nodes:
        print("Error: Model has no active Material with an image texture.")
        sys.exit(1)
        
    img_node = next((n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and getattr(n, 'image', None)), None)
    if not img_node:
        print("Error: No Image Texture found baked into the material.")
        sys.exit(1)
        
    img = img_node.image
    width, height = img.size
    print(f"[*] Found baked texture map: {width}x{height}")
    
    # Fast pixel extraction
    pixels = [0.0] * (width * height * 4)
    img.pixels.foreach_get(pixels)

    print("[*] Analyzing face colors and mathematically generating bounds...")
    bpy.ops.object.mode_set(mode='OBJECT')
    
    unique_colors = {}
    mesh.materials.clear() 
    
    for poly in mesh.polygons:
        # Centroid sampling
        u_sum = v_sum = 0.0
        for loop_idx in poly.loop_indices:
            uv = uv_layer.data[loop_idx].uv
            u_sum += uv.x
            v_sum += uv.y
            
        u_center = u_sum / len(poly.loop_indices)
        v_center = v_sum / len(poly.loop_indices)
        
        u = u_center % 1.0
        v = v_center % 1.0
        x = min(int(u * width), width - 1)
        y = min(int(v * height), height - 1)
        idx = (y * width + x) * 4
        
        # Color crushing (1.0 = primary basic colors)
        steps = 1.0 
        r = round(pixels[idx] * steps) / steps
        g = round(pixels[idx+1] * steps) / steps
        b = round(pixels[idx+2] * steps) / steps
        
        color_key = (r, g, b)
        
        if color_key not in unique_colors:
            mat_name = f"Segment_{len(unique_colors) + 1}"
            new_mat = bpy.data.materials.new(name=mat_name)
            new_mat.use_nodes = True
            if new_mat.node_tree:
                bsdf = new_mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
            mesh.materials.append(new_mat)
            unique_colors[color_key] = len(mesh.materials) - 1
            
        poly.material_index = unique_colors[color_key]

    print(f"[*] Identified {len(unique_colors)} solid color groups.")
    
    print("[*] Slicing mesh geometry...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='MATERIAL')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 4. EXPORTING
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_filepath))[0]
    
    # Find all the new split pieces in the scene
    separated = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    
    print(f"[*] Successfully sliced into {len(separated)} pieces. Exporting independent files...")
    
    for i, part in enumerate(separated):
        bpy.ops.object.select_all(action='DESELECT')
        part.select_set(True)
        bpy.context.view_layer.objects.active = part
        
        out_path = os.path.join(output_dir, f"{basename}_part_{i+1}.glb")
        
        bpy.ops.export_scene.gltf(
            filepath=out_path,
            use_selection=True,
            export_materials='EXPORT',
            export_format='GLB'
        )
        print(f"  -> Saved: {out_path}")

    print(f"[*] FINISHED! All pieces saved to folder: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python headless_split.py <input_mesh.glb> <output_folder_path>")
        sys.exit(1)
        
    split_mesh_by_texture(sys.argv[1], sys.argv[2])
