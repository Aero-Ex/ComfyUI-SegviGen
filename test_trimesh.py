import trimesh
import numpy as np
import sys
import os

def trimesh_texture_split(glb_file, quantization_steps=1.0):
    print(f"Loading {glb_file}...")
    scene = trimesh.load(glb_file, force='scene')
    
    geom_names = list(scene.geometry.keys())
    if not geom_names:
        print("No geometry found in the file.")
        return []
        
    geom_name = geom_names[0]
    mesh = scene.geometry[geom_name]
    
    if not hasattr(mesh, "visual") or not hasattr(mesh.visual, "material"):
        print("Mesh has no visual or material.")
        return []
        
    mat = mesh.visual.material
    pil_img = None
    if getattr(mat, "baseColorTexture", None) is not None:
        pil_img = mat.baseColorTexture
    elif getattr(mat, "image", None) is not None:
        pil_img = mat.image

    if pil_img is None:
        print("Material has no image texture.")
        return []
    img_np = np.array(pil_img.convert('RGB')) / 255.0  
    h, w = img_np.shape[:2]
    
    print(f"Image Resolution: {w}x{h}")
    face_uvs = mesh.visual.uv[mesh.faces].mean(axis=1)  
    
    u = face_uvs[:, 0] % 1.0
    v = face_uvs[:, 1] % 1.0
    
    pixel_x = np.clip((u * w).astype(int), 0, w - 1)
    pixel_y = np.clip(((1.0 - v) * h).astype(int), 0, h - 1)
    
    face_colors = img_np[pixel_y, pixel_x] 
    crushed_colors = np.round(face_colors * quantization_steps) / quantization_steps
    unique_colors, face_color_indices = np.unique(crushed_colors, axis=0, return_inverse=True)
    
    split_pieces = []
    
    from trimesh.visual import material, TextureVisuals
    
    for i, color in enumerate(unique_colors):
        mask = (face_color_indices == i)
        sub_mesh = mesh.submesh([mask], append=True)
        r, g, b = (color * 255).astype(np.uint8)
        
        # Solid PBR material export
        mat = material.PBRMaterial(baseColorFactor=[r, g, b, 255])
        sub_mesh.visual = TextureVisuals(material=mat)
        
        split_pieces.append(sub_mesh)

    print(f"Instantly mathematically sliced into {len(split_pieces)} independent pieces!")
    return split_pieces

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_trimesh.py <mesh.glb>")
        sys.exit(1)
        
    in_file = sys.argv[1]
    pieces = trimesh_texture_split(in_file, quantization_steps=1.0)
    
    out_path = in_file.replace(".glb", "_single_split.glb")
    if out_path == in_file:
        out_path += "_single_split.glb"
        
    scene_out = trimesh.Scene()
    for i, piece in enumerate(pieces):
        scene_out.add_geometry(piece, geom_name=f"part_{i}")
        print(f"Packed part_{i} ({len(piece.faces)} faces)")
        
    scene_out.export(out_path)
    print(f"\nSuccessfully exported all pieces into ONE file: {out_path}")
