import os
import struct
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import trimesh
from PIL import Image

# =========================
# 你只需要改这里
# =========================
# INPUT_GLB = "/mnt/pfs/users/huangzehuan/projects/SegviGen/examples/trellis2_output.glb"


# UID = "demonic_warrior_red_bronze_armor"
# UID = "playful_pose_white_top_portrait"
# UID = "african_inspired_metallic_silver_ensemble_with_headwrap"
# UID = "cyberpunk_bowser_motorcycle"
# UID = "crimson_battle_mecha_with_spikes"
UID = "black_lace_lingerie_ensemble"

INPUT_GLB = (
    f"/mnt/pfs/users/maxueqi/studio/datasets/dense_mesh/segvigen_bak/{UID}/output.glb"
)

# 只用 RGB（忽略透明度/alpha）
COLOR_QUANT_STEP = 16  # RGB 量化步长：0/4/8/16（越大越“合并”）
PALETTE_SAMPLE_PIXELS = 2_000_000
PALETTE_MIN_PIXELS = 500  # 少于该像素数的颜色当噪声丢掉（边界抗锯齿中间色）
PALETTE_MAX_COLORS = 256  # 最多保留多少个主颜色
PALETTE_MERGE_DIST = 32  # ✅ 合并 palette 内近似颜色（解决“看着同色却拆两块”）

SAMPLES_PER_FACE = 4  # 1 或 4（推荐 4，能明显减少边界采样误差）
FLIP_V = True  # glTF 常见需要 flip V
UV_WRAP_REPEAT = True  # True: repeat (mod 1)；False: clamp 到 [0,1]

MIN_FACES_PER_PART = 50
BAKE_TRANSFORMS = True
DEBUG_PRINT = True
# =========================


CHUNK_TYPE_JSON = 0x4E4F534A  # b'JSON'
CHUNK_TYPE_BIN = 0x004E4942  # b'BIN\0'


def _default_out_path(in_path: str) -> str:
    root, ext = os.path.splitext(in_path)
    if ext.lower() not in [".glb", ".gltf"]:
        ext = ".glb"
    return f"{root}_seg.glb"


def _quantize_rgb(rgb: np.ndarray, step: int) -> np.ndarray:
    """
    rgb: (...,3) uint8
    """
    if step is None or step <= 0:
        return rgb
    q = (rgb.astype(np.int32) + step // 2) // step * step
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q


def _load_glb_json_and_bin(glb_path: str) -> Tuple[dict, bytes]:
    data = open(glb_path, "rb").read()
    if len(data) < 12:
        raise RuntimeError("Invalid GLB: too small")

    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise RuntimeError("Not a GLB file (missing glTF header)")

    offset = 12
    gltf_json = None
    bin_chunk = None

    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset : offset + chunk_len]
        offset += chunk_len

        if chunk_type == CHUNK_TYPE_JSON:
            gltf_json = chunk_data.decode("utf-8", errors="replace")
        elif chunk_type == CHUNK_TYPE_BIN:
            bin_chunk = chunk_data

    if gltf_json is None:
        raise RuntimeError("GLB missing JSON chunk")
    if bin_chunk is None:
        raise RuntimeError("GLB missing BIN chunk")

    import json

    return json.loads(gltf_json), bin_chunk


def _extract_basecolor_texture_image(glb_path: str) -> np.ndarray:
    """
    从 GLB 内嵌资源里拿 baseColorTexture 的 PNG/JPG，返回 (H,W,4) uint8 RGBA
    """
    gltf, bin_chunk = _load_glb_json_and_bin(glb_path)

    materials = gltf.get("materials", [])
    textures = gltf.get("textures", [])
    images = gltf.get("images", [])
    buffer_views = gltf.get("bufferViews", [])

    if not materials:
        raise RuntimeError("No materials in GLB")

    # 这里按 material[0] 取 baseColorTexture（你的 glb 只有一个材质/primitive）
    pbr = materials[0].get("pbrMetallicRoughness", {})
    base_tex_index = pbr.get("baseColorTexture", {}).get("index", None)
    if base_tex_index is None:
        raise RuntimeError("Material has no baseColorTexture")

    if base_tex_index >= len(textures):
        raise RuntimeError("baseColorTexture index out of range")

    tex = textures[base_tex_index]
    img_index = tex.get("source", None)
    if img_index is None or img_index >= len(images):
        raise RuntimeError("Texture has no valid image source")

    img_info = images[img_index]
    bv_index = img_info.get("bufferView", None)
    mime = img_info.get("mimeType", None)
    if bv_index is None:
        uri = img_info.get("uri", None)
        raise RuntimeError(f"Image is not embedded (bufferView missing). uri={uri}")

    if bv_index >= len(buffer_views):
        raise RuntimeError("image.bufferView out of range")

    bv = buffer_views[bv_index]
    bo = int(bv.get("byteOffset", 0))
    bl = int(bv.get("byteLength", 0))
    img_bytes = bin_chunk[bo : bo + bl]

    if DEBUG_PRINT:
        print(
            f"[Texture] baseColorTextureIndex={base_tex_index}, imageIndex={img_index}, "
            f"bufferView={bv_index}, mime={mime}, bytes={len(img_bytes)}"
        )

    pil = Image.open(trimesh.util.wrap_as_stream(img_bytes)).convert("RGBA")
    return np.array(pil, dtype=np.uint8)


def _merge_palette_rgb(
    palette_rgb: np.ndarray, counts: np.ndarray, merge_dist: float
) -> np.ndarray:
    """
    对 palette 内 RGB 做“近似合并”，用 counts 作为权重更新中心。
    palette_rgb: (K,3) uint8
    counts: (K,) int
    """
    if palette_rgb is None or len(palette_rgb) == 0:
        return palette_rgb
    if merge_dist is None or merge_dist <= 0:
        return palette_rgb

    rgb = palette_rgb.astype(np.float32)
    counts = counts.astype(np.int64)

    order = np.argsort(-counts)

    centers = []
    center_w = []
    thr2 = float(merge_dist) * float(merge_dist)

    for idx in order:
        x = rgb[idx]
        w = int(counts[idx])

        if not centers:
            centers.append(x.copy())
            center_w.append(w)
            continue

        C = np.stack(centers, axis=0)  # (M,3)
        d2 = np.sum((C - x[None, :]) ** 2, axis=1)
        k = int(np.argmin(d2))

        if float(d2[k]) <= thr2:
            cw = center_w[k]
            centers[k] = (centers[k] * cw + x * w) / (cw + w)
            center_w[k] = cw + w
        else:
            centers.append(x.copy())
            center_w.append(w)

    merged = np.clip(np.rint(np.stack(centers, axis=0)), 0, 255).astype(np.uint8)

    if DEBUG_PRINT:
        print(
            f"[PaletteMerge] before={len(palette_rgb)} after={len(merged)} merge_dist={merge_dist}"
        )

    return merged


def _build_palette_rgb(tex_rgba: np.ndarray) -> np.ndarray:
    """
    从贴图中提取 RGB 主颜色调色板（忽略 alpha）。
    返回: (K,3) uint8
    """
    rgb = tex_rgba[:, :, :3].reshape(-1, 3)
    n = rgb.shape[0]

    if n > PALETTE_SAMPLE_PIXELS:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=PALETTE_SAMPLE_PIXELS, replace=False)
        rgb = rgb[idx]

    rgb = _quantize_rgb(rgb, COLOR_QUANT_STEP)

    uniq, counts = np.unique(rgb, axis=0, return_counts=True)
    order = np.argsort(-counts)
    uniq = uniq[order]
    counts = counts[order]

    keep = counts >= PALETTE_MIN_PIXELS
    uniq = uniq[keep]
    counts = counts[keep]

    if len(uniq) > PALETTE_MAX_COLORS:
        uniq = uniq[:PALETTE_MAX_COLORS]
        counts = counts[:PALETTE_MAX_COLORS]

    if DEBUG_PRINT:
        print(
            f"[Palette] quant_step={COLOR_QUANT_STEP} palette_size(before_merge)={len(uniq)} "
            f"min_pixels={PALETTE_MIN_PIXELS}"
        )
        for i in range(min(15, len(uniq))):
            r, g, b = [int(x) for x in uniq[i]]
            print(f"  {i:02d} rgb=({r},{g},{b}) count={int(counts[i])}")

    uniq = _merge_palette_rgb(uniq.astype(np.uint8), counts, PALETTE_MERGE_DIST)

    if DEBUG_PRINT:
        print(f"[Palette] palette_size(after_merge)={len(uniq)}")
        for i in range(min(15, len(uniq))):
            r, g, b = [int(x) for x in uniq[i]]
            print(f"  {i:02d} rgb=({r},{g},{b})")

    return uniq.astype(np.uint8)


def _unwrap_uv3_for_seam(uv3: np.ndarray) -> np.ndarray:
    """
    uv3: (F,3,2). 若跨 seam（跨度>0.5），把小于0.5的一侧 +1，避免均值跑到另一边。
    """
    out = uv3.copy()
    for d in range(2):
        v = out[:, :, d]
        vmin = v.min(axis=1)
        vmax = v.max(axis=1)
        seam = (vmax - vmin) > 0.5
        if np.any(seam):
            vv = v[seam]
            vv = np.where(vv < 0.5, vv + 1.0, vv)
            out[seam, :, d] = vv
    return out


def _barycentric_samples(uv3: np.ndarray, samples_per_face: int) -> np.ndarray:
    """
    uv3: (F,3,2)
    return: (F,S,2)
    """
    uv3 = _unwrap_uv3_for_seam(uv3)

    if samples_per_face == 1:
        w = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        uvs = uv3[:, 0, :] * w[0] + uv3[:, 1, :] * w[1] + uv3[:, 2, :] * w[2]
        return uvs[:, None, :]

    # 4 个点：中心 + 三个靠近顶点的内点（尽量远离边界抗锯齿带）
    ws = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float32,
    )
    uvs = (
        uv3[:, None, 0, :] * ws[None, :, 0, None]
        + uv3[:, None, 1, :] * ws[None, :, 1, None]
        + uv3[:, None, 2, :] * ws[None, :, 2, None]
    )
    return uvs


def _wrap_or_clamp_uv(uv: np.ndarray) -> np.ndarray:
    if UV_WRAP_REPEAT:
        return np.mod(uv, 1.0)
    return np.clip(uv, 0.0, 1.0)


def _sample_texture_nearest_rgb(tex_rgba: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """
    tex_rgba: (H,W,4) uint8
    uv: (N,2) float
    return: (N,3) uint8
    """
    h, w = tex_rgba.shape[0], tex_rgba.shape[1]
    uv = _wrap_or_clamp_uv(uv)

    u = uv[:, 0]
    v = uv[:, 1]
    if FLIP_V:
        v = 1.0 - v

    x = np.rint(u * (w - 1)).astype(np.int32)
    y = np.rint(v * (h - 1)).astype(np.int32)
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    return tex_rgba[y, x, :3].astype(np.uint8)


def _map_to_palette_rgb(
    colors_rgb: np.ndarray, palette_rgb: np.ndarray, chunk: int = 20000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    把采样到的 RGB 映射到最近的 palette RGB.
    如果 palette 为空，则用 colors_rgb 的 unique 作为“临时 palette”.
    返回:
      labels: (N,) int
      used_palette_rgb: (K,3) uint8
    """
    if palette_rgb is None or len(palette_rgb) == 0:
        uniq, inv = np.unique(colors_rgb, axis=0, return_inverse=True)
        return inv.astype(np.int32), uniq.astype(np.uint8)

    c = colors_rgb.astype(np.float32)
    p = palette_rgb.astype(np.float32)

    out = np.empty((c.shape[0],), dtype=np.int32)
    for i in range(0, c.shape[0], chunk):
        cc = c[i : i + chunk]
        d2 = ((cc[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
        out[i : i + chunk] = np.argmin(d2, axis=1).astype(np.int32)

    return out, palette_rgb


def _face_labels_from_texture_rgb(
    mesh: trimesh.Trimesh,
    tex_rgba: np.ndarray,
    palette_rgb: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    用 TEXCOORD_0 + baseColorTexture，为每个 face 采样 RGB，并映射到 palette label。
    返回:
      face_label: (F,) int
      label_rgb: (K,3) uint8
    """
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        return None

    uv = np.asarray(uv, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] != len(mesh.vertices):
        return None

    faces = mesh.faces
    uv3 = uv[faces]  # (F,3,2)

    uvs = _barycentric_samples(uv3, SAMPLES_PER_FACE)  # (F,S,2)
    F, S = uvs.shape[0], uvs.shape[1]
    flat_uv = uvs.reshape(-1, 2)

    sampled_rgb = _sample_texture_nearest_rgb(tex_rgba, flat_uv)  # (F*S,3)
    sampled_rgb = _quantize_rgb(sampled_rgb, COLOR_QUANT_STEP)

    sample_label, used_palette = _map_to_palette_rgb(sampled_rgb, palette_rgb)
    sample_label = sample_label.reshape(F, S)

    if S == 1:
        return sample_label[:, 0].astype(np.int32), used_palette

    # 4 票投票（向量化）
    l0, l1, l2, l3 = (
        sample_label[:, 0],
        sample_label[:, 1],
        sample_label[:, 2],
        sample_label[:, 3],
    )
    c0 = 1 + (l0 == l1) + (l0 == l2) + (l0 == l3)
    c1 = 1 + (l1 == l0) + (l1 == l2) + (l1 == l3)
    c2 = 1 + (l2 == l0) + (l2 == l1) + (l2 == l3)
    c3 = 1 + (l3 == l0) + (l3 == l1) + (l3 == l2)

    counts = np.stack([c0, c1, c2, c3], axis=1)  # (F,4)
    vals = np.stack([l0, l1, l2, l3], axis=1)  # (F,4)
    best = vals[np.arange(F), np.argmax(counts, axis=1)]
    return best.astype(np.int32), used_palette


# =========================
# 拓扑纠错
# =========================

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _get_physical_face_adjacency(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    忽略 UV 接缝，计算纯物理空间上的面片相邻关系。
    """
    # 1. 四舍五入顶点坐标（处理浮点数微小误差），找出空间中真正唯一的物理顶点
    v_rounded = np.round(mesh.vertices, decimals=3)
    v_unique, inv_indices = np.unique(v_rounded, axis=0, return_inverse=True)

    # 2. 将原本的面片索引，映射到这些“唯一物理顶点”上
    # 这样，跨越 UV 接缝的面片，此时它们引用的顶点索引就变成一样的了
    physical_faces = inv_indices[mesh.faces]

    # 3. 创建一个临时的“影子网格”（process=False 极其重要，防止 trimesh 内部重排面片）
    tmp_mesh = trimesh.Trimesh(vertices=v_unique, faces=physical_faces, process=False)

    # 返回影子网格的物理相邻边
    return tmp_mesh.face_adjacency


def smooth_face_labels_by_topology(
    mesh: trimesh.Trimesh, face_label: np.ndarray, min_faces: int = 50
) -> np.ndarray:
    """
    通过真实的 3D 物理拓扑关系过滤飞点，跨越 UV 接缝合并色块。

    Phase 1: 在同色连通图上，把挨着大块的小块吞并到大块中。
    Phase 2: 对残留小块（邻居全是小块），回退到全物理邻接，
             按物理邻居中的多数 label 吞并。
    Phase 3: 对完全孤立的面片（无物理邻接边），按面片质心距离
             找最近的非孤立面片，继承其 label。
    """
    labels = face_label.copy()
    edges = _get_physical_face_adjacency(mesh)
    F = len(mesh.faces)

    # ---- Phase 1: 同色连通域平滑 ----
    for iteration in range(3):
        same_label = labels[edges[:, 0]] == labels[edges[:, 1]]
        sub_edges = edges[same_label]

        if len(sub_edges) > 0:
            data = np.ones(len(sub_edges), dtype=bool)
            graph = coo_matrix((data, (sub_edges[:, 0], sub_edges[:, 1])), shape=(F, F))
            graph = graph.maximum(graph.T)
            n_components, comp_labels = connected_components(graph, directed=False)
        else:
            n_components = F
            comp_labels = np.arange(F)

        comp_sizes = np.bincount(comp_labels, minlength=n_components)
        small_comps = np.where(comp_sizes < min_faces)[0]
        if len(small_comps) == 0:
            break

        is_small = np.isin(comp_labels, small_comps)

        mask0 = is_small[edges[:, 0]]
        mask1 = is_small[edges[:, 1]]

        boundary_edges_0 = edges[mask0 & ~mask1]
        boundary_edges_1 = edges[mask1 & ~mask0]

        b_inner = np.concatenate([boundary_edges_0[:, 0], boundary_edges_1[:, 1]])
        b_outer = np.concatenate([boundary_edges_0[:, 1], boundary_edges_1[:, 0]])

        if len(b_inner) == 0:
            break

        outer_labels = labels[b_outer]
        inner_comps = comp_labels[b_inner]

        for cid in np.unique(inner_comps):
            cid_mask = inner_comps == cid
            surrounding_labels = outer_labels[cid_mask]
            if len(surrounding_labels) > 0:
                best_label = np.bincount(surrounding_labels).argmax()
                labels[comp_labels == cid] = best_label

    # ---- Phase 2: 用全物理邻接处理残留小块 ----
    # 重新计算同色连通域，找出还残留的小块
    same_label = labels[edges[:, 0]] == labels[edges[:, 1]]
    sub_edges = edges[same_label]
    if len(sub_edges) > 0:
        data = np.ones(len(sub_edges), dtype=bool)
        graph = coo_matrix((data, (sub_edges[:, 0], sub_edges[:, 1])), shape=(F, F))
        graph = graph.maximum(graph.T)
        n_components, comp_labels = connected_components(graph, directed=False)
    else:
        n_components = F
        comp_labels = np.arange(F)

    comp_sizes = np.bincount(comp_labels, minlength=n_components)
    small_comps_set = set(np.where(comp_sizes < min_faces)[0])

    if small_comps_set:
        is_small = np.array([comp_labels[i] in small_comps_set for i in range(F)])

        # 构建全物理邻接查找表: face -> set of neighbor faces
        adj = defaultdict(set)
        for e0, e1 in edges:
            adj[int(e0)].add(int(e1))
            adj[int(e1)].add(int(e0))

        # 迭代：每轮让小块面片从物理邻居（忽略颜色）中投票取多数 label
        for _ in range(3):
            changed = False
            small_comps_now = set(
                int(c)
                for c in range(n_components)
                if comp_sizes[c] < min_faces and c in small_comps_set
            )
            if not small_comps_now:
                break

            for cid in small_comps_now:
                cid_faces = np.where(comp_labels == cid)[0]
                # 收集所有物理邻居中不属于本连通域的面片的 label
                neighbor_labels = []
                for fi in cid_faces:
                    for nf in adj[int(fi)]:
                        if comp_labels[nf] != cid:
                            neighbor_labels.append(labels[nf])

                if len(neighbor_labels) > 0:
                    best_label = int(np.bincount(neighbor_labels).argmax())
                    labels[cid_faces] = best_label
                    changed = True

            if not changed:
                break

            # 重新计算连通域
            same_label = labels[edges[:, 0]] == labels[edges[:, 1]]
            sub_edges = edges[same_label]
            if len(sub_edges) > 0:
                data = np.ones(len(sub_edges), dtype=bool)
                graph = coo_matrix(
                    (data, (sub_edges[:, 0], sub_edges[:, 1])), shape=(F, F)
                )
                graph = graph.maximum(graph.T)
                n_components, comp_labels = connected_components(graph, directed=False)
            else:
                n_components = F
                comp_labels = np.arange(F)
            comp_sizes = np.bincount(comp_labels, minlength=n_components)
            small_comps_set = set(np.where(comp_sizes < min_faces)[0])

    # ---- Phase 3: 完全孤立面片（无物理邻接边），按质心距离继承 label ----
    same_label = labels[edges[:, 0]] == labels[edges[:, 1]]
    sub_edges = edges[same_label]
    if len(sub_edges) > 0:
        data = np.ones(len(sub_edges), dtype=bool)
        graph = coo_matrix((data, (sub_edges[:, 0], sub_edges[:, 1])), shape=(F, F))
        graph = graph.maximum(graph.T)
        _, comp_labels = connected_components(graph, directed=False)
    else:
        comp_labels = np.arange(F)
    comp_sizes = np.bincount(comp_labels)
    orphan_comps = set(np.where(comp_sizes < min_faces)[0])

    if orphan_comps:
        orphan_mask = np.array([comp_labels[i] in orphan_comps for i in range(F)])
        non_orphan_mask = ~orphan_mask
        if non_orphan_mask.any() and orphan_mask.any():
            centroids = mesh.triangles_center
            orphan_indices = np.where(orphan_mask)[0]
            non_orphan_indices = np.where(non_orphan_mask)[0]
            non_orphan_centroids = centroids[non_orphan_indices]

            for oi in orphan_indices:
                dists = np.linalg.norm(non_orphan_centroids - centroids[oi], axis=1)
                nearest = non_orphan_indices[np.argmin(dists)]
                labels[oi] = labels[nearest]

        if DEBUG_PRINT:
            n_orphan = int(orphan_mask.sum())
            print(f"  [Phase3] Assigned {n_orphan} orphan faces by centroid proximity")

    return labels


# =========================
# 分割主函数
# =========================


# def split_glb_by_texture_palette_rgb(
#     in_glb_path: str,
#     out_glb_path: Optional[str] = None,
#     min_faces_per_part: int = 1,
#     bake_transforms: bool = True,
# ) -> str:
#     """
#     输入：glb（无 COLOR_0，但有 baseColorTexture + TEXCOORD_0）
#     输出：先从贴图提取 RGB 主色 palette（忽略 alpha），再按 palette label 分割
#     """
#     if out_glb_path is None:
#         out_glb_path = _default_out_path(in_glb_path)

#     tex_rgba = _extract_basecolor_texture_image(in_glb_path)
#     palette_rgb = _build_palette_rgb(tex_rgba)

#     scene = trimesh.load(in_glb_path, force="scene", process=False)
#     out_scene = trimesh.Scene()

#     part_count = 0
#     base = os.path.splitext(os.path.basename(in_glb_path))[0]

#     for node_name in scene.graph.nodes_geometry:
#         geom_name = scene.graph[node_name][1]
#         if geom_name is None:
#             continue

#         geom = scene.geometry.get(geom_name, None)
#         if geom is None or not isinstance(geom, trimesh.Trimesh):
#             continue

#         mesh = geom.copy()

#         if bake_transforms:
#             T, _ = scene.graph.get(node_name)
#             if T is not None:
#                 mesh.apply_transform(T)

#         res = _face_labels_from_texture_rgb(mesh, tex_rgba, palette_rgb)
#         if res is None:
#             if DEBUG_PRINT:
#                 print(f"[{node_name}] no uv / cannot sample -> keep orig")
#             out_scene.add_geometry(mesh, geom_name=f"{base}__{node_name}__orig")
#             continue

#         face_label, label_rgb = res

#         # =========================
#         # 🔥 新增调用：进行拓扑纠错，合并飞点
#         # =========================
#         face_label = smooth_face_labels_by_topology(mesh, face_label, min_faces=100)

#         if DEBUG_PRINT:
#             uniq_labels, cnts = np.unique(face_label, return_counts=True)
#             order = np.argsort(-cnts)
#             print(
#                 f"[{node_name}] faces={len(mesh.faces)} labels_used={len(uniq_labels)} palette_size={len(label_rgb)}"
#             )
#             for i in order[:10]:
#                 lab = int(uniq_labels[i])
#                 r, g, b = (
#                     [int(x) for x in label_rgb[lab]]
#                     if 0 <= lab < len(label_rgb)
#                     else (0, 0, 0)
#                 )
#                 print(f"  label={lab} rgb=({r},{g},{b}) faces={int(cnts[i])}")

#         groups = defaultdict(list)
#         for fi, lab in enumerate(face_label):
#             groups[int(lab)].append(fi)

#         for lab, face_ids in groups.items():
#             if len(face_ids) < min_faces_per_part:
#                 continue

#             sub = mesh.submesh(
#                 [np.array(face_ids, dtype=np.int64)], append=True, repair=False
#             )
#             if sub is None:
#                 continue
#             if isinstance(sub, (list, tuple)):
#                 if not sub:
#                     continue
#                 sub = sub[0]

#             if 0 <= lab < len(label_rgb):
#                 r, g, b = [int(x) for x in label_rgb[lab]]
#                 part_name = f"{base}__{node_name}__label_{lab}__rgb_{r}_{g}_{b}"
#             else:
#                 part_name = f"{base}__{node_name}__label_{lab}"

#             out_scene.add_geometry(sub, geom_name=part_name)
#             part_count += 1

#     if part_count == 0:
#         if DEBUG_PRINT:
#             print("[INFO] part_count==0, fallback to original scene export.")
#         out_scene = scene

#     out_scene.export(out_glb_path)
#     return out_glb_path


def split_glb_by_texture_palette_rgb(
    in_glb_path: str,
    out_glb_path: Optional[str] = None,
    min_faces_per_part: int = 1,
    bake_transforms: bool = True,
    color_quant_step: int = 16,
    palette_sample_pixels: int = 2_000_000,
    palette_min_pixels: int = 500,
    palette_max_colors: int = 256,
    palette_merge_dist: int = 32,
    samples_per_face: int = 4,
    flip_v: bool = True,
    uv_wrap_repeat: bool = True,
    transition_conf_thresh: float = 1.0,
    transition_prop_iters: int = 6,
    transition_neighbor_min: int = 1,
    small_component_action: str = "reassign",
    small_component_min_faces: int = 50,
    postprocess_iters: int = 3,
    debug_print: bool = True,
) -> str:
    """
    Input: GLB (no COLOR_0, but with baseColorTexture + TEXCOORD_0)
    Output: Split based on palette labels derived from baseColorTexture
    """
    if out_glb_path is None:
        out_glb_path = _default_out_path(in_glb_path)

    tex_rgba = _extract_basecolor_texture_image(in_glb_path)
    palette_rgb = _build_palette_rgb(tex_rgba)

    scene = trimesh.load(in_glb_path, force="scene", process=False)
    out_scene = trimesh.Scene()

    part_count = 0
    base = os.path.splitext(os.path.basename(in_glb_path))[0]

    for node_name in scene.graph.nodes_geometry:
        geom_name = scene.graph[node_name][1]
        if geom_name is None:
            continue

        geom = scene.geometry.get(geom_name, None)
        if geom is None or not isinstance(geom, trimesh.Trimesh):
            continue

        mesh = geom.copy()

        if bake_transforms:
            T, _ = scene.graph.get(node_name)
            if T is not None:
                mesh.apply_transform(T)

        res = _face_labels_from_texture_rgb(mesh, tex_rgba, palette_rgb)
        if res is None:
            if debug_print:
                print(f"[{node_name}] no uv / cannot sample -> keep orig")
            out_scene.add_geometry(mesh, geom_name=f"{base}__{node_name}__orig")
            continue

        face_label, label_rgb = res

        # =========================
        # 🔥 New: Apply topology correction to merge small disconnected components
        # =========================
        face_label = smooth_face_labels_by_topology(mesh, face_label, min_faces=100)

        if debug_print:
            uniq_labels, cnts = np.unique(face_label, return_counts=True)
            order = np.argsort(-cnts)
            print(
                f"[{node_name}] faces={len(mesh.faces)} labels_used={len(uniq_labels)} palette_size={len(label_rgb)}"
            )
            for i in order[:10]:
                lab = int(uniq_labels[i])
                r, g, b = (
                    [int(x) for x in label_rgb[lab]]
                    if 0 <= lab < len(label_rgb)
                    else (0, 0, 0)
                )
                print(f"  label={lab} rgb=({r},{g},{b}) faces={int(cnts[i])}")

        groups = defaultdict(list)
        for fi, lab in enumerate(face_label):
            groups[int(lab)].append(fi)

        for lab, face_ids in groups.items():
            if len(face_ids) < min_faces_per_part:
                continue

            sub = mesh.submesh([np.array(face_ids, dtype=np.int64)], append=True, repair=False)
            if sub is None:
                continue
            if isinstance(sub, (list, tuple)):
                if not sub:
                    continue
                sub = sub[0]

            if 0 <= lab < len(label_rgb):
                r, g, b = [int(x) for x in label_rgb[lab]]
                part_name = f"{base}__{node_name}__label_{lab}__rgb_{r}_{g}_{b}"
            else:
                part_name = f"{base}__{node_name}__label_{lab}"

            out_scene.add_geometry(sub, geom_name=part_name)
            part_count += 1

    if part_count == 0:
        if debug_print:
            print("[INFO] part_count==0, fallback to original scene export.")
        out_scene = scene

    out_scene.export(out_glb_path)
    return out_glb_path

def main():
    out_path = split_glb_by_texture_palette_rgb(
        INPUT_GLB,
        out_glb_path=None,
        min_faces_per_part=MIN_FACES_PER_PART,
        bake_transforms=BAKE_TRANSFORMS,
    )
    print("Done. Exported:", out_path)


if __name__ == "__main__":
    main()
