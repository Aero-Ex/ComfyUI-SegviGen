import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "flash_attn_3"

import urllib.request

COMFY_MODELS_DIR = "/home/aero/comfy/ComfyUI/models"


CKPT_FULL_SEG = os.path.join(COMFY_MODELS_DIR, "checkpoints", "full_seg.safetensors")
CKPT_W_2D_MAP = os.path.join(COMFY_MODELS_DIR, "checkpoints", "full_seg_w_2d_map.safetensors")

def _has_local_any(ckpt_path):
    return os.path.exists(ckpt_path)

if not _has_local_any(CKPT_FULL_SEG):
    print(f"Downloading {CKPT_FULL_SEG}...")
    os.makedirs(os.path.dirname(CKPT_FULL_SEG), exist_ok=True)
    urllib.request.urlretrieve(
        "https://huggingface.co/Aero-Ex/SegviGen/resolve/main/full_seg.safetensors",
        CKPT_FULL_SEG,
    )

if not _has_local_any(CKPT_W_2D_MAP):
    print(f"Downloading {CKPT_W_2D_MAP}...")
    os.makedirs(os.path.dirname(CKPT_W_2D_MAP), exist_ok=True)
    urllib.request.urlretrieve(
        "https://huggingface.co/Aero-Ex/SegviGen/resolve/main/full_seg_w_2d_map.safetensors",
        CKPT_W_2D_MAP,
    )

import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import inference_full as inf
import split as splitter


TRANSFORMS_JSON = "./data_toolkit/transforms.json"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(ROOT_DIR, "_tmp_gradio_seg")
EXAMPLES_CACHE_DIR = os.path.join(TMP_DIR, "examples_cache")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(EXAMPLES_CACHE_DIR, exist_ok=True)

os.environ["GRADIO_TEMP_DIR"] = TMP_DIR
os.environ["GRADIO_EXAMPLES_CACHE"] = EXAMPLES_CACHE_DIR

import gradio as gr

EXAMPLES_DIR = "examples"


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _normalize_path(x):
    """
    Compatible with different Gradio versions: File/Model3D might be str / dict / object
    """
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("name") or x.get("path") or x.get("data")
    return getattr(x, "name", None) or getattr(x, "path", None) or None


def _raise_user_error(msg: str):
    if hasattr(gr, "Error"):
        raise gr.Error(msg)
    raise RuntimeError(msg)


def _collect_examples(example_dir: str) -> List[List[str]]:
    """
    Scan example_dir for pairs: <name>.glb + <name>.png
    Return a list of examples: [[glb_path, png_path], ...]
    """
    d = Path(example_dir)
    if not d.is_dir():
        return []

    examples: List[List[str]] = []

    # Search recursively in case you add subfolders later
    glb_files = sorted(d.rglob("*.glb"))
    for glb_path in glb_files:
        png_path = glb_path.with_suffix(".png")
        if png_path.is_file():
            examples.append([str(glb_path), str(png_path)])
        # If png is missing, skip to keep examples consistent (2 inputs required)

    return examples


# Build examples once at startup
FULL_SEG_EXAMPLES = _collect_examples(EXAMPLES_DIR)


def _update_img_box(mode: str):
    is_generate = str(mode).startswith("Generate")

    if is_generate:
        return gr.update(
            interactive=False,
            label="2D Segmentation Map",
            value=None,
        )

    return gr.update(
        interactive=True,
        label="2D Segmentation Map",
        value=None,
    )


def run_seg(glb_in, map_mode, img_in, ckpt_choice="Default (Auto)", bake_mode=False, generate_uv=False):
    """
    Segment button: generates whole segmented GLB and displays in the second box.

    Auto mode:
        - If image is provided -> use CKPT_W_2D_MAP
        - If image is not provided -> keep original logic and use CKPT_FULL_SEG

    Generate mode:
        - Generate a 2D map first
        - Use the generated map as if it were the uploaded image
        - Therefore use CKPT_W_2D_MAP

    Returns:
        segmented_glb_path, segmented_glb_path(state), image_preview
    """
    try:
        glb_path = _normalize_path(glb_in)
        img_path = _normalize_path(img_in)

        if (glb_path is None) or (not os.path.isfile(glb_path)):
            _raise_user_error("Please upload an input GLB first.")

        # Create workdir
        workdir = os.path.join(TMP_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        _ensure_dir(workdir)

        in_glb = os.path.join(workdir, "input.glb")
        shutil.copy(glb_path, in_glb)

        out_glb = os.path.join(workdir, "segmented.glb")
        in_vxz = os.path.join(workdir, "input.vxz")

        effective_img_path = None
        is_generate = str(map_mode).startswith("Generate")

        # New path: generate a 2D map first, then use it as input image
        if is_generate:
            render_img = os.path.join(workdir, "render.png")
            generated_img = os.path.join(workdir, "2d_map_generated.png")

            inf.generate_2d_map_from_glb(
                glb_path=in_glb,
                transforms_path=TRANSFORMS_JSON,
                out_img_path=generated_img,
                render_img_path=render_img,
            )

            if not os.path.isfile(generated_img):
                _raise_user_error("2D map generation failed: generated image not found.")

            effective_img_path = generated_img

        # Original logic is preserved here
        elif img_path is not None and os.path.isfile(img_path):
            copied_img = os.path.join(workdir, "2d_map.png")
            shutil.copy(img_path, copied_img)
            effective_img_path = copied_img

        # ---------------- Checkpoint Selection ----------------
        if ckpt_choice == "full_seg":
            ckpt = CKPT_FULL_SEG
        elif ckpt_choice == "full_seg_w_2d_map":
            ckpt = CKPT_W_2D_MAP
        else:
            # Default (Auto) logic
            if effective_img_path is not None and os.path.isfile(effective_img_path):
                ckpt = CKPT_W_2D_MAP
            else:
                ckpt = CKPT_FULL_SEG

        # Keep the original branching logic for item initialization
        if effective_img_path is not None and os.path.isfile(effective_img_path):
            item = {
                "2d_map": True,
                "glb": in_glb,
                "input_vxz": in_vxz,
                "img": effective_img_path,
                "export_glb": out_glb,
                "bake": bake_mode,
                "generate_uv": generate_uv,
            }
            preview_img = effective_img_path
        else:
            render_img = os.path.join(workdir, "render.png")
            item = {
                "2d_map": False,
                "glb": in_glb,
                "input_vxz": in_vxz,
                "transforms": TRANSFORMS_JSON,
                "img": render_img,
                "export_glb": out_glb,
                "bake": bake_mode,
                "generate_uv": generate_uv,
            }
            preview_img = None

        inf.inference_with_loaded_models(ckpt, item)

        if not os.path.isfile(out_glb):
            _raise_user_error("Export failed: output glb not found.")

        # Apply X90 rotation for whole segmented output
        # _apply_root_x90_rotation_glb(out_glb)

        return out_glb, out_glb, preview_img

    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(err)
        raise


def run_refine_segmentation(
    seg_glb_path_state,
    color_quant_step,
    palette_sample_pixels,
    palette_min_pixels,
    palette_max_colors,
    palette_merge_dist,
    samples_per_face,
    flip_v,
    uv_wrap_repeat,
    transition_conf_thresh,
    transition_prop_iters,
    transition_neighbor_min,
    small_component_action,
    small_component_min_faces,
    postprocess_iters,
    min_faces_per_part,
    bake_transforms,
):
    """
    Refine Segmentation button: splits the segmented GLB into smaller parts GLB and displays in the fourth box.
    """
    try:
        seg_glb_path = seg_glb_path_state if isinstance(seg_glb_path_state, str) else None
        if (seg_glb_path is None) or (not os.path.isfile(seg_glb_path)):
            _raise_user_error("Please run Segmentation first (the segmented GLB is missing).")

        out_dir = os.path.dirname(seg_glb_path)
        out_parts_glb = os.path.join(out_dir, "segmented_parts.glb")

        splitter.split_glb_by_texture_palette_rgb(
            in_glb_path=seg_glb_path,
            out_glb_path=out_parts_glb,
            min_faces_per_part=min_faces_per_part,
            bake_transforms=bool(bake_transforms),
            color_quant_step=color_quant_step,
            palette_sample_pixels=palette_sample_pixels,
            palette_min_pixels=palette_min_pixels,
            palette_max_colors=palette_max_colors,
            palette_merge_dist=palette_merge_dist,
            samples_per_face=samples_per_face,
            flip_v=flip_v,
            uv_wrap_repeat=uv_wrap_repeat,
            transition_conf_thresh=transition_conf_thresh,
            transition_prop_iters=transition_prop_iters,
            transition_neighbor_min=transition_neighbor_min,
            small_component_action=small_component_action,
            small_component_min_faces=small_component_min_faces,
            postprocess_iters=postprocess_iters,
            debug_print=True,
        )

        if not os.path.isfile(out_parts_glb):
            _raise_user_error("Split failed: output parts glb not found.")

        # If bake_transforms=False, split output will not have the wrapper transform baked, so we need to apply X90 rotation fix
        # if (not bool(bake_transforms)) and APPLY_OUTPUT_X90_FIX:
        #     _apply_root_x90_rotation_glb(out_parts_glb)

        return out_parts_glb

    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(err)
        raise


CSS_TEXT = """
<style>
#in_glb  { height: 520px !important; }
#seg_glb { height: 520px !important; }
#part_glb{ height: 520px !important; }
#img     { height: 520px !important; }
</style>
"""

with gr.Blocks() as demo:
    gr.HTML(CSS_TEXT)
    gr.Markdown(
        """
# SegviGen: Repurposing 3D Generative Model for Part Segmentation
"""
    )

    # ---------------- 2x2 Layout ----------------
    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            in_glb = gr.Model3D(label="Input GLB", elem_id="in_glb")
        with gr.Column(scale=1, min_width=260):
            seg_glb = gr.Model3D(label="Processed GLB", elem_id="seg_glb")

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            with gr.Accordion("2D Segmentation Map (Optional)", open=False):
                map_mode = gr.Radio(
                    choices=["Upload", "Generate (Use FLUX.2 to generate segmentation map)"],
                    value="Upload",
                    label="2D Map Mode",
                )
                in_img = gr.Image(
                    label="2D Segmentation Map",
                    type="filepath",
                    elem_id="img",
                    interactive=True,
                )

            with gr.Accordion("Model & Post-Processing Settings", open=True):
                ckpt_choice = gr.Dropdown(
                    choices=["Default (Auto)", "full_seg", "full_seg_w_2d_map"],
                    value="Default (Auto)",
                    label="SegviGen Checkpoint",
                )
                bake_mode = gr.Checkbox(
                    label="Bake to Original Mesh", 
                    value=False,
                )
                generate_uv = gr.Checkbox(
                    label="Generate UVs (xatlas)", 
                    value=False,
                )

            seg_btn = gr.Button("Process", variant="primary")

            # ✅ Examples directly under the Process button
            if FULL_SEG_EXAMPLES:
                gr.Examples(
                    examples=FULL_SEG_EXAMPLES,
                    inputs=[in_glb, in_img],
                    label="Examples",
                    examples_per_page=3,
                    cache_examples=False,
                )
            else:
                gr.Markdown(f"**No examples found** in: `{EXAMPLES_DIR}` (expected: `*.glb` + same-name `*.png`).")

            with gr.Accordion("Advanced segmentation options", open=False):
                def _g(name, default):
                    return getattr(splitter, name, default)

                color_quant_step = gr.Slider(
                    1, 64, value=_g("COLOR_QUANT_STEP", 16), step=1, label="COLOR_QUANT_STEP"
                )
                gr.Markdown(
                    "*COLOR_QUANT_STEP controls the RGB quantization step, where a larger value merges similar colors more aggressively and a smaller value preserves finer color differences.*"
                )

                palette_sample_pixels = gr.Number(
                    value=_g("PALETTE_SAMPLE_PIXELS", 2_000_000), precision=0, label="PALETTE_SAMPLE_PIXELS"
                )
                gr.Markdown(
                    "*PALETTE_SAMPLE_PIXELS sets the maximum number of sampled pixels used to estimate the palette, where more samples improve stability but increase runtime.*"
                )

                palette_min_pixels = gr.Number(
                    value=_g("PALETTE_MIN_PIXELS", 500), precision=0, label="PALETTE_MIN_PIXELS"
                )
                gr.Markdown(
                    "*PALETTE_MIN_PIXELS specifies the minimum pixel count required to keep a color in the palette, where a higher threshold suppresses noise but may discard small parts.*"
                )

                palette_max_colors = gr.Number(
                    value=_g("PALETTE_MAX_COLORS", 256), precision=0, label="PALETTE_MAX_COLORS"
                )
                gr.Markdown(
                    "*PALETTE_MAX_COLORS limits the maximum number of colors retained in the palette, where a larger limit yields finer partitions and a smaller limit enforces stronger merging.*"
                )

                palette_merge_dist = gr.Number(
                    value=_g("PALETTE_MERGE_DIST", 32), precision=0, label="PALETTE_MERGE_DIST"
                )
                gr.Markdown(
                    "*PALETTE_MERGE_DIST defines the distance threshold for merging nearby palette colors in RGB space, where a larger threshold merges near duplicates more often and a smaller threshold keeps colors distinct.*"
                )

                samples_per_face = gr.Dropdown(
                    choices=[1, 4], value=_g("SAMPLES_PER_FACE", 4), label="SAMPLES_PER_FACE"
                )
                gr.Markdown(
                    "*SAMPLES_PER_FACE sets the number of UV samples per triangle used for label voting, where more samples improve robustness near boundaries but increase computation.*"
                )

                flip_v = gr.Checkbox(value=_g("FLIP_V", True), label="FLIP_V")
                gr.Markdown(
                    "*FLIP_V toggles whether the V coordinate is flipped to match common glTF texture conventions, and you should disable it only if the texture appears vertically inverted.*"
                )

                uv_wrap_repeat = gr.Checkbox(value=_g("UV_WRAP_REPEAT", True), label="UV_WRAP_REPEAT")
                gr.Markdown(
                    "*UV_WRAP_REPEAT selects how out of range UVs are handled by either repeating via modulo or clamping to the unit interval, and repeating is typically preferred for tiled textures.*"
                )

                transition_conf_thresh = gr.Slider(
                    0.25, 1.0, value=float(_g("TRANSITION_CONF_THRESH", 1.0)), step=0.25, label="TRANSITION_CONF_THRESH"
                )
                gr.Markdown(
                    "*TRANSITION_CONF_THRESH sets the confidence threshold for transition handling, where a higher value makes refinement more conservative and a lower value enables more aggressive smoothing.*"
                )

                transition_prop_iters = gr.Number(
                    value=_g("TRANSITION_PROP_ITERS", 6), precision=0, label="TRANSITION_PROP_ITERS"
                )
                gr.Markdown(
                    "*TRANSITION_PROP_ITERS specifies the number of propagation iterations used in transition refinement, where more iterations strengthen diffusion effects but increase runtime.*"
                )

                transition_neighbor_min = gr.Number(
                    value=_g("TRANSITION_NEIGHBOR_MIN", 1), precision=0, label="TRANSITION_NEIGHBOR_MIN"
                )
                gr.Markdown(
                    "*TRANSITION_NEIGHBOR_MIN requires a minimum number of supporting neighbors to propagate a label, where a higher requirement is more conservative and a lower requirement is more permissive.*"
                )

                small_component_action = gr.Dropdown(
                    choices=["reassign", "drop"], value=_g("SMALL_COMPONENT_ACTION", "reassign"), label="SMALL_COMPONENT_ACTION"
                )
                gr.Markdown(
                    "*SMALL_COMPONENT_ACTION determines how small connected components are handled by either reassigning them to neighboring labels or dropping them entirely.*"
                )

                small_component_min_faces = gr.Number(
                    value=_g("SMALL_COMPONENT_MIN_FACES", 50), precision=0, label="SMALL_COMPONENT_MIN_FACES"
                )
                gr.Markdown(
                    "*SMALL_COMPONENT_MIN_FACES defines the face count threshold used to classify a component as small, where a higher threshold merges or removes more fragments and a lower threshold preserves more small parts.*"
                )

                postprocess_iters = gr.Number(
                    value=_g("POSTPROCESS_ITERS", 3), precision=0, label="POSTPROCESS_ITERS"
                )
                gr.Markdown(
                    "*POSTPROCESS_ITERS sets the number of post processing iterations, where more iterations produce stronger cleanup at the cost of additional computation.*"
                )

                min_faces_per_part = gr.Number(
                    value=_g("MIN_FACES_PER_PART", 1), precision=0, label="MIN_FACES_PER_PART"
                )
                gr.Markdown(
                    "*MIN_FACES_PER_PART enforces a minimum number of faces per exported part, where a larger value filters tiny outputs and a smaller value retains fine components.*"
                )

                bake_transforms = gr.Checkbox(value=_g("BAKE_TRANSFORMS", True), label="BAKE_TRANSFORMS")
                gr.Markdown(
                    "*BAKE_TRANSFORMS controls whether scene graph transforms are baked into geometry before splitting, where enabling it improves consistency in world space and disabling it preserves node transforms.*"
                )

        with gr.Column(scale=1, min_width=260):
            refine_btn = gr.Button("Segment", variant="secondary")
            part_glb = gr.Model3D(label="Segmented GLB", elem_id="part_glb")

    seg_glb_state = gr.State(None)

    map_mode.change(
        fn=_update_img_box,
        inputs=[map_mode],
        outputs=[in_img],
    )

    seg_btn.click(
        fn=run_seg,
        inputs=[in_glb, map_mode, in_img, ckpt_choice, bake_mode, generate_uv],
        outputs=[seg_glb, seg_glb_state, in_img],
    )

    refine_btn.click(
        fn=run_refine_segmentation,
        inputs=[
            seg_glb_state,
            color_quant_step,
            palette_sample_pixels,
            palette_min_pixels,
            palette_max_colors,
            palette_merge_dist,
            samples_per_face,
            flip_v,
            uv_wrap_repeat,
            transition_conf_thresh,
            transition_prop_iters,
            transition_neighbor_min,
            small_component_action,
            small_component_min_faces,
            postprocess_iters,
            min_faces_per_part,
            bake_transforms
        ],
        outputs=[part_glb],
    )

if __name__ == "__main__":
    inf.PIPE.load_all_models()

    # preload
    inf.PIPE.load_ckpt_if_needed(CKPT_W_2D_MAP)

    demo.launch()