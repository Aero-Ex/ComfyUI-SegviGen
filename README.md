# ComfyUI-SegviGen

A ComfyUI implementation of SegviGen, providing precise 3D texturing and interactive segmentation for TRELLIS 2.0.

## Installation

1.  Clone this repository to your `ComfyUI/custom_nodes/` folder.
2.  Install the required base dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the specialized installation script to handle custom CUDA wheels and internal libraries:
    ```bash
    python install.py
    ```

## Key Features

- **Automated Model Downloading**: All model components (Trellis base, SegviGen checkpoints, BiRefNet, DinoV3) are automatically downloaded on first use.
- **Granular Pipeline**: Modular nodes for preprocessing, conditioning, sampling, and post-processing (VXZ, Latent Slats, Voxel, GLB).

## Node Overview

- **SegviGenTrellisLoader**: Loads the base TRELLIS model components.
- **SegviGenCheckpointLoader**: Loads the SegviGen flow checkpoints (Full/Interactive).
- **SegviGenImagePreprocessor**: Automates background removal (BiRefNet) and image preparation.
- **SegviGenImageToCond**: Generates conditioning embeddings (DinoV3).
- **SegviGenSampler**: Performs the core texture sampling with optional point-based guidance.

## Acknowledgements

Original SegviGen implementation by [fenghora](https://huggingface.co/fenghora/SegviGen).
Uses [TRELLIS 2.0](https://huggingface.co/microsoft/TRELLIS.2-4B) by Microsoft.# ComfyUI-SegviGen
