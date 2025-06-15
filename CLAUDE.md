# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a YOLOv9 heatmap visualization tool that extracts and visualizes intermediate Conv layer outputs from a YOLOv9 ONNX model. The tool helps understand which areas the human body detection model focuses on by generating heatmaps and overlay images.

## Commands

### Dependencies Installation
```bash
pip install onnx onnx-graphsurgeon onnxruntime opencv-python numpy tqdm
```

### Common Usage Commands

```bash
# Generate heatmaps for all Conv layers with default 40% transparency
python generate_heatmaps_unified.py

# Use a custom ONNX model file
python generate_heatmaps_unified.py --model path/to/your/model.onnx

# Generate with custom transparency (0.0-1.0)
python generate_heatmaps_unified.py --alpha 0.4

# Generate inverted heatmaps (blue for body, red for background)
python generate_heatmaps_unified.py --invert

# Generate heatmaps only without overlays
python generate_heatmaps_unified.py --no-overlay

# Extract specific layers (e.g., model.7, cv3.0)
python generate_heatmaps_unified.py --layers model.7 cv3.0

# Recommended visualization for human body attention areas
python generate_heatmaps_unified.py --layers "cv3.0.cv3" --alpha 0.4
```

## Architecture

The codebase consists of a single main script (`generate_heatmaps_unified.py`) that:

1. **Model Modification**: Uses ONNX GraphSurgeon to modify the YOLOv9 model to expose Conv layer outputs as additional model outputs
2. **Image Processing**: Preprocesses input images to match YOLOv9 input requirements (640x480, RGB, normalized)
3. **Inference**: Runs the modified model to get both detection outputs and intermediate Conv features
4. **Heatmap Generation**: Converts Conv layer outputs to heatmaps by summing across channels and applying colormaps
5. **Overlay Creation**: Optionally creates overlay images combining original images with heatmaps

Key functions:
- `find_conv_outputs()`: Locates all Conv layers in the ONNX graph
- `modify_onnx_model()`: Modifies ONNX model to expose selected Conv outputs
- `preprocess_image()`: Handles YOLOv9-specific image preprocessing
- `generate_heatmap()`: Creates heatmap visualizations from feature maps
- `create_overlay()`: Blends heatmaps with original images

## Output Structure

Generated files are organized into directories:
- `heatmaps/`: Standard heatmap images
- `heatmaps_inverted/`: Inverted color heatmaps
- `overlays_XX/`: Overlay images with XX% transparency
- `overlays_XX_inverted/`: Inverted overlay images

## Model Details

The repository includes:
- `yolov9_e_wholebody34_0100_1x3x480x640.onnx`: YOLOv9 model with 34 wholebody keypoint classes
- Input size: 640x480 (width x height)
- The model is temporarily modified during execution but cleaned up afterward