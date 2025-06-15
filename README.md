# YOLOv9 Heatmap Visualization

A tool for visualizing intermediate layer (Conv layer) outputs from YOLOv9 models and generating heatmaps.

## Overview

This tool extracts outputs from each Conv layer in a YOLOv9 ONNX model and visualizes them as heatmaps. It helps understand which regions the human body detection model is focusing on. The sample ONNX models committed are incomplete and unfinished models.

## Required Libraries

```bash
pip install onnx onnx-graphsurgeon onnxruntime opencv-python numpy
```

## Usage

### Basic Usage

```bash
# Generate heatmaps for all Conv layers (default 40% transparency)
python generate_heatmaps_unified.py

# Generate overlay with 60% transparency
python generate_heatmaps_unified.py --alpha 0.6

# Generate heatmaps with inverted colors
python generate_heatmaps_unified.py --invert

# Generate heatmaps only (no overlay)
python generate_heatmaps_unified.py --no-overlay

# Use a custom ONNX model file
python generate_heatmaps_unified.py --model path/to/your/model.onnx
```

### Specifying Specific Layers

```bash
# Extract all Conv layers from model.7
python generate_heatmaps_unified.py --layers model.7

# Extract all layers containing cv3.0
python generate_heatmaps_unified.py --layers cv3.0

# Specify multiple patterns
python generate_heatmaps_unified.py --layers model.7 model.9 cv3.0

# Specify a complete layer name
python generate_heatmaps_unified.py --layers "/model.7/cv3/cv3.0/cv3/conv/Conv_output_0"
```

## Parameters

- `--model`: Path to ONNX model file (default: yolov9_e_wholebody34_0100_1x3x480x640.onnx)
- `--invert`: Invert heatmap colors (blue for body regions, red for background)
- `--layers`: Specify layer names or patterns to extract (multiple values allowed)
- `--alpha`: Specify overlay transparency (0.0-1.0, default: 0.4)
- `--no-overlay`: Skip overlay image generation

## Output Directories

- `heatmaps/`: Normal heatmap images
- `heatmaps_inverted/`: Inverted heatmap images
- `overlays_XX/`: Overlay images with XX% transparency
- `overlays_XX_inverted/`: Inverted overlay images with XX% transparency

## File Structure

- `generate_heatmaps_unified.py`: Main script
- `yolov9_e_wholebody34_0100_1x3x480x640.onnx`: YOLOv9 model (34 classes)
- `000000001000.jpg`: Sample input image

## Examples

### Visualizing Layers Suitable for Human Body Detection

The cv3.0.cv3 layers in intermediate layers (model.7, model.9) moderately abstract human body shapes and are suitable for visualizing attention regions:

```bash
python generate_heatmaps_unified.py --layers "cv3.0.cv3"
```

### Comparison Grid Images

`comparison_grid.png` and `comparison_grid_middle.png` show comparisons of layers at different depths.

## Notes

- The ONNX model is temporarily modified but deleted after execution
- Processing a large number of Conv layers may take time
- Generated images will have the same size as the original input image

## License

- **Code**: MIT License
- **Model**: GPLv3 License