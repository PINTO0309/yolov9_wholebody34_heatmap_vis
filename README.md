# YOLOv9 Heatmap Visualization

A tool for visualizing intermediate layer (Conv layer) outputs from YOLOv9 models and generating heatmaps.

## Overview

This tool extracts outputs from each Conv layer in a YOLOv9 ONNX model and visualizes them as heatmaps. It helps understand which regions the human body detection model is focusing on. The sample ONNX models committed are incomplete and unfinished models.

![overlay__model 35_cv3_cv3 0_cv1_conv_Conv_output_0](https://github.com/user-attachments/assets/9e28b8d8-3a48-4d89-875f-38ae9967b6a4)

## Required Libraries

```bash
pip install onnx onnx-graphsurgeon onnxruntime opencv-python numpy
```

## Download Test Model

Download the test ONNX model from the following URL:
```bash
wget https://github.com/PINTO0309/yolov9_wholebody34_heatmap_vis/releases/download/onnx/yolov9_e_wholebody34_0100_1x3x480x640.onnx
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

For more precise control, you can specify the exact layer path to visualize a specific Conv output:

```bash
python generate_heatmaps_unified.py --layers "/model.35/cv3/cv3.0/cv1/conv/Conv_output_0"
```

![overlay__model 35_cv3_cv3 0_cv1_conv_Conv_output_0](https://github.com/user-attachments/assets/ced40c7a-f286-4ee7-b396-ec1bd61828a5)

### Comparison Grid Images

When multiple layers are processed (e.g., when using `--layers "/model.35"`), the tool automatically generates comparison grid images:

- `comparison_grid.png`: Shows the first half of all generated heatmaps/overlays in a grid layout
- `comparison_grid_middle.png`: Shows the middle portion (25% to 75%) of all heatmaps/overlays

These grid images are saved in both the heatmaps and overlays directories (unless `--no-overlay` is used).

## Notes

- The ONNX model is temporarily modified but deleted after execution
- Processing a large number of Conv layers may take time
- Generated images will have the same size as the original input image

## License

- **Code**: MIT License
- **Model**: GPLv3 License
