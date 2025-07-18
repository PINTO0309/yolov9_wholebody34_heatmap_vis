#!/bin/bash

# Test script for YOLOv9-C segmentation feature extraction

echo "Testing YOLOv9-C best layers for instance segmentation..."

# Test 1: Extract multi-scale features for segmentation
echo -e "\n1. Extracting multi-scale features (4x, 8x, 16x downsampling):"
python generate_heatmaps_unified.py \
    --model yolov9_c_wholebody25_Nx3x640x640.onnx \
    --layers "/model.2/Concat_output_0" "/model.4/Concat_output_0" "/model.5/Concat_output_0" \
    --alpha 0.4

# Test 2: Extract Concat layers (best for feature fusion)
echo -e "\n2. Extracting top Concat layers:"
python generate_heatmaps_unified.py \
    --model yolov9_c_wholebody25_Nx3x640x640.onnx \
    --layers "/model.4/Concat_output_0" "/model.15/Concat_output_0" "/model.6/Concat_output_0" \
    --alpha 0.4

# Test 3: Extract activated features (Swish outputs)
echo -e "\n3. Extracting activated features (Swish outputs):"
python generate_heatmaps_unified.py \
    --model yolov9_c_wholebody25_Nx3x640x640.onnx \
    --layers "/model.4/cv4/act/Mul_output_0" "/model.6/cv1/act/Mul_output_0" \
    --alpha 0.4 \
    --activated

echo -e "\nAnalysis complete! Check the heatmaps and overlays directories for results."
echo "Best layers for segmentation:"
echo "  - High res (4x): /model.2/Concat_output_0 (256x160x160)"
echo "  - Medium res (8x): /model.4/Concat_output_0 (512x80x80)"
echo "  - Low res (16x): /model.5/Concat_output_0 (512x40x40)"