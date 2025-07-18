# YOLOv9-C Instance Segmentation Analysis Report

## Model Information
- **Model**: `yolov9_c_wholebody25_Nx3x640x640.onnx`
- **Total nodes**: 545
- **Input size**: 640x640
- **Model size**: ~101MB (Compact variant)

## Layer Statistics
| Layer Type | Count |
|------------|-------|
| concat_layers | 35 |
| swish_layers | 138 |
| cv3_layers | 206 |
| cv4_layers | 25 |
| cv5_layers | 3 |

## Top 10 Layers for Instance Segmentation
| Rank | Layer Name | Shape | Downsample | Score |
|------|------------|-------|------------|-------|
| 1 | `/model.0/conv/Conv_output_0` | 64x320x320 | 2x | 0.3875 |
| 2 | `/model.0/act/Sigmoid_output_0` | 64x320x320 | 2x | 0.3875 |
| 3 | `/model.0/act/Mul_output_0` | 64x320x320 | 2x | 0.3875 |
| 4 | `/model.4/Concat_output_0` | 512x80x80 | 8x | 0.3875 |
| 5 | `/model.4/cv4/conv/Conv_output_0` | 512x80x80 | 8x | 0.3875 |
| 6 | `/model.4/cv4/act/Sigmoid_output_0` | 512x80x80 | 8x | 0.3875 |
| 7 | `/model.4/cv4/act/Mul_output_0` | 512x80x80 | 8x | 0.3875 |
| 8 | `/model.13/Resize_output_0` | 512x80x80 | 8x | 0.3875 |
| 9 | `/model.14/Concat_output_0` | 1024x80x80 | 8x | 0.3875 |
| 10 | `/model.15/Concat_output_0` | 512x80x80 | 8x | 0.3875 |

## Multi-Scale Recommendations

For optimal instance segmentation, use features from multiple scales:

### 4x_downsampling
- **/model.2/Concat_output_0** - Shape: 256x160x160, Score: 0.3250
- **/model.2/cv4/conv/Conv_output_0** - Shape: 256x160x160, Score: 0.3250
- **/model.2/cv4/act/Sigmoid_output_0** - Shape: 256x160x160, Score: 0.3250
- **/model.2/cv4/act/Mul_output_0** - Shape: 256x160x160, Score: 0.3250
- **/model.3/AveragePool_output_0** - Shape: 256x159x159, Score: 0.3239

### 8x_downsampling
- **/model.4/Concat_output_0** - Shape: 512x80x80, Score: 0.3875
- **/model.4/cv4/conv/Conv_output_0** - Shape: 512x80x80, Score: 0.3875
- **/model.4/cv4/act/Sigmoid_output_0** - Shape: 512x80x80, Score: 0.3875
- **/model.4/cv4/act/Mul_output_0** - Shape: 512x80x80, Score: 0.3875
- **/model.13/Resize_output_0** - Shape: 512x80x80, Score: 0.3875

### 16x_downsampling
- **/model.5/Concat_output_0** - Shape: 512x40x40, Score: 0.3438
- **/model.6/cv1/conv/Conv_output_0** - Shape: 512x40x40, Score: 0.3438
- **/model.6/cv1/act/Sigmoid_output_0** - Shape: 512x40x40, Score: 0.3438
- **/model.6/cv1/act/Mul_output_0** - Shape: 512x40x40, Score: 0.3438
- **/model.6/Concat_output_0** - Shape: 1024x40x40, Score: 0.3438

### 32x_downsampling
- **/model.7/Concat_output_0** - Shape: 512x20x20, Score: 0.3219
- **/model.8/cv1/conv/Conv_output_0** - Shape: 512x20x20, Score: 0.3219
- **/model.8/cv1/act/Sigmoid_output_0** - Shape: 512x20x20, Score: 0.3219
- **/model.8/cv1/act/Mul_output_0** - Shape: 512x20x20, Score: 0.3219
- **/model.8/Concat_output_0** - Shape: 1024x20x20, Score: 0.3219

## Scoring Methodology
```
Score = (spatial_resolution_factor * 0.7) + (channel_depth_factor * 0.3)
where:
  spatial_resolution_factor = sqrt(height * width) / sqrt(640 * 640)
  channel_depth_factor = min(channels / 512, 1.0)
```

## Best Layers for Instance Segmentation

Based on the analysis, here are the recommended layers for multi-scale instance segmentation:

### High Resolution (4x downsampling)
- **/model.2/Concat_output_0** - 256x160x160

### Medium Resolution (8x downsampling)
- **/model.4/Concat_output_0** - 512x80x80
- **/model.4/cv4/conv/Conv_output_0** - 512x80x80

### Low Resolution (16x downsampling)
- **/model.5/Concat_output_0** - 512x40x40

## Recommended Usage
```python
# Extract features for segmentation
python generate_heatmaps_unified.py --model yolov9_c_wholebody25_Nx3x640x640.onnx \
    --layers "/model.4/Concat_output_0" "/model.5/Concat_output_0" \
    --alpha 0.4
```