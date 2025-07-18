# YOLOv9-S Segmentation Layer Analysis

**Model**: `yolov9_s_wholebody25_Nx3x640x640.onnx`
**Total Nodes**: 674
**Analyzed Layers**: 209

## Summary

- Concat layers: 30
- Swish activation layers: 179
- CV3 modules: 84
- CV4 modules: 8
- CV5 modules: 1

## Top 10 Layers for Segmentation

| Rank | Layer Name | Type | Resolution | Channels | Score |
|------|------------|------|------------|----------|-------|
| 1 | `/model.0/act/Mul_output_0` | Swish (Mul after Sigmoid) | 320x320 (1/2x) | 32 | 0.3687 |
| 2 | `/model.8/Concat_output_0` | Concat | 20x20 (1/32x) | 512 | 0.3219 |
| 3 | `/model.9/Concat_output_0` | Concat | 20x20 (1/32x) | 512 | 0.3219 |
| 4 | `/model.21/Concat_output_0` | Concat | 20x20 (1/32x) | 512 | 0.3219 |
| 5 | `/model.11/Concat_output_0` | Concat | 40x40 (1/16x) | 448 | 0.3063 |
| 6 | `/model.14/Concat_output_0` | Concat | 80x80 (1/8x) | 320 | 0.2750 |
| 7 | `/model.6/Concat_output_0` | Concat | 40x40 (1/16x) | 384 | 0.2687 |
| 8 | `/model.12/Concat_output_0` | Concat | 40x40 (1/16x) | 384 | 0.2687 |
| 9 | `/model.18/Concat_output_0` | Concat | 40x40 (1/16x) | 384 | 0.2687 |
| 10 | `/model.2/Concat_output_0` | Concat | 160x160 (1/4x) | 128 | 0.2500 |

## Concat Layers

| Layer Name | Resolution | Channels | Score |
|------------|------------|----------|-------|
| `/model.2/Concat_output_0` | 160x160 (1/4x) | 128 | 0.2500 |
| `/model.4/cv2/cv2.0/Concat_output_0` | 80x80 (1/8x) | 64 | 0.1250 |
| `/model.4/cv3/cv3.0/Concat_output_0` | 80x80 (1/8x) | 64 | 0.1250 |
| `/model.4/Concat_output_0` | 80x80 (1/8x) | 256 | 0.2375 |
| `/model.6/cv2/cv2.0/Concat_output_0` | 40x40 (1/16x) | 96 | 0.1000 |
| `/model.6/cv3/cv3.0/Concat_output_0` | 40x40 (1/16x) | 96 | 0.1000 |
| `/model.6/Concat_output_0` | 40x40 (1/16x) | 384 | 0.2687 |
| `/model.8/cv2/cv2.0/Concat_output_0` | 20x20 (1/32x) | 128 | 0.0969 |
| `/model.8/cv3/cv3.0/Concat_output_0` | 20x20 (1/32x) | 128 | 0.0969 |
| `/model.8/Concat_output_0` | 20x20 (1/32x) | 512 | 0.3219 |
| `/model.9/Concat_output_0` | 20x20 (1/32x) | 512 | 0.3219 |
| `/model.11/Concat_output_0` | 40x40 (1/16x) | 448 | 0.3063 |
| `/model.12/cv2/cv2.0/Concat_output_0` | 40x40 (1/16x) | 96 | 0.1000 |
| `/model.12/cv3/cv3.0/Concat_output_0` | 40x40 (1/16x) | 96 | 0.1000 |
| `/model.12/Concat_output_0` | 40x40 (1/16x) | 384 | 0.2687 |
| `/model.14/Concat_output_0` | 80x80 (1/8x) | 320 | 0.2750 |
| `/model.15/cv2/cv2.0/Concat_output_0` | 80x80 (1/8x) | 64 | 0.1250 |
| `/model.15/cv3/cv3.0/Concat_output_0` | 80x80 (1/8x) | 64 | 0.1250 |
| `/model.15/Concat_output_0` | 80x80 (1/8x) | 256 | 0.2375 |
| `/model.17/Concat_output_0` | 40x40 (1/16x) | 288 | 0.2125 |
| `/model.18/cv2/cv2.0/Concat_output_0` | 40x40 (1/16x) | 96 | 0.1000 |
| `/model.18/cv3/cv3.0/Concat_output_0` | 40x40 (1/16x) | 96 | 0.1000 |
| `/model.18/Concat_output_0` | 40x40 (1/16x) | 384 | 0.2687 |
| `/model.20/Concat_output_0` | 20x20 (1/32x) | 384 | 0.2469 |
| `/model.21/cv2/cv2.0/Concat_output_0` | 20x20 (1/32x) | 128 | 0.0969 |
| `/model.21/cv3/cv3.0/Concat_output_0` | 20x20 (1/32x) | 128 | 0.0969 |
| `/model.21/Concat_output_0` | 20x20 (1/32x) | 512 | 0.3219 |
| `/model.22/Concat_output_0` | 80x80 (1/8x) | 89 | 0.1396 |
| `/model.22/Concat_1_output_0` | 40x40 (1/16x) | 89 | 0.0959 |
| `/model.22/Concat_2_output_0` | 20x20 (1/32x) | 89 | 0.0740 |

## Recommendations

Based on the analysis, the best layers for segmentation are:

1. **High Resolution (160x160, 1/4x)**: Good for fine details
   - `/model.2/Concat_output_0`: Score 0.2500

2. **Medium Resolution (80x80, 1/8x)**: Balance of detail and context
   - `/model.14/Concat_output_0`: Score 0.2750

3. **Low Resolution (40x40, 1/16x)**: Good for semantic context
   - `/model.11/Concat_output_0`: Score 0.3063
   - `/model.6/Concat_output_0`: Score 0.2687
   - `/model.12/Concat_output_0`: Score 0.2687
   - `/model.18/Concat_output_0`: Score 0.2687
