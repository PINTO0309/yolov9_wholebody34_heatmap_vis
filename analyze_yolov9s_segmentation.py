#!/usr/bin/env python3
"""
Analyze YOLOv9-S model layers for instance segmentation suitability.
Similar to YOLOv9-T and YOLOv9-E analysis.
"""

import onnx
import numpy as np
from collections import defaultdict
import json

def load_onnx_model(model_path):
    """Load ONNX model."""
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    return model

def get_node_output_shape(model, output_name):
    """Get output shape for a given node output name."""
    for value_info in model.graph.value_info:
        if value_info.name == output_name:
            shape = []
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Dynamic dimension
            return shape
    
    # Check outputs
    for output in model.graph.output:
        if output.name == output_name:
            shape = []
            for dim in output.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)
            return shape
    
    return None

def calculate_segmentation_score(height, width, channels):
    """Calculate segmentation suitability score."""
    # Spatial resolution factor (0-1)
    spatial_resolution = height * width
    max_resolution = 640 * 640
    spatial_factor = np.sqrt(spatial_resolution) / np.sqrt(max_resolution)
    
    # Channel depth factor (0-1)
    channel_factor = min(channels / 512, 1.0)
    
    # Combined score with weights
    score = (spatial_factor * 0.7) + (channel_factor * 0.3)
    
    return score, spatial_factor, channel_factor

def analyze_layers(model):
    """Analyze all layers in the model."""
    layers_info = []
    concat_layers = []
    swish_layers = []
    cv_modules = defaultdict(list)
    
    # Get all nodes
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Concat":
            # Get output shape
            output_shape = get_node_output_shape(model, node.output[0])
            if output_shape and len(output_shape) == 4:
                _, channels, height, width = output_shape
                score, spatial_f, channel_f = calculate_segmentation_score(height, width, channels)
                
                layer_info = {
                    'name': node.output[0],
                    'type': 'Concat',
                    'shape': output_shape,
                    'height': height,
                    'width': width,
                    'channels': channels,
                    'downsampling': 640 // width if width > 0 else 0,
                    'score': score,
                    'spatial_factor': spatial_f,
                    'channel_factor': channel_f
                }
                concat_layers.append(layer_info)
                layers_info.append(layer_info)
        
        elif node.op_type == "Mul":
            # Check if it's a Swish activation (Mul after Sigmoid)
            if i > 0 and model.graph.node[i-1].op_type == "Sigmoid":
                output_shape = get_node_output_shape(model, node.output[0])
                if output_shape and len(output_shape) == 4:
                    _, channels, height, width = output_shape
                    score, spatial_f, channel_f = calculate_segmentation_score(height, width, channels)
                    
                    layer_info = {
                        'name': node.output[0],
                        'type': 'Swish (Mul after Sigmoid)',
                        'shape': output_shape,
                        'height': height,
                        'width': width,
                        'channels': channels,
                        'downsampling': 640 // width if width > 0 else 0,
                        'score': score,
                        'spatial_factor': spatial_f,
                        'channel_factor': channel_f
                    }
                    swish_layers.append(layer_info)
                    layers_info.append(layer_info)
                    
                    # Categorize cv modules
                    if 'cv3' in node.output[0]:
                        cv_modules['cv3'].append(layer_info)
                    elif 'cv4' in node.output[0]:
                        cv_modules['cv4'].append(layer_info)
                    elif 'cv5' in node.output[0]:
                        cv_modules['cv5'].append(layer_info)
    
    return layers_info, concat_layers, swish_layers, cv_modules

def main():
    # Load model
    model_path = "yolov9_s_wholebody25_Nx3x640x640.onnx"
    model = load_onnx_model(model_path)
    
    # Count total nodes
    total_nodes = len(model.graph.node)
    print(f"\nTotal nodes in model: {total_nodes}")
    
    # Analyze layers
    all_layers, concat_layers, swish_layers, cv_modules = analyze_layers(model)
    
    # Sort layers by score
    all_layers.sort(key=lambda x: x['score'], reverse=True)
    
    # Print analysis
    print(f"\n{'='*80}")
    print("YOLOv9-S SEGMENTATION ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nTotal layers analyzed: {len(all_layers)}")
    print(f"Concat layers: {len(concat_layers)}")
    print(f"Swish activation layers: {len(swish_layers)}")
    
    # Print Concat layers
    print(f"\n{'='*60}")
    print("CONCAT LAYERS")
    print(f"{'='*60}")
    for layer in concat_layers:
        print(f"\nLayer: {layer['name']}")
        print(f"  Shape: {layer['shape']}")
        print(f"  Resolution: {layer['height']}x{layer['width']} (1/{layer['downsampling']}x)")
        print(f"  Channels: {layer['channels']}")
        print(f"  Score: {layer['score']:.4f}")
    
    # Print top Swish layers
    print(f"\n{'='*60}")
    print("TOP SWISH ACTIVATION LAYERS")
    print(f"{'='*60}")
    top_swish = sorted(swish_layers, key=lambda x: x['score'], reverse=True)[:15]
    for layer in top_swish:
        print(f"\nLayer: {layer['name']}")
        print(f"  Shape: {layer['shape']}")
        print(f"  Resolution: {layer['height']}x{layer['width']} (1/{layer['downsampling']}x)")
        print(f"  Channels: {layer['channels']}")
        print(f"  Score: {layer['score']:.4f}")
    
    # Print cv module analysis
    print(f"\n{'='*60}")
    print("CV MODULE ANALYSIS")
    print(f"{'='*60}")
    for cv_type, layers in cv_modules.items():
        if layers:
            print(f"\n{cv_type.upper()} modules: {len(layers)} layers")
            # Show best layer from each resolution
            resolutions = {}
            for layer in layers:
                res = f"{layer['height']}x{layer['width']}"
                if res not in resolutions or layer['score'] > resolutions[res]['score']:
                    resolutions[res] = layer
            
            for res, layer in sorted(resolutions.items(), key=lambda x: x[1]['score'], reverse=True):
                print(f"  {layer['name']}: {res} @ {layer['channels']}ch (score: {layer['score']:.4f})")
    
    # Print top 10 layers for segmentation
    print(f"\n{'='*60}")
    print("TOP 10 LAYERS FOR SEGMENTATION")
    print(f"{'='*60}")
    for i, layer in enumerate(all_layers[:10], 1):
        print(f"\n{i}. {layer['name']}")
        print(f"   Type: {layer['type']}")
        print(f"   Resolution: {layer['height']}x{layer['width']} (1/{layer['downsampling']}x)")
        print(f"   Channels: {layer['channels']}")
        print(f"   Score: {layer['score']:.4f}")
        print(f"   Spatial Factor: {layer['spatial_factor']:.4f}")
        print(f"   Channel Factor: {layer['channel_factor']:.4f}")
    
    # Save results
    results = {
        'model': model_path,
        'total_nodes': total_nodes,
        'total_analyzed_layers': len(all_layers),
        'concat_layers': len(concat_layers),
        'swish_layers': len(swish_layers),
        'top_10_layers': all_layers[:10],
        'all_concat_layers': concat_layers,
        'cv_modules': {k: len(v) for k, v in cv_modules.items()}
    }
    
    with open('yolov9s_segmentation_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Analysis saved to yolov9s_segmentation_analysis.json")
    
    # Generate markdown report
    with open('yolov9s_segmentation_report.md', 'w') as f:
        f.write("# YOLOv9-S Segmentation Layer Analysis\n\n")
        f.write(f"**Model**: `{model_path}`\n")
        f.write(f"**Total Nodes**: {total_nodes}\n")
        f.write(f"**Analyzed Layers**: {len(all_layers)}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Concat layers: {len(concat_layers)}\n")
        f.write(f"- Swish activation layers: {len(swish_layers)}\n")
        f.write(f"- CV3 modules: {len(cv_modules['cv3'])}\n")
        f.write(f"- CV4 modules: {len(cv_modules['cv4'])}\n")
        f.write(f"- CV5 modules: {len(cv_modules['cv5'])}\n\n")
        
        f.write("## Top 10 Layers for Segmentation\n\n")
        f.write("| Rank | Layer Name | Type | Resolution | Channels | Score |\n")
        f.write("|------|------------|------|------------|----------|-------|\n")
        for i, layer in enumerate(all_layers[:10], 1):
            f.write(f"| {i} | `{layer['name']}` | {layer['type']} | "
                   f"{layer['height']}x{layer['width']} (1/{layer['downsampling']}x) | "
                   f"{layer['channels']} | {layer['score']:.4f} |\n")
        
        f.write("\n## Concat Layers\n\n")
        f.write("| Layer Name | Resolution | Channels | Score |\n")
        f.write("|------------|------------|----------|-------|\n")
        for layer in concat_layers:
            f.write(f"| `{layer['name']}` | {layer['height']}x{layer['width']} "
                   f"(1/{layer['downsampling']}x) | {layer['channels']} | {layer['score']:.4f} |\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("Based on the analysis, the best layers for segmentation are:\n\n")
        f.write("1. **High Resolution (160x160, 1/4x)**: Good for fine details\n")
        for layer in all_layers[:10]:
            if layer['height'] == 160:
                f.write(f"   - `{layer['name']}`: Score {layer['score']:.4f}\n")
        
        f.write("\n2. **Medium Resolution (80x80, 1/8x)**: Balance of detail and context\n")
        for layer in all_layers[:10]:
            if layer['height'] == 80:
                f.write(f"   - `{layer['name']}`: Score {layer['score']:.4f}\n")
        
        f.write("\n3. **Low Resolution (40x40, 1/16x)**: Good for semantic context\n")
        for layer in all_layers[:10]:
            if layer['height'] == 40:
                f.write(f"   - `{layer['name']}`: Score {layer['score']:.4f}\n")
    
    print(f"Report saved to yolov9s_segmentation_report.md")

if __name__ == "__main__":
    main()