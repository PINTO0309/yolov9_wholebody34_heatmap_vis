#!/usr/bin/env python3
"""
Analyze YOLOv9-T ONNX model to find best feature maps for instance segmentation
"""

import onnx
import numpy as np
from collections import defaultdict
import json

def analyze_yolov9_for_segmentation(model_path):
    """Analyze YOLOv9 model structure and identify best layers for segmentation"""
    
    # Load the ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Count total layers
    total_nodes = len(graph.node)
    print(f"Total nodes in model: {total_nodes}")
    
    # Get value info for all tensors
    value_info = {}
    for vi in graph.value_info:
        shape = []
        for dim in vi.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)  # Dynamic dimension
        value_info[vi.name] = shape
    
    # Also add input/output info
    for input_tensor in graph.input:
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)
        value_info[input_tensor.name] = shape
    
    for output_tensor in graph.output:
        shape = []
        for dim in output_tensor.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)
        value_info[output_tensor.name] = shape
    
    # Collect layer information
    concat_layers = []
    swish_layers = []  # Mul nodes that follow Sigmoid (Swish activation)
    cv_modules = []
    all_conv_layers = []
    
    # Track Sigmoid outputs to identify Swish patterns
    sigmoid_outputs = set()
    
    for i, node in enumerate(graph.node):
        # Track Sigmoid outputs
        if node.op_type == "Sigmoid":
            if node.output[0] in value_info:
                sigmoid_outputs.add(node.output[0])
        
        # Collect Concat layers
        if node.op_type == "Concat":
            if node.output[0] in value_info:
                shape = value_info[node.output[0]]
                if len(shape) == 4:  # NCHW format
                    concat_layers.append({
                        'name': node.output[0],
                        'shape': shape,
                        'index': i
                    })
        
        # Collect Swish layers (Mul after Sigmoid)
        elif node.op_type == "Mul":
            # Check if one of the inputs is from a Sigmoid
            if any(inp in sigmoid_outputs for inp in node.input):
                if node.output[0] in value_info:
                    shape = value_info[node.output[0]]
                    if len(shape) == 4:  # NCHW format
                        swish_layers.append({
                            'name': node.output[0],
                            'shape': shape,
                            'index': i
                        })
        
        # Collect Conv layers
        elif node.op_type == "Conv":
            if node.output[0] in value_info:
                shape = value_info[node.output[0]]
                if len(shape) == 4:  # NCHW format
                    layer_info = {
                        'name': node.output[0],
                        'shape': shape,
                        'index': i
                    }
                    all_conv_layers.append(layer_info)
                    
                    # Check if it's a cv module
                    output_name = node.output[0]
                    if any(cv_type in output_name for cv_type in ['cv3', 'cv4', 'cv5', 'cv2']):
                        cv_modules.append(layer_info)
    
    # Print analysis results
    print(f"\nTotal Concat layers: {len(concat_layers)}")
    print(f"Total Swish activation layers: {len(swish_layers)}")
    print(f"Total cv modules: {len(cv_modules)}")
    print(f"Total Conv layers: {len(all_conv_layers)}")
    
    # Detailed layer listings
    print("\n=== CONCAT LAYERS ===")
    for layer in concat_layers[:10]:  # Show first 10
        print(f"Layer: {layer['name']}, Shape: {layer['shape']}")
    if len(concat_layers) > 10:
        print(f"... and {len(concat_layers) - 10} more")
    
    print("\n=== SWISH ACTIVATION LAYERS ===")
    for layer in swish_layers[:10]:  # Show first 10
        print(f"Layer: {layer['name']}, Shape: {layer['shape']}")
    if len(swish_layers) > 10:
        print(f"... and {len(swish_layers) - 10} more")
    
    print("\n=== CV MODULES ===")
    for layer in cv_modules:
        print(f"Layer: {layer['name']}, Shape: {layer['shape']}")
    
    # Calculate segmentation scores
    def calculate_segmentation_score(shape):
        """Calculate segmentation score based on resolution and channel depth"""
        if len(shape) != 4:
            return 0
        
        _, channels, height, width = shape
        
        # Skip if dimensions are dynamic
        if height == -1 or width == -1:
            return 0
        
        # Spatial resolution factor
        spatial_resolution_factor = np.sqrt(height * width) / np.sqrt(640 * 640)
        
        # Channel depth factor
        channel_depth_factor = min(channels / 512, 1.0)
        
        # Combined score
        score = (spatial_resolution_factor * 0.7) + (channel_depth_factor * 0.3)
        
        return score
    
    # Score all layers
    all_layers = []
    
    # Add all Conv layers
    for layer in all_conv_layers:
        layer['score'] = calculate_segmentation_score(layer['shape'])
        layer['type'] = 'Conv'
        if layer['score'] > 0:
            all_layers.append(layer)
    
    # Add Concat layers
    for layer in concat_layers:
        layer['score'] = calculate_segmentation_score(layer['shape'])
        layer['type'] = 'Concat'
        if layer['score'] > 0:
            all_layers.append(layer)
    
    # Add Swish layers
    for layer in swish_layers:
        layer['score'] = calculate_segmentation_score(layer['shape'])
        layer['type'] = 'Swish'
        if layer['score'] > 0:
            all_layers.append(layer)
    
    # Sort by score
    all_layers.sort(key=lambda x: x['score'], reverse=True)
    
    # Print top layers for segmentation
    print("\n=== TOP 10 LAYERS FOR SEGMENTATION ===")
    print(f"{'Rank':<5} {'Layer Name':<50} {'Type':<10} {'Resolution':<15} {'Channels':<10} {'Score':<10}")
    print("-" * 110)
    
    for i, layer in enumerate(all_layers[:10]):
        shape = layer['shape']
        if len(shape) == 4:
            _, channels, height, width = shape
            resolution = f"{height}x{width}"
            downsampling = 640 // height if height > 0 else 0
            print(f"{i+1:<5} {layer['name']:<50} {layer['type']:<10} {resolution:<15} {channels:<10} {layer['score']:.4f}")
            print(f"      Downsampling: {downsampling}x")
    
    # Group by resolution scale
    resolution_groups = defaultdict(list)
    for layer in all_layers:
        if len(layer['shape']) == 4:
            _, _, height, width = layer['shape']
            if height > 0:
                downsampling = 640 // height
                resolution_groups[downsampling].append(layer)
    
    print("\n=== BEST LAYERS BY RESOLUTION SCALE ===")
    for scale in sorted(resolution_groups.keys()):
        if scale in [4, 8, 16, 32]:  # Focus on common scales
            print(f"\n{scale}x downsampling ({640//scale}x{640//scale}):")
            for layer in resolution_groups[scale][:3]:  # Top 3 per scale
                _, channels, _, _ = layer['shape']
                print(f"  - {layer['name']} ({layer['type']}, {channels} channels, score: {layer['score']:.4f})")
    
    # Save detailed results to JSON
    results = {
        'total_nodes': total_nodes,
        'concat_layers': len(concat_layers),
        'swish_layers': len(swish_layers),
        'cv_modules': len(cv_modules),
        'top_10_layers': [
            {
                'rank': i+1,
                'name': layer['name'],
                'type': layer['type'],
                'shape': layer['shape'],
                'score': layer['score'],
                'downsampling': 640 // layer['shape'][2] if layer['shape'][2] > 0 else 0
            }
            for i, layer in enumerate(all_layers[:10])
        ],
        'layers_by_scale': {
            f"{scale}x": [
                {
                    'name': layer['name'],
                    'type': layer['type'],
                    'shape': layer['shape'],
                    'score': layer['score']
                }
                for layer in resolution_groups[scale][:3]
            ]
            for scale in sorted(resolution_groups.keys()) if scale in [4, 8, 16, 32]
        }
    }
    
    with open('yolov9_t_segmentation_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDetailed results saved to yolov9_t_segmentation_analysis.json")
    
    return results

if __name__ == "__main__":
    model_path = "yolov9_t_wholebody25_Nx3x640x640.onnx"
    analyze_yolov9_for_segmentation(model_path)