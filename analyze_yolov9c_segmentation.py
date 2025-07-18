#!/usr/bin/env python3
"""
Analyze YOLOv9-C model for best feature maps for instance segmentation.
"""

import onnx
import onnx_graphsurgeon as gs
import numpy as np
from typing import List, Tuple, Dict
import math
import json

def calculate_segmentation_score(height: int, width: int, channels: int, input_size: int = 640) -> float:
    """
    Calculate segmentation score based on spatial resolution and channel depth.
    
    Score = (spatial_resolution_factor * 0.7) + (channel_depth_factor * 0.3)
    """
    # Spatial resolution factor (relative to input size)
    spatial_resolution_factor = math.sqrt(height * width) / math.sqrt(input_size * input_size)
    
    # Channel depth factor (normalized by 512 channels)
    channel_depth_factor = min(channels / 512, 1.0)
    
    # Combined score
    score = (spatial_resolution_factor * 0.7) + (channel_depth_factor * 0.3)
    
    return score

def get_downsampling_factor(size: int, input_size: int = 640) -> int:
    """Get downsampling factor from output size."""
    return input_size // size

def analyze_model_for_segmentation(model_path: str) -> Dict:
    """Analyze ONNX model and find best layers for segmentation."""
    
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)
    
    # Get total layer count
    total_nodes = len(graph.nodes)
    print(f"Total nodes in model: {total_nodes}")
    
    # Analyze different types of layers
    concat_layers = []
    swish_layers = []  # Mul outputs after Sigmoid
    cv3_layers = []
    cv4_layers = []
    cv5_layers = []
    all_analyzed_layers = []
    
    # First pass - identify Swish activations (Sigmoid -> Mul pattern)
    sigmoid_to_mul = {}  # Map Sigmoid outputs to Mul nodes
    
    for node in graph.nodes:
        if node.op == "Sigmoid":
            sigmoid_output = node.outputs[0]
            # Find Mul nodes that use this Sigmoid output
            for other_node in graph.nodes:
                if other_node.op == "Mul" and sigmoid_output in other_node.inputs:
                    sigmoid_to_mul[sigmoid_output.name] = other_node
    
    # Second pass - analyze all nodes
    for node in graph.nodes:
        output_shape = None
        if node.outputs and hasattr(node.outputs[0], 'shape') and node.outputs[0].shape:
            try:
                shape = node.outputs[0].shape
                if isinstance(shape, list) and len(shape) == 4:
                    _, channels, height, width = shape
                    if isinstance(height, int) and isinstance(width, int) and height == width:
                        output_shape = (height, width, channels)
            except:
                continue
        
        if not output_shape:
            continue
        
        height, width, channels = output_shape
        
        # Skip very small feature maps
        if height < 10 or width < 10:
            continue
            
        # Analyze Concat layers
        if node.op == "Concat":
            score = calculate_segmentation_score(height, width, channels)
            downsample = get_downsampling_factor(height)
            concat_layers.append({
                'name': node.outputs[0].name,
                'shape': f"{channels}x{height}x{width}",
                'resolution': f"{height}x{width}",
                'channels': channels,
                'downsample_factor': f"{downsample}x",
                'score': score,
                'height': height,
                'width': width
            })
        
        # Analyze Swish layers (Mul after Sigmoid)
        if node.op == "Mul" and node in sigmoid_to_mul.values():
            score = calculate_segmentation_score(height, width, channels)
            downsample = get_downsampling_factor(height)
            swish_layers.append({
                'name': node.outputs[0].name,
                'shape': f"{channels}x{height}x{width}",
                'resolution': f"{height}x{width}",
                'channels': channels,
                'downsample_factor': f"{downsample}x",
                'score': score,
                'height': height,
                'width': width
            })
        
        # Check for cv3, cv4, cv5 modules
        output_name = node.outputs[0].name
        if 'cv3' in output_name:
            score = calculate_segmentation_score(height, width, channels)
            downsample = get_downsampling_factor(height)
            cv3_layers.append({
                'name': output_name,
                'shape': f"{channels}x{height}x{width}",
                'resolution': f"{height}x{width}",
                'channels': channels,
                'downsample_factor': f"{downsample}x",
                'score': score,
                'height': height,
                'width': width
            })
        elif 'cv4' in output_name:
            score = calculate_segmentation_score(height, width, channels)
            downsample = get_downsampling_factor(height)
            cv4_layers.append({
                'name': output_name,
                'shape': f"{channels}x{height}x{width}",
                'resolution': f"{height}x{width}",
                'channels': channels,
                'downsample_factor': f"{downsample}x",
                'score': score,
                'height': height,
                'width': width
            })
        elif 'cv5' in output_name:
            score = calculate_segmentation_score(height, width, channels)
            downsample = get_downsampling_factor(height)
            cv5_layers.append({
                'name': output_name,
                'shape': f"{channels}x{height}x{width}",
                'resolution': f"{height}x{width}",
                'channels': channels,
                'downsample_factor': f"{downsample}x",
                'score': score,
                'height': height,
                'width': width
            })
        
        # Add to all analyzed layers
        score = calculate_segmentation_score(height, width, channels)
        downsample = get_downsampling_factor(height)
        all_analyzed_layers.append({
            'name': output_name,
            'type': node.op,
            'shape': f"{channels}x{height}x{width}",
            'resolution': f"{height}x{width}",
            'channels': channels,
            'downsample_factor': f"{downsample}x",
            'score': score,
            'height': height,
            'width': width
        })
    
    # Sort all layers by score
    all_analyzed_layers.sort(key=lambda x: x['score'], reverse=True)
    concat_layers.sort(key=lambda x: x['score'], reverse=True)
    swish_layers.sort(key=lambda x: x['score'], reverse=True)
    cv3_layers.sort(key=lambda x: x['score'], reverse=True)
    cv4_layers.sort(key=lambda x: x['score'], reverse=True)
    cv5_layers.sort(key=lambda x: x['score'], reverse=True)
    
    # Get top 10 layers overall
    top_10_layers = all_analyzed_layers[:10]
    
    # Find best layers at each scale from all layers
    scale_4x = [l for l in all_analyzed_layers if l['downsample_factor'] == '4x'][:5]
    scale_8x = [l for l in all_analyzed_layers if l['downsample_factor'] == '8x'][:5]
    scale_16x = [l for l in all_analyzed_layers if l['downsample_factor'] == '16x'][:5]
    scale_32x = [l for l in all_analyzed_layers if l['downsample_factor'] == '32x'][:5]
    
    # Create analysis report
    analysis = {
        'model_info': {
            'model_path': model_path,
            'total_nodes': total_nodes,
            'input_size': '640x640'
        },
        'layer_counts': {
            'concat_layers': len(concat_layers),
            'swish_layers': len(swish_layers),
            'cv3_layers': len(cv3_layers),
            'cv4_layers': len(cv4_layers),
            'cv5_layers': len(cv5_layers)
        },
        'top_10_layers_for_segmentation': top_10_layers,
        'multi_scale_recommendations': {
            '4x_downsampling': scale_4x,
            '8x_downsampling': scale_8x,
            '16x_downsampling': scale_16x,
            '32x_downsampling': scale_32x
        },
        'layer_details': {
            'concat_layers': concat_layers[:10],
            'swish_layers': swish_layers[:10],
            'cv3_layers': cv3_layers[:5],
            'cv4_layers': cv4_layers[:5],
            'cv5_layers': cv5_layers[:3]
        },
        'best_segmentation_layers': {
            'high_resolution': scale_4x[:1] if scale_4x else [],
            'medium_resolution': scale_8x[:2] if scale_8x else [],
            'low_resolution': scale_16x[:1] if scale_16x else []
        }
    }
    
    return analysis

def print_analysis_report(analysis: Dict):
    """Print formatted analysis report."""
    
    print("\n" + "="*80)
    print("YOLOv9-C INSTANCE SEGMENTATION ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n📊 Model Information:")
    print(f"   - Model: {analysis['model_info']['model_path']}")
    print(f"   - Total nodes: {analysis['model_info']['total_nodes']}")
    print(f"   - Input size: {analysis['model_info']['input_size']}")
    
    print(f"\n📈 Layer Statistics:")
    for layer_type, count in analysis['layer_counts'].items():
        print(f"   - {layer_type}: {count}")
    
    print(f"\n🏆 TOP 10 LAYERS FOR INSTANCE SEGMENTATION:")
    print(f"{'Rank':<6} {'Layer Name':<40} {'Shape':<20} {'Downsample':<12} {'Score':<8}")
    print("-" * 90)
    
    for i, layer in enumerate(analysis['top_10_layers_for_segmentation'], 1):
        print(f"{i:<6} {layer['name']:<40} {layer['shape']:<20} {layer['downsample_factor']:<12} {layer['score']:.4f}")
    
    print(f"\n🎯 MULTI-SCALE RECOMMENDATIONS:")
    for scale, layers in analysis['multi_scale_recommendations'].items():
        if layers:
            print(f"\n   {scale}:")
            for layer in layers:
                print(f"      - {layer['name']} ({layer['shape']}, score: {layer['score']:.4f})")
    
    print(f"\n📋 DETAILED LAYER ANALYSIS:")
    
    # Print detailed info for each layer type
    for layer_type, layers in analysis['layer_details'].items():
        if layers:
            print(f"\n   {layer_type.upper()}:")
            for layer in layers[:3]:  # Show top 3 of each type
                print(f"      - {layer['name']}")
                print(f"        Shape: {layer['shape']}, Score: {layer['score']:.4f}")

def main():
    model_path = "yolov9_c_wholebody25_Nx3x640x640.onnx"
    
    try:
        analysis = analyze_model_for_segmentation(model_path)
        print_analysis_report(analysis)
        
        # Save to JSON file
        output_file = "yolov9c_segmentation_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✅ Analysis saved to: {output_file}")
        
        # Create a markdown report
        md_report = generate_markdown_report(analysis)
        md_file = "yolov9c_segmentation_analysis_report.md"
        with open(md_file, 'w') as f:
            f.write(md_report)
        print(f"📄 Markdown report saved to: {md_file}")
        
    except Exception as e:
        print(f"❌ Error analyzing model: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_markdown_report(analysis: Dict) -> str:
    """Generate a markdown report from analysis."""
    
    md = []
    md.append("# YOLOv9-C Instance Segmentation Analysis Report\n")
    
    md.append("## Model Information")
    md.append(f"- **Model**: `{analysis['model_info']['model_path']}`")
    md.append(f"- **Total nodes**: {analysis['model_info']['total_nodes']}")
    md.append(f"- **Input size**: {analysis['model_info']['input_size']}")
    md.append(f"- **Model size**: ~101MB (Compact variant)\n")
    
    md.append("## Layer Statistics")
    md.append("| Layer Type | Count |")
    md.append("|------------|-------|")
    for layer_type, count in analysis['layer_counts'].items():
        md.append(f"| {layer_type} | {count} |")
    
    md.append("\n## Top 10 Layers for Instance Segmentation")
    md.append("| Rank | Layer Name | Shape | Downsample | Score |")
    md.append("|------|------------|-------|------------|-------|")
    
    for i, layer in enumerate(analysis['top_10_layers_for_segmentation'], 1):
        md.append(f"| {i} | `{layer['name']}` | {layer['shape']} | {layer['downsample_factor']} | {layer['score']:.4f} |")
    
    md.append("\n## Multi-Scale Recommendations")
    md.append("\nFor optimal instance segmentation, use features from multiple scales:")
    
    for scale, layers in analysis['multi_scale_recommendations'].items():
        if layers:
            md.append(f"\n### {scale}")
            for layer in layers:
                md.append(f"- **{layer['name']}** - Shape: {layer['shape']}, Score: {layer['score']:.4f}")
    
    md.append("\n## Scoring Methodology")
    md.append("```")
    md.append("Score = (spatial_resolution_factor * 0.7) + (channel_depth_factor * 0.3)")
    md.append("where:")
    md.append("  spatial_resolution_factor = sqrt(height * width) / sqrt(640 * 640)")
    md.append("  channel_depth_factor = min(channels / 512, 1.0)")
    md.append("```")
    
    md.append("\n## Best Layers for Instance Segmentation")
    md.append("\nBased on the analysis, here are the recommended layers for multi-scale instance segmentation:")
    
    best = analysis.get('best_segmentation_layers', {})
    if best.get('high_resolution'):
        md.append("\n### High Resolution (4x downsampling)")
        for layer in best['high_resolution']:
            md.append(f"- **{layer['name']}** - {layer['shape']}")
    
    if best.get('medium_resolution'):
        md.append("\n### Medium Resolution (8x downsampling)")
        for layer in best['medium_resolution']:
            md.append(f"- **{layer['name']}** - {layer['shape']}")
    
    if best.get('low_resolution'):
        md.append("\n### Low Resolution (16x downsampling)")
        for layer in best['low_resolution']:
            md.append(f"- **{layer['name']}** - {layer['shape']}")
    
    md.append("\n## Recommended Usage")
    md.append("```python")
    md.append("# Extract features for segmentation")
    
    # Build command with best layers
    best_layers = []
    scale_8x = analysis['multi_scale_recommendations'].get('8x_downsampling', [])
    scale_16x = analysis['multi_scale_recommendations'].get('16x_downsampling', [])
    
    if scale_8x:
        best_layers.append(scale_8x[0]['name'])
    if scale_16x:
        best_layers.append(scale_16x[0]['name'])
    
    if best_layers:
        layers_str = ' '.join([f'"{l}"' for l in best_layers])
        md.append(f"python generate_heatmaps_unified.py --model yolov9_c_wholebody25_Nx3x640x640.onnx \\")
        md.append(f"    --layers {layers_str} \\")
        md.append(f"    --alpha 0.4")
    else:
        md.append("python generate_heatmaps_unified.py --model yolov9_c_wholebody25_Nx3x640x640.onnx \\")
        md.append("    --layers \"/model.22/Concat_output_0\" \"/model.15/Concat_output_0\" \\")
        md.append("    --alpha 0.4")
    md.append("```")
    
    return "\n".join(md)

if __name__ == "__main__":
    main()