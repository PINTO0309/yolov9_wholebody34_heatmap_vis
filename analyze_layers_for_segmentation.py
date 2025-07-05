#!/usr/bin/env python3
"""
Analyze activated Conv layers to find the most suitable ones for instance segmentation.
"""
import onnx
import onnx_graphsurgeon as gs
import json
import numpy as np
from collections import defaultdict

def analyze_activated_conv_layers(model_path):
    """Analyze all activated Conv layers and their properties."""
    # Load the model
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)
    
    # Input size
    input_shape = model.graph.input[0].type.tensor_type.shape
    input_height = input_shape.dim[2].dim_value  # 480
    input_width = input_shape.dim[3].dim_value   # 640
    
    activated_layers = []
    
    # Find Conv->Sigmoid->Mul patterns (Swish activation)
    for node in graph.nodes:
        if node.op == "Conv":
            conv_output = node.outputs[0]
            
            # Find Sigmoid and Mul nodes
            for next_node in graph.nodes:
                if conv_output in next_node.inputs and next_node.op == "Sigmoid":
                    sigmoid_output = next_node.outputs[0]
                    
                    for mul_node in graph.nodes:
                        if mul_node.op == "Mul" and sigmoid_output in mul_node.inputs:
                            if conv_output in mul_node.inputs:
                                # Get output shape from the Mul node
                                output_name = mul_node.outputs[0].name
                                
                                # Try to get shape information
                                shape = None
                                if hasattr(mul_node.outputs[0], 'shape') and mul_node.outputs[0].shape:
                                    shape = [dim if isinstance(dim, int) else None for dim in mul_node.outputs[0].shape]
                                
                                layer_info = {
                                    'conv_name': node.outputs[0].name,
                                    'output_name': output_name,
                                    'shape': shape,
                                    'spatial_size': None,
                                    'channels': None,
                                    'downsampling_factor': None,
                                    'receptive_field_estimate': None
                                }
                                
                                # Parse layer name to understand position in network
                                if 'model.' in node.outputs[0].name:
                                    parts = node.outputs[0].name.split('/')
                                    for part in parts:
                                        if part.startswith('model.') and part[6:].isdigit():
                                            layer_info['model_stage'] = int(part[6:])
                                            break
                                
                                activated_layers.append(layer_info)
                                break
    
    return activated_layers, input_height, input_width

def run_inference_to_get_shapes(model_path, image_path='000000001000.jpg'):
    """Run inference to get actual output shapes."""
    import onnxruntime as ort
    import cv2
    
    # Prepare input
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 480))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    # Modify model to expose all activated outputs
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)
    
    output_names = []
    for node in graph.nodes:
        if node.op == "Conv":
            conv_output = node.outputs[0]
            for next_node in graph.nodes:
                if conv_output in next_node.inputs and next_node.op == "Sigmoid":
                    sigmoid_output = next_node.outputs[0]
                    for mul_node in graph.nodes:
                        if mul_node.op == "Mul" and sigmoid_output in mul_node.inputs and conv_output in mul_node.inputs:
                            if mul_node.outputs[0] not in graph.outputs:
                                graph.outputs.append(mul_node.outputs[0])
                                output_names.append(mul_node.outputs[0].name)
                            break
    
    # Save modified model temporarily
    model_modified = gs.export_onnx(graph)
    onnx.save(model_modified, "temp_model_analysis.onnx")
    
    # Run inference
    session = ort.InferenceSession("temp_model_analysis.onnx")
    input_name = session.get_inputs()[0].name
    
    # Get shapes from actual inference
    outputs = session.run(None, {input_name: img_batch})
    
    shapes_dict = {}
    for i, output in enumerate(session.get_outputs()):
        if output.name in output_names:
            shape = outputs[i].shape
            shapes_dict[output.name] = shape
    
    # Clean up
    import os
    os.remove("temp_model_analysis.onnx")
    
    return shapes_dict

def analyze_layers_for_segmentation(model_path='yolov9_e_wholebody34_0100_1x3x480x640.onnx'):
    """Analyze which layers are best for segmentation."""
    
    print("Analyzing activated Conv layers...")
    layers, input_h, input_w = analyze_activated_conv_layers(model_path)
    
    print(f"Found {len(layers)} activated Conv layers")
    print(f"Input size: {input_w}x{input_h}")
    
    # Get actual shapes through inference
    print("\nGetting actual output shapes through inference...")
    shapes_dict = run_inference_to_get_shapes(model_path)
    
    # Update layer info with actual shapes
    for layer in layers:
        if layer['output_name'] in shapes_dict:
            shape = shapes_dict[layer['output_name']]
            layer['shape'] = shape
            if len(shape) == 4:  # NCHW format
                layer['channels'] = shape[1]
                layer['spatial_size'] = (shape[2], shape[3])
                layer['downsampling_factor'] = (input_h // shape[2], input_w // shape[3])
    
    # Score layers for segmentation suitability
    print("\nScoring layers for segmentation suitability...")
    
    for layer in layers:
        if layer['spatial_size']:
            h, w = layer['spatial_size']
            
            # Scoring criteria:
            # 1. Spatial resolution (higher is better for fine details)
            spatial_score = (h * w) / (input_h * input_w)
            
            # 2. Semantic level (middle layers balance low-level and high-level features)
            # Prefer layers in the middle of the network
            stage = layer.get('model_stage', 20)
            if stage < 10:  # Early layers
                semantic_score = 0.3
            elif stage < 25:  # Middle layers
                semantic_score = 1.0
            else:  # Late layers
                semantic_score = 0.7
            
            # 3. Channel depth (moderate depth is good)
            channels = layer['channels']
            if channels < 64:
                channel_score = 0.5
            elif channels <= 256:
                channel_score = 1.0
            else:
                channel_score = 0.8
            
            # 4. Downsampling factor (prefer 8x or 16x for good balance)
            ds_h, ds_w = layer['downsampling_factor']
            if ds_h <= 4:  # Too high resolution, less semantic
                ds_score = 0.6
            elif ds_h <= 16:  # Good balance
                ds_score = 1.0
            else:  # Too low resolution
                ds_score = 0.4
            
            # Combined score
            layer['segmentation_score'] = (
                spatial_score * 0.3 +
                semantic_score * 0.3 +
                channel_score * 0.2 +
                ds_score * 0.2
            )
    
    # Sort by segmentation score
    scored_layers = [l for l in layers if 'segmentation_score' in l]
    scored_layers.sort(key=lambda x: x['segmentation_score'], reverse=True)
    
    # Print top candidates
    print("\nTop 10 layers for instance segmentation:")
    print("-" * 100)
    print(f"{'Rank':<5} {'Output Name':<50} {'Shape':<20} {'Downsample':<12} {'Score':<8}")
    print("-" * 100)
    
    for i, layer in enumerate(scored_layers[:10]):
        shape_str = f"{layer['channels']}x{layer['spatial_size'][0]}x{layer['spatial_size'][1]}"
        ds_str = f"{layer['downsampling_factor'][0]}x{layer['downsampling_factor'][1]}"
        print(f"{i+1:<5} {layer['output_name'][:50]:<50} {shape_str:<20} {ds_str:<12} {layer['segmentation_score']:.4f}")
    
    # Save detailed results
    results = {
        'analysis_summary': {
            'total_layers': len(layers),
            'input_size': f"{input_w}x{input_h}",
            'top_recommendations': []
        },
        'all_layers': scored_layers
    }
    
    # Add recommendations
    for layer in scored_layers[:5]:
        results['analysis_summary']['top_recommendations'].append({
            'name': layer['output_name'],
            'shape': f"{layer['channels']}x{layer['spatial_size'][0]}x{layer['spatial_size'][1]}",
            'downsampling': f"{layer['downsampling_factor'][0]}x",
            'score': layer['segmentation_score'],
            'reasoning': get_reasoning(layer)
        })
    
    with open('segmentation_layer_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed analysis saved to: segmentation_layer_analysis.json")
    
    return scored_layers[:5]

def get_reasoning(layer):
    """Get reasoning for why this layer is good for segmentation."""
    reasons = []
    
    h, w = layer['spatial_size']
    ds = layer['downsampling_factor'][0]
    
    if ds <= 8:
        reasons.append("High spatial resolution preserves fine details")
    elif ds <= 16:
        reasons.append("Good balance between spatial detail and semantic information")
    
    stage = layer.get('model_stage', 20)
    if 10 <= stage <= 25:
        reasons.append("Middle-layer features combine low-level details with semantic understanding")
    
    if 64 <= layer['channels'] <= 256:
        reasons.append("Optimal channel depth for rich feature representation")
    
    return "; ".join(reasons)

if __name__ == "__main__":
    analyze_layers_for_segmentation()