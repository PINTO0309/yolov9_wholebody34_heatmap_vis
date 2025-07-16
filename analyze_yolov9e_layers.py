import onnx
import json
from collections import defaultdict

def analyze_model_layers(model_path):
    """Extract and analyze all layers from ONNX model"""
    model = onnx.load(model_path)
    
    # Get input shape
    input_shape = None
    for input_tensor in model.graph.input:
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value else 'dynamic')
        input_shape = shape
        break
    
    print(f"Model input shape: {input_shape}")
    
    # Analyze all nodes
    layer_info = {}
    concat_layers = []
    swish_layers = []
    
    # Create output name to shape mapping
    output_shapes = {}
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    
    for node in model.graph.node:
        node_type = node.op_type
        node_name = node.name if node.name else f"{node_type}_{len(layer_info)}"
        
        # Get output shape if available
        output_shape = None
        if node.output and node.output[0] in value_infos:
            vi = value_infos[node.output[0]]
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value else 'dynamic')
            output_shape = shape
        
        layer_info[node_name] = {
            'type': node_type,
            'inputs': list(node.input),
            'outputs': list(node.output),
            'output_shape': output_shape
        }
        
        # Identify Concat layers
        if node_type == 'Concat':
            concat_layers.append({
                'name': node.output[0] if node.output else node_name,
                'node_name': node_name,
                'shape': output_shape
            })
        
        # Identify Swish activation (Sigmoid + Mul pattern)
        if node_type == 'Mul' and len(node.input) == 2:
            # Check if one input comes from Sigmoid
            for other_node in model.graph.node:
                if other_node.op_type == 'Sigmoid' and other_node.output[0] in node.input:
                    swish_layers.append({
                        'name': node.output[0] if node.output else node_name,
                        'node_name': node_name,
                        'shape': output_shape
                    })
                    break
    
    # Calculate resolution and channels for each layer
    def calculate_resolution_info(shape):
        if shape and len(shape) == 4:  # NCHW format
            _, channels, height, width = shape
            if isinstance(height, int) and isinstance(width, int):
                downsampling = input_shape[2] // height if input_shape and isinstance(input_shape[2], int) else None
                return {
                    'resolution': f"{height}×{width}",
                    'channels': channels,
                    'downsampling': downsampling
                }
        return None
    
    # Process concat layers
    print("\n=== Concat Layers ===")
    for layer in concat_layers:
        if layer['shape']:
            res_info = calculate_resolution_info(layer['shape'])
            if res_info:
                layer.update(res_info)
                print(f"{layer['name']}: {res_info['resolution']} ({res_info['downsampling']}x downsampling), {res_info['channels']} channels")
    
    # Process swish layers
    print("\n=== Swish Activation Layers ===")
    for layer in swish_layers:
        if layer['shape']:
            res_info = calculate_resolution_info(layer['shape'])
            if res_info:
                layer.update(res_info)
                print(f"{layer['name']}: {res_info['resolution']} ({res_info['downsampling']}x downsampling), {res_info['channels']} channels")
    
    # Save results
    results = {
        'input_shape': input_shape,
        'concat_layers': concat_layers,
        'swish_layers': swish_layers,
        'total_layers': len(layer_info)
    }
    
    with open('yolov9e_layer_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = analyze_model_layers("yolov9_e_wholebody25_0100_1x3x640x640.onnx")
    print(f"\nTotal layers analyzed: {results['total_layers']}")
    print(f"Concat layers found: {len(results['concat_layers'])}")
    print(f"Swish activation layers found: {len(results['swish_layers'])}")