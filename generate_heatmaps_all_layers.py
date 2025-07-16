#!/usr/bin/env python3
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
import onnx_graphsurgeon as gs
import os
import glob
import argparse
from tqdm import tqdm
import shutil

def find_layer_outputs(onnx_model, layer_types=['Conv', 'Concat', 'Mul'], target_layers=None):
    """Find all outputs from specified layer types"""
    graph = onnx_model.graph
    layer_outputs = []
    
    for node in graph.node:
        if node.op_type in layer_types:
            # Check if this is a target layer (if specified)
            if target_layers:
                # Match by output name
                if any(target in node.output[0] for target in target_layers):
                    if node.output:
                        layer_outputs.append({
                            'name': node.output[0],
                            'node_name': node.name,
                            'type': node.op_type
                        })
            else:
                # Include all layers of specified types
                if node.output:
                    layer_outputs.append({
                        'name': node.output[0],
                        'node_name': node.name,
                        'type': node.op_type
                    })
    
    return layer_outputs

def modify_onnx_model(model_path, layer_outputs, output_path):
    """Modify ONNX model to expose intermediate outputs"""
    # Load model with graph surgeon
    graph = gs.import_onnx(onnx.load(model_path))
    
    # Find tensors to expose
    tensors_to_expose = []
    for layer_info in layer_outputs:
        tensor_name = layer_info['name']
        # Find the tensor in the graph
        for tensor in graph.tensors().values():
            if tensor.name == tensor_name:
                tensors_to_expose.append(tensor)
                break
    
    # Add these tensors as outputs
    graph.outputs.extend(tensors_to_expose)
    
    # Clean up and export
    graph.cleanup()
    onnx_model = gs.export_onnx(graph)
    onnx.save(onnx_model, output_path)
    
    return len(tensors_to_expose)

def preprocess_image(image_path, input_size=(640, 640)):
    """Preprocess image for YOLO model"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_image = image.copy()
    
    # Resize to model input size
    image = cv2.resize(image, input_size)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image, original_image

def generate_heatmap(feature_map, colormap=cv2.COLORMAP_JET):
    """Generate heatmap from feature map"""
    # Handle different feature map dimensions
    if len(feature_map.shape) == 4:  # Batch x Channels x Height x Width
        # Average across channels
        heatmap = np.mean(feature_map[0], axis=0)
    elif len(feature_map.shape) == 3:  # Channels x Height x Width
        # Average across channels
        heatmap = np.mean(feature_map, axis=0)
    elif len(feature_map.shape) == 2:  # Height x Width
        heatmap = feature_map
    else:
        raise ValueError(f"Unexpected feature map shape: {feature_map.shape}")
    
    # Normalize to 0-255
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    return heatmap_colored

def create_overlay(original_image, heatmap, alpha=0.5):
    """Create overlay of heatmap on original image"""
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(original_image, 1-alpha, heatmap_resized, alpha, 0)
    
    return overlay

def process_images(model_path, image_pattern, layer_outputs, alpha=0.5, 
                  generate_overlay=True, colormap=cv2.COLORMAP_JET):
    """Process all images and generate heatmaps"""
    # Create output directories
    os.makedirs("heatmaps", exist_ok=True)
    if generate_overlay:
        overlay_dir = f"overlays_{int(alpha*100)}"
        os.makedirs(overlay_dir, exist_ok=True)
    
    # Get all image files
    image_files = glob.glob(image_pattern)
    if not image_files:
        print(f"No images found matching pattern: {image_pattern}")
        return
    
    # Create modified model
    modified_model_path = "temp_modified_model.onnx"
    modify_onnx_model(model_path, layer_outputs, modified_model_path)
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(modified_model_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_size = (input_shape[3], input_shape[2])  # width, height
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Preprocess image
        preprocessed_img, original_img = preprocess_image(img_path, input_size)
        
        # Run inference
        outputs = session.run(None, {input_name: preprocessed_img})
        
        # Generate heatmaps for each layer
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Skip the original model outputs (first few outputs)
        original_output_count = len(outputs) - len(layer_outputs)
        
        for idx, (output, layer_info) in enumerate(zip(outputs[original_output_count:], layer_outputs)):
            # Generate heatmap
            heatmap = generate_heatmap(output, colormap)
            
            # Save heatmap
            layer_name = layer_info['name'].replace('/', '_').replace('.', '_')
            heatmap_path = f"heatmaps/{base_name}_{layer_name}.png"
            cv2.imwrite(heatmap_path, heatmap)
            
            # Generate overlay if requested
            if generate_overlay:
                overlay = create_overlay(original_img, heatmap, alpha)
                overlay_path = f"{overlay_dir}/{base_name}_{layer_name}_overlay.png"
                cv2.imwrite(overlay_path, overlay)
    
    # Clean up temporary model
    os.remove(modified_model_path)
    
    print(f"Generated heatmaps for {len(image_files)} images and {len(layer_outputs)} layers")

def main():
    parser = argparse.ArgumentParser(description='Generate heatmaps from YOLO model layers')
    parser.add_argument('--model', type=str, default='yolov9_e_wholebody34_0100_1x3x480x640.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--images', type=str, default='*.jpg',
                        help='Pattern for input images')
    parser.add_argument('--layers', nargs='+', default=None,
                        help='Specific layer names to visualize')
    parser.add_argument('--layer-types', nargs='+', default=['Conv', 'Concat', 'Mul'],
                        help='Types of layers to include')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency for overlay (0.0-1.0)')
    parser.add_argument('--no-overlay', action='store_true',
                        help='Skip overlay generation')
    parser.add_argument('--colormap', type=str, default='jet',
                        help='Colormap for heatmap visualization')
    
    args = parser.parse_args()
    
    # Map colormap names to OpenCV constants
    colormap_dict = {
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'turbo': cv2.COLORMAP_TURBO,
    }
    colormap = colormap_dict.get(args.colormap.lower(), cv2.COLORMAP_JET)
    
    # Load model and find layers
    print(f"Loading model: {args.model}")
    model = onnx.load(args.model)
    
    print(f"Finding {args.layer_types} layers...")
    layer_outputs = find_layer_outputs(model, args.layer_types, args.layers)
    
    if not layer_outputs:
        print("No matching layers found!")
        return
    
    print(f"Found {len(layer_outputs)} matching layers:")
    for layer in layer_outputs[:10]:  # Show first 10
        print(f"  - {layer['name']} ({layer['type']})")
    if len(layer_outputs) > 10:
        print(f"  ... and {len(layer_outputs) - 10} more")
    
    # Process images
    process_images(
        args.model,
        args.images,
        layer_outputs,
        alpha=args.alpha,
        generate_overlay=not args.no_overlay,
        colormap=colormap
    )

if __name__ == "__main__":
    main()