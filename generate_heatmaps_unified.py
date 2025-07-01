import onnx
import onnx_graphsurgeon as gs
import numpy as np
import onnxruntime as ort
import cv2
import os
import argparse
from typing import List, Tuple
import math

def find_conv_outputs(graph: gs.Graph) -> List[Tuple[str, gs.Variable]]:
    """Find all Conv layer output tensors in the graph"""
    conv_outputs = []
    for node in graph.nodes:
        if node.op == "Conv":
            output_name = node.outputs[0].name
            conv_outputs.append((output_name, node.outputs[0]))
    return conv_outputs

def get_model_input_shape(model_path: str) -> Tuple[int, int]:
    """Get the input shape (width, height) from ONNX model"""
    model = onnx.load(model_path)
    input_shape = model.graph.input[0].type.tensor_type.shape
    
    # Extract dimensions - typically NCHW format
    height = None
    width = None
    
    for i, dim in enumerate(input_shape.dim):
        if i == 2:  # Height dimension in NCHW
            height = dim.dim_value
        elif i == 3:  # Width dimension in NCHW
            width = dim.dim_value
    
    if height and width:
        return (width, height)  # Return as (width, height) for cv2.resize
    else:
        # Default fallback
        print("Warning: Could not detect input shape from model, using default 640x480")
        return (640, 480)

def modify_onnx_model(input_model_path: str, output_model_path: str, target_layers: List[str] = None) -> Tuple[List[str], Tuple[int, int]]:
    """Modify ONNX model to expose Conv layer outputs and return input shape"""
    # Load the model
    model = onnx.load(input_model_path)
    graph = gs.import_onnx(model)
    
    # Get input shape
    input_shape = get_model_input_shape(input_model_path)

    # Find all Conv outputs
    all_conv_outputs = find_conv_outputs(graph)
    print(f"Found {len(all_conv_outputs)} Conv layers in total")

    # Filter by target layers if specified
    if target_layers:
        conv_outputs = []
        for name, tensor in all_conv_outputs:
            # Check if any target pattern matches the layer name
            for target in target_layers:
                if target in name:
                    conv_outputs.append((name, tensor))
                    break
        print(f"Filtered to {len(conv_outputs)} Conv layers matching targets: {target_layers}")
    else:
        conv_outputs = all_conv_outputs

    # Add Conv outputs to model outputs
    conv_output_names = []
    for name, tensor in conv_outputs:
        if tensor not in graph.outputs:
            graph.outputs.append(tensor)
            conv_output_names.append(name)
            print(f"Added output: {name}")

    # Export modified model
    model_modified = gs.export_onnx(graph)
    onnx.save(model_modified, output_model_path)

    return conv_output_names, input_shape

def preprocess_image(image_path: str, input_size: Tuple[int, int] = (640, 480)) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    """Preprocess image for YOLOv9 model"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    original_size = (img.shape[1], img.shape[0])

    # Resize to model input size
    img_resized = cv2.resize(img, input_size)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # Transpose to NCHW format
    img_transposed = np.transpose(img_normalized, (2, 0, 1))

    # Add batch dimension
    img_batch = np.expand_dims(img_transposed, axis=0)

    return img_batch, original_size, img

def generate_heatmap(feature_map: np.ndarray, original_size: Tuple[int, int],
                    layer_name: str, output_dir: str, invert: bool = False) -> None:
    """Generate and save heatmap from feature map"""
    # Handle different feature map dimensions
    if feature_map.ndim == 4:  # NCHW format
        # Sum across channels
        heatmap_data = np.sum(feature_map[0], axis=0)
    elif feature_map.ndim == 3:  # CHW format
        heatmap_data = np.sum(feature_map, axis=0)
    elif feature_map.ndim == 2:  # HW format
        heatmap_data = feature_map
    else:
        print(f"Unexpected feature map shape for {layer_name}: {feature_map.shape}")
        return

    # Normalize to 0-255
    if heatmap_data.max() > heatmap_data.min():
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min()) * 255
    else:
        heatmap_data = np.zeros_like(heatmap_data) * 255

    heatmap_data = heatmap_data.astype(np.uint8)

    # Invert if requested
    if invert:
        heatmap_data = 255 - heatmap_data

    # Resize to original image size
    heatmap_resized = cv2.resize(heatmap_data, original_size, interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Save heatmap
    output_path = os.path.join(output_dir, f"heatmap_{layer_name.replace('/', '_')}.png")
    cv2.imwrite(output_path, heatmap_colored)
    print(f"Saved heatmap: {output_path}")

def create_overlay(original_img: np.ndarray, heatmap_path: str, alpha: float, output_path: str) -> None:
    """Create and save overlay of heatmap on original image"""
    # Load heatmap
    heatmap_img = cv2.imread(heatmap_path)
    if heatmap_img is None:
        print(f"Warning: Could not load heatmap from {heatmap_path}")
        return

    # Ensure same size
    if original_img.shape[:2] != heatmap_img.shape[:2]:
        heatmap_img = cv2.resize(heatmap_img, (original_img.shape[1], original_img.shape[0]))

    # Create overlay
    overlay = cv2.addWeighted(original_img, 1-alpha, heatmap_img, alpha, 0)

    # Save overlay
    cv2.imwrite(output_path, overlay)

def create_comparison_grid(image_paths: List[str], output_path: str, grid_size: Tuple[int, int] = None,
                          target_size: Tuple[int, int] = None) -> None:
    """Create a grid comparison image from multiple heatmap images"""
    if not image_paths:
        return

    # Load first image to get dimensions
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        print(f"Warning: Could not load image from {image_paths[0]}")
        return

    if target_size is None:
        target_size = (first_img.shape[1] // 2, first_img.shape[0] // 2)  # Default to half size

    # Determine grid size if not specified
    if grid_size is None:
        n_images = len(image_paths)
        cols = math.ceil(math.sqrt(n_images))
        rows = math.ceil(n_images / cols)
        grid_size = (cols, rows)

    cols, rows = grid_size

    # Create empty grid
    grid_width = target_size[0] * cols
    grid_height = target_size[1] * rows
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Fill grid with images
    for idx, img_path in enumerate(image_paths[:cols * rows]):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize image to target size
        img_resized = cv2.resize(img, target_size)

        # Calculate position in grid
        row = idx // cols
        col = idx % cols
        y_start = row * target_size[1]
        y_end = y_start + target_size[1]
        x_start = col * target_size[0]
        x_end = x_start + target_size[0]

        # Place image in grid
        grid[y_start:y_end, x_start:x_end] = img_resized

        # Add layer name as text
        layer_name = os.path.basename(img_path).replace('heatmap_', '').replace('overlay_', '').replace('.png', '')
        # Truncate long names
        if len(layer_name) > 30:
            layer_name = layer_name[:27] + '...'

        # Add white background for text
        cv2.rectangle(grid, (x_start, y_start), (x_end, y_start + 25), (255, 255, 255), -1)
        cv2.putText(grid, layer_name, (x_start + 5, y_start + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Save grid
    cv2.imwrite(output_path, grid)
    print(f"Saved comparison grid: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate heatmaps from YOLOv9 ONNX model')
    parser.add_argument('--model', type=str, default='yolov9_e_wholebody34_0100_1x3x480x640.onnx', help='Path to ONNX model file (default: yolov9_e_wholebody34_0100_1x3x480x640.onnx)')
    parser.add_argument('--image', type=str, default='000000001000.jpg', help='Path to input image file (default: 000000001000.jpg)')
    parser.add_argument('--invert', action='store_true', help='Invert heatmap colors')
    parser.add_argument('--layers', nargs='+', help='Target Conv layer names or patterns (e.g., model.7 cv3.0)')
    parser.add_argument('--alpha', type=float, default=0.4, help='Overlay transparency (0.0-1.0, default: 0.4)')
    parser.add_argument('--no-overlay', action='store_true', help='Skip overlay generation')
    args = parser.parse_args()

    # Paths
    input_model_path = args.model
    modified_model_path = "yolov9_modified_temp.onnx"
    image_path = args.image

    # Check if model file exists
    if not os.path.exists(input_model_path):
        print(f"Error: Model file not found: {input_model_path}")
        return

    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return

    # Output directories
    suffix = "_inverted" if args.invert else ""
    heatmaps_dir = f"heatmaps{suffix}"
    overlay_dir = f"overlays_{int(args.alpha*100)}{suffix}"

    # Create output directories
    os.makedirs(heatmaps_dir, exist_ok=True)
    if not args.no_overlay:
        os.makedirs(overlay_dir, exist_ok=True)

    # Step 1: Inspect and modify ONNX model
    print("Modifying ONNX model to expose Conv outputs...")
    conv_output_names, model_input_size = modify_onnx_model(input_model_path, modified_model_path, args.layers)
    print(f"Detected model input size: {model_input_size[0]}x{model_input_size[1]} (width x height)")

    if not conv_output_names:
        print("No matching Conv layers found!")
        return

    # Step 2: Load and preprocess image
    print("\nPreprocessing image...")
    img_data, original_size, original_img = preprocess_image(image_path, model_input_size)
    print(f"Input shape: {img_data.shape}")

    # Step 3: Run inference with modified model
    print("\nRunning inference...")
    session = ort.InferenceSession(modified_model_path)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Get all output names
    output_names = [output.name for output in session.get_outputs()]
    print(f"Total outputs: {len(output_names)}")

    # Run inference
    outputs = session.run(output_names, {input_name: img_data})

    # Step 4: Generate heatmaps for Conv outputs
    print("\nGenerating heatmaps...")
    generated_count = 0
    for i, output_name in enumerate(output_names):
        if any(conv_name in output_name for conv_name in conv_output_names):
            print(f"\nProcessing {output_name}")
            print(f"Shape: {outputs[i].shape}, Min: {outputs[i].min():.4f}, Max: {outputs[i].max():.4f}")
            generate_heatmap(outputs[i], original_size, output_name, heatmaps_dir, args.invert)
            generated_count += 1

    print(f"\nGenerated {generated_count} heatmaps in '{heatmaps_dir}' directory")

    # Step 5: Generate overlays if requested
    if not args.no_overlay:
        print(f"\nGenerating overlays with {int(args.alpha*100)}% opacity...")
        overlay_count = 0

        for output_name in conv_output_names:
            heatmap_filename = f"heatmap_{output_name.replace('/', '_')}.png"
            heatmap_path = os.path.join(heatmaps_dir, heatmap_filename)

            if os.path.exists(heatmap_path):
                overlay_filename = heatmap_filename.replace("heatmap_", "overlay_")
                overlay_path = os.path.join(overlay_dir, overlay_filename)
                create_overlay(original_img, heatmap_path, args.alpha, overlay_path)
                overlay_count += 1

        print(f"Generated {overlay_count} overlays in '{overlay_dir}' directory")

    # Also save the original image for reference
    cv2.imwrite(os.path.join(heatmaps_dir, "original_image.jpg"), original_img)

    # Generate comparison grids if multiple layers were processed
    if generated_count > 1:
        print("\nGenerating comparison grids...")

        # Collect all heatmap paths
        heatmap_paths = []
        overlay_paths = []

        for output_name in conv_output_names:
            heatmap_filename = f"heatmap_{output_name.replace('/', '_')}.png"
            heatmap_path = os.path.join(heatmaps_dir, heatmap_filename)
            if os.path.exists(heatmap_path):
                heatmap_paths.append(heatmap_path)

            if not args.no_overlay:
                overlay_filename = heatmap_filename.replace("heatmap_", "overlay_")
                overlay_path = os.path.join(overlay_dir, overlay_filename)
                if os.path.exists(overlay_path):
                    overlay_paths.append(overlay_path)

        # Create comparison grid for heatmaps
        if heatmap_paths:
            # First half of images for comparison_grid.png
            half_count = len(heatmap_paths) // 2
            if half_count > 0:
                create_comparison_grid(heatmap_paths[:half_count],
                                     os.path.join(heatmaps_dir, "comparison_grid.png"))

            # Middle portion for comparison_grid_middle.png
            quarter = len(heatmap_paths) // 4
            if quarter > 0:
                middle_start = quarter
                middle_end = quarter * 3
                create_comparison_grid(heatmap_paths[middle_start:middle_end],
                                     os.path.join(heatmaps_dir, "comparison_grid_middle.png"))

        # Create comparison grid for overlays if available
        if overlay_paths and not args.no_overlay:
            # First half of overlays
            half_count = len(overlay_paths) // 2
            if half_count > 0:
                create_comparison_grid(overlay_paths[:half_count],
                                     os.path.join(overlay_dir, "comparison_grid.png"))

            # Middle portion for overlays
            quarter = len(overlay_paths) // 4
            if quarter > 0:
                middle_start = quarter
                middle_end = quarter * 3
                create_comparison_grid(overlay_paths[middle_start:middle_end],
                                     os.path.join(overlay_dir, "comparison_grid_middle.png"))

    # Clean up temporary model
    if os.path.exists(modified_model_path):
        os.remove(modified_model_path)

    print("\nDone!")

if __name__ == "__main__":
    main()