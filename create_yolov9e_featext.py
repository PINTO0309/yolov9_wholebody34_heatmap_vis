#!/usr/bin/env python3
import onnx
import onnx_graphsurgeon as gs

def create_feature_extraction_model(input_model_path, output_model_path, feature_layers):
    """Create a modified ONNX model with additional outputs for feature extraction"""
    print(f"Loading model: {input_model_path}")
    graph = gs.import_onnx(onnx.load(input_model_path))
    
    # Find tensors to expose as additional outputs
    tensors_to_add = []
    found_layers = []
    
    for tensor_name in feature_layers:
        found = False
        for tensor in graph.tensors().values():
            if tensor.name == tensor_name:
                tensors_to_add.append(tensor)
                found_layers.append(tensor_name)
                found = True
                print(f"Found layer: {tensor_name}")
                break
        
        if not found:
            print(f"Warning: Layer {tensor_name} not found in model")
    
    if not tensors_to_add:
        print("No matching layers found!")
        return False
    
    # Add these tensors as additional outputs
    print(f"\nAdding {len(tensors_to_add)} layers as additional outputs...")
    graph.outputs.extend(tensors_to_add)
    
    # Clean up the graph
    graph.cleanup()
    
    # Export the modified model
    onnx_model = gs.export_onnx(graph)
    
    # Save the model
    onnx.save(onnx_model, output_model_path)
    
    print(f"\nModified model saved to: {output_model_path}")
    print(f"Original outputs: {len(graph.outputs) - len(tensors_to_add)}")
    print(f"Added outputs: {len(tensors_to_add)}")
    print(f"Total outputs: {len(graph.outputs)}")
    
    # Print added output names
    print("\nAdded feature extraction outputs:")
    for i, layer_name in enumerate(found_layers):
        print(f"  {i+1}. {layer_name}")
    
    return True

def main():
    # Input and output model paths
    input_model = "yolov9_e_wholebody25_0100_1x3x640x640.onnx"
    output_model = "yolov9_e_wholebody25_0100_1x3x640x640_featext.onnx"
    
    # Recommended feature extraction layers based on analysis
    feature_layers = [
        "/model.3/Concat_output_0",      # Top-1: 160×160, 256ch, score 0.76
        "/model.19/Concat_output_0",     # Top-2: 160×160, 256ch, score 0.76
        "/model.5/Concat_output_0",      # Top-3: 80×80, 512ch, score 0.70
        "/model.22/Concat_output_0",     # Top-4: 80×80, 512ch, score 0.70
        "/model.34/Concat_output_0"      # Top-5: 80×80, 1024ch, score 0.70
    ]
    
    print("Creating feature extraction model for YOLOv9e")
    print("=" * 60)
    
    success = create_feature_extraction_model(input_model, output_model, feature_layers)
    
    if success:
        print("\n" + "=" * 60)
        print("Feature extraction model created successfully!")
        print(f"\nUsage example:")
        print(f"  import onnxruntime as ort")
        print(f"  session = ort.InferenceSession('{output_model}')")
        print(f"  outputs = session.run(None, {{'images': input_image}})")
        print(f"  # outputs[0-2]: Original YOLOv9 outputs")
        print(f"  # outputs[3]: /model.3/Concat_output_0 (160×160)")
        print(f"  # outputs[4]: /model.19/Concat_output_0 (160×160)")
        print(f"  # outputs[5]: /model.5/Concat_output_0 (80×80)")
        print(f"  # outputs[6]: /model.22/Concat_output_0 (80×80)")
        print(f"  # outputs[7]: /model.34/Concat_output_0 (80×80)")

if __name__ == "__main__":
    main()