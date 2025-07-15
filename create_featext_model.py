#!/usr/bin/env python3
"""
Create YOLOv9n model with additional feature extraction outputs for segmentation
"""
import onnx
import onnx_graphsurgeon as gs

def create_feature_extraction_model():
    """Add recommended segmentation layer outputs to YOLOv9n model"""
    
    # Load the original model
    input_model_path = "yolov9_n_wholebody25_0100_1x3x640x640.onnx"
    output_model_path = "yolov9_n_wholebody25_0100_1x3x640x640_featext.onnx"
    
    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)
    graph = gs.import_onnx(model)
    
    # Define the recommended layers to add as outputs
    # Based on the segmentation analysis report
    recommended_layers = [
        # Concat layers (best for segmentation)
        "/model.22/Concat_output_0",      # Best overall: 80x80, 89ch, score 0.7293
        "/model.14/Concat_output_0",      # FPN fusion: 80x80, 80ch, score 0.7117
        "/model.11/Concat_output_0",      # Semantic: 40x40, 112ch, score 0.6501
        
        # Swish activation layers (alternative options)
        "/model.22/cv3.0/cv3.0.0/act/Mul_output_0",  # 80x80, 50ch, score 0.5543
        "/model.2/cv3/act/Mul_output_0",             # High-res: 160x160, 8ch, score 0.4975
        "/model.22/cv3.1/cv3.1.0/act/Mul_output_0",  # Semantic: 40x40, 50ch
    ]
    
    # Track which outputs were successfully added
    added_outputs = []
    not_found = []
    
    # Add each recommended layer as an output
    for layer_name in recommended_layers:
        found = False
        for tensor in graph.tensors().values():
            if tensor.name == layer_name:
                # Check if it's already an output
                if tensor not in graph.outputs:
                    graph.outputs.append(tensor)
                    added_outputs.append(layer_name)
                    print(f"✓ Added output: {layer_name}")
                else:
                    print(f"  Already an output: {layer_name}")
                found = True
                break
        
        if not found:
            not_found.append(layer_name)
            print(f"✗ Not found: {layer_name}")
    
    # Export the modified model
    print(f"\nExporting modified model...")
    modified_model = gs.export_onnx(graph)
    
    # Save the model
    onnx.save(modified_model, output_model_path)
    print(f"Saved to: {output_model_path}")
    
    # Summary
    print(f"\nSummary:")
    print(f"- Original outputs: {len(model.graph.output)}")
    print(f"- Final outputs: {len(modified_model.graph.output)}")
    print(f"- Added outputs: {len(added_outputs)}")
    
    if added_outputs:
        print(f"\nAdded feature extraction outputs:")
        for name in added_outputs:
            print(f"  - {name}")
    
    if not_found:
        print(f"\nWarning: The following layers were not found:")
        for name in not_found:
            print(f"  - {name}")
    
    return output_model_path, added_outputs

if __name__ == "__main__":
    output_path, added = create_feature_extraction_model()
    
    if added:
        print(f"\n✓ Successfully created {output_path} with {len(added)} additional outputs")
        print("\nYou can now use this model for feature extraction and segmentation tasks.")
        print("The added outputs provide multi-scale features suitable for instance segmentation.")
    else:
        print("\n✗ No outputs were added. Please check the layer names.")