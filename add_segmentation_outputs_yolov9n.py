#!/usr/bin/env python3
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def add_segmentation_outputs_yolov9n(input_model_path, output_model_path):
    """
    Add recommended segmentation layer outputs to YOLOv9n model
    while preserving the original ir_version
    """
    
    # Load the ONNX model
    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)
    
    # Store original ir_version
    original_ir_version = model.ir_version
    print(f"Original IR version: {original_ir_version}")
    
    # Convert to GraphSurgeon
    graph = gs.import_onnx(model)
    
    # Recommended layers from the analysis report for YOLOv9n
    recommended_layers = [
        "/model.22/Concat_output_0",      # 80x80, 89ch, score: 0.7293
        "/model.14/Concat_output_0",      # 80x80, 80ch, score: 0.7117
        "/model.2/cv3/act/Mul_output_0",  # 160x160, 8ch, score: 0.4975 (Swish)
        "/model.11/Concat_output_0"       # 40x40, 112ch, score: 0.6501
    ]
    
    # Find the tensors for the recommended layers
    added_outputs = []
    for layer_name in recommended_layers:
        # Search for the tensor in the graph
        tensors = [tensor for tensor in graph.tensors().values() if tensor.name == layer_name]
        
        if tensors:
            tensor = tensors[0]
            # Create new output with descriptive name
            readable_name = layer_name.replace("/", "_").replace(".", "_")
            output_name = f"segmentation{readable_name}"
            
            # Create output variable
            output = gs.Variable(
                name=output_name,
                dtype=tensor.dtype,
                shape=tensor.shape
            )
            
            # Add identity node to connect the tensor to output
            identity_node = gs.Node(
                op="Identity",
                name=f"Identity{readable_name}",
                inputs=[tensor],
                outputs=[output]
            )
            
            graph.nodes.append(identity_node)
            graph.outputs.append(output)
            added_outputs.append((layer_name, output_name))
            print(f"Added output: {layer_name} -> {output_name}")
        else:
            print(f"Warning: Could not find tensor {layer_name}")
    
    # Cleanup and export
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    
    # Restore original ir_version
    model.ir_version = original_ir_version
    print(f"\nRestored IR version: {model.ir_version}")
    
    # Save the modified model
    onnx.save(model, output_model_path)
    
    print(f"\nModified model saved to: {output_model_path}")
    print(f"Total outputs added: {len(added_outputs)}")
    print(f"Total outputs in model: {len(graph.outputs)}")
    
    # Verify the saved model
    try:
        # Load and check ir_version
        saved_model = onnx.load(output_model_path)
        print(f"\nVerification - Saved model IR version: {saved_model.ir_version}")
        
        onnx.checker.check_model(output_model_path)
        print("Model validation: PASSED")
    except Exception as e:
        print(f"Model validation: FAILED - {e}")
    
    return added_outputs

if __name__ == "__main__":
    input_model = "yolov9_n_wholebody25_Nx3x640x640.onnx"
    output_model = "yolov9_n_wholebody25_Nx3x640x640_featext.onnx"
    
    added = add_segmentation_outputs_yolov9n(input_model, output_model)
    
    if added:
        print("\nSuccessfully added segmentation outputs:")
        for i, (original, new) in enumerate(added, 1):
            print(f"  {i}. {original} -> {new}")
    else:
        print("\nNo outputs were added.")