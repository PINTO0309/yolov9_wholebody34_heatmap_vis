#!/usr/bin/env python3
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def add_segmentation_outputs(input_model_path, output_model_path):
    # Load ONNX model
    model = onnx.load(input_model_path)
    graph = gs.import_onnx(model)
    
    # Recommended Concat layer outputs from segmentation_analysis_report_yolov9e.md
    concat_outputs = [
        "/model.3/Concat_output_0",   # 160x160, 256ch, score 0.76
        "/model.19/Concat_output_0",  # 160x160, 256ch, score 0.76
        "/model.5/Concat_output_0",   # 80x80, 512ch, score 0.70
        "/model.22/Concat_output_0",  # 80x80, 512ch, score 0.70
        "/model.34/Concat_output_0"   # 80x80, 1024ch, score 0.70
    ]
    
    # Find and add outputs
    added_outputs = []
    for output_name in concat_outputs:
        # Find the tensor in the graph
        tensors = [tensor for tensor in graph.tensors().values() if tensor.name == output_name]
        if tensors:
            tensor = tensors[0]
            # Create new output with descriptive name
            readable_name = output_name.replace("/", "_").replace(".", "_")
            output = gs.Variable(name=f"segmentation{readable_name}", 
                               dtype=tensor.dtype, 
                               shape=tensor.shape)
            
            # Add identity node to connect the tensor to output
            identity_node = gs.Node(op="Identity",
                                  name=f"Identity{readable_name}",
                                  inputs=[tensor],
                                  outputs=[output])
            
            graph.nodes.append(identity_node)
            graph.outputs.append(output)
            added_outputs.append(output_name)
            print(f"Added output: {output_name} -> segmentation{readable_name}")
        else:
            print(f"Warning: Could not find tensor {output_name}")
    
    # Cleanup and save
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    
    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"\nModified model saved to: {output_model_path}")
    print(f"Total outputs added: {len(added_outputs)}")
    
    return added_outputs

if __name__ == "__main__":
    input_model = "yolov9_e_wholebody25_Nx3x640x640.onnx"
    output_model = "yolov9_e_wholebody25_Nx3x640x640_featext.onnx"
    
    print(f"Loading model: {input_model}")
    added = add_segmentation_outputs(input_model, output_model)
    
    if added:
        print("\nSuccessfully added segmentation outputs:")
        for i, name in enumerate(added, 1):
            print(f"  {i}. {name}")
    else:
        print("\nNo outputs were added.")