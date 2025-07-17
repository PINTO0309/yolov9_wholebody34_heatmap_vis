#!/usr/bin/env python3
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def remove_output0_and_optimize_preserve_ir(input_model_path, output_model_path):
    """
    Remove output0 from the model and optimize by cleaning up unused nodes
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
    
    # Find and remove output0
    original_output_count = len(graph.outputs)
    outputs_to_keep = []
    removed_outputs = []
    
    for output in graph.outputs:
        if output.name == "output0":
            removed_outputs.append(output.name)
            print(f"Removing output: {output.name}")
        else:
            outputs_to_keep.append(output)
    
    # Update graph outputs
    graph.outputs = outputs_to_keep
    
    print(f"\nOriginal outputs: {original_output_count}")
    print(f"Removed outputs: {len(removed_outputs)}")
    print(f"Remaining outputs: {len(graph.outputs)}")
    
    # List remaining outputs
    print("\nRemaining outputs:")
    for i, output in enumerate(graph.outputs):
        shape_str = str(output.shape) if output.shape else "unknown"
        print(f"  {i+1}. {output.name} (shape: {shape_str})")
    
    # Optimize the graph by removing unused nodes
    print("\nOptimizing model...")
    
    # Count nodes before optimization
    nodes_before = len(graph.nodes)
    
    # Cleanup will remove nodes that don't contribute to any output
    graph.cleanup()
    
    # Toposort to ensure proper node ordering
    graph.toposort()
    
    # Count nodes after optimization
    nodes_after = len(graph.nodes)
    
    print(f"Nodes before optimization: {nodes_before}")
    print(f"Nodes after optimization: {nodes_after}")
    print(f"Nodes removed: {nodes_before - nodes_after}")
    
    # Export the optimized model
    model = gs.export_onnx(graph)
    
    # Restore original ir_version
    model.ir_version = original_ir_version
    print(f"\nRestored IR version: {model.ir_version}")
    
    # Save the optimized model
    onnx.save(model, output_model_path)
    print(f"\nOptimized model saved to: {output_model_path}")
    
    # Verify the saved model
    try:
        # Load and check ir_version
        saved_model = onnx.load(output_model_path)
        print(f"Verification - Saved model IR version: {saved_model.ir_version}")
        
        onnx.checker.check_model(output_model_path)
        print("Model validation: PASSED")
    except Exception as e:
        print(f"Model validation: FAILED - {e}")
    
    return removed_outputs

if __name__ == "__main__":
    input_model = "yolov9_e_wholebody25_Nx3x640x640_featext.onnx"
    output_model = "yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx"
    
    removed = remove_output0_and_optimize_preserve_ir(input_model, output_model)
    
    if removed:
        print(f"\nSuccessfully removed {len(removed)} output(s):")
        for name in removed:
            print(f"  - {name}")
    else:
        print("\nNo outputs were removed.")