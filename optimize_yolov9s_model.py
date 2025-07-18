#!/usr/bin/env python3
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def remove_output0_and_optimize_yolov9s(input_model_path, output_model_path):
    """
    Remove output0 from YOLOv9-S featext model and optimize by cleaning up unused nodes
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
    
    print("\nAnalyzing outputs...")
    for output in graph.outputs:
        if output.name == "output0":
            removed_outputs.append(output.name)
            print(f"  - Marked for removal: {output.name} (detection output)")
        else:
            outputs_to_keep.append(output)
            print(f"  - Keeping: {output.name}")
    
    # Update graph outputs
    graph.outputs = outputs_to_keep
    
    print(f"\nOriginal outputs: {original_output_count}")
    print(f"Removed outputs: {len(removed_outputs)}")
    print(f"Remaining outputs: {len(graph.outputs)}")
    
    # List remaining outputs with descriptions
    print("\nRemaining segmentation outputs:")
    segmentation_descriptions = {
        "segmentation_model_11_Concat_output_0": "Primary feature (16x downsample, 448ch)",
        "segmentation_model_14_Concat_output_0": "High-resolution (8x downsample, 320ch)",
        "segmentation_model_8_Concat_output_0": "Deep semantic (32x downsample, 512ch)",
        "segmentation_model_2_Concat_output_0": "Early fusion (4x downsample, 128ch)",
        "segmentation_model_0_act_Mul_output_0": "Ultra high-res (2x downsample, 32ch)"
    }
    
    for i, output in enumerate(graph.outputs):
        shape_str = str(output.shape) if output.shape else "unknown"
        desc = segmentation_descriptions.get(output.name, "")
        print(f"  {i+1}. {output.name}")
        print(f"      Shape: {shape_str}")
        if desc:
            print(f"      Description: {desc}")
    
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
    
    # Calculate model size reduction estimate
    reduction_percentage = ((nodes_before - nodes_after) / nodes_before) * 100
    print(f"Node reduction: {reduction_percentage:.1f}%")
    
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
        print(f"\nVerification - Saved model IR version: {saved_model.ir_version}")
        print(f"Verification - Total outputs: {len(saved_model.graph.output)}")
        
        # Check that output0 is not in the model
        output_names = [o.name for o in saved_model.graph.output]
        if "output0" not in output_names:
            print("Verification - output0 successfully removed: ✓")
        else:
            print("Verification - output0 still present: ✗")
        
        onnx.checker.check_model(output_model_path)
        print("Model validation: PASSED")
    except Exception as e:
        print(f"Model validation: FAILED - {e}")
    
    return removed_outputs

if __name__ == "__main__":
    input_model = "yolov9_s_wholebody25_Nx3x640x640_featext.onnx"
    output_model = "yolov9_s_wholebody25_Nx3x640x640_featext_optimized.onnx"
    
    print("YOLOv9-S Feature Extraction Model Optimizer")
    print("=" * 60)
    print("This script removes the detection output (output0) and")
    print("optimizes the model for pure segmentation feature extraction")
    print("=" * 60)
    
    removed = remove_output0_and_optimize_yolov9s(input_model, output_model)
    
    if removed:
        print(f"\n" + "=" * 60)
        print(f"Successfully removed {len(removed)} output(s):")
        for name in removed:
            print(f"  - {name}")
        print("\nThe optimized model now contains only segmentation outputs.")
        print("This reduces model size and improves inference efficiency.")
        print("\nYOLOv9-S provides a good balance between speed and accuracy,")
        print("with max 512 channels for rich feature representation.")
    else:
        print("\nNo outputs were removed.")