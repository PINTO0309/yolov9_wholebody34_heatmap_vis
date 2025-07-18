#!/usr/bin/env python3
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def add_segmentation_outputs_yolov9c_preserve_ir(input_model_path, output_model_path):
    """
    Add recommended segmentation layer outputs to YOLOv9-C model
    while preserving the original ir_version
    Based on recommendations from segmentation_analysis_report_yolov9c.md
    """
    
    # Load the ONNX model
    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)
    
    # Store original ir_version
    original_ir_version = model.ir_version
    print(f"Original IR version: {original_ir_version}")
    
    # Convert to GraphSurgeon
    graph = gs.import_onnx(model)
    
    # Recommended layers from segmentation_analysis_report_yolov9c.md
    # Priority order based on the report recommendations
    recommended_layers = [
        # 最優先推奨
        "/model.4/Concat_output_0",    # 80x80, 512ch, score: 0.3875 - 理想的なバランス
        
        # 中解像度融合層
        "/model.5/Concat_output_0",    # 40x40, 512ch, score: 0.3438 - 深い意味特徴
        
        # 高解像度融合層
        "/model.2/Concat_output_0",    # 160x160, 256ch, score: 0.3250 - 詳細な境界線
        
        # 超高解像度層
        "/model.0/act/Mul_output_0",   # 320x320, 64ch, score: 0.3875 - 最高解像度
        
        # 深層融合層
        "/model.7/Concat_output_0",    # 20x20, 512ch, score: 0.3219 - グローバル特徴
        
        # CV4モジュール高品質特徴（追加オプション）
        "/model.4/cv4/conv/Conv_output_0",    # 80x80, 512ch - CV4特徴
        "/model.4/cv4/act/Mul_output_0"       # 80x80, 512ch - CV4活性化後
    ]
    
    # Layer descriptions for documentation
    layer_descriptions = {
        "/model.4/Concat_output_0": "Primary feature (8x downsample, ideal balance)",
        "/model.5/Concat_output_0": "Mid-resolution feature (16x downsample, deep semantic)",
        "/model.2/Concat_output_0": "High-resolution feature (4x downsample, fine details)",
        "/model.0/act/Mul_output_0": "Ultra high-resolution feature (2x downsample)",
        "/model.7/Concat_output_0": "Deep global feature (32x downsample)",
        "/model.4/cv4/conv/Conv_output_0": "CV4 module feature (high quality)",
        "/model.4/cv4/act/Mul_output_0": "CV4 module activated feature"
    }
    
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
            
            # Print with description
            desc = layer_descriptions.get(layer_name, "")
            print(f"Added output: {layer_name} -> {output_name}")
            if desc:
                print(f"  Description: {desc}")
            print(f"  Shape: {tensor.shape}")
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
    
    # Print summary of outputs
    print("\nOutput summary:")
    print("Original outputs:")
    original_output_count = len(graph.outputs) - len(added_outputs)
    for i in range(original_output_count):
        print(f"  - {graph.outputs[i].name}")
    
    print("\nAdded segmentation outputs:")
    for original, new in added_outputs:
        print(f"  - {new} (from {original})")
    
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
    input_model = "yolov9_c_wholebody25_Nx3x640x640.onnx"
    output_model = "yolov9_c_wholebody25_Nx3x640x640_featext.onnx"
    
    print("YOLOv9-C Segmentation Feature Extraction Model Generator")
    print("=" * 60)
    print("This script adds recommended segmentation layer outputs")
    print("based on segmentation_analysis_report_yolov9c.md")
    print("=" * 60)
    
    added = add_segmentation_outputs_yolov9c_preserve_ir(input_model, output_model)
    
    if added:
        print("\n" + "=" * 60)
        print("Successfully added segmentation outputs:")
        for i, (original, new) in enumerate(added, 1):
            print(f"  {i}. {original} -> {new}")
        print("\nThese outputs can be used for instance segmentation tasks.")
        print("\nRecommended usage based on requirements:")
        print("  - Balance: model.4 + model.5 (best accuracy/speed)")
        print("  - Speed: model.4 only (single layer, 512 channels)")
        print("  - High accuracy: model.2 + model.4 + model.5 (multi-scale)")
        print("  - Research: All layers including CV4 modules")
        print("\nYOLOv9-C features:")
        print("  - Compact size (~101MB)")
        print("  - Max 1024 channels (same as YOLOv9-E)")
        print("  - Rich CV3/CV4 modules (206/25)")
    else:
        print("\nNo outputs were added.")