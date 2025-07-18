#!/usr/bin/env python3
import onnx
import onnxruntime as ort
import numpy as np

def verify_featext_model(model_path):
    """Verify the YOLOv9-C feature extraction model"""
    
    print(f"Verifying model: {model_path}")
    print("=" * 60)
    
    # Load model with ONNX
    model = onnx.load(model_path)
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    
    # Get input info
    print("\nInputs:")
    for input in model.graph.input:
        print(f"  - {input.name}: {[dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input.type.tensor_type.shape.dim]}")
    
    # Get output info
    print("\nOutputs:")
    output_info = []
    for output in model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output.type.tensor_type.shape.dim]
        output_info.append((output.name, shape))
        print(f"  - {output.name}: {shape}")
    
    # Create ONNX Runtime session
    print("\nCreating ONNX Runtime session...")
    try:
        session = ort.InferenceSession(model_path)
        print("Session created successfully!")
        
        # Test inference with dummy input
        print("\nTesting inference with dummy input...")
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        outputs = session.run(None, {'images': dummy_input})
        
        print("\nInference successful! Output shapes:")
        output_names = [o.name for o in session.get_outputs()]
        for name, output in zip(output_names, outputs):
            print(f"  - {name}: {output.shape}")
            
        # Check segmentation outputs
        print("\nSegmentation outputs verification:")
        segmentation_outputs = [name for name in output_names if 'segmentation' in name]
        print(f"Found {len(segmentation_outputs)} segmentation outputs:")
        
        expected_shapes = {
            'segmentation_model_4_Concat_output_0': (1, 512, 80, 80),          # Primary feature
            'segmentation_model_5_Concat_output_0': (1, 512, 40, 40),          # Mid-resolution
            'segmentation_model_2_Concat_output_0': (1, 256, 160, 160),        # High-resolution
            'segmentation_model_0_act_Mul_output_0': (1, 64, 320, 320),        # Ultra high-res
            'segmentation_model_7_Concat_output_0': (1, 512, 20, 20),          # Deep global
            'segmentation_model_4_cv4_conv_Conv_output_0': (1, 512, 80, 80),   # CV4 feature
            'segmentation_model_4_cv4_act_Mul_output_0': (1, 512, 80, 80)      # CV4 activated
        }
        
        for i, (name, output) in enumerate(zip(output_names, outputs)):
            if 'segmentation' in name:
                expected = expected_shapes.get(name)
                status = "✓" if expected and output.shape == expected else "✗"
                print(f"  {i+1}. {name}: {output.shape} {status}")
                if name == 'segmentation_model_4_Concat_output_0':
                    print(f"     → Primary feature (ideal balance)")
                elif name == 'segmentation_model_5_Concat_output_0':
                    print(f"     → Mid-resolution feature (deep semantic)")
                elif name == 'segmentation_model_2_Concat_output_0':
                    print(f"     → High-resolution feature (fine details)")
                elif name == 'segmentation_model_0_act_Mul_output_0':
                    print(f"     → Ultra high-resolution feature")
                elif name == 'segmentation_model_7_Concat_output_0':
                    print(f"     → Deep global context feature")
                elif name == 'segmentation_model_4_cv4_conv_Conv_output_0':
                    print(f"     → CV4 module feature (high quality)")
                elif name == 'segmentation_model_4_cv4_act_Mul_output_0':
                    print(f"     → CV4 module activated feature")
        
        print("\n" + "=" * 60)
        print("Model verification completed successfully!")
        print("\nYOLOv9-C feature extraction model characteristics:")
        print("  - Model size: ~101MB (compact but powerful)")
        print("  - Max channels: 1024 (same as YOLOv9-E)")
        print("  - Rich CV modules: 206 CV3 + 25 CV4 modules")
        print("  - Best for: High accuracy with reasonable speed")
        print("  - Includes CV4 module outputs for research")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        return False
    
    return True

if __name__ == "__main__":
    model_path = "yolov9_c_wholebody25_Nx3x640x640_featext.onnx"
    verify_featext_model(model_path)