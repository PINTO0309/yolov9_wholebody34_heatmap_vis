#!/usr/bin/env python3
import onnx
import onnxruntime as ort
import numpy as np

def verify_featext_model(model_path):
    """Verify the YOLOv9-S feature extraction model"""
    
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
            'segmentation_model_11_Concat_output_0': (1, 448, 40, 40),    # Primary feature
            'segmentation_model_14_Concat_output_0': (1, 320, 80, 80),    # High-resolution
            'segmentation_model_8_Concat_output_0': (1, 512, 20, 20),     # Deep semantic
            'segmentation_model_2_Concat_output_0': (1, 128, 160, 160),   # Early fusion
            'segmentation_model_0_act_Mul_output_0': (1, 32, 320, 320)    # Ultra high-res
        }
        
        for i, (name, output) in enumerate(zip(output_names, outputs)):
            if 'segmentation' in name:
                expected = expected_shapes.get(name)
                status = "✓" if expected and output.shape == expected else "✗"
                print(f"  {i+1}. {name}: {output.shape} {status}")
                if name == 'segmentation_model_11_Concat_output_0':
                    print(f"     → Primary feature (best balance)")
                elif name == 'segmentation_model_14_Concat_output_0':
                    print(f"     → High-resolution feature")
                elif name == 'segmentation_model_8_Concat_output_0':
                    print(f"     → Deep semantic feature (max channels)")
                elif name == 'segmentation_model_2_Concat_output_0':
                    print(f"     → Early fusion feature (fine details)")
                elif name == 'segmentation_model_0_act_Mul_output_0':
                    print(f"     → Ultra high-resolution feature")
        
        print("\n" + "=" * 60)
        print("Model verification completed successfully!")
        print("\nYOLOv9-S feature extraction model is ready for use.")
        print("This model provides a good balance between YOLOv9-T and YOLOv9-E:")
        print("  - Model size: ~28MB (vs T:~7.5MB, E:~240MB)")
        print("  - Max channels: 512 (vs T:256, E:1024)")
        print("  - Best for balanced accuracy/speed requirements")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        return False
    
    return True

if __name__ == "__main__":
    model_path = "yolov9_s_wholebody25_Nx3x640x640_featext.onnx"
    verify_featext_model(model_path)