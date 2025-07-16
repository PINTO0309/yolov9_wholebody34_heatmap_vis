#!/usr/bin/env python3
import onnx
import onnxruntime as ort
import numpy as np

def verify_model(model_path):
    """Verify the feature extraction model outputs"""
    print(f"Verifying model: {model_path}")
    
    # Load model
    model = onnx.load(model_path)
    
    # Print all outputs
    print(f"\nModel outputs ({len(model.graph.output)} total):")
    for i, output in enumerate(model.graph.output):
        shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append('dynamic')
        print(f"  Output {i}: {output.name}")
        print(f"    Shape: {shape}")
    
    # Test inference
    print("\nTesting inference...")
    session = ort.InferenceSession(model_path)
    
    # Get input info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_shape = [dim if isinstance(dim, int) else 1 for dim in input_shape]
    
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    
    # Create dummy input
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_name: dummy_input})
    
    print(f"\nInference successful! Got {len(outputs)} outputs:")
    for i, output in enumerate(outputs):
        print(f"  Output {i}: shape {output.shape}")
    
    # Expected feature extraction outputs
    expected_outputs = {
        "Original detection output": 0,
        "/model.3/Concat_output_0 (160×160, 256ch)": 1,
        "/model.19/Concat_output_0 (160×160, 256ch)": 2,
        "/model.5/Concat_output_0 (80×80, 512ch)": 3,
        "/model.22/Concat_output_0 (80×80, 512ch)": 4,
        "/model.34/Concat_output_0 (80×80, 1024ch)": 5
    }
    
    print("\nExpected outputs mapping:")
    for desc, idx in expected_outputs.items():
        if idx < len(outputs):
            print(f"  outputs[{idx}]: {desc} - Shape: {outputs[idx].shape}")

if __name__ == "__main__":
    verify_model("yolov9_e_wholebody25_0100_1x3x640x640_featext.onnx")