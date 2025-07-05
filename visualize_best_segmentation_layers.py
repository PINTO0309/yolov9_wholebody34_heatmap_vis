#!/usr/bin/env python3
"""
Visualize the best layers for segmentation with heatmaps.
"""
import os
import json

def generate_heatmaps_for_best_layers():
    """Generate heatmaps for the best segmentation layers."""
    
    # Load the analysis results
    with open('segmentation_layer_analysis.json', 'r') as f:
        results = json.load(f)
    
    # Get top 3 recommendations
    top_layers = results['analysis_summary']['top_recommendations'][:3]
    
    print("Top layers for instance segmentation:")
    print("=" * 80)
    
    for i, layer in enumerate(top_layers, 1):
        print(f"\n{i}. Layer: {layer['name']}")
        print(f"   Shape: {layer['shape']}")
        print(f"   Downsampling: {layer['downsampling']}")
        print(f"   Score: {layer['score']:.4f}")
        print(f"   Reasoning: {layer['reasoning']}")
    
    print("\n" + "=" * 80)
    print("\nGenerating heatmaps for the top 3 layers...")
    
    # Generate heatmaps using the activated flag
    layer_names = [layer['name'] for layer in top_layers[:3]]
    
    # Create command to generate heatmaps
    cmd = f"python generate_heatmaps_unified.py --activated --layers {' '.join(layer_names)} --alpha 0.6"
    
    print(f"\nCommand to run:")
    print(cmd)
    
    # Also create a comparison script
    print("\n" + "=" * 80)
    print("Recommendation for segmentation tasks:")
    print("\nBased on the analysis, the best layers for instance segmentation are:")
    
    print("\n1. PRIMARY CHOICE: /model.22/cv3/cv3.0/cv3/act/Mul_output_0")
    print("   - Resolution: 128x60x80 (8x downsampling)")
    print("   - Stage: model.22 (middle of network)")
    print("   - Combines semantic understanding with spatial detail")
    print("   - Part of cv3 module which aggregates multi-scale features")
    
    print("\n2. ALTERNATIVE: /model.19/cv3/cv3.0/cv3/act/Mul_output_0")
    print("   - Resolution: 64x120x160 (4x downsampling)")
    print("   - Stage: model.19 (earlier in network)")
    print("   - Higher spatial resolution for finer details")
    print("   - Good for small object segmentation")
    
    print("\n3. FOR SEMANTIC SEGMENTATION: /model.25/cv3/cv3.0/cv3/act/Mul_output_0")
    print("   - Resolution: 256x30x40 (16x downsampling)")
    print("   - Stage: model.25 (later in network)")
    print("   - Stronger semantic features")
    print("   - Better for whole-object understanding")
    
    # Save recommendations
    recommendations = {
        "instance_segmentation": {
            "best_layer": "/model.22/cv3/cv3.0/cv3/act/Mul_output_0",
            "command": "python generate_heatmaps_unified.py --activated --layers /model.22/cv3/cv3.0/cv3/act/Mul_output_0 --alpha 0.6",
            "characteristics": {
                "resolution": "60x80",
                "channels": 128,
                "downsampling": "8x",
                "stage": "model.22"
            }
        },
        "fine_detail_segmentation": {
            "best_layer": "/model.19/cv3/cv3.0/cv3/act/Mul_output_0",
            "command": "python generate_heatmaps_unified.py --activated --layers /model.19/cv3/cv3.0/cv3/act/Mul_output_0 --alpha 0.6",
            "characteristics": {
                "resolution": "120x160",
                "channels": 64,
                "downsampling": "4x",
                "stage": "model.19"
            }
        },
        "semantic_segmentation": {
            "best_layer": "/model.25/cv3/cv3.0/cv3/act/Mul_output_0",
            "command": "python generate_heatmaps_unified.py --activated --layers /model.25/cv3/cv3.0/cv3/act/Mul_output_0 --alpha 0.6",
            "characteristics": {
                "resolution": "30x40",
                "channels": 256,
                "downsampling": "16x",
                "stage": "model.25"
            }
        }
    }
    
    with open('segmentation_layer_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print("\nRecommendations saved to: segmentation_layer_recommendations.json")

if __name__ == "__main__":
    generate_heatmaps_for_best_layers()