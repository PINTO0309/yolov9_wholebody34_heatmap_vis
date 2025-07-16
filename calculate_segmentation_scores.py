import json
import numpy as np

def calculate_segmentation_score(resolution, channels, downsampling):
    """
    Calculate segmentation suitability score based on:
    - Resolution: Higher is better for detail (weight: 0.4)
    - Channel depth: More channels = richer features (weight: 0.3)
    - Optimal downsampling: 8x is ideal balance (weight: 0.3)
    """
    # Parse resolution
    height, width = map(int, resolution.split('×'))
    
    # Resolution score (normalized to 0-1, 160x160 as reference)
    resolution_score = (height * width) / (160 * 160)
    resolution_score = min(resolution_score, 1.0)  # Cap at 1.0
    
    # Channel score (normalized, assuming 512 channels as excellent)
    channel_score = min(channels / 512, 1.0)
    
    # Downsampling score (8x is optimal, penalize deviation)
    if downsampling == 8:
        downsampling_score = 1.0
    elif downsampling == 4:
        downsampling_score = 0.7  # Good for detail but less semantic
    elif downsampling == 16:
        downsampling_score = 0.8  # Good semantic but less detail
    elif downsampling == 32:
        downsampling_score = 0.5  # Too coarse
    else:
        downsampling_score = 0.3  # Very high resolution, limited semantic
    
    # Weighted total score
    total_score = (resolution_score * 0.4 + 
                   channel_score * 0.3 + 
                   downsampling_score * 0.3)
    
    return {
        'total_score': round(total_score, 4),
        'resolution_score': round(resolution_score, 4),
        'channel_score': round(channel_score, 4),
        'downsampling_score': round(downsampling_score, 4)
    }

def analyze_layers_for_segmentation():
    """Analyze and rank layers for segmentation suitability"""
    
    # Load layer analysis
    with open('yolov9e_layer_analysis.json', 'r') as f:
        data = json.load(f)
    
    # Score all layers
    scored_concat_layers = []
    scored_swish_layers = []
    
    # Process Concat layers
    for layer in data['concat_layers']:
        if 'resolution' in layer and 'channels' in layer and 'downsampling' in layer:
            scores = calculate_segmentation_score(
                layer['resolution'], 
                layer['channels'], 
                layer['downsampling']
            )
            layer.update(scores)
            scored_concat_layers.append(layer)
    
    # Process Swish layers
    for layer in data['swish_layers']:
        if 'resolution' in layer and 'channels' in layer and 'downsampling' in layer:
            scores = calculate_segmentation_score(
                layer['resolution'], 
                layer['channels'], 
                layer['downsampling']
            )
            layer.update(scores)
            scored_swish_layers.append(layer)
    
    # Sort by total score
    scored_concat_layers.sort(key=lambda x: x['total_score'], reverse=True)
    scored_swish_layers.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Print top recommendations
    print("=== Top 10 Concat Layers for Segmentation ===")
    for i, layer in enumerate(scored_concat_layers[:10]):
        print(f"{i+1}. {layer['name']}")
        print(f"   Resolution: {layer['resolution']} ({layer['downsampling']}x downsampling)")
        print(f"   Channels: {layer['channels']}")
        print(f"   Score: {layer['total_score']}")
        print(f"   Breakdown: res={layer['resolution_score']}, ch={layer['channel_score']}, ds={layer['downsampling_score']}")
        print()
    
    print("\n=== Top 10 Swish Layers for Segmentation ===")
    for i, layer in enumerate(scored_swish_layers[:10]):
        print(f"{i+1}. {layer['name']}")
        print(f"   Resolution: {layer['resolution']} ({layer['downsampling']}x downsampling)")
        print(f"   Channels: {layer['channels']}")
        print(f"   Score: {layer['total_score']}")
        print(f"   Breakdown: res={layer['resolution_score']}, ch={layer['channel_score']}, ds={layer['downsampling_score']}")
        print()
    
    # Save detailed results
    results = {
        'concat_layers_ranked': scored_concat_layers,
        'swish_layers_ranked': scored_swish_layers,
        'top_concat': scored_concat_layers[:5],
        'top_swish': scored_swish_layers[:5]
    }
    
    with open('yolov9e_segmentation_scores.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = analyze_layers_for_segmentation()