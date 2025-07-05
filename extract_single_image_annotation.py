#!/usr/bin/env python3
"""
Extract COCO annotations for a specific image and save to a new JSON file.
"""
import json
import os

def extract_image_annotations(coco_json_path, target_image_name, output_path):
    """
    Extract annotations for a specific image from COCO format JSON.
    
    Args:
        coco_json_path: Path to the COCO format JSON file
        target_image_name: Target image filename (e.g., '000000001000.jpg')
        output_path: Path to save the extracted annotations
    """
    # Load COCO JSON
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Find the target image
    target_image = None
    image_id = None
    
    for image in coco_data['images']:
        if image['file_name'] == target_image_name:
            target_image = image
            image_id = image['id']
            break
    
    if target_image is None:
        print(f"Image '{target_image_name}' not found in the dataset")
        return False
    
    print(f"Found image: {target_image_name} (ID: {image_id})")
    
    # Extract annotations for this image
    image_annotations = []
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image_id:
            image_annotations.append(annotation)
    
    print(f"Found {len(image_annotations)} annotations for this image")
    
    # Get categories info
    categories = coco_data.get('categories', [])
    
    # Create output structure with COCO format
    output_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': [target_image],  # Only include the target image
        'annotations': image_annotations,
        'categories': categories
    }
    
    # Save to output file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved extracted annotations to: {output_path}")
    
    # Also create a simplified version with just the essential data
    simplified_output = {
        'image': target_image,
        'annotations': image_annotations,
        'annotation_count': len(image_annotations)
    }
    
    simplified_path = output_path.replace('.json', '_simplified.json')
    with open(simplified_path, 'w') as f:
        json.dump(simplified_output, f, indent=2)
    
    print(f"Also saved simplified version to: {simplified_path}")
    
    return True

def main():
    # Define paths
    annotations_dir = "annotations"
    target_image = "000000001000.jpg"
    
    # Check both train and val datasets
    datasets = [
        ('instances_val2017_person_only_no_crowd.json', 'val2017'),
        ('instances_train2017_person_only_no_crowd.json', 'train2017')
    ]
    
    found = False
    for json_file, dataset_type in datasets:
        json_path = os.path.join(annotations_dir, json_file)
        if os.path.exists(json_path):
            print(f"\nChecking {dataset_type} dataset...")
            output_path = f"annotations_000000001000_{dataset_type}.json"
            
            if extract_image_annotations(json_path, target_image, output_path):
                found = True
                break
    
    if not found:
        print(f"\nImage '{target_image}' was not found in any dataset")

if __name__ == "__main__":
    main()