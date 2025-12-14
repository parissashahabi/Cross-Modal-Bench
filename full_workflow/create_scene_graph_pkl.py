#!/usr/bin/env python3
"""
Create scene-graph.pkl from segmentation masks for description generation.

This script creates a minimal scene graph pickle file from segmentation data
when full scene graph inference is not available.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def create_scene_graph_from_segmentation(scene_graphs_dir: Path, output_pkl: Path):
    """
    Create a scene-graph.pkl file from segmentation and annotations.
    
    Args:
        scene_graphs_dir: Directory containing anno.json and segmentation masks
        output_pkl: Path to output pickle file
    """
    # Load annotation data
    anno_path = scene_graphs_dir / "anno.json"
    if not anno_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_path}")
    
    with open(anno_path, "r") as f:
        anno_data = json.load(f)
    
    pkl_data = []
    json_data_dict = {}
    
    thing_classes = anno_data["thing_classes"]
    stuff_classes = anno_data["stuff_classes"]
    all_classes = thing_classes + stuff_classes
    
    for entry in anno_data["data"]:
        image_id = entry["image_id"]
        seg_file = entry["pan_seg_file_name"]
        seg_path = scene_graphs_dir / seg_file
        
        if not seg_path.exists():
            print(f"Warning: Segmentation mask not found: {seg_path}")
            continue
        
        # Load segmentation mask and convert to label map
        seg_img = Image.open(seg_path)
        seg_array = np.array(seg_img)
        
        # Convert RGB back to IDs
        if len(seg_array.shape) == 3:
            # RGB format - convert back to ID
            id_map = (seg_array[:, :, 0].astype(np.uint32) + 
                     (seg_array[:, :, 1].astype(np.uint32) << 8) + 
                     (seg_array[:, :, 2].astype(np.uint32) << 16))
        else:
            id_map = seg_array
        
        # Create box index mapping (simple sequential mapping)
        unique_ids = np.unique(id_map)
        unique_ids = unique_ids[unique_ids != 0]  # Remove background
        
        # Create mask with box indices (1-indexed)
        mask = np.zeros_like(id_map, dtype=np.int32)
        boxes = []
        
        for idx, (seg_info, anno_box) in enumerate(zip(entry["segments_info"], entry["annotations"]), start=1):
            seg_id = seg_info["id"]
            category_id = seg_info["category_id"]
            
            # Mark pixels with this box index
            mask[id_map == seg_id] = idx
            
            # Get bounding box
            bbox_xywh = anno_box["bbox"]
            bbox_xyxy = [
                bbox_xywh[0],
                bbox_xywh[1],
                bbox_xywh[0] + bbox_xywh[2],
                bbox_xywh[1] + bbox_xywh[3]
            ]
            
            # Get label name
            label_name = all_classes[category_id] if category_id < len(all_classes) else f"class_{category_id}"
            
            boxes.append({
                "index": idx,
                "bbox_xyxy": bbox_xyxy,
                "bbox_xywh": bbox_xywh,
                "label": label_name,
                "category_id": category_id,
                "score": seg_info.get("score", 1.0)
            })
        
        # Add to pickle data
        pkl_data.append({
            "img_id": image_id,
            "mask": mask,
        })
        
        # Create JSON entry for scene graph
        json_entry = {
            "image_id": image_id,
            "file_name": entry["file_name"],
            "height": entry["height"],
            "width": entry["width"],
            "boxes": boxes,
            "relationships": []  # No relationships without inference
        }
        
        # Save individual JSON file
        json_path = scene_graphs_dir / f"scene-graph_{image_id}.json"
        with open(json_path, "w") as f:
            json.dump(json_entry, f, indent=2)
        
        json_data_dict[str(image_id)] = json_entry
        print(f"Processed image {image_id}: {len(boxes)} objects")
    
    # Save pickle file
    with open(output_pkl, "wb") as f:
        pickle.dump(pkl_data, f)
    
    print(f"\nCreated scene graph pickle: {output_pkl}")
    print(f"Total images: {len(pkl_data)}")
    print(f"Total objects: {sum(len(e['boxes']) for e in json_data_dict.values())}")


def main():
    parser = argparse.ArgumentParser(description="Create scene-graph.pkl from segmentation data")
    parser.add_argument("--scene-graphs-dir", type=str, required=True,
                       help="Directory containing anno.json and segmentation masks")
    parser.add_argument("--output", type=str, default=None,
                       help="Output pickle file path (default: scene-graphs-dir/scene-graph.pkl)")
    
    args = parser.parse_args()
    
    scene_graphs_dir = Path(args.scene_graphs_dir)
    output_pkl = Path(args.output) if args.output else scene_graphs_dir / "scene-graph.pkl"
    
    create_scene_graph_from_segmentation(scene_graphs_dir, output_pkl)
    return 0


if __name__ == "__main__":
    exit(main())
