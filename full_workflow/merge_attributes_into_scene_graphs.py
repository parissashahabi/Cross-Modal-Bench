#!/usr/bin/env python3
"""
Merge attributes into scene graph JSON files.

This script takes attributes.json (from attribute generation) and merges them
into scene graph JSON files by matching image_id, label, and bbox.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List


def bbox_key(bbox: List[float], tolerance: int = 5) -> Tuple[int, int, int, int]:
    """Normalize bbox to int tuple for matching with tolerance.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2] or [x, y, w, h]
        tolerance: Tolerance in pixels for rounding (default: 5)
    
    Returns:
        Normalized bbox tuple
    """
    if len(bbox) == 4:
        # Convert [x, y, w, h] to [x1, y1, x2, y2] if needed
        if bbox[2] > bbox[0] + 1000:  # Likely x2, y2 format
            bbox_xyxy = bbox
        else:  # Likely x, y, w, h format
            bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    else:
        bbox_xyxy = bbox
    
    # Round with tolerance to handle small differences
    return tuple(int(round(x / tolerance)) * tolerance for x in bbox_xyxy)


def build_attr_index(attr_list: List[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Index attributes by (image_id, index).
    Uses index from scene graph for reliable matching.
    Normalizes image_id to int for consistent matching.
    """
    attr_index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for entry in attr_list:
        image_id = entry["image_id"]
        index = entry.get("index")
        
        if index is None:
            # Skip entries without index - they can't be matched
            print(f"Warning: Attribute entry missing index: image_id={image_id}, category={entry.get('category')}")
            continue
        
        # Normalize image_id to int
        if isinstance(image_id, str):
            try:
                image_id = int(image_id)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert image_id to int: {image_id}")
                continue
        
        key = (image_id, index)
        if key in attr_index:
            print(f"Warning: Duplicate key (image_id={image_id}, index={index}), overwriting")
        attr_index[key] = entry.get("attributes", {})
    return attr_index


def merge_attributes_into_scene_graph(scene_graph: Dict[str, Any], attr_index: Dict[Tuple[int, int], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge attributes into scene graph boxes by matching image_id and index.
    Uses index from scene graph for reliable matching.
    Handles both int and string image_id types.
    """
    image_id = scene_graph.get("image_id")
    # Normalize image_id to int for matching
    if isinstance(image_id, str):
        try:
            image_id = int(image_id)
        except (ValueError, TypeError):
            pass
    
    matched_count = 0
    for box in scene_graph.get("boxes", []):
        index = box.get("index")
        
        if index is None:
            # Skip if no index
            box["attributes"] = {}
            continue
        
        # Try both int and the original type for image_id
        key = (image_id, index)
        attrs = attr_index.get(key, {})
        
        # If not found and image_id is int, try as string
        if not attrs and isinstance(image_id, int):
            key_str = (str(image_id), index)
            attrs = attr_index.get(key_str, {})
        # If not found and image_id is string, try as int
        elif not attrs and isinstance(image_id, str):
            try:
                key_int = (int(image_id), index)
                attrs = attr_index.get(key_int, {})
            except (ValueError, TypeError):
                pass
        
        if attrs:
            matched_count += 1
        
        # Add attributes to box (even if empty)
        box["attributes"] = attrs
    
    return scene_graph


def main():
    parser = argparse.ArgumentParser(
        description="Merge attributes into scene graph JSON files"
    )
    parser.add_argument(
        "--attributes-json",
        type=str,
        required=True,
        help="Path to attributes.json file"
    )
    parser.add_argument(
        "--scene-graphs-dir",
        type=str,
        required=True,
        help="Directory containing scene-graph_*.json files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to scene-graphs-dir)"
    )
    
    args = parser.parse_args()
    
    attrs_path = Path(args.attributes_json)
    sg_dir = Path(args.scene_graphs_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sg_dir
    
    if not attrs_path.exists():
        print(f"Error: Attributes file not found: {attrs_path}")
        return 1
    
    if not sg_dir.exists():
        print(f"Error: Scene graphs directory not found: {sg_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load attributes
    print(f"Loading attributes from: {attrs_path}")
    with open(attrs_path, 'r') as f:
        attr_list = json.load(f)
    
    if not attr_list:
        print("Warning: No attributes found in attributes.json")
        return 0
    
    print(f"Loaded {len(attr_list)} attribute entries")
    
    # Build attribute index
    print("Building attribute index...")
    attr_index = build_attr_index(attr_list)
    print(f"Indexed {len(attr_index)} attributes")
    
    # Debug: show some example keys
    if attr_index:
        example_keys = list(attr_index.keys())[:5]
        print(f"Example index keys: {example_keys}")
        for key in example_keys:
            print(f"  Key {key}: {len(attr_index[key])} attribute types")
    
    # Find all scene graph files
    sg_files = list(sg_dir.glob("scene-graph_*.json"))
    if not sg_files:
        print(f"Warning: No scene graph files found in {sg_dir}")
        return 0
    
    print(f"\nProcessing {len(sg_files)} scene graph files...")
    
    merged_count = 0
    for sg_file in sg_files:
        # Load scene graph
        with open(sg_file, 'r') as f:
            scene_graph = json.load(f)
        
        # Merge attributes
        merged_sg = merge_attributes_into_scene_graph(scene_graph, attr_index)
        
        # Count how many boxes got attributes
        boxes_with_attrs = sum(1 for box in merged_sg.get("boxes", []) if box.get("attributes"))
        total_boxes = len(merged_sg.get("boxes", []))
        
        # Save merged scene graph
        output_file = output_dir / sg_file.name
        with open(output_file, 'w') as f:
            json.dump(merged_sg, f, indent=2)
        
        merged_count += 1
        print(f"  {sg_file.name}: {boxes_with_attrs}/{total_boxes} boxes with attributes")
        
        # Debug: show which boxes didn't get attributes and why
        if boxes_with_attrs < total_boxes:
            unmatched = []
            image_id = merged_sg.get("image_id")
            if isinstance(image_id, str):
                try:
                    image_id = int(image_id)
                except (ValueError, TypeError):
                    pass
            
            for box in merged_sg.get("boxes", []):
                if not box.get("attributes"):
                    index = box.get("index")
                    key = (image_id, index) if index is not None else None
                    in_index = key in attr_index if key else False
                    unmatched.append(f"index={index}, label={box.get('label')}, key_in_index={in_index}")
            if unmatched:
                print(f"    Unmatched boxes: {', '.join(unmatched[:5])}{'...' if len(unmatched) > 5 else ''}")
    
    print(f"\n✓ Merged attributes into {merged_count} scene graph files")
    print(f"✓ Output saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())

