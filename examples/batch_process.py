#!/usr/bin/env python3
"""
Example script for batch processing multiple images.

Usage:
    python examples/batch_process.py --input_dir data/input --output_dir data/output
"""

import argparse
import sys
from pathlib import Path
import yaml
import os
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scene_graph import SceneGraphGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'OPENAI_API_KEY' in os.environ:
        config['vlm']['api_key'] = os.environ['OPENAI_API_KEY']
    
    return config


def get_image_files(input_dir: Path) -> list:
    """Get all image files from directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]


def main():
    parser = argparse.ArgumentParser(description='Batch process images to generate scene graphs')
    parser.add_argument('--input_dir', '-i', required=True, help='Input directory with images')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory for scene graphs')
    parser.add_argument('--config', '-c', default='configs/config.yaml', help='Config file')
    parser.add_argument('--model', '-m', choices=['kmax', 'fcclip'], help='Override model type')
    parser.add_argument('--skip_existing', action='store_true', help='Skip already processed images')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Get image files
    image_files = get_image_files(input_dir)
    print(f"Found {len(image_files)} images in {input_dir}")
    
    if not image_files:
        print("No images found!")
        sys.exit(1)
    
    # Load config
    config = load_config(args.config)
    if args.model:
        config['segmentation']['model'] = args.model
    
    # Create generator
    generator = SceneGraphGenerator.from_config(config)
    
    # Process images
    success_count = 0
    failed_images = []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        output_path = output_dir / f"{image_file.stem}_scene_graph.json"
        
        # Skip if exists
        if args.skip_existing and output_path.exists():
            print(f"Skipping {image_file.name} (already exists)")
            continue
        
        try:
            scene_graph = generator.generate(
                image_path=str(image_file),
                output_path=str(output_path)
            )
            success_count += 1
            print(f"✓ {image_file.name}: {len(scene_graph.nodes)} nodes, {len(scene_graph.edges)} edges")
            
        except Exception as e:
            print(f"✗ {image_file.name}: {str(e)}")
            failed_images.append(image_file.name)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_images)}")
    
    if failed_images:
        print(f"\nFailed images:")
        for img in failed_images:
            print(f"  - {img}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
