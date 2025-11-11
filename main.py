#!/usr/bin/env python3
"""
Main CLI for Cross-Modal Bench Scene Graph Generation.

Usage:
    python main.py --image path/to/image.jpg
    python main.py --image path/to/image.jpg --model fcclip
    python main.py --image path/to/image.jpg --output scene_graph.json
"""

import argparse
import sys
from pathlib import Path
import yaml
import os

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from scene_graph import SceneGraphGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'OPENAI_API_KEY' in os.environ:
        if 'vlm' not in config:
            config['vlm'] = {}
        config['vlm']['gpt4o_api_key'] = os.environ['OPENAI_API_KEY']
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Generate scene graph from image using panoptic segmentation + VLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --image data/input/photo.jpg
  
  # Use FC-CLIP instead of kMaX
  python main.py --image photo.jpg --model fcclip
  
  # Specify output path
  python main.py --image photo.jpg --output results/scene_graph.json
  
  # Add text caption for context
  python main.py --image photo.jpg --caption "A busy street scene"
  
  # Use different config
  python main.py --image photo.jpg --config configs/my_config.yaml
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to input image'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file (default: configs/config.yaml)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save output scene graph JSON (default: auto-generated in output_dir)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['kmax', 'fcclip'],
        default=None,
        help='Override segmentation model (kmax or fcclip)'
    )
    
    parser.add_argument(
        '--caption',
        type=str,
        default=None,
        help='Optional text caption for context'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input image
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print(f"Please create a config file or use --config to specify a different path")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Override config with CLI arguments
    if args.model:
        config['segmentation']['model'] = args.model
    
    if args.verbose:
        config['verbose'] = True
    
    # Validate VLM configuration
    vlm_type = config.get('vlm', {}).get('type', 'vllm')
    
    if vlm_type == 'gpt4o':
        api_key = config.get('vlm', {}).get('gpt4o_api_key', '')
        if not api_key or api_key == 'your-api-key-here':
            print("Error: GPT-4o API key not configured!")
            print("Please either:")
            print("  1. Set OPENAI_API_KEY environment variable")
            print("  2. Update vlm.gpt4o_api_key in configs/config.yaml")
            print("  3. Change vlm.type to 'vllm' to use local model instead")
            sys.exit(1)
    elif vlm_type == 'vllm':
        pass
    else:
        print(f"Error: Unknown VLM type: {vlm_type}")
        print("Please set vlm.type to either 'vllm' or 'gpt4o' in config")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(config['data']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        image_name = Path(args.image).stem
        output_path = str(output_dir / f"{image_name}_scene_graph.json")
    
    # Create generator
    try:
        generator = SceneGraphGenerator.from_config(config)
    except Exception as e:
        print(f"Error initializing generator: {e}")
        print("\nMake sure:")
        print("  1. Segmentation model code is in external/ directory")
        print("  2. Model weights are downloaded")
        print("  3. Config paths are correct")
        if vlm_type == 'vllm':
            print("  4. VLLM is installed: pip install vllm")
            print("  5. Free GPUs available: export CUDA_VISIBLE_DEVICES=2")
        sys.exit(1)
    
    # Generate scene graph
    try:
        scene_graph = generator.generate(
            image_path=args.image,
            caption=args.caption,
            output_path=output_path
        )
        
        print(f"\n✓ Success! Scene graph saved to: {output_path}")
        print(f"\nScene Graph Summary:")
        print(f"  Objects: {len(scene_graph.nodes)}")
        print(f"  Relationships: {len(scene_graph.edges)}")
        
        # Print sample nodes
        if scene_graph.nodes and config.get('verbose', False):
            print(f"\n  Sample objects:")
            for node in scene_graph.nodes[:5]:
                attrs = ', '.join([f"{k}:{v}" for k, v in node.attributes.items()][:3])
                print(f"    - {node.object_class} ({attrs})")
        
    except Exception as e:
        print(f"\n✗ Error generating scene graph: {e}")
        import traceback
        if config.get('verbose', False):
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()