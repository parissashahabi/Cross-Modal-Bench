"""
Main scene graph generator that orchestrates the full pipeline.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scene_graph.schema import SceneGraph, Node, Edge
from segmentation import BaseSegmentor, KMaxSegmentor, FCCLIPSegmentor
from vlm import GPT4oClient, VLLMClient


class SceneGraphGenerator:
    """
    Main orchestrator for scene graph generation.
    
    Pipeline:
    1. Load and run panoptic segmentation model
    2. Get visualization with labeled masks
    3. Send to VLM for scene graph extraction
    4. Parse and structure the scene graph
    5. Save as JSON
    """
    
    def __init__(
        self,
        segmentor: BaseSegmentor,
        vlm_client: Union[GPT4oClient, VLLMClient],
        verbose: bool = True
    ):
        """
        Initialize the scene graph generator.
        
        Args:
            segmentor: Segmentation model (kMaX or FC-CLIP)
            vlm_client: VLM client (GPT-4o or VLLM)
            verbose: Print progress messages
        """
        self.segmentor = segmentor
        self.vlm_client = vlm_client
        self.verbose = verbose
    
    def generate(
        self,
        image_path: str,
        caption: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> SceneGraph:
        """
        Generate scene graph from an image.
        
        Args:
            image_path: Path to input image
            caption: Optional text caption for context
            output_path: Optional path to save scene graph JSON
        
        Returns:
            SceneGraph object
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Generating Scene Graph")
            print(f"{'='*60}")
            print(f"Image: {image_path}")
            if caption:
                print(f"Caption: {caption}")
        
        start_time = time.time()
        
        # Step 1: Run segmentation
        if self.verbose:
            vlm_name = self.vlm_client.__class__.__name__
            print(f"\n[1/3] Running panoptic segmentation...")
        
        masks, labels, vis_path = self.segmentor.segment(image_path)
        
        # Step 2: Extract scene graph using VLM
        if self.verbose:
            print(f"\n[2/3] Extracting scene graph with {vlm_name}...")
        
        raw_scene_graph = self.vlm_client.extract_scene_graph(
            visualization_path=vis_path,
            caption=caption
        )
        
        # Step 3: Convert to SceneGraph object
        if self.verbose:
            print(f"\n[3/3] Structuring scene graph...")
        
        scene_graph = self._build_scene_graph(raw_scene_graph, image_path, vis_path)
        
        # Save if output path provided
        if output_path:
            scene_graph.save(output_path)
            if self.verbose:
                print(f"✓ Scene graph saved to: {output_path}")
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✓ Scene Graph Generation Complete")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Nodes: {len(scene_graph.nodes)}")
            print(f"  Edges: {len(scene_graph.edges)}")
            print(f"{'='*60}\n")
        
        return scene_graph
    
    def _build_scene_graph(
        self,
        raw_data: Dict[str, Any],
        image_path: str,
        vis_path: str
    ) -> SceneGraph:
        """
        Convert raw VLM output to structured SceneGraph.
        
        Args:
            raw_data: Raw dictionary from VLM
            image_path: Original image path
            vis_path: Visualization path
        
        Returns:
            Structured SceneGraph object
        """
        scene_graph = SceneGraph()
        
        # Add metadata
        scene_graph.metadata = {
            "image_path": image_path,
            "visualization_path": vis_path,
            "model": self.segmentor.__class__.__name__
        }
        
        # Process objects/nodes
        objects = raw_data.get("objects", [])
        for obj in objects:
            # Extract attributes
            attributes = {}
            if "attributes" in obj:
                if isinstance(obj["attributes"], list):
                    # List format: [{"name": "x", "value": "y"}]
                    for attr in obj["attributes"]:
                        if isinstance(attr, dict) and "name" in attr and "value" in attr:
                            attributes[attr["name"]] = attr["value"]
                elif isinstance(obj["attributes"], dict):
                    # Dict format: {"color": "red", "size": "large"}
                    attributes = obj["attributes"]
            
            node = Node(
                id=str(obj.get("id", "")),
                object_class=obj.get("class", obj.get("object", "")),
                description=obj.get("description"),
                attributes=attributes
            )
            scene_graph.add_node(node)
        
        # Process relationships/edges
        relationships = raw_data.get("relationships", [])
        for rel in relationships:
            edge = Edge(
                subject_id=str(rel.get("subject_id", rel.get("subject", ""))),
                object_id=str(rel.get("object_id", rel.get("object", ""))),
                relation=rel.get("relation", rel.get("relationship", ""))
            )
            scene_graph.add_edge(edge)
        
        return scene_graph
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SceneGraphGenerator':
        """
        Create SceneGraphGenerator from unified configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Initialized SceneGraphGenerator
        """
        seg_config = config['segmentation']
        model_type = seg_config['model']
        
        # Build segmentor config based on model type
        if model_type == 'kmax':
            # Map unified config keys to what KMaxSegmentor expects
            segmentor_config = {
                'kmax_path': seg_config['kmax_path'],
                'config_file': seg_config['kmax_config_file'],  # Map kmax_config_file -> config_file
                'weights': seg_config['kmax_weights'],          # Map kmax_weights -> weights
                'output_dir': seg_config['output_dir'],
                'use_cuda': seg_config['use_cuda']
            }
            segmentor = KMaxSegmentor(segmentor_config)
            
        elif model_type == 'fcclip':
            # Map unified config keys to what FCCLIPSegmentor expects
            segmentor_config = {
                'fcclip_path': seg_config['fcclip_path'],
                'config_file': seg_config['fcclip_config_file'],  # Map fcclip_config_file -> config_file
                'weights': seg_config['fcclip_weights'],          # Map fcclip_weights -> weights
                'output_dir': seg_config['output_dir'],
                'use_cuda': seg_config['use_cuda']
            }
            segmentor = FCCLIPSegmentor(segmentor_config)
        else:
            raise ValueError(f"Unknown segmentation model: {model_type}. Must be 'kmax' or 'fcclip'")
        
        # Initialize VLM client
        vlm_config = config['vlm']
        vlm_type = vlm_config.get('type', 'vllm')
        
        if vlm_type == 'vllm':
            vlm_client = VLLMClient(
                model_name=vlm_config.get('vllm_model', 'Qwen/Qwen2-VL-7B-Instruct'),
                tensor_parallel_size=vlm_config.get('tensor_parallel_size', 1),
                max_tokens=vlm_config.get('max_tokens', 2000)
            )
        elif vlm_type == 'gpt4o':
            vlm_client = GPT4oClient(
                api_key=vlm_config.get('gpt4o_api_key'),
                model=vlm_config.get('gpt4o_model', 'gpt-4o')
            )
        else:
            raise ValueError(f"Unknown VLM type: {vlm_type}. Must be 'vllm' or 'gpt4o'")
        
        # Create generator
        return cls(
            segmentor=segmentor,
            vlm_client=vlm_client,
            verbose=config.get('verbose', True)
        )