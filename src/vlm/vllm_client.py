"""
VLLM-based Vision Language Model client for scene graph extraction.
Alternative to GPT-4o that runs locally.
"""

import json
import re
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM not installed. Install with: pip install vllm")


class VLLMClient:
    """
    Client for using local VLLM models (Qwen2-VL, Qwen3-VL) for scene graph extraction.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        tensor_parallel_size: int = 4,
        max_tokens: int = 2000
    ):
        """
        Initialize VLLM client.
        
        Args:
            model_name: HuggingFace model name
            tensor_parallel_size: Number of GPUs to use
            max_tokens: Maximum tokens for response
        """
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM is required. Install with: pip install vllm")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Initialize model
        engine_args = {
            "model": model_name,
            "seed": 42,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": 4096,
            "max_num_seqs": 5,
            "dtype": "bfloat16",
            "limit_mm_per_prompt": {"image": 1},
            "mm_processor_cache_gb": 4,
            "trust_remote_code": True,
        }
        
        # Add model-specific kwargs
        if "Qwen2-VL" in model_name:
            engine_args["mm_processor_kwargs"] = {
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            }
        elif "Qwen3-VL" in model_name:
            engine_args["mm_processor_kwargs"] = {
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            }
        
        print(f"Loading VLLM model: {model_name}...")
        self.llm = LLM(**engine_args)
        print("✓ VLLM model loaded")
    
    def _build_prompt(self, caption: Optional[str] = None) -> str:
        """
        Build prompt for scene graph extraction.
        
        Args:
            caption: Optional caption for context
        
        Returns:
            Formatted prompt string
        """
        if caption:
            base_instruction = f"""Analyze this image with overlaid segmentation masks for the caption: "{caption}"

Generate a scene graph in JSON format."""
        else:
            base_instruction = """Analyze this image with overlaid segmentation masks.

Generate a scene graph in JSON format."""
        
        json_format = """{
  "objects": [
    {
      "id": "unique_id",
      "class": "object_class_name",
      "attributes": [
        {"name": "attribute_name", "value": "attribute_value"}
      ]
    }
  ],
  "relationships": [
    {
      "subject_id": "id_of_subject_object",
      "object_id": "id_of_object_object",
      "relation": "relationship_type"
    }
  ]
}"""
        
        full_prompt = f"""{base_instruction}

Output format:
{json_format}

IMPORTANT:
- Use the labeled segments visible in the image
- Provide ONLY the JSON object, no additional text
- Extract as many attributes and relationships as possible
- Ensure valid JSON format"""
        
        return full_prompt
    
    def _format_prompt_for_model(self, question: str) -> str:
        """
        Format prompt according to model template.
        
        Args:
            question: The question/instruction
        
        Returns:
            Formatted prompt with model-specific template
        """
        if "Qwen2-VL" in self.model_name or "Qwen3-VL" in self.model_name:
            placeholder = "<|image_pad|>"
            return (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            # Generic format
            return f"<image>\n{question}"
    
    def extract_scene_graph(
        self,
        visualization_path: str,
        caption: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract scene graph from visualization using VLLM.
        
        Args:
            visualization_path: Path to segmentation visualization
            caption: Optional text caption for context
            max_tokens: Override default max_tokens
        
        Returns:
            Dictionary containing scene graph structure
        """
        # Build prompt
        question = self._build_prompt(caption)
        full_prompt = self._format_prompt_for_model(question)
        
        try:
            image_data = Image.open(visualization_path)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
        
        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=max_tokens or self.max_tokens,
        )
        
        # Prepare inputs
        inputs = {
            "prompt": full_prompt,
            "multi_modal_data": {"image": image_data},
            "multi_modal_uuids": {"image": visualization_path},
        }
        
        # Run inference
        print(f"Running VLLM inference...")
        outputs = self.llm.generate([inputs], sampling_params=sampling_params)
        response_text = outputs[0].outputs[0].text.strip()
        
        # Parse response
        scene_graph_data = self._parse_response(response_text)
        
        return scene_graph_data
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse VLLM response and extract JSON.
        
        Args:
            response_text: Raw response from VLLM
        
        Returns:
            Parsed scene graph dictionary
        """
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        # Remove any leading/trailing whitespace
        response_text = response_text.strip()
        
        # Try to parse the JSON
        try:
            data = json.loads(response_text)
            return data
        except json.JSONDecodeError as e:
            cleaned_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
            
            try:
                data = json.loads(cleaned_text)
                return data
            except json.JSONDecodeError:
                raise ValueError(
                    f"Failed to parse VLLM response as JSON: {str(e)}\n\n"
                    f"Response: {response_text}"
                )
    
    def validate_scene_graph(self, data: Dict[str, Any]) -> bool:
        """
        Validate scene graph structure.
        
        Args:
            data: Scene graph dictionary
        
        Returns:
            True if valid
        """
        if "objects" not in data:
            return False
        
        if not isinstance(data["objects"], list):
            return False
        
        for obj in data["objects"]:
            if not isinstance(obj, dict):
                return False
            if "id" not in obj or "class" not in obj:
                return False
        
        if "relationships" in data:
            if not isinstance(data["relationships"], list):
                return False
            
            for rel in data["relationships"]:
                if not isinstance(rel, dict):
                    return False
                required = ["subject_id", "object_id", "relation"]
                if not all(k in rel for k in required):
                    return False
        
        return True