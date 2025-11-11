"""
GPT-4o Vision Language Model client for scene graph extraction.
"""

import base64
import json
import re
from typing import Dict, Any, Optional
from pathlib import Path

from openai import OpenAI


class GPT4oClient:
    """
    Client for interacting with GPT-4o to extract scene graphs from images.
    
    Following the paper's approach:
    - Input: Image with overlaid segmentation masks
    - Output: Scene graph in JSON format
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the GPT-4o client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _build_prompt(self, caption: Optional[str] = None) -> str:
        """
        Build the prompt for GPT-4o following the paper's specification.
        
        The paper's prompt format:
        "Could you please generate the scene graph for this caption "[caption]" 
        by considering the overlaid mask image? The output should be formatted 
        as follow: 'objects':..., attributes:name:xx, value:xx, name:xx, value:xx, 
        'relationship':xx:relationship:[xx]."
        
        Args:
            caption: Optional caption describing the image
        
        Returns:
            Formatted prompt string
        """
        if caption:
            prompt = f"""Could you please generate the scene graph for this caption "{caption}" by considering the overlaid mask image?

The output should be formatted as a valid JSON object with the following structure:
{{
  "objects": [
    {{
      "id": "unique_id",
      "class": "object_class_name",
      "attributes": [
        {{"name": "attribute_name", "value": "attribute_value"}}
      ]
    }}
  ],
  "relationships": [
    {{
      "subject_id": "id_of_subject_object",
      "object_id": "id_of_object_object",
      "relation": "relationship_type"
    }}
  ]
}}

IMPORTANT: 
- Use the labeled segments visible in the image
- Provide ONLY the JSON object, no additional text
- Ensure the JSON is valid and properly formatted
- Extract as many attributes and relationships as you can identify"""
        else:
            prompt = """Please analyze this image with overlaid segmentation masks and generate a scene graph.

The output should be formatted as a valid JSON object with the following structure:
{
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
}

IMPORTANT:
- Use the labeled segments visible in the image
- Provide ONLY the JSON object, no additional text
- Ensure the JSON is valid and properly formatted
- Extract as many attributes and relationships as you can identify"""
        
        return prompt
    
    def extract_scene_graph(
        self,
        visualization_path: str,
        caption: Optional[str] = None,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Extract scene graph from a visualization image using GPT-4o.
        
        Args:
            visualization_path: Path to the segmentation visualization
            caption: Optional text caption for context
            max_tokens: Maximum tokens for the response
        
        Returns:
            Dictionary containing the scene graph structure
        
        Raises:
            ValueError: If GPT-4o returns invalid JSON
            Exception: If API call fails
        """
        # Encode the image
        image_base64 = self._encode_image(visualization_path)
        
        # Build the prompt
        prompt = self._build_prompt(caption)
        
        # Create the API request
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Parse the JSON response
            scene_graph_data = self._parse_response(response_text)
            
            return scene_graph_data
            
        except Exception as e:
            raise Exception(f"GPT-4o API call failed: {str(e)}")
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse GPT-4o response and extract JSON.
        
        The paper mentions ~5% format errors. This method handles:
        - Extracting JSON from markdown code blocks
        - Cleaning up malformed JSON
        
        Args:
            response_text: Raw response from GPT-4o
        
        Returns:
            Parsed scene graph dictionary
        
        Raises:
            ValueError: If JSON cannot be parsed
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
            # Remove trailing commas
            cleaned_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
            
            try:
                data = json.loads(cleaned_text)
                return data
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse GPT-4o response as JSON: {str(e)}\n\nResponse: {response_text}")
    
    def validate_scene_graph(self, data: Dict[str, Any]) -> bool:
        """
        Validate that the scene graph has the expected structure.
        
        Args:
            data: Scene graph dictionary
        
        Returns:
            True if valid, False otherwise
        """
        required_keys = ["objects"]
        for key in required_keys:
            if key not in data:
                return False
        
        # Validate objects structure
        if not isinstance(data["objects"], list):
            return False
        
        for obj in data["objects"]:
            if not isinstance(obj, dict):
                return False
            if "id" not in obj or "class" not in obj:
                return False
        
        # Validate relationships if present
        if "relationships" in data:
            if not isinstance(data["relationships"], list):
                return False
            
            for rel in data["relationships"]:
                if not isinstance(rel, dict):
                    return False
                if "subject_id" not in rel or "object_id" not in rel or "relation" not in rel:
                    return False
        
        return True
