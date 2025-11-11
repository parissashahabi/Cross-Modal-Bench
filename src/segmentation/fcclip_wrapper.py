"""
Wrapper for FC-CLIP open-vocabulary segmentation model.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.base import BaseSegmentor


class FCCLIPSegmentor(BaseSegmentor):
    """
    Wrapper for FC-CLIP model.
    
    This assumes FC-CLIP is installed in external/fc-clip/
    and follows the Detectron2-based inference API similar to kMaX.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FC-CLIP segmentor.
        
        Args:
            config: Configuration dict with keys:
                - fcclip_path: Path to FC-CLIP repository
                - config_file: Path to model config file
                - weights: Path to model weights
                - output_dir: Directory to save visualizations
        """
        super().__init__(config)
        self.fcclip_path = Path(config['fcclip_path'])
        self.config_file = config['config_file']
        self.weights = config['weights']
        self.output_dir = Path(config.get('output_dir', './data/output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if str(self.fcclip_path) not in sys.path:
            sys.path.insert(0, str(self.fcclip_path))
    
    def load_model(self) -> None:
        """
        Load FC-CLIP model using Detectron2.
        
        This method initializes the model following the FC-CLIP
        inference pattern.
        """
        try:
            # Import detectron2 and fc-clip components
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.projects.deeplab import add_deeplab_config
            
            # Try to import FC-CLIP config
            try:
                from fcclip import add_maskformer2_config, add_fcclip_config
            except ImportError:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "fcclip_config",
                    self.fcclip_path / "fcclip" / "config.py"
                )
                fcclip_config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(fcclip_config_module)
                add_maskformer2_config = fcclip_config_module.add_maskformer2_config
                add_fcclip_config = fcclip_config_module.add_fcclip_config
            
            # Setup config
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_maskformer2_config(cfg)
            add_fcclip_config(cfg)
            cfg.merge_from_file(self.config_file)
            cfg.MODEL.WEIGHTS = self.weights
            cfg.MODEL.DEVICE = 'cuda' if self.config.get('use_cuda', True) else 'cpu'
            
            # Create predictor
            self.model = DefaultPredictor(cfg)
            self.cfg = cfg
            
            print(f"✓ FC-CLIP model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import FC-CLIP dependencies. "
                f"Make sure detectron2 and FC-CLIP are properly installed.\n"
                f"Error: {str(e)}"
            )
    
    def segment(self, image_path: str) -> Tuple[np.ndarray, List[str], str]:
        """
        Run panoptic segmentation using FC-CLIP.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Tuple of (masks, labels, visualization_path)
        """
        if self.model is None:
            self.load_model()
        
        self.validate_image(image_path)
        
        import cv2
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        
        image = cv2.imread(image_path)
        
        # Run inference
        outputs = self.model(image)
        
        # Extract panoptic segmentation
        panoptic_seg = outputs["panoptic_seg"]
        segments_info = panoptic_seg[1]
        masks = panoptic_seg[0].cpu().numpy()
        
        # Get metadata for class names
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        
        # Extract labels for each segment
        labels = []
        for segment in segments_info:
            category_id = segment['category_id']
            # FC-CLIP uses open-vocabulary, so it might have different label handling
            if hasattr(metadata, 'stuff_classes') and category_id < len(metadata.stuff_classes):
                label = metadata.stuff_classes[category_id]
            elif hasattr(metadata, 'thing_classes'):
                thing_idx = category_id - len(metadata.stuff_classes) if hasattr(metadata, 'stuff_classes') else category_id
                label = metadata.thing_classes[thing_idx]
            else:
                label = f"object_{category_id}"
            labels.append(label)
        
        # Create visualization
        visualizer = Visualizer(image[:, :, ::-1], metadata, scale=1.0)
        vis_output = visualizer.draw_panoptic_seg(panoptic_seg[0], segments_info)
        
        # Save visualization
        vis_image = vis_output.get_image()
        output_filename = Path(image_path).stem + "_fcclip_seg.jpg"
        vis_path = str(self.output_dir / output_filename)
        cv2.imwrite(vis_path, vis_image[:, :, ::-1])
        
        print(f"✓ Segmentation complete: {len(labels)} segments detected")
        print(f"✓ Visualization saved to: {vis_path}")
        
        return masks, labels, vis_path
