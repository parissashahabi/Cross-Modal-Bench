#!/usr/bin/env python3
"""
Step 2: Scene Graph Generation

Generate scene graphs with relationship predictions using DSFormer.
Supports GT/PT split processing.
"""

from pathlib import Path
from .base_step import BaseStep


class Step2SceneGraphGeneration(BaseStep):
    """Step 2: Generate scene graphs with relationship predictions."""
    
    def execute(self) -> bool:
        """
        Step 2: Generate scene graphs with relationship predictions.
        
        Uses DSFormer for relationship prediction.
        Supports GT/PT split processing.
        """
        print("\n" + "="*70)
        print("STEP 2: SCENE GRAPH GENERATION")
        print("="*70)
        
        cfg = self.config['scene_graph_generation']
        seg_cfg = self.config['segmentation']
        
        process_gt_pt_separately = cfg.get('process_separately', False)
        
        if process_gt_pt_separately:
            process_mode = self.get_process_mode()
            print(f"Process mode: {process_mode}")
            print("Processing GT and PT directories separately...")
            
            gt_success = True
            pt_success = True
            
            if self.should_process_gt():
                gt_success = self._process_scene_graph_generation(
                    cfg,
                    seg_cfg,
                    cfg.get('img_dir_gt', cfg.get('img_dir')),
                    self.dirs['scene_graphs_gt'],
                    "GT"
                )
            
            if self.should_process_pt():
                pt_success = self._process_scene_graph_generation(
                    cfg,
                    seg_cfg,
                    cfg.get('img_dir_pt', cfg.get('img_dir')),
                    self.dirs['scene_graphs_pt'],
                    "PT"
                )
            
            return gt_success and pt_success
        else:
            raise ValueError(
                "Single directory mode is no longer supported. "
                "Please set 'process_separately': true in scene_graph_generation config."
            )
    
    def _process_scene_graph_generation(self, cfg: dict, seg_cfg: dict, img_dir: str, out_dir: Path, label: str) -> bool:
        """Helper to process scene graph generation for a directory."""
        sg_env = cfg.get('conda_env')
        
        if sg_env:
            cmd = [
                'conda', 'run', '-n', sg_env, '--no-capture-output',
                'python', cfg['script'],
                '--img-dir', str(img_dir),
                '--out-dir', str(out_dir),
                '--psg-meta', cfg['psg_meta'],
                '--top-k', str(cfg.get('top_k', 15))
            ]
        else:
            cmd = [
                'python', cfg['script'],
                '--img-dir', str(img_dir),
                '--out-dir', str(out_dir),
                '--psg-meta', cfg['psg_meta'],
                '--top-k', str(cfg.get('top_k', 15))
            ]
        
        if cfg.get('model_dir'):
            cmd.extend(['--model-dir', cfg['model_dir']])
        else:
            cmd.append('--skip-inference')
        
        if seg_cfg.get('skip', False) or seg_cfg.get('method') == 'kmax':
            anno_path = out_dir / 'anno.json'
            if not anno_path.exists():
                if label == 'GT':
                    anno_path = self.dirs['segmentation_gt'] / 'anno.json'
                elif label == 'PT':
                    anno_path = self.dirs['segmentation_pt'] / 'anno.json'
                else:
                    anno_path = self.dirs['segmentation'] / 'anno.json'
            
            if anno_path.exists():
                cmd.append('--skip-segmentation')
                cmd.extend(['--anno-path', str(anno_path)])
            else:
                print(f"Warning: anno.json not found at {anno_path}")
        else:
            cmd.extend([
                '--seg-model', seg_cfg.get('seg_model', 'facebook/mask2former-swin-large-coco-panoptic'),
                '--workers', str(seg_cfg.get('workers', 2))
            ])
        
        return self.run_command(cmd, f'scene_graph_generation_{label}') == 0


