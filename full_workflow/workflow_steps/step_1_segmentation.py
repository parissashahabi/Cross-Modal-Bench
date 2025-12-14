#!/usr/bin/env python3
"""
Step 1: Segmentation

Generate segmentation masks using either Mask2Former or kMaX-DeepLab.
"""

import shutil
from pathlib import Path

from .base_step import BaseStep


class Step1Segmentation(BaseStep):
    """Step 1: Generate segmentation masks."""
    
    def execute(self) -> bool:
        """
        Step 1: Generate segmentation masks.
        
        Uses either Mask2Former or kMaX-DeepLab based on configuration.
        """
        print("\n" + "="*70)
        print("STEP 1: SEGMENTATION")
        print("="*70)
        
        cfg = self.config['segmentation']
        
        if cfg.get('skip', False):
            print("Skipping segmentation (using existing anno.json)")
            if 'anno_path' in cfg:
                src = Path(cfg['anno_path'])
                dst = self.dirs['segmentation'] / 'anno.json'
                if src.exists():
                    shutil.copy(src, dst)
                    print(f"Copied existing anno.json from: {src}")
                else:
                    print(f"ERROR: anno.json not found at {src}")
                    return False
            return True
        
        if cfg.get('method') == 'kmax':
            kmax_env = cfg.get('kmax_env')
            
            sg_cfg = self.config.get('scene_graph_generation', {})
            img_dir_gt = sg_cfg.get('img_dir_gt')
            img_dir_pt = sg_cfg.get('img_dir_pt')
            
            if not img_dir_gt or not img_dir_pt:
                base_img_dir = Path(cfg['img_dir'])
                img_dir_gt = str(base_img_dir / 'gt')
                img_dir_pt = str(base_img_dir / 'pt')
            
            process_mode = self.get_process_mode()
            print(f"Process mode: {process_mode}")
            
            if self.should_process_gt():
                print(f"Processing GT images from: {img_dir_gt}")
            if self.should_process_pt():
                print(f"Processing PT images from: {img_dir_pt}")
            
            success = True
            
            if self.should_process_gt():
                out_dir_gt = self.dirs['segmentation_gt']
                if kmax_env:
                    cmd_gt = [
                        'conda', 'run', '-n', kmax_env, '--no-capture-output',
                        'python', cfg['kmax_script'],
                        '--img-dir', str(img_dir_gt),
                        '--out-dir', str(out_dir_gt),
                        '--psg-meta', cfg['psg_meta'],
                        '--kmax-path', cfg['kmax_path'],
                        '--kmax-config', cfg['kmax_config'],
                        '--kmax-weights', cfg['kmax_weights']
                    ]
                    if 'max_images' in cfg:
                        cmd_gt.extend(['--max-images', str(cfg['max_images'])])
                else:
                    cmd_gt = [
                        'python', cfg['kmax_script'],
                        '--img-dir', str(img_dir_gt),
                        '--out-dir', str(out_dir_gt),
                        '--psg-meta', cfg['psg_meta'],
                        '--kmax-path', cfg['kmax_path'],
                        '--kmax-config', cfg['kmax_config'],
                        '--kmax-weights', cfg['kmax_weights']
                    ]
                    if 'max_images' in cfg:
                        cmd_gt.extend(['--max-images', str(cfg['max_images'])])
                
                if self.run_command(cmd_gt, 'segmentation_kmax_gt') != 0:
                    success = False
            
            if self.should_process_pt():
                out_dir_pt = self.dirs['segmentation_pt']
                if kmax_env:
                    cmd_pt = [
                        'conda', 'run', '-n', kmax_env, '--no-capture-output',
                        'python', cfg['kmax_script'],
                        '--img-dir', str(img_dir_pt),
                        '--out-dir', str(out_dir_pt),
                        '--psg-meta', cfg['psg_meta'],
                        '--kmax-path', cfg['kmax_path'],
                        '--kmax-config', cfg['kmax_config'],
                        '--kmax-weights', cfg['kmax_weights']
                    ]
                    if 'max_images' in cfg:
                        cmd_pt.extend(['--max-images', str(cfg['max_images'])])
                else:
                    cmd_pt = [
                        'python', cfg['kmax_script'],
                        '--img-dir', str(img_dir_pt),
                        '--out-dir', str(out_dir_pt),
                        '--psg-meta', cfg['psg_meta'],
                        '--kmax-path', cfg['kmax_path'],
                        '--kmax-config', cfg['kmax_config'],
                        '--kmax-weights', cfg['kmax_weights']
                    ]
                    if 'max_images' in cfg:
                        cmd_pt.extend(['--max-images', str(cfg['max_images'])])
                
                if self.run_command(cmd_pt, 'segmentation_kmax_pt') != 0:
                    success = False
                
            return success
        else:
            print("Using Mask2Former (segmentation will run in scene graph generation)")
            return True


