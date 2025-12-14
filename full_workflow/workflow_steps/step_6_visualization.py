#!/usr/bin/env python3
"""
Step 6: Visualization

Generate visualizations of matched graphs.
Creates side-by-side comparisons with masks.
"""

from .base_step import BaseStep


class Step6Visualization(BaseStep):
    """Step 6: Generate visualizations of matched graphs."""
    
    def execute(self) -> bool:
        """
        Step 6: Generate visualizations of matched graphs.
        
        Creates side-by-side comparisons with masks.
        """
        print("\n" + "="*70)
        print("STEP 6: VISUALIZATION")
        print("="*70)
        
        cfg = self.config['visualization']
        
        if cfg.get('skip', False):
            print("Skipping visualization")
            return True
        
        process_mode = self.get_process_mode()
        if process_mode != 'both':
            print(f"Warning: Visualization typically requires both GT and PT, but process_mode is '{process_mode}'")
            print("Skipping visualization. Use process_mode='both' to enable visualization.")
            return True
        
        scene_graphs_json = self.dirs['matching'] / 'scene_graphs_for_matching.json'
        matching_results = self.dirs['matching'] / 'scene_graphs_matched.json'
        
        if not scene_graphs_json.exists() or not matching_results.exists():
            print("WARNING: Missing required files for visualization")
            print(f"  Scene graphs: {scene_graphs_json}")
            print(f"  Matching results: {matching_results}")
            return False
        
        vis_env = cfg.get('conda_env', 'graph-matching')
        
        cmd = [
            'conda', 'run', '-n', vis_env, '--no-capture-output',
            'python', cfg['script'],
            '--scene-graphs', str(scene_graphs_json),
            '--matching-results', str(matching_results),
            '--psg-json', cfg['psg_json'],
            '--coco-images', cfg['coco_images'],
            '--output-dir', str(self.dirs['visualizations'])
        ]
        
        if cfg.get('pred_masks'):
            cmd.extend(['--pred-masks', cfg['pred_masks']])
        
        if cfg.get('panoptic_dir'):
            cmd.extend(['--panoptic-dir', cfg['panoptic_dir']])
        
        return self.run_command(cmd, 'visualization') == 0


