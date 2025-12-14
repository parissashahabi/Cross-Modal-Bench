#!/usr/bin/env python3
"""
Step 5: Graph Matching

Run graph matching using Hungarian algorithm.
Matches predicted scene graphs with ground truth.
"""

from pathlib import Path

from .base_step import BaseStep


class Step5GraphMatching(BaseStep):
    """Step 5: Run graph matching using Hungarian algorithm."""
    
    def execute(self) -> bool:
        """
        Step 5: Run graph matching using Hungarian algorithm.
        
        Matches predicted scene graphs with ground truth.
        Supports matching between gt and pt directories by image ID.
        """
        print("\n" + "="*70)
        print("STEP 5: GRAPH MATCHING")
        print("="*70)
        
        cfg = self.config['graph_matching']
        
        if cfg.get('skip', False):
            print("Skipping graph matching")
            return True
        
        process_mode = self.get_process_mode()
        if process_mode != 'both':
            print(f"Warning: Graph matching requires both GT and PT, but process_mode is '{process_mode}'")
            print("Skipping graph matching. Use process_mode='both' to enable matching.")
            return True
        
        matching_input = self.dirs['matching'] / 'scene_graphs_for_matching.json'
        
        print("Converting scene graphs to matching format...")
        converter_script = Path(__file__).parent.parent / 'convert_to_matching_format.py'
        
        gt_sg_dir = cfg.get('gt_scene_graphs_dir', self.dirs.get('scene_graphs_gt'))
        pt_sg_dir = cfg.get('pt_scene_graphs_dir', self.dirs.get('scene_graphs_pt'))
        
        convert_cmd = ['python', str(converter_script)]
        
        use_gt_pt_mode = False
        if not gt_sg_dir:
            gt_sg_dir = self.dirs.get('scene_graphs_gt')
        if not pt_sg_dir:
            pt_sg_dir = self.dirs.get('scene_graphs_pt')
            
        if gt_sg_dir and pt_sg_dir:
            gt_sg_dir = Path(gt_sg_dir)
            pt_sg_dir = Path(pt_sg_dir)
            
            if gt_sg_dir.exists() and pt_sg_dir.exists():
                gt_files = list(gt_sg_dir.glob('scene-graph_*.json'))
                pt_files = list(pt_sg_dir.glob('scene-graph_*.json'))
                
                if gt_files and pt_files:
                    use_gt_pt_mode = True
                    print(f"Matching ground truth from: {gt_sg_dir} ({len(gt_files)} files)")
                    print(f"Matching predictions from: {pt_sg_dir} ({len(pt_files)} files)")
                elif gt_files or pt_files:
                    print(f"Warning: GT or PT scene graph directories are empty (GT: {len(gt_files)}, PT: {len(pt_files)})")
                else:
                    print(f"Warning: GT and PT scene graph directories exist but are empty")
        
        if use_gt_pt_mode:
            convert_cmd.extend([
                '--gt-dir', str(gt_sg_dir),
                '--pt-dir', str(pt_sg_dir),
                '--output', str(matching_input)
            ])
        else:
            raise ValueError(
                "Single directory mode is no longer supported. "
                "GT and PT scene graph directories are required for graph matching."
            )
        
        if self.run_command(convert_cmd, 'format_conversion') != 0:
            print("ERROR: Failed to convert scene graphs to matching format")
            return False
        
        if not matching_input.exists():
            print("ERROR: Matching input file was not created")
            return False
        
        print(f"✓ Created matching input at: {matching_input}")
        
        matching_env = cfg.get('conda_env', 'graph-matching')
        
        cmd = [
            'conda', 'run', '-n', matching_env, '--no-capture-output',
            'python', cfg['script'],
            '--input', str(matching_input),
            '--output', str(self.dirs['matching'] / 'scene_graphs_matched.json'),
            '--model', cfg.get('model', 'all-MiniLM-L6-v2')
        ]
        
        if cfg.get('high_recall', True):
            cmd.append('--high-recall')
        
        match_success = self.run_command(cmd, 'graph_matching') == 0
        
        if match_success:
            print("Generating prediction masks for visualization...")
            extract_masks_script = Path(__file__).parent.parent / 'extract_pred_masks_from_scenegraph.py'
            
            sg_pt_dir = self.dirs.get('scene_graphs_pt')
            
            compact_cfg = self.config.get('compact_extraction', {})
            if compact_cfg.get('pt_scene_graphs_dir'):
                sg_pt_dir = Path(compact_cfg['pt_scene_graphs_dir'])
            
            if sg_pt_dir and (sg_pt_dir / 'scene-graph.pkl').exists():
                pred_masks_out = self.dirs['matching'] / 'prediction_masks_from_scenegraph.pkl'
                
                cmd_mask = [
                    'conda', 'run', '-n', matching_env, '--no-capture-output',
                    'python', str(extract_masks_script),
                    '--scene-graphs-dir', str(sg_pt_dir),
                    '--output', str(pred_masks_out)
                ]
                
                if self.run_command(cmd_mask, 'extract_pred_masks') != 0:
                    print("Warning: Failed to generate prediction masks pickle")
            else:
                print(f"Warning: scene-graph.pkl not found at {sg_pt_dir}. Skipping mask extraction.")
                
        return match_success


