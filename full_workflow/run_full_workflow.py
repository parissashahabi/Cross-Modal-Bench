#!/usr/bin/env python3
"""
Complete Scene Graph Generation and Matching Workflow

This script orchestrates the entire pipeline:
1. Scene Graph Generation (segmentation + relation prediction)
2. Attribute Generation (VLM-based object attributes)
3. Compact Format Extraction
4. Graph Matching (Hungarian algorithm)
5. Visualization

All intermediate results are stored in organized subdirectories.
"""

import argparse
import json
import sys
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

from workflow_steps import (
    Step1Segmentation,
    Step2SceneGraphGeneration,
    Step3GraphMerging,
    Step4AttributeGeneration,
    Step5GraphMatching,
    Step6Visualization,
)


class WorkflowRunner:
    """Manages the full workflow execution."""
    
    def __init__(self, config_path: str, workflow_dir: Optional[Union[str, Path]] = None):
        """
        Initialize workflow runner.
        
        Args:
            config_path: Path to configuration JSON file
            workflow_dir: Base directory for workflow (defaults to script location)
        """
        self.config = self.load_config(config_path)
        
        if workflow_dir is None:
            workflow_dir = Path(__file__).parent.resolve()
        else:
            workflow_dir = Path(workflow_dir).resolve()

        # Ensure instance attribute is a Path for consistent path operations
        self.workflow_dir: Path = workflow_dir
        
        # Change working directory to workflow directory to ensure relative paths work
        print(f"Changing working directory to: {self.workflow_dir}")
        os.chdir(self.workflow_dir)
        
        self.setup_directories()
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {config_path}")
        return config
    
    def get_process_mode(self) -> str:
        """Get process mode from config: 'both', 'gt_only', or 'pt_only'."""
        return self.config.get('process_mode', 'both')
    
    def should_process_gt(self) -> bool:
        """Check if GT should be processed based on process_mode."""
        mode = self.get_process_mode()
        return mode in ('both', 'gt_only')
    
    def should_process_pt(self) -> bool:
        """Check if PT should be processed based on process_mode."""
        mode = self.get_process_mode()
        return mode in ('both', 'pt_only')
    
    def setup_directories(self):
        """Create all necessary subdirectories."""
        self.dirs = {
            'segmentation_gt': self.workflow_dir / '1_segmentation_gt',
            'segmentation_pt': self.workflow_dir / '1_segmentation_pt',
            'scene_graphs_gt': self.workflow_dir / '2_scene_graphs_gt',
            'scene_graphs_pt': self.workflow_dir / '2_scene_graphs_pt',
            'graph_merge_gt': self.workflow_dir / '3_graph_merge_gt',
            'graph_merge_pt': self.workflow_dir / '3_graph_merge_pt',
            'attributes_gt': self.workflow_dir / '4_attributes_gt',
            'attributes_pt': self.workflow_dir / '4_attributes_pt',
            'matching': self.workflow_dir / '5_matching',
            'visualizations': self.workflow_dir / '6_visualizations',
            'logs': self.workflow_dir / 'logs',
        }
        
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"Directory ready: {path}")
    
    def run_command(self, cmd: list, step_name: str, log_file: Optional[Union[str, Path]] = None) -> int:
        """
        Run a command and log output.
        
        Args:
            cmd: Command as list of strings
            step_name: Name of the step for logging
            log_file: Optional log file path
            
        Returns:
            Return code
        """
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.dirs['logs'] / f"{step_name}_{timestamp}.log"
        
        print(f"\n{'='*70}")
        print(f"Running: {step_name}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Log file: {log_file}")
        print(f"{'='*70}\n")
        
        # Allow passing either a Path or str for the log file
        with open(log_file, 'w', buffering=1) as log:
            log.write(f"Command: {' '.join(cmd)}\n")
            log.write(f"Started: {datetime.now()}\n")
            log.write("="*70 + "\n\n")
            
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            if process.stdout:
                for line in process.stdout:
                    print(f"[{step_name}] {line}", end='')
                    log.write(line)
            
            process.wait()
            
            log.write(f"\n{'='*70}\n")
            log.write(f"Finished: {datetime.now()}\n")
            log.write(f"Return code: {process.returncode}\n")
        
        print(f"\n{'='*70}")
        if process.returncode == 0:
            print(f"✓ {step_name} completed successfully")
        else:
            print(f"✗ {step_name} failed with return code {process.returncode}")
            print(f"Check log file: {log_file}")
        
        return process.returncode
    
    def run_workflow(self, start_step: int = 1, end_step: int = 6) -> bool:
        """
        Run the complete workflow or a range of steps.
        
        Args:
            start_step: First step to run (1-6)
            end_step: Last step to run (1-6)
            
        Returns:
            True if all steps succeeded
        """
        steps = {
            1: ('Segmentation', Step1Segmentation(self)),
            2: ('Scene Graph Generation', Step2SceneGraphGeneration(self)),
            3: ('Graph Merging', Step3GraphMerging(self)),
            4: ('Attribute Generation', Step4AttributeGeneration(self)),
            5: ('Graph Matching', Step5GraphMatching(self)),
            6: ('Visualization', Step6Visualization(self)),
        }
        
        print("\n" + "="*70)
        print("FULL WORKFLOW EXECUTION")
        print("="*70)
        print(f"Workflow directory: {self.workflow_dir}")
        print(f"Running steps {start_step} to {end_step}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success = True
        for step_num in range(start_step, end_step + 1):
            if step_num not in steps:
                print(f"Invalid step number: {step_num}")
                continue
            
            step_name, step_instance = steps[step_num]
            
            try:
                result = step_instance.execute()
                if not result:
                    print(f"\n✗ Step {step_num} ({step_name}) FAILED")
                    success = False
                    if self.config.get('stop_on_error', True):
                        print("Stopping workflow due to error")
                        break
                else:
                    print(f"\n✓ Step {step_num} ({step_name}) completed successfully")
            except Exception as e:
                print(f"\n✗ Step {step_num} ({step_name}) ERROR: {e}")
                import traceback
                traceback.print_exc()
                success = False
                if self.config.get('stop_on_error', True):
                    break
        
        print("\n" + "="*70)
        print("WORKFLOW SUMMARY")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if success:
            print("✓ All steps completed successfully")
            print("\nOutput directories:")
            for name, path in self.dirs.items():
                if name != 'logs':
                    # For steps 2, 3: only count GT and PT directories
                    if name in ['scene_graphs', 'attributes']:
                        # Skip main directories, only show GT/PT
                        continue
                    elif name in ['segmentation_gt', 'segmentation_pt', 'scene_graphs_gt', 'scene_graphs_pt', 'graph_merge_gt', 'graph_merge_pt', 'attributes_gt', 'attributes_pt']:
                        file_count = len(list(path.glob('*')))
                        print(f"  {name}: {path} ({file_count} files)")
                    else:
                        # For other directories (matching, visualizations, etc.), show normally
                        file_count = len(list(path.glob('*')))
                        print(f"  {name}: {path} ({file_count} files)")
        else:
            print("✗ Workflow completed with errors")
            print(f"Check logs in: {self.dirs['logs']}")
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Complete scene graph generation and matching workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow
  python run_full_workflow.py --config config.json
  
  # Run only segmentation and scene graph generation
  python run_full_workflow.py --config config.json --start 1 --end 2
  
  # Run from graph merging onward
  python run_full_workflow.py --config config.json --start 3
  
  # Run from attributes onward
  python run_full_workflow.py --config config.json --start 4
  
  # Dry run (show what would be executed)
  python run_full_workflow.py --config config.json --dry-run
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--workflow-dir',
        type=str,
        default=None,
        help='Base directory for workflow (default: script directory)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help='First step to run (1-6, default: 1)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=6,
        choices=[1, 2, 3, 4, 5, 6],
        help='Last step to run (1-6, default: 6)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )
    
    args = parser.parse_args()
    
    if args.start > args.end:
        print("ERROR: --start must be <= --end")
        return 1
    
    try:
        runner = WorkflowRunner(args.config, args.workflow_dir)
        
        if args.dry_run:
            print("\nDRY RUN - Configuration loaded successfully")
            print("Would execute steps:", args.start, "to", args.end)
            return 0
        
        success = runner.run_workflow(args.start, args.end)
        return 0 if success else 1
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
