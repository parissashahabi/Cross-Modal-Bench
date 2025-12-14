#!/usr/bin/env python3
"""
Step 3: Graph Merging

Merge overlapping bounding boxes in scene graphs into merged nodes with merged bboxes.
Runs after scene graph generation and before attribute generation.
"""

import json
import shutil
from pathlib import Path

from .base_step import BaseStep


class Step3GraphMerging(BaseStep):
    """Step 3: Merge overlapping bounding boxes in scene graphs."""
    
    def execute(self) -> bool:
        """
        Step 3: Merge overlapping bounding boxes in scene graphs.
        
        Merges overlapping segments into merged nodes with merged bboxes.
        This runs after scene graph generation and before attribute generation.
        """
        print("\n" + "="*70)
        print("STEP 3: GRAPH MERGING")
        print("="*70)
        
        cfg = self.config.get('graph_merging', {})
        
        if cfg.get('skip', False):
            print("Skipping graph merging")
            return True
        
        sg_cfg = self.config.get('scene_graph_generation', {})
        process_gt_pt_separately = sg_cfg.get('process_separately', False)
        
        if process_gt_pt_separately:
            process_mode = self.get_process_mode()
            print(f"Process mode: {process_mode}")
            print("Processing GT and PT directories separately...")
            
            gt_success = True
            pt_success = True
            
            if self.should_process_gt():
                gt_success = self._process_graph_merging(
                    cfg,
                    self.dirs['segmentation_gt'],
                    self.dirs['scene_graphs_gt'],
                    self.dirs['graph_merge_gt'],
                    "GT"
                )
            
            if self.should_process_pt():
                pt_success = self._process_graph_merging(
                    cfg,
                    self.dirs['segmentation_pt'],
                    self.dirs['scene_graphs_pt'],
                    self.dirs['graph_merge_pt'],
                    "PT"
                )
            
            return gt_success and pt_success
        else:
            raise ValueError(
                "Graph merging requires process_separately: true in scene_graph_generation config."
            )
    
    def _process_graph_merging(self, cfg: dict, seg_dir: Path, scene_graphs_dir: Path, output_dir: Path, label: str) -> bool:
        """
        Helper to process graph merging for a directory.
        
        Steps:
        1. Run merge_graph.py to create anno_merged.json
        2. Run merge_masks.py to create merged masks
        3. Run merge_edges.py to create merged graph JSON
        4. Convert merged graph format to scene graph format
        """
        merge_env = cfg.get('conda_env')
        
        anno_path = seg_dir / 'anno.json'
        if not anno_path.exists():
            print(f"Warning: anno.json not found at {anno_path}")
            return False
        
        scene_pkl_path = scene_graphs_dir / 'scene-graph.pkl'
        if not scene_pkl_path.exists():
            print(f"Warning: scene-graph.pkl not found at {scene_pkl_path}")
            return False
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        merge_graph_script = cfg.get('merge_graph_script', '../psg_generation/graph_merge/merge_graph.py')
        padding = cfg.get('padding', 10)
        
        anno_merged_path = output_dir / 'anno_merged.json'
        
        if merge_env:
            cmd1 = [
                'conda', 'run', '-n', merge_env, '--no-capture-output',
                'python', merge_graph_script,
                '--input', str(anno_path),
                '--padding', str(padding)
            ]
        else:
            cmd1 = [
                'python', merge_graph_script,
                '--input', str(anno_path),
                '--padding', str(padding)
            ]
        
        temp_merged_path = anno_path.parent / anno_path.name.replace('.json', '_merged.json')
        
        if self.run_command(cmd1, f'merge_graph_{label}') != 0:
            return False
        
        if temp_merged_path.exists():
            shutil.move(str(temp_merged_path), str(anno_merged_path))
        else:
            print(f"Warning: Expected merged file not found at {temp_merged_path}")
            return False
        
        merge_masks_script = cfg.get('merge_masks_script', '../psg_generation/graph_merge/merge_masks.py')
        merged_masks_dir = output_dir / 'merged_segmentations'
        
        if merge_env:
            cmd2 = [
                'conda', 'run', '-n', merge_env, '--no-capture-output',
                'python', merge_masks_script,
                '--anno', str(anno_path),
                '--merged', str(anno_merged_path),
                '--output-dir', str(merged_masks_dir)
            ]
        else:
            cmd2 = [
                'python', merge_masks_script,
                '--anno', str(anno_path),
                '--merged', str(anno_merged_path),
                '--output-dir', str(merged_masks_dir)
            ]
        
        if self.run_command(cmd2, f'merge_masks_{label}') != 0:
            return False
        
        merge_edges_script = cfg.get('merge_edges_script', '../psg_generation/graph_merge/merge_edges.py')
        agg = cfg.get('agg', 'mean')
        merged_graph_path = output_dir / 'anno_merged_edges.json'
        
        if merge_env:
            cmd3 = [
                'conda', 'run', '-n', merge_env, '--no-capture-output',
                'python', merge_edges_script,
                '--anno', str(anno_path),
                '--merged', str(anno_merged_path),
                '--scene-pkl', str(scene_pkl_path),
                '--output', str(merged_graph_path),
                '--format', 'json',
                '--agg', agg
            ]
        else:
            cmd3 = [
                'python', merge_edges_script,
                '--anno', str(anno_path),
                '--merged', str(anno_merged_path),
                '--scene-pkl', str(scene_pkl_path),
                '--output', str(merged_graph_path),
                '--format', 'json',
                '--agg', agg
            ]
        
        if self.run_command(cmd3, f'merge_edges_{label}') != 0:
            return False
        
        if not merged_graph_path.exists():
            print(f"Warning: Merged graph file not found at {merged_graph_path}")
            return False
        
        return self._convert_merged_to_scene_graph(merged_graph_path, output_dir, label)
    
    def _convert_merged_to_scene_graph(self, merged_graph_path: Path, output_dir: Path, label: str) -> bool:
        """
        Convert merged graph format to scene graph format.
        
        Input format: {image_id: {groups: [...], edges: [...]}}
        Output format: {image_id, file_name, boxes: [{index, label, bbox_xyxy}], relations: [...]}
        """
        with open(merged_graph_path, 'r') as f:
            merged_data = json.load(f)
        
        for image_id_str, graph_data in merged_data.items():
            image_id = graph_data.get('image_id')
            file_name = graph_data.get('file_name')
            groups = graph_data.get('groups', [])
            edges = graph_data.get('edges', [])
            
            group_id_to_index = {}
            boxes = []
            
            for idx, group in enumerate(sorted(groups, key=lambda g: g.get('group_id', 0))):
                group_id = group.get('group_id')
                group_label = group.get('label', '')
                bbox = group.get('bbox', [])
                
                if len(bbox) == 4:
                    bbox_xyxy = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                else:
                    print(f"Warning: Invalid bbox format for group {group_id}: {bbox}")
                    bbox_xyxy = [0, 0, 0, 0]
                
                group_id_to_index[group_id] = idx
                boxes.append({
                    "index": idx,
                    "label": group_label,
                    "bbox_xyxy": bbox_xyxy
                })
            
            relations = []
            for edge in edges:
                sub_gid = edge.get('subject_group_id')
                obj_gid = edge.get('object_group_id')
                
                if sub_gid not in group_id_to_index or obj_gid not in group_id_to_index:
                    continue
                
                sub_idx = group_id_to_index[sub_gid]
                obj_idx = group_id_to_index[obj_gid]
                
                sub_label = boxes[sub_idx].get('label', '')
                obj_label = boxes[obj_idx].get('label', '')
                
                relation = {
                    "subject_index": sub_idx,
                    "subject_label": sub_label,
                    "object_index": obj_idx,
                    "object_label": obj_label,
                    "predicate": edge.get('best_predicate', ''),
                    "predicate_score": edge.get('best_predicate_score', 0.0),
                    "no_relation_score": edge.get('no_relation_score', 0.0)
                }
                
                relations.append(relation)
            
            scene_graph = {
                "image_id": image_id,
                "file_name": file_name,
                "boxes": boxes,
                "relations": relations
            }
            
            output_file = output_dir / f"scene-graph_{image_id}.json"
            with open(output_file, 'w') as f:
                json.dump(scene_graph, f, indent=2)
            
            print(f"Converted merged graph for image {image_id}: {len(boxes)} boxes, {len(relations)} relations")
        
        return True


