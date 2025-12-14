#!/usr/bin/env python3
"""
Step 4: Attribute Generation

Generate structured attributes for objects using VLM (Qwen3-VL-32B-Instruct) via VLLM.
"""

import json
from pathlib import Path

from .base_step import BaseStep


class Step4AttributeGeneration(BaseStep):
    """Step 4: Generate structured attributes for objects."""
    
    def execute(self) -> bool:
        """
        Step 4: Generate structured attributes for objects.
        
        Uses VLM (Qwen3-VL-32B-Instruct) via VLLM with guided decoding.
        """
        print("\n" + "="*70)
        print("STEP 4: ATTRIBUTE GENERATION")
        print("="*70)
        
        cfg = self.config['attribute_generation']
        
        if cfg.get('skip', False):
            print("Skipping attribute generation")
            return True
        
        process_gt_pt_separately = cfg.get('process_gt_pt_separately', False)
        
        if process_gt_pt_separately:
            process_mode = self.get_process_mode()
            print(f"Process mode: {process_mode}")
            print("Processing GT and PT directories separately...")
            
            gt_attr_success = True
            pt_attr_success = True
            
            if self.should_process_gt():
                img_dir_gt = cfg.get('img_dir_gt', cfg.get('img_dir'))
                
                anno_path_gt = self.dirs['segmentation_gt'] / 'anno.json'
                if not anno_path_gt.exists():
                    anno_path_gt = self.dirs['segmentation'] / 'anno.json'
            
                merge_cfg = self.config.get('graph_merging', {})
                if not merge_cfg.get('skip', False):
                    merged_sg_dir_gt = self.dirs['graph_merge_gt']
                    
                    if list(merged_sg_dir_gt.glob('scene-graph_*.json')):
                        print("Using merged scene graphs for GT")
                        scene_graphs_dir_gt = merged_sg_dir_gt
                    else:
                        print("Using original scene graphs for GT (merged not found)")
                        scene_graphs_dir_gt = self.dirs['scene_graphs_gt']
                else:
                    scene_graphs_dir_gt = self.dirs['scene_graphs_gt']
                
                gt_attr_success = self._process_attribute_generation(
                    cfg,
                    img_dir_gt,
                    anno_path_gt,
                    self.dirs['segmentation_gt'],
                    scene_graphs_dir_gt,
                    "GT"
                )
            else:
                gt_attr_success = True
            
            if self.should_process_pt():
                img_dir_pt = cfg.get('img_dir_pt', cfg.get('img_dir'))
                
                anno_path_pt = self.dirs['segmentation_pt'] / 'anno.json'
                if not anno_path_pt.exists():
                    anno_path_pt = self.dirs['segmentation'] / 'anno.json'
                
                merge_cfg = self.config.get('graph_merging', {})
                if not merge_cfg.get('skip', False):
                    merged_sg_dir_pt = self.dirs['graph_merge_pt']
                    
                    if list(merged_sg_dir_pt.glob('scene-graph_*.json')):
                        print("Using merged scene graphs for PT")
                        scene_graphs_dir_pt = merged_sg_dir_pt
                    else:
                        print("Using original scene graphs for PT (merged not found)")
                        scene_graphs_dir_pt = self.dirs['scene_graphs_pt']
                else:
                    scene_graphs_dir_pt = self.dirs['scene_graphs_pt']
                
                pt_attr_success = self._process_attribute_generation(
                    cfg,
                    img_dir_pt,
                    anno_path_pt,
                    self.dirs['segmentation_pt'],
                    scene_graphs_dir_pt,
                    "PT"
                )
            else:
                pt_attr_success = True
            
            gt_merge_success = True
            pt_merge_success = True
            
            if self.should_process_gt() and gt_attr_success:
                merge_cfg = self.config.get('graph_merging', {})
                if not merge_cfg.get('skip', False) and list(self.dirs['graph_merge_gt'].glob('scene-graph_*.json')):
                    scene_graphs_dir_gt = self.dirs['graph_merge_gt']
                else:
                    scene_graphs_dir_gt = self.dirs['scene_graphs_gt']
                
                gt_merge_success = self._merge_attributes_into_scene_graphs(
                    self.dirs['attributes_gt'] / 'attributes.json',
                    scene_graphs_dir_gt,
                    self.dirs['attributes_gt'],
                    "GT"
                )
            
            if self.should_process_pt() and pt_attr_success:
                merge_cfg = self.config.get('graph_merging', {})
                if not merge_cfg.get('skip', False) and list(self.dirs['graph_merge_pt'].glob('scene-graph_*.json')):
                    scene_graphs_dir_pt = self.dirs['graph_merge_pt']
                else:
                    scene_graphs_dir_pt = self.dirs['scene_graphs_pt']
                
                pt_merge_success = self._merge_attributes_into_scene_graphs(
                    self.dirs['attributes_pt'] / 'attributes.json',
                    scene_graphs_dir_pt,
                    self.dirs['attributes_pt'],
                    "PT"
                )
            else:
                pt_merge_success = True
            
            return gt_attr_success and gt_merge_success and pt_attr_success and pt_merge_success
        else:
            raise ValueError(
                "Single directory mode is no longer supported. "
                "Please set 'process_gt_pt_separately': true in attribute_generation config."
            )
    
    def _merge_attributes_into_scene_graphs(self, attributes_json: Path, scene_graphs_dir: Path, output_dir: Path, label: str) -> bool:
        """Merge attributes into scene graphs and save to output directory."""
        if not attributes_json.exists():
            print(f"Warning: Attributes file not found: {attributes_json}")
            return False
        
        with open(attributes_json, 'r') as f:
            attr_list = json.load(f)
        
        category_id_to_name = {}
        for entry in attr_list:
            cat_id = entry.get("category_id")
            cat_name = entry.get("category")
            if cat_id is not None and cat_name:
                category_id_to_name[cat_id] = cat_name
        
        attr_index_by_bbox = {}
        attr_index_by_index = {}
        
        for entry in attr_list:
            image_id = entry.get("image_id")
            category_id = entry.get("category_id")
            bbox = entry.get("bbox", [])
            index = entry.get("index")
            attrs = entry.get("attributes", {})
            
            if image_id is None:
                continue
            
            if isinstance(image_id, str):
                try:
                    image_id = int(image_id)
                except:
                    pass
            
            if index is not None:
                attr_index_by_index[(image_id, index)] = attrs
            
            if category_id is not None and len(bbox) == 4:
                bbox_xyxy = bbox
                bbox_normalized = tuple(int(round(x / 5)) * 5 for x in bbox_xyxy)
                attr_index_by_bbox[(image_id, category_id, bbox_normalized)] = attrs
        
        print(f"Built attribute index: {len(attr_index_by_index)} by index, {len(attr_index_by_bbox)} by bbox")
        
        name_to_category_id = {v: k for k, v in category_id_to_name.items()}
        
        scene_graph_files = list(scene_graphs_dir.glob("scene-graph_*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        merged_count = 0
        total_matched = 0
        
        for sg_file in scene_graph_files:
            with open(sg_file, 'r') as f:
                scene_graph = json.load(f)
            
            image_id = scene_graph.get("image_id")
            if isinstance(image_id, str):
                try:
                    image_id = int(image_id)
                except:
                    pass
            
            for box in scene_graph.get("boxes", []):
                index = box.get("index")
                label_name = box.get("label")
                bbox_xyxy = box.get("bbox_xyxy", [])
                
                attrs = {}
                
                if index is not None:
                    key = (image_id, index)
                    attrs = attr_index_by_index.get(key, {})
                
                if not attrs and label_name and len(bbox_xyxy) == 4:
                    category_id = name_to_category_id.get(label_name)
                    if category_id is not None:
                        bbox_normalized = tuple(int(round(x / 5)) * 5 for x in bbox_xyxy)
                        key = (image_id, category_id, bbox_normalized)
                        attrs = attr_index_by_bbox.get(key, {})
                
                if attrs:
                    total_matched += 1
                
                box["attributes"] = attrs
            
            output_file = output_dir / sg_file.name
            with open(output_file, 'w') as f:
                json.dump(scene_graph, f, indent=2)
            merged_count += 1
        
        print(f"✓ Merged attributes into {merged_count} scene graphs in {output_dir}")
        print(f"✓ Matched {total_matched} boxes with attributes")
        return True
    
    def _process_attribute_generation(self, cfg: dict, img_dir: str, anno_path: Path, seg_dir: Path, scene_graphs_dir: Path, label: str) -> bool:
        """
        Helper to process attribute generation for a directory.
        Attributes are generated and automatically merged into scene graph files.
        """
        if not scene_graphs_dir.exists():
            print(f"Warning: Scene graphs directory not found: {scene_graphs_dir}")
            return False
        
        if not anno_path.exists():
            print(f"Warning: anno.json not found: {anno_path}")
            return False
        
        attr_env = cfg.get('conda_env')
        
        mapping_json = cfg.get('mapping_json', str(Path(__file__).parent.parent.parent / 'psg_generation' / 'attribute_generation' / 'category_mapping.json'))
        
        base_cmd = [
            '--anno-json', str(anno_path),
            '--img-dir', str(img_dir),
            '--seg-dir', str(seg_dir),
            '--scene-graphs-dir', str(scene_graphs_dir),
            '--output-dir', str(self.dirs[f'attributes_{label.lower()}']),
            '--mapping-json', mapping_json,
            '--num-gpus', str(cfg.get('num_gpus', 4)),
            '--batch-size', str(cfg.get('batch_size', 8))
        ]
        
        if cfg.get('fake', False):
            base_cmd.append('--fake')
        
        if attr_env:
            cmd = ['conda', 'run', '-n', attr_env, '--no-capture-output', 'python', cfg['script']] + base_cmd
        else:
            cmd = ['python', cfg['script']] + base_cmd
        
        return self.run_command(cmd, f'attribute_generation_{label}') == 0


