# Full Scene Graph Generation and Matching Workflow

Complete pipeline for generating scene graphs with attributes and performing graph matching on the cluster.

## Overview

This workflow combines two main components:

1. **Scene Graph Generation with Attributes** (2-step process)
   - Generate segmentation masks (Mask2Former or kMaX-DeepLab)
   - Predict scene graph relationships (DSFormer)
   - Generate structured object attributes (VLM)
   - Extract compact format

2. **Graph Matching**
   - Match predicted graphs with ground truth using Hungarian algorithm
   - Compute precision, recall, and F1 scores
   - Generate visualizations with matched masks

## Directory Structure

```
full_workflow/
├── run_full_workflow.py          # Main orchestration script
├── config.json                   # Configuration file
├── config_example_kmax.json      # Example config with kMaX
├── setup.sh                      # Setup script
├── run_workflow.sh               # Local/interactive execution
├── README.md                     # This file
│
├── 1_segmentation/              # Segmentation outputs
│   ├── anno.json                # Annotations
│   └── seg_*.png                # Segmentation masks
│
├── 2_scene_graphs/              # Scene graph outputs
│   ├── custom_psg.json          # PSG format
│   ├── scene-graph.pkl          # Inference results
│   ├── scene-graph_*.json       # Readable scene graphs
│   └── scene-graph_*.png        # Visualizations
│
├── 3_graph_merge/              # Merged scene graphs
├── 4_attributes/                # VLM-generated attributes
│   └── attributes.json
│
├── 4_compact/                   # Compact format
│   └── scene-graph-description_*_compact.json
│
├── 5_matching/                  # Graph matching results
│   ├── scene_graphs_for_matching.json
│   └── scene_graphs_matched.json
│
├── 6_visualizations/            # Visual outputs
│   └── matched_*.png
│
└── logs/                        # Execution logs
    └── *.log
```

## Environment Requirements

This workflow uses **different environments** for different steps:

### Poetry Environment (Steps 2-4)
- Scene Graph Generation (Step 2)
- Attribute Generation (Step 3)
- Compact Extraction (Step 4)

**Setup**: Navigate to `vlm-benchmark/psg_generation/` and activate:
```bash
cd /sc/home/anton.hackl/master-project/vlm-benchmark/psg_generation
poetry install
poetry shell  # or use: poetry run python <script>
```

### Conda Environment: kmax_seg (Step 1, if using kMaX)
- kMaX-DeepLab Segmentation (Step 1, optional)

**Setup**:
```bash
conda env create -f vlm-benchmark/psg_generation/kmax_environment.yml
conda activate kmax_seg
```

### Conda Environment: graph-matching (Steps 5-6)
- Graph Matching (Step 5)
- Visualization (Step 6)

**Setup**:
```bash
conda env create -f graph_matching/environment.yml
conda activate graph-matching
```

**Note**: The workflow scripts automatically handle environment switching using `poetry run` and `conda run` commands.

## Quick Start

### 1. Setup

```bash
cd /sc/home/anton.hackl/master-project/full_workflow

# Run setup script to create directories
./setup.sh

# Edit configuration file with your paths
nano config.json
```

### 2. Configuration

Edit `config.json` to set your paths and parameters:

```json
{
  "segmentation": {
    "method": "mask2former",
    "img_dir": "/path/to/images",
    "psg_meta": "/path/to/psg.json",
    ...
  },
  "scene_graph_generation": {
    "model_dir": "/path/to/dsformer/model",
    ...
  },
  ...
}
```

Key configuration options:

- **Segmentation method**: `"mask2former"` (default) or `"kmax"`
- **Model paths**: Update paths to your trained models
- **GPU settings**: Adjust `num_gpus` and `batch_size`
- **Skip flags**: Set `"skip": true` to skip specific steps

### 3. Run Workflow

#### Interactive/Local Execution

```bash
# Run complete workflow
./run_workflow.sh

# Run specific steps
./run_workflow.sh --start 1 --end 3

# Run with custom config
./run_workflow.sh --config my_config.json
```

#### Direct Python Execution

```bash
# Full workflow
python run_full_workflow.py --config config.json

# Specific steps
python run_full_workflow.py --config config.json --start 2 --end 4

# Dry run (check configuration)
python run_full_workflow.py --config config.json --dry-run
```

## Workflow Steps

### Step 1: Segmentation

Generates panoptic segmentation masks from input images.

**Options:**
- **Mask2Former** (default): Uses HuggingFace transformers
- **kMaX-DeepLab**: Better quality, requires separate conda environment and model checkpoint

**Requirements for kMaX-DeepLab:**
- Model checkpoint: `kmax_convnext_large.pth` must be placed in `./models/`
- Download from the kMaX-DeepLab repository or copy from your trained models

**Outputs:**
- `1_segmentation/anno.json`: Annotation file
- `1_segmentation/seg_*.png`: Segmentation masks

**Skip this step** if you already have segmentation results:
```json
"segmentation": {
  "skip": true,
  "anno_path": "/path/to/existing/anno.json"
}
```

### Step 2: Scene Graph Generation

Generates scene graphs with object relationships using DSFormer.

**Environment**: Poetry (fair-psg)

**Requirements:**
- Segmentation results from Step 1 (or existing anno.json)
- Trained DSFormer model (optional, can skip inference)
- Poetry environment activated

**Outputs:**
- `2_scene_graphs/custom_psg.json`: PSG format
- `2_scene_graphs/scene-graph.pkl`: Raw inference results
- `2_scene_graphs/scene-graph_*.json`: Readable scene graphs
- `2_scene_graphs/scene-graph_*.png`: Visualizations

### Step 3: Graph Merging

Merges overlapping bounding boxes in scene graphs into merged nodes with merged bboxes.

**Environment**: Same as scene graph generation (kmax_deeplab)

**Requirements:**
- Scene graph results from Step 2 (scene-graph.pkl and anno.json)

**Outputs:**
- `3_graph_merge/anno_merged.json`: Merge mapping
- `3_graph_merge/anno_merged_edges.json`: Merged graph with groups and edges
- `3_graph_merge/scene-graph_*.json`: Converted scene graphs with merged nodes

**Skip this step** if you don't need merging:
```json
"graph_merging": {
  "skip": true
}
```

### Step 4: Attribute Generation

Generates structured attributes for objects using VLM (Qwen3-VL-32B-Instruct) with guided decoding.

**Environment**: Poetry (with VLLM)

**Requirements:**
- Segmentation results from Step 1 (anno.json)
- Scene graphs from Step 2 or merged graphs from Step 3
- Multiple GPUs recommended (default: 4)
- Category mapping file (defaults to `category_mapping.json`)

**Outputs:**
- `4_attributes/attributes.json`: Structured attributes for all objects
- `4_attributes/scene-graph_*.json`: Scene graphs with merged attributes

**Skip this step** if you don't need attributes:
```json
"attribute_generation": {
  "skip": true
}
```

### Compact Extraction (Optional Step)

Extracts simplified JSON format with attributes and predicates.

**Outputs:**
- `4_compact/scene-graph-description_*_compact.json`: Compact format

**Format:**
```json
{
  "image_id": 123456,
  "file_name": "...",
  "full_image_description": "...",
  "descriptions": [
    {"index": 0, "label": "person", "description": "..."},
    ...
  ],
  "predicates": [
    {"subject_index": 0, "object_index": 1, "predicate": "on"},
    ...
  ]
}
```

**Note**: Attributes need to be merged into scene graphs before compact extraction. See `vlm-benchmark/psg_generation/attribute_merge/merge_attributes.py` for merging attributes into scene graphs.

### Step 5: Graph Matching

Matches predicted scene graphs with ground truth using Hungarian algorithm.

**Environment**: Conda (graph-matching)

**Requirements:**
- Scene graphs in matching format
- Sentence transformer model for embeddings
- Conda environment with sentence-transformers

**Outputs:**
- `5_matching/scene_graphs_matched.json`: Matching results with metrics

**Note:** This step may require format adaptation depending on your scene graph format. Check the script for expected input format.

### Step 6: Visualization

Generates side-by-side visualizations of matched graphs with masks.

**Environment**: Conda (graph-matching)

**Outputs:**
- `6_visualizations/matched_*.png`: Visual comparisons

### Resuming Failed Workflows

If a step fails, fix the issue and resume from that step:

```bash
# Workflow failed at step 3
# Fix the issue, then:
python run_full_workflow.py --config config.json --start 3
```

### Using Existing Intermediate Results

You can skip steps by providing existing results:

1. **Skip segmentation**: Set `"skip": true` and provide `"anno_path"`
2. **Skip attributes**: Set `"skip": true` in attribute_generation
3. **Skip inference**: Remove or set `"model_dir": null` in scene_graph_generation

## Logs and Debugging

All execution logs are saved in `logs/`:

```bash
# View latest log
ls -lt logs/*.log | head -1 | xargs cat

# Monitor running workflow
tail -f logs/scene_graph_generation_*.log

# Check for errors
grep -i error logs/*.log
```

## Resource Requirements

### Memory

- **Segmentation**: 16-32 GB RAM
- **Scene Graph Generation**: 32-64 GB RAM
- **Attribute Generation**: 64-128 GB RAM (depends on model and batch size)
- **Graph Matching**: 8-16 GB RAM
- **Visualization**: 8-16 GB RAM

### GPU

- **Segmentation**: 1 GPU (8-16 GB VRAM)
- **Scene Graph Generation**: 1-2 GPUs (16-24 GB VRAM)
- **Attribute Generation**: 4 GPUs recommended (24-40 GB VRAM each)
- **Graph Matching**: CPU only
- **Visualization**: CPU only

### Time Estimates (for 100 images)

- **Segmentation**: 30-60 minutes
- **Scene Graph Generation**: 1-2 hours
- **Attribute Generation**: 2-4 hours
- **Compact Extraction**: 5-10 minutes
- **Graph Matching**: 30-60 minutes
- **Visualization**: 30-60 minutes

**Total**: ~5-9 hours for 100 images

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Activate correct conda environment
   - Install missing packages: `pip install -r requirements.txt`

2. **CUDA out of memory**
   - Reduce batch size in config
   - Reduce number of GPUs for attribute generation
   - Process fewer images at once

3. **File not found errors**
   - Check all paths in config.json are absolute paths
   - Ensure input images exist
   - Verify model checkpoints are downloaded

4. **Segmentation produces no masks**
   - Check image format (should be RGB .jpg)
   - Verify PSG metadata is correct
   - Try different segmentation threshold

5. **Graph matching fails**
   - Ensure scene graph format matches expected input
   - Check that both prediction and ground truth graphs exist
   - Verify embedding model is available