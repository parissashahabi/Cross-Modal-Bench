# Step-by-Step Usage Guide

This guide walks you through using the scene graph generation pipeline from start to finish.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.8+ installed
- [ ] Git installed
- [ ] (Optional) NVIDIA GPU with CUDA
- [ ] OpenAI API account with GPT-4o access
- [ ] At least 10GB free disk space

## Part 1: Installation (30-60 minutes)

### Step 1.1: Set Up Project Directory

```bash
# Navigate to where you want the project
cd ~/projects  # or your preferred location

# If you received the project as a ZIP
unzip cross-modal-bench.zip
cd cross-modal-bench

# OR if cloning from Git
git clone <your-repo-url> cross-modal-bench
cd cross-modal-bench
```

### Step 1.2: Create Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Verify activation (should show venv in prompt)
which python  # Should point to venv/bin/python
```

### Step 1.3: Install Base Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# This installs:
# - openai (for GPT-4o)
# - pyyaml (for config)
# - pillow, opencv-python (for images)
# - torch, torchvision (deep learning)
# - numpy, tqdm (utilities)
```

**Expected time**: 5-10 minutes

### Step 1.4: Install Detectron2

```bash
# For CUDA 11.8 (check your CUDA version with: nvcc --version)
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# For CPU only
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html

# OR install from source (if above fails)
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**Expected time**: 5-10 minutes

**Verify installation:**
```bash
python -c "import detectron2; print('✓ Detectron2 installed')"
```

### Step 1.5: Install Segmentation Model (kMaX-DeepLab)

```bash
# Create external directory
mkdir -p external
cd external

# Clone kMaX-DeepLab
git clone https://github.com/bytedance/kmax-deeplab.git kmax-deeplab

# Install dependencies
cd kmax-deeplab
pip install -r requirements.txt

# Go back to project root
cd ../..
```

**Expected time**: 2-5 minutes

### Step 1.6: Download Model Weights

```bash
# Install gdown for Google Drive downloads
pip install gdown

# Create models directory
mkdir -p models
cd models

# Download ResNet-50 weights (recommended for testing)
gdown https://drive.google.com/uc?id=1YB_5dct0U7ys2KTJNjDIqXLSZneWTyr-
# This downloads kmax_r50.pth (~180MB)

# OR for better quality, download ConvNeXt-Large
# gdown https://drive.google.com/uc?id=1b6rEnKw4PNTdqSdWpmb0P9dsvN0pkOiN

cd ..
```

**Expected time**: 2-5 minutes (depending on internet speed)

**Verify download:**
```bash
ls -lh models/kmax_r50.pth
# Should show ~180MB file
```

### Step 1.7: Configure OpenAI API Key

```bash
# Option 1: Set environment variable (recommended)
export OPENAI_API_KEY='sk-your-actual-api-key-here'

# To make it permanent, add to ~/.bashrc or ~/.zshrc:
echo 'export OPENAI_API_KEY="sk-your-actual-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# Option 2: Edit config file
nano configs/default.yaml
# Update the api_key field
```

**Get API key**: https://platform.openai.com/api-keys

### Step 1.8: Update Configuration

```bash
# Open config file
nano configs/default.yaml

# Update these paths:
# - kmax_path: './external/kmax-deeplab'
# - config_file: './external/kmax-deeplab/configs/coco/panoptic_segmentation/kmax_r50.yaml'
# - weights: './models/kmax_r50.pth'
# - api_key: (if not using environment variable)

# Save and exit (Ctrl+X, Y, Enter in nano)
```

### Step 1.9: Verify Installation

```bash
# Run verification script
python -c "
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')

import detectron2
print(f'✓ Detectron2 installed')

from openai import OpenAI
print(f'✓ OpenAI library installed')

import sys
from pathlib import Path
sys.path.insert(0, 'src')
from scene_graph import SceneGraphGenerator
print(f'✓ Project modules loaded')

print('\n✅ Installation complete!')
"
```

**Expected output:**
```
✓ PyTorch 2.x.x
✓ CUDA available: True
✓ Detectron2 installed
✓ OpenAI library installed
✓ Project modules loaded

✅ Installation complete!
```

## Part 2: First Run (5-10 minutes)

### Step 2.1: Prepare Test Image

```bash
# Create input directory
mkdir -p data/input

# Option 1: Use your own image
cp /path/to/your/photo.jpg data/input/test.jpg

# Option 2: Download sample image
wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/demo/input.jpg \
  -O data/input/test.jpg

# Option 3: Use any image from the internet
# Just make sure it's a .jpg or .png file
```

### Step 2.2: Run Your First Scene Graph Generation

```bash
# Basic command
python main.py --image data/input/test.jpg --verbose

# Watch the progress:
# [1/3] Running panoptic segmentation...
# [2/3] Extracting scene graph with GPT-4o...
# [3/3] Structuring scene graph...
```

**Expected output:**
```
============================================================
Generating Scene Graph
============================================================
Image: data/input/test.jpg

[1/3] Running panoptic segmentation...
✓ kMaX-DeepLab model loaded successfully
✓ Segmentation complete: 15 segments detected
✓ Visualization saved to: data/output/test_kmax_seg.jpg

[2/3] Extracting scene graph with GPT-4o...

[3/3] Structuring scene graph...
✓ Scene graph saved to: data/output/test_scene_graph.json

============================================================
✓ Scene Graph Generation Complete
  Time: 18.45s
  Nodes: 15
  Edges: 12
============================================================

✓ Success! Scene graph saved to: data/output/test_scene_graph.json

Scene Graph Summary:
  Objects: 15
  Relationships: 12

  Sample objects:
    - car (color:blue, type:sedan)
    - person (position:standing, clothing:casual)
    - building (material:brick, size:large)
```

**Time breakdown:**
- Model loading: ~5-10 seconds (first time only)
- Segmentation: ~2-5 seconds
- GPT-4o API call: ~5-15 seconds
- **Total: ~10-25 seconds**

### Step 2.3: View Results

```bash
# View the segmentation visualization
# (Open in your image viewer)
open data/output/test_kmax_seg.jpg  # Mac
# OR
xdg-open data/output/test_kmax_seg.jpg  # Linux
# OR
start data/output/test_kmax_seg.jpg  # Windows

# View the scene graph JSON
cat data/output/test_scene_graph.json

# Or pretty-print it
python -m json.tool data/output/test_scene_graph.json
```

**Scene graph structure:**
```json
{
  "nodes": [
    {
      "id": "1",
      "object": "car",
      "description": "A blue sedan",
      "attributes": {
        "color": "blue",
        "type": "sedan"
      }
    },
    ...
  ],
  "edges": [
    {
      "subject": "1",
      "object": "2",
      "relation": "parked_on"
    },
    ...
  ],
  "metadata": {
    "image_path": "data/input/test.jpg",
    "model": "KMaxSegmentor"
  }
}
```

## Part 3: Common Use Cases

### Use Case 1: Process Multiple Images

```bash
# Put all images in input directory
cp /path/to/photos/*.jpg data/input/

# Run batch processing
python examples/batch_process.py \
  --input_dir data/input \
  --output_dir data/output \
  --skip_existing

# Results will be in data/output/
ls data/output/
```

### Use Case 2: Add Context with Caption

```bash
# Providing a caption helps GPT-4o understand the scene better
python main.py \
  --image data/input/street.jpg \
  --caption "A busy urban street with cars, pedestrians, and buildings" \
  --output data/output/street_graph.json
```

### Use Case 3: Use Different Model (FC-CLIP)

First, install FC-CLIP:

```bash
cd external/
git clone https://github.com/bytedance/fc-clip.git fc-clip
cd fc-clip
pip install -r requirements.txt
cd ../..

# Download weights
cd models
gdown https://drive.google.com/uc?id=1-91PIns86vyNaL3CzMmDD39zKGnPMtvj
cd ..

# Update config to use FC-CLIP
# Edit configs/default.yaml:
# - model: 'fcclip'
# - config_file: './external/fc-clip/configs/...'
# - weights: './models/fcclip_convnext_large.pth'

# Run with FC-CLIP
python main.py --image data/input/test.jpg --model fcclip
```

### Use Case 4: Custom Output Location

```bash
# Specify exact output path
python main.py \
  --image data/input/photo.jpg \
  --output results/my_analysis/photo_scene_graph.json

# The directory will be created if it doesn't exist
```

### Use Case 5: Programmatic Use

Create `my_script.py`:

```python
import sys
sys.path.insert(0, 'src')

from scene_graph import SceneGraphGenerator
import yaml

# Load config
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create generator
generator = SceneGraphGenerator.from_config(config)

# Generate scene graph
scene_graph = generator.generate(
    image_path='data/input/test.jpg',
    caption='My test image',
    output_path='results/test_graph.json'
)

# Access data programmatically
print(f"Found {len(scene_graph.nodes)} objects")
for node in scene_graph.nodes:
    print(f"  - {node.object_class}: {node.attributes}")

print(f"\nFound {len(scene_graph.edges)} relationships")
for edge in scene_graph.edges:
    print(f"  - {edge.subject_id} {edge.relation} {edge.object_id}")
```

Run it:
```bash
python my_script.py
```

## Part 4: Troubleshooting

### Problem: "CUDA out of memory"

**Solution 1**: Use CPU mode
```bash
# Edit configs/default.yaml:
# use_cuda: false

python main.py --image data/input/test.jpg
```

**Solution 2**: Use smaller model
```bash
# Use ResNet-50 instead of ConvNeXt-Large
# Update config weights to kmax_r50.pth
```

**Solution 3**: Reduce image size
```bash
# Resize image before processing
convert input.jpg -resize 1024x1024 input_small.jpg
```

### Problem: "API key not configured"

**Check:**
```bash
# Verify environment variable
echo $OPENAI_API_KEY

# If empty, set it
export OPENAI_API_KEY='sk-your-key'

# Or pass via command line
python main.py --image test.jpg --api-key 'sk-your-key'
```

### Problem: "Model weights not found"

**Check file exists:**
```bash
ls -lh models/kmax_r50.pth
```

**If missing:**
```bash
cd models
gdown https://drive.google.com/uc?id=1YB_5dct0U7ys2KTJNjDIqXLSZneWTyr-
cd ..
```

**Update config with correct path:**
```bash
# Edit configs/default.yaml
# weights: './models/kmax_r50.pth'  # Check this path
```

### Problem: "GPT-4o returns invalid JSON"

This happens ~5% of the time (per paper).

**Solution:**
```bash
# Just re-run the same image
python main.py --image data/input/test.jpg

# The code includes automatic JSON cleaning
# Most cases will succeed on retry
```

### Problem: "Detectron2 import error"

**Reinstall detectron2:**
```bash
pip uninstall detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Part 5: Next Steps

### Explore Your Data

```bash
# Count total objects across all images
python -c "
import json
from pathlib import Path

total_nodes = 0
total_edges = 0

for file in Path('data/output').glob('*_scene_graph.json'):
    with open(file) as f:
        sg = json.load(f)
        total_nodes += len(sg['nodes'])
        total_edges += len(sg['edges'])

print(f'Total objects: {total_nodes}')
print(f'Total relationships: {total_edges}')
"
```

### Visualize Scene Graphs

You can use tools like:
- Graphviz
- NetworkX
- Neo4j
- Gephi

Example with NetworkX:
```python
import json
import networkx as nx
import matplotlib.pyplot as plt

# Load scene graph
with open('data/output/test_scene_graph.json') as f:
    sg = json.load(f)

# Create graph
G = nx.DiGraph()

# Add nodes
for node in sg['nodes']:
    G.add_node(node['id'], label=node['object'])

# Add edges
for edge in sg['edges']:
    G.add_edge(edge['subject'], edge['object'], label=edge['relation'])

# Draw
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=2000, font_size=10, arrows=True)
plt.savefig('scene_graph_visualization.png')
print("Saved visualization to scene_graph_visualization.png")
```

### Integrate into Your Workflow

The pipeline is modular, so you can:
1. Use just the segmentation part
2. Replace GPT-4o with another VLM
3. Add custom post-processing
4. Integrate with your existing codebase

### Move to Production

For production use:
1. Add proper logging
2. Implement retry logic for API calls
3. Add caching for repeated images
4. Set up monitoring
5. Consider Docker deployment

## Part 6: Quick Reference

### Key Commands

```bash
# Basic usage
python main.py --image <image_path>

# With options
python main.py --image <image_path> --caption "..." --output <output_path>

# Batch processing
python examples/batch_process.py --input_dir <dir> --output_dir <dir>

# Get help
python main.py --help
```

### Key Files

- `configs/default.yaml` - Main configuration
- `data/input/` - Place images here
- `data/output/` - Find results here
- `main.py` - CLI entry point
- `src/scene_graph/generator.py` - Main pipeline logic

### Useful Tips

1. **Reuse loaded model**: Batch processing is much faster
2. **Use GPU**: 10-20x speedup
3. **Provide captions**: Better scene understanding
4. **Check visualizations**: Verify segmentation quality
5. **Save API responses**: Avoid redundant API calls

## Summary

You now know how to:
- ✅ Install and configure the pipeline
- ✅ Generate scene graphs from images
- ✅ Handle common issues
- ✅ Use advanced features
- ✅ Integrate into your workflow

For more details, see:
- `README.md` - Full documentation
- `ARCHITECTURE.md` - System design
- `SETUP_GUIDE.md` - Detailed installation

Happy scene graph generation! 🎉
