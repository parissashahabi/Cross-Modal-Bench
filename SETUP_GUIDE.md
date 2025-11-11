# Detailed Setup Guide

This guide provides step-by-step instructions for setting up the Cross-Modal Bench scene graph generation pipeline.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Testing](#testing)
5. [Common Issues](#common-issues)

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
  - For kMaX-DeepLab: GTX 1080 Ti or better
  - For FC-CLIP: RTX 2080 or better
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and dependencies

### Software
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.3+ (if using GPU)
- **Git**: For cloning repositories

## Installation Steps

### 1. Set Up Base Environment

```bash
# Clone or navigate to project
cd /path/to/cross-modal-bench

# Create Python virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 2. Install Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Contents of requirements.txt:**
- openai>=1.0.0
- pyyaml>=6.0
- pillow>=10.0.0
- opencv-python>=4.8.0
- numpy>=1.24.0
- torch>=2.0.0
- torchvision>=0.15.0
- tqdm>=4.65.0

### 3. Install Detectron2

Detectron2 is required by both kMaX-DeepLab and FC-CLIP.

**For Linux with CUDA 11.8:**
```bash
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

**For other CUDA versions or CPU:**
Visit: https://detectron2.readthedocs.io/en/latest/tutorials/install.html

**From source (if above fails):**
```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

### 4. Install Segmentation Models

#### kMaX-DeepLab

```bash
cd external/

# Clone repository
git clone https://github.com/bytedance/kmax-deeplab.git kmax-deeplab

# Install dependencies
cd kmax-deeplab
pip install -r requirements.txt
cd ../..
```

**Download Model Weights:**

1. Go to: https://github.com/bytedance/kmax-deeplab#model-zoo
2. Choose a model (e.g., ResNet-50 for COCO)
3. Download the checkpoint

Using `gdown`:
```bash
pip install gdown
mkdir -p models

# ResNet-50 COCO (53.3 PQ)
gdown https://drive.google.com/uc?id=1YB_5dct0U7ys2KTJNjDIqXLSZneWTyr- \
  -O models/kmax_r50.pth

# ConvNeXt-Large COCO (57.9 PQ) - better quality
gdown https://drive.google.com/uc?id=1b6rEnKw4PNTdqSdWpmb0P9dsvN0pkOiN \
  -O models/kmax_convnext_large.pth
```

#### FC-CLIP (Optional)

```bash
cd external/

# Clone repository
git clone https://github.com/bytedance/fc-clip.git fc-clip

# Install dependencies
cd fc-clip
pip install -r requirements.txt
cd ../..
```

**Download Model Weights:**

1. Go to: https://github.com/bytedance/fc-clip#model-zoo
2. Choose a model

Using `gdown`:
```bash
# ConvNeXt-Large (26.8 PQ on ADE20K)
gdown https://drive.google.com/uc?id=1-91PIns86vyNaL3CzMmDD39zKGnPMtvj \
  -O models/fcclip_convnext_large.pth
```

### 5. Configure OpenAI API

You need a GPT-4o API key from OpenAI: https://platform.openai.com/api-keys

**Method 1: Environment Variable (Recommended)**
```bash
# Add to ~/.bashrc or ~/.zshrc for persistence
export OPENAI_API_KEY='sk-your-actual-key-here'

# Or set temporarily
export OPENAI_API_KEY='sk-your-actual-key-here'
```

**Method 2: Config File**

Edit `configs/default.yaml`:
```yaml
vlm:
  api_key: 'sk-your-actual-key-here'
```

## Configuration

### Update Config File

Edit `configs/default.yaml` with your setup:

```yaml
segmentation:
  # Choose model: 'kmax' or 'fcclip'
  model: 'kmax'
  
  # Paths to repositories
  kmax_path: './external/kmax-deeplab'
  fcclip_path: './external/fc-clip'
  
  # Model config file
  config_file: './external/kmax-deeplab/configs/coco/panoptic_segmentation/kmax_r50.yaml'
  
  # Model weights
  weights: './models/kmax_r50.pth'
  
  # Output directory
  output_dir: './data/output'
  
  # Use GPU (set false for CPU)
  use_cuda: true

vlm:
  api_key: 'your-api-key-here'
  model: 'gpt-4o'
  max_tokens: 2000

verbose: true
```

### Important Path Notes

- **Relative paths**: Start from project root
- **Absolute paths**: Full system paths (e.g., `/home/user/...`)
- **Config file**: Must match the model architecture
  - ResNet-50: `kmax_r50.yaml`
  - ConvNeXt-Large: `kmax_convnext_large.yaml`

## Testing

### Verify Installation

```bash
python -c "
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')

import detectron2
print(f'✓ Detectron2 installed')

from openai import OpenAI
print(f'✓ OpenAI library installed')

print('\n✓ All core dependencies OK!')
"
```

### Test Segmentation Model

```bash
cd external/kmax-deeplab

# Download a test image
wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/demo/input.jpg

# Test inference
python demo/demo.py \
  --config-file configs/coco/panoptic_segmentation/kmax_r50.yaml \
  --input input.jpg \
  --output output/ \
  --opts MODEL.WEIGHTS ../../models/kmax_r50.pth

cd ../..
```

### Test Full Pipeline

```bash
# Place a test image
mkdir -p data/input
cp /path/to/test.jpg data/input/

# Run pipeline
python main.py --image data/input/test.jpg --verbose

# Check outputs
ls data/output/
# Should see:
#   - test_kmax_seg.jpg (visualization)
#   - test_scene_graph.json (scene graph)
```

### Validate Output

```bash
# Check scene graph structure
python -c "
import json
with open('data/output/test_scene_graph.json', 'r') as f:
    sg = json.load(f)
print(f'Nodes: {len(sg[\"nodes\"])}')
print(f'Edges: {len(sg[\"edges\"])}')
print('✓ Scene graph valid')
"
```

## Common Issues

### Issue: ImportError: No module named 'detectron2'

**Solution:**
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Issue: CUDA out of memory

**Solutions:**
1. Use smaller model (ResNet-50 instead of ConvNeXt-Large)
2. Reduce image size
3. Use CPU mode: set `use_cuda: false` in config

### Issue: Model weights not loading

**Check:**
1. Weights file exists: `ls -lh models/kmax_r50.pth`
2. Config file path matches architecture
3. Use absolute paths if relative paths fail

### Issue: OpenAI API errors

**Check:**
1. API key is valid: `echo $OPENAI_API_KEY`
2. You have GPT-4o access (not all accounts have it)
3. Check API usage/billing: https://platform.openai.com/usage

### Issue: Segmentation visualization not created

**Debug:**
```bash
# Test segmentation directly
cd external/kmax-deeplab
python demo/demo.py \
  --config-file configs/coco/panoptic_segmentation/kmax_r50.yaml \
  --input /path/to/image.jpg \
  --output test_output/ \
  --opts MODEL.WEIGHTS ../../models/kmax_r50.pth
```

### Issue: JSON parsing errors from GPT-4o

This is expected ~5% of the time. The code includes retry logic.

**Manual fix:**
1. Check visualization: `data/output/*_seg.jpg`
2. If segmentation looks good, just re-run
3. If segmentation is bad, try different model or image

## Performance Optimization

### Speed Tips

1. **Use GPU**: 10-20x faster than CPU
2. **Batch processing**: Process multiple images in one run
3. **Smaller models**: ResNet-50 vs ConvNeXt-Large
4. **Reduce resolution**: Resize large images before processing

### Quality Tips

1. **Better segmentation**:
   - Use ConvNeXt-Large for higher quality
   - Use FC-CLIP for open-vocabulary scenarios
2. **Better scene graphs**:
   - Provide text captions with `--caption`
   - Use high-quality input images
   - Adjust GPT-4o prompt in `src/vlm/gpt4o.py`

## Next Steps

1. ✓ Installation complete
2. → Process your own images
3. → Explore batch processing (`examples/batch_process.py`)
4. → Integrate into your workflow
5. → Extend for T2I evaluation (future work)

## Getting Help

- **Documentation**: See `README.md`
- **Quick start**: See `QUICKSTART.md`
- **Examples**: See `examples/` directory
- **Issues**: Check GitHub issues or create new one

## Verification Checklist

Before proceeding, verify:

- [ ] Python environment activated
- [ ] All pip packages installed
- [ ] Detectron2 installed
- [ ] kMaX-DeepLab or FC-CLIP cloned
- [ ] Model weights downloaded
- [ ] OpenAI API key configured
- [ ] Config file updated with correct paths
- [ ] Test image processed successfully
- [ ] Scene graph JSON created

If all checked, you're ready to go! 🚀
