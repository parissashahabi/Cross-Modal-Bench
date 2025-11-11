# Scene Graph Generation Pipeline

Generate structured scene graphs from images using panoptic segmentation and vision-language models.

**Pipeline:** Image → Segmentation (kMaX/FC-CLIP) → VLM (Qwen2-VL/GPT-4o) → Scene Graph JSON

---

## Quick Start
```bash
# 1. Setup environment
conda create -n cross-modal python=3.10 -y
conda activate cross-modal

# 2. Clone project
git clone <your-repo-url>
cd cross-modal-bench

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml pillow opencv-python numpy tqdm openai vllm
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 4. Setup segmentation models (see Installation section)
cd external/
git clone https://github.com/bytedance/kmax-deeplab.git kmax-deeplab
cd kmax-deeplab && pip install -r requirements.txt && cd ../..

# 5. Download weights (see Installation section)
mkdir -p models && cd models
pip install gdown
gdown https://drive.google.com/uc?id=1YB_5dct0U7ys2KTJNjDIqXLSZneWTyr-
cd ..

# 6. Configure (edit configs/config.yaml)
nano configs/config.yaml  # Update paths if needed

# 7. Run pipeline
export CUDA_VISIBLE_DEVICES=2  # Use free GPU
python main.py --image data/input/test.jpg
```

**Output:** `data/output/test_scene_graph.json`

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.10
- CUDA-capable GPU (8GB+ VRAM recommended)
- ~20GB disk space

### Step 1: Create Environment
```bash
conda create -n cross-modal python=3.10 -y
conda activate cross-modal
```

### Step 2: Install PyTorch
```bash
# Check your CUDA version
nvidia-smi  # Look for "CUDA Version: XX.X"

# For CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Core Dependencies
```bash
pip install pyyaml pillow opencv-python numpy tqdm openai vllm
```

### Step 4: Install Detectron2
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# If that fails, try:
# cd external/
# git clone https://github.com/facebookresearch/detectron2.git detectron2
# cd detectron2
# pip install --no-build-isolation -e .
# cd ../..
```

### Step 5: Install Segmentation Models

**Option A: kMaX-DeepLab (recommended for real-world images)**
```bash
cd external/
git clone https://github.com/bytedance/kmax-deeplab.git kmax-deeplab
cd kmax-deeplab
pip install -r requirements.txt
cd ../..
```

**Option B: FC-CLIP (for open-vocabulary/synthetic images)**
```bash
cd external/
git clone https://github.com/bytedance/fc-clip.git fc-clip
cd fc-clip
pip install -r requirements.txt
cd ../..
```

### Step 6: Download Model Weights
```bash
pip install gdown
mkdir -p models && cd models

# kMaX-DeepLab ResNet-50 (~180MB)
gdown https://drive.google.com/uc?id=1YB_5dct0U7ys2KTJNjDIqXLSZneWTyr-
# This downloads: kmax_r50.pth

# FC-CLIP ConvNeXt-Large (~700MB) - optional
gdown https://drive.google.com/uc?id=1-91PIns86vyNaL3CzMmDD39zKGnPMtvj
# This downloads: fcclip_convnext_large.pth (rename if needed)

cd ..
```

### Step 7: Verify Installation
```bash
python -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')

import detectron2
print('✓ Detectron2: OK')

from vllm import LLM
print('✓ VLLM: OK')

print('\n✅ All dependencies installed!')
"
```

---

## Configuration

**Single config file:** `configs/config.yaml`

### Default Configuration
```yaml
segmentation:
  model: 'kmax'  # or 'fcclip'
  
  # Paths (update these to match your setup)
  kmax_path: './external/kmax-deeplab'
  fcclip_path: './external/fc-clip'
  
  kmax_config_file: './external/kmax-deeplab/configs/coco/panoptic_segmentation/kmax_r50.yaml'
  kmax_weights: './models/kmax_r50.pth'
  
  fcclip_config_file: './external/fc-clip/configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml'
  fcclip_weights: './models/fcclip_convnext_large.pth'
  
  output_dir: './data/output'
  use_cuda: true

vlm:
  type: 'vllm'  # or 'gpt4o'
  
  # VLLM settings (local, no API costs)
  vllm_model: 'Qwen/Qwen2-VL-7B-Instruct'
  tensor_parallel_size: 1  # Number of GPUs
  
  # GPT-4o settings (if using OpenAI)
  gpt4o_api_key: 'your-api-key-here'
  gpt4o_model: 'gpt-4o'
  
  max_tokens: 2000

verbose: true
```

### Common Configurations

#### 1. Use kMaX + VLLM (Default, Recommended)
```yaml
segmentation:
  model: 'kmax'

vlm:
  type: 'vllm'
  vllm_model: 'Qwen/Qwen2-VL-7B-Instruct'
  tensor_parallel_size: 1
```

#### 2. Use FC-CLIP + VLLM (Open-vocabulary)
```yaml
segmentation:
  model: 'fcclip'  # Change this

vlm:
  type: 'vllm'
  vllm_model: 'Qwen/Qwen2-VL-7B-Instruct'
  tensor_parallel_size: 1
```

#### 3. Use kMaX + GPT-4o (API-based)
```yaml
segmentation:
  model: 'kmax'

vlm:
  type: 'gpt4o'  # Change this
  gpt4o_api_key: 'sk-your-actual-api-key'
```

#### 4. Use Larger VLLM Model
```yaml
segmentation:
  model: 'kmax'

vlm:
  type: 'vllm'
  vllm_model: 'Qwen/Qwen3-VL-30B-A3B-Instruct'  # Larger model
  tensor_parallel_size: 4  # More GPUs needed
```

### GPU Configuration
```bash
# Check free GPUs
nvidia-smi

# Use specific GPUs
export CUDA_VISIBLE_DEVICES=2,3,4,5

# For single GPU (7B model)
export CUDA_VISIBLE_DEVICES=2

# For multi-GPU (30B model)
export CUDA_VISIBLE_DEVICES=2,3,4,5
# Update config: tensor_parallel_size: 4
```

---

## Usage

### Basic Usage
```bash
# Set GPU
export CUDA_VISIBLE_DEVICES=2

# Generate scene graph
python main.py --image data/input/photo.jpg
```

**Output files:**
- `data/output/photo_kmax_seg.jpg` - Segmentation visualization
- `data/output/photo_scene_graph.json` - Scene graph

### With Options
```bash
# Specify output path
python main.py \
  --image data/input/photo.jpg \
  --output results/my_scene_graph.json

# Add caption for context
python main.py \
  --image data/input/street.jpg \
  --caption "Urban street with cars and buildings"

# Override segmentation model
python main.py \
  --image data/input/photo.jpg \
  --model fcclip

# Enable verbose output
python main.py \
  --image data/input/photo.jpg \
  --verbose
```

### Batch Processing
```bash
# Put all images in data/input/
cp /path/to/images/*.jpg data/input/

# Process all images
python examples/batch_process.py \
  --input_dir data/input \
  --output_dir data/output
```

### Command-Line Arguments
```
--image, -i      Path to input image (required)
--config, -c     Config file (default: configs/config.yaml)
--output, -o     Output path for scene graph JSON
--model, -m      Override segmentation model: 'kmax' or 'fcclip'
--caption        Optional text caption for context
--verbose, -v    Enable verbose output
```

---

## Output Format

Scene graphs are saved as JSON:
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
    {
      "id": "2",
      "object": "road",
      "attributes": {
        "surface": "asphalt"
      }
    }
  ],
  "edges": [
    {
      "subject": "1",
      "object": "2",
      "relation": "parked_on"
    }
  ],
  "metadata": {
    "image_path": "data/input/photo.jpg",
    "visualization_path": "data/output/photo_kmax_seg.jpg",
    "model": "KMaxSegmentor"
  }
}
```

---

## Troubleshooting

### Common Issues

#### "CUDA out of memory"

**Solutions:**
```bash
# 1. Use free GPU
export CUDA_VISIBLE_DEVICES=2

# 2. Use CPU mode
# Edit config: use_cuda: false

# 3. Use smaller model
# Edit config: vllm_model: 'Qwen/Qwen2-VL-7B-Instruct'
# Edit config: tensor_parallel_size: 1

# 4. Reduce image size
convert input.jpg -resize 1024x1024 input_small.jpg
```

#### "VLLM placement group error"

**Solution:**
```bash
# Use specific free GPUs
nvidia-smi  # Check which GPUs are free

export CUDA_VISIBLE_DEVICES=2,3,4,5

# Or use single GPU
export CUDA_VISIBLE_DEVICES=2
# Edit config: tensor_parallel_size: 1
```

#### "Detectron2 not found"

**Solution:**
```bash
# Reinstall
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Or add to PYTHONPATH
export PYTHONPATH="$PWD/external/detectron2:$PYTHONPATH"
```

#### "Model weights not found"

**Solution:**
```bash
# Check file exists
ls -lh models/kmax_r50.pth

# Re-download
cd models
gdown https://drive.google.com/uc?id=1YB_5dct0U7ys2KTJNjDIqXLSZneWTyr-
cd ..

# Update config with correct path
nano configs/config.yaml
```

#### "GPT-4o API quota exceeded"

**Solution:**
```bash
# Switch to VLLM (free, local)
# Edit configs/config.yaml:
vlm:
  type: 'vllm'  # Change from 'gpt4o' to 'vllm'
```

### Performance Optimization

**Slow inference?**
- ✅ Use GPU: `export CUDA_VISIBLE_DEVICES=2`
- ✅ Use free GPU only
- ✅ Batch process multiple images (faster)
- ✅ Use smaller model: `Qwen2-VL-7B-Instruct`

**Out of memory?**
- ✅ Single GPU: `tensor_parallel_size: 1`
- ✅ Smaller model: `Qwen2-VL-7B-Instruct`
- ✅ Reduce image size before processing
- ✅ CPU mode: `use_cuda: false`

---

## Model Selection

### Segmentation Models

| Model | Best For | Speed | Memory | Classes |
|-------|----------|-------|--------|---------|
| **kMaX-DeepLab** | Real-world images | Fast | ~2GB | COCO (80) |
| **FC-CLIP** | Synthetic images, rare objects | Medium | ~3GB | Open-vocab |

**When to use:**
- **kMaX-DeepLab**: Standard photos, common objects (cars, people, buildings)
- **FC-CLIP**: Unusual objects, synthetic images, open-vocabulary needs

### VLM Models

| Model | Size | Quality | Speed | GPUs | Cost |
|-------|------|---------|-------|------|------|
| **Qwen2-VL-7B** | 7B | Good | Fast | 1 | Free |
| Qwen3-VL-30B | 30B | Better | Medium | 2-4 | Free |
| GPT-4o | - | Best | Fast | 0 | API costs |

**When to use:**
- **Qwen2-VL-7B**: Default choice, good quality, single GPU
- **Qwen3-VL-30B**: Better quality, need multiple GPUs
- **GPT-4o**: Best quality, but requires API credits

---

## Project Structure
```
cross-modal-bench/
├── src/
│   ├── segmentation/       # Model wrappers
│   │   ├── base.py
│   │   ├── kmax_wrapper.py
│   │   └── fcclip_wrapper.py
│   ├── scene_graph/        # Core logic
│   │   ├── schema.py
│   │   └── generator.py
│   └── vlm/               # VLM clients
│       ├── gpt4o.py
│       └── vllm_client.py
├── external/              # External models
│   ├── kmax-deeplab/     # Clone here
│   ├── fc-clip/          # Clone here
│   └── detectron2/       # Clone here
├── models/               # Model weights
│   ├── kmax_r50.pth
│   └── fcclip_convnext_large.pth
├── configs/
│   └── config.yaml       # Single config file
├── data/
│   ├── input/           # Your images
│   └── output/          # Results
├── examples/
│   └── batch_process.py
├── main.py              # CLI entry point
└── README.md            # This file
```

---

## Examples

### Example 1: Basic Usage
```bash
export CUDA_VISIBLE_DEVICES=2
python main.py --image data/input/street.jpg
```

### Example 2: With Caption
```bash
python main.py \
  --image data/input/scene.jpg \
  --caption "A busy urban intersection with traffic"
```

### Example 3: Use FC-CLIP
```bash
# Edit config first
nano configs/config.yaml
# Change: model: 'fcclip'

python main.py --image data/input/unusual_objects.jpg
```

### Example 4: Batch Processing
```bash
python examples/batch_process.py \
  --input_dir data/input \
  --output_dir data/output
```

---

## Performance Benchmarks

**Expected timing (per image on H100 GPU):**
- Model loading: ~10-20 seconds (one-time)
- Segmentation: ~2-5 seconds
- VLM inference: ~5-15 seconds
- **Total: ~10-25 seconds per image**

**Batch processing is much faster:** Model loads once, then processes all images.

---

## Citation

Based on "Leveraging Panoptic Scene Graph for Evaluating Fine-Grained Text-to-Image Generation" (ICCV 2025)
```bibtex
@inproceedings{deng2025leveraging,
  title={Leveraging Panoptic Scene Graph for Evaluating Fine-Grained Text-to-Image Generation},
  author={Deng, Xueqing and Yang, Linjie and Yu, Qihang and Yang, Chenglin and Chen, Liang-Chieh},
  booktitle={ICCV},
  year={2025}
}
```

---

## Quick Reference
```bash
# Setup (one-time)
conda create -n cross-modal python=3.10 -y
conda activate cross-modal
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml pillow opencv-python numpy tqdm openai vllm
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install segmentation model
cd external/
git clone https://github.com/bytedance/kmax-deeplab.git kmax-deeplab
cd kmax-deeplab && pip install -r requirements.txt && cd ../..

# Download weights
mkdir -p models && cd models
pip install gdown
gdown https://drive.google.com/uc?id=1YB_5dct0U7ys2KTJNjDIqXLSZneWTyr-
cd ..

# Configure
nano configs/config.yaml  # Update paths if needed

# Run
export CUDA_VISIBLE_DEVICES=2
python main.py --image data/input/test.jpg

# View output
cat data/output/test_scene_graph.json
```

---

## Getting Help

1. **Read this README** - Most common issues are covered
2. **Check configuration** - `configs/config.yaml`
3. **Verify GPU** - `nvidia-smi`
4. **Check paths** - Ensure model weights and external repos exist

---

## Tips for Teammates

### First-time Setup Checklist

- [ ] Create conda environment with Python 3.10
- [ ] Install PyTorch with correct CUDA version
- [ ] Install core dependencies (yaml, pillow, etc.)
- [ ] Install detectron2
- [ ] Clone kMaX-DeepLab (or FC-CLIP) to `external/`
- [ ] Download model weights to `models/`
- [ ] Update paths in `configs/config.yaml`
- [ ] Test installation: `python -c "import torch; import detectron2; import vllm"`
- [ ] Run test: `python main.py --image data/input/test.jpg`

### Daily Usage
```bash
# 1. Activate environment
conda activate cross-modal

# 2. Set GPU
export CUDA_VISIBLE_DEVICES=2

# 3. Run
python main.py --image your_image.jpg

# Done!
```

---

**That's it! You're ready to generate scene graphs.** 🎉

For questions or issues, check the [Troubleshooting](#troubleshooting) section above.