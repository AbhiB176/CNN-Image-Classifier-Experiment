# CNN Image Classifier — ITCS 6169/8169 Assignment 1

16-class scene recognition using transfer learning and partial fine-tuning on 2,400 training images.

**Final result:** ConvNeXt-Tiny partial fine-tune · 96.04% val · 96.50% test

## Repository contents

| File | Contents |
|---|---|
| `submission.ipynb` | Full experiment notebook — setup, Experiments 0–6, final test evaluation, ablation summary |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1

# GPU (CUDA 12.x — check your version with nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Data

Download the dataset from the assignment handout and place it as:

```
data/
  train/
    <class_name>/
    ...
  test/
    <class_name>/
    ...
```

Update `PROJECT_ROOT` in the first setup cell of `submission.ipynb` to point to your directory.

## Reproducing results

Open `submission.ipynb` and run cells top to bottom. All experiments are defined sequentially —
transforms and data loaders are created once at the top, then each experiment uses them directly.
Expected runtime on an RTX 4060: ~15 min for the full run (Exp 4 and 5 dominate at ~8 min each).

## Experiment summary

| Configuration | Val Acc. |
|---|---:|
| Baseline CNN (TNet, from scratch) | 49.79% |
| + Pretrained backbone (ResNet18 frozen) | 90.21% |
| + Partial fine-tune + training strategy | 94.37% |
| + ConvNeXt-Tiny backbone (final model) | **96.04%** |
| Final model — test set | **96.50%** |

## Model checkpoint

The final model weights (`convnext_final.pth`) are too large for GitHub. Download from:

> **[convnext_final.pth — Google Drive](LINK_HERE)**

To load:
```python
import torch, torch.nn as nn
from torchvision.models import convnext_tiny

model = convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(768, 16)
model.load_state_dict(torch.load('convnext_final.pth', map_location='cpu'))
model.eval()
```

## Environment

- Python 3.11 · PyTorch 2.6 + CUDA 12.4
- NVIDIA GeForce RTX 4060 Laptop GPU
- SEED = 0
