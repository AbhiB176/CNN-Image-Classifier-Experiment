# CNN Image Classifier — ITCS 6169/8169 Assignment 1

16-class scene recognition using a CNN trained on 2,400 images.

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

## Running

Open `ITCS_6169_8169_Assignment1_2026_Starter.ipynb` and run cells top to bottom.
Each experiment section is self-contained. The `plot_history()` utility is defined once and reused after every training run.

## Documentation

| File | Contents |
|---|---|
| `EXPERIMENTS.md` | Structured log of every experiment — hypothesis, config, results |
| `NARRATIVE.md` | Decision rationale, failure analysis, project arc |
| `AI_USAGE.md` | Human-AI collaboration record (assignment requirement) |

## Environment

- Python 3.x
- PyTorch 2.x + CUDA 12.x
- NVIDIA GeForce RTX 4060
- SEED = 0
