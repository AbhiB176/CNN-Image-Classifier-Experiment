# Experiment Log

All experiments use SEED=0, 80/20 train/val split, CrossEntropyLoss unless noted.
Test set is not evaluated until the final model is selected.

---

## Experiment 0: Baseline TNet

**Hypothesis:** Establish a sanity-check floor — confirm the pipeline works and get a starting val accuracy to beat.

**Change from previous:** N/A — starter model as provided.

**Config:**
- Model: TNet (1 Conv layer, 16 channels, kernel=3, MaxPool 4×4)
- Input: grayscale, 64×64
- Optimizer: Adam, lr=0.002
- Epochs: 20, batch size: 64

**Result:**
- Best val accuracy: 49.79%
- Final epoch: train loss 0.1181, val loss 2.2755, val acc 0.4667
- Train loss: smooth L-curve descent to near zero
- Val loss: plateaued then rose from ~epoch 13 onward
- Val accuracy: plateaued ~epoch 12–15, never recovered

**Plot:** `plots/exp0_baseline.png`

**Conclusion:** Severe overfitting despite a weak model. No regularization + Adam at lr=0.002 is sufficient to memorize 1,920 training images even with a 1-layer CNN. Model ceiling is too low to fix with regularization alone — move to pretrained ResNet18.

---

---

## Experiment 1: ResNet18 Feature Extraction (Frozen Backbone)

**Hypothesis:** Pretrained ImageNet features redirected to 16 classes via a new head should substantially outperform TNet, establishing how much transfer learning alone is worth.

**Change from previous:** Model swapped to ResNet18 pretrained on ImageNet-1K. All conv layers frozen — only `Linear(512, 16)` head trained. Input changed to RGB 224×224 with ImageNet normalization.

**Config:**
- Backbone: ResNet18 (frozen), ~8k trainable params out of 11M
- Optimizer: Adam, lr=1e-3, head only
- Epochs: 20, batch size: 64

**Result:**
- Best val accuracy: 90.21%
- Final epoch: train loss 0.1838, val loss 0.3267, val acc 0.8833
- Both loss curves smooth, close together, val loss diverges slightly from train around epoch 3
- Max acc > final acc — best checkpoint was earlier, mild regression at end

**Plot:** `plots/exp1_resnet18_frozen.png`

**Conclusion:** Massive gain from transfer learning alone (+40 points). Fit is healthy — no severe overfitting. The small train/val gap with only 8k trainable params means regularization on the head won't move the needle. Bottleneck is now representational: a single linear layer doing linear separation of ImageNet features not optimized for scenes.

<!-- Add new experiments below following the same template -->

---

## Final Ablation Summary

> Filled in at the end of the project.

| Stage | What changed | Val Accuracy | Observation |
|---|---|---|---|
| Baseline CNN (TNet) | Starter model | -- | -- |
| | | | |
| Final model | All components | -- | -- |
