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

---

## Experiment 2: ResNet18 Frozen + Data Augmentation

**Hypothesis:** Augmenting the training set exposes the classifier head to more varied scene views, improving generalization without touching the frozen backbone.

**Change from Exp 1:** Train transform only — added `RandomResizedCrop(224, scale=0.7–1.0)`, `RandomHorizontalFlip(p=0.5)`, `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)`. Val/test transforms unchanged.

**Config:**
- Backbone: ResNet18 frozen (identical to Exp 1)
- Optimizer: Adam, lr=1e-3, head only
- Epochs: 20, batch size: 64

**Result:**
- Best val accuracy: 89.17% (vs 90.21% in Exp 1 — slight regression)
- Final epoch: train loss 0.2398, val loss 0.3348, val acc 0.8833
- Val loss diverges from train later (~epoch 8 vs ~epoch 3 in Exp 1)
- Train loss higher than Exp 1 (expected — augmentation makes training harder)

**Plot:** `plots/exp2_resnet18_aug.png`

**Conclusion:** Augmentation slightly hurt on a frozen backbone. The head sees harder, more varied inputs (random crops, color shifts) but the backbone can't adapt its feature extraction to match — it always uses the same fixed weights. Augmented views produce noisier, less consistent 512-dim feature vectors, making the linear classification problem harder without giving the model a way to compensate.

---

## Experiment 3: ResNet18 Frozen + Cosine LR Scheduling

**Hypothesis:** Flat LR in Exp 1 caused late-epoch oscillation (best acc at ~epoch 13, regression after). Cosine decay should smooth final convergence and reduce that regression.

**Change from Exp 1:** Added `CosineAnnealingLR(T_max=20)`. No augmentation — scheduler effect isolated.

**Config:**
- Backbone: ResNet18 frozen (identical to Exp 1)
- Optimizer: Adam, lr=1e-3 → cosine decay to ~0
- Scheduler: CosineAnnealingLR, T_max=20
- Epochs: 20, batch size: 64

**Result:**
- Best val accuracy: 89.58% (vs 90.21% in Exp 1 — slight regression)
- Final epoch: train loss 0.2624, val loss 0.3562, val acc 0.8917
- Val loss diverges from train earlier (~epoch 6 vs ~epoch 3 in Exp 1)
- Train loss higher than Exp 1 despite same data — cosine decay reduced LR before full convergence

**Plot:** `plots/exp3_resnet18_cosine.png`

**Conclusion:** Cosine scheduling slightly hurt on a frozen backbone. The linear classification problem the head solves is simple and fast-converging — Adam already finds a good solution without scheduling help. Decaying the LR constrains the optimizer before it reaches the optimum, slightly underfitting. Scheduling is more valuable when the loss landscape is complex (i.e., when fine-tuning a full network).

---

## Experiment 4: ResNet18 Partial Fine-Tune (layer4 + head)

**Hypothesis:** Unfreezing layer4 lets the highest-level features adapt toward scene discrimination. Augmentation and cosine scheduling should now help since the backbone can respond to varied inputs.

**Changes from Exp 1:** layer4 + head trainable (~2.6M params), layers 1–3 frozen. lr=1e-4 (all trainable params). Augmented train transform. CosineAnnealingLR. 30 epochs.

**Config:**
- Backbone: ResNet18, layer4 + fc trainable (23.2% of params)
- Optimizer: Adam, lr=1e-4
- Scheduler: CosineAnnealingLR, T_max=30
- Epochs: 30, batch size: 64, augmented train data

**Result:**
- Best val accuracy: 94.37%  (+4.16 points over Exp 1)
- Final epoch: train loss 0.0120, val loss 0.2144, val acc 0.9417
- Val loss higher than train from epoch 1 — normal with pretrained weights (model arrives already good)
- Val accuracy plateaus quickly (~epoch 8–10), consistent with fast layer4 convergence + cosine decay reducing LR early

**Plot:** `plots/exp4_resnet18_partial_finetune.png`

**Conclusion:** Confirmed that partial fine-tuning breaks the frozen backbone ceiling. Layer4 adapted quickly — the pretrained weights were already close to useful for scenes, needing only small adjustments. Augmentation and cosine scheduling helped here (unlike on frozen backbone) because the backbone could actually respond to varied inputs. Train loss reached near-zero (0.012) with augmentation, indicating layer4 thoroughly adapted to the training distribution.

<!-- Add new experiments below following the same template -->

---

## Final Ablation Summary

> Filled in at the end of the project.

| Stage | What changed | Val Accuracy | Observation |
|---|---|---|---|
| Baseline CNN (TNet) | Starter model | -- | -- |
| | | | |
| Final model | All components | -- | -- |
