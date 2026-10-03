# AI Usage

## How this document was produced

The decisions and reasoning here are mine. I explained them in conversation with Claude during
the project, then had Claude organize them into this format using the conversation history. I
reviewed the result against my notebook and training logs.

## Tools Used

- **Claude Code (claude-sonnet-4-6)**, VSCode extension: implementation, debugging,
  experiment documentation, and discussion of experiment design.

## Representative AI-Assisted Tasks

1. **Training loop and diagnostics.** Claude implemented `train_model`, `plot_history`, and
   `per_class_accuracy`. I checked them by running Experiment 0 and confirming the output
   matched the training logs (train loss ~0.12, val loss ~2.28 at the final epoch).
2. **Backbone freezing.** Claude implemented the `requires_grad` freezing pattern for ResNet18
   (layer4 + head) and ConvNeXt-Tiny (`features[6]`, `features[7]` + head). I checked the
   printed trainable-parameter counts before training.
3. **Debugging.** Claude diagnosed a `KeyError` from reading ImageNet normalisation constants
   out of torchvision's weights metadata and switched to the canonical hardcoded values. I
   confirmed training ran and the curves looked as expected.
4. **Experiment documentation.** Claude drafted `EXPERIMENTS.md` entries. I
   checked the factual entries against the logged numbers from each run.
5. **Submission notebook.** Claude assembled `submission.ipynb` as a clean sequential version
   of the experiments. I reviewed it before committing.

## Incorrect / Ineffective AI Suggestion

Claude proposed replacing the ConvNeXt linear head with
`Dropout(0.3) -> Linear(768,256) -> GELU -> Dropout(0.2) -> Linear(256,16)` (Exp 6) to add
non-linear capacity. I ran it instead of assuming it would help. Val accuracy dropped from
96.04% to 95.42% and val loss rose (0.173 vs 0.134) with train loss nearly unchanged,
consistent with the dropout over-regularising. This suggests the pretrained features are
already close to linearly separable, so head capacity is not the bottleneck. This is a single
seed and the accuracy gap is about 3 images, so I treat it as suggestive.

## Decisions I Made

- **ResNet18 before ConvNeXt:** a simple, well-understood pretrained model first, then a
  stronger backbone.
- **Frozen backbone first:** to measure how far a pretrained model with only a new head gets
  compared to TNet (a linear-probe baseline).
- **Unfreezing only the last stage:** a fully frozen backbone only re-interprets ImageNet
  features. Deeper layers encode more task-specific patterns while early layers (edges,
  textures) generalise. I did not test other depths.
- **Stopping after Exp 6:** Claude listed further experiments (discriminative learning rates,
  weight decay, label smoothing, higher resolution, larger variants). I chose not to run them
  because the six experiments already told a complete story. These remain untested.

## Summary

AI sped up implementation and debugging. The choice of experiments and the interpretation of
results were mine, and Claude helped by explaining tradeoffs and writing code.