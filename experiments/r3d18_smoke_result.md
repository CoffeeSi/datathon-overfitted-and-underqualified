# R3D-18 baseline acceptance run

Executed locally on 2026-10-17, on NVIDIA GeForce RTX 4050 Laptop GPU.
This is a real-video one-epoch acceptance experiment, separate from the earlier
synthetic Brev L4 environment check.

## Configuration and data

- Base configuration: [configs/r3d18.yaml](../configs/r3d18.yaml).
- Overrides: `epochs=1`, output directories `runs/r3d18_smoke` and
  `runs/r3d18_smoke_repeat` for two independent process runs.
- Python 3.14.5, torch 2.13.0+cu126, torchvision 0.28.0+cu126, CUDA runtime 12.6.
- Seed 42, frozen pretrained backbone, binary head, batch size 2, AdamW lr 0.001.
- 4-second windows, 16 frames, the pretrained weights' own transform.
- Fixed 30/10 canonical video split: 31 train windows (17 positive, 14 negative)
  and 10 validation windows (5 positive, 5 negative).
- X-4/X-47 contributes two source events to one canonical training video.
- Source/split hashes and all window assignments are saved in each run's manifest.

## Observed results

Train: loss 0.9569284623, accuracy 0.483871, precision 0.521739,
recall 0.705882, F1 0.600000.

Validation: loss 0.7603230059, accuracy 0.500000, precision 0.500000,
recall 1.000000, F1 0.666667. Confusion matrix, true labels by rows and predicted
labels by columns in `[no_violation, violation]` order: `[[0, 5], [0, 5]]`.

The model predicted violation for every validation window. This is no improvement
over an always-positive predictor on this validation set. The run demonstrates a
working training/checkpoint pipeline, not satisfactory detection quality.

## Acceptance checks

- All 13 EDA and ML tests passed.
- Both runs produced identical metric histories and bitwise-identical model tensors.
- All non-head tensors, including BatchNorm buffers, equal the original pretrained state.
- Only `fc.weight` and `fc.bias` are trainable. The optimizer recorded 16 steps
  for 31 train windows at batch size 2; validation adds no optimizer steps.
- `best.pt` loaded successfully with `weights_only=True` and reproduced the saved
  validation metrics on GPU.
- Effective config, environment, data manifest, metric history, predictions,
  `best.pt` and `last.pt` were saved under each output directory.

Weights and run outputs are local and ignored by Git. The code, configuration and
this report are versionable. Reproduction commands and output descriptions are in
[docs/r3d18-baseline.md](../docs/r3d18-baseline.md).

These are annotation-centred window metrics on a small within-room validation set.
They do not measure event localization or independent-session generalization.
The baseline implementation acceptance is satisfied; further experiments are needed
to obtain a useful classifier and full-video inference.
