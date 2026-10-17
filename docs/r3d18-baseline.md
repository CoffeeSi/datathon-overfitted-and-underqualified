# R3D-18 baseline

## Video-level quality evaluation

Evaluate an existing checkpoint without training or downloading pretrained weights:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m overfitted_underqualified.ml.evaluate --checkpoint runs/r3d18_smoke/best.pt --output-dir reports/r3d18_new_evaluation --device cuda
```

Linux/Brev: use `PYTHONPATH=src python -m overfitted_underqualified.ml.evaluate` with
the same arguments. Explicit input overrides are `--events-path`, `--split-path`,
`--video-dir`; their contents must match checkpoint provenance. Use a new output directory.

`evaluation.json` records the checkpoint SHA-256, epoch, saved/effective configuration,
full data manifest, environment, window/video metrics and synchronized timing.
`video_predictions.csv` counts each canonical video once; `window_predictions.csv`
retains each sampled window. The fixed aggregation is max violation probability per
video, threshold 0.5. `confusion_matrix.png` and `report.md` summarize the result.
Timing covers one cold batched pass over the selected windows including video decoding;
preparation time is separate. The per-video time is an amortized mean, not request latency.

The [measured smoke-baseline report](../reports/r3d18_smoke_evaluation/report.md) includes
a [manual error analysis](../reports/r3d18_smoke_evaluation/error_analysis.md).
It measures annotation-centred window aggregation, not blind full-video detection.
No threshold selection, training, label changes or independent-session evaluation occurs.

## Data and model

`events.csv` supplies binary labels and single event timestamps. The tracked
`split_proposal.csv` supplies the fixed 30/10 canonical video assignment. The loader
checks source coverage, aliases, labels, timestamps and actual video SHA-256 before
training. A changed label/time or video requires an explicitly versioned split update.
EDA validates the existing CSV without rewriting it and fails on a different proposal;
archive the prior split and experiment provenance before intentionally creating a replacement.
There is no dependency on generated EDA inventory files or a second annotation table.

Each positive event produces one 4-second window with 16 evenly sampled frames.
At a video boundary the window shifts inward; videos shorter than the window use
their available span. Each negative video uses one midpoint window by default.
`negative_windows` enables evenly spaced additional windows for both subsets.
Multiple source event markers on the same canonical video are preserved. Unannotated
parts of positive videos are not automatically labeled negative.

OpenCV decodes RGB frames, with sequential fallback for failed seeks. The transform
comes directly from `R3D_18_Weights.KINETICS400_V1.transforms()`: resize to 128×171,
center crop 112×112, scale/normalize using pretrained statistics, output C×T×H×W.
This standard crop may lose useful peripheral detail in portrait footage; investigate
that limitation in later experiments rather than changing preprocessing silently.
See the [official model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html).

The pretrained R3D-18 head is replaced by a two-logit linear layer:
`no_violation=0`, `violation=1`. Default training freezes the backbone, including
BatchNorm statistics, and updates only the head with AdamW and cross-entropy.
`freeze_backbone: false` allows a separately configured fine-tuning experiment.

## Run

Install the general and ML requirements using the existing Brev setup in README.
Run from the repository root, with the original videos in `data/raw/videos`.
The first run downloads approximately 127 MB of pretrained weights into
`models/pretrained/checkpoints`; this cache and experiment outputs are ignored by Git.

PowerShell:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m overfitted_underqualified.ml.train --config configs/r3d18.yaml
```

Linux / Brev:

```bash
PYTHONPATH=src python -m overfitted_underqualified.ml.train --config configs/r3d18.yaml
```

One-epoch experiment, retaining the full fixed train and validation sets:

```bash
PYTHONPATH=src python -m overfitted_underqualified.ml.train --config configs/r3d18.yaml --epochs 1 --output-dir runs/r3d18_new_smoke
```

`--video-dir` overrides the input directory; `--device cpu` explicitly selects CPU.
All YAML paths resolve relative to the repository root. Each experiment needs a new
output directory; nonempty directories are rejected to preserve earlier results.
Use the saved config with the same data and software/hardware for a repeat run.
Seeds cover Python, NumPy, PyTorch, DataLoader ordering and workers; deterministic
algorithms are enabled. Bitwise equality across different devices or library versions
is not promised.

## Notebook workflow

Open Jupyter with the environment containing both requirements files. Run each notebook
top to bottom in a fresh kernel, from the root or `notebooks/`; paths are discovered
from the current directory on Windows and Linux/Brev.

1. `01_eda.ipynb`: inspect files, validate `events.csv`, examine distributions/contact
   sheets, verify the fixed split and write readiness reports. Functions live in
   `eda/video.py`, `annotations.py`, `split.py`, `plots.py`, and `reports.py`.
2. `02_detection.ipynb`: load YAML → `prepare_training` (hashes, source coverage,
   datasets/loaders) → show sampled windows → `build_model` → `train_prepared` →
   loss/accuracy curves, validation results, compatible-run comparisons and conclusions.
   `RUN_TRAINING=False` by default skips model initialization and training; it reads
   `EXISTING_RUN` for results. Copy `.env.example` to `.env` in the repository root
   and set `RUN_TRAINING=true` explicitly to train. Notebook 02 rereads this file
   in its settings cell; existing shell variables take precedence. The CLI `run`
   uses the same phases and retains its existing explicit training behavior.
3. `03_evaluation.ipynb`: `prepare_evaluation` restores saved model/configuration
   and checks annotation/split hashes, canonical video records and validation windows;
   `evaluate_prepared` computes loss, precision, recall, F1, accuracy and confusion
   matrix with threshold 0.5; research cells separately analyze FP/FN, evidence frames
   and sampling offsets. Timestamp prediction error is N/A: the model has no temporal
   localization output. Known annotation-centred windows cannot establish detection accuracy.

Notebook environment overrides (set before launching Jupyter, or edit the settings cell):

- `VIDEO_DIR`: location of the original videos.
- `TRAIN_EPOCHS`: optional epoch override; `1` is enough to check the pipeline.
- `TRAIN_DEVICE`: explicit `cpu` or `cuda`; default comes from YAML.
- `TRAIN_OUTPUT_DIR`: optional fresh run directory; default includes a UTC timestamp.
- `CHECKPOINT_PATH`: checkpoint to evaluate; default is `runs/r3d18_smoke/best.pt`.
- `EVAL_DEVICE`: defaults to `cpu`; set `cuda` for GPU evaluation.

PowerShell example (one epoch only after setting `RUN_TRAINING=True` in notebook 02,
then select the printed checkpoint in notebook 03):

```powershell
$env:TRAIN_EPOCHS = "1"
.\.venv\Scripts\jupyter.exe lab
```

Linux/Brev equivalent:

```bash
TRAIN_EPOCHS=1 jupyter lab
```

On another machine, copy the chosen checkpoint and the matching original data.
Notebook 03 explicitly supplies current repository input paths, overriding historical
absolute paths saved on Windows; all hashes must still match. It does not modify the
saved configuration or run artifacts. `R3D_18_Weights.transforms()` supplies preprocessing
without downloading weights. Model restoration uses `weights=None` and strict state loading.
Evaluation constructs no optimizer, disables gradients and sets every layer to eval mode.
It does not tune thresholds or update BatchNorm. Running evaluation never starts training.

Outputs and execution counts should be cleared from all notebooks before committing;
experiment provenance belongs in `runs/` and deliberate summaries in `experiments/`.

## Outputs and acceptance

- `config.yaml`: effective settings, including CLI overrides and resolved paths.
- `data_manifest.json`: source/split hashes, canonical videos, aliases and sampled windows.
- `environment.json`: runtime versions, device, preprocessing and seed.
- `history.json`: train and validation loss, accuracy, precision, recall, F1, confusion matrix.
- `last.pt`, `best.pt`: full model state, optimizer, epoch, configuration and manifest.
  Best checkpoint is selected by validation loss; no gradient or BatchNorm updates
  occur during validation.
- `best_validation_predictions.csv`: per-window probabilities for the best checkpoint.

`load_checkpoint` in `ml/train.py` reconstructs a saved model without downloading weights.
The threshold is fixed at 0.5. Validation is used for checkpoint selection only;
there is no final held-out test set.

These metrics evaluate **annotation-centred window classification**: positive validation
windows use known event timestamps, while negative windows use fixed positions.
They do not evaluate blind event localization or full-video detection. The small,
within-room split may share people, backgrounds and recording sessions. An independent
evaluation, sliding-window inference and backend integration are separate tasks.

Run tests with `python -m unittest discover -s tests` after installing dependencies.
Tests exercise real video decoding, preprocessing, duplicate/source validation,
frozen BatchNorm and absence of validation parameter updates. A successful small
experiment must also save and reload a real R3D-18 checkpoint.

The first real-video acceptance run and exact reproducibility checks are recorded
in [experiments/r3d18_smoke_result.md](../experiments/r3d18_smoke_result.md).
That smoke run predicted violation for all validation windows (accuracy 0.5, recall 1.0).
It establishes execution and reproducibility, not useful detection quality. Refactoring
does not change those historical metrics or turn them into an independent evaluation.
