# Overfitted and Underqualified

Datathon workspace for the **Overfitted and Underqualified** team.

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

If PowerShell blocks environment activation, run the commands below from a terminal where Python is available:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\jupyter.exe lab
```

## ML dependencies on Brev L4

Open a terminal on the Brev L4 instance, clone this repository (or pull the latest commit), and
run these commands from the repository root. The instance may not have `python3.11` installed.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to get the exact Python
3.11.9 version recorded in `.python-version`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11.9
uv venv --seed --python 3.11.9 .venv
source .venv/bin/activate
python --version      # expected: Python 3.11.9
nvidia-smi            # expected GPU: NVIDIA L4
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
python -c "import torch, torchvision; print('torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'PyTorch CUDA:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())"
```

`requirements-ml.txt` pins PyTorch 2.13.0 and torchvision 0.28.0 using the official
[CUDA 12.6 wheel index](https://pytorch.org/get-started/previous-versions/) specified inside that
file. The PyTorch check should print `PyTorch CUDA: 12.6` and `CUDA available: True`.
The CUDA version displayed by `nvidia-smi` describes the driver, so it may be different.
The [experiment config](experiments/brev_l4_smoke.yaml) records the seed, device, and synthetic
one-step forward/backward settings for that verification. The [Brev L4 result](experiments/brev_l4_smoke_result.md)
records the observed versions, GPU, and successful optimizer step.

## Project layout

```text
data/
  raw/          # Original data; contents are ignored by Git
  processed/    # Generated data; contents are ignored by Git
notebooks/      # Exploration and experiments
reports/figures/ # Charts and exported figures
src/overfitted_underqualified/ # Reusable project code
```

Start with `notebooks/01_eda.ipynb`. Put the open-set videos into `open/` (or `data/open/`). Put reusable functions in `src/overfitted_underqualified/` once they are useful in more than one notebook. Keep notebooks focused, restart and run them top to bottom before sharing results, and record important findings and data sources here or in the notebook.

## EDA inputs and outputs

The notebook also discovers videos in `data/raw/videos/`. Set `VIDEO_DIR` to use an explicit directory.
Run the notebook with the repository's `.venv` interpreter, from top to bottom.
EDA functions are defined in the sections that use them: video reading, annotation
validation, plots, contact sheets and scene grouping. Shared frame reading is defined
in the first section because video inspection also uses it. There is no separate
EDA Python module. Tests load imports and the `eda-definitions` cells without
executing the video analysis pipeline.

- `data/annotations/events.csv` is the editable annotation source. EDA never overwrites it.
  `source_label` and `timecode` preserve the supplied supervision. Descriptions, event tags, candidate categories and rule references
  are provisional interpretations of the supplied text, used only for exploration.
  They are not verified multiclass targets or classifier input features. The former
  `confidence` and `review_status` columns have been removed.
- `data/processed/validation_issues.csv` lists missing files, duplicate IDs, invalid timestamps,
  decoding problems and conflicting duplicate labels. Failed MOV seeks are retried by
  sequential decoding against frame timestamps before reporting an end-frame warning.
- `data/processed/events_enriched.csv` combines eligible annotations with current video metadata.
  It keeps every source record and adds `canonical_clip_id` for identical files.
- `data/processed/duplicate_map.csv` maps source IDs to canonical video IDs by SHA-256;
  handled copies are informational entries in `dataset_notes.csv`.
- `data/processed/training_videos.csv` contains one row per unique video, with all original
  IDs in `source_clip_ids` and all distinct event markers in `event_times_sec` (JSON arrays).
  Use this table for video-level sampling; retain `events_enriched.csv` for event annotations.
- `reports/figures/event_frames/` and `normal_frames/` contain contact sheets for both classes.
  `data/processed/frame_extraction.csv` records whether every requested frame was decoded.
- `data/processed/scene_candidates.csv` lists visually similar scene candidates; these are heuristics
  to review, not confirmed camera identities. Inspect the scene preview and contact sheets.
- Copy `data/processed/scenes_template.csv` to `data/annotations/scenes.csv` and fill in camera,
  session or scene groups after reviewing the footage. IDs refer to the same camera/session across clips.
  Set `reviewed=true` only after checking each clip. This optional source file is never overwritten.
- `data/processed/split_proposal.csv` proposes a reproducible train/validation split of canonical
  video IDs from `training_videos.csv`. Exact copies,
  candidate scenes and known cameras/sessions stay together. The proposal is provisional until scenes
  have been reviewed. If no suitable group split exists, the file is empty and the reason is reported.
  All clips from a group must remain together in downstream frame/segment datasets.
- `data/processed/eda_checklist.csv` records current-run checks; `eda_run.json` records input hashes
  and parameters. Fix errors and review groups before using the proposed split for training.

Optional frame zones live in `data/annotations/zones.json`, using
`{"X-N": {"zone_name": [x1, y1, x2, y2]}}`. Coordinates are checked against the decoded frame.

Run the validation tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
