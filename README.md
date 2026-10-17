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

Use Python 3.11.9, recorded in `.python-version`. On the Linux Brev instance, check that
`python3.11 --version` reports exactly `3.11.9`, then install the general dependencies and
the pinned CUDA 12.6 PyTorch wheels:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
```

`requirements-ml.txt` pins PyTorch 2.13.0 and torchvision 0.28.0 using the official
[CUDA 12.6 wheel index](https://pytorch.org/get-started/previous-versions/). The Brev GPU driver
must be compatible with this runtime; that still needs verification on the L4 instance.

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
Run the notebook with the repository's `.venv` interpreter. Checks are implemented in
`src/overfitted_underqualified/eda.py`.

- `data/annotations/events.csv` is the editable annotation source. EDA never overwrites it.
- `data/processed/validation_issues.csv` lists missing files, duplicate IDs, invalid timestamps,
  decoding problems, exact copies and conflicting duplicate labels.
- `data/processed/events_enriched.csv` combines eligible annotations with current video metadata.
- `reports/figures/event_frames/` and `normal_frames/` contain contact sheets for both classes.
  `data/processed/frame_extraction.csv` records whether every requested frame was decoded.
- `data/processed/scene_candidates.csv` lists visually similar scene candidates; these are heuristics
  to review, not confirmed camera identities. Inspect the scene preview and contact sheets.
- Copy `data/processed/scenes_template.csv` to `data/annotations/scenes.csv` and fill in camera,
  session or scene groups after reviewing the footage. IDs refer to the same camera/session across clips.
  Set `reviewed=true` only after checking each clip. This optional source file is never overwritten.
- `data/processed/split_proposal.csv` proposes a reproducible train/validation split. Exact copies,
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
