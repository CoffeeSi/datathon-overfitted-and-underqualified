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

- `data/annotations/events.csv` is the sole editable source of event labels and timestamps, accepted by the team. EDA never overwrites it or requires a separate approval table.
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
- `data/processed/split_proposal.csv` records the agreed preliminary train/validation split
  by unique video SHA-256, approximately 75/25 with seed 42. Both subsets contain both
  classes. Each row includes canonical ID, hash, source aliases and all event markers.
  Exact copies never cross subsets. The file is allowed in Git to retain the baseline assignment;
  it is generated from `events.csv` and is not another editable annotation source.
- Scene similarity remains diagnostic only for this protocol. The room, people and unknown
  recording sessions may overlap between subsets. Results describe within-room performance,
  not generalization to independent sessions or new cameras. There is no final held-out test set.
- `data/processed/eda_checklist.csv` records current-run checks; `eda_run.json` records
  input hashes and parameters. `training_readiness.json` distinguishes preliminary baseline
  readiness from independent evaluation readiness. P0 completion refers to the agreed
  preliminary protocol only.

See [the annotation workflow](docs/temporal-annotation-workflow.md) for the single-source annotation protocol and current split limitation.

Optional frame zones live in `data/annotations/zones.json`, using
`{"X-N": {"zone_name": [x1, y1, x2, y2]}}`. Coordinates are checked against the decoded frame.

Run the validation tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
