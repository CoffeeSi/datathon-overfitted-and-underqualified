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

## Backend dependencies

Install the API dependencies separately from the notebook environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
```

The main `requirements.txt` includes this file, so a full project installation
also installs the backend dependencies. SQLite is provided by Python; ML model
dependencies belong to the selected analyzer and are not included here.

For the local trained cash-sequence analyzer on Windows (requires all three
checkpoints referenced by `configs/action_service_cash_sequence.yaml` and CUDA):

```powershell
# Set the shared FORTEBANK_API_KEY in .env.compose (also read by the Docker frontend).
.\scripts\start-backend.ps1 -Port 8000
```

The launcher explicitly selects the real action provider and disables mocks,
overriding stale terminal settings. Check `GET http://127.0.0.1:8000/readyz` for
200 and `local-cash-sequence`. The supported scenario is receipt, client departure
and cash remaining on the counter. It does not detect every legal type; no candidate
returns insufficient_data. See [the live integration report](reports/cash_sequence_backend_20261018.md).
To reproduce the old action-head service explicitly, pass
`-ActionConfig configs/action_service_backend_smoke.yaml`.

## Action model improvement study

The current action model is compared with a newly trained visible-cash sequence in
[the action improvement study](reports/action_improvement_20261018.md).
`PYTHONPATH=src python -m overfitted_underqualified.ml.action_improvement --config configs/action_improvement.yaml`
runs the fixed baseline evaluation, three training stages and a common reviewed-grid
comparison. Use a fresh `output_dir` for each run. It preserves the split and old
weights; this candidate recognizes one cash scenario, not every legal type. The
running backend is not switched automatically.

## Frontend — Overfitted and Underqualified

The research-review UI lives in [`frontend/`](frontend/README.md): MOV upload,
job status, synchronized event timeline, evidence frames and local recent-job metadata.
See [frontend setup and configuration](frontend/README.md) for npm commands,
server-only API credentials and explicitly enabled synthetic examples. A typed ML
Analyzer is still required for real findings; the UI preserves the unavailable-provider
state and never substitutes demo results for a failed analysis.

## ML dependencies on Brev L4

The current requirements and `.python-version` match the local Python 3.14.5
environment. NumPy 2.5.4 requires Python >=3.12. Historical Brev experiment records
retain the Python 3.11.9 runtime used for those experiments.

Open a terminal on the Brev L4 instance, clone this repository (or pull the latest commit), and
run these commands from the repository root. The instance may not have Python 3.14 installed.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to get the exact Python
3.14.5 version recorded in `.python-version`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.14.5
uv venv --seed --python 3.14.5 .venv
source .venv/bin/activate
python --version      # expected: Python 3.14.5
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

## R3D-18 baseline

The implementation and run commands are documented in [docs/r3d18-baseline.md](docs/r3d18-baseline.md).
Settings live in `configs/r3d18.yaml`; training uses the fixed 30/10 split and saves
checkpoints, effective configuration and metrics under `runs/`.
Video-level evaluation, confusion matrix, error analysis and processing time are recorded
in [reports/r3d18_smoke_evaluation/report.md](reports/r3d18_smoke_evaluation/report.md).
The evaluation CLI is documented in [the baseline guide](docs/r3d18-baseline.md#video-level-quality-evaluation).

## Project layout

```text
data/
  raw/          # Original data; contents are ignored by Git
  processed/    # Generated data; contents are ignored by Git
notebooks/      # Exploration and experiments
reports/figures/ # Charts and exported figures
src/overfitted_underqualified/ # Reusable project code
```

Run `01_eda.ipynb` → `02_detection.ipynb` → `03_evaluation.ipynb` with the project's
Python kernel. Each notebook locates `src/` from either the repository root or
`notebooks/`, on Windows or Linux/Brev. ML notebooks require `requirements-ml.txt`.
Restart the kernel and run cells from top to bottom. Clear outputs before committing.

- `eda/video.py`: discovery, metadata, hashes and timestamp-based frame checks.
- `eda/annotations.py`: source label and timestamp validation.
- `eda/split.py`: canonical copies, scene diagnostics and reproducible split checks.
- `eda/plots.py`, `eda/reports.py`: figures, contact sheets and generated reports.
- `ml/dataset.py`: validated sources and 16-frame training/validation windows.
- `ml/train.py`: preparation, model loading, training and the CLI.
- `ml/evaluate.py`, `ml/plots.py`: checkpoint evaluation, loss, confusion matrix and error views.
- `ml/inference.py`: annotation-free sliding-window scanning, untyped candidates and local evidence.
- `ml/full_video_evaluation.py`: full-video validation, one-to-one temporal matching and private report/ZIP export.

The baseline notebook uses YAML settings. `RUN_TRAINING=False` by default lets you
inspect data and existing experiments without loading pretrained weights or training.
Copy `.env.example` to the repository-root `.env` and set `RUN_TRAINING=true` there
to create a fresh experiment directory; notebook 02 rereads `.env` in its settings cell.
Shell environment variables override `.env`. The local `.env` is ignored by Git.
`TRAIN_EPOCHS=1` limits that explicitly enabled run to one epoch.
Set `TRAIN_DEVICE=cpu` explicitly if needed.
For evaluation, set `CHECKPOINT_PATH` in `.env` to the printed `best.pt` path. If unset,
the only available `runs/*/best.pt` is selected; multiple files require an explicit choice. Evaluation verifies
source/split/video provenance, uses threshold 0.5 and does not download pretrained weights.
Its final cell automatically saves results under `reports/evaluation_<run>_<UTC>_<id>/`
and displays a ZIP download link, without weights or banking images.
See [the notebook and CLI instructions](docs/r3d18-baseline.md#notebook-workflow).

Notebooks own research choices, inspection tables, experiment comparisons and written
conclusions; modules own decoding, datasets, preprocessing, training and metric computation.
Evaluation separates FP/FN and shows local evidence frames. Protocol A (known windows)
has no predicted timestamp; sampling offsets are not localization errors. Protocol B
(full-video scan) derives candidate timestamps from peak windows and reports matching
errors and missed markers separately. It does not identify concrete action types.

Full-video settings and commands: [docs/full-video-inference.md](docs/full-video-inference.md).
In 02, set `RUN_FULL_SCAN=true` and optionally `SCAN_VIDEO`; training is controlled separately.
In 03, select `EVAL_PROTOCOL=annotation_windows`, `full_video`, or `both`.
All full-video parameters are fixed in `configs/full_video.yaml`, without validation tuning.
The catalog's data coverage and action-recognition limitations are recorded in
[reports/catalog_coverage.md](reports/catalog_coverage.md).

## EDA inputs and outputs

Private research evidence, annotation snapshots and supplemental audits live in
`data/local_artifacts/` (ignored by Git). This replaces the former `outputs/`
directory. Historical annotation snapshots are immutable provenance; `events.csv`
remains the editable annotation source. Model checkpoints remain in `runs/`.

The notebook also discovers videos in `data/raw/videos/`. Set `VIDEO_DIR` to use an explicit directory.
Run the notebook with the repository's `.venv` interpreter, from top to bottom.
Reusable EDA functions live in `src/overfitted_underqualified/eda/`; the notebook
contains explanations, settings and explicit calls. Importing modules does not run
analysis or create output files. Tests import ordinary Python modules.
EDA retains timestamp/PTS fallback decoding; ML retains its fixed frame-index sampler.
These readers intentionally have different semantics.

- `data/annotations/events.csv` is the sole editable source of event labels and timestamps, accepted by the team. EDA never overwrites it or requires a separate approval table.
  `source_label` and `timecode` preserve the supplied supervision. Descriptions, event tags, candidate categories and rule references
  were initially provisional interpretations. On 18 October the owner confirmed the
  listed observations, but not completeness against the catalog or negative action
  labels. The separate action experiment masks unknown targets and currently blocks
  training; descriptions and rule text are never model input features. The former
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
  EDA checks an existing split without rewriting it and stops if the regenerated proposal differs.
- Scene similarity remains diagnostic only for this protocol. The room, people and unknown
  recording sessions may overlap between subsets. Results describe within-room performance,
  not generalization to independent sessions or new cameras. There is no final held-out test set.
- `data/processed/eda_checklist.csv` records current-run checks; `eda_run.json` records
  input hashes and parameters. `training_readiness.json` distinguishes preliminary baseline
  readiness from independent evaluation readiness. P0 completion refers to the agreed
  preliminary protocol only.

Historical baseline metrics describe windows around known annotations. New full-video
metrics use a separate protocol and cannot be equated directly with window F1.
The first smoke run predicted violation for every validation window;
this verifies the pipeline, not useful detector quality. Historical metrics remain in
[the experiment record](experiments/r3d18_smoke_result.md).

See [the annotation workflow](docs/temporal-annotation-workflow.md) for the single-source annotation protocol and current split limitation.

P1 rules catalog: [`src/overfitted_underqualified/rules/catalog.json`](src/overfitted_underqualified/rules/catalog.json).

Action recognition: [implementation and commands](docs/action-recognition.md),
[current P0 status](reports/action_recognition_status.md). Separate multilabel R3D-18
training, per-window recognition, typed event merging, catalog enrichment, evaluation
exports and a real async backend adapter are implemented. Trained research action
checkpoints exist; their measured limitations are documented below. Real readiness
stays false until both models and evaluated types are configured. Notebook 02 audits
coverage and offers explicit action training; notebook 03 has an opt-in action protocol.
Existing binary experiments, labels and split remain unchanged.
Two real five-epoch action experiments now exist (unweighted and train-only loss
balancing). Their only evaluable positive validation example is missed at the fixed
0.7 threshold: F1=0. See [action experiment results](reports/action_training_20261018.md).
Use `configs/actions_balanced.yaml` and the explicit research service config to
reproduce; default service type activation remains disabled.
The subsequent [X-18/X-2 geometry study](reports/identity_geometry_20261018.md)
compares a single-class head with the original crop and an aspect-preserving ROI.
Both fixed 30-epoch runs fit train but miss X-2; additional positive videos are needed.
The notebooks include the saved comparison; action preprocessing is shared with inference.
The [subsequent cash-storage check](reports/unsecured_cash_20261018.md) resolves X-38
as a positive action window and verifies three negative windows, masking four uncertain
negatives. Its fixed-threshold model still misses X-38 (F1=0); a full-scan alarm on
X-2 is false. This is a measured failure, not successful P0 quality acceptance.
The historical 231-only CSV revision changed only `legal_basis`. The current source
also cites №14 and №53. Verified №14 paragraph 8 is research context; final №53
paragraph 53 remains unverified. Active action links still use №231; applicability
remains conditional. See [current action readiness](reports/action_preparation_20261018.md). An explicit
[annotation revision record](data/annotation_revisions.json) allows historical binary
checkpoints to evaluate that exact revision while preserving both source hashes.
See [migration checks](reports/annotations_231_migration.md); arbitrary annotation drift is rejected.
Missing descriptions on 20 source rows were filled from [sampled video review](reports/visible_descriptions_20261018.md).
The 19 unique videos were reviewed at 1 Hz with selected original frames. This changes
visible prose only; binary supervision and unassigned action tags remain unchanged.
The exact registered metadata revisions also preserve historical checkpoint evaluation.
Current [annotation/catalog audit](reports/action_annotation_review_20261018.md)
identifies four uncertain mappings. `python experiments/review_action_annotations.py`
creates a private review page in ignored `data/local_artifacts/`. Per-type present/absent/unknown
decisions can be imported into the sole `events.csv`; omitted types remain masked.
The default action config uses this partial-review protocol, not catalog-wide approval.

Binary improvement study: [measured results and checkpoints](reports/binary_improvements_20261018.md).
On the same 10 previously examined validation videos, the selected 4-second ROI
candidate reached full-video F1 0.714 (baseline 0.615), matched 5/5 known markers
within ±5 seconds and reduced negative-video alarm candidates from 8 to 5.
Four of five negative videos still receive false alarms; this is exploratory progress,
not independent validation or production readiness. All five trained variants and
the failed initial attempt are retained. Same-seed retraining reproduced identical weights.
Verification notes and acceptance checklist:
[docs/rules-catalog-review.md](docs/rules-catalog-review.md).
Only `api_supported` event types may be attached to a finding; binary `violation`
is not an event type. Unconfirmed claims (X-1, X-28, door/№14, X-46) stay excluded.

Optional frame zones live in `data/annotations/zones.json`, using
`{"X-N": {"zone_name": [x1, y1, x2, y2]}}`. Coordinates are checked against the decoded frame.

Visible-primitives research now separates client presence, cash on the counter and
cash handover from the temporal service-history hypothesis. It uses sparse reviewed
facts inside the same `events.csv`, causal 16-frame inputs and the unchanged 30/10
split. Training and full-video scanning are opt-in in 02/03; reusable code lives in
`ml/primitive_data.py`, `primitive_model.py`, `cash_sequence.py`,
`primitive_inference.py` and `primitive_evaluation.py`.
See [workflow and CLI commands](docs/visible-primitives.md) and
[measured experiment results](reports/primitive_study_20261018.md).
Candidates require review; this experiment does not replace the FastAPI analyzer.

The spatial experiment adds a small counter-banknote detector and audited
train-only external augmentations. It retains the fixed split and old checkpoints;
the synthetic S-scenarios are recorded separately and never counted as real test
videos. Notebooks 01/02/03 include source coverage, box inspection, explicit training,
hybrid evaluation, error trajectories and downloadable reports. See
[banknote workflow](docs/banknote-sequence.md). All images stay local and notebook
outputs must be cleared before committing.
The [measured detector/tracking comparison](reports/banknote_study_20261018.md)
now finds X-38 at 29 s, within 4 s of its current action marker. Three reviewed
negative videos have no candidates. This meets the minimum local prototype check
for one visual scenario; four repeatedly inspected videos do not establish
independent legal-event quality. Other validation videos remain unknown for this
type, and the FastAPI analyzer is not automatically switched.

Additional source videos are imported into ignored `data/raw/supplemental/`.
The [augmentation workflow](docs/augmented-videos.md) trains binary R3D-18 on
canonical train descendants and evaluates validation descendants separately,
preserving the existing split and the sole annotation source `events.csv`.
The [matched augmentation protocol](docs/matched-augmentation.md) fixes optimizer
updates and compares original-only sampling with an 80/20 mixture. Final
checkpoints are frozen before validation; accuracy is reported without threshold
or split adjustments.

Run the validation tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
