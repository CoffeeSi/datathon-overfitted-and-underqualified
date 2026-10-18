# Full-video binary scanning and evaluation

`ml/inference.py` implements annotation-free scanning. `ml/full_video_evaluation.py`
implements the separate full-video validation protocol. Existing Dataset, training,
annotation-centred evaluation, rules catalog and public FastAPI schemas are retained.
No import starts inference, trains a model or writes files.

## What the result means

The scanner outputs binary window scores and **untyped event candidates**. It does
not identify an action, establish a legal violation or infer conditions outside the
camera view. `event_type` and `type_confidence` remain null and `source_links` empty.
The sole editable annotation source is still `events.csv`; source IDs, timestamps
and the agreed 30/10 canonical split are unchanged. No interval annotation file is added.

`configs/full_video.yaml` fixes the exploratory choices before evaluation:
4-second windows, 2-second step, 16 frames, batch size 2, threshold 0.5, merge gap 0,
matching tolerance 5 seconds. These choices have not been optimized on validation.
Changing the temporal context changes the input distribution; preserve the resulting
configuration and evaluate it as a new experiment.

## Scanning and timing

Windows cover the nominal first through last frame. The last window shifts inward;
short videos produce one window over the available span, with repeated indices if
fewer than 16 frames exist. Steps greater than the window are rejected to avoid gaps.
The scanner reuses `dataset.decode_frames` and the checkpoint's KINETICS400_V1
transform. It restores R3D-18 with `weights=None`; no pretrained download occurs.
`eval()` and `torch.inference_mode()` prevent gradient and BatchNorm updates.

Memory holds one raw 16-frame window and at most one batch of transformed windows,
plus numeric output metadata. Metadata scales with video duration; decoded video
does not accumulate in RAM. Failed seeks use the existing sequential fallback.
Unreadable metadata, frames or nonfinite model outputs abort the pass; no window is
silently dropped. Sequential fallback may be slow on long or badly indexed files.
This is an offline research scanner, not a real-time streaming implementation.

Positive supports that overlap or are separated by at most `merge_gap_sec` merge.
An overlapping chain can bridge a negative window. Each merged support yields one
candidate, even if it contains several actual actions. Its representative time is
the centre of the maximum-score window; equal scores choose the earliest window.
This is an estimated event time, not a learned onset. Support intervals are predictions,
not new training annotations.

An opt-in `candidate_policy: window_peaks` instead extracts local maxima above the
same fixed threshold and applies temporal suppression with `peak_separation_sec: 4`.
It can retain multiple candidates inside one long alarm. It has no annotation lookup;
plateau ties choose the earliest point. More peaks can increase negative-video false
alarms, so the old `support_union` remains the default. The study's policy comparison
reports this tradeoff from saved scores without retraining or threshold search.
Neither policy provides learned event boundaries or verified action types.

Sampling uses nominal FPS and frame indices, as in training. Variable FPS/container
timing can differ from this convention. `evidence_frames` selects up to three nearby
indices, sequentially decodes them and records `frame_index`, nominal time and actual
OpenCV decoder PTS separately. Invalid PTS is an error. JPEG `image_base64` and
`timestamp_sec` are compatible with `EvidenceFrame`; metadata is retained internally.
Frames are shown locally only and excluded from reports and ZIP archives.

## Notebook workflow

- `01_eda.ipynb`: existing EDA, source checks, canonical split and local contact sheets.
- `02_detection.ipynb`: existing interactive training plus catalog coverage, selected
  full-video scan, probability timeline, candidate table and local evidence view.
  `RUN_TRAINING=false` and `RUN_FULL_SCAN=false` by default. Enable each explicitly.
  Training creates a new run. When training and scanning together, the scan uses
  that new `best.pt`; otherwise `CHECKPOINT_PATH` chooses the saved model.
- `03_evaluation.ipynb`: `EVAL_PROTOCOL=annotation_windows` (A), `full_video` (B),
  or `both`. Separate metrics, errors, temporal matching, local evidence, conclusions
  and automatic exports. Protocol B uses all 10 validation videos and creates a new
  `reports/evaluation_full_<run>_<UTC>_<id>/` and neighboring ZIP.

Use the project kernel from the repository root or `notebooks/`. `CHECKPOINT_PATH`,
`EVAL_DEVICE`, `VIDEO_DIR`, `SCAN_VIDEO`, `RUN_FULL_SCAN`, `EVAL_PROTOCOL` are read from
root `.env`; shell environment values take precedence. Relative video paths resolve
against the repository root. No evaluation protocol starts training.
Executed notebook copies, figures and images are private local artifacts. Committed
notebooks must have empty outputs, execution counts and attachments.

## CLI

From the repository root, PowerShell first sets `$env:PYTHONPATH = "src"`.
Linux/Brev uses `PYTHONPATH=src python ...` instead of that PowerShell setup.

Single unseen video (does not read events/split):

```powershell
python -m overfitted_underqualified.ml.inference --checkpoint runs/my_run/best.pt --video data/raw/videos/X-1.mov --config configs/full_video.yaml --device cuda --output reports/evaluation_single_run/scan.json
```

Complete validation with matched checkpoint/data provenance and automatic ZIP:

```powershell
python -m overfitted_underqualified.ml.full_video_evaluation --checkpoint runs/my_run/best.pt --config configs/full_video.yaml --device cuda --output-dir reports/evaluation_full_my_run
```

Explicit `--events-path`, `--split-path`, `--video-dir` relocate the same data. Their
contents must match checkpoint provenance. Output destinations must be new; prior
reports and checkpoints are never overwritten. CPU is available via `--device cpu`.

## Metrics and reproducibility

A measures classification of windows centred on known positive markers, with selected
negative windows. B scans the whole video without consulting annotations. B's video
decision is whether at least one window reaches the fixed threshold. Do not treat
the difference between their F1 values as an improvement on one unchanged task.

B saves video-level precision, recall, F1, accuracy and confusion matrix `[TN, FP;
FN, TP]`; candidate counts; false candidates per accepted negative video and per
minute of accepted negative recordings. These false alarm rates are relative to the
team's binary labels, not independently verified exhaustive frame annotations.

Temporal matching maximizes the number of pairs within the fixed tolerance, then
minimizes total absolute error. Every prediction and marker can occur in only one
pair. Additional candidates near an already matched marker stay unmatched. Distinct
X-4/X-47 markers remain distinct despite sharing a canonical video. Metrics include
matched MAE/max error, known-marker recall, candidate match fraction, missed marker
count and unmatched prediction count. The fraction is deliberately not called proven
event precision: unmatched candidates on positive videos are **unresolved**, since
the point annotations may not describe all events. MAE without its match/miss counts
would be misleading. An empty matched set yields null, not zero MAE.

`evaluation.json` records checkpoint hash/epoch, saved model config, scan config,
tolerance, data manifest, evaluated video hashes/aliases/markers, implementation
file hashes, software/device and timing. Preparation is separate; processing includes
metadata/hash checks, decoding, preprocessing, transfer, inference, merging and matching.
GPU timing is synchronized. There is no dedicated warmup; earlier notebook work may
have warmed the runtime. Timings exclude report serialization and evidence display;
mean video time is amortized, not HTTP request latency.

Exports contain only `evaluation.json`, `window_predictions.csv`, `candidates.csv`,
`video_predictions.csv`, `matching.csv`, `video_errors.csv` and `report.md`.
ZIP uses that explicit allowlist. It excludes weights, original videos, banking
frames, `.env` and unrelated files. Generated `reports/evaluation_*` are Git-ignored.
Numeric predictions and manifests still belong to the project team; they are not
automatically published. Historical reports remain unchanged.

The checkpoint was selected on the same small within-room validation set. Shared
room, people and unknown sessions prevent an independent generalization claim.
The original smoke run predicted violation for every validation window. Full-video
scanning alone does not repair poor class discrimination or train temporal localization.

## Catalog and FastAPI integration boundary

Coverage is audited by `rules/coverage.py`; see
[the coverage report](../reports/catalog_coverage.md). The historical audit does
not assert action-label completeness. On 18 October the
owner confirmed listed observations but not catalog-wide negative labels. The new
[action audit and pipeline](action-recognition.md) preserve that distinction;
validation is never used for training or to move examples between subsets.

`ActionRecognizer.recognize(video_path, candidate) -> ActionObservation | None`
is the single-observation extension point. `action_inference.R3DActionRecognizer`
adds `recognize_many` and per-window multi-label decisions before merging. The
implementation exists, but a real action checkpoint is blocked by missing confirmed
negative labels. Its output includes a supported catalog type, a separate
type confidence, component provenance and visible description. `link_action` validates
against `supported_event_types()`, calls `source_links_for()` and retains applicability
and conditions video cannot establish. It does not convert binary scores to types or
prove that applicability conditions hold. Catalog interpretation is inherited from
the existing reviewed repository catalog, not newly legally verified by this change.

The public `Analyzer.async analyze(video_path) -> AnalysisResult`, `Finding` and
`EvidenceFrame` remain compatible. `backend.local_analyzer` supplies a threaded
adapter with cooperative cancellation and retained admission slots. UI-only metadata
carries rules and unresolved candidates; the jury endpoint projects the old schema.
The real provider remains unavailable without both checkpoints and explicitly enabled,
evaluated types. See [configuration and remaining blockers](action-recognition.md).
Do not set
`FORTEBANK_ANALYZER_FACTORY` to this scanner: it intentionally returns an internal ML
result, not a completed legal analysis. Existing configured/mock backend behavior
is preserved; no ground-truth lookup is added to production inference.
