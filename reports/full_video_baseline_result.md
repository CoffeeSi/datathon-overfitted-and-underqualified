# Full-video baseline: local acceptance run

Executed 2026-10-18 local time (Asia/Qyzylorda); artifact names use 2026-10-17 UTC.
This is exploratory evaluation within the existing room, not an independent test.

## Artifacts and provenance

- Training: `runs/r3d18_full_video_20261017T201156_904223Z/`; 5 epochs, seed 42,
  frozen pretrained backbone, learning rate 0.001, original 30/10 canonical split.
- Best checkpoint: epoch 4, SHA-256
  `c0e4a7f6c0a5743dc10348fb74884cebda2b8cdc1c91a74160b50fcce65df511`.
- Source SHA-256: `2ec181a7af08257f882aeb006b6f98b5ccb0f204470adb92d252616a67b1e062`.
- Split SHA-256: `19c33f7e336b51fe5705f62d1804fb05e9c557076217aa5c0d6f986eb22f363d`.
- Full-video report and adjacent ZIP:
  `reports/evaluation_full_r3d18_full_video_20261017T201156_904223Z_20261017T201706Z_546d55bf/`.
- Known-window report and adjacent ZIP:
  `reports/evaluation_r3d18_full_video_20261017T201156_904223Z_20261017T201412Z_630a46b3/`.
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU. Python 3.14.5,
  PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126, OpenCV 5.0.0, NumPy 2.5.4.

Generated reports, checkpoints, executed notebooks and evidence are local and ignored
by Git. The JSON report records effective scan/model settings, evaluated video hashes,
original aliases/markers, library versions, implementation hashes and measured timings.
Previous experiments and metrics were not overwritten.

## Protocol A: known-window classification

The separately retained protocol reproduced accuracy 0.60, precision 0.571429,
recall 0.80, F1 0.666667; confusion matrix `[[2, 3], [1, 4]]`.
These windows use known positive timestamps. Their F1 is not the same measurement as B.

## Protocol B: full-video pass

All 10 canonical validation videos were scanned: 202 windows, 14 merged candidates.
Fixed parameters: 4-second window, 2-second step, 16 frames, batch size 2,
threshold 0.5, merge gap 0, matching tolerance 5 seconds. No threshold or tolerance
selection used validation outcomes.

- Video accuracy: **0.50**; precision: **0.50**; recall: **0.80**; F1: **0.615385**.
- Confusion matrix `[TN, FP; FN, TP]`: **`[[1, 4], [1, 4]]`**.
- Eight false candidate alarms across five accepted negative videos:
  **1.6 candidates/negative video**, **1.693918 candidates/negative minute**
  over 4.722778 negative minutes.
- Temporal matching: **3/5** known markers matched; **2** missed markers.
  Matched MAE: **2.666667 seconds**, maximum matched error: **5 seconds**.
- Eleven unmatched candidates: eight on accepted negative videos and three unresolved
  on positive videos. Candidate match fraction is 3/14, **not established event precision**.
- Preparation: **1.177 seconds**; full processing: **57.383 seconds**, mean **5.738 seconds/video**.
  Timing includes decoding, hashing and inference; report/evidence display is excluded.
  There was no dedicated warmup; earlier work may have warmed the runtime.

One-to-one matches: X-2 at predicted 8 vs reference 10 seconds; X-28 at 12 vs 7;
X-38 at 34 vs 33. X-34 has a candidate at 2 seconds but a known marker at 13 seconds,
outside tolerance. X-44 produces no candidate and misses its 16-second reference.
Full-video classification of X-34 as positive therefore does not imply correct localization.

## Error analysis and local visual review

FP videos: X-14, X-16, X-31, X-39. FN video: X-44. X-10 is the sole TN.
The review inspected eight decoded context frames, covering both X-14 candidates,
one candidate each from X-16/X-31/X-39, X-34's predicted and reference times, and
X-44's missed reference. This is a small visual check, not full reannotation.
The images remain in ignored `runs/full_video_error_context.png` and are excluded from ZIPs.

- X-14: peaks at 4 and 38 seconds occur while a standing employee handles a bag at
  the workstation. The accepted video label is negative; these are false alarms
  relative to that label. Similar objects and desk activity also appear in positive
  examples, but the frames alone do not establish the cause of the model's response.
- X-16: the inspected peak shows a visitor with a bag opposite a seated employee.
  X-31's peak shows an employee moving toward the doorway. These accepted negative
  examples illustrate that familiar scene/action context is insufficient for a violation decision.
- X-39: the inspected 36-second peak shows a person near the partition and an
  employee reaching toward storage. Four separated candidates occur in this negative
  video. Merging windows does not eliminate this underlying classification problem.
- X-34: predicted and reference frames show different positions of the visitor and
  different employee hand positions. A high early window score can yield the correct
  binary video label while missing the annotated action time by 11 seconds.
- X-44: the reference frame shows the employee's hands working at the desk with no
  client visible in that selected frame. Maximum full-video score is 0.487103, below
  the fixed threshold. This observation does not establish transaction status or justify
  lowering the threshold based on this validation example.

Useful next hypotheses are whether the portrait centre crop retains relevant small
objects and whether training examples separate ordinary desk/bag movements from the
intended observable actions. Test them using training data and additional independent
examples. Do not relabel these videos or tune the threshold from this error review.

## Validation performed

- **35 tests passed** with `python -m unittest discover -s tests -v`.
- All three source notebooks ran sequentially in fresh kernels locally. EDA completed,
  notebook 02 trained a new five-epoch experiment and scanned a real video; notebook
  03 evaluated both protocols and produced reports/ZIPs.
- Final 02 and 03 sources were also checked in fresh kernels without retraining:
  02 rendered six inline figures; 03 in `full_video` mode rendered five. Both root
  and `notebooks/` working directories were exercised on Windows.
- Tests cover short/boundary windows, broken decoding, immutable model/BatchNorm,
  no annotation reads in scanner, real R3D checkpoint restoration with `weights=None`,
  merge/peak rules, one-to-one marker matching, two events on a canonical copy,
  provenance checks, overwrite rejection, JPEG schema compatibility and ZIP allowlist.
- `events.csv` and split SHA-256 remained identical. Committed notebook sources contain
  no outputs, execution counts or embedded banking images. `git diff --check` passed.
- Linux/Brev was not executed during this local run. Paths/API usage are portable;
  actual remote environment compatibility still requires its normal environment setup.

## Remaining scope

The implementation finds binary candidates and estimated times. It does not recognize
catalog action types or prove legal conditions. [Catalog coverage](catalog_coverage.md)
shows too few verified independent examples for a reliable action classifier.
`ActionRecognizer` and `link_action` provide a tested extension boundary; a real action
recognizer and applicability handling are still required before producing typed
FastAPI Findings. Public Analyzer/AnalysisResult/Finding/EvidenceFrame contracts are unchanged.

The ten-video split shares a room and potentially people/sessions; checkpoint selection
already used this validation set. Point annotations may be incomplete. The measured
results establish a working full-video research pipeline, not a production-ready detector.
