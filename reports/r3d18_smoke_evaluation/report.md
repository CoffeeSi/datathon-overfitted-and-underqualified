# Baseline evaluation

Подробный разбор ложных тревог и отсутствующих пропусков: [error_analysis.md](error_analysis.md).

Checkpoint SHA-256: `0da7d29c0e0a6a6891bf1bb5f123a89b00a3c8bb56196307e80a41a11ca67c47`; epoch 1.
Events SHA-256: `2ec181a7af08257f882aeb006b6f98b5ccb0f204470adb92d252616a67b1e062`.
Split SHA-256: `19c33f7e336b51fe5705f62d1804fb05e9c557076217aa5c0d6f986eb22f363d`.

Protocol: max probability per canonical validation video over known annotation-centred windows; threshold 0.5.

Videos: 10; windows: 10.
Precision=0.5000; recall=1.0000; F1=0.6667; accuracy=0.5000.
Confusion matrix (true rows / predicted columns; no_violation, violation): `[[0, 5], [0, 5]]`.

[Confusion matrix figure](confusion_matrix.png). Configuration, environment and full provenance: [evaluation.json](evaluation.json).

Processing: 3.771 s; preparation: 1.589 s; amortized 0.377 s/video.
Timing includes decoding/preprocessing/inference and window metrics; one cold pass, no full-video scan.

## Errors

- FP X-10: p=0.657528, true=0, predicted=1.
- FP X-14: p=0.554476, true=0, predicted=1.
- FP X-16: p=0.740437, true=0, predicted=1.
- FP X-31: p=0.769438, true=0, predicted=1.
- FP X-39: p=0.781319, true=0, predicted=1.

No visual cause is inferred from a score alone. Review evidence and annotations before proposing changes.

## Limitations

- Known positive timestamps determine evaluated windows; no blind full-video detection
- Shared room/people and unknown sessions may cross subsets
- Validation was used for checkpoint selection; no independent held-out test
- Max aggregation can increase false alarms when the number of windows increases
- Single timing pass is not a latency benchmark; per-video time is an amortized mean
