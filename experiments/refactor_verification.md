# EDA / R3D-18 refactor verification — 2026-10-17

This is a code-organization and execution check, not a new model-quality claim.
The historical [smoke-run record](r3d18_smoke_result.md) and its artifacts remain unchanged.

## Structure

- EDA implementations now live in `src/overfitted_underqualified/eda/`:
  `video`, `annotations`, `split`, `plots`, and `reports`.
- `01_eda.ipynb` contains settings, ordered calls and interpretation. Tests import
  Python modules instead of extracting definitions from notebook JSON.
- `02_training.ipynb` uses the same preparation/model/training phases as the CLI.
- `03_evaluation.ipynb` restores a checkpoint, validates provenance, evaluates the
  fixed validation windows and displays errors without training or threshold tuning.

## Executed checks

Local Windows repository `.venv`, NVIDIA GeForce RTX 4050 Laptop GPU:

- `python -m unittest discover -s tests -v`: 15 tests passed. Coverage includes real
  decoding/preprocessing, aliases and distinct event times, source/hash drift,
  duplicate leakage, split preservation, frozen BatchNorm and inference without
  parameter updates. The checkpoint test asserts construction with `weights=None`,
  exact restored tensors, explicit path relocation and provenance rejection.
- All three notebooks executed sequentially in separate fresh kernels. EDA and
  evaluation were run from the repository root; training from `notebooks/`.
  After moving the final split checks into the module, final EDA was also executed
  from `notebooks/` in another fresh kernel.
- Training used one epoch and a new directory:
  `runs/refactor_verification/training/`. It preserved both earlier runs and used
  the complete fixed train/validation sets. Its metrics and every model-state
  tensor exactly matched the original smoke checkpoint on this local environment.
- Evaluation restored the historical `runs/r3d18_smoke/best.pt` using explicit current
  data paths. No new pretrained download or optimizer was needed.
- SHA-256 checks confirmed that `events.csv`, `split_proposal.csv`, and every file
  in the two previous smoke-run directories were unchanged (16 protected files).
  The 30/10 assignment, original-ID coverage, hashes and X-4/X-47 markers (57/68 s)
  were checked again through the ML loader.
- Source notebooks validate as notebook format 4 and contain no function definitions,
  outputs, execution counts or embedded attachments. Executed copies remain locally
  under ignored `runs/refactor_verification/`.
- Training CLI `--help` and `git diff --check` passed. No commit or push was made.

Linux/Brev execution was not repeated here. Path discovery uses `pathlib`; evaluation
accepts explicit relocated input paths instead of depending on historical Windows paths.
Bitwise reproducibility across devices or library versions is not claimed.

## Remaining limitations

The task remains annotation-centred window classification, not full-video event
detection. Both subsets may share the room, people and unknown recording sessions.
Validation is small and used for checkpoint selection, not an independent final test.
The original smoke model predicted violation for every validation window. Reproducing
that result confirms the refactor preserved behavior; it does not establish useful quality.
