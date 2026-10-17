# Brev L4 smoke-check result

Recorded from terminal output supplied by the Brev operator on 2026-10-17.
Settings: [brev_l4_smoke.yaml](brev_l4_smoke.yaml).

The separate `nvidia-smi` output identified GPU 0 as NVIDIA L4, driver 595.91.07,
and driver-reported CUDA version 13.2. The check below used the CUDA 12.6
PyTorch wheels pinned in `requirements-ml.txt`.

```text
Python: 3.11.9
PyTorch: 2.13.0+cu126
torchvision: 0.28.0+cu126
PyTorch CUDA: 12.6
CUDA available: True
GPU: NVIDIA L4
PASS: forward, backward, and one optimizer step; loss: 0.4047415852546692
```

The reported loss is from this synthetic run; its exact value is not an acceptance threshold.
