"""Versioned video geometry shared by training, evaluation and binary inference."""
import math

import torch
from torchvision.models.video import R3D_18_Weights
from torchvision.transforms import functional as F


class AspectPreservingVideo:
    """Optional normalized ROI, then letterbox to 112 square; no hidden centre crop.

    RGB inputs are uint8 T,C,H,W. The KINETICS400_V1 mean/std are retained;
    padding uses their mean so padded model pixels are approximately zero.
    """
    def __init__(self, roi=None):
        roi = [0., 0., 1., 1.] if roi is None else list(roi)
        if (len(roi) != 4 or any(not math.isfinite(v) for v in roi)
                or not 0 <= roi[0] < roi[2] <= 1 or not 0 <= roi[1] < roi[3] <= 1):
            raise ValueError("ROI must be normalized [left, top, right, bottom]")
        self.roi = roi
        reference = R3D_18_Weights.KINETICS400_V1.transforms()
        self.mean, self.std = reference.mean, reference.std

    def __call__(self, frames):
        if frames.ndim != 4 or frames.shape[1] != 3 or frames.dtype != torch.uint8:
            raise ValueError("Expected uint8 RGB T,C,H,W")
        h, w = frames.shape[-2:]
        left, top, right, bottom = self.roi
        x0, y0, x1, y1 = round(left*w), round(top*h), round(right*w), round(bottom*h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("ROI is empty at this source resolution")
        frames = frames[:, :, y0:y1, x0:x1]
        h, w = frames.shape[-2:]
        scale = 112 / max(h, w)
        height, width = max(1, round(h*scale)), max(1, round(w*scale))
        resized = F.resize(frames, [height, width], antialias=True).float() / 255
        mean = torch.tensor(self.mean, device=frames.device).view(1, 3, 1, 1)
        result = mean.expand(len(frames), 3, 112, 112).clone()
        y, x = (112-height)//2, (112-width)//2
        result[:, :, y:y+height, x:x+width] = resized
        std = torch.tensor(self.std, device=frames.device).view(1, 3, 1, 1)
        return ((result - mean) / std).permute(1, 0, 2, 3).contiguous()

    def __repr__(self):
        return f"AspectPreservingVideo(v1, roi={self.roi}, output=112x112, mean={self.mean}, std={self.std})"


def video_transform(config):
    mode = config.get("preprocessing", "kinetics400_v1")
    if config.get("weights") != "KINETICS400_V1":
        raise ValueError("Unsupported normalization weights")
    if mode == "kinetics400_v1":
        if config.get("roi") is not None:
            raise ValueError("ROI requires preprocessing=aspect_letterbox_v1")
        return R3D_18_Weights.KINETICS400_V1.transforms()
    if mode == "aspect_letterbox_v1":
        return AspectPreservingVideo(config.get("roi"))
    raise ValueError(f"Unsupported preprocessing version: {mode}")
