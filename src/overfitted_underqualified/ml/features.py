"""Per-run frozen feature cache and sample weights; no validation-driven selection."""
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class WeightedWindows(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        video_counts = Counter(r.clip_id for r, _ in dataset.samples)
        class_videos = {label: len({r.clip_id for r, _ in dataset.samples if r.label == label}) for label in (0, 1)}
        if not all(class_videos.values()):
            raise ValueError("Weighted training requires both classes")
        n = len(dataset)
        self.weights = [n / (2 * class_videos[r.label] * video_counts[r.clip_id]) for r, _ in dataset.samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return dict(self.dataset[index], sample_weight=self.weights[index])


def cache_frozen_loaders(model, loaders, device, batch_size, seed):
    """Exactly the same fixed backbone, preprocessing and windows; train only fc.

    Features are held in CPU memory for this run only. No cache can cross a data,
    preprocessing, split or backbone change. Intended for deterministic transforms.
    """
    if any(p.requires_grad for name, p in model.named_parameters() if not name.startswith("fc.")):
        raise ValueError("Feature cache requires a fully frozen backbone")
    head = model.fc
    cached = {}
    model.eval()
    model.fc = nn.Identity()
    try:
        with torch.inference_mode():
            for subset, loader in loaders.items():
                samples = []
                sequential = DataLoader(loader.dataset, batch_size=min(batch_size, 2), shuffle=False, num_workers=0,
                                        generator=torch.Generator().manual_seed(seed))
                for batch in sequential:
                    features = model(batch["video"].to(device)).cpu()
                    if features.ndim != 2 or features.shape[1] != head.in_features or not torch.isfinite(features).all():
                        raise ValueError("Invalid cached backbone features")
                    for i in range(len(features)):
                        targets = ({"targets": batch["targets"][i], "mask": batch["mask"][i]}
                                   if "targets" in batch else {"label": batch["label"][i]})
                        samples.append(dict(features=features[i], **targets,
                            clip_id=batch["clip_id"][i], center_sec=batch["center_sec"][i],
                            **({"sample_weight": batch["sample_weight"][i]} if "sample_weight" in batch else {})))
                cached[subset] = DataLoader(samples, batch_size=batch_size, shuffle=subset == "train", num_workers=0,
                                           generator=torch.Generator().manual_seed(seed))
    finally:
        model.fc = head
        model.eval()
    return cached
