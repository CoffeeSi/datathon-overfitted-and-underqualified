"""Geometry parity, coverage negatives and frozen-cache training equivalence."""
from pathlib import Path
import io
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights

from overfitted_underqualified.ml.dataset import VideoWindowDataset, sample_times
from overfitted_underqualified.ml.features import WeightedWindows, cache_frozen_loaders
from overfitted_underqualified.ml.preprocessing import AspectPreservingVideo, video_transform
from overfitted_underqualified.ml.train import run_epoch
from overfitted_underqualified.ml.action_train import action_epoch
from overfitted_underqualified.ml.inference import ScanConfig, merge_candidates


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm3d(3)
        self.bn.requires_grad_(False)
        self.fc = nn.Linear(3, 2)
    def forward(self, x):
        return self.fc(self.bn(x).mean((2, 3, 4)))


class TrainingImprovementsTests(unittest.TestCase):
    def test_action_feature_cache_preserves_masks_gradients_and_backbone(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        raw=Tiny();cached=Tiny();cached.load_state_dict(raw.state_dict())
        samples=[dict(video=torch.rand(3,16,8,8),targets=torch.tensor([1.,0.]),
                      mask=torch.tensor([1.,float(i%2)]),clip_id=str(i),center_sec=2.) for i in range(4)]
        loaders={s:DataLoader(samples,batch_size=2,shuffle=False) for s in ('train','validation')}
        cached_loaders=cache_frozen_loaders(cached,loaders,'cpu',2,42)
        self.assertEqual(next(iter(cached_loaders['validation']))['mask'].shape,(2,2))
        # A single batch removes order differences while comparing identical losses.
        raw_loader=DataLoader(samples,batch_size=4,shuffle=False)
        cached_samples=list(cached_loaders['validation'].dataset)
        cached_loader=DataLoader(cached_samples,batch_size=4,shuffle=False)
        bn=raw.bn.running_mean.clone()
        for model,loader in ((raw,raw_loader),(cached,cached_loader)):
            action_epoch(model,loader,'cpu',['one','two'],.7,torch.optim.SGD(model.fc.parameters(),lr=.1),torch.tensor([2.,1.]))
        torch.testing.assert_close(raw.fc.weight,cached.fc.weight,rtol=1e-6,atol=1e-7)
        self.assertTrue(torch.equal(cached.bn.running_mean,bn))

    def test_peak_policy_retains_separate_peaks_inside_one_long_alarm(self):
        windows = [dict(window_index=i, start_sec=float(i*2), end_sec=float(i*2+4),
                        center_sec=float(i*2+2), probability=p) for i, p in enumerate((.8,.6,.7,.9,.6,.75))]
        self.assertEqual(len(merge_candidates(windows, ScanConfig())), 1)
        peaks = merge_candidates(windows, ScanConfig(candidate_policy="window_peaks", peak_separation_sec=4.))
        self.assertEqual([p["timestamp_sec"] for p in peaks], [2.,8.,12.])
        self.assertTrue(all(p["event_type"] is None for p in peaks))
        flat = [{**w, "probability": .7} for w in windows]
        self.assertEqual(len(merge_candidates(flat, ScanConfig(candidate_policy="window_peaks"))), 1)
        self.assertEqual(merge_candidates([{**w, "probability": .1} for w in windows], ScanConfig(candidate_policy="window_peaks")), [])

    def test_legacy_transform_is_unchanged_and_invalid_versions_fail(self):
        frames = torch.randint(0, 256, (16, 3, 160, 80), dtype=torch.uint8)
        expected = R3D_18_Weights.KINETICS400_V1.transforms()(frames)
        torch.testing.assert_close(video_transform(dict(weights="KINETICS400_V1"))(frames), expected, rtol=0, atol=0)
        with self.assertRaises(ValueError): video_transform(dict(weights="KINETICS400_V1", preprocessing="typo"))

    def test_letterbox_preserves_shapes_and_padding_is_zero(self):
        frames = torch.full((16, 3, 200, 100), 255, dtype=torch.uint8)
        result = AspectPreservingVideo()(frames)
        self.assertEqual(tuple(result.shape), (3, 16, 112, 112))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result[:, :, :, :28], torch.zeros_like(result[:, :, :, :28])))
        self.assertTrue((result[:, :, :, 28:84] > 0).all())
        with self.assertRaises(ValueError): AspectPreservingVideo([0, .8, 1, .3])
        with self.assertRaises(ValueError): AspectPreservingVideo([0, 0, float("nan"), 1])

    def test_coverage_includes_boundaries_and_preserves_positive_alias_times(self):
        positive = SimpleNamespace(clip_id="pos", split="train", label=1, event_times=(5., 12.), last_time=20.)
        negative = SimpleNamespace(clip_id="neg", split="train", label=0, event_times=(), last_time=20.)
        data = VideoWindowDataset([positive, negative], "train", None, 4., 16, 5, negative_sampling="coverage")
        centers = [t for r, t in data.samples if not r.label]
        self.assertTrue(all(type(t) is float for t in centers))
        self.assertEqual(centers, [2., 6., 10., 14., 18.])
        self.assertEqual(sample_times(centers[0], 20., 4.)[0], 0)
        self.assertEqual(sample_times(centers[-1], 20., 4.)[-1], 20)
        weighted = WeightedWindows(data)
        totals = [sum(w for (r, _), w in zip(data.samples, weighted.weights) if r.label == label) for label in (0, 1)]
        self.assertAlmostEqual(totals[0], totals[1])
        self.assertEqual([t for r, t in data.samples if r.label], [5., 12.])
        buffer = io.BytesIO()
        torch.save(dict(centers=centers), buffer)
        buffer.seek(0)
        self.assertEqual(torch.load(buffer, weights_only=True)["centers"], centers)

    def test_cached_fc_gradients_match_pixels_and_bn_does_not_change(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        samples = [dict(video=torch.rand(3, 16, 8, 8), label=torch.tensor(i%2), clip_id=str(i), center_sec=1., sample_weight=1.) for i in range(4)]
        original, cached_model = Tiny(), Tiny()
        cached_model.load_state_dict(original.state_dict())
        bn_before = original.bn.running_mean.clone()
        loaders = {subset: DataLoader(samples, batch_size=2, shuffle=subset == "train", generator=torch.Generator().manual_seed(42)) for subset in ("train", "validation")}
        cached = cache_frozen_loaders(cached_model, loaders, "cpu", 2, 42)
        self.assertTrue(torch.equal(cached_model.bn.running_mean, bn_before))
        for model, loader in ((original, loaders["train"]), (cached_model, cached["train"])):
            run_epoch(model, loader, "cpu", torch.optim.SGD(model.fc.parameters(), lr=.1), True)
        torch.testing.assert_close(original.fc.weight, cached_model.fc.weight, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(original.fc.bias, cached_model.fc.bias, rtol=1e-6, atol=1e-7)
        self.assertTrue(torch.equal(cached_model.bn.running_mean, bn_before))
        cached_model.bn.requires_grad_(True)
        with self.assertRaises(ValueError): cache_frozen_loaders(cached_model, loaders, "cpu", 2, 42)


if __name__ == "__main__": unittest.main()
