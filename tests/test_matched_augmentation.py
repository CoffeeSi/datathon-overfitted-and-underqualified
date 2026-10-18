"""Sampling mass and fixed-update training guards for the matched ablation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

from overfitted_underqualified.ml.matched_augmentation import (
    fit_head,
    training_probabilities,
)


VARIANTS = ("original", "mirror", "warm_sepia", "cool_camera", "cctv_low_quality")


class MatchedAugmentationTests(unittest.TestCase):
    @staticmethod
    def weighted_sources():
        # Unequal numbers of timestamps and videos per class must not change
        # canonical video or class mass when the source mixture changes.
        specifications = (("X-1", 0, (2., 4., 6.)),
                          ("X-4", 1, (57., 68.)),
                          ("X-7", 1, (10.,)))
        rows = [dict(clip_id=clip, label=label, center_sec=center, variant=variant,
                     split="train")
                for clip, label, centers in specifications
                for center in centers for variant in VARIANTS]
        weights = torch.tensor([
            len(rows) / (2 * (1 if row["label"] == 0 else 2)
                         * sum(other["clip_id"] == row["clip_id"] for other in rows))
            for row in rows
        ], dtype=torch.float64)
        return rows, weights

    def test_mixture_preserves_classes_canonical_videos_and_both_alias_times(self):
        rows, weights = self.weighted_sources()
        probabilities = training_probabilities(rows, weights, .8)
        self.assertEqual(probabilities.shape, weights.shape)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.)
        for variant in VARIANTS:
            mass = sum(float(p) for p, row in zip(probabilities, rows)
                       if row["variant"] == variant)
            self.assertAlmostEqual(mass, .8 if variant == "original" else .05)
        for label in (0, 1):
            self.assertAlmostEqual(sum(float(p) for p, row in zip(probabilities, rows)
                                       if row["label"] == label), .5)
        for clip, expected in (("X-1", .5), ("X-4", .25), ("X-7", .25)):
            self.assertAlmostEqual(sum(float(p) for p, row in zip(probabilities, rows)
                                       if row["clip_id"] == clip), expected)
        for center in (57., 68.):
            self.assertAlmostEqual(sum(float(p) for p, row in zip(probabilities, rows)
                                       if row["clip_id"] == "X-4" and row["center_sec"] == center), .125)

    def test_control_samples_only_originals_without_losing_canonical_mass(self):
        rows, weights = self.weighted_sources()
        probabilities = training_probabilities(rows, weights, 1.)
        self.assertAlmostEqual(float(probabilities.sum()), 1.)
        self.assertTrue(all(float(p) == 0. for p, row in zip(probabilities, rows)
                            if row["variant"] != "original"))
        self.assertAlmostEqual(sum(float(p) for p, row in zip(probabilities, rows)
                                   if row["label"] == 1), .5)

    def test_invalid_weights_and_missing_original_are_rejected(self):
        rows, weights = self.weighted_sources()
        for bad in (weights[:-1], torch.zeros_like(weights),
                    weights.clone().index_fill(0, torch.tensor([0]), float("nan")),
                    weights.clone().index_fill(0, torch.tensor([0]), -1.)):
            with self.subTest(weights=bad):
                with self.assertRaises(ValueError):
                    training_probabilities(rows, bad, .8)
        missing = next(index for index, row in enumerate(rows)
                       if row["variant"] == "original")
        keep = [index for index in range(len(rows)) if index != missing]
        with self.assertRaises(ValueError):
            training_probabilities([rows[index] for index in keep], weights[keep], .8)
        for fraction in (-.1, 1.1, float("nan")):
            with self.subTest(original_fraction=fraction):
                with self.assertRaises(ValueError):
                    training_probabilities(rows, weights, fraction)

    @staticmethod
    def head_inputs():
        features = torch.tensor([[-2., -1.], [-1., -2.], [-2., -2.], [-1., -1.],
                                 [2., 1.], [1., 2.], [2., 2.], [1., 1.]])
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        probabilities = torch.full((8,), 1. / 8, dtype=torch.float64)
        initial = {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}
        return features, labels, probabilities, initial

    @staticmethod
    def fit(features, labels, probabilities, initial):
        return fit_head(features, labels, probabilities, initial, steps=7,
                        batch_size=4, learning_rate=.01, weight_decay=.0001,
                        seed=42, device="cpu")

    def test_fixed_update_count_and_same_seed_are_reproducible(self):
        features, labels, probabilities, initial = self.head_inputs()
        first, optimizer, history = self.fit(features, labels, probabilities, initial)
        second, second_optimizer, repeated = self.fit(features, labels, probabilities, initial)
        self.assertIsInstance(first, nn.Linear)
        self.assertEqual((first.in_features, first.out_features), (2, 2))
        self.assertEqual(len(history), 7)
        self.assertEqual(len(repeated), 7)
        self.assertEqual(int(optimizer.state[first.weight]["step"]), 7)
        self.assertEqual(int(second_optimizer.state[second.weight]["step"]), 7)
        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertTrue(torch.equal(first.bias, second.bias))
        self.assertFalse(torch.equal(first.weight.detach(), initial["weight"]))
        with torch.inference_mode():
            self.assertTrue(torch.equal(first(features).argmax(1), labels))

    def test_training_does_not_mutate_shared_initialization_or_cached_inputs(self):
        features, labels, probabilities, initial = self.head_inputs()
        snapshots = [tensor.clone() for tensor in (features, labels, probabilities)]
        initial_copy = {key: tensor.clone() for key, tensor in initial.items()}
        self.fit(features, labels, probabilities, initial)
        for tensor, before in zip((features, labels, probabilities), snapshots):
            self.assertTrue(torch.equal(tensor, before))
        for key in initial:
            self.assertTrue(torch.equal(initial[key], initial_copy[key]))

    def test_invalid_sampling_probabilities_fail_before_training(self):
        features, labels, probabilities, initial = self.head_inputs()
        invalid = (probabilities[:-1], torch.zeros_like(probabilities),
                   probabilities.clone().index_fill(0, torch.tensor([0]), -1.),
                   probabilities.clone().index_fill(0, torch.tensor([0]), float("nan")),
                   probabilities.clone().index_fill(0, torch.tensor([0]), float("inf")))
        for bad in invalid:
            with self.subTest(probabilities=bad):
                with self.assertRaises(ValueError):
                    self.fit(features, labels, bad, initial)


if __name__ == "__main__":
    unittest.main()
