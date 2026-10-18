"""Typed window decisions and per-action localization. No annotation lookup at inference."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .action_train import load_action_checkpoint
from .preprocessing import video_transform
from .dataset import decode_frames, sha256
from .inference import (ActionObservation, ScanConfig, check_cancelled, inspect_video,
                        link_action, prepare_scanner, synchronize)
from ..rules.catalog import event_by_type, supported_event_types


class R3DActionRecognizer:
    """Multi-label frozen model with fixed positive/background thresholds and abstention.

    'background' means low scores for the deployed classes, not proof of no action.
    Unknown/open-set detection is not guaranteed by low neural-network confidence.
    """
    def __init__(self, checkpoint_path, device="cpu", *, enabled_types):
        self.model, self.checkpoint, self.digest = load_action_checkpoint(checkpoint_path, device)
        self.device = torch.device(device)
        self.classes = self.checkpoint["class_names"]
        self.enabled_types = set(enabled_types)
        if (not self.enabled_types or not self.enabled_types <= set(self.checkpoint["validated_types"])
                or not self.enabled_types <= set(self.classes) or not self.enabled_types <= supported_event_types()):
            raise ValueError("Enable only trained types with positive and negative validation support")
        config = self.checkpoint["config"]
        self.threshold, self.background_ceiling = config["threshold"], config["background_ceiling"]
        if not 0 <= self.background_ceiling < self.threshold <= 1:
            raise ValueError("Invalid action decision thresholds")
        self.window_sec = config["window_sec"]
        self.transform = video_transform(config)
        self.provenance = dict(checkpoint_sha256=self.digest, checkpoint_path=str(Path(checkpoint_path).resolve()),
                               class_names=self.classes, enabled_types=sorted(self.enabled_types),
                               threshold=self.threshold, background_ceiling=self.background_ceiling,
                               preprocessing=config.get("preprocessing", "kinetics400_v1"), roi=config.get("roi"),
                               schema=self.checkpoint["schema"])

    def predict_window(self, video, window):
        times = np.linspace(window["start_sec"], window["end_sec"], 16)
        tensor = self.transform(decode_frames(video, times))
        if tuple(tensor.shape) != (3, 16, 112, 112) or not torch.isfinite(tensor).all():
            raise ValueError("Invalid action input")
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(tensor.unsqueeze(0).to(self.device))
        if tuple(logits.shape) != (1, len(self.classes)) or not torch.isfinite(logits).all():
            raise ValueError("Invalid action logits")
        probabilities = dict(zip(self.classes, logits.sigmoid()[0].cpu().tolist()))
        observations = []
        for name, score in probabilities.items():
            if name in self.enabled_types and score >= self.threshold:
                event = event_by_type(name)
                description = event["visible_action"]
                observations.append(ActionObservation(name, score, "r3d18-actions:" + self.digest,
                                                       f"Возможное действие по оценке модели: {description} Требуется просмотр контекста."))
        status = "typed" if observations else ("background" if max(probabilities.values()) <= self.background_ceiling else "unknown")
        uncertain = any(self.background_ceiling < score < self.threshold or
                        (name not in self.enabled_types and score >= self.threshold)
                        for name, score in probabilities.items())
        return dict(status=status, observations=observations, probabilities=probabilities,
                    unresolved=not observations or uncertain)

    def recognize_many(self, video_path, candidate):
        return self.predict_window(inspect_video(video_path), candidate)["observations"]

    def recognize(self, video_path, candidate):
        observations = self.recognize_many(video_path, candidate)
        # Compatibility with the old single-observation protocol; never drop a second type.
        return observations[0] if len(observations) == 1 else None


def merge_typed_windows(windows, gap_sec=0.0):
    """Merge only consecutive windows of the same action; an abstention breaks continuity."""
    groups = []
    active = {}
    for window in sorted(windows, key=lambda w: w["window_index"]):
        current = {a["event_type"]: a for a in window.get("actions", [])}
        active = {name: group for name, group in active.items() if name in current}
        for name, action in current.items():
            if name not in supported_event_types():
                raise ValueError("Unsupported type in typed window")
            previous = active.get(name)
            if (previous is None or window["window_index"] != previous["window_indices"][-1] + 1
                    or window["start_sec"] > previous["end_sec"] + gap_sec + 1e-9):
                previous = dict(start_sec=window["start_sec"], end_sec=window["end_sec"],
                                timestamp_sec=window["center_sec"], binary_score=window["probability"],
                                type_confidence=action["confidence"], action=action,
                                event_type=name, window_indices=[])
                groups.append(previous)
                active[name] = previous
            previous["end_sec"] = max(previous["end_sec"], window["end_sec"])
            previous["window_indices"].append(window["window_index"])
            if action["confidence"] > previous["type_confidence"]:
                previous.update(timestamp_sec=window["center_sec"], type_confidence=action["confidence"],
                                binary_score=window["probability"], action=action)
    return [dict(candidate_id=i, **group) for i, group in enumerate(sorted(groups, key=lambda g: (g["timestamp_sec"], g["event_type"])))]


class TypedVideoPipeline:
    def __init__(self, scanner, recognizer):
        if abs(scanner.config.window_sec - recognizer.window_sec) > 1e-9:
            raise ValueError("Binary and action window lengths must match")
        self.scanner, self.recognizer = scanner, recognizer

    def scan(self, video_path, *, cancel_event=None):
        started = time.perf_counter()
        result = self.scanner.scan(video_path, cancel_event=cancel_event)
        video = inspect_video(video_path)
        unresolved = []
        for window in result["windows"]:
            check_cancelled(cancel_event)
            window["actions"] = []
            if window["probability"] < self.scanner.config.threshold:
                window["action_status"] = "binary_negative"
                continue
            decision = self.recognizer.predict_window(video, window)
            window["action_status"] = decision["status"]
            window["action_probabilities"] = decision["probabilities"]
            window["actions"] = [link_action(o) for o in decision["observations"]]
            if decision.get("unresolved", not window["actions"]):
                unresolved.append(dict(window_index=window["window_index"], timestamp_sec=window["center_sec"],
                                       binary_score=window["probability"],
                                       reason="background" if decision["status"] == "background" else "unknown"))
        result["typed_events"] = merge_typed_windows(result["windows"], self.scanner.config.merge_gap_sec)
        result["unresolved_candidates"] = unresolved
        result["verdict"] = ("violation" if result["typed_events"] else
                             "insufficient_data" if unresolved else "no_violation")
        result["provenance"] = dict(binary=self.scanner.provenance, action=self.recognizer.provenance,
                                    binary_gate=True, legal_conclusion=None)
        if sha256(video.path) != result["sha256"]:
            raise ValueError("Video changed during action inference")
        synchronize(self.recognizer.device)
        result["processing_sec"] = time.perf_counter() - started
        return result


def prepare_typed_pipeline(binary_checkpoint, action_checkpoint, *, enabled_types, scan=None, device="cpu"):
    if not enabled_types:
        raise ValueError("No evaluated action types explicitly enabled")
    for path in (binary_checkpoint, action_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    scanner = prepare_scanner(binary_checkpoint, scan or ScanConfig(), device)
    recognizer = R3DActionRecognizer(action_checkpoint, device, enabled_types=enabled_types)
    return TypedVideoPipeline(scanner, recognizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-checkpoint", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--enabled-type", action="append", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pipeline = prepare_typed_pipeline(args.binary_checkpoint, args.action_checkpoint,
                                      enabled_types=args.enabled_type, device=args.device)
    result = pipeline.scan(args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
