"""Whole-video primitive scan. Inputs are a video and checkpoint, never annotations."""
import argparse
import math
from pathlib import Path
import time

import torch

from .action_train import write_json
from .cash_sequence import CashSequence
from .dataset import decode_frames, sha256
from .inference import inspect_video, synchronize, check_cancelled
from .primitive_data import CLASSES, PrimitiveTransform, causal_times
from .primitive_model import load_primitive_checkpoint
from .train import seed_everything
from ..rules.catalog import source_links_for, CATALOG_PATH


class PrimitiveScanner:
    def __init__(self, checkpoint_path, device="cpu", *, banknote_checkpoint=None, banknote_tracking=False):
        if banknote_tracking and banknote_checkpoint is None:
            raise ValueError("Banknote tracking requires a banknote checkpoint")
        self.model, self.checkpoint, self.checkpoint_sha256 = load_primitive_checkpoint(checkpoint_path, device)
        self.config = self.checkpoint["config"]
        self.device = torch.device(device)
        seed_everything(self.config["seed"], self.config["cpu_threads"])
        self.transform = PrimitiveTransform(self.config)
        self.banknote_detector = None
        self.banknote_tracking_config = None
        if banknote_checkpoint is not None:
            from .banknote_model import load
            self.banknote_detector = load(banknote_checkpoint, device)
            if self.banknote_detector.config["split_sha256"] != self.config["split_sha256"]:
                raise ValueError("Primitive and banknote checkpoints use different splits")
            if banknote_tracking:
                from .banknote_tracking import BanknoteTracker
                self.banknote_tracking_config = dict(BanknoteTracker().config)

    def scan(self, video_path, *, cancel_event=None):
        start = time.perf_counter()
        record = inspect_video(Path(video_path))
        digest = sha256(record.path)
        step = self.config["scan_step_sec"]
        stamps = [i * step for i in range(math.floor(record.last_time / step) + 1)]
        if record.last_time > stamps[-1] + 1e-8:
            stamps.append(record.last_time)
        sequence = CashSequence(**self.config["sequence"])
        tracker = None
        if getattr(self, "banknote_tracking_config", None) is not None:
            from .banknote_tracking import BanknoteTracker
            tracker = BanknoteTracker()
            if tracker.config != self.banknote_tracking_config:
                raise ValueError("Tracking configuration changed since scanner initialization")
        observations = []
        self.model.eval()
        # Small sequential batches avoid retaining decoded full-resolution video in RAM.
        with torch.inference_mode():
            for offset in range(0, len(stamps), 2):
                check_cancelled(cancel_event)
                selected = stamps[offset:offset + 2]
                decoded = [decode_frames(record, causal_times(t, record.last_time,
                           self.config["window_sec"])) for t in selected]
                batch = torch.stack([self.transform(frames) for frames in decoded]).to(self.device)
                probabilities = self.model(batch).sigmoid().cpu().tolist()
                for stamp, values, frames in zip(selected, probabilities, decoded):
                    scores = dict(zip(CLASSES, values))
                    original_scores = dict(scores)
                    detections = None
                    if self.banknote_detector is not None:
                        rgb = frames[-1].permute(1, 2, 0).numpy()
                        detections = self.banknote_detector.predict(rgb)
                        if tracker is not None:
                            tracked = tracker.update(rgb, stamp, detections["objects"])
                            detections = dict(detections, raw_detection_score=detections["score"],
                                              detector_objects=detections["objects"],
                                              objects=tracked["objects"],
                                              boxes=[o["box"] for o in tracked["objects"]],
                                              score=max((o["score"] for o in tracked["objects"]), default=0.),
                                              tracking_evidence=tracked["tracking_evidence"],
                                              absence_certified=False)
                        scores["banknotes_on_counter"] = detections["score"]
                    trace = sequence.update(stamp, scores)
                    observation = dict(**trace, probabilities=scores)
                    if detections is not None:
                        observation.update(r3d_probabilities=original_scores, banknotes=detections)
                    observations.append(observation)
        synchronize(self.device)
        if sha256(record.path) != digest:
            raise ValueError("Video changed during primitive scan")
        events = [dict(event, rule_candidates=source_links_for(event["event_type"])) for event in sequence.events]
        uncertain = sum(any(v is None for v in row["state"].values()) for row in observations)
        result = dict(schema="primitive_scan_v1", video_sha256=digest, checkpoint_sha256=self.checkpoint_sha256,
                    catalog_sha256=sha256(CATALOG_PATH), clip_id=record.clip_id,
                    duration_sec=record.frame_count / record.fps, fps=record.fps, frame_count=record.frame_count,
                    processing_sec=time.perf_counter() - start, observations=observations, events=events,
                    unresolved=sequence.unresolved, uncertain_observations=uncertain,
                    status="research_candidates" if events else "insufficient_data",
                    note="No candidate does not certify absence. No legal decision, verified cash identity/ownership or independent test.")
        if self.banknote_detector is not None:
            result["banknote_checkpoint_sha256"] = self.banknote_detector.checkpoint_sha256
            result["schema"] = "banknote_sequence_scan_v1"
        if tracker is not None:
            result["banknote_tracking_config"] = dict(self.banknote_tracking_config)
            result["banknote_score_semantics"] = "Minimum of originating model score and fresh image-match score; not a calibrated probability."
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--banknote-checkpoint", type=Path, help="Optional reviewed counter-banknote detector")
    p.add_argument("--track-banknotes", action="store_true", help="Require fresh image matching between detector observations")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = PrimitiveScanner(args.checkpoint, args.device, banknote_checkpoint=args.banknote_checkpoint,
                              banknote_tracking=args.track_banknotes).scan(args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
