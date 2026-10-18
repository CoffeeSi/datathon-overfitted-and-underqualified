"""Annotation-free cash-sequence pipeline for the existing typed service contract."""
import math
from pathlib import Path

import torch

from .dataset import decode_frames, sample_times, sha256
from .inference import ActionObservation, ScanConfig, check_cancelled, inspect_video, link_action, prepare_scanner
from .primitive_inference import PrimitiveScanner

EVENT_TYPE = "unsecured_accepted_cash"


class CashSequencePipeline:
    """Visual receipt/departure/cash evidence determines type and time, not binary scores."""
    provider_name = "local-cash-sequence"

    def __init__(self, primitive_scanner, binary_scanner):
        if primitive_scanner.banknote_detector is None:
            raise ValueError("Cash sequence service requires the trained banknote detector")
        self.scanner = primitive_scanner
        self.binary_scanner = binary_scanner

    def binary_score(self, video, center, cancel_event):
        """Real auxiliary binary score; never gates or supplies a typed event."""
        check_cancelled(cancel_event)
        scanner = self.binary_scanner
        times = sample_times(center, video.last_time, scanner.config.window_sec, scanner.config.frames)
        inputs = scanner.transform(decode_frames(video, times)).unsqueeze(0).to(scanner.device)
        scanner.model.eval()
        with torch.inference_mode():
            logits = scanner.model(inputs)
        if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
            raise ValueError("Invalid auxiliary binary prediction")
        return float(logits.softmax(dim=1)[0, 1].cpu())

    def scan(self, video_path, *, cancel_event=None):
        raw = self.scanner.scan(video_path, cancel_event=cancel_event)
        video = inspect_video(Path(video_path))
        if sha256(video.path) != raw["video_sha256"]:
            raise ValueError("Uploaded video changed after sequence scan")
        scores = {}

        def auxiliary_score(stamp):
            if stamp not in scores:
                scores[stamp] = self.binary_score(video, stamp, cancel_event)
            return scores[stamp]

        events = []
        for event in raw["events"]:
            check_cancelled(cancel_event)
            if event["event_type"] != EVENT_TYPE:
                raise ValueError("Unsupported type from cash sequence scanner")
            stamp, confidence = event["timestamp_sec"], event["confidence"]
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid sequence score")
            description = (
                f"Возможный эпизод: передача наличных кассиру замечена на {event['receipt_sec']:.1f} с, "
                f"отсутствие клиента у стойки — с {event['departure_sec']:.1f} с; "
                f"купюры остаются на открытой рабочей поверхности до {stamp:.1f} с. "
                "Проверьте контекст: принадлежность денег и завершение учёта по видео не установлены."
            )
            action = link_action(ActionObservation(EVENT_TYPE, confidence,
                "cash-sequence:" + self.scanner.checkpoint_sha256 + ":" + self.scanner.banknote_detector.checkpoint_sha256,
                description))
            events.append(dict(event_type=EVENT_TYPE, timestamp_sec=stamp,
                type_confidence=confidence, binary_score=auxiliary_score(stamp), action=action))
        unresolved = []
        for item in raw.get("unresolved", []):
            stamp = item["timestamp_sec"]
            unresolved.append(dict(timestamp_sec=stamp, binary_score=auxiliary_score(stamp), reason="unknown"))
        check_cancelled(cancel_event)
        if sha256(video.path) != raw["video_sha256"]:
            raise ValueError("Uploaded video changed during sequence conversion")
        return dict(typed_events=events, unresolved_candidates=unresolved,
            verdict="violation" if events else "insufficient_data", recognition_scope=[EVENT_TYPE],
            provenance=dict(video_sha256=raw["video_sha256"], primitive_sha256=self.scanner.checkpoint_sha256,
                banknote_sha256=self.scanner.banknote_detector.checkpoint_sha256, binary_gate=False,
                legal_conclusion=None, requires_review=True))


def prepare_cash_sequence_pipeline(config, root):
    root = Path(root)
    if config.get("enabled_types") != [EVENT_TYPE] or type(config.get("banknote_tracking")) is not bool:
        raise ValueError("Cash sequence supports exactly one explicit type and a boolean tracking mode")
    for key in ("primitive", "banknote", "binary"):
        path = root / config[key + "_checkpoint"]
        if sha256(path) != config[key + "_sha256"]:
            raise ValueError(f"Unexpected {key} checkpoint digest")
    scanner = PrimitiveScanner(root / config["primitive_checkpoint"], config["device"],
        banknote_checkpoint=root / config["banknote_checkpoint"], banknote_tracking=config["banknote_tracking"])
    binary = prepare_scanner(root / config["binary_checkpoint"], ScanConfig(), config["device"])
    if binary.provenance["training_manifest"]["split_sha256"] != scanner.config["split_sha256"]:
        raise ValueError("Auxiliary binary model has a different split")
    return CashSequencePipeline(scanner, binary)
