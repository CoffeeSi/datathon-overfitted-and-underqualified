"""Short-lived image evidence for banknote detector flicker, without model guessing.

A track must match a textured object patch in every newly supplied image. It is
discarded on mismatch, time gaps, geometry changes or expiry. This propagates the
detector's object hypothesis; it neither proves banknote identity nor absence.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

SCHEMA = "banknote_image_tracker_v1"


@dataclass
class _Track:
    track_id: int
    box: tuple
    original_box: tuple
    template: np.ndarray
    origin_sec: float
    last_detection_sec: float
    last_sec: float
    model_score: float
    detection_area_px: int | None


def _iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    a = (first[2] - first[0]) * (first[3] - first[1])
    b = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (a + b - intersection) if intersection else 0.


class BanknoteTracker:
    """Fixed, conservative template tracker; inputs are RGB uint8 and xyxy boxes.

    ``objects`` are the existing detector's component dictionaries (``box`` and
    ``score``). Returned score is a conservative combined evidence score, not a
    newly calibrated model probability. Original model and match scores remain
    separate. No frame, timestamp or label source is read by this class.
    """
    def __init__(self, *, detection_threshold=.7, match_threshold=.9,
                 max_displacement_fraction=.015, max_total_displacement_fraction=.03,
                 min_patch_contrast=8., max_rgb_difference=12.,
                 max_gap_sec=2., max_age_sec=10., cut_rgb_difference=35., max_tracks=16):
        values = (detection_threshold, match_threshold, max_displacement_fraction,
                  max_total_displacement_fraction, min_patch_contrast, max_rgb_difference,
                  max_gap_sec, max_age_sec, cut_rgb_difference)
        if (any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values)
                or not 0 < detection_threshold < 1 or not 0 < match_threshold <= 1
                or max_displacement_fraction > .1 or max_total_displacement_fraction > .2
                or type(max_tracks) is not int or max_tracks < 1):
            raise ValueError("Invalid banknote image tracker settings")
        self.config = dict(schema=SCHEMA, detection_threshold=detection_threshold, match_threshold=match_threshold,
                           max_displacement_fraction=max_displacement_fraction,
                           max_total_displacement_fraction=max_total_displacement_fraction,
                           min_patch_contrast=min_patch_contrast, max_rgb_difference=max_rgb_difference,
                           max_gap_sec=max_gap_sec, max_age_sec=max_age_sec,
                           cut_rgb_difference=cut_rgb_difference, max_tracks=max_tracks)
        self.schema = SCHEMA
        self.tracks, self.shape, self.last_sec, self.preview = [], None, None, None
        self.next_id = 1

    def reset(self):
        self.tracks, self.shape, self.last_sec, self.preview = [], None, None, None

    @staticmethod
    def _pixel_box(box, shape):
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
            raise ValueError("Tracker boxes must be normalized full-frame xyxy")
        height, width = shape[:2]
        return (max(0, math.floor(box[0] * width)), max(0, math.floor(box[1] * height)),
                min(width, math.ceil(box[2] * width)), min(height, math.ceil(box[3] * height)))

    @staticmethod
    def _normalize(box, shape):
        height, width = shape[:2]
        return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]

    def _trackable_patch(self, rgb, box):
        left, top, right, bottom = box
        if min(right - left, bottom - top) < 3:
            return None
        patch = rgb[top:bottom, left:right].copy()
        contrast = float(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).std())
        return patch if contrast >= self.config["min_patch_contrast"] else None

    def _match(self, rgb, stamp, track):
        if stamp - track.last_detection_sec > self.config["max_age_sec"]:
            return None, "last_detection_expired"
        if stamp - track.last_sec > self.config["max_gap_sec"]:
            return None, "image_gap"
        height, width = rgb.shape[:2]
        left, top, right, bottom = track.box
        dx = max(1, math.ceil(width * self.config["max_displacement_fraction"]))
        dy = max(1, math.ceil(height * self.config["max_displacement_fraction"]))
        search_left, search_top = max(0, left - dx), max(0, top - dy)
        search_right, search_bottom = min(width, right + dx), min(height, bottom + dy)
        search = rgb[search_top:search_bottom, search_left:search_right]
        response = cv2.matchTemplate(search, track.template, cv2.TM_CCOEFF_NORMED)
        _, similarity, _, position = cv2.minMaxLoc(response)
        if not math.isfinite(similarity) or similarity < self.config["match_threshold"]:
            return None, "template_mismatch"
        candidate_left, candidate_top = search_left + position[0], search_top + position[1]
        th, tw = track.template.shape[:2]
        box = (candidate_left, candidate_top, candidate_left + tw, candidate_top + th)
        original = track.original_box
        if (abs(box[0] - original[0]) > width * self.config["max_total_displacement_fraction"]
                or abs(box[1] - original[1]) > height * self.config["max_total_displacement_fraction"]):
            return None, "excessive_displacement"
        patch = self._trackable_patch(rgb, box)
        if patch is None:
            return None, "flat_or_occluded_patch"
        difference = float(np.abs(patch.astype(np.float32) - track.template.astype(np.float32)).mean())
        if difference > self.config["max_rgb_difference"]:
            return None, "rgb_mismatch"
        # Keep the originating template: updating it can slowly drift onto desk
        # texture after the banknote disappears. Every match reuses image evidence.
        tracked = _Track(track.track_id, box, track.original_box, track.template,
                         track.origin_sec, track.last_detection_sec, stamp, track.model_score, track.detection_area_px)
        evidence = dict(box=self._normalize(box, rgb.shape), score=min(track.model_score, float(similarity)),
                        evidence_type="tracked", evidence_method="template_and_rgb_image_match", model_score=track.model_score,
                        tracking_score=float(similarity), rgb_mean_absolute_difference=difference,
                        originating_detection_sec=track.origin_sec, last_detection_sec=track.last_detection_sec,
                        matched_at_sec=stamp, area_px=track.detection_area_px,
                        area_px_source="last detector component in its resized ROI",
                        track_id=track.track_id, absence_certified=False)
        return (tracked, evidence), None

    def update(self, rgb, time_sec, objects):
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 3):
            raise ValueError("Tracker expects an RGB uint8 HWC image")
        if (not isinstance(time_sec, (int, float)) or not math.isfinite(time_sec) or time_sec < 0
                or self.last_sec is not None and time_sec <= self.last_sec):
            raise ValueError("Tracker timestamps must be finite and strictly increasing")
        height, width = rgb.shape[:2]
        preview = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        reason = None
        if self.shape is not None and self.shape != rgb.shape:
            reason = "geometry_changed"
        elif self.last_sec is not None and time_sec - self.last_sec > self.config["max_gap_sec"]:
            reason = "image_gap"
        elif self.preview is not None and float(np.abs(preview.astype(np.float32) - self.preview).mean()) > self.config["cut_rgb_difference"]:
            reason = "scene_cut_or_large_image_change"
        evidence = []
        if reason:
            evidence.extend(dict(track_id=track.track_id, status="discarded", reason=reason) for track in self.tracks)
            self.tracks = []
        current, detected_boxes, associations, associated_ids = [], [], [], set()
        for source in objects:
            score = source.get("score")
            if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Detector object requires a finite score in [0,1]")
            box = self._pixel_box(source.get("box"), rgb.shape)
            if score < self.config["detection_threshold"]:
                continue
            available = [track for track in self.tracks if track.track_id not in associated_ids and _iou(track.box, box) >= .2]
            previous = max(available, key=lambda track: _iou(track.box, box)) if available else None
            if previous is not None:
                associated_ids.add(previous.track_id)
            associations.append(previous)
            detected_boxes.append(box)
            item = dict(source, evidence_type="model_detection", model_score=float(score), tracking_score=None,
                        originating_detection_sec=previous.origin_sec if previous else float(time_sec),
                        last_detection_sec=float(time_sec), matched_at_sec=float(time_sec),
                        absence_certified=False)
            current.append(item)
        survivors = []
        for track in self.tracks:
            if any(_iou(track.box, box) >= .2 for box in detected_boxes):
                continue  # current model evidence replaces that old track
            matched, failure = self._match(rgb, time_sec, track)
            if failure:
                evidence.append(dict(track_id=track.track_id, status="discarded", reason=failure))
            elif matched:
                survivor, item = matched
                # A shifted tracker may converge on a current detection; keep
                # the detector evidence once, without duplicating the object.
                if any(_iou(survivor.box, box) >= .2 for box in detected_boxes):
                    continue
                survivors.append(survivor); current.append(item)
                evidence.append(dict(track_id=track.track_id, status="image_match", tracking_score=item["tracking_score"],
                                     rgb_mean_absolute_difference=item["rgb_mean_absolute_difference"]))
        for item, box, previous in zip(current[:len(detected_boxes)], detected_boxes, associations):
            patch = self._trackable_patch(rgb, box)
            if patch is None:
                item["trackable"] = False
                continue
            identity = previous.track_id if previous else self.next_id
            track = _Track(identity, box, box, patch, item["originating_detection_sec"], float(time_sec),
                           float(time_sec), float(item["score"]), item.get("area_px"))
            if previous is None:
                self.next_id += 1
            item["track_id"], item["trackable"] = track.track_id, True
            survivors.append(track)
        self.tracks = sorted(survivors, key=lambda track: (-track.model_score, -track.origin_sec, track.track_id))[:self.config["max_tracks"]]
        self.shape, self.last_sec, self.preview = rgb.shape, float(time_sec), preview.astype(np.float32)
        return dict(schema=self.schema, objects=current, tracking_evidence=evidence, absence_certified=False,
                    tracking_config=dict(self.config), reset_reason=reason)
