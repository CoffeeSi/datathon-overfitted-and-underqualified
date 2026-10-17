"""Shared analyzer and API result schemas."""

from __future__ import annotations

import base64
import binascii
import math
from io import BytesIO
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator, model_validator


class EvidenceFrame(BaseModel):
    timestamp_sec: float = Field(ge=0)
    image_base64: str = Field(min_length=1)

    @field_validator("timestamp_sec")
    @classmethod
    def timestamp_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("timestamp_sec must be finite")
        return value

    @field_validator("image_base64")
    @classmethod
    def image_must_be_base64_jpeg(cls, value: str) -> str:
        try:
            image_bytes = base64.b64decode(value, validate=True)
        except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
            raise ValueError("image_base64 must contain valid base64 data") from exc

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format != "JPEG":
                    raise ValueError("image_base64 must decode to a JPEG image")
                if (
                    image.width < 1
                    or image.height < 1
                    or image.width * image.height > 40_000_000
                ):
                    raise ValueError("JPEG dimensions are invalid or too large")
                image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("image_base64 must contain a valid JPEG image") from exc
        return value


class Finding(BaseModel):
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)

    @field_validator("timecode_sec")
    @classmethod
    def timecode_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("timecode_sec must be finite")
        return value


class AnalysisResult(BaseModel):
    verdict: Literal["no_violation", "violation", "insufficient_data"]
    violations: list[Finding] = Field(default_factory=list)

    @model_validator(mode="after")
    def verdict_matches_findings(self) -> "AnalysisResult":
        if bool(self.violations) != (self.verdict == "violation"):
            raise ValueError("violation verdicts must have findings, and other verdicts must not")
        return self
