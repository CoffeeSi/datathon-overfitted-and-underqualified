"""Shared analyzer and API result schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceFrame(BaseModel):
    timestamp_sec: float = Field(ge=0)
    image_base64: str = Field(min_length=1)


class Finding(BaseModel):
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class AnalysisResult(BaseModel):
    verdict: Literal["no_violation", "violation", "insufficient_data"]
    violations: list[Finding] = Field(default_factory=list)

    @model_validator(mode="after")
    def verdict_matches_findings(self) -> "AnalysisResult":
        if bool(self.violations) != (self.verdict == "violation"):
            raise ValueError("violation verdicts must have findings, and other verdicts must not")
        return self
