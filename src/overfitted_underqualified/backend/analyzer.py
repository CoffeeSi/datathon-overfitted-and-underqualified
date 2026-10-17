"""Replaceable local analyzer interface and development-only mock provider."""

from __future__ import annotations

import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Protocol

from .schemas import AnalysisResult
from .settings import Settings


class Analyzer(Protocol):
    @property
    def name(self) -> str: ...

    async def analyze(self, video_path: Path) -> AnalysisResult: ...


class MockAnalyzer:
    """A deterministic integration stub; it does not inspect video content."""

    name = "mock"

    def __init__(self, fixture_path: Path | None = None) -> None:
        self._fixture: AnalysisResult | None = None
        if fixture_path is not None:
            self._fixture = AnalysisResult.model_validate_json(
                fixture_path.read_text(encoding="utf-8")
            )

    async def analyze(self, video_path: Path) -> AnalysisResult:
        # Preserve the async provider contract for future local inference implementations.
        await asyncio.sleep(0)
        if self._fixture is not None:
            return self._fixture
        return AnalysisResult(verdict="no_violation", violations=[])


def configured_analyzer(settings: Settings) -> Analyzer:
    """Load a local provider factory or use the development mock."""
    if not settings.analyzer_factory:
        return MockAnalyzer(settings.mock_fixture_path)
    try:
        module_name, factory_name = settings.analyzer_factory.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            "FORTEBANK_ANALYZER_FACTORY must use the form 'python.module:create_analyzer'"
        ) from exc
    factory = getattr(importlib.import_module(module_name), factory_name)
    provider = factory(settings)
    analyze_method = getattr(provider, "analyze", None)
    if (
        not callable(analyze_method)
        or not inspect.iscoroutinefunction(analyze_method)
        or not isinstance(getattr(provider, "name", None), str)
    ):
        raise TypeError("Configured analyzer must expose a string name and async analyze method")
    return provider
