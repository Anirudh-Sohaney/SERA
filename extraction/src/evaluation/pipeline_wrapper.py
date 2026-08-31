"""
Pipeline wrapper for evaluator compatibility.

Wraps TwoStagePipeline to return objects compatible with
EndToEndEvaluator's expectations (extract() returns .spans, .has_spans).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ..inference.pipeline import TwoStagePipeline


class PipelineWrapper:
    """Wraps TwoStagePipeline for evaluator compatibility.

    The EndToEndEvaluator calls extractor.extract(prompt) and accesses
    .spans and .has_spans. TwoStagePipeline.extract() returns a
    PipelineResult which already has these attributes. This wrapper
    provides a thin compatibility layer.
    """

    def __init__(self, pipeline: TwoStagePipeline) -> None:
        self._pipeline = pipeline

    def extract(self, prompt: str) -> Any:
        """Run the two-stage pipeline and return a result with .spans, .has_spans."""
        return self._pipeline.extract(prompt)

    @property
    def stage2_threshold(self) -> float:
        return self._pipeline.stage2_threshold

    @stage2_threshold.setter
    def stage2_threshold(self, value: float) -> None:
        self._pipeline.stage2_threshold = value


class ExtractionSLMWrapper:
    """Wraps a single ExtractionSLM to return objects with .spans, .has_spans.

    Used for Stage-1-only evaluation without SpanFilter.
    """

    def __init__(self, extractor: Any) -> None:
        self._extractor = extractor

    def extract(self, prompt: str) -> Any:
        """Run extraction and return result with .spans, .has_spans."""
        return self._extractor.extract(prompt)
