"""
E7 Two-Stage Pipeline.

Orchestrates Stage 1 (E6-A) and Stage 2 (Span Filter) for the final
extractive span detection system.

Architecture:
    USER PROMPT
        ↓
    STAGE 1 — E6-A (candidate generation)
        ↓
    candidate spans
        ↓
    STAGE 2 — Span Filter (KEEP / REJECT)
        ↓
    accepted spans
        ↓
    exact character-offset spans

The pipeline ensures:
- Stage 1 produces candidates
- Stage 2 classifies each candidate
- Only KEEP candidates are returned
- Original Stage 1 offsets are preserved exactly
- No text generation or modification
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from ..models.span_filter import SpanFilter
from ..models.token_classifier import ExtractionClassifier

logger = logging.getLogger(__name__)


class PipelineResult:
    """Result of the two-stage pipeline."""

    def __init__(self, spans: List[Dict], prompt: str, metrics: Dict):
        self.spans = spans
        self.prompt = prompt
        self.metrics = metrics

    def to_dict(self) -> Dict:
        return {
            "spans": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "confidence": s["confidence"],
                    "stage1_confidence": s.get("stage1_confidence", 0),
                    "stage2_prob": s.get("stage2_prob", 0),
                }
                for s in self.spans
            ],
            "num_spans": len(self.spans),
            "prompt_length": len(self.prompt),
            "metrics": self.metrics,
        }


class TwoStagePipeline:
    """
    Two-stage span extraction pipeline.

    Stage 1: E6-A generates high-recall candidate spans
    Stage 2: Span filter classifies KEEP/REJECT
    """

    def __init__(
        self,
        stage1_model: ExtractionClassifier,
        stage1_tokenizer: AutoTokenizer,
        stage2_model: SpanFilter,
        stage2_tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
        stage1_max_length: int = 128,
        stage2_max_length: int = 256,
        stage2_threshold: float = 0.5,
    ):
        self.stage1_model = stage1_model
        self.stage1_tokenizer = stage1_tokenizer
        self.stage2_model = stage2_model
        self.stage2_tokenizer = stage2_tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.stage1_model.to(self.device)
        self.stage1_model.eval()
        self.stage2_model.to(self.device)
        self.stage2_model.eval()

        self.stage1_max_length = stage1_max_length
        self.stage2_max_length = stage2_max_length
        self.stage2_threshold = stage2_threshold

    @classmethod
    def from_checkpoints(
        cls,
        stage1_checkpoint: str,
        stage2_checkpoint: str,
        stage2_model_name: str = "google/bert_uncased_L-6_H-512_A-8",
        stage2_threshold: float = 0.5,
    ) -> "TwoStagePipeline":
        """Load both stages from checkpoints."""
        # Load Stage 1
        s1_path = Path(stage1_checkpoint)
        s1_config_path = s1_path / "config.json"
        if s1_config_path.exists():
            with open(s1_config_path) as f:
                s1_config = json.load(f)
            s1_model_name = s1_config.get("model_name", "google/bert_uncased_L-6_H-512_A-8")
        else:
            s1_model_name = "google/bert_uncased_L-6_H-512_A-8"

        stage1_model = ExtractionClassifier(model_name=s1_model_name)
        s1_model_path = s1_path / "model.pt"
        if s1_model_path.exists():
            state_dict = torch.load(s1_model_path, map_location="cpu", weights_only=True)
            stage1_model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Stage 1 model not found at {s1_model_path}")

        stage1_tokenizer = AutoTokenizer.from_pretrained(s1_model_name)

        # Load Stage 2
        s2_path = Path(stage2_checkpoint)
        stage2_model = SpanFilter(model_name=stage2_model_name)
        s2_model_path = s2_path / "model.pt"
        if s2_model_path.exists():
            state_dict = torch.load(s2_model_path, map_location="cpu", weights_only=True)
            stage2_model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Stage 2 model not found at {s2_model_path}")

        stage2_tokenizer = AutoTokenizer.from_pretrained(stage2_model_name)

        # Load threshold if available
        threshold_path = s2_path.parent / "threshold_results.json"
        if threshold_path.exists():
            with open(threshold_path) as f:
                threshold_data = json.load(f)
            if "best" in threshold_data:
                stage2_threshold = threshold_data["best"]["threshold"]
                logger.info(f"Loaded threshold: {stage2_threshold}")

        return cls(
            stage1_model=stage1_model,
            stage1_tokenizer=stage1_tokenizer,
            stage2_model=stage2_model,
            stage2_tokenizer=stage2_tokenizer,
            stage2_threshold=stage2_threshold,
        )

    def extract(self, prompt: str) -> PipelineResult:
        """
        Extract project-memory spans using the two-stage pipeline.

        Args:
            prompt: The user's coding/project prompt

        Returns:
            PipelineResult with filtered spans
        """
        start_time = time.time()

        if not prompt or not prompt.strip():
            return PipelineResult([], prompt, {})

        # Stage 1: Generate candidates
        stage1_start = time.time()
        candidates = self._stage1_extract(prompt)
        stage1_time = time.time() - stage1_start

        # Stage 2: Classify candidates
        stage2_start = time.time()
        accepted = self._stage2_classify(prompt, candidates)
        stage2_time = time.time() - stage2_start

        total_time = time.time() - start_time

        # Validate exact offsets
        validated = self._validate_offsets(prompt, accepted)

        metrics = {
            "stage1_candidates": len(candidates),
            "stage2_accepted": len(validated),
            "rejection_rate": 1 - (len(validated) / max(len(candidates), 1)),
            "stage1_latency_ms": stage1_time * 1000,
            "stage2_latency_ms": stage2_time * 1000,
            "total_latency_ms": total_time * 1000,
        }

        return PipelineResult(validated, prompt, metrics)

    def _stage1_extract(self, prompt: str) -> List[Dict]:
        """Run Stage 1 to generate candidate spans."""
        # Tokenize
        encoding = self.stage1_tokenizer(
            prompt,
            max_length=self.stage1_max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
        )

        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([encoding["attention_mask"]], dtype=torch.long)

        # Predict
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            outputs = self.stage1_model(input_ids, attention_mask)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            confidences = probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)

        # Decode spans
        candidates = self._decode_stage1(
            predictions[0].cpu().numpy(),
            confidences[0].cpu().numpy(),
            attention_mask[0].cpu().numpy(),
            encoding["offset_mapping"],
            prompt,
        )

        return candidates

    def _decode_stage1(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        attention_mask: np.ndarray,
        offset_mapping: List,
        prompt: str,
    ) -> List[Dict]:
        """Decode Stage 1 BIO predictions to candidate spans."""
        candidates = []
        active_mask = attention_mask == 1
        pred_seq = predictions[active_mask]
        conf_seq = confidences[active_mask]
        offsets = offset_mapping[: len(pred_seq)]

        in_span = False
        span_start = None
        span_conf = []

        for i, (tag, conf, (char_start, char_end)) in enumerate(
            zip(pred_seq, conf_seq, offsets)
        ):
            if tag == 1:  # B-PROJECT_INFO
                if in_span and span_start is not None:
                    avg_conf = float(np.mean(span_conf))
                    text = prompt[span_start:char_start]
                    if text.strip():
                        candidates.append({
                            "start": span_start,
                            "end": char_start,
                            "text": text,
                            "confidence": avg_conf,
                        })

                span_start = char_start
                span_conf = [float(conf)]
                in_span = True

            elif tag == 2 and in_span:  # I-PROJECT_INFO
                span_conf.append(float(conf))

            else:  # O
                if in_span and span_start is not None:
                    avg_conf = float(np.mean(span_conf))
                    text = prompt[span_start:char_start]
                    if text.strip():
                        candidates.append({
                            "start": span_start,
                            "end": char_start,
                            "text": text,
                            "confidence": avg_conf,
                        })

                in_span = False
                span_start = None
                span_conf = []

        # Close final span
        if in_span and span_start is not None:
            last_offset = offsets[-1] if offsets else (0, 0)
            avg_conf = float(np.mean(span_conf))
            text = prompt[span_start:last_offset[1]]
            if text.strip():
                candidates.append({
                    "start": span_start,
                    "end": last_offset[1],
                    "text": text,
                    "confidence": avg_conf,
                })

        return candidates

    def _stage2_classify(
        self, prompt: str, candidates: List[Dict]
    ) -> List[Dict]:
        """Run Stage 2 to classify candidates."""
        if not candidates:
            return []

        accepted = []

        for candidate in candidates:
            # Tokenize as sequence pair
            encoding = self.stage2_tokenizer(
                prompt,
                candidate["text"],
                max_length=self.stage2_max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Classify
            with torch.no_grad():
                outputs = self.stage2_model(input_ids, attention_mask)
                prob = outputs["probs"].item()

            # Apply threshold
            if prob >= self.stage2_threshold:
                candidate["stage2_prob"] = prob
                accepted.append(candidate)

        return accepted

    def _validate_offsets(self, prompt: str, spans: List[Dict]) -> List[Dict]:
        """Validate that all spans are exact substrings."""
        validated = []
        for span in spans:
            extracted = prompt[span["start"]:span["end"]]
            if extracted == span["text"]:
                validated.append(span)
            else:
                # Try to find the text as a fallback
                idx = prompt.find(span["text"])
                if idx >= 0:
                    span["start"] = idx
                    span["end"] = idx + len(span["text"])
                    validated.append(span)
                # else: discard invalid span

        return validated


def extract_two_stage(
    prompt: str,
    pipeline: TwoStagePipeline,
) -> Dict:
    """
    Convenience function for two-stage extraction.

    Args:
        prompt: User's coding/project prompt
        pipeline: Loaded TwoStagePipeline

    Returns:
        Dict with spans and metrics
    """
    result = pipeline.extract(prompt)
    return result.to_dict()
