"""
Inference API for project-memory span extraction.

Provides a clean `extract()` function that:
1. Takes a user prompt as input
2. Returns exact character-offset spans
3. Validates all spans are exact substrings
4. Supports zero, one, or multiple spans
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from ..models.token_classifier import ExtractionClassifier

logger = logging.getLogger(__name__)


class ExtractionResult:
    """Result of span extraction."""

    def __init__(self, spans: List[Dict], prompt: str):
        self.spans = spans
        self.prompt = prompt

    def to_dict(self) -> Dict:
        return {
            "spans": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "label": s["label"],
                    "confidence": s["confidence"],
                    "text": s["text"],
                }
                for s in self.spans
            ],
            "num_spans": len(self.spans),
            "prompt_length": len(self.prompt),
        }

    @property
    def texts(self) -> List[str]:
        """Extracted texts in order."""
        return [s["text"] for s in self.spans]

    @property
    def has_spans(self) -> bool:
        """Whether any spans were extracted."""
        return len(self.spans) > 0


class ExtractionSLM:
    """
    Inference wrapper for the extraction SLM.

    Usage:
        model = ExtractionSLM.from_checkpoint("checkpoints/best")
        result = model.extract("Use FastAPI for the backend")
        for span in result.spans:
            print(f"{span['text']} ({span['confidence']:.2f})")
    """

    def __init__(
        self,
        model: ExtractionClassifier,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.5,
        max_length: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        model_name: str = "microsoft/deberta-v3-small",
        confidence_threshold: float = 0.5,
    ) -> "ExtractionSLM":
        """Load model from a checkpoint directory."""
        checkpoint_path = Path(checkpoint_dir)

        # Load config if available
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            model_name = config.get("model_name", model_name)

        # Create model
        model = ExtractionClassifier(model_name=model_name)

        # Load weights
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"No model.pt found in {checkpoint_dir}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return cls(
            model=model,
            tokenizer=tokenizer,
            confidence_threshold=confidence_threshold,
        )

    def extract(self, prompt: str) -> ExtractionResult:
        """
        Extract project-memory spans from a user prompt.

        Args:
            prompt: The user's coding/project prompt

        Returns:
            ExtractionResult with validated spans
        """
        if not prompt or not prompt.strip():
            return ExtractionResult([], prompt)

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
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

            outputs = self.model(input_ids, attention_mask)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            confidences = probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)

        # Decode spans
        spans = self._decode_spans(
            predictions[0].cpu().numpy(),
            confidences[0].cpu().numpy(),
            attention_mask[0].cpu().numpy(),
            encoding["offset_mapping"],
            prompt,
        )

        return ExtractionResult(spans, prompt)

    def _decode_spans(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        attention_mask: np.ndarray,
        offset_mapping: List,
        prompt: str,
    ) -> List[Dict]:
        """Decode BIO predictions to character-offset spans."""
        spans = []
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
                    if avg_conf >= self.confidence_threshold:
                        text = prompt[span_start:char_end]
                        if text.strip():  # Skip empty spans
                            spans.append({
                                "start": span_start,
                                "end": char_end,
                                "label": "project_info",
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
                    if avg_conf >= self.confidence_threshold:
                        text = prompt[span_start:char_start]
                        if text.strip():
                            spans.append({
                                "start": span_start,
                                "end": char_start,
                                "label": "project_info",
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
            if avg_conf >= self.confidence_threshold:
                text = prompt[span_start:last_offset[1]]
                if text.strip():
                    spans.append({
                        "start": span_start,
                        "end": last_offset[1],
                        "label": "project_info",
                        "text": text,
                        "confidence": avg_conf,
                    })

        # Hard validation: all spans must be exact substrings
        validated_spans = []
        for span in spans:
            extracted = prompt[span["start"]:span["end"]]
            if extracted == span["text"]:
                validated_spans.append(span)
            else:
                # Try to find the text as a fallback
                idx = prompt.find(span["text"])
                if idx >= 0:
                    span["start"] = idx
                    span["end"] = idx + len(span["text"])
                    validated_spans.append(span)
                # else: discard invalid span

        # Handle overlapping spans: keep all (configurable)
        return validated_spans

    def extract_batch(self, prompts: List[str]) -> List[ExtractionResult]:
        """Extract spans from multiple prompts."""
        results = []
        for prompt in prompts:
            results.append(self.extract(prompt))
        return results

    def save(self, output_dir: str):
        """Save the extraction SLM to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), output_path / "model.pt")

        with open(output_path / "config.json", "w") as f:
            json.dump({
                "model_name": self.model.model_name,
                "confidence_threshold": self.confidence_threshold,
                "max_length": self.max_length,
                "num_labels": self.model.num_labels,
            }, f, indent=2)

        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Saved extraction SLM to {output_path}")


def extract(prompt: str, model: ExtractionSLM) -> Dict:
    """
    Convenience function for span extraction.

    Args:
        prompt: User's coding/project prompt
        model: Loaded ExtractionSLM

    Returns:
        Dict with spans list
    """
    result = model.extract(prompt)
    return result.to_dict()
