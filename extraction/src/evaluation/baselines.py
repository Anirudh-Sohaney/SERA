"""
Baseline models for comparison.

Baseline 1: Lexical matching — find output values as substrings.
Baseline 2: Pretrained encoder + token classification (main model).
Baseline 3: Span prediction model (optional, for future work).

The lexical baseline provides a lower bound: if the model cannot beat
simple substring matching, it has not learned meaningful representations.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ..data.alignment import extract_output_values, find_substring_offsets

logger = logging.getLogger(__name__)


class LexicalBaseline:
    """
    Lexical matching baseline.

    For each output field value, search for it as a substring in the prompt.
    This is essentially the alignment pipeline run at inference time.
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        min_span_length: int = 2,
    ):
        self.case_sensitive = case_sensitive
        self.min_span_length = min_span_length

    def predict(self, prompt: str, record: Dict) -> List[Dict]:
        """
        Predict spans using lexical matching.

        Args:
            prompt: The user prompt
            record: The full record (for accessing output fields)

        Returns:
            List of span dicts with start, end, label, text
        """
        output = record.get("output", {})
        output_values = extract_output_values(output)

        spans = []
        for field, value in output_values:
            if len(value) < self.min_span_length:
                continue

            offsets = find_substring_offsets(prompt, value, self.case_sensitive)

            if offsets:
                # Use first occurrence
                start, end = offsets[0]
                spans.append({
                    "start": start,
                    "end": end,
                    "label": "project_info",
                    "text": prompt[start:end],
                    "field": field,
                    "confidence": 1.0,
                })

        return spans

    def evaluate(
        self,
        aligned_records: List[Dict],
        indices: List[int],
    ) -> Dict:
        """Evaluate lexical baseline on a set of records."""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        exact_matches = 0
        total_records = 0

        all_errors = []

        for idx in indices:
            ar = aligned_records[idx]
            record = ar["record"]
            gold_spans = ar["spans"]
            prompt = record.get("input", {}).get("user_prompt", "")

            if not prompt or not gold_spans:
                continue

            total_records += 1
            pred_spans = self.predict(prompt, record)

            # Match predicted to gold
            matched_gold = set()
            for p in pred_spans:
                p_text = p.get("text", "").lower().strip()
                found = False
                for g_idx, g in enumerate(gold_spans):
                    g_text = g.get("text", "").lower().strip()
                    if g_idx not in matched_gold and p_text == g_text:
                        total_tp += 1
                        matched_gold.add(g_idx)
                        found = True
                        break
                if not found:
                    total_fp += 1
                    all_errors.append({
                        "type": "false_positive",
                        "prompt": prompt[:200],
                        "predicted": p_text,
                    })

            total_fn += len(gold_spans) - len(matched_gold)
            for g_idx, g in enumerate(gold_spans):
                if g_idx not in matched_gold:
                    all_errors.append({
                        "type": "false_negative",
                        "prompt": prompt[:200],
                        "expected": g.get("text", ""),
                    })

            if len(matched_gold) == len(gold_spans) and len(pred_spans) == len(gold_spans):
                exact_matches += 1

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "exact_match_rate": exact_matches / max(total_records, 1),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "total_records": total_records,
            "error_examples": all_errors[:50],
        }


def run_lexical_baseline(
    aligned_records: List[Dict],
    splits: Dict[str, List[int]],
) -> Dict:
    """Run lexical baseline evaluation on all splits."""
    baseline = LexicalBaseline()

    results = {}
    for split_name, indices in splits.items():
        results[split_name] = baseline.evaluate(aligned_records, indices)
        logger.info(
            f"Lexical baseline {split_name}: "
            f"P={results[split_name]['precision']:.4f}, "
            f"R={results[split_name]['recall']:.4f}, "
            f"F1={results[split_name]['f1']:.4f}"
        )

    return results
