"""
E7 Span Filter Evaluator.

Evaluates the two-stage pipeline against the existing evaluation suite.
Reports precision, recall, F1, unseen-vocabulary F1, context-positive F1,
and other required metrics.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ..models.span_filter import SpanFilter
from ..models.token_classifier import ExtractionClassifier
from ..data.alignment import build_bio_labels
from ..inference.pipeline import TwoStagePipeline

logger = logging.getLogger(__name__)


class EvalDS(Dataset):
    """Dataset for evaluation examples."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 128):
        self.data = []
        for e in examples:
            prompt = e.get("prompt", "")
            spans = e.get("spans", [])
            if not prompt:
                continue
            enc = build_bio_labels(prompt, spans, tokenizer, max_length)
            self.data.append({
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(enc["labels"], dtype=torch.long),
                "prompt": prompt,
                "spans": spans,
                "offset_mapping": enc["offset_mapping"],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["input_ids"],
            "attention_mask": self.data[idx]["attention_mask"],
            "labels": self.data[idx]["labels"],
        }


def evaluate_stage1_only(
    model: ExtractionClassifier,
    tokenizer: AutoTokenizer,
    eval_examples: List[Dict],
    max_length: int = 128,
    batch_size: int = 64,
) -> Dict:
    """
    Evaluate Stage 1 only (E6-A baseline).

    Returns precision, recall, F1 at token level.
    """
    model.eval()
    device = next(model.parameters()).device

    tp = 0
    fp = 0
    fn = 0

    ds = EvalDS(eval_examples, tokenizer, max_length)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs["logits"], dim=-1)

        pred_np = preds.cpu().numpy()
        label_np = labels.numpy()
        mask_np = batch["attention_mask"].numpy()

        for p, l, m in zip(pred_np, label_np, mask_np):
            active = m == 1
            p_pos = p[active] >= 1
            l_pos = l[active] >= 1
            tp += int((p_pos & l_pos).sum())
            fp += int((p_pos & ~l_pos).sum())
            fn += int((~p_pos & l_pos).sum())

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n": len(ds),
    }


def evaluate_two_stage(
    pipeline: TwoStagePipeline,
    eval_examples: List[Dict],
) -> Dict:
    """
    Evaluate the two-stage pipeline.

    For each example:
    1. Run Stage 1 to get candidates
    2. Run Stage 2 to filter
    3. Compare filtered spans with gold spans
    """
    tp = 0
    fp = 0
    fn = 0
    total_candidates = 0
    total_accepted = 0

    for ex in eval_examples:
        prompt = ex.get("prompt", "")
        gold_spans = ex.get("spans", [])

        if not prompt:
            continue

        # Run pipeline
        result = pipeline.extract(prompt)
        pred_spans = [(s["start"], s["end"]) for s in result.spans]
        gold_set = set((s["start"], s["end"]) for s in gold_spans)

        total_candidates += result.metrics.get("stage1_candidates", 0)
        total_accepted += len(result.spans)

        # Count TP, FP, FN
        for pred in pred_spans:
            if pred in gold_set:
                tp += 1
            else:
                fp += 1

        for gold in gold_set:
            if gold not in [p for p in pred_spans]:
                fn += 1

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_candidates": total_candidates,
        "total_accepted": total_accepted,
        "rejection_rate": round(1 - (total_accepted / max(total_candidates, 1)), 4),
        "n": len(eval_examples),
    }


def evaluate_pipeline(
    pipeline: TwoStagePipeline,
    evaluation_suite_path: str,
) -> Dict:
    """
    Run full evaluation suite on the two-stage pipeline.

    Evaluates:
    - random test
    - unseen vocabulary
    - context positive
    - context negative
    """
    with open(evaluation_suite_path) as f:
        eval_suite = json.load(f)

    results = {}

    # Random test
    if "random_test" in eval_suite:
        logger.info("Evaluating on random test set...")
        results["random_test"] = evaluate_two_stage(
            pipeline, eval_suite["random_test"]["examples"]
        )

    # Unseen vocabulary
    if "unseen_vocabulary" in eval_suite:
        logger.info("Evaluating on unseen vocabulary...")
        results["unseen_vocab"] = evaluate_two_stage(
            pipeline, eval_suite["unseen_vocabulary"]["unseen_examples"]
        )

    # Context positive
    if "context_reversal" in eval_suite:
        logger.info("Evaluating on context positive examples...")
        results["context_positive"] = evaluate_two_stage(
            pipeline, eval_suite["context_reversal"]["positive_examples"]
        )

    # Context negative
    if "context_reversal" in eval_suite:
        logger.info("Evaluating on context negative examples...")
        results["context_negative"] = evaluate_two_stage(
            pipeline, eval_suite["context_reversal"]["negative_examples"]
        )

    return results
