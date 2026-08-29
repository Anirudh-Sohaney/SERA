"""
Comprehensive evaluation suite.

Implements:
- Random test set evaluation
- Concept-holdout evaluation
- Unseen-vocabulary evaluation
- Cross-language evaluation
- Adversarial evaluation (Tests A-E)
- Error analysis with taxonomy
- Baseline comparisons
"""

import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ..data.alignment import find_substring_offsets
from ..data.label_schema import LABEL_NAMES, NUM_LABELS

logger = logging.getLogger(__name__)


class ErrorTaxonomy:
    """Categorization of extraction errors."""

    CATEGORIES = [
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE",
        "BOUNDARY_ERROR",
        "CONTEXT_ERROR",
        "NEGATION_ERROR",
        "CONTRAST_ERROR",
        "DUPLICATE_MATCH",
        "NON_EXTRACTIVE_TARGET",
        "CODE_SPAN_ERROR",
        "PATH_ERROR",
        "VERSION_ERROR",
        "MULTIWORD_ERROR",
        "MULTI_SPAN_ERROR",
        "UNSEEN_VOCABULARY_ERROR",
        "LANGUAGE_GENERALIZATION_ERROR",
    ]

    def __init__(self):
        self.errors = {cat: [] for cat in self.CATEGORIES}
        self.counts = Counter()

    def categorize_error(
        self,
        prompt: str,
        predicted_span: Dict,
        gold_span: Optional[Dict],
        all_gold_spans: List[Dict],
    ) -> str:
        """Categorize a single error."""
        pred_text = predicted_span.get("text", "")
        pred_start = predicted_span.get("start", 0)
        pred_end = predicted_span.get("end", 0)

        if gold_span is None:
            # False positive — predicted something that shouldn't be extracted
            # Check for negation context
            before = prompt[max(0, pred_start - 50):pred_start].lower()
            if any(neg in before for neg in ["do not", "don't", "avoid", "instead of", "rather than"]):
                return "NEGATION_ERROR"
            # Check for contrast
            if any(contr in before for contr in ["but", "however", "although", "though"]):
                return "CONTRAST_ERROR"
            return "FALSE_POSITIVE"

        # False negative — missed something that should be extracted
        if predicted_span is None and gold_span is not None:
            return "FALSE_NEGATIVE"

        # Boundary error — partial overlap
        gold_start = gold_span.get("start", 0)
        gold_end = gold_span.get("end", 0)
        gold_text = gold_span.get("text", "")

        if pred_start != gold_start or pred_end != gold_end:
            # Check if it's a multi-word boundary issue
            if len(gold_text.split()) > 1 and pred_text in gold_text:
                return "MULTIWORD_ERROR"
            return "BOUNDARY_ERROR"

        return None

    def add_error(self, category: str, example: Dict):
        """Add an error example to a category."""
        if category in self.errors:
            self.errors[category].append(example)
            self.counts[category] += 1

    def get_summary(self) -> Dict:
        """Get error summary."""
        return {
            "counts": dict(self.counts),
            "total_errors": sum(self.counts.values()),
            "examples": {
                cat: examples[:10] for cat, examples in self.errors.items()
            },
        }


class Evaluator:
    """
    Comprehensive evaluation suite for the extraction model.
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        config,
        device: torch.device = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.error_taxonomy = ErrorTaxonomy()

    @torch.no_grad()
    def predict_batch(self, input_ids, attention_mask):
        """Run model inference on a batch."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_ids, attention_mask)
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        confidences = probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)

        return predictions.cpu().numpy(), confidences.cpu().numpy()

    def decode_spans(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        attention_mask: np.ndarray,
        offset_mappings: List,
        prompt: str,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Decode BIO predictions to exact character-offset spans.

        Returns list of span dicts with start, end, label, confidence, text.
        """
        spans = []
        active_mask = attention_mask == 1
        pred_seq = predictions[active_mask]
        conf_seq = confidences[active_mask]
        offsets = offset_mappings[:len(pred_seq)]

        in_span = False
        span_start = None
        span_conf = []

        for i, (tag, conf, (char_start, char_end)) in enumerate(
            zip(pred_seq, conf_seq, offsets)
        ):
            if tag == 1:  # B-PROJECT_INFO
                # Close any open span
                if in_span and span_start is not None:
                    avg_conf = np.mean(span_conf)
                    if avg_conf >= threshold:
                        text = prompt[span_start:char_end]
                        spans.append({
                            "start": span_start,
                            "end": char_end,
                            "label": "project_info",
                            "text": text,
                            "confidence": float(avg_conf),
                        })

                span_start = char_start
                span_conf = [float(conf)]
                in_span = True

            elif tag == 2 and in_span:  # I-PROJECT_INFO
                span_conf.append(float(conf))

            else:  # O
                if in_span and span_start is not None:
                    avg_conf = np.mean(span_conf)
                    if avg_conf >= threshold:
                        text = prompt[span_start:char_start]
                        spans.append({
                            "start": span_start,
                            "end": char_start,
                            "label": "project_info",
                            "text": text,
                            "confidence": float(avg_conf),
                        })

                in_span = False
                span_start = None
                span_conf = []

        # Close final span
        if in_span and span_start is not None:
            last_offset = offsets[-1] if offsets else (0, 0)
            avg_conf = np.mean(span_conf)
            if avg_conf >= threshold:
                text = prompt[span_start:last_offset[1]]
                spans.append({
                    "start": span_start,
                    "end": last_offset[1],
                    "label": "project_info",
                    "text": text,
                    "confidence": float(avg_conf),
                })

        # Hard validation: assert all spans are exact substrings
        valid_spans = []
        for span in spans:
            extracted = prompt[span["start"]:span["end"]]
            if extracted == span["text"]:
                valid_spans.append(span)
            else:
                # Try to find the text in the prompt
                idx = prompt.find(span["text"])
                if idx >= 0:
                    span["start"] = idx
                    span["end"] = idx + len(span["text"])
                    valid_spans.append(span)
                # else: discard invalid span

        return valid_spans

    def evaluate_dataset(
        self,
        dataset,
        split_name: str = "test",
        threshold: float = 0.5,
    ) -> Dict:
        """
        Evaluate model on a dataset.

        Returns comprehensive metrics.
        """
        logger.info(f"Evaluating on {split_name} set ({len(dataset)} examples)")

        all_predictions = []
        all_labels = []
        all_masks = []
        all_spans = []
        all_prompts = []
        all_gold_spans = []

        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            predictions, confidences = self.predict_batch(input_ids, attention_mask)

            all_predictions.append(predictions)
            all_labels.append(batch["labels"].numpy())
            all_masks.append(attention_mask.numpy())

            # Decode spans for each example in batch
            for b in range(input_ids.shape[0]):
                metadata = dataset.get_metadata(b)
                pred_spans = self.decode_spans(
                    predictions[b],
                    confidences[b],
                    attention_mask[b].numpy(),
                    metadata["offset_mapping"],
                    metadata["prompt"],
                    threshold,
                )
                all_spans.append(pred_spans)
                all_prompts.append(metadata["prompt"])
                all_gold_spans.append(metadata["spans"])

        # Concatenate token-level predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # Compute token-level metrics
        from ..training.trainer import compute_metrics
        token_metrics = compute_metrics(all_predictions, all_labels, all_masks)

        # Compute span-level metrics
        span_metrics = self._compute_span_metrics(all_spans, all_gold_spans)

        # Error analysis
        error_analysis = self._analyze_errors(
            all_prompts, all_spans, all_gold_spans
        )

        results = {
            "split": split_name,
            "num_examples": len(dataset),
            "token_metrics": token_metrics,
            "span_metrics": span_metrics,
            "error_analysis": error_analysis,
            "threshold": threshold,
        }

        logger.info(f"{split_name} results:")
        logger.info(f"  Token F1: {token_metrics['f1']:.4f}")
        logger.info(f"  Token Precision: {token_metrics['precision']:.4f}")
        logger.info(f"  Token Recall: {token_metrics['recall']:.4f}")
        logger.info(f"  Span F1: {span_metrics['f1']:.4f}")
        logger.info(f"  Exact Span Accuracy: {span_metrics['exact_span_accuracy']:.4f}")

        return results

    def _compute_span_metrics(
        self,
        predicted_spans: List[List[Dict]],
        gold_spans: List[List[Dict]],
    ) -> Dict:
        """Compute span-level precision, recall, F1."""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        exact_matches = 0
        total_gold = 0

        for pred, gold in zip(predicted_spans, gold_spans):
            # Match predicted spans to gold spans
            matched_gold = set()
            for p in pred:
                p_text = p.get("text", "").lower().strip()
                found = False
                for g_idx, g in enumerate(gold):
                    g_text = g.get("text", "").lower().strip()
                    if g_idx not in matched_gold and p_text == g_text:
                        total_tp += 1
                        matched_gold.add(g_idx)
                        found = True
                        break
                if not found:
                    total_fp += 1

            total_fn += len(gold) - len(matched_gold)
            total_gold += len(gold)

            # Exact match: all gold spans must be found exactly
            if len(matched_gold) == len(gold) and len(pred) == len(gold):
                exact_matches += 1

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        exact_accuracy = exact_matches / max(len(predicted_spans), 1)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "exact_span_accuracy": float(exact_accuracy),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "total_gold": total_gold,
        }

    def _analyze_errors(
        self,
        prompts: List[str],
        predicted_spans: List[List[Dict]],
        gold_spans: List[List[Dict]],
        max_examples: int = 100,
    ) -> Dict:
        """Analyze and categorize errors."""
        self.error_taxonomy = ErrorTaxonomy()

        for prompt, pred, gold in zip(prompts, predicted_spans, gold_spans):
            if len(self.error_taxonomy.counts) > max_examples:
                break

            # Check for false positives
            for p in pred:
                p_text = p.get("text", "")
                found_match = any(
                    g.get("text", "").lower() == p_text.lower() for g in gold
                )
                if not found_match:
                    cat = self.error_taxonomy.categorize_error(prompt, p, None, gold)
                    if cat:
                        self.error_taxonomy.add_error(cat, {
                            "prompt": prompt[:200],
                            "predicted": p_text,
                            "start": p.get("start"),
                            "end": p.get("end"),
                        })

            # Check for false negatives
            for g in gold:
                g_text = g.get("text", "")
                found_match = any(
                    p.get("text", "").lower() == g_text.lower() for p in pred
                )
                if not found_match:
                    self.error_taxonomy.add_error("FALSE_NEGATIVE", {
                        "prompt": prompt[:200],
                        "expected": g_text,
                        "start": g.get("start"),
                        "end": g.get("end"),
                    })

        return self.error_taxonomy.get_summary()

    def evaluate_cross_language(
        self,
        dataset,
        language_groups: Dict[str, List[int]],
    ) -> Dict:
        """Evaluate separately across programming languages."""
        results = {}

        for lang, indices in language_groups.items():
            if len(indices) < 10:
                continue  # Skip languages with too few examples

            # Create subset dataset
            subset = torch.utils.data.Subset(dataset, indices)
            eval_results = self.evaluate_dataset(subset, f"lang_{lang}")
            results[lang] = eval_results

        return results

    def evaluate_adversarial(
        self,
        adversarial_sets: Dict,
        threshold: float = 0.5,
    ) -> Dict:
        """Evaluate on adversarial test sets."""
        results = {}

        for test_name, examples in adversarial_sets.items():
            if not examples:
                continue

            correct = 0
            total = 0
            details = []

            for example in examples:
                if test_name == "template_variation":
                    prompt = example["original"]
                elif test_name == "contextual_reversal":
                    prompt = example["negative_prompt"]
                elif test_name == "semantic_substitution":
                    prompt = example["substituted"]
                else:
                    prompt = example.get("prompt", example.get("original", ""))

                # Tokenize and predict
                encoding = self.tokenizer(
                    prompt,
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                    return_offsets_mapping=True,
                )

                input_ids = torch.tensor([encoding["input_ids"]])
                attention_mask = torch.tensor([encoding["attention_mask"]])

                predictions, confidences = self.predict_batch(input_ids, attention_mask)

                spans = self.decode_spans(
                    predictions[0],
                    confidences[0],
                    attention_mask[0].numpy(),
                    encoding["offset_mapping"],
                    prompt,
                    threshold,
                )

                has_extraction = len(spans) > 0
                total += 1

                details.append({
                    "prompt": prompt[:200],
                    "extractions": [s["text"] for s in spans],
                    "has_extraction": has_extraction,
                })

            results[test_name] = {
                "total": total,
                "details": details,
            }

        return results

    def generate_report(self, results: Dict, output_dir: str):
        """Generate a comprehensive evaluation report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "evaluation_report.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Generate markdown summary
        lines = ["# Evaluation Report\n"]

        if "random_test" in results:
            rt = results["random_test"]
            lines.append("## Random Test Set Results\n")
            lines.append(f"- **Examples**: {rt['num_examples']}")
            lines.append(f"- **Token F1**: {rt['token_metrics']['f1']:.4f}")
            lines.append(f"- **Token Precision**: {rt['token_metrics']['precision']:.4f}")
            lines.append(f"- **Token Recall**: {rt['token_metrics']['recall']:.4f}")
            lines.append(f"- **Span F1**: {rt['span_metrics']['f1']:.4f}")
            lines.append(f"- **Exact Span Accuracy**: {rt['span_metrics']['exact_span_accuracy']:.4f}")
            lines.append("")

        if "concept_holdout" in results:
            ch = results["concept_holdout"]
            lines.append("## Concept Holdout Results\n")
            lines.append(f"- **Examples**: {ch['num_examples']}")
            lines.append(f"- **Token F1**: {ch['token_metrics']['f1']:.4f}")
            lines.append(f"- **Span F1**: {ch['span_metrics']['f1']:.4f}")
            lines.append("")

        if "unseen_vocabulary" in results:
            uv = results["unseen_vocabulary"]
            lines.append("## Unseen Vocabulary Results\n")
            lines.append(f"- **Examples**: {uv['num_examples']}")
            lines.append(f"- **Token F1**: {uv['token_metrics']['f1']:.4f}")
            lines.append(f"- **Span F1**: {uv['span_metrics']['f1']:.4f}")
            lines.append("")

        if "cross_language" in results:
            lines.append("## Cross-Language Results\n")
            lines.append("| Language | F1 | Precision | Recall |")
            lines.append("|----------|------|-----------|--------|")
            for lang, lr in results["cross_language"].items():
                lines.append(
                    f"| {lang} | {lr['span_metrics']['f1']:.4f} | "
                    f"{lr['span_metrics']['precision']:.4f} | "
                    f"{lr['span_metrics']['recall']:.4f} |"
                )
            lines.append("")

        if "error_analysis" in results:
            ea = results["error_analysis"].get("counts", {})
            lines.append("## Error Analysis\n")
            lines.append("| Category | Count |")
            lines.append("|----------|-------|")
            for cat, count in sorted(ea.items(), key=lambda x: -x[1]):
                lines.append(f"| {cat} | {count} |")
            lines.append("")

        with open(output_path / "evaluation_report.md", "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Evaluation report saved to {output_path}")
