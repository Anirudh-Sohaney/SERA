"""
Stage-1 candidate error analysis for SERA.

Analyzes why Stage 1 generates false positives and misses true positives.
Breaks down errors by multiple dimensions to inform targeted data acquisition.

Usage:
    python -m src.evaluation.candidate_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.extractor import ExtractionSLM
from src.memory.transitions import _infer_category_from_text

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text analysis helpers
# ---------------------------------------------------------------------------

def _span_length_bins(text: str) -> Dict[str, bool]:
    """Classify span by length."""
    word_count = len(text.split())
    char_count = len(text)
    return {
        "single_word": word_count == 1,
        "short_2_3": 2 <= word_count <= 3,
        "medium_4_6": 4 <= word_count <= 6,
        "long_7_plus": word_count >= 7,
        "char_under_10": char_count < 10,
        "char_10_30": 10 <= char_count < 30,
        "char_30_plus": char_count >= 30,
    }


def _position_in_prompt(text: str, prompt: str) -> Dict[str, Any]:
    """Determine where in the prompt the span appears."""
    idx = prompt.find(text)
    if idx < 0:
        return {"position": "not_found", "relative_position": 0.5}

    prompt_len = len(prompt)
    relative = idx / max(prompt_len, 1)

    return {
        "position": "start" if relative < 0.33 else ("middle" if relative < 0.66 else "end"),
        "relative_position": relative,
        "char_offset": idx,
    }


def _is_technology(text: str) -> bool:
    """Check if text looks like a technology mention."""
    tech_indicators = [
        r"\b(python|java|javascript|typescript|rust|go|ruby|php|c\+\+|c#)\b",
        r"\b(flask|django|fastapi|express|react|vue|angular|svelte)\b",
        r"\b(postgresql|mysql|sqlite|mongodb|redis|elasticsearch)\b",
        r"\b(aws|gcp|azure|docker|kubernetes|terraform)\b",
        r"\b(node|npm|pip|cargo|brew)\b",
        r"\b(html|css|sql|json|yaml|toml|xml)\b",
        r"\b git\b",
        r"\bapi\b",
        r"\bcli\b",
        r"\bsdk\b",
    ]
    text_lower = text.lower()
    for pattern in tech_indicators:
        if re.search(pattern, text_lower):
            return True
    return False


def _is_description(text: str) -> bool:
    """Check if text is description language rather than a specific entity."""
    desc_patterns = [
        r"^(a|an|the)\s+",
        r"\s+(that|which|who)\s+",
        r"\s+(to|for|of|in|on|at)\s+",
        r"^(create|build|make|write|implement|develop|set up|design)\b",
        r"^(how|what|where|when|why)\s+",
        r"\?$",
        r"^(good|bad|nice|great|better|worse|fast|slow|simple|easy|complex)\b",
    ]
    text_lower = text.lower()
    for pattern in desc_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def _is_code_syntax(text: str) -> bool:
    """Check if text looks like code syntax."""
    code_patterns = [
        r"[{}\[\]()]",
        r"->|=>|::",
        r"import\s+\w+",
        r"from\s+\w+\s+import",
        r"def\s+\w+",
        r"class\s+\w+",
        r"function\s+\w+",
        r"const\s+\w+",
        r"let\s+\w+",
        r"var\s+\w+",
    ]
    for pattern in code_patterns:
        if re.search(pattern, text):
            return True
    return False


def _is_negated(text: str, prompt: str) -> bool:
    """Check if span is in a negation context."""
    negation_patterns = [
        r"\bnot?\s+use\b",
        r"\bdo\s+not\s+use\b",
        r"\bdon'?t\s+use\b",
        r"\bavoid\b",
        r"\binstead\s+of\b",
        r"\brather\s+than\b",
        r"\bno\s+longer\b",
        r"\bstop\s+using\b",
    ]
    # Check context around the span
    idx = prompt.find(text)
    if idx < 0:
        return False
    context = prompt[max(0, idx - 50):idx + len(text) + 50].lower()
    for pattern in negation_patterns:
        if re.search(pattern, context):
            return True
    return False


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def analyze_candidates(
    conversations: List[Dict],
    extractor: ExtractionSLM,
    output_dir: str,
) -> Dict[str, Any]:
    """Run full candidate error analysis."""
    os.makedirs(output_dir, exist_ok=True)

    all_fp_examples = []
    all_fn_examples = []
    all_tp_examples = []

    # Distributions
    fp_conf_dist = []
    fn_conf_dist = []
    tp_conf_dist = []

    fp_length_dist = []
    fn_length_dist = []

    fp_category_dist = Counter()
    fn_category_dist = Counter()
    tp_category_dist = Counter()

    fp_position_dist = Counter()
    fn_position_dist = Counter()

    fp_technology_count = 0
    fp_description_count = 0
    fp_code_count = 0
    fp_negated_count = 0

    fn_technology_count = 0
    fn_description_count = 0

    total_candidates = 0
    total_gold = 0

    for idx, convo in enumerate(conversations):
        if (idx + 1) % 200 == 0:
            logger.info(f"  [{idx + 1}/{len(conversations)}]")

        for turn_entry in convo["turns"]:
            record = turn_entry["record"]
            gold_spans = turn_entry.get("spans", [])
            prompt = record.get("input", {}).get("user_prompt", "")

            if not prompt:
                continue

            # Run extractor
            try:
                result = extractor.extract(prompt)
                pred_spans = result.spans if result.has_spans else []
            except Exception:
                pred_spans = []

            gold_texts = {s["text"].lower().strip() for s in gold_spans}
            pred_texts = {s["text"].lower().strip() for s in pred_spans}

            # True positives
            tp_texts = gold_texts & pred_texts
            # False positives
            fp_texts = pred_texts - gold_texts
            # False negatives
            fn_texts = gold_texts - pred_texts

            total_candidates += len(pred_spans)
            total_gold += len(gold_spans)

            # Analyze TPs
            for span in pred_spans:
                if span["text"].lower().strip() in tp_texts:
                    tp_conf_dist.append(span["confidence"])
                    cat = _infer_category_from_text(span["text"])
                    tp_category_dist[cat.value] += 1
                    all_tp_examples.append({
                        "text": span["text"],
                        "confidence": span["confidence"],
                        "category": cat.value,
                        "prompt": prompt[:200],
                    })

            # Analyze FPs
            for span in pred_spans:
                if span["text"].lower().strip() in fp_texts:
                    conf = span["confidence"]
                    fp_conf_dist.append(conf)

                    # Length analysis
                    length_info = _span_length_bins(span["text"])
                    fp_length_dist.append(length_info)

                    # Category
                    cat = _infer_category_from_text(span["text"])
                    fp_category_dist[cat.value] += 1

                    # Position
                    pos_info = _position_in_prompt(span["text"], prompt)
                    fp_position_dist[pos_info["position"]] += 1

                    # Type analysis
                    is_tech = _is_technology(span["text"])
                    is_desc = _is_description(span["text"])
                    is_code = _is_code_syntax(span["text"])
                    is_neg = _is_negated(span["text"], prompt)

                    if is_tech:
                        fp_technology_count += 1
                    if is_desc:
                        fp_description_count += 1
                    if is_code:
                        fp_code_count += 1
                    if is_neg:
                        fp_negated_count += 1

                    all_fp_examples.append({
                        "text": span["text"],
                        "confidence": conf,
                        "category": cat.value,
                        "is_technology": is_tech,
                        "is_description": is_desc,
                        "is_code": is_code,
                        "is_negated": is_neg,
                        "prompt": prompt[:200],
                        "length_words": len(span["text"].split()),
                        "position": pos_info["position"],
                    })

            # Analyze FNs
            for gold_span in gold_spans:
                if gold_span["text"].lower().strip() in fn_texts:
                    cat = _infer_category_from_text(gold_span["text"])
                    fn_category_dist[cat.value] += 1
                    fn_position_dist[_position_in_prompt(gold_span["text"], prompt)["position"]] += 1

                    is_tech = _is_technology(gold_span["text"])
                    is_desc = _is_description(gold_span["text"])
                    if is_tech:
                        fn_technology_count += 1
                    if is_desc:
                        fn_description_count += 1

                    all_fn_examples.append({
                        "text": gold_span["text"],
                        "category": cat.value,
                        "is_technology": is_tech,
                        "is_description": is_desc,
                        "prompt": prompt[:200],
                        "field": gold_span.get("field", ""),
                    })

    # Summarize
    fp_count = len(all_fp_examples)
    fn_count = len(all_fn_examples)
    tp_count = len(all_tp_examples)

    distribution = {
        "total_candidates": total_candidates,
        "total_gold_spans": total_gold,
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "precision": tp_count / max(tp_count + fp_count, 1),
        "recall": tp_count / max(tp_count + fn_count, 1),
    }

    fp_analysis = {
        "count": fp_count,
        "confidence_distribution": {
            "mean": float(np.mean(fp_conf_dist)) if fp_conf_dist else 0,
            "median": float(np.median(fp_conf_dist)) if fp_conf_dist else 0,
            "std": float(np.std(fp_conf_dist)) if fp_conf_dist else 0,
            "p25": float(np.percentile(fp_conf_dist, 25)) if fp_conf_dist else 0,
            "p75": float(np.percentile(fp_conf_dist, 75)) if fp_conf_dist else 0,
        },
        "category_distribution": dict(fp_category_dist.most_common()),
        "position_distribution": dict(fp_position_dist.most_common()),
        "type_counts": {
            "technology": fp_technology_count,
            "description": fp_description_count,
            "code_syntax": fp_code_count,
            "negated": fp_negated_count,
        },
        "length_distribution": {
            "single_word": sum(1 for l in fp_length_dist if l.get("single_word")),
            "short_2_3": sum(1 for l in fp_length_dist if l.get("short_2_3")),
            "medium_4_6": sum(1 for l in fp_length_dist if l.get("medium_4_6")),
            "long_7_plus": sum(1 for l in fp_length_dist if l.get("long_7_plus")),
        },
    }

    fn_analysis = {
        "count": fn_count,
        "category_distribution": dict(fn_category_dist.most_common()),
        "position_distribution": dict(fn_position_dist.most_common()),
        "type_counts": {
            "technology": fn_technology_count,
            "description": fn_description_count,
        },
    }

    tp_analysis = {
        "count": tp_count,
        "confidence_distribution": {
            "mean": float(np.mean(tp_conf_dist)) if tp_conf_dist else 0,
            "median": float(np.median(tp_conf_dist)) if tp_conf_dist else 0,
        },
        "category_distribution": dict(tp_category_dist.most_common()),
    }

    # Save all
    results = {
        "distribution": distribution,
        "false_positive_analysis": fp_analysis,
        "false_negative_analysis": fn_analysis,
        "true_positive_analysis": tp_analysis,
    }

    with open(os.path.join(output_dir, "distribution.json"), "w") as f:
        json.dump(distribution, f, indent=2)

    with open(os.path.join(output_dir, "false_positive_analysis.json"), "w") as f:
        json.dump(fp_analysis, f, indent=2)

    with open(os.path.join(output_dir, "false_negative_analysis.json"), "w") as f:
        json.dump(fn_analysis, f, indent=2)

    with open(os.path.join(output_dir, "true_positive_analysis.json"), "w") as f:
        json.dump(tp_analysis, f, indent=2)

    # Save examples (limit to 500 each)
    with open(os.path.join(output_dir, "fp_examples.jsonl"), "w") as f:
        for ex in all_fp_examples[:500]:
            f.write(json.dumps(ex) + "\n")

    with open(os.path.join(output_dir, "fn_examples.jsonl"), "w") as f:
        for ex in all_fn_examples[:500]:
            f.write(json.dumps(ex) + "\n")

    # Build summary
    summary = _build_summary(results)
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write(summary)

    logger.info(f"Analysis saved to {output_dir}")
    return results


def _build_summary(results: Dict) -> str:
    dist = results["distribution"]
    fp = results["false_positive_analysis"]
    fn = results["false_negative_analysis"]
    tp = results["true_positive_analysis"]

    lines = [
        "# Stage-1 Candidate Error Analysis",
        "",
        f"**Timestamp:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Distribution",
        "",
        f"- **Total candidates:** {dist['total_candidates']}",
        f"- **Total gold spans:** {dist['total_gold_spans']}",
        f"- **True positives:** {dist['true_positives']}",
        f"- **False positives:** {dist['false_positives']}",
        f"- **False negatives:** {dist['false_negatives']}",
        f"- **Precision:** {dist['precision']:.4f}",
        f"- **Recall:** {dist['recall']:.4f}",
        "",
        "## False Positive Analysis",
        "",
        f"**Count:** {fp['count']}",
        "",
        "### Confidence Distribution",
        f"- Mean: {fp['confidence_distribution']['mean']:.4f}",
        f"- Median: {fp['confidence_distribution']['median']:.4f}",
        f"- P25: {fp['confidence_distribution']['p25']:.4f}",
        f"- P75: {fp['confidence_distribution']['p75']:.4f}",
        "",
        "### Category Distribution",
    ]
    for cat, count in sorted(fp["category_distribution"].items(), key=lambda x: -x[1])[:10]:
        lines.append(f"- {cat}: {count} ({count/max(fp['count'],1)*100:.1f}%)")

    lines.extend(["", "### Type Analysis"])
    for typ, count in fp["type_counts"].items():
        lines.append(f"- {typ}: {count} ({count/max(fp['count'],1)*100:.1f}%)")

    lines.extend(["", "### Length Distribution"])
    for length, count in fp["length_distribution"].items():
        lines.append(f"- {length}: {count} ({count/max(fp['count'],1)*100:.1f}%)")

    lines.extend(["", "### Position Distribution"])
    for pos, count in fp["position_distribution"].items():
        lines.append(f"- {pos}: {count} ({count/max(fp['count'],1)*100:.1f}%)")

    lines.extend([
        "",
        "## False Negative Analysis",
        "",
        f"**Count:** {fn['count']}",
        "",
        "### Category Distribution",
    ])
    for cat, count in sorted(fn["category_distribution"].items(), key=lambda x: -x[1])[:10]:
        lines.append(f"- {cat}: {count} ({count/max(fn['count'],1)*100:.1f}%)")

    lines.extend(["", "### Type Analysis"])
    for typ, count in fn["type_counts"].items():
        lines.append(f"- {typ}: {count} ({count/max(fn['count'],1)*100:.1f}%)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SERA Candidate Error Analysis")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"logs/candidate_analysis/{timestamp}"

    data_dir = PROJECT_ROOT / "data" / "processed"
    aligned_path = str(data_dir / "aligned_records.jsonl")
    splits_path = str(data_dir / "splits.json")

    # Load extractor
    logger.info("Loading extractor...")
    extractor = ExtractionSLM.from_checkpoint(
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints" / "oracle_e6a" / "best"),
        model_name="google/bert_uncased_L-6_H-512_A-8",
        confidence_threshold=0.0,
    )

    # Load conversations
    logger.info("Loading conversations...")
    from src.evaluation.end_to_end import EndToEndEvaluator
    evaluator = EndToEndEvaluator(extractor)
    conversations = evaluator.load_conversations(
        aligned_path, splits_path, split=args.split, min_turns=args.min_turns,
    )
    logger.info(f"Loaded {len(conversations)} conversations")

    # Run analysis
    results = analyze_candidates(conversations, extractor, output_dir)

    # Print summary
    dist = results["distribution"]
    print(f"\nTotal candidates: {dist['total_candidates']}")
    print(f"True positives: {dist['true_positives']}")
    print(f"False positives: {dist['false_positives']}")
    print(f"False negatives: {dist['false_negatives']}")
    print(f"Precision: {dist['precision']:.4f}")
    print(f"Recall: {dist['recall']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
