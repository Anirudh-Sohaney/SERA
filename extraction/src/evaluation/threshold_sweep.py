"""
Threshold sweep for SERA product baseline.

Efficiently sweeps SpanFilter thresholds by caching Stage 1 candidates
and SpanFilter probabilities. Only the threshold cutoff changes.

Usage:
    python -m src.evaluation.threshold_sweep
    python -m src.evaluation.threshold_sweep --thresholds 0.5,0.6,0.7,0.8,0.9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.end_to_end import (
    build_gold_state,
    build_gold_transitions,
    error_attribution,
    false_lock_analysis,
)
from src.evaluation.product_baseline import (
    _aggregate_state_metrics,
    _gold_trans_dicts_to_objects,
    _write_outputs,
)
from src.inference.extractor import ExtractionSLM
from src.memory.engine import ProjectMemoryEngine
from src.memory.metrics import StateMetrics, compute_metrics
from src.memory.schema import MemoryCategory, ProjectState, Transition, TransitionType

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate cache: stores Stage 1 output + SpanFilter scores
# ---------------------------------------------------------------------------

class CandidateCache:
    """Caches Stage 1 candidates and SpanFilter probabilities for efficient
    threshold sweeping."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[Dict]] = {}  # prompt -> candidates

    def get_or_compute(
        self,
        prompt: str,
        stage1_fn,
        stage2_fn,
    ) -> List[Dict]:
        """Get cached candidates or compute and cache them."""
        if prompt in self._cache:
            return self._cache[prompt]

        # Stage 1: generate candidates
        candidates = stage1_fn(prompt)

        # Stage 2: score each candidate
        for cand in candidates:
            prob = stage2_fn(prompt, cand["text"])
            cand["stage2_prob"] = prob

        self._cache[prompt] = candidates
        return candidates

    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Threshold sweep evaluator
# ---------------------------------------------------------------------------

class ThresholdSweepEvaluator:
    """Evaluates SERA at multiple SpanFilter thresholds efficiently."""

    def __init__(
        self,
        stage1_model: ExtractionSLM,
        stage2_model: Any,
        stage2_tokenizer: Any,
        stage2_device: torch.device,
    ) -> None:
        self._stage1 = stage1_model
        self._stage2 = stage2_model
        self._stage2_tokenizer = stage2_tokenizer
        self._stage2_device = stage2_device
        self._cache = CandidateCache()

    def _stage1_extract(self, prompt: str) -> List[Dict]:
        """Run Stage 1 extraction."""
        result = self._stage1.extract(prompt)
        return result.spans if result.has_spans else []

    def _stage2_score(self, prompt: str, candidate_text: str) -> float:
        """Run SpanFilter scoring on a single candidate."""
        encoding = self._stage2_tokenizer(
            prompt,
            candidate_text,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self._stage2_device)
        attention_mask = encoding["attention_mask"].to(self._stage2_device)

        with torch.no_grad():
            outputs = self._stage2(input_ids, attention_mask)
            prob = outputs["probs"].item()

        return prob

    def load_conversations(
        self,
        aligned_records_path: str,
        splits_path: str,
        split: str = "test",
        min_turns: int = 1,
    ) -> List[Dict]:
        """Load conversations."""
        from src.evaluation.end_to_end import EndToEndEvaluator
        evaluator = EndToEndEvaluator(self._stage1)
        return evaluator.load_conversations(
            aligned_records_path, splits_path, split=split, min_turns=min_turns,
        )

    def evaluate_at_threshold(
        self,
        conversations: List[Dict],
        threshold: float,
        high_confidence: float = 0.9,
        medium_confidence: float = 0.7,
    ) -> Dict:
        """Evaluate all conversations at a specific threshold."""
        all_results = []
        all_metrics = []
        all_errors = []
        all_false_locks = []

        for idx, convo in enumerate(conversations):
            conv_id = convo["conversation_id"]
            if (idx + 1) % 100 == 0:
                logger.info(f"    [{idx + 1}/{len(conversations)}] threshold={threshold}")

            try:
                result = self._evaluate_conversation_at_threshold(
                    convo["turns"], threshold, high_confidence, medium_confidence,
                    conv_id,
                )
                all_results.append(result)

                m = StateMetrics(**{
                    k: v for k, v in result["aggregate_metrics"].items()
                    if hasattr(StateMetrics, k)
                })
                all_metrics.append(m)
                all_errors.extend(result.get("error_attribution", []))
                all_false_locks.extend(
                    result.get("false_lock_analysis", {}).get("false_locks", [])
                )
            except Exception as e:
                logger.error(f"Failed: {conv_id}: {e}")
                continue

        aggregate = _aggregate_state_metrics(all_metrics) if all_metrics else StateMetrics().__dict__

        error_summary = defaultdict(int)
        for err in all_errors:
            error_summary[err.get("code", "?")] += 1

        total_adds = sum(
            r.get("false_lock_analysis", {}).get("total_adds", 0)
            for r in all_results
        )
        total_flocks = sum(
            r.get("false_lock_analysis", {}).get("false_lock_count", 0)
            for r in all_results
        )

        return {
            "threshold": threshold,
            "aggregate": aggregate,
            "num_conversations": len(all_results),
            "num_failed": len(conversations) - len(all_results),
            "error_attribution_summary": dict(error_summary),
            "false_lock_summary": {
                "total_adds": total_adds,
                "total_false_locks": total_flocks,
                "false_lock_rate": total_flocks / total_adds if total_adds > 0 else 0.0,
            },
        }

    def _evaluate_conversation_at_threshold(
        self,
        turns: List[Dict],
        threshold: float,
        high_confidence: float,
        medium_confidence: float,
        conv_id: str,
    ) -> Dict:
        """Evaluate one conversation at a specific threshold."""
        engine = ProjectMemoryEngine(
            project_id=f"sw_{conv_id}",
            high_confidence_threshold=0.0,
            medium_confidence_threshold=0.0,
        )

        gold_states = build_gold_state(turns)
        gold_trans = build_gold_transitions(turns)

        all_pred_spans = []
        all_gold_spans = []
        all_pred_transitions = []
        all_gold_transitions_flat = []
        per_turn = []

        for turn_idx, turn_entry in enumerate(turns):
            record = turn_entry["record"]
            gold_spans = turn_entry.get("spans", [])
            turn_number = int(record.get("turn", 0)) + 1
            prompt = record.get("input", {}).get("user_prompt", "")

            # Get cached candidates
            candidates = self._cache.get_or_compute(
                prompt, self._stage1_extract, self._stage2_score,
            )

            # Apply threshold
            accepted = [c for c in candidates if c.get("stage2_prob", 0) >= threshold]

            # Validate offsets
            validated = []
            for span in accepted:
                extracted = prompt[span["start"]:span["end"]]
                if extracted == span["text"]:
                    validated.append(span)
                else:
                    idx = prompt.find(span["text"])
                    if idx >= 0:
                        span["start"] = idx
                        span["end"] = idx + len(span["text"])
                        validated.append(span)

            all_gold_spans.append(gold_spans)
            all_pred_spans.append(validated)

            # Process through engine
            try:
                engine_result = engine.process_turn(
                    prompt=prompt,
                    extracted_spans=validated,
                    turn_number=turn_number,
                )
            except Exception:
                engine_result = {"transitions": [], "state_snapshot": {"active_memories": [], "all_memories_count": 0, "current_turn": turn_number}, "audit_records": [], "validation_result": type("R", (), {"valid": False, "errors": [], "warnings": []})()}

            pred_transitions = engine_result["transitions"]
            all_pred_transitions.extend(pred_transitions)

            gold_turn_trans = _gold_trans_dicts_to_objects(
                gold_trans[turn_idx] if turn_idx < len(gold_trans) else [],
                prompt, turn_number,
            )
            all_gold_transitions_flat.extend(gold_turn_trans)

            turn_metrics = compute_metrics(
                predicted_state=engine.get_project_state(),
                expected_state=ProjectState.from_dict(gold_states[turn_idx]) if turn_idx < len(gold_states) else ProjectState(),
                predicted_spans=validated,
                gold_spans=gold_spans,
                predicted_transitions=pred_transitions,
                gold_transitions=gold_turn_trans,
            )

            per_turn.append({
                "turn": turn_number,
                "turn_idx": turn_idx,
                "prompt": prompt,
                "conversation_id": conv_id,
                "num_pred_spans": len(validated),
                "num_gold_spans": len(gold_spans),
                "num_transitions": len(pred_transitions),
                "candidate_count": len(candidates),
                "accepted_count": len(validated),
                "discarded_count": len(candidates) - len(validated),
                "transition_types": {t.transition_type.value: sum(1 for tt in pred_transitions if tt.transition_type == t.transition_type) for t in pred_transitions} if pred_transitions else {},
                "metrics": turn_metrics.__dict__,
                "predicted_transitions": pred_transitions,
            })

        aggregate = compute_metrics(
            predicted_state=engine.get_project_state(),
            expected_state=ProjectState.from_dict(gold_states[-1]) if gold_states else ProjectState(),
            predicted_spans=[s for ts in all_pred_spans for s in ts],
            gold_spans=[s for ts in all_gold_spans for s in ts],
            predicted_transitions=all_pred_transitions,
            gold_transitions=all_gold_transitions_flat,
        )

        pred_state = engine.get_project_state()
        gold_final = ProjectState.from_dict(gold_states[-1]) if gold_states else ProjectState()
        attribution = error_attribution(
            predicted_memory=pred_state,
            gold_memory=gold_final,
            predicted_transitions=all_pred_transitions,
            gold_transitions=gold_trans,
            predicted_spans=all_pred_spans,
            gold_spans=all_gold_spans,
        )
        flocks = false_lock_analysis(per_turn, gold_trans)

        return {
            "conversation_id": conv_id,
            "num_turns": len(turns),
            "aggregate_metrics": aggregate.__dict__,
            "error_attribution": attribution,
            "false_lock_analysis": flocks,
        }

    def run_sweep(
        self,
        conversations: List[Dict],
        thresholds: List[float],
        output_dir: str,
    ) -> List[Dict]:
        """Run threshold sweep and save results."""
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for thresh in thresholds:
            logger.info(f"=== Evaluating threshold={thresh} ===")
            start = time.time()
            result = self.evaluate_at_threshold(conversations, thresh)
            elapsed = time.time() - start
            result["elapsed_seconds"] = elapsed
            results.append(result)

            agg = result["aggregate"]
            flock = result["false_lock_summary"]
            logger.info(
                f"  threshold={thresh}: "
                f"FLR={flock['false_lock_rate']:.4f}, "
                f"P={agg['state_precision']:.4f}, "
                f"R={agg['state_recall']:.4f}, "
                f"F1={agg['state_f1']:.4f}, "
                f"SpanF1={agg['span_f1']:.4f}, "
                f"time={elapsed:.1f}s"
            )

        # Save sweep results
        sweep_data = []
        for r in results:
            sweep_data.append({
                "threshold": r["threshold"],
                "false_lock_rate": r["false_lock_summary"]["false_lock_rate"],
                "total_adds": r["false_lock_summary"]["total_adds"],
                "total_false_locks": r["false_lock_summary"]["total_false_locks"],
                "span_precision": r["aggregate"]["span_precision"],
                "span_recall": r["aggregate"]["span_recall"],
                "span_f1": r["aggregate"]["span_f1"],
                "state_precision": r["aggregate"]["state_precision"],
                "state_recall": r["aggregate"]["state_recall"],
                "state_f1": r["aggregate"]["state_f1"],
                "add_count": r["aggregate"]["add_count"],
                "modify_count": r["aggregate"]["modify_count"],
                "reject_count": r["aggregate"]["reject_count"],
                "stale_memory_rate": r["aggregate"]["stale_memory_rate"],
                "elapsed_seconds": r["elapsed_seconds"],
            })

        with open(os.path.join(output_dir, "threshold_sweep.json"), "w") as f:
            json.dump(sweep_data, f, indent=2)

        # Build summary
        summary = self._build_sweep_summary(sweep_data)
        with open(os.path.join(output_dir, "summary.md"), "w") as f:
            f.write(summary)

        logger.info(f"Sweep results saved to {output_dir}")
        return results

    def _build_sweep_summary(self, sweep_data: List[Dict]) -> str:
        """Build Markdown summary of threshold sweep."""
        lines = [
            "# SERA Threshold Sweep Results",
            "",
            f"**Timestamp:** {datetime.now(timezone.utc).isoformat()}",
            f"**Thresholds tested:** {len(sweep_data)}",
            "",
            "## Results by Threshold",
            "",
            "| Threshold | FLR | State P | State R | State F1 | Span F1 | ADDs | False Locks | Time (s) |",
            "|-----------|-----|---------|---------|----------|---------|------|-------------|----------|",
        ]

        for d in sweep_data:
            lines.append(
                f"| {d['threshold']:.2f} "
                f"| {d['false_lock_rate']:.4f} "
                f"| {d['state_precision']:.4f} "
                f"| {d['state_recall']:.4f} "
                f"| {d['state_f1']:.4f} "
                f"| {d['span_f1']:.4f} "
                f"| {d['add_count']} "
                f"| {d['total_false_locks']} "
                f"| {d['elapsed_seconds']:.1f} |"
            )

        # Find best by product constraints
        # Minimum acceptable: FLR < 0.5, State P > 0.15, State R > 0.10
        candidates = [
            d for d in sweep_data
            if d["false_lock_rate"] < 0.5
            and d["state_precision"] > 0.15
            and d["state_recall"] > 0.10
        ]

        if candidates:
            best = min(candidates, key=lambda x: x["false_lock_rate"])
            lines.extend([
                "",
                "## Best Threshold (Product Constraints)",
                "",
                f"Constraints: FLR < 0.5, State P > 0.15, State R > 0.10",
                f"**Best threshold: {best['threshold']:.2f}**",
                f"- False Lock Rate: {best['false_lock_rate']:.4f}",
                f"- State Precision: {best['state_precision']:.4f}",
                f"- State Recall: {best['state_recall']:.4f}",
                f"- State F1: {best['state_f1']:.4f}",
            ])
        else:
            # Find best FLR that still has reasonable recall
            recall_candidates = [d for d in sweep_data if d["state_recall"] > 0.05]
            if recall_candidates:
                best = min(recall_candidates, key=lambda x: x["false_lock_rate"])
                lines.extend([
                    "",
                    "## Best Threshold (No constraint combination met)",
                    "",
                    f"Best FLR with recall > 0.05: threshold={best['threshold']:.2f}",
                    f"- False Lock Rate: {best['false_lock_rate']:.4f}",
                    f"- State Precision: {best['state_precision']:.4f}",
                    f"- State Recall: {best['state_recall']:.4f}",
                    f"- State F1: {best['state_f1']:.4f}",
                ])
            else:
                lines.extend([
                    "",
                    "## No threshold met minimum recall constraint",
                    "",
                    "All thresholds produced state recall < 0.05.",
                ])

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SERA Threshold Sweep")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated thresholds (default: 16-point sweep)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]
    else:
        thresholds = [
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
            0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
        ]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"logs/product_baseline/threshold_sweep_{timestamp}"

    data_dir = PROJECT_ROOT / "data" / "processed"
    aligned_path = str(data_dir / "aligned_records.jsonl")
    splits_path = str(data_dir / "splits.json")

    # Load models
    logger.info("Loading Stage 1 (E6-A)...")
    stage1 = ExtractionSLM.from_checkpoint(
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints" / "oracle_e6a" / "best"),
        model_name="google/bert_uncased_L-6_H-512_A-8",
        confidence_threshold=0.0,
    )

    logger.info("Loading Stage 2 (SpanFilter)...")
    from src.models.span_filter import SpanFilter
    from transformers import AutoTokenizer

    s2_model = SpanFilter(model_name="google/bert_uncased_L-6_H-512_A-8")
    s2_ckpt = PROJECT_ROOT / "checkpoints" / "experiment_e7_span_filter" / "best" / "model.pt"
    state_dict = torch.load(str(s2_ckpt), map_location="cpu", weights_only=True)
    s2_model.load_state_dict(state_dict)
    s2_model.eval()

    s2_tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-6_H-512_A-8")
    device = torch.device("cpu")

    logger.info("Loading conversations...")
    evaluator = ThresholdSweepEvaluator(
        stage1_model=stage1,
        stage2_model=s2_model,
        stage2_tokenizer=s2_tokenizer,
        stage2_device=device,
    )
    conversations = evaluator.load_conversations(
        aligned_path, splits_path, split=args.split, min_turns=args.min_turns,
    )
    logger.info(f"Loaded {len(conversations)} conversations")

    # Run sweep
    logger.info(f"Running threshold sweep: {thresholds}")
    results = evaluator.run_sweep(conversations, thresholds, output_dir)

    # Print best
    best_flr = min(results, key=lambda r: r["false_lock_summary"]["false_lock_rate"])
    print(f"\nBest FLR: threshold={best_flr['threshold']}, FLR={best_flr['false_lock_summary']['false_lock_rate']:.4f}")


if __name__ == "__main__":
    main()
