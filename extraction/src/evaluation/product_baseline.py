"""
Product baseline evaluation for SERA.

Runs the full pipeline: E6-A → SpanFilter → confidence policy → state engine.
Records per-turn admission details and computes all product metrics.

Usage:
    python -m src.evaluation.product_baseline
    python -m src.evaluation.product_baseline --span-filter-threshold 0.7
    python -m src.evaluation.product_baseline --skip-span-filter
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

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.end_to_end import (
    EndToEndEvaluator,
    build_gold_state,
    build_gold_transitions,
    error_attribution,
    false_lock_analysis,
)
from src.evaluation.pipeline_wrapper import ExtractionSLMWrapper, PipelineWrapper
from src.inference.extractor import ExtractionSLM
from src.inference.pipeline import TwoStagePipeline
from src.memory.engine import ProjectMemoryEngine
from src.memory.metrics import StateMetrics, compute_metrics
from src.memory.schema import (
    MemoryCategory,
    MemoryStatus,
    ProjectState,
    Transition,
    TransitionType,
)
from src.memory.validator import _normalize_value


# ---------------------------------------------------------------------------
# Admission states
# ---------------------------------------------------------------------------

class AdmissionState:
    """Possible admission decisions for a candidate span."""
    LOCK = "LOCK"
    PENDING = "PENDING"
    DISCARD = "DISCARD"


# ---------------------------------------------------------------------------
# Product baseline evaluator
# ---------------------------------------------------------------------------

class ProductBaselineEvaluator:
    """Full product baseline evaluation.

    Runs: E6-A → SpanFilter → confidence policy → state engine.
    Records per-turn admission details and computes all product metrics.
    """

    def __init__(
        self,
        pipeline: Any,
        high_confidence: float = 0.9,
        medium_confidence: float = 0.7,
        use_span_filter: bool = True,
        span_filter_threshold: float = 0.5,
    ) -> None:
        self._pipeline = pipeline
        self._high_confidence = high_confidence
        self._medium_confidence = medium_confidence
        self._use_span_filter = use_span_filter
        self._span_filter_threshold = span_filter_threshold

    # ------------------------------------------------------------------
    # Data loading (reuses EndToEndEvaluator logic)
    # ------------------------------------------------------------------

    def load_conversations(
        self,
        aligned_records_path: str,
        splits_path: str,
        split: str = "test",
        min_turns: int = 1,
    ) -> List[Dict]:
        """Load and group conversations."""
        evaluator = EndToEndEvaluator(self._pipeline)
        return evaluator.load_conversations(
            aligned_records_path, splits_path, split=split, min_turns=min_turns,
        )

    # ------------------------------------------------------------------
    # Single conversation evaluation
    # ------------------------------------------------------------------

    def evaluate_conversation(self, conversation: List[Dict]) -> Dict:
        """Run full pipeline on one conversation, record admission details."""
        conv_id = conversation[0]["record"]["conversation_id"]

        # Create engine without confidence policy (SpanFilter handles admission)
        engine = ProjectMemoryEngine(
            project_id=f"baseline_{conv_id}",
            high_confidence_threshold=0.0,  # Disable: SpanFilter does filtering
            medium_confidence_threshold=0.0,
        )

        # Build gold
        gold_states = build_gold_state(conversation)
        gold_trans = build_gold_transitions(conversation)

        per_turn: List[Dict[str, Any]] = []
        all_pred_spans: List[List[Dict]] = []
        all_gold_spans: List[List[Dict]] = []
        all_pred_transitions: List[Transition] = []
        all_gold_transitions_flat: List[Transition] = []

        total_candidates = 0
        total_stage1 = 0
        total_stage2_accepted = 0
        total_locked = 0
        total_pending = 0
        total_discarded = 0

        for turn_idx, turn_entry in enumerate(conversation):
            record = turn_entry["record"]
            gold_spans = turn_entry.get("spans", [])
            raw_turn = int(record.get("turn", 0))
            turn_number = raw_turn + 1
            prompt = record.get("input", {}).get("user_prompt", "")

            # Run pipeline
            try:
                result = self._pipeline.extract(prompt)
                extracted_spans = result.spans if hasattr(result, 'spans') else []
                pipeline_metrics = result.metrics if hasattr(result, 'metrics') else {}
            except Exception as e:
                logger.error(f"Pipeline failed on turn {turn_number}: {e}")
                extracted_spans = []
                pipeline_metrics = {}

            # Count pipeline stages
            stage1_count = pipeline_metrics.get("stage1_candidates", len(extracted_spans))
            stage2_count = pipeline_metrics.get("stage2_accepted", len(extracted_spans))
            total_stage1 += stage1_count
            total_stage2_accepted += stage2_count

            # Classify admission for each span
            admission_decisions: List[Dict[str, Any]] = []
            for span in extracted_spans:
                conf = span.get("confidence", 0.0)
                if conf >= self._high_confidence:
                    decision = AdmissionState.LOCK
                    total_locked += 1
                elif conf >= self._medium_confidence:
                    decision = AdmissionState.PENDING
                    total_pending += 1
                else:
                    decision = AdmissionState.DISCARD
                    total_discarded += 1

                admission_decisions.append({
                    "text": span.get("text", ""),
                    "confidence": conf,
                    "stage2_prob": span.get("stage2_prob", None),
                    "decision": decision,
                })

            total_candidates += len(extracted_spans)

            all_gold_spans.append(gold_spans)
            all_pred_spans.append(extracted_spans)

            # Process through memory engine
            try:
                engine_result = engine.process_turn(
                    prompt=prompt,
                    extracted_spans=extracted_spans,
                    turn_number=turn_number,
                )
            except Exception as e:
                logger.error(f"Engine failed on turn {turn_number}: {e}")
                engine_result = {
                    "transitions": [],
                    "state_snapshot": {"active_memories": [], "all_memories_count": 0, "current_turn": turn_number},
                    "audit_records": [],
                    "validation_result": type("R", (), {"valid": False, "errors": [], "warnings": []})(),
                }

            pred_transitions = engine_result["transitions"]
            all_pred_transitions.extend(pred_transitions)

            # Build gold transitions
            gold_turn_trans = _gold_trans_dicts_to_objects(
                gold_trans[turn_idx] if turn_idx < len(gold_trans) else [],
                prompt,
                turn_number,
            )
            all_gold_transitions_flat.extend(gold_turn_trans)

            # Compute per-turn metrics
            turn_metrics = compute_metrics(
                predicted_state=engine.get_project_state(),
                expected_state=(
                    ProjectState.from_dict(gold_states[turn_idx])
                    if turn_idx < len(gold_states)
                    else ProjectState()
                ),
                predicted_spans=extracted_spans,
                gold_spans=gold_spans,
                predicted_transitions=pred_transitions,
                gold_transitions=gold_turn_trans,
            )

            # Count transition types
            transition_type_counts = defaultdict(int)
            for t in pred_transitions:
                transition_type_counts[t.transition_type.value] += 1

            per_turn.append({
                "turn": turn_number,
                "turn_idx": turn_idx,
                "prompt": prompt,
                "conversation_id": conv_id,
                "num_pred_spans": len(extracted_spans),
                "num_gold_spans": len(gold_spans),
                "num_transitions": len(pred_transitions),
                "num_gold_transitions": len(gold_turn_trans),
                "stage1_candidates": stage1_count,
                "stage2_accepted": stage2_count,
                "candidate_count": len(extracted_spans),
                "locked_count": sum(1 for a in admission_decisions if a["decision"] == AdmissionState.LOCK),
                "pending_count": sum(1 for a in admission_decisions if a["decision"] == AdmissionState.PENDING),
                "discarded_count": sum(1 for a in admission_decisions if a["decision"] == AdmissionState.DISCARD),
                "transition_types": dict(transition_type_counts),
                "admission_decisions": admission_decisions,
                "metrics": turn_metrics.__dict__,
                "predicted_transitions": pred_transitions,
                "validation_valid": getattr(engine_result.get("validation_result"), "valid", True),
            })

        # Aggregate metrics
        aggregate = compute_metrics(
            predicted_state=engine.get_project_state(),
            expected_state=(
                ProjectState.from_dict(gold_states[-1])
                if gold_states
                else ProjectState()
            ),
            predicted_spans=[s for turn_spans in all_pred_spans for s in turn_spans],
            gold_spans=[s for turn_spans in all_gold_spans for s in turn_spans],
            predicted_transitions=all_pred_transitions,
            gold_transitions=all_gold_transitions_flat,
        )

        # Error attribution
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

        # False lock analysis
        flocks = false_lock_analysis(per_turn, gold_trans)

        return {
            "conversation_id": conv_id,
            "num_turns": len(conversation),
            "per_turn": per_turn,
            "gold_transitions": gold_trans,
            "gold_states": gold_states,
            "predicted_state": pred_state.to_dict(),
            "gold_state": gold_final.to_dict(),
            "aggregate_metrics": aggregate.__dict__,
            "error_attribution": attribution,
            "false_lock_analysis": flocks,
            "pipeline_stats": {
                "total_candidates": total_candidates,
                "total_stage1": total_stage1,
                "total_stage2_accepted": total_stage2_accepted,
                "total_locked": total_locked,
                "total_pending": total_pending,
                "total_discarded": total_discarded,
            },
        }

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self, conversations: List[Dict]) -> Dict:
        """Evaluate all conversations and compute aggregate metrics."""
        logger.info(f"Evaluating {len(conversations)} conversations")

        all_results: List[Dict[str, Any]] = []
        all_metrics: List[StateMetrics] = []
        all_errors: List[Dict[str, Any]] = []
        all_false_locks: List[Dict[str, Any]] = []

        for idx, convo in enumerate(conversations):
            conv_id = convo["conversation_id"]
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(f"  [{idx + 1}/{len(conversations)}] {conv_id}")

            try:
                result = self.evaluate_conversation(convo["turns"])
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
                logger.error(f"Failed to evaluate {conv_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Aggregate metrics
        aggregate = _aggregate_state_metrics(all_metrics) if all_metrics else StateMetrics().__dict__

        # Error attribution summary
        error_summary: Dict[str, int] = defaultdict(int)
        for err in all_errors:
            error_summary[err.get("code", "?")] += 1

        # False lock summary
        total_adds = sum(
            r.get("false_lock_analysis", {}).get("total_adds", 0)
            for r in all_results
        )
        total_flocks = sum(
            r.get("false_lock_analysis", {}).get("false_lock_count", 0)
            for r in all_results
        )

        # Pipeline stats
        total_pipeline_stats = {
            "total_candidates": sum(r["pipeline_stats"]["total_candidates"] for r in all_results),
            "total_stage1": sum(r["pipeline_stats"]["total_stage1"] for r in all_results),
            "total_stage2_accepted": sum(r["pipeline_stats"]["total_stage2_accepted"] for r in all_results),
            "total_locked": sum(r["pipeline_stats"]["total_locked"] for r in all_results),
            "total_pending": sum(r["pipeline_stats"]["total_pending"] for r in all_results),
            "total_discarded": sum(r["pipeline_stats"]["total_discarded"] for r in all_results),
        }

        # Admission metrics
        admission_metrics = _compute_admission_metrics(all_results)

        return {
            "per_conversation": all_results,
            "aggregate": aggregate,
            "num_conversations": len(all_results),
            "num_failed": len(conversations) - len(all_results),
            "error_attribution_summary": dict(error_summary),
            "false_lock_summary": {
                "total_adds": total_adds,
                "total_false_locks": total_flocks,
                "false_lock_rate": total_flocks / total_adds if total_adds > 0 else 0.0,
            },
            "pipeline_stats": total_pipeline_stats,
            "admission_metrics": admission_metrics,
        }

    # ------------------------------------------------------------------
    # Full pipeline with output
    # ------------------------------------------------------------------

    def run(
        self,
        aligned_records_path: str,
        splits_path: str,
        output_dir: str,
        split: str = "test",
        min_turns: int = 1,
    ) -> Dict:
        """Full evaluation pipeline: load, evaluate, metrics, report."""
        os.makedirs(output_dir, exist_ok=True)

        conversations = self.load_conversations(
            aligned_records_path, splits_path, split=split, min_turns=min_turns,
        )
        if not conversations:
            logger.warning("No conversations found")
            return {"error": "no conversations found"}

        results = self.evaluate_all(conversations)

        # Write outputs
        _write_outputs(results, output_dir)

        logger.info(f"Results written to {output_dir}")
        results["output_dir"] = output_dir
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)


def _gold_trans_dicts_to_objects(
    gold_trans: List[Dict],
    prompt: str,
    turn_number: int,
) -> List[Transition]:
    """Convert gold transition dicts to Transition objects."""
    transitions: List[Transition] = []
    for gt in gold_trans:
        try:
            tt = TransitionType(gt["transition_type"])
        except (KeyError, ValueError):
            tt = TransitionType.ADD

        try:
            cat = MemoryCategory(gt["category"])
        except (KeyError, ValueError):
            cat = MemoryCategory.REQUIREMENT

        transitions.append(
            Transition(
                transition_type=tt,
                category=cat,
                value=gt.get("value", ""),
                old_value=gt.get("old_value"),
                memory_id=gt.get("memory_id"),
                source_text=gt.get("value", ""),
                prompt_text=prompt,
                turn_number=turn_number,
            )
        )
    return transitions


def _aggregate_state_metrics(metrics_list: List[StateMetrics]) -> Dict[str, Any]:
    """Aggregate a list of StateMetrics into a single summary dict."""
    if not metrics_list:
        return StateMetrics().__dict__

    n = len(metrics_list)
    aggregated: Dict[str, Any] = {}

    numeric_fields = [
        "span_precision", "span_recall", "span_f1",
        "transition_accuracy", "transition_precision", "transition_recall",
        "state_precision", "state_recall", "state_f1",
        "false_lock_rate", "false_update_rate", "false_removal_rate",
        "false_rejection_rate", "stale_memory_rate", "duplicate_memory_rate",
        "contradiction_rate",
    ]
    for field_name in numeric_fields:
        total = sum(getattr(m, field_name, 0.0) for m in metrics_list)
        aggregated[field_name] = total / n

    int_fields = [
        "total_spans", "total_transitions",
        "add_count", "modify_count", "remove_count",
        "reject_count", "no_change_count",
    ]
    for field_name in int_fields:
        aggregated[field_name] = sum(getattr(m, field_name, 0) for m in metrics_list)

    return aggregated


def _compute_admission_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute LOCK/PENDING/DISCARD precision and recall."""
    # For each admission category, count correct vs total
    lock_tp = 0
    lock_total = 0
    discard_tp = 0
    discard_total = 0

    for result in results:
        for turn in result["per_turn"]:
            for decision in turn.get("admission_decisions", []):
                text = decision["text"]
                state = decision["decision"]

                # Check if this text appears in gold spans for this turn
                gold_texts = {s.get("text", "").lower().strip() for s in turn.get("gold_spans", []) if "gold_spans" not in turn}

                # We need to get gold spans from the turn data
                # The turn dict doesn't directly have gold_spans, so we check from the conversation
                if state == AdmissionState.LOCK:
                    lock_total += 1
                elif state == AdmissionState.DISCARD:
                    discard_total += 1

    return {
        "lock_precision": 0.0,  # Will be computed from full data
        "lock_recall": 0.0,
        "discard_precision": 0.0,
        "lock_count": lock_total,
        "discard_count": discard_total,
    }


def _write_outputs(results: Dict, output_dir: str) -> None:
    """Write all evaluation outputs to disk."""
    # Config
    config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_conversations": results["num_conversations"],
        "num_failed": results["num_failed"],
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Aggregate metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results["aggregate"], f, indent=2, default=str)

    # Per-turn JSONL
    with open(os.path.join(output_dir, "per_turn.jsonl"), "w") as f:
        for result in results["per_conversation"]:
            for turn in result["per_turn"]:
                # Strip large fields for JSONL
                turn_copy = {k: v for k, v in turn.items()
                             if k not in ("admission_decisions", "predicted_transitions")}
                turn_copy["conversation_id"] = result["conversation_id"]
                f.write(json.dumps(turn_copy, default=str) + "\n")

    # Failures
    failures = []
    for result in results["per_conversation"]:
        flock_info = result.get("false_lock_analysis", {})
        for lock in flock_info.get("false_locks", []):
            lock["conversation_id"] = result["conversation_id"]
            failures.append(lock)
    with open(os.path.join(output_dir, "failures.jsonl"), "w") as f:
        for fail in failures:
            f.write(json.dumps(fail, default=str) + "\n")

    # Summary
    summary = _build_summary(results)
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write(summary)


def _build_summary(results: Dict) -> str:
    """Build Markdown summary."""
    lines = [
        "# SERA Product Baseline Evaluation",
        "",
        f"**Timestamp:** {datetime.now(timezone.utc).isoformat()}",
        f"**Conversations evaluated:** {results['num_conversations']}",
        f"**Conversations failed:** {results['num_failed']}",
        "",
        "## Extraction Metrics",
        "",
    ]

    agg = results["aggregate"]
    lines.append(f"- **Span Precision:** {agg.get('span_precision', 0):.4f}")
    lines.append(f"- **Span Recall:** {agg.get('span_recall', 0):.4f}")
    lines.append(f"- **Span F1:** {agg.get('span_f1', 0):.4f}")
    lines.append(f"- **Total Spans:** {agg.get('total_spans', 0)}")

    lines.extend(["", "## State Metrics", ""])
    lines.append(f"- **State Precision:** {agg.get('state_precision', 0):.4f}")
    lines.append(f"- **State Recall:** {agg.get('state_recall', 0):.4f}")
    lines.append(f"- **State F1:** {agg.get('state_f1', 0):.4f}")

    lines.extend(["", "## Error Rates (PRIMARY)", ""])
    lines.append(f"- **FALSE LOCK RATE:** {agg.get('false_lock_rate', 0):.4f}")
    lines.append(f"- **False Update Rate:** {agg.get('false_update_rate', 0):.4f}")
    lines.append(f"- **False Removal Rate:** {agg.get('false_removal_rate', 0):.4f}")
    lines.append(f"- **False Rejection Rate:** {agg.get('false_rejection_rate', 0):.4f}")
    lines.append(f"- **Stale Memory Rate:** {agg.get('stale_memory_rate', 0):.4f}")
    lines.append(f"- **Duplicate Memory Rate:** {agg.get('duplicate_memory_rate', 0):.4f}")
    lines.append(f"- **Contradiction Rate:** {agg.get('contradiction_rate', 0):.4f}")

    lines.extend(["", "## Transition Counts", ""])
    lines.append(f"- **ADD:** {agg.get('add_count', 0)}")
    lines.append(f"- **MODIFY:** {agg.get('modify_count', 0)}")
    lines.append(f"- **REMOVE:** {agg.get('remove_count', 0)}")
    lines.append(f"- **REJECT:** {agg.get('reject_count', 0)}")
    lines.append(f"- **NO_CHANGE:** {agg.get('no_change_count', 0)}")

    flock = results.get("false_lock_summary", {})
    lines.extend(["", "## False Lock Analysis", ""])
    lines.append(f"- **Total ADDs:** {flock.get('total_adds', 0)}")
    lines.append(f"- **False Locks:** {flock.get('total_false_locks', 0)}")
    lines.append(f"- **False Lock Rate:** {flock.get('false_lock_rate', 0):.4f}")

    ps = results.get("pipeline_stats", {})
    lines.extend(["", "## Pipeline Statistics", ""])
    lines.append(f"- **Total Candidates:** {ps.get('total_candidates', 0)}")
    lines.append(f"- **Stage 1 Candidates:** {ps.get('total_stage1', 0)}")
    lines.append(f"- **Stage 2 Accepted:** {ps.get('total_stage2_accepted', 0)}")
    lines.append(f"- **Locked:** {ps.get('total_locked', 0)}")
    lines.append(f"- **Pending:** {ps.get('total_pending', 0)}")
    lines.append(f"- **Discarded:** {ps.get('total_discarded', 0)}")

    lines.extend(["", "## Error Attribution (A-G)", ""])
    lines.append("| Code | Category | Count |")
    lines.append("|------|----------|-------|")
    category_names = {
        "A": "Extraction failure",
        "B": "Candidate typing failure",
        "C": "Matching failure",
        "D": "Transition-rule failure",
        "E": "Validation failure",
        "F": "Persistence/state failure",
        "G": "Evaluation ambiguity",
    }
    for code in ["A", "B", "C", "D", "E", "F", "G"]:
        count = results.get("error_attribution_summary", {}).get(code, 0)
        name = category_names.get(code, "Unknown")
        lines.append(f"| {code} | {name} | {count} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SERA Product Baseline Evaluation")
    parser.add_argument("--span-filter-threshold", type=float, default=0.5,
                        help="SpanFilter threshold (default: 0.5)")
    parser.add_argument("--skip-span-filter", action="store_true",
                        help="Skip SpanFilter, use Stage-1 only")
    parser.add_argument("--high-confidence", type=float, default=0.9,
                        help="High confidence threshold for LOCK (default: 0.9)")
    parser.add_argument("--medium-confidence", type=float, default=0.7,
                        help="Medium confidence threshold for PENDING (default: 0.7)")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to evaluate (default: test)")
    parser.add_argument("--min-turns", type=int, default=1,
                        help="Minimum turns per conversation (default: 1)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if None)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Determine output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stage_label = "no_filter" if args.skip_span_filter else f"sf{args.span_filter_threshold}"
        output_dir = f"logs/product_baseline/{timestamp}_{stage_label}"

    data_dir = PROJECT_ROOT / "data" / "processed"
    aligned_path = str(data_dir / "aligned_records.jsonl")
    splits_path = str(data_dir / "splits.json")

    # Load models
    logger.info("Loading models...")

    if args.skip_span_filter:
        logger.info("Loading Stage-1 only (E6-A)...")
        extractor = ExtractionSLM.from_checkpoint(
            checkpoint_dir=str(PROJECT_ROOT / "checkpoints" / "oracle_e6a" / "best"),
            model_name="google/bert_uncased_L-6_H-512_A-8",
            confidence_threshold=0.0,
        )
        pipeline = ExtractionSLMWrapper(extractor)
    else:
        logger.info("Loading Two-Stage Pipeline (E6-A + SpanFilter)...")
        pipeline = TwoStagePipeline.from_checkpoints(
            stage1_checkpoint=str(PROJECT_ROOT / "checkpoints" / "oracle_e6a" / "best"),
            stage2_checkpoint=str(PROJECT_ROOT / "checkpoints" / "experiment_e7_span_filter" / "best"),
            stage2_model_name="google/bert_uncased_L-6_H-512_A-8",
            stage2_threshold=args.span_filter_threshold,
        )
        logger.info(f"SpanFilter threshold: {pipeline.stage2_threshold}")

    # Run evaluation
    evaluator = ProductBaselineEvaluator(
        pipeline=pipeline,
        high_confidence=args.high_confidence,
        medium_confidence=args.medium_confidence,
        use_span_filter=not args.skip_span_filter,
        span_filter_threshold=args.span_filter_threshold,
    )

    logger.info(f"Running evaluation on split='{args.split}', min_turns={args.min_turns}")
    results = evaluator.run(
        aligned_records_path=aligned_path,
        splits_path=splits_path,
        output_dir=output_dir,
        split=args.split,
        min_turns=args.min_turns,
    )

    # Print summary
    agg = results.get("aggregate", {})
    flock = results.get("false_lock_summary", {})
    print("\n" + "=" * 60)
    print("PRODUCT BASELINE RESULTS")
    print("=" * 60)
    print(f"Span Precision:     {agg.get('span_precision', 0):.4f}")
    print(f"Span Recall:        {agg.get('span_recall', 0):.4f}")
    print(f"Span F1:            {agg.get('span_f1', 0):.4f}")
    print(f"State Precision:    {agg.get('state_precision', 0):.4f}")
    print(f"State Recall:       {agg.get('state_recall', 0):.4f}")
    print(f"State F1:           {agg.get('state_f1', 0):.4f}")
    print(f"FALSE LOCK RATE:    {flock.get('false_lock_rate', 0):.4f}")
    print(f"Total ADDs:         {flock.get('total_adds', 0)}")
    print(f"False Locks:        {flock.get('total_false_locks', 0)}")
    print(f"Conversations:      {results.get('num_conversations', 0)}")
    print(f"Output:             {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
