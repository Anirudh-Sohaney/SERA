"""
Evidence-based admission evaluation for SERA.

Runs Policies A-D on the same data and compares false lock rates.
Uses cached Stage 1 candidates + SpanFilter scores for efficiency.

Usage:
    python -m src.evaluation.admission_eval
    python -m src.evaluation.admission_eval --policies A,B,C,D
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
from typing import Any, Dict, List, Optional

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
from src.evaluation.product_baseline import _aggregate_state_metrics, _gold_trans_dicts_to_objects
from src.inference.extractor import ExtractionSLM
from src.memory.admission import (
    AdmissionDecision,
    AdmissionPolicy,
    EvidenceBasedAdmission,
    POLICY_A,
    POLICY_B,
    POLICY_C,
    POLICY_D,
)
from src.memory.engine import ProjectMemoryEngine
from src.memory.metrics import StateMetrics, compute_metrics
from src.memory.schema import MemoryCategory, ProjectState, Transition, TransitionType

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy evaluator
# ---------------------------------------------------------------------------

class AdmissionPolicyEvaluator:
    """Evaluates multiple admission policies on the same data."""

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
        self._candidate_cache: Dict[str, List[Dict]] = {}

    def _extract_and_score(self, prompt: str) -> List[Dict]:
        """Extract candidates and score with SpanFilter (cached)."""
        if prompt in self._candidate_cache:
            return self._candidate_cache[prompt]

        # Stage 1
        result = self._stage1.extract(prompt)
        candidates = result.spans if result.has_spans else []

        # Stage 2: score each candidate
        for cand in candidates:
            encoding = self._stage2_tokenizer(
                prompt, cand["text"],
                max_length=256, truncation=True, padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self._stage2_device)
            attention_mask = encoding["attention_mask"].to(self._stage2_device)
            with torch.no_grad():
                outputs = self._stage2(input_ids, attention_mask)
                cand["stage2_prob"] = outputs["probs"].item()

        self._candidate_cache[prompt] = candidates
        return candidates

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

    def evaluate_policy(
        self,
        conversations: List[Dict],
        policy: AdmissionPolicy,
        policy_name: str,
    ) -> Dict:
        """Evaluate a single admission policy across all conversations."""
        logger.info(f"  Evaluating policy: {policy_name}")
        admission = EvidenceBasedAdmission(policy)

        all_results = []
        all_metrics = []
        all_errors = []
        all_false_locks = []

        for idx, convo in enumerate(conversations):
            conv_id = convo["conversation_id"]
            if (idx + 1) % 200 == 0:
                logger.info(f"    [{idx + 1}/{len(conversations)}]")

            try:
                result = self._evaluate_conversation(
                    convo["turns"], admission, conv_id,
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

        # Admission statistics
        lock_count = sum(
            r["admission_stats"]["lock"] for r in all_results
        )
        pending_count = sum(
            r["admission_stats"]["pending"] for r in all_results
        )
        discard_count = sum(
            r["admission_stats"]["discard"] for r in all_results
        )

        return {
            "policy_name": policy_name,
            "aggregate": aggregate,
            "num_conversations": len(all_results),
            "num_failed": len(conversations) - len(all_results),
            "error_attribution_summary": dict(error_summary),
            "false_lock_summary": {
                "total_adds": total_adds,
                "total_false_locks": total_flocks,
                "false_lock_rate": total_flocks / total_adds if total_adds > 0 else 0.0,
            },
            "admission_stats": {
                "lock": lock_count,
                "pending": pending_count,
                "discard": discard_count,
                "total": lock_count + pending_count + discard_count,
            },
        }

    def _evaluate_conversation(
        self,
        turns: List[Dict],
        admission: EvidenceBasedAdmission,
        conv_id: str,
    ) -> Dict:
        """Evaluate one conversation with a given admission policy."""
        engine = ProjectMemoryEngine(
            project_id=f"adm_{conv_id}",
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

        lock_count = 0
        pending_count = 0
        discard_count = 0

        for turn_idx, turn_entry in enumerate(turns):
            record = turn_entry["record"]
            gold_spans = turn_entry.get("spans", [])
            turn_number = int(record.get("turn", 0)) + 1
            prompt = record.get("input", {}).get("user_prompt", "")

            # Get candidates
            candidates = self._extract_and_score(prompt)

            # Apply admission policy
            decisions = admission.decide_batch(
                candidates, prompt, engine.get_project_state(),
            )

            # Filter to LOCKED only
            locked_spans = []
            for decision in decisions:
                if decision.decision == AdmissionDecision.LOCK:
                    locked_spans.append({
                        "text": decision.text,
                        "confidence": decision.extractor_confidence,
                        "stage2_prob": decision.spanfilter_confidence,
                        "start": next((c.get("start", 0) for c in candidates if c.get("text") == decision.text), 0),
                        "end": next((c.get("end", 0) for c in candidates if c.get("text") == decision.text), 0),
                    })
                    lock_count += 1
                elif decision.decision == AdmissionDecision.PENDING:
                    pending_count += 1
                else:
                    discard_count += 1

            all_gold_spans.append(gold_spans)
            all_pred_spans.append(locked_spans)

            # Process through engine
            try:
                engine_result = engine.process_turn(
                    prompt=prompt,
                    extracted_spans=locked_spans,
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
                predicted_spans=locked_spans,
                gold_spans=gold_spans,
                predicted_transitions=pred_transitions,
                gold_transitions=gold_turn_trans,
            )

            per_turn.append({
                "turn": turn_number,
                "turn_idx": turn_idx,
                "prompt": prompt,
                "conversation_id": conv_id,
                "num_pred_spans": len(locked_spans),
                "num_gold_spans": len(gold_spans),
                "num_transitions": len(pred_transitions),
                "candidate_count": len(candidates),
                "locked_count": len(locked_spans),
                "pending_count": sum(1 for d in decisions if d.decision == AdmissionDecision.PENDING),
                "discarded_count": sum(1 for d in decisions if d.decision == AdmissionDecision.DISCARD),
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
            "admission_stats": {
                "lock": lock_count,
                "pending": pending_count,
                "discard": discard_count,
            },
        }

    def run_comparison(
        self,
        conversations: List[Dict],
        policies: Dict[str, AdmissionPolicy],
        output_dir: str,
    ) -> Dict:
        """Run all policies and compare."""
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        for name, policy in policies.items():
            start = time.time()
            result = self.evaluate_policy(conversations, policy, name)
            elapsed = time.time() - start
            result["elapsed_seconds"] = elapsed
            results[name] = result

            flock = result["false_lock_summary"]
            adm = result["admission_stats"]
            logger.info(
                f"  {name}: FLR={flock['false_lock_rate']:.4f}, "
                f"P={result['aggregate']['state_precision']:.4f}, "
                f"R={result['aggregate']['state_recall']:.4f}, "
                f"Lock={adm['lock']}, Pending={adm['pending']}, Discard={adm['discard']}, "
                f"time={elapsed:.1f}s"
            )

        # Save results
        comparison = []
        for name, r in results.items():
            comparison.append({
                "policy": name,
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
                "lock_count": r["admission_stats"]["lock"],
                "pending_count": r["admission_stats"]["pending"],
                "discard_count": r["admission_stats"]["discard"],
                "elapsed_seconds": r["elapsed_seconds"],
            })

        with open(os.path.join(output_dir, "policy_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)

        # Build summary
        summary = self._build_summary(comparison)
        with open(os.path.join(output_dir, "summary.md"), "w") as f:
            f.write(summary)

        logger.info(f"Results saved to {output_dir}")
        return results

    def _build_summary(self, comparison: List[Dict]) -> str:
        lines = [
            "# SERA Admission Policy Comparison",
            "",
            f"**Timestamp:** {datetime.now(timezone.utc).isoformat()}",
            f"**Policies tested:** {len(comparison)}",
            "",
            "## Results by Policy",
            "",
            "| Policy | FLR | State P | State R | State F1 | Span F1 | Locks | Pending | Discard | Time (s) |",
            "|--------|-----|---------|---------|----------|---------|-------|---------|---------|----------|",
        ]

        for d in comparison:
            lines.append(
                f"| {d['policy']} "
                f"| {d['false_lock_rate']:.4f} "
                f"| {d['state_precision']:.4f} "
                f"| {d['state_recall']:.4f} "
                f"| {d['state_f1']:.4f} "
                f"| {d['span_f1']:.4f} "
                f"| {d['lock_count']} "
                f"| {d['pending_count']} "
                f"| {d['discard_count']} "
                f"| {d['elapsed_seconds']:.1f} |"
            )

        # Find best
        if comparison:
            best = min(comparison, key=lambda x: x["false_lock_rate"])
            lines.extend([
                "",
                "## Best Policy (Lowest False Lock Rate)",
                "",
                f"**{best['policy']}**",
                f"- False Lock Rate: {best['false_lock_rate']:.4f}",
                f"- State Precision: {best['state_precision']:.4f}",
                f"- State Recall: {best['state_recall']:.4f}",
                f"- State F1: {best['state_f1']:.4f}",
            ])

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SERA Admission Policy Evaluation")
    parser.add_argument("--policies", type=str, default="A,B,C,D",
                        help="Comma-separated policy letters (default: A,B,C,D)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"logs/product_baseline/admission_eval_{timestamp}"

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

    # Select policies
    policy_map = {"A": POLICY_A, "B": POLICY_B, "C": POLICY_C, "D": POLICY_D}
    selected = [p.strip().upper() for p in args.policies.split(",")]
    policies = {f"Policy_{k}": policy_map[k] for k in selected if k in policy_map}

    # Load conversations
    logger.info("Loading conversations...")
    evaluator = AdmissionPolicyEvaluator(
        stage1_model=stage1,
        stage2_model=s2_model,
        stage2_tokenizer=s2_tokenizer,
        stage2_device=device,
    )
    conversations = evaluator.load_conversations(
        aligned_path, splits_path, split=args.split, min_turns=args.min_turns,
    )
    logger.info(f"Loaded {len(conversations)} conversations")

    # Run comparison
    logger.info(f"Running admission policy comparison: {list(policies.keys())}")
    results = evaluator.run_comparison(conversations, policies, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("ADMISSION POLICY COMPARISON")
    print("=" * 60)
    for name, r in results.items():
        flock = r["false_lock_summary"]
        print(f"{name}: FLR={flock['false_lock_rate']:.4f}, P={r['aggregate']['state_precision']:.4f}, R={r['aggregate']['state_recall']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
