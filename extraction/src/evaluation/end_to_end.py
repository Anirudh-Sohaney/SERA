"""
End-to-end evaluation harness for SERA.

Runs E6-A extractor + memory engine on real multi-turn conversations.
Measures extraction quality, transition quality, and final state quality.
Performs layer-by-layer error attribution.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.memory.engine import ProjectMemoryEngine
from src.memory.metrics import (
    StateMetrics,
    compare_states,
    compute_metrics,
    format_metrics,
)
from src.memory.schema import (
    MemoryCategory,
    MemoryItem,
    MemoryStatus,
    ProjectState,
    Transition,
    TransitionType,
)
from src.memory.transitions import (
    FIELD_CATEGORY_MAP,
    _category_from_field,
    _infer_category_from_text,
)
from src.memory.validator import _normalize_value

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Negation patterns for gold transition detection
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS = [
    r"\bnot?\s+use\b",
    r"\bdo\s+not\s+use\b",
    r"\bdon'?t\s+use\b",
    r"\bavoid\b",
    r"\binstead\s+of\b",
    r"\brather\s+than\b",
    r"\bno\s+longer\b",
    r"\bstop\s+using\b",
    r"\bremove\b",
    r"\bwithout\b",
    r"\binstead\b",
]


def _contains_negation(text: str) -> bool:
    """Check if text contains negation patterns."""
    lower = text.lower()
    for pattern in _NEGATION_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Gold state / transition builders
# ---------------------------------------------------------------------------

def _spans_to_memory_items(
    spans: List[Dict[str, Any]],
    prompt_text: str,
    turn_number: int,
) -> List[MemoryItem]:
    """Convert a list of span dicts into MemoryItems with resolved categories."""
    items: List[MemoryItem] = []
    for span in spans:
        text = str(span.get("text", "")).strip()
        if not text:
            continue

        # Resolve category (same order as build_memory_candidates)
        category: Optional[MemoryCategory] = None
        field_name = span.get("field")
        if field_name:
            category = _category_from_field(field_name)
        if category is None:
            raw_cat = span.get("category")
            if raw_cat:
                try:
                    category = MemoryCategory(raw_cat)
                except (ValueError, KeyError):
                    category = None
        if category is None:
            category = _infer_category_from_text(text)

        items.append(
            MemoryItem(
                category=category,
                value=text,
                source_text=text,
                source_start=int(span.get("start", 0)),
                source_end=int(span.get("end", len(prompt_text))),
                prompt_text=prompt_text,
                status=MemoryStatus.ACTIVE,
                created_turn=turn_number,
                updated_turn=turn_number,
                confidence=float(span.get("confidence", 1.0)),
            )
        )
    return items


def build_gold_transitions(conversation: List[Dict]) -> List[List[Dict]]:
    """Build expected transitions for each turn from gold spans.

    For each turn, determines what transition SHOULD have happened based on
    comparison against the accumulated gold state up to that point.

    Args:
        conversation: List of turn dicts with ``record``, ``spans`` keys.

    Returns:
        List of transition-lists, one per turn (same order as conversation).
    """
    gold_state = ProjectState(project_id="gold")
    all_turn_transitions: List[List[Dict]] = []

    for turn_entry in conversation:
        record = turn_entry["record"]
        spans = turn_entry.get("spans", [])
        # Engine requires turn_number >= 1; data is 0-indexed
        turn_number = int(record.get("turn", 0)) + 1
        prompt = record.get("input", {}).get("user_prompt", "")

        turn_items = _spans_to_memory_items(spans, prompt, turn_number)
        turn_transitions: List[Dict] = []

        for item in turn_items:
            # Check existing gold memories in same category
            existing = gold_state.get_active_by_category(item.category)
            matched = None
            for mem in existing:
                if mem.matches_value(item.value):
                    matched = mem
                    break

            if matched is not None:
                # Value already exists → NO_CHANGE
                turn_transitions.append({
                    "transition_type": "NO_CHANGE",
                    "category": item.category.value,
                    "value": item.value,
                    "memory_id": matched.memory_id,
                })
            else:
                # Check if any existing memory in same category has a different value
                if existing:
                    # Different value in same category → MODIFY (replace first match)
                    target = existing[0]
                    old_value = target.value
                    gold_state.modify_memory(
                        memory_id=target.memory_id,
                        new_value=item.value,
                        new_source=item.source_text,
                        new_source_start=item.source_start,
                        new_source_end=item.source_end,
                        new_prompt=item.prompt_text,
                        turn=turn_number,
                    )
                    turn_transitions.append({
                        "transition_type": "MODIFY",
                        "category": item.category.value,
                        "value": item.value,
                        "old_value": old_value,
                        "memory_id": target.memory_id,
                    })
                else:
                    # New value, no existing memory → ADD
                    gold_state.add_memory(item)
                    turn_transitions.append({
                        "transition_type": "ADD",
                        "category": item.category.value,
                        "value": item.value,
                        "memory_id": item.memory_id,
                    })

        all_turn_transitions.append(turn_transitions)

    return all_turn_transitions


def build_gold_state(conversation: List[Dict]) -> List[Dict]:
    """Build expected ProjectState after each turn from gold spans.

    Accumulates state across turns: turn 1 adds Python, turn 2 adds Flask,
    so gold state after turn 2 has both.

    Args:
        conversation: List of turn dicts with ``record``, ``spans`` keys.

    Returns:
        List of ProjectState dicts, one per turn.
    """
    gold_state = ProjectState(project_id="gold")
    state_snapshots: List[Dict] = []

    for turn_entry in conversation:
        record = turn_entry["record"]
        spans = turn_entry.get("spans", [])
        # Engine requires turn_number >= 1; data is 0-indexed
        turn_number = int(record.get("turn", 0)) + 1
        prompt = record.get("input", {}).get("user_prompt", "")

        turn_items = _spans_to_memory_items(spans, prompt, turn_number)

        for item in turn_items:
            existing = gold_state.get_active_by_category(item.category)
            matched = any(mem.matches_value(item.value) for mem in existing)

            if not matched:
                if existing:
                    # MODIFY: replace existing value in same category
                    gold_state.modify_memory(
                        memory_id=existing[0].memory_id,
                        new_value=item.value,
                        new_source=item.source_text,
                        new_source_start=item.source_start,
                        new_source_end=item.source_end,
                        new_prompt=item.prompt_text,
                        turn=turn_number,
                    )
                else:
                    # ADD: new category
                    gold_state.add_memory(item)

        state_snapshots.append(gold_state.to_dict())

    return state_snapshots


# ---------------------------------------------------------------------------
# Error attribution
# ---------------------------------------------------------------------------

class ErrorCategory:
    """Error classification categories (A-G)."""

    EXTRACTION = "A"        # Extraction failure: wrong span extracted
    CANDIDATE_TYPING = "B"  # Candidate typing failure: right span, wrong category
    MATCHING = "C"          # Matching failure: right category, wrong match
    TRANSITION_RULE = "D"   # Transition-rule failure: right match, wrong type
    VALIDATION = "E"        # Validation failure
    PERSISTENCE = "F"       # Persistence/state failure
    EVAL_AMBIGUITY = "G"    # Evaluation ambiguity


def error_attribution(
    predicted_memory: ProjectState,
    gold_memory: ProjectState,
    predicted_transitions: List[Transition],
    gold_transitions: List[List[Dict]],
    predicted_spans: List[List[Dict]],
    gold_spans: List[List[Dict]],
) -> List[Dict]:
    """Classify each error into categories A-G.

    Args:
        predicted_memory:      Final predicted ProjectState.
        gold_memory:           Final gold ProjectState.
        predicted_transitions: All predicted transitions across turns.
        gold_transitions:      Gold transitions per turn.
        predicted_spans:       Predicted spans per turn.
        gold_spans:            Gold spans per turn.

    Returns:
        List of error dicts with ``category``, ``description``, ``turn``,
        ``predicted``, ``expected`` keys.
    """
    errors: List[Dict[str, Any]] = []

    # --- A: Extraction failures (false positives / false negatives) ---
    for turn_idx, (pred_spans, gold_spans_turn) in enumerate(
        zip(predicted_spans, gold_spans)
    ):
        pred_texts = {s.get("text", "").lower().strip() for s in pred_spans}
        gold_texts = {s.get("text", "").lower().strip() for s in gold_spans_turn}

        for gt in gold_texts:
            if gt and gt not in pred_texts:
                errors.append({
                    "category": ErrorCategory.EXTRACTION,
                    "code": "A",
                    "description": f"Missed span: '{gt}'",
                    "turn": turn_idx,
                    "predicted": None,
                    "expected": gt,
                })

        for pt in pred_texts:
            if pt and pt not in gold_texts:
                errors.append({
                    "category": ErrorCategory.EXTRACTION,
                    "code": "A",
                    "description": f"Spurious span: '{pt}'",
                    "turn": turn_idx,
                    "predicted": pt,
                    "expected": None,
                })

    # --- B: Candidate typing failures ---
    pred_key_set = {
        (_normalize_value(m.value), m.category.value)
        for m in predicted_memory.active_memories
    }
    gold_key_set = {
        (_normalize_value(m.value), m.category.value)
        for m in gold_memory.active_memories
    }

    # Find spans that exist in both but with different categories
    pred_val_cat = {m.value.lower().strip(): m.category.value for m in predicted_memory.active_memories}
    gold_val_cat = {m.value.lower().strip(): m.category.value for m in gold_memory.active_memories}

    for val_lower in pred_val_cat:
        if val_lower in gold_val_cat:
            if pred_val_cat[val_lower] != gold_val_cat[val_lower]:
                errors.append({
                    "category": ErrorCategory.CANDIDATE_TYPING,
                    "code": "B",
                    "description": (
                        f"Category mismatch for '{val_lower}': "
                        f"predicted={pred_val_cat[val_lower]}, "
                        f"expected={gold_val_cat[val_lower]}"
                    ),
                    "turn": -1,
                    "predicted": pred_val_cat[val_lower],
                    "expected": gold_val_cat[val_lower],
                })

    # --- C: Matching failures ---
    # Predicted has the right value+category but wrong memory_id association
    # (detected by looking at transition mismatches)
    pred_trans_by_key: Dict[Tuple[str, str], Transition] = {}
    for t in predicted_transitions:
        key = (_normalize_value(t.value), t.category.value)
        if key not in pred_trans_by_key:
            pred_trans_by_key[key] = t

    # --- D: Transition-rule failures ---
    for turn_idx, gold_turn in enumerate(gold_transitions):
        for gt in gold_turn:
            g_key = (_normalize_value(gt["value"]), gt["category"])
            g_type = gt["transition_type"]

            # Find matching predicted transition
            if g_key in pred_trans_by_key:
                pred_t = pred_trans_by_key[g_key]
                if pred_t.transition_type.value != g_type:
                    errors.append({
                        "category": ErrorCategory.TRANSITION_RULE,
                        "code": "D",
                        "description": (
                            f"Transition type mismatch for '{gt['value']}': "
                            f"predicted={pred_t.transition_type.value}, "
                            f"expected={g_type}"
                        ),
                        "turn": turn_idx,
                        "predicted": pred_t.transition_type.value,
                        "expected": g_type,
                    })
            else:
                # Expected transition was not produced at all
                errors.append({
                    "category": ErrorCategory.TRANSITION_RULE,
                    "code": "D",
                    "description": f"Missing expected transition: {g_type}({gt['value']})",
                    "turn": turn_idx,
                    "predicted": None,
                    "expected": g_type,
                })

    # --- E: Validation failures (structural issues) ---
    # These are detected by the engine's validator; surface them
    # Check for obvious structural issues
    pred_active_vals = [m.value for m in predicted_memory.active_memories]
    gold_active_vals = [m.value for m in gold_memory.active_memories]

    if len(pred_active_vals) != len(set(pred_active_vals)):
        errors.append({
            "category": ErrorCategory.VALIDATION,
            "code": "E",
            "description": "Duplicate active memories in predicted state",
            "turn": -1,
            "predicted": f"{len(pred_active_vals)} memories",
            "expected": f"{len(set(pred_active_vals))} unique",
        })

    # --- F: Persistence/state failures ---
    # State has memories that should have been removed, or is missing ones
    for val_lower in pred_val_cat:
        if val_lower not in gold_val_cat:
            errors.append({
                "category": ErrorCategory.PERSISTENCE,
                "code": "F",
                "description": f"Extra memory in state: '{val_lower}' (category={pred_val_cat[val_lower]})",
                "turn": -1,
                "predicted": pred_val_cat[val_lower],
                "expected": None,
            })

    for val_lower in gold_val_cat:
        if val_lower not in pred_val_cat:
            errors.append({
                "category": ErrorCategory.PERSISTENCE,
                "code": "F",
                "description": f"Missing memory from state: '{val_lower}' (category={gold_val_cat[val_lower]})",
                "turn": -1,
                "predicted": None,
                "expected": gold_val_cat[val_lower],
            })

    return errors


# ---------------------------------------------------------------------------
# False lock analysis
# ---------------------------------------------------------------------------

def false_lock_analysis(
    results: List[Dict[str, Any]],
    gold_transitions: List[List[Dict]],
) -> Dict[str, Any]:
    """Analyze every false lock in detail.

    A false lock occurs when an incorrect memory was added to persistent
    state (i.e., a predicted ADD that should not have been an ADD, or
    an ADD with wrong value/category).

    Args:
        results:           Per-turn evaluation results.
        gold_transitions:  Gold transitions per turn.

    Returns:
        Dict with ``false_locks`` list, ``total_adds``, ``false_lock_count``,
        ``false_lock_rate``, and ``details`` per lock.
    """
    false_locks: List[Dict[str, Any]] = []
    total_adds = 0

    for turn_idx, turn_result in enumerate(results):
        pred_transitions = turn_result.get("predicted_transitions", [])
        gold_turn = gold_transitions[turn_idx] if turn_idx < len(gold_transitions) else []

        gold_adds = {
            (_normalize_value(g["value"]), g["category"])
            for g in gold_turn
            if g["transition_type"] == "ADD"
        }

        for t in pred_transitions:
            if t.transition_type == TransitionType.ADD:
                total_adds += 1
                t_key = (_normalize_value(t.value), t.category.value)

                if t_key not in gold_adds:
                    # This is a false lock
                    detail = {
                        "turn": turn_idx,
                        "value": t.value,
                        "category": t.category.value,
                        "confidence": t.confidence,
                        "source_text": t.source_text,
                        "prompt_text": t.prompt_text[:200] if t.prompt_text else "",
                    }

                    # Check if this value exists in gold with different category
                    gold_val_cats = {
                        g["category"]: g
                        for g in gold_turn
                        if _normalize_value(g["value"]) == _normalize_value(t.value)
                    }
                    if gold_val_cats:
                        detail["possible_category_mismatch"] = True
                        detail["expected_categories"] = list(gold_val_cats.keys())

                    # Check if this was supposed to be a MODIFY
                    gold_modify = [
                        g for g in gold_turn
                        if g["transition_type"] == "MODIFY"
                        and _normalize_value(g["value"]) == _normalize_value(t.value)
                    ]
                    if gold_modify:
                        detail["should_be_modify"] = True

                    false_locks.append(detail)

    false_lock_count = len(false_locks)
    false_lock_rate = false_lock_count / total_adds if total_adds > 0 else 0.0

    return {
        "false_locks": false_locks,
        "total_adds": total_adds,
        "false_lock_count": false_lock_count,
        "false_lock_rate": false_lock_rate,
    }


# ---------------------------------------------------------------------------
# End-to-end evaluator
# ---------------------------------------------------------------------------

class EndToEndEvaluator:
    """Evaluates SERA extraction + memory engine on real conversations.

    Runs the full pipeline: load data → extract spans → build candidates →
    process through memory engine → compare with gold → compute metrics.

    Args:
        extractor:     Callable ``extract(prompt) -> List[Dict]``.
        engine_class:  Class to instantiate for the memory engine.
    """

    def __init__(
        self,
        extractor: Callable[[str], List[Dict[str, Any]]],
        engine_class: type = ProjectMemoryEngine,
    ) -> None:
        """Initialize with loaded extractor."""
        self._extractor = extractor
        self._engine_class = engine_class

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_conversations(
        self,
        aligned_records_path: str,
        splits_path: str,
        split: str = "test",
        min_turns: int = 2,
    ) -> List[Dict]:
        """Load and group conversations from aligned records.

        Reads the JSONL file, filters by split indices, groups by
        conversation_id, sorts by turn, and filters to conversations
        with at least ``min_turns`` turns.

        Args:
        aligned_records_path: Path to aligned_records.jsonl.
        splits_path:          Path to splits.json.
        split:                Which split to load (``"train"``, ``"test"``, ``"validation"``).
        min_turns:            Minimum number of turns required.

        Returns:
        List of conversation dicts, each with ``conversation_id`` and
        ``turns`` (sorted list of turn records).
        """
        logger.info(f"Loading conversations from {aligned_records_path}")

        with open(splits_path, "r") as f:
            splits = json.load(f)

        split_indices = set(splits.get(split, []))
        if not split_indices:
            logger.warning(f"No indices found for split '{split}'")
            return []

        # Read and filter records
        convos: Dict[str, List[Dict]] = defaultdict(list)
        total_lines = 0
        with open(aligned_records_path, "r") as f:
            for i, line in enumerate(f):
                total_lines += 1
                if i in split_indices:
                    try:
                        record = json.loads(line)
                        conv_id = record["record"]["conversation_id"]
                        convos[conv_id].append(record)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping malformed line {i}: {e}")

        logger.info(
            f"Loaded {len(convos)} conversations from {total_lines} records "
            f"(split='{split}')"
        )

        # Filter to multi-turn and sort
        multi_turn = []
        for conv_id, turns in convos.items():
            sorted_turns = sorted(turns, key=lambda r: r["record"]["turn"])
            if len(sorted_turns) >= min_turns:
                multi_turn.append({
                    "conversation_id": conv_id,
                    "turns": sorted_turns,
                })

        logger.info(
            f"Filtered to {len(multi_turn)} conversations with {min_turns}+ turns"
        )
        return multi_turn

    # ------------------------------------------------------------------
    # Gold state builder
    # ------------------------------------------------------------------

    def build_gold_state(self, conversation: List[Dict]) -> List[Dict]:
        """Build expected state after each turn from gold spans.

        Accumulates across turns: turn 1 adds Python, turn 2 adds Flask,
        gold state after turn 2 has both.

        Args:
            conversation: List of turn dicts (``record``, ``spans``).

        Returns:
            List of ProjectState dicts, one per turn.
        """
        return build_gold_state(conversation)

    # ------------------------------------------------------------------
    # Single conversation evaluation
    # ------------------------------------------------------------------

    def evaluate_conversation(self, conversation: List[Dict]) -> Dict:
        """Run full pipeline on one conversation, return per-turn results.

        Steps:
        1. Build gold state and gold transitions.
        2. For each turn, run extractor, build candidates, process through
           engine, compare with gold.
        3. Compute per-turn and cumulative metrics.

        Args:
            conversation: List of turn dicts (``record``, ``spans``).

        Returns:
            Dict with ``per_turn``, ``gold_transitions``, ``gold_states``,
            ``predicted_transitions``, ``final_state``, ``aggregate_metrics``.
        """
        conv_id = conversation[0]["record"]["conversation_id"]
        engine = self._engine_class(project_id=f"eval_{conv_id}")

        # Build gold
        gold_states = self.build_gold_state(conversation)
        gold_trans = build_gold_transitions(conversation)

        # Accumulate gold spans and predicted spans across turns
        all_gold_spans: List[List[Dict]] = []
        all_pred_spans: List[List[Dict]] = []
        all_pred_transitions: List[Transition] = []
        all_gold_transitions_flat: List[Transition] = []
        per_turn: List[Dict[str, Any]] = []

        for turn_idx, turn_entry in enumerate(conversation):
            record = turn_entry["record"]
            spans = turn_entry.get("spans", [])
            raw_turn = int(record.get("turn", 0))
            # Engine requires turn_number >= 1; data is 0-indexed
            turn_number = raw_turn + 1
            prompt = record.get("input", {}).get("user_prompt", "")

            # Run extractor
            try:
                result = self._extractor.extract(prompt)
                extracted_spans = result.spans if result.has_spans else []
            except Exception as e:
                logger.error(f"Extractor failed on turn {turn_number}: {e}")
                extracted_spans = []

            all_gold_spans.append(spans)
            all_pred_spans.append(extracted_spans)

            # Process through memory engine
            try:
                result = engine.process_turn(
                    prompt=prompt,
                    extracted_spans=extracted_spans,
                    turn_number=turn_number,
                )
            except Exception as e:
                logger.error(f"Engine failed on turn {turn_number}: {e}")
                result = {
                    "transitions": [],
                    "state_snapshot": {"active_memories": [], "all_memories_count": 0, "current_turn": turn_number},
                    "audit_records": [],
                    "validation_result": type("R", (), {"valid": False, "errors": [], "warnings": []})(),
                }

            pred_transitions = result["transitions"]
            all_pred_transitions.extend(pred_transitions)

            # Build gold transitions as Transition objects for this turn
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
                gold_spans=spans,
                predicted_transitions=pred_transitions,
                gold_transitions=gold_turn_trans,
            )

            per_turn.append({
                "turn": turn_number,
                "turn_idx": turn_idx,
                "prompt": prompt,
                "num_pred_spans": len(extracted_spans),
                "num_gold_spans": len(spans),
                "num_transitions": len(pred_transitions),
                "num_gold_transitions": len(gold_turn_trans),
                "transition_types": {
                    t.transition_type.value: sum(
                        1 for tt in pred_transitions
                        if tt.transition_type == t.transition_type
                    )
                    for t in pred_transitions
                } if pred_transitions else {},
                "metrics": turn_metrics.__dict__,
                "predicted_transitions": pred_transitions,
                "validation_valid": getattr(result.get("validation_result"), "valid", True),
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
        }

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self, conversations: List[Dict]) -> Dict:
        """Evaluate all conversations and compute aggregate metrics.

        Args:
            conversations: List of conversation dicts (each with ``turns``).

        Returns:
            Dict with ``per_conversation``, ``aggregate``, ``summary``,
            ``error_attribution_summary``, ``false_lock_summary``.
        """
        logger.info(f"Evaluating {len(conversations)} conversations")

        all_results: List[Dict[str, Any]] = []
        all_metrics: List[StateMetrics] = []
        all_errors: List[Dict[str, Any]] = []
        all_false_locks: List[Dict[str, Any]] = []

        for idx, convo in enumerate(conversations):
            conv_id = convo["conversation_id"]
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(f"  Evaluating conversation {idx + 1}/{len(conversations)}: {conv_id}")

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
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        aligned_records_path: str,
        splits_path: str,
        output_dir: str,
        split: str = "test",
        min_turns: int = 2,
    ) -> Dict:
        """Full evaluation pipeline: load, evaluate, metrics, report.

        Args:
            aligned_records_path: Path to aligned_records.jsonl.
            splits_path:          Path to splits.json.
            output_dir:           Directory to write results.
            split:                Data split to evaluate on.
            min_turns:            Minimum turns per conversation.

        Returns:
            Dict with aggregate metrics and output paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load
        conversations = self.load_conversations(
            aligned_records_path, splits_path, split=split, min_turns=min_turns,
        )
        if not conversations:
            logger.warning("No conversations found; aborting evaluation")
            return {"error": "no conversations found"}

        # Evaluate
        results = self.evaluate_all(conversations)

        # Write outputs
        aggregate_path = os.path.join(output_dir, "aggregate_metrics.json")
        with open(aggregate_path, "w") as f:
            json.dump(results["aggregate"], f, indent=2, default=str)

        # Write per-conversation summaries (not full results to save space)
        per_conv_path = os.path.join(output_dir, "per_conversation.json")
        per_conv_summaries = []
        for r in results["per_conversation"]:
            per_conv_summaries.append({
                "conversation_id": r["conversation_id"],
                "num_turns": r["num_turns"],
                "aggregate_metrics": r["aggregate_metrics"],
                "error_count": len(r.get("error_attribution", [])),
                "false_lock_count": r.get("false_lock_analysis", {}).get("false_lock_count", 0),
            })
        with open(per_conv_path, "w") as f:
            json.dump(per_conv_summaries, f, indent=2, default=str)

        # Write error attribution
        attribution_path = os.path.join(output_dir, "error_attribution.json")
        all_attribution = []
        for r in results["per_conversation"]:
            all_attribution.extend(r.get("error_attribution", []))
        with open(attribution_path, "w") as f:
            json.dump(all_attribution, f, indent=2, default=str)

        # Write false locks
        flock_path = os.path.join(output_dir, "false_locks.json")
        all_flocks = []
        for r in results["per_conversation"]:
            flock_info = r.get("false_lock_analysis", {})
            for lock in flock_info.get("false_locks", []):
                lock["conversation_id"] = r["conversation_id"]
                all_flocks.append(lock)
        with open(flock_path, "w") as f:
            json.dump(all_flocks, f, indent=2, default=str)

        # Write summary report
        summary_md = self._build_summary(results)
        summary_path = os.path.join(output_dir, "summary.md")
        with open(summary_path, "w") as f:
            f.write(summary_md)

        logger.info(f"Results written to {output_dir}")
        results["output_dir"] = output_dir
        return results

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    def _build_summary(self, results: Dict) -> str:
        """Build a Markdown summary of the evaluation."""
        lines = [
            "# SERA End-to-End Evaluation Summary",
            "",
            f"**Conversations evaluated:** {results['num_conversations']}",
            f"**Conversations failed:** {results['num_failed']}",
            "",
            "## Aggregate Metrics",
            "",
        ]

        agg = results["aggregate"]
        if isinstance(agg, dict):
            lines.append(f"- **Span Precision:** {agg.get('span_precision', 0):.4f}")
            lines.append(f"- **Span Recall:** {agg.get('span_recall', 0):.4f}")
            lines.append(f"- **Span F1:** {agg.get('span_f1', 0):.4f}")
            lines.append(f"- **State Precision:** {agg.get('state_precision', 0):.4f}")
            lines.append(f"- **State Recall:** {agg.get('state_recall', 0):.4f}")
            lines.append(f"- **State F1:** {agg.get('state_f1', 0):.4f}")
            lines.append(f"- **False Lock Rate:** {agg.get('false_lock_rate', 0):.4f}")
            lines.append(f"- **Total Spans:** {agg.get('total_spans', 0)}")
            lines.append(f"- **Total Transitions:** {agg.get('total_transitions', 0)}")

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

        flock = results.get("false_lock_summary", {})
        lines.extend([
            "",
            "## False Lock Analysis",
            "",
            f"- **Total ADDs:** {flock.get('total_adds', 0)}",
            f"- **False Locks:** {flock.get('total_false_locks', 0)}",
            f"- **False Lock Rate:** {flock.get('false_lock_rate', 0):.4f}",
        ])

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
