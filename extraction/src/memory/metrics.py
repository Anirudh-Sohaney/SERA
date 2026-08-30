"""
State-engine metrics.

Measures extraction quality, transition quality, and final state quality
separately so that each layer of the pipeline can be evaluated independently.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.memory.schema import (
    MemoryCategory,
    MemoryItem,
    MemoryStatus,
    ProjectState,
    Transition,
    TransitionType,
)
from src.memory.validator import _normalize_value


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class StateMetrics:
    """Comprehensive metrics comparing predicted state against expected.

    Attributes:
        span_*:            Extraction-level metrics (predicted spans vs gold).
        transition_*:      Transition-level metrics (predicted transitions vs gold).
        state_*:           State-level metrics (final predicted vs expected).
        false_lock_rate:   (false adds) / (total adds).
        false_update_rate: (incorrect modifies) / (total modifies).
        false_removal_rate:(incorrect removes) / (total removes).
        false_rejection_rate: (incorrect rejects) / (total rejects).
        stale_memory_rate: Active memories that should be removed or rejected.
        duplicate_memory_rate: Active memories with duplicate category+value.
        contradiction_rate: Transitions that contradict each other.
        total_spans:       Number of spans extracted.
        total_transitions:  Number of transitions produced.
        add_count / modify_count / remove_count / reject_count / no_change_count:
            Breakdown by transition type.
    """

    # Extraction metrics
    span_precision: float = 0.0
    span_recall: float = 0.0
    span_f1: float = 0.0

    # Transition metrics
    transition_accuracy: float = 0.0
    transition_precision: float = 0.0
    transition_recall: float = 0.0

    # State metrics
    state_precision: float = 0.0
    state_recall: float = 0.0
    state_f1: float = 0.0

    # Error rates
    false_lock_rate: float = 0.0
    false_update_rate: float = 0.0
    false_removal_rate: float = 0.0
    false_rejection_rate: float = 0.0
    stale_memory_rate: float = 0.0
    duplicate_memory_rate: float = 0.0
    contradiction_rate: float = 0.0

    # Counts
    total_spans: int = 0
    total_transitions: int = 0
    add_count: int = 0
    modify_count: int = 0
    remove_count: int = 0
    reject_count: int = 0
    no_change_count: int = 0


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _memory_key(item: MemoryItem) -> Tuple[MemoryCategory, str]:
    """Canonical key for a memory (category + normalized value)."""
    return (item.category, _normalize_value(item.value))


def _transition_key(t: Transition) -> Tuple[TransitionType, MemoryCategory, str]:
    """Canonical key for a transition (type + category + normalized value)."""
    return (t.transition_type, t.category, _normalize_value(t.value))


def _transition_match_key(t: Transition) -> Tuple[MemoryCategory, str]:
    """Key for matching transitions that affect the same logical memory."""
    return (t.category, _normalize_value(t.value))


def compute_metrics(
    predicted_state: ProjectState,
    expected_state: ProjectState,
    predicted_spans: Optional[List[Dict[str, Any]]] = None,
    gold_spans: Optional[List[Dict[str, Any]]] = None,
    predicted_transitions: Optional[List[Transition]] = None,
    gold_transitions: Optional[List[Transition]] = None,
) -> StateMetrics:
    """Compute all metrics by comparing predicted against expected.

    At minimum, ``predicted_state`` and ``expected_state`` must be provided.
    Optional inputs enable finer-grained span-level and transition-level
    metrics.

    Args:
        predicted_state:        The engine's final state.
        expected_state:         Ground-truth state.
        predicted_spans:        Raw spans from the extractor (optional).
        gold_spans:             Ground-truth spans (optional).
        predicted_transitions:  Transitions produced by the engine (optional).
        gold_transitions:       Ground-truth transitions (optional).

    Returns:
        A fully populated :class:`StateMetrics`.
    """
    metrics = StateMetrics()

    # --- Span-level metrics ------------------------------------------------
    if predicted_spans is not None and gold_spans is not None:
        pred_span_set = _span_set(predicted_spans)
        gold_span_set = _span_set(gold_spans)

        tp = len(pred_span_set & gold_span_set)
        fp = len(pred_span_set - gold_span_set)
        fn = len(gold_span_set - pred_span_set)

        metrics.span_precision = _safe_div(tp, tp + fp)
        metrics.span_recall = _safe_div(tp, tp + fn)
        metrics.span_f1 = _harmonic(metrics.span_precision, metrics.span_recall)
        metrics.total_spans = len(predicted_spans)

    # --- Transition-level metrics -------------------------------------------
    pred_trans = predicted_transitions or []
    gold_trans = gold_transitions or []

    if pred_trans and gold_trans:
        pred_keys = {_transition_key(t) for t in pred_trans}
        gold_keys = {_transition_key(t) for t in gold_trans}

        tp_t = len(pred_keys & gold_keys)
        fp_t = len(pred_keys - gold_keys)
        fn_t = len(gold_keys - pred_keys)

        metrics.transition_precision = _safe_div(tp_t, tp_t + fp_t)
        metrics.transition_recall = _safe_div(tp_t, tp_t + fn_t)
        metrics.transition_f1 = _harmonic(metrics.transition_precision, metrics.transition_recall)
        metrics.transition_accuracy = _safe_div(tp_t, len(gold_keys))

    # --- Count transitions by type -----------------------------------------
    for t in pred_trans:
        metrics.total_transitions += 1
        if t.transition_type == TransitionType.ADD:
            metrics.add_count += 1
        elif t.transition_type == TransitionType.MODIFY:
            metrics.modify_count += 1
        elif t.transition_type == TransitionType.REMOVE:
            metrics.remove_count += 1
        elif t.transition_type == TransitionType.REJECT:
            metrics.reject_count += 1
        elif t.transition_type == TransitionType.NO_CHANGE:
            metrics.no_change_count += 1

    # --- State-level metrics -----------------------------------------------
    pred_keys_s = {_memory_key(m) for m in predicted_state.active_memories}
    gold_keys_s = {_memory_key(m) for m in expected_state.active_memories}

    tp_s = len(pred_keys_s & gold_keys_s)
    fp_s = len(pred_keys_s - gold_keys_s)
    fn_s = len(gold_keys_s - pred_keys_s)

    metrics.state_precision = _safe_div(tp_s, tp_s + fp_s)
    metrics.state_recall = _safe_div(tp_s, tp_s + fn_s)
    metrics.state_f1 = _harmonic(metrics.state_precision, metrics.state_recall)

    # --- Error rates -------------------------------------------------------
    metrics.false_lock_rate = _compute_false_lock_rate(pred_trans, gold_trans)
    metrics.false_update_rate = _compute_false_update_rate(pred_trans, gold_trans)
    metrics.false_removal_rate = _compute_false_removal_rate(pred_trans, gold_trans)
    metrics.false_rejection_rate = _compute_false_rejection_rate(pred_trans, gold_trans)
    metrics.stale_memory_rate = _compute_stale_memory_rate(predicted_state, expected_state)
    metrics.duplicate_memory_rate = _compute_duplicate_rate(predicted_state)
    metrics.contradiction_rate = _compute_contradiction_rate(pred_trans)

    return metrics


# ---------------------------------------------------------------------------
# Detailed state comparison
# ---------------------------------------------------------------------------

def compare_states(
    predicted: ProjectState,
    expected: ProjectState,
) -> Dict[str, Any]:
    """Detailed comparison showing matching, missing, and extra memories.

    Args:
        predicted: The engine's final state.
        expected:  Ground-truth state.

    Returns:
        Dict with ``matching``, ``missing``, ``extra``, ``mismatched_values``,
        and ``mismatched_status`` lists.
    """
    pred_map: Dict[Tuple[MemoryCategory, str], MemoryItem] = {}
    for m in predicted.active_memories:
        key = (m.category, _normalize_value(m.value))
        pred_map[key] = m

    exp_map: Dict[Tuple[MemoryCategory, str], MemoryItem] = {}
    for m in expected.active_memories:
        key = (m.category, _normalize_value(m.value))
        exp_map[key] = m

    matching: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    extra: List[Dict[str, Any]] = []

    matched_pred_keys: Set[Tuple[MemoryCategory, str]] = set()

    for key, exp_item in exp_map.items():
        if key in pred_map:
            matched_pred_keys.add(key)
            pred_item = pred_map[key]
            if pred_item.status == exp_item.status:
                matching.append(
                    {
                        "category": key[0].value,
                        "value": exp_item.value,
                        "predicted_status": pred_item.status.value,
                        "expected_status": exp_item.status.value,
                    }
                )
            else:
                matching.append(
                    {
                        "category": key[0].value,
                        "value": exp_item.value,
                        "predicted_status": pred_item.status.value,
                        "expected_status": exp_item.status.value,
                        "status_mismatch": True,
                    }
                )
        else:
            missing.append(
                {
                    "category": key[0].value,
                    "value": exp_item.value,
                    "status": exp_item.status.value,
                }
            )

    for key, pred_item in pred_map.items():
        if key not in exp_map:
            extra.append(
                {
                    "category": key[0].value,
                    "value": pred_item.value,
                    "status": pred_item.status.value,
                }
            )

    return {
        "matching": matching,
        "missing": missing,
        "extra": extra,
        "matching_count": len(matching),
        "missing_count": len(missing),
        "extra_count": len(extra),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_metrics(metrics: StateMetrics) -> str:
    """Human-readable metrics report.

    Args:
        metrics: The metrics to format.

    Returns:
        Multi-line string with all metric values.
    """
    lines = [
        "=== SERA State-Engine Metrics ===",
        "",
        "--- Extraction (Span) ---",
        f"  Precision:  {metrics.span_precision:.4f}",
        f"  Recall:     {metrics.span_recall:.4f}",
        f"  F1:         {metrics.span_f1:.4f}",
        f"  Total:      {metrics.total_spans}",
        "",
        "--- Transition ---",
        f"  Accuracy:   {metrics.transition_accuracy:.4f}",
        f"  Precision:  {metrics.transition_precision:.4f}",
        f"  Recall:     {metrics.transition_recall:.4f}",
        f"  Total:      {metrics.total_transitions}",
        f"    ADD:      {metrics.add_count}",
        f"    MODIFY:   {metrics.modify_count}",
        f"    REMOVE:   {metrics.remove_count}",
        f"    REJECT:   {metrics.reject_count}",
        f"    NO_CHANGE:{metrics.no_change_count}",
        "",
        "--- State ---",
        f"  Precision:  {metrics.state_precision:.4f}",
        f"  Recall:     {metrics.state_recall:.4f}",
        f"  F1:         {metrics.state_f1:.4f}",
        "",
        "--- Error Rates ---",
        f"  False lock:    {metrics.false_lock_rate:.4f}",
        f"  False update:  {metrics.false_update_rate:.4f}",
        f"  False removal: {metrics.false_removal_rate:.4f}",
        f"  False rejection: {metrics.false_rejection_rate:.4f}",
        f"  Stale memory:  {metrics.stale_memory_rate:.4f}",
        f"  Duplicates:    {metrics.duplicate_memory_rate:.4f}",
        f"  Contradictions:{metrics.contradiction_rate:.4f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: int | float, denominator: int | float) -> float:
    """Safe division that returns 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _harmonic(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _span_set(spans: List[Dict[str, Any]]) -> Set[str]:
    """Extract a set of normalized span texts from raw span dicts."""
    result: Set[str] = set()
    for span in spans:
        text = str(span.get("text", "")).strip()
        if text:
            result.add(_normalize_value(text))
    return result


def _compute_false_lock_rate(
    predicted: List[Transition],
    gold: List[Transition],
) -> float:
    """Fraction of predicted ADDs that are not in gold ADDs."""
    gold_adds = {
        _transition_match_key(t) for t in gold
        if t.transition_type == TransitionType.ADD
    }
    pred_adds = [
        t for t in predicted if t.transition_type == TransitionType.ADD
    ]
    if not pred_adds:
        return 0.0
    false_locks = sum(
        1 for t in pred_adds if _transition_match_key(t) not in gold_adds
    )
    return _safe_div(false_locks, len(pred_adds))


def _compute_false_update_rate(
    predicted: List[Transition],
    gold: List[Transition],
) -> float:
    """Fraction of predicted MODIFYs that are not in gold MODIFYs."""
    gold_modifies = {
        _transition_match_key(t) for t in gold
        if t.transition_type == TransitionType.MODIFY
    }
    pred_modifies = [
        t for t in predicted if t.transition_type == TransitionType.MODIFY
    ]
    if not pred_modifies:
        return 0.0
    false_updates = sum(
        1 for t in pred_modifies if _transition_match_key(t) not in gold_modifies
    )
    return _safe_div(false_updates, len(pred_modifies))


def _compute_false_removal_rate(
    predicted: List[Transition],
    gold: List[Transition],
) -> float:
    """Fraction of predicted REMOVEs that are not in gold REMOVEs."""
    gold_removes = {
        _transition_match_key(t) for t in gold
        if t.transition_type == TransitionType.REMOVE
    }
    pred_removes = [
        t for t in predicted if t.transition_type == TransitionType.REMOVE
    ]
    if not pred_removes:
        return 0.0
    false_removes = sum(
        1 for t in pred_removes if _transition_match_key(t) not in gold_removes
    )
    return _safe_div(false_removes, len(pred_removes))


def _compute_false_rejection_rate(
    predicted: List[Transition],
    gold: List[Transition],
) -> float:
    """Fraction of predicted REJECTs that are not in gold REJECTs."""
    gold_rejects = {
        _transition_match_key(t) for t in gold
        if t.transition_type == TransitionType.REJECT
    }
    pred_rejects = [
        t for t in predicted if t.transition_type == TransitionType.REJECT
    ]
    if not pred_rejects:
        return 0.0
    false_rejects = sum(
        1 for t in pred_rejects if _transition_match_key(t) not in gold_rejects
    )
    return _safe_div(false_rejects, len(pred_rejects))


def _compute_stale_memory_rate(
    predicted: ProjectState,
    expected: ProjectState,
) -> float:
    """Fraction of active predicted memories that should be removed or rejected."""
    if not predicted.active_memories:
        return 0.0

    expected_active_keys = {
        _memory_key(m) for m in expected.active_memories
    }
    expected_non_active = {
        (m.category, _normalize_value(m.value))
        for m in expected.all_memories
        if m.status in (MemoryStatus.REMOVED, MemoryStatus.REJECTED)
    }

    stale_count = 0
    for m in predicted.active_memories:
        key = _memory_key(m)
        if key in expected_non_active or key not in expected_active_keys:
            stale_count += 1

    return _safe_div(stale_count, len(predicted.active_memories))


def _compute_duplicate_rate(state: ProjectState) -> float:
    """Fraction of active memories that share category+value with another."""
    if not state.active_memories:
        return 0.0

    seen: Dict[Tuple[MemoryCategory, str], int] = defaultdict(int)
    for m in state.active_memories:
        key = _memory_key(m)
        seen[key] += 1

    duplicate_count = sum(1 for count in seen.values() if count > 1)
    return _safe_div(duplicate_count, len(state.active_memories))


def _compute_contradiction_rate(transitions: List[Transition]) -> float:
    """Fraction of transitions that contradict another in the same turn.

    A contradiction is two transitions with the same (category, normalized
    value) but different non-NO_CHANGE transition types.
    """
    if len(transitions) <= 1:
        return 0.0

    turn_groups: Dict[int, List[Transition]] = defaultdict(list)
    for t in transitions:
        turn_groups[t.turn_number].append(t)

    contradictions = 0
    for turn, group in turn_groups.items():
        by_key: Dict[Tuple[MemoryCategory, str], List[TransitionType]] = defaultdict(list)
        for t in group:
            if t.transition_type == TransitionType.NO_CHANGE:
                continue
            key = (t.category, _normalize_value(t.value))
            by_key[key].append(t.transition_type)

        for key, types in by_key.items():
            unique_types = set(types)
            if len(unique_types) > 1:
                contradictions += 1

    return _safe_div(contradictions, len(transitions))
