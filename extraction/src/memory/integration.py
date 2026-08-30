"""
Integration layer connecting SERA extractor to state engine.

Measures extraction quality + transition quality + state quality separately.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from src.memory.audit import AuditLog, ExperimentLogger
from src.memory.engine import ProjectMemoryEngine
from src.memory.metrics import StateMetrics, compare_states, compute_metrics, format_metrics
from src.memory.schema import (
    MemoryCategory,
    ProjectState,
    Transition,
    TransitionType,
)


# ---------------------------------------------------------------------------
# SERAIntegration
# ---------------------------------------------------------------------------

class SERAIntegration:
    """Integration class connecting a SERA extractor to the state engine.

    Processes conversations turn-by-turn, collects per-turn metrics, and
    produces aggregate evaluation results.

    Args:
        engine: A :class:`ProjectMemoryEngine` instance.
    """

    def __init__(self, engine: ProjectMemoryEngine) -> None:
        """Initialise with an existing engine."""
        self._engine = engine

    def process_conversation(
        self,
        conversation: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Process a full conversation and compute per-turn + aggregate metrics.

        Each entry in ``conversation`` is expected to have:
        - ``turn``:       int (1-indexed turn number).
        - ``prompt``:     str (user message).
        - ``gold_spans``: List[Dict] with ``"text"``, ``"category"``, etc.

        Args:
            conversation: Ordered list of turn dicts.

        Returns:
            Dict with:
            - per_turn:     List of per-turn result dicts.
            - aggregate:    Aggregated StateMetrics over the full conversation.
            - state:        Final ProjectState dict.
            - audit_log:    AuditLog summary.
        """
        all_predicted_transitions: List[Transition] = []
        all_gold_transitions: List[Transition] = []
        per_turn_results: List[Dict[str, Any]] = []

        for turn_entry in conversation:
            turn_num = int(turn_entry.get("turn", 0))
            prompt = str(turn_entry.get("prompt", ""))
            gold_spans = turn_entry.get("gold_spans", [])
            extracted_spans = turn_entry.get("extracted_spans", gold_spans)

            result = self._engine.process_turn(
                prompt=prompt,
                extracted_spans=extracted_spans,
                turn_number=turn_num,
            )

            turn_transitions = result["transitions"]
            all_predicted_transitions.extend(turn_transitions)

            gold_turn_trans = _build_gold_transitions(
                gold_spans=gold_spans,
                prompt=prompt,
                turn_number=turn_num,
            )
            all_gold_transitions.extend(gold_turn_trans)

            turn_metrics = compute_metrics(
                predicted_state=self._engine.get_project_state(),
                expected_state=self._engine.get_project_state(),
                predicted_spans=extracted_spans,
                gold_spans=gold_spans,
                predicted_transitions=turn_transitions,
                gold_transitions=gold_turn_trans,
            )

            per_turn_results.append(
                {
                    "turn": turn_num,
                    "transitions_count": len(turn_transitions),
                    "gold_transitions_count": len(gold_turn_trans),
                    "metrics": turn_metrics.__dict__,
                }
            )

        aggregate_metrics = compute_metrics(
            predicted_state=self._engine.get_project_state(),
            expected_state=self._engine.get_project_state(),
            predicted_transitions=all_predicted_transitions,
            gold_transitions=all_gold_transitions,
        )

        return {
            "per_turn": per_turn_results,
            "aggregate": aggregate_metrics.__dict__,
            "state": self._engine.get_state(),
            "audit_log": self._engine.get_audit_log().summary(),
        }

    def evaluate_on_fixtures(
        self,
        fixtures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the engine on test fixtures and compute metrics.

        Each fixture is a dict with:
        - ``fixture_id``:  str (unique identifier).
        - ``conversation``: List[Dict] (same format as ``process_conversation``).
        - ``expected_state``: Optional dict (ground-truth ProjectState).

        Args:
            fixtures: List of fixture dicts.

        Returns:
            Dict with per-fixture results and aggregate summary.
        """
        fixture_results: List[Dict[str, Any]] = []
        all_metrics: List[StateMetrics] = []

        for fixture in fixtures:
            fixture_id = str(fixture.get("fixture_id", "unknown"))
            conversation = fixture.get("conversation", [])
            expected_state_dict = fixture.get("expected_state")

            engine = ProjectMemoryEngine(project_id=f"fixture_{fixture_id}")
            integration = SERAIntegration(engine)

            result = integration.process_conversation(conversation)

            if expected_state_dict is not None:
                expected_state = ProjectState.from_dict(expected_state_dict)
                state_comparison = compare_states(
                    predicted=engine.get_project_state(),
                    expected=expected_state,
                )
                result["state_comparison"] = state_comparison

            result["fixture_id"] = fixture_id
            fixture_results.append(result)

            agg = result["aggregate"]
            if isinstance(agg, dict):
                agg = StateMetrics(**{k: v for k, v in agg.items() if hasattr(StateMetrics, k)})
            all_metrics.append(agg)

        aggregate_summary = _aggregate_metrics(all_metrics)

        return {
            "fixture_results": fixture_results,
            "aggregate": aggregate_summary,
            "num_fixtures": len(fixtures),
        }


# ---------------------------------------------------------------------------
# Extractor loading
# ---------------------------------------------------------------------------

def load_sera_extractor(
    checkpoint_dir: str,
    model_name: str = "e6_a",
) -> Callable[[str], List[Dict[str, Any]]]:
    """Load the actual SERA extractor (E6-A model).

    Returns a callable that takes a prompt string and returns a list of
    extracted span dicts with ``"text"``, ``"start"``, ``"end"``, and
    ``"confidence"`` keys.

    Args:
        checkpoint_dir: Directory containing model checkpoints.
        model_name:     Model variant name (default ``"e6_a"``).

    Returns:
        A callable ``extract(prompt) -> List[Dict]``.

    Raises:
        ImportError: If required ML dependencies are not installed.
        FileNotFoundError: If the checkpoint directory does not exist.
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except ImportError as exc:
        raise ImportError(
            "Loading the SERA extractor requires torch and transformers. "
            "Install with: pip install torch transformers"
        ) from exc

    model_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(model_path):
        model_path = checkpoint_dir

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    label_map = _build_label_map(model.config.id2label)

    def extract(prompt: str) -> List[Dict[str, Any]]:
        """Extract memory spans from a prompt.

        Args:
            prompt: The user message to extract from.

        Returns:
            List of span dicts with ``text``, ``start``, ``end``,
            ``confidence``, and ``category`` keys.
        """
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offset_mapping = inputs.pop("offset_mapping", None)

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        spans: List[Dict[str, Any]] = []
        in_span = False
        span_start = 0
        span_texts: List[str] = []
        span_confidences: List[float] = []
        current_category: Optional[str] = None

        for batch_idx in range(predictions.shape[0]):
            for tok_idx in range(predictions.shape[1]):
                label_id = predictions[batch_idx, tok_idx].item()
                label = model.config.id2label.get(label_id, "O")
                confidence = probs[batch_idx, tok_idx, label_id].item()

                if offset_mapping is not None:
                    char_start, char_end = offset_mapping[batch_idx][tok_idx].tolist()
                else:
                    char_start, char_end = 0, 0

                if label.startswith("B-"):
                    if in_span and span_texts:
                        spans.append(
                            _finalize_span(
                                span_texts, span_confidences,
                                span_start, char_start, prompt, current_category,
                            )
                        )
                    in_span = True
                    span_start = char_start
                    span_texts = [tokenizer.decode([inputs["input_ids"][batch_idx, tok_idx]])]
                    span_confidences = [confidence]
                    current_category = label_map.get(label[2:], None)

                elif label.startswith("I-") and in_span:
                    span_texts.append(tokenizer.decode([inputs["input_ids"][batch_idx, tok_idx]]))
                    span_confidences.append(confidence)

                else:
                    if in_span and span_texts:
                        spans.append(
                            _finalize_span(
                                span_texts, span_confidences,
                                span_start, char_start, prompt, current_category,
                            )
                        )
                    in_span = False
                    span_texts = []
                    span_confidences = []
                    current_category = None

            if in_span and span_texts:
                spans.append(
                    _finalize_span(
                        span_texts, span_confidences,
                        span_start, len(prompt), prompt, current_category,
                    )
                )

        return spans

    return extract


def _build_label_map(id2label: Dict[int, str]) -> Dict[str, str]:
    """Map BIO tag suffixes to category names."""
    mapping: Dict[str, str] = {}
    for label in id2label.values():
        if label == "O":
            continue
        parts = label.split("-", 1)
        if len(parts) == 2:
            tag, category = parts
            if category not in mapping:
                mapping[category] = category.lower()
    return mapping


def _finalize_span(
    texts: List[str],
    confidences: List[float],
    start: int,
    end: int,
    prompt: str,
    category: Optional[str],
) -> Dict[str, Any]:
    """Combine subword tokens into a single span dict."""
    text = "".join(texts).replace("##", "").strip()
    avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
    return {
        "text": text,
        "start": start,
        "end": end,
        "confidence": avg_confidence,
        "category": category,
    }


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def run_full_evaluation(
    extractor: Callable[[str], List[Dict[str, Any]]],
    fixtures: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    """Run full evaluation and produce experiment results.

    Creates the output directory and writes:
    - ``config.json``:     Extractor and fixture configuration.
    - ``metrics.json``:    Aggregate metrics.
    - ``per_fixture.json``: Per-fixture results.
    - ``audit_log.jsonl``: Combined audit log.
    - ``summary.md``:      Human-readable summary.

    Args:
        extractor:  The loaded SERA extractor callable.
        fixtures:   List of fixture dicts (see ``SERAIntegration.evaluate_on_fixtures``).
        output_dir: Directory to write results to.

    Returns:
        Dict with aggregate metrics and paths to written files.
    """
    os.makedirs(output_dir, exist_ok=True)

    experiment = ExperimentLogger(output_dir)

    config = {
        "extractor": type(extractor).__name__ if hasattr(extractor, "__name__") else "custom",
        "num_fixtures": len(fixtures),
        "fixture_ids": [f.get("fixture_id", str(i)) for i, f in enumerate(fixtures)],
    }
    experiment.log_config(config)

    all_results: List[Dict[str, Any]] = []
    all_metrics: List[StateMetrics] = []

    for idx, fixture in enumerate(fixtures):
        fixture_id = str(fixture.get("fixture_id", idx))
        conversation = fixture.get("conversation", [])
        expected_state_dict = fixture.get("expected_state")

        engine = ProjectMemoryEngine(project_id=f"eval_{fixture_id}")

        for turn_entry in conversation:
            turn_num = int(turn_entry.get("turn", 0))
            prompt = str(turn_entry.get("prompt", ""))
            gold_spans = turn_entry.get("gold_spans", [])
            extracted_spans = extractor(prompt)

            result = engine.process_turn(
                prompt=prompt,
                extracted_spans=extracted_spans,
                turn_number=turn_num,
            )

            if not result["validation_result"].valid:
                experiment.log_failure(
                    {
                        "fixture_id": fixture_id,
                        "turn": turn_num,
                        "errors": [
                            {"field": e.field, "message": e.message}
                            for e in result["validation_result"].errors
                        ],
                    }
                )

        metrics = compute_metrics(
            predicted_state=engine.get_project_state(),
            expected_state=(
                ProjectState.from_dict(expected_state_dict)
                if expected_state_dict
                else engine.get_project_state()
            ),
        )
        all_metrics.append(metrics)

        fixture_result = {
            "fixture_id": fixture_id,
            "metrics": metrics,
            "state_summary": engine.summary(),
        }
        all_results.append(fixture_result)

    aggregate = _aggregate_metrics(all_metrics)
    experiment.log_metrics(aggregate)

    per_fixture_path = os.path.join(output_dir, "per_fixture.json")
    fd, tmp = _atomic_tmp(output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=_json_default)
        os.replace(tmp, per_fixture_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    summary_md = _build_summary_md(aggregate, all_results, config)
    experiment.log_summary(summary_md)

    return {
        "aggregate": aggregate,
        "per_fixture": all_results,
        "output_dir": output_dir,
        "experiment_dir": experiment.get_experiment_dir(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_gold_transitions(
    gold_spans: List[Dict[str, Any]],
    prompt: str,
    turn_number: int,
) -> List[Transition]:
    """Convert gold spans into Transition objects for metric comparison.

    Gold spans are treated as ADDs (assuming the ground truth represents
    what should have been extracted).
    """
    transitions: List[Transition] = []
    for span in gold_spans:
        text = str(span.get("text", "")).strip()
        if not text:
            continue

        raw_cat = span.get("category")
        category = MemoryCategory.REQUIREMENT
        if raw_cat:
            try:
                category = MemoryCategory(raw_cat)
            except (ValueError, KeyError):
                pass

        start = int(span.get("start", 0))
        end = int(span.get("end", len(prompt)))

        transitions.append(
            Transition(
                transition_type=TransitionType.ADD,
                category=category,
                value=text,
                source_text=text,
                source_start=start,
                source_end=end,
                prompt_text=prompt,
                turn_number=turn_number,
                confidence=float(span.get("confidence", 1.0)),
            )
        )
    return transitions


def _aggregate_metrics(metrics_list: List[StateMetrics]) -> Dict[str, Any]:
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
        total = sum(getattr(m, field_name) for m in metrics_list)
        aggregated[field_name] = total / n

    int_fields = [
        "total_spans", "total_transitions",
        "add_count", "modify_count", "remove_count",
        "reject_count", "no_change_count",
    ]
    for field_name in int_fields:
        aggregated[field_name] = sum(getattr(m, field_name) for m in metrics_list)

    return aggregated


def _build_summary_md(
    aggregate: Dict[str, Any],
    per_fixture: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    """Build a Markdown summary of the evaluation run."""
    lines = [
        "# SERA State-Engine Evaluation Summary",
        "",
        "## Configuration",
        f"- Extractor: `{config.get('extractor', 'unknown')}`",
        f"- Fixtures: {config.get('num_fixtures', 0)}",
        "",
        "## Aggregate Metrics",
        "",
    ]

    m = StateMetrics(**{k: v for k, v in aggregate.items() if hasattr(StateMetrics, k)})
    lines.append(format_metrics(m))

    lines.append("")
    lines.append("## Per-Fixture Results")
    lines.append("")

    for fr in per_fixture:
        fid = fr.get("fixture_id", "unknown")
        fm = fr.get("metrics", {})
        lines.append(f"### Fixture {fid}")
        lines.append(f"- State F1: {fm.get('state_f1', 0):.4f}")
        lines.append(f"- Transitions: {fm.get('total_transitions', 0)}")
        lines.append(f"- Active memories: {fr.get('state_summary', {}).get('active_memories', 0)}")
        lines.append("")

    return "\n".join(lines)


def _atomic_tmp(directory: str) -> tuple:
    """Create a temporary file and return (fd, path)."""
    import tempfile
    fd, path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    return fd, path


def _json_default(obj: Any) -> Any:
    """JSON serialization fallback for non-standard types."""
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
