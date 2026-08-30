"""
SERA Project-Memory State Engine.

Top-level orchestrator that connects extraction output to persistent memory.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from src.memory.audit import AuditLog, AuditRecord
from src.memory.schema import (
    MemoryCandidate,
    ProjectState,
    Transition,
    TransitionType,
    _new_id,
    _now_iso,
)
from src.memory.transitions import TransitionEngine, build_memory_candidates
from src.memory.validator import Result as ValidationResult
from src.memory.validator import StateValidator


class ProjectMemoryEngine:
    """Top-level state engine for SERA project memory.

    Orchestrates the full pipeline: candidate building, transition
    classification, validation, and audit logging.

    Args:
        project_id: Unique identifier for this project.
    """

    def __init__(self, project_id: str = "default") -> None:
        """Initialize with empty state."""
        self._state = ProjectState(project_id=project_id)
        self._audit_log = AuditLog()
        self._validator = StateValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_turn(
        self,
        prompt: str,
        extracted_spans: List[Dict],
        turn_number: int,
    ) -> Dict[str, Any]:
        """Process a single conversation turn.

        Args:
            prompt:           The original user message.
            extracted_spans:  Spans from SERA extractor, each with at
                              least ``"text"``; optional ``"start"``,
                              ``"end"``, ``"confidence"``, ``"field"``.
            turn_number:      The turn number (1-indexed).

        Returns:
            Dict with:
            - transitions:        List[Transition] produced this turn.
            - state_snapshot:     Dict of current active memories.
            - audit_records:      List[AuditRecord] for this turn.
            - validation_result:  Aggregated ValidationResult.
        """
        if turn_number < 1:
            raise ValueError(f"turn_number must be >= 1, got {turn_number}")

        self._state.current_turn = max(self._state.current_turn, turn_number)

        candidates = build_memory_candidates(
            spans=extracted_spans,
            prompt_text=prompt,
            turn_number=turn_number,
        )

        transition_engine = TransitionEngine(self._state)
        transitions = transition_engine.process_candidates(candidates)

        all_validation_errors: List[Dict[str, Any]] = []
        any_invalid = False
        for transition in transitions:
            state_before = self._state.to_dict()
            validation = self._validator.validate_transition(transition, self._state)
            audit_errors: List[Dict[str, Any]] = []
            if not validation.valid:
                any_invalid = True
                for err in validation.errors:
                    audit_errors.append(
                        {"field": err.field, "message": err.message, "severity": err.severity}
                    )
                    all_validation_errors.append(
                        {"field": err.field, "message": err.message, "severity": err.severity}
                    )
            for warn in validation.warnings:
                audit_errors.append(
                    {"field": warn.field, "message": warn.message, "severity": warn.severity}
                )
                all_validation_errors.append(
                    {"field": warn.field, "message": warn.message, "severity": warn.severity}
                )

            record = AuditRecord(
                turn_number=turn_number,
                transition=transition,
                state_before=state_before,
                state_after=self._state.to_dict(),
                validation_passed=validation.valid,
                validation_errors=audit_errors,
            )
            self._audit_log.add_record(record)

        aggregated = ValidationResult()
        aggregated.valid = not any_invalid
        for err_info in all_validation_errors:
            if err_info["severity"] == "error":
                aggregated.add_error(err_info["field"], err_info["message"])
            else:
                aggregated.add_warning(err_info["field"], err_info["message"])

        return {
            "transitions": transitions,
            "state_snapshot": {
                "active_memories": [m.to_dict() for m in self._state.active_memories],
                "all_memories_count": len(self._state.all_memories),
                "current_turn": self._state.current_turn,
            },
            "audit_records": self._audit_log.get_records_by_turn(turn_number),
            "validation_result": aggregated,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return current state as dict."""
        return self._state.to_dict()

    def get_transitions(self) -> List[Dict[str, Any]]:
        """Return all transitions as dicts."""
        return [t.to_dict() for t in self._state.transition_log]

    def get_audit_log(self) -> AuditLog:
        """Return the audit log."""
        return self._audit_log

    def get_project_state(self) -> ProjectState:
        """Return the underlying ProjectState for metrics/evaluation."""
        return self._state

    def save(self, directory: str) -> None:
        """Save state, transitions, and audit log to directory.

        Creates the directory if it does not exist. Writes:
        - state.json:          Full ProjectState.
        - transitions.json:    Transition log.
        - audit_log.jsonl:     Audit log in JSONL format.

        Args:
            directory: Target directory path.
        """
        os.makedirs(directory, exist_ok=True)

        state_path = os.path.join(directory, "state.json")
        self._state.save(state_path)

        transitions_path = os.path.join(directory, "transitions.json")
        fd, tmp = _atomic_tmp(directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.get_transitions(), f, indent=2, ensure_ascii=False)
            os.replace(tmp, transitions_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

        audit_path = os.path.join(directory, "audit_log.jsonl")
        self._audit_log.save(audit_path)

    @classmethod
    def load(cls, directory: str) -> "ProjectMemoryEngine":
        """Load from directory.

        Expects the directory to contain state.json and audit_log.jsonl
        (written by a prior ``save`` call).

        Args:
            directory: Directory containing saved engine files.

        Returns:
            A reconstructed ProjectMemoryEngine.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        state_path = os.path.join(directory, "state.json")
        state = ProjectState.load(state_path)

        engine = cls.__new__(cls)
        engine._state = state
        engine._validator = StateValidator()

        audit_path = os.path.join(directory, "audit_log.jsonl")
        if os.path.exists(audit_path):
            engine._audit_log = AuditLog.load(audit_path)
        else:
            engine._audit_log = AuditLog()

        return engine

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        transition_counts: Dict[str, int] = {}
        for t in self._state.transition_log:
            key = t.transition_type.value
            transition_counts[key] = transition_counts.get(key, 0) + 1

        category_counts: Dict[str, int] = {}
        for m in self._state.active_memories:
            key = m.category.value
            category_counts[key] = category_counts.get(key, 0) + 1

        return {
            "project_id": self._state.project_id,
            "current_turn": self._state.current_turn,
            "active_memories": len(self._state.active_memories),
            "total_memories": len(self._state.all_memories),
            "total_transitions": len(self._state.transition_log),
            "transition_counts": transition_counts,
            "category_counts": category_counts,
            "audit_records": len(self._audit_log.records),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_tmp(directory: str) -> tuple:
    """Create a temporary file in *directory* and return (fd, path)."""
    import tempfile
    fd, path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    return fd, path
