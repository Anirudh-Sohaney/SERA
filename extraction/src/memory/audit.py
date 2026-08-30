"""
Audit logging for memory state changes.

Every modification must produce an explicit audit record.
Never silently modify memory.

The audit system records every transition with full provenance:
state snapshots before and after, validation results, and timestamps.
The log is append-only and can be serialized to JSONL for persistence.

Usage::

    from src.memory.audit import AuditLog, AuditRecord

    log = AuditLog()
    record = AuditRecord(
        transition=transition,
        state_before=state.to_dict(),
        state_after=new_state.to_dict(),
        validation_passed=True,
        validation_errors=[],
    )
    log.add_record(record)
    log.save("audit_log.jsonl")
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from src.memory.schema import (
    MemoryCategory,
    MemoryItem,
    ProjectState,
    Transition,
    TransitionType,
)


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------

@dataclass
class AuditRecord:
    """A single audit record capturing a state transition with provenance.

    Attributes:
        record_id:         Unique identifier (UUID4).
        timestamp:         ISO 8601 UTC timestamp of when this record was
                           created.
        turn_number:       Conversation turn that produced the transition.
        transition:        The :class:`Transition` that was applied.
        state_before:      Serialised state snapshot BEFORE the transition.
        state_after:       Serialised state snapshot AFTER the transition.
        validation_passed: Whether the transition passed validation.
        validation_errors: List of validation error dicts (empty if passed).
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    turn_number: int = 0
    transition: Transition = field(default_factory=Transition)
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
    validation_passed: bool = True
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        The ``transition`` field is serialized via its own ``to_dict``
        method so that enum values are stored as strings.
        """
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "turn_number": self.turn_number,
            "transition": self.transition.to_dict(),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AuditRecord:
        """Deserialize from a dictionary.

        Raises:
            ValueError: If required fields are missing or malformed.
        """
        if not d:
            raise ValueError("Cannot deserialize AuditRecord from empty dict")
        try:
            transition = Transition.from_dict(d["transition"])
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid transition in AuditRecord: {e}")

        return cls(
            record_id=str(d.get("record_id", str(uuid.uuid4()))),
            timestamp=str(d.get("timestamp", datetime.now(timezone.utc).isoformat())),
            turn_number=int(d.get("turn_number", 0)),
            transition=transition,
            state_before=d.get("state_before", {}),
            state_after=d.get("state_after", {}),
            validation_passed=bool(d.get("validation_passed", True)),
            validation_errors=d.get("validation_errors", []),
        )

    def to_json(self) -> str:
        """Serialize to a single-line JSON string (suitable for JSONL)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> AuditRecord:
        """Deserialize from a JSON string.

        Raises:
            json.JSONDecodeError: If the string is not valid JSON.
            ValueError: If the resulting dict is invalid.
        """
        if not isinstance(s, str) or not s.strip():
            raise ValueError("Cannot deserialize AuditRecord from empty string")
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class AuditLog:
    """Append-only audit log for memory state transitions.

    The log stores :class:`AuditRecord` objects and supports querying
    by turn, category, and transition type.  It can be serialized to
    and deserialized from JSONL files.

    Usage::

        log = AuditLog()
        log.add_record(record)
        print(log.summary())

        # Persist
        log.save("audit_log.jsonl")

        # Reload
        log2 = AuditLog.load("audit_log.jsonl")
    """

    def __init__(self) -> None:
        """Initialise an empty audit log."""
        self.records: List[AuditRecord] = []

    def add_record(self, record: AuditRecord) -> None:
        """Append an audit record to the log.

        Args:
            record: The :class:`AuditRecord` to append.

        Raises:
            TypeError: If ``record`` is not an :class:`AuditRecord`.
        """
        if not isinstance(record, AuditRecord):
            raise TypeError(
                f"Expected AuditRecord, got {type(record).__name__}"
            )
        self.records.append(record)

    def get_records_by_turn(self, turn: int) -> List[AuditRecord]:
        """Return all records for a specific turn number.

        Args:
            turn: The conversation turn to filter by.

        Returns:
            List of matching :class:`AuditRecord` objects.
        """
        return [r for r in self.records if r.turn_number == turn]

    def get_records_by_category(
        self, category: MemoryCategory
    ) -> List[AuditRecord]:
        """Return all records that affect a specific memory category.

        Args:
            category: The :class:`MemoryCategory` to filter by.

        Returns:
            List of matching :class:`AuditRecord` objects.
        """
        return [
            r for r in self.records
            if r.transition.category == category
        ]

    def get_records_by_type(
        self, transition_type: TransitionType
    ) -> List[AuditRecord]:
        """Return all records for a specific transition type.

        Args:
            transition_type: The :class:`TransitionType` to filter by.

        Returns:
            List of matching :class:`AuditRecord` objects.
        """
        return [
            r for r in self.records
            if r.transition.transition_type == transition_type
        ]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize the entire log to a list of dictionaries.

        Returns:
            A list of JSON-compatible dicts, one per audit record.
        """
        return [r.to_dict() for r in self.records]

    def save(self, path: str) -> None:
        """Write the audit log to a JSONL file.

        Each record is written as a single JSON line.  Parent directories
        are created if they do not exist.

        Args:
            path: File path to write the JSONL to.
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(record.to_json())
                f.write("\n")

    @classmethod
    def load(cls, path: str) -> AuditLog:
        """Load an audit log from a JSONL file.

        Each line is parsed as a JSON object and converted to an
        :class:`AuditRecord`.  Malformed lines are skipped with a
        warning printed to stderr.

        Args:
            path: File path to read the JSONL from.

        Returns:
            A new :class:`AuditLog` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        log = cls()
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = AuditRecord.from_json(line)
                    log.records.append(record)
                except (json.JSONDecodeError, ValueError) as exc:
                    import sys
                    print(
                        f"WARNING: Skipping malformed audit record "
                        f"at line {line_no}: {exc}",
                        file=sys.stderr,
                    )
        return log

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the audit log.

        The summary includes:

        * ``total_records``: Total number of records.
        * ``turns``: Set of turn numbers present.
        * ``transition_counts``: Count of each transition type.
        * ``category_counts``: Count of records per category.
        * ``validation_failures``: Number of records where validation
          failed.
        * ``first_timestamp`` / ``last_timestamp``: Time range.

        Returns:
            A dict with summary statistics.
        """
        transition_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}
        turns: set = set()
        validation_failures = 0

        for record in self.records:
            tt = record.transition.transition_type.value
            transition_counts[tt] = transition_counts.get(tt, 0) + 1

            cat = record.transition.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

            turns.add(record.turn_number)

            if not record.validation_passed:
                validation_failures += 1

        first_ts = self.records[0].timestamp if self.records else None
        last_ts = self.records[-1].timestamp if self.records else None

        return {
            "total_records": len(self.records),
            "turns": sorted(turns),
            "transition_counts": transition_counts,
            "category_counts": category_counts,
            "validation_failures": validation_failures,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
        }


# ---------------------------------------------------------------------------
# Experiment logger (spec section 19)
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """Structured logging for experiment runs.

    Creates an immutable experiment directory with the layout::

        logs/state_engine/experiment_XXX/
            config.json
            metrics.json
            failures.jsonl
            summary.md
            audit_log.jsonl

    Each file is written atomically (write-to-temp then rename) to
    prevent corruption from interrupted writes.

    Args:
        experiment_dir: Root directory for this experiment.
    """

    def __init__(self, experiment_dir: str) -> None:
        """Initialise the experiment logger.

        Creates the experiment directory if it does not exist.

        Args:
            experiment_dir: Absolute or relative path to the experiment
                            directory.
        """
        self._experiment_dir = os.path.abspath(experiment_dir)
        os.makedirs(self._experiment_dir, exist_ok=True)

    def get_experiment_dir(self) -> str:
        """Return the absolute path of the experiment directory.

        Returns:
            The experiment directory path.
        """
        return self._experiment_dir

    def log_config(self, config: Dict[str, Any]) -> None:
        """Save the experiment configuration to ``config.json``.

        The config dict is serialized with 2-space indentation for
        readability.

        Args:
            config: The experiment configuration to persist.
        """
        self._atomic_write_json("config.json", config)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save experiment metrics to ``metrics.json``.

        Args:
            metrics: The metrics dict to persist.
        """
        self._atomic_write_json("metrics.json", metrics)

    def log_failure(self, failure: Dict[str, Any]) -> None:
        """Append a failure record to ``failures.jsonl``.

        Each call appends a single JSON line.  The file is created
        if it does not exist.

        Args:
            failure: The failure record to append.
        """
        path = os.path.join(self._experiment_dir, "failures.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(failure, ensure_ascii=False))
            f.write("\n")

    def log_summary(self, summary: str) -> None:
        """Save an experiment summary to ``summary.md``.

        Args:
            summary: Markdown-formatted summary text.
        """
        path = os.path.join(self._experiment_dir, "summary.md")
        fd, tmp_path = tempfile.mkstemp(
            dir=self._experiment_dir, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(summary)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _atomic_write_json(
        self, filename: str, data: Dict[str, Any]
    ) -> None:
        """Atomically write a JSON file.

        Writes to a temporary file in the same directory, then renames
        to the target path.

        Args:
            filename: Target filename (relative to experiment dir).
            data:     The dict to serialize.
        """
        import tempfile

        target_path = os.path.join(self._experiment_dir, filename)
        fd, tmp_path = tempfile.mkstemp(
            dir=self._experiment_dir, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, target_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


# Need to import tempfile for log_summary
import tempfile


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def format_audit_record(record: AuditRecord) -> str:
    """Pretty-print a single audit record for human reading.

    The output includes the record ID, timestamp, transition details,
    validation status, and a before/after state summary.

    Args:
        record: The :class:`AuditRecord` to format.

    Returns:
        A multi-line human-readable string.
    """
    t = record.transition
    lines = [
        f"Audit Record: {record.record_id}",
        f"  Timestamp:      {record.timestamp}",
        f"  Turn:           {record.turn_number}",
        f"  Transition:     {t.transition_type.value}",
        f"  Category:       {t.category.value}",
        f"  Value:          {t.value!r}",
    ]

    if t.old_value is not None:
        lines.append(f"  Old Value:      {t.old_value!r}")
    if t.memory_id is not None:
        lines.append(f"  Memory ID:      {t.memory_id}")

    lines.append(
        f"  Source:         [{t.source_start}:{t.source_end}] "
        f"{t.source_text!r}"
    )

    # Validation status
    status = "PASSED" if record.validation_passed else "FAILED"
    lines.append(f"  Validation:     {status}")
    if record.validation_errors:
        for err in record.validation_errors:
            lines.append(f"    - {err.get('field', '?')}: {err.get('message', '?')}")

    # State summary
    before_active = len(record.state_before.get("active_memories", []))
    after_active = len(record.state_after.get("active_memories", []))
    before_total = len(record.state_before.get("all_memories", []))
    after_total = len(record.state_after.get("all_memories", []))
    lines.append(
        f"  State Before:   {before_active} active, "
        f"{before_total} total memories"
    )
    lines.append(
        f"  State After:    {after_active} active, "
        f"{after_total} total memories"
    )

    return "\n".join(lines)
