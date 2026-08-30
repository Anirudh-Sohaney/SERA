"""
Deterministic state validator.

Validates transitions and state consistency before committing changes.
Invalid transitions must fail loudly — never silently repair corrupted state.

The validator is a pure function layer: it never mutates state. Every
``validate_*`` method returns a :class:`Result` that the caller inspects
before deciding whether to proceed with the commit.

Usage::

    from src.memory.validator import StateValidator

    validator = StateValidator()
    result = validator.validate_transition(transition, state)
    if not result.valid:
        for err in result.errors:
            print(f"ERROR: {err.field}: {err.message}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.memory.schema import (
    MemoryCategory,
    MemoryItem,
    MemoryStatus,
    ProjectState,
    Transition,
    TransitionType,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    """A single validation issue.

    Attributes:
        field:    Dot-notation path to the offending field
                  (e.g. ``"transition.value"``).
        message:  Human-readable description of the problem.
        severity: ``"error"`` (blocks commit) or ``"warning"`` (informational).
    """

    field: str
    message: str
    severity: str = "error"

    def __post_init__(self) -> None:
        if self.severity not in ("error", "warning"):
            raise ValueError(
                f"severity must be 'error' or 'warning', got {self.severity!r}"
            )


@dataclass
class Result:
    """Aggregated validation outcome.

    Attributes:
        valid:    ``True`` when there are zero errors.  Warnings alone do
                  not invalidate the result.
        errors:   List of :class:`ValidationError` with severity ``"error"``.
        warnings: List of :class:`ValidationError` with severity ``"warning"``.
    """

    valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def add_error(self, fld: str, message: str) -> None:
        """Append an error and flip ``valid`` to ``False``.

        Args:
            fld:     Field path that failed validation.
            message: Description of the failure.
        """
        self.errors.append(ValidationError(field=fld, message=message, severity="error"))
        self.valid = False

    def add_warning(self, fld: str, message: str) -> None:
        """Append a non-blocking warning.

        Args:
            fld:     Field path that triggered the warning.
            message: Description of the concern.
        """
        self.warnings.append(ValidationError(field=fld, message=message, severity="warning"))


# ---------------------------------------------------------------------------
# Normalization helper (matches matcher.normalize_text)
# ---------------------------------------------------------------------------

def _normalize_value(text: str) -> str:
    """Case-insensitive, stripped, whitespace-collapsed normalization.

    This mirrors the normalization used by :class:`MemoryItem.matches_value`
    and :class:`StateMatcher` to ensure duplicate detection is consistent.
    """
    if not text:
        return ""
    import re
    import unicodedata
    result = unicodedata.normalize("NFKD", text)
    result = result.lower().strip()
    result = re.sub(r"\s+", " ", result)
    return result


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class StateValidator:
    """Deterministic state validator.

    Every ``validate_*`` method returns a :class:`Result`.  The caller
    MUST check ``result.valid`` before committing any state mutation.

    Examples::

        validator = StateValidator()
        result = validator.validate_transition(transition, state)
        if not result.valid:
            raise ValueError(f"Invalid transition: {result.errors}")
    """

    def __init__(self) -> None:
        """Initialise the validator (stateless)."""
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_transition(
        self, transition: Transition, state: ProjectState
    ) -> Result:
        """Validate a single transition against the current state.

        Checks performed:

        * ``transition_type`` is a valid :class:`TransitionType`.
        * ADD: ``memory_id`` is ``None``, ``value`` is non-empty.
        * MODIFY: ``memory_id`` exists and is ACTIVE, ``old_value`` matches
          the current value.
        * REMOVE: ``memory_id`` exists and is ACTIVE.
        * REJECT: ``memory_id`` exists and is ACTIVE **or** ``None``
          (rejecting a never-added value).
        * NO_CHANGE: ``memory_id`` exists and is ACTIVE, ``value`` matches.
        * Source-text fields (``source_text``, ``source_start``,
          ``source_end``) are self-consistent.
        * ``turn_number`` > 0.
        * No duplicate active memories for the same category + value after
          the transition would be applied.

        Args:
            transition: The transition to validate.
            state:      The current project state.

        Returns:
            A :class:`Result` describing all issues found.
        """
        result = Result()

        # -- transition_type must be valid --------------------------------
        if not isinstance(transition.transition_type, TransitionType):
            result.add_error(
                "transition.transition_type",
                f"Invalid transition type: {transition.transition_type!r}",
            )
            # Cannot continue without knowing the type
            return result

        tt = transition.transition_type

        # -- turn_number must be > 0 --------------------------------------
        if transition.turn_number <= 0:
            result.add_error(
                "transition.turn_number",
                f"turn_number must be > 0, got {transition.turn_number}",
            )

        # -- source_text consistency --------------------------------------
        if transition.source_text and transition.source_start < 0:
            result.add_error(
                "transition.source_start",
                "source_start must be >= 0 when source_text is provided",
            )
        if transition.source_text and transition.source_end <= transition.source_start:
            result.add_error(
                "transition.source_end",
                "source_end must be > source_start when source_text is provided",
            )
        if transition.source_text and transition.prompt_text:
            if transition.source_start >= len(transition.prompt_text):
                result.add_error(
                    "transition.source_start",
                    "source_start exceeds prompt_text length",
                )
            elif transition.source_end > len(transition.prompt_text):
                result.add_error(
                    "transition.source_end",
                    "source_end exceeds prompt_text length",
                )
            else:
                extracted = transition.prompt_text[
                    transition.source_start : transition.source_end
                ]
                if extracted != transition.source_text:
                    result.add_error(
                        "transition.source_text",
                        (
                            "prompt_text[source_start:source_end] does not "
                            "match source_text"
                        ),
                    )

        # -- type-specific rules ------------------------------------------
        if tt == TransitionType.ADD:
            self._validate_add(transition, state, result)
        elif tt == TransitionType.MODIFY:
            self._validate_modify(transition, state, result)
        elif tt == TransitionType.REMOVE:
            self._validate_remove(transition, state, result)
        elif tt == TransitionType.REJECT:
            self._validate_reject(transition, state, result)
        elif tt == TransitionType.NO_CHANGE:
            self._validate_no_change(transition, state, result)

        # -- post-transition duplicate check ------------------------------
        self._check_post_transition_duplicates(transition, state, result)

        return result

    def validate_state(self, state: ProjectState) -> Result:
        """Validate the full project state for internal consistency.

        Checks performed:

        * No duplicate active memories (same category + normalised value).
        * All active memories have a valid schema.
        * All ``source_text`` references are non-empty.
        * Transition log is ordered by ``turn_number`` (monotonic).
        * Every MODIFY/REMOVE/REJECT in the log references a valid
          ``memory_id`` present in ``all_memories``.
        * No memory is both active and removed/rejected.

        Args:
            state: The project state to validate.

        Returns:
            A :class:`Result` describing all issues found.
        """
        result = Result()

        # -- duplicate active memories ------------------------------------
        dup_errors = validate_no_duplicates(state)
        for err in dup_errors:
            if err.severity == "error":
                result.add_error(err.field, err.message)
            else:
                result.add_warning(err.field, err.message)

        # -- active memory schema validation ------------------------------
        for idx, item in enumerate(state.active_memories):
            schema_result = self.validate_schema(item)
            if not schema_result.valid:
                for err in schema_result.errors:
                    result.add_error(
                        f"active_memories[{idx}].{err.field}", err.message
                    )
            for warn in schema_result.warnings:
                result.add_warning(
                    f"active_memories[{idx}].{warn.field}", warn.message
                )

        # -- active memory source_text non-empty --------------------------
        for idx, item in enumerate(state.active_memories):
            if not item.source_text:
                result.add_warning(
                    f"active_memories[{idx}].source_text",
                    "source_text is empty for active memory",
                )

        # -- transition log ordering (monotonic turn_number) --------------
        for i in range(1, len(state.transition_log)):
            prev_turn = state.transition_log[i - 1].turn_number
            curr_turn = state.transition_log[i].turn_number
            if curr_turn < prev_turn:
                result.add_error(
                    "transition_log",
                    (
                        f"Transition at index {i} has turn_number {curr_turn} "
                        f"< previous turn_number {prev_turn}"
                    ),
                )

        # -- MODIFY/REMOVE/REJECT reference valid memory_ids --------------
        known_ids = {m.memory_id for m in state.all_memories}
        for i, t in enumerate(state.transition_log):
            if t.transition_type in (
                TransitionType.MODIFY,
                TransitionType.REMOVE,
                TransitionType.REJECT,
            ):
                if t.memory_id is not None and t.memory_id not in known_ids:
                    result.add_error(
                        f"transition_log[{i}].memory_id",
                        (
                            f"{t.transition_type.value} references "
                            f"memory_id {t.memory_id!r} not found in all_memories"
                        ),
                    )

        # -- no memory is both ACTIVE and removed/rejected ----------------
        active_ids = {m.memory_id for m in state.active_memories}
        for item in state.all_memories:
            if item.status in (MemoryStatus.REMOVED, MemoryStatus.REJECTED):
                if item.memory_id in active_ids:
                    result.add_error(
                        "state",
                        (
                            f"Memory {item.memory_id} has status "
                            f"{item.status.value} but also appears in "
                            f"active_memories"
                        ),
                    )

        # -- consistency checks -------------------------------------------
        consistency_errors = validate_state_consistency(state)
        for err in consistency_errors:
            if err.severity == "error":
                result.add_error(err.field, err.message)
            else:
                result.add_warning(err.field, err.message)

        return result

    def validate_schema(self, item: MemoryItem) -> Result:
        """Validate that a single MemoryItem has a valid schema.

        Checks performed:

        * ``memory_id`` is a non-empty string.
        * ``category`` is a valid :class:`MemoryCategory`.
        * ``value`` is non-empty.
        * ``source_text`` is non-empty.
        * ``source_start`` >= 0.
        * ``source_end`` > ``source_start``.
        * ``confidence`` is between 0.0 and 1.0 (inclusive).
        * ``created_turn`` >= 1.
        * ``updated_turn`` >= ``created_turn``.

        Args:
            item: The memory item to validate.

        Returns:
            A :class:`Result` describing all issues found.
        """
        result = Result()

        if not item.memory_id or not isinstance(item.memory_id, str):
            result.add_error("memory_id", "memory_id must be a non-empty string")

        if not isinstance(item.category, MemoryCategory):
            result.add_error(
                "category", f"Invalid category: {item.category!r}"
            )

        if not item.value or not item.value.strip():
            result.add_error("value", "value must be non-empty")

        if not item.source_text or not item.source_text.strip():
            result.add_warning("source_text", "source_text is empty")

        if item.source_start < 0:
            result.add_error(
                "source_start",
                f"source_start must be >= 0, got {item.source_start}",
            )

        if item.source_end <= item.source_start:
            result.add_error(
                "source_end",
                f"source_end must be > source_start, got source_end={item.source_end}, source_start={item.source_start}",
            )

        if not (0.0 <= item.confidence <= 1.0):
            result.add_error(
                "confidence",
                f"confidence must be in [0.0, 1.0], got {item.confidence}",
            )

        if item.created_turn < 1:
            result.add_error(
                "created_turn",
                f"created_turn must be >= 1, got {item.created_turn}",
            )

        if item.updated_turn < item.created_turn:
            result.add_error(
                "updated_turn",
                (
                    f"updated_turn ({item.updated_turn}) must be >= "
                    f"created_turn ({item.created_turn})"
                ),
            )

        return result

    def validate_source_text(self, item: MemoryItem) -> Result:
        """Validate that source_text is an exact substring of prompt_text.

        Checks performed:

        * ``source_text`` is a substring of ``prompt_text`` (exact match).
        * ``prompt_text[source_start:source_end] == source_text``.
        * ``source_start`` and ``source_end`` are within ``prompt_text`` bounds.

        Args:
            item: The memory item whose source-text fields to validate.

        Returns:
            A :class:`Result` describing all issues found.
        """
        result = Result()

        if not item.prompt_text:
            result.add_error("prompt_text", "prompt_text is empty")
            return result

        if not item.source_text:
            result.add_error("source_text", "source_text is empty")
            return result

        pt_len = len(item.prompt_text)

        # Bounds check
        if item.source_start < 0:
            result.add_error(
                "source_start",
                f"source_start ({item.source_start}) must be >= 0",
            )
            return result

        if item.source_end > pt_len:
            result.add_error(
                "source_end",
                (
                    f"source_end ({item.source_end}) exceeds "
                    f"prompt_text length ({pt_len})"
                ),
            )
            return result

        if item.source_start >= pt_len:
            result.add_error(
                "source_start",
                (
                    f"source_start ({item.source_start}) exceeds "
                    f"prompt_text length ({pt_len})"
                ),
            )
            return result

        if item.source_end <= item.source_start:
            result.add_error(
                "source_end",
                (
                    f"source_end ({item.source_end}) must be > "
                    f"source_start ({item.source_start})"
                ),
            )
            return result

        # Exact substring match
        extracted = item.prompt_text[item.source_start : item.source_end]
        if extracted != item.source_text:
            result.add_error(
                "source_text",
                (
                    f"prompt_text[{item.source_start}:{item.source_end}] "
                    f"does not match source_text. Expected "
                    f"{item.source_text!r}, got {extracted!r}"
                ),
            )

        # Prompt contains source_text somewhere (not necessarily at the
        # indicated offsets — could be a secondary occurrence).
        if item.source_text not in item.prompt_text:
            result.add_error(
                "source_text",
                "source_text is not a substring of prompt_text",
            )

        return result

    # ------------------------------------------------------------------
    # Private: type-specific transition checks
    # ------------------------------------------------------------------

    def _validate_add(
        self, transition: Transition, state: ProjectState, result: Result
    ) -> None:
        """Validate ADD-specific constraints."""
        if transition.memory_id is not None:
            result.add_error(
                "transition.memory_id",
                "ADD transition must have memory_id=None (new item)",
            )

        if not transition.value or not transition.value.strip():
            result.add_error(
                "transition.value",
                "ADD transition must have a non-empty value",
            )

    def _validate_modify(
        self, transition: Transition, state: ProjectState, result: Result
    ) -> None:
        """Validate MODIFY-specific constraints."""
        if transition.memory_id is None:
            result.add_error(
                "transition.memory_id",
                "MODIFY transition must have a non-null memory_id",
            )
            return

        # memory_id must exist and be ACTIVE
        target = self._find_memory(transition.memory_id, state)
        if target is None:
            result.add_error(
                "transition.memory_id",
                (
                    f"MODIFY references memory_id {transition.memory_id!r} "
                    f"not found in state"
                ),
            )
            return

        if target.status != MemoryStatus.ACTIVE:
            result.add_error(
                "transition.memory_id",
                (
                    f"MODIFY references memory_id {transition.memory_id!r} "
                    f"with status {target.status.value}, expected ACTIVE"
                ),
            )
            return

        # old_value must match current value
        if transition.old_value is not None:
            if not target.matches_value(transition.old_value):
                result.add_error(
                    "transition.old_value",
                    (
                        f"old_value {transition.old_value!r} does not match "
                        f"current value {target.value!r} for memory "
                        f"{transition.memory_id!r}"
                    ),
                )

        # value must be non-empty
        if not transition.value or not transition.value.strip():
            result.add_error(
                "transition.value",
                "MODIFY transition must have a non-empty value",
            )

    def _validate_remove(
        self, transition: Transition, state: ProjectState, result: Result
    ) -> None:
        """Validate REMOVE-specific constraints."""
        if transition.memory_id is None:
            result.add_error(
                "transition.memory_id",
                "REMOVE transition must have a non-null memory_id",
            )
            return

        target = self._find_memory(transition.memory_id, state)
        if target is None:
            result.add_error(
                "transition.memory_id",
                (
                    f"REMOVE references memory_id {transition.memory_id!r} "
                    f"not found in state"
                ),
            )
            return

        if target.status != MemoryStatus.ACTIVE:
            result.add_error(
                "transition.memory_id",
                (
                    f"REMOVE references memory_id {transition.memory_id!r} "
                    f"with status {target.status.value}, expected ACTIVE"
                ),
            )

    def _validate_reject(
        self, transition: Transition, state: ProjectState, result: Result
    ) -> None:
        """Validate REJECT-specific constraints."""
        if transition.memory_id is None:
            # REJECT with None memory_id is allowed (rejecting a
            # never-added value).  Validate value instead.
            if not transition.value or not transition.value.strip():
                result.add_error(
                    "transition.value",
                    (
                        "REJECT with memory_id=None must have a "
                        "non-empty value"
                    ),
                )
            return

        target = self._find_memory(transition.memory_id, state)
        if target is None:
            result.add_error(
                "transition.memory_id",
                (
                    f"REJECT references memory_id {transition.memory_id!r} "
                    f"not found in state"
                ),
            )
            return

        if target.status != MemoryStatus.ACTIVE:
            result.add_error(
                "transition.memory_id",
                (
                    f"REJECT references memory_id {transition.memory_id!r} "
                    f"with status {target.status.value}, expected ACTIVE"
                ),
            )

    def _validate_no_change(
        self, transition: Transition, state: ProjectState, result: Result
    ) -> None:
        """Validate NO_CHANGE-specific constraints."""
        if transition.memory_id is None:
            result.add_error(
                "transition.memory_id",
                "NO_CHANGE transition must have a non-null memory_id",
            )
            return

        target = self._find_memory(transition.memory_id, state)
        if target is None:
            result.add_error(
                "transition.memory_id",
                (
                    f"NO_CHANGE references memory_id {transition.memory_id!r} "
                    f"not found in state"
                ),
            )
            return

        if target.status != MemoryStatus.ACTIVE:
            result.add_error(
                "transition.memory_id",
                (
                    f"NO_CHANGE references memory_id {transition.memory_id!r} "
                    f"with status {target.status.value}, expected ACTIVE"
                ),
            )

        # Value must match the current value
        if not target.matches_value(transition.value):
            result.add_error(
                "transition.value",
                (
                    f"NO_CHANGE value {transition.value!r} does not match "
                    f"current value {target.value!r} for memory "
                    f"{transition.memory_id!r}"
                ),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_memory(
        memory_id: str, state: ProjectState
    ) -> MemoryItem | None:
        """Find a memory by ID across all_memories.

        Args:
            memory_id: The ID to search for.
            state:     The current project state.

        Returns:
            The matching :class:`MemoryItem`, or ``None``.
        """
        for item in state.all_memories:
            if item.memory_id == memory_id:
                return item
        return None

    def _check_post_transition_duplicates(
        self, transition: Transition, state: ProjectState, result: Result
    ) -> None:
        """Check for duplicate active memories after a transition.

        Only ADD and MODIFY transitions can introduce duplicates.

        Args:
            transition: The transition being validated.
            state:      The current project state.
            result:     The result to append warnings to.
        """
        if transition.transition_type not in (TransitionType.ADD, TransitionType.MODIFY):
            return

        # Build a hypothetical active set
        active_ids: dict[str, tuple[MemoryCategory, str]] = {}
        for m in state.active_memories:
            key = _normalize_value(m.value)
            active_ids[m.memory_id] = (m.category, key)

        tt = transition.transition_type
        if tt == TransitionType.ADD:
            # The new item would be added as active
            norm_val = _normalize_value(transition.value)
            cat_val = (transition.category, norm_val)
            # Check for duplicates
            for existing_cat, existing_val in active_ids.values():
                if existing_cat == transition.category and existing_val == norm_val:
                    result.add_warning(
                        "transition",
                        (
                            f"ADD would create a duplicate active memory: "
                            f"category={transition.category.value}, "
                            f"value={transition.value!r} already active"
                        ),
                    )
                    return

        elif tt == TransitionType.MODIFY and transition.memory_id:
            # The modified item would have the new value
            norm_val = _normalize_value(transition.value)
            for mid, (cat, val) in active_ids.items():
                if mid == transition.memory_id:
                    continue
                if cat == transition.category and val == norm_val:
                    result.add_warning(
                        "transition",
                        (
                            f"MODIFY would create a duplicate active memory: "
                            f"category={transition.category.value}, "
                            f"value={transition.value!r} already active "
                            f"(memory_id={mid})"
                        ),
                    )
                    return


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def validate_no_duplicates(state: ProjectState) -> List[ValidationError]:
    """Check that no two active memories share category + normalised value.

    Args:
        state: The project state to check.

    Returns:
        A list of :class:`ValidationError` objects.  Empty list means
        no duplicates were found.
    """
    errors: List[ValidationError] = []
    seen: dict[tuple[MemoryCategory, str], List[str]] = {}

    for item in state.active_memories:
        norm_val = _normalize_value(item.value)
        key = (item.category, norm_val)
        if key not in seen:
            seen[key] = []
        seen[key].append(item.memory_id)

    for (cat, val), ids in seen.items():
        if len(ids) > 1:
            errors.append(
                ValidationError(
                    field="active_memories",
                    message=(
                        f"Duplicate active memories: category={cat.value}, "
                        f"normalized_value={val!r}, "
                        f"memory_ids={ids}"
                    ),
                    severity="error",
                )
            )

    return errors


def validate_state_consistency(state: ProjectState) -> List[ValidationError]:
    """Check that the state is internally consistent.

    Checks performed:

    * Every active memory is also in all_memories.
    * Every active memory has status == ACTIVE.
    * No orphaned transition references (memory_id in transition_log
      not present in all_memories).

    Args:
        state: The project state to check.

    Returns:
        A list of :class:`ValidationError` objects.
    """
    errors: List[ValidationError] = []

    all_ids = {m.memory_id: m for m in state.all_memories}

    # -- active_memories must be a subset of all_memories -----------------
    for item in state.active_memories:
        if item.memory_id not in all_ids:
            errors.append(
                ValidationError(
                    field="active_memories",
                    message=(
                        f"Active memory {item.memory_id} not found in "
                        f"all_memories"
                    ),
                    severity="error",
                )
            )
        elif item.status != MemoryStatus.ACTIVE:
            errors.append(
                ValidationError(
                    field="active_memories",
                    message=(
                        f"Active memory {item.memory_id} has status "
                        f"{item.status.value} instead of ACTIVE"
                    ),
                    severity="error",
                )
            )

    # -- transition_log references must exist in all_memories -------------
    for idx, t in enumerate(state.transition_log):
        if t.memory_id is not None and t.memory_id not in all_ids:
            errors.append(
                ValidationError(
                    field=f"transition_log[{idx}].memory_id",
                    message=(
                        f"{t.transition_type.value} at index {idx} "
                        f"references memory_id {t.memory_id!r} not found "
                        f"in all_memories"
                    ),
                    severity="error",
                )
            )

    # -- MODIFY transitions should have old_value set ---------------------
    for idx, t in enumerate(state.transition_log):
        if t.transition_type == TransitionType.MODIFY and t.old_value is None:
            errors.append(
                ValidationError(
                    field=f"transition_log[{idx}].old_value",
                    message=(
                        f"MODIFY at index {idx} has no old_value recorded"
                    ),
                    severity="warning",
                )
            )

    return errors
