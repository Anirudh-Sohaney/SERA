"""
Canonical data types for the SERA deterministic project-memory state engine.

This module defines the core data structures that represent persistent
project knowledge extracted from conversation turns. The memory engine
maintains a deterministic, auditable state that can be serialized to JSON
and loaded across sessions.

Architecture:
    MemoryItem      — A single fact extracted from a prompt (e.g. "PostgreSQL").
    Transition      — An immutable record of a state change (ADD, MODIFY, etc.).
    ProjectState    — The full persistent state: all memories + transition log.
    MemoryCandidate — A candidate span extracted by the model, awaiting evaluation.

Invariants:
    - Every MemoryItem has a globally unique memory_id (UUID4).
    - Transitions are append-only; the transition_log is the source of truth.
    - active_memories is always consistent with all_memories (status == ACTIVE).
    - Source text is always preserved verbatim for auditability.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryCategory(str, Enum):
    """Canonical memory categories for project knowledge."""

    PROJECT = "project"
    LANGUAGE = "language"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    DATABASE = "database"
    PLATFORM = "platform"
    DEPLOYMENT = "deployment"
    ARCHITECTURE = "architecture"
    DIRECTORY = "directory"
    FILE = "file"
    INTERFACE = "interface"
    INPUT = "input"
    OUTPUT = "output"
    CONSTRAINT = "constraint"
    REQUIREMENT = "requirement"
    DESIGN = "design"
    TOOL = "tool"
    RUNTIME = "runtime"
    TESTING = "testing"
    CONFIGURATION = "configuration"


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory item."""

    ACTIVE = "active"
    REMOVED = "removed"
    REJECTED = "rejected"


class TransitionType(str, Enum):
    """Type of state transition recorded in the log."""

    ADD = "ADD"
    MODIFY = "MODIFY"
    REMOVE = "REMOVE"
    REJECT = "REJECT"
    NO_CHANGE = "NO_CHANGE"


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryItem:
    """A single piece of persistent project memory.

    Each MemoryItem represents a canonical fact extracted from a conversation
    prompt. It preserves the full source context (original text, character
    offsets, and the complete prompt) for auditability and traceability.

    Attributes:
        memory_id: Globally unique identifier (UUID4, auto-generated).
        category: Semantic category of the memory (e.g. LANGUAGE, DATABASE).
        value: Canonical normalized value (e.g. "PostgreSQL", "FastAPI").
        source_text: The exact substring from the original prompt.
        source_start: Character offset start in the original prompt.
        source_end: Character offset end in the original prompt.
        prompt_text: The full original prompt text.
        status: Current lifecycle status (active, removed, rejected).
        created_turn: Conversation turn when this memory was first added.
        updated_turn: Conversation turn when this memory was last modified.
        confidence: Model confidence for this extraction (0.0–1.0).
        metadata: Arbitrary key-value metadata for extensibility.
    """

    memory_id: str = field(default_factory=_new_id)
    category: MemoryCategory = MemoryCategory.PROJECT
    value: str = ""
    source_text: str = ""
    source_start: int = 0
    source_end: int = 0
    prompt_text: str = ""
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_turn: int = 0
    updated_turn: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Enums are converted to their string values for portability.
        """
        d = asdict(self)
        d["category"] = self.category.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryItem:
        """Deserialize from a dictionary.

        Handles both string and enum values for category and status fields.
        Raises ValueError if required fields are missing or invalid.
        """
        if not d:
            raise ValueError("Cannot deserialize MemoryItem from empty dict")
        try:
            cat = d["category"]
            if isinstance(cat, str):
                cat = MemoryCategory(cat)
            elif not isinstance(cat, MemoryCategory):
                raise ValueError(f"Invalid category type: {type(cat)}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid or missing 'category': {e}")

        try:
            status = d.get("status", MemoryStatus.ACTIVE)
            if isinstance(status, str):
                status = MemoryStatus(status)
            elif not isinstance(status, MemoryStatus):
                raise ValueError(f"Invalid status type: {type(status)}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid 'status': {e}")

        metadata = d.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = dict(metadata) if metadata else {}

        return cls(
            memory_id=d.get("memory_id", _new_id()),
            category=cat,
            value=str(d.get("value", "")),
            source_text=str(d.get("source_text", "")),
            source_start=int(d.get("source_start", 0)),
            source_end=int(d.get("source_end", 0)),
            prompt_text=str(d.get("prompt_text", "")),
            status=status,
            created_turn=int(d.get("created_turn", 0)),
            updated_turn=int(d.get("updated_turn", 0)),
            confidence=float(d.get("confidence", 1.0)),
            metadata=metadata,
        )

    def to_json(self) -> str:
        """Serialize to a JSON string with 2-space indentation."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> MemoryItem:
        """Deserialize from a JSON string.

        Raises json.JSONDecodeError if the string is not valid JSON,
        or ValueError if the resulting dict is invalid.
        """
        if not isinstance(s, str) or not s.strip():
            raise ValueError("Cannot deserialize MemoryItem from empty string")
        d = json.loads(s)
        return cls.from_dict(d)

    def matches_value(self, other_value: str) -> bool:
        """Case-insensitive, stripped comparison of values.

        Returns True if the normalized value matches after stripping
        whitespace and lowering case.
        """
        if not isinstance(other_value, str):
            return False
        return self.value.strip().lower() == other_value.strip().lower()

    def __eq__(self, other: object) -> bool:
        """Equality based on memory_id only."""
        if not isinstance(other, MemoryItem):
            return NotImplemented
        return self.memory_id == other.memory_id

    def __hash__(self) -> int:
        """Hash based on memory_id for use in sets and dicts."""
        return hash(self.memory_id)

    def __repr__(self) -> str:
        return (
            f"MemoryItem(id={self.memory_id[:8]}, "
            f"cat={self.category.value}, "
            f"val={self.value!r}, "
            f"status={self.status.value})"
        )


@dataclass
class Transition:
    """An immutable record of a memory state change.

    Transitions are the audit log of the memory engine. Every ADD, MODIFY,
    REMOVE, or REJECT operation produces exactly one Transition. The
    transition_log is append-only and serves as the source of truth.

    Attributes:
        transition_id: Globally unique identifier (UUID4, auto-generated).
        transition_type: The type of change (ADD, MODIFY, REMOVE, REJECT, NO_CHANGE).
        category: Semantic category of the memory.
        value: The new value (ADD/MODIFY) or the value being removed/rejected.
        old_value: For MODIFY, the previous value. For REMOVE, the value removed.
        memory_id: For MODIFY/REMOVE/REJECT, the affected memory item's ID.
        source_text: The exact substring from the original prompt.
        source_start: Character offset start in the original prompt.
        source_end: Character offset end in the original prompt.
        prompt_text: The full original prompt text.
        turn_number: Conversation turn that produced this transition.
        timestamp: ISO 8601 UTC timestamp of when the transition was recorded.
        confidence: Model confidence for this extraction (0.0–1.0).
        metadata: Arbitrary key-value metadata for extensibility.
    """

    transition_id: str = field(default_factory=_new_id)
    transition_type: TransitionType = TransitionType.ADD
    category: MemoryCategory = MemoryCategory.PROJECT
    value: str = ""
    old_value: Optional[str] = None
    memory_id: Optional[str] = None
    source_text: str = ""
    source_start: int = 0
    source_end: int = 0
    prompt_text: str = ""
    turn_number: int = 0
    timestamp: str = field(default_factory=_now_iso)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Enums are converted to their string values for portability.
        """
        d = asdict(self)
        d["transition_type"] = self.transition_type.value
        d["category"] = self.category.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Transition:
        """Deserialize from a dictionary.

        Handles both string and enum values for enum fields.
        Raises ValueError if required fields are missing or invalid.
        """
        if not d:
            raise ValueError("Cannot deserialize Transition from empty dict")
        try:
            tt = d["transition_type"]
            if isinstance(tt, str):
                tt = TransitionType(tt)
            elif not isinstance(tt, TransitionType):
                raise ValueError(f"Invalid transition_type type: {type(tt)}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid or missing 'transition_type': {e}")

        try:
            cat = d["category"]
            if isinstance(cat, str):
                cat = MemoryCategory(cat)
            elif not isinstance(cat, MemoryCategory):
                raise ValueError(f"Invalid category type: {type(cat)}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid or missing 'category': {e}")

        metadata = d.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = dict(metadata) if metadata else {}

        return cls(
            transition_id=d.get("transition_id", _new_id()),
            transition_type=tt,
            category=cat,
            value=str(d.get("value", "")),
            old_value=d.get("old_value"),
            memory_id=d.get("memory_id"),
            source_text=str(d.get("source_text", "")),
            source_start=int(d.get("source_start", 0)),
            source_end=int(d.get("source_end", 0)),
            prompt_text=str(d.get("prompt_text", "")),
            turn_number=int(d.get("turn_number", 0)),
            timestamp=str(d.get("timestamp", _now_iso())),
            confidence=float(d.get("confidence", 1.0)),
            metadata=metadata,
        )

    def to_json(self) -> str:
        """Serialize to a JSON string with 2-space indentation."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> Transition:
        """Deserialize from a JSON string.

        Raises json.JSONDecodeError if the string is not valid JSON,
        or ValueError if the resulting dict is invalid.
        """
        if not isinstance(s, str) or not s.strip():
            raise ValueError("Cannot deserialize Transition from empty string")
        d = json.loads(s)
        return cls.from_dict(d)

    def __repr__(self) -> str:
        return (
            f"Transition(id={self.transition_id[:8]}, "
            f"type={self.transition_type.value}, "
            f"cat={self.category.value}, "
            f"val={self.value!r}, "
            f"turn={self.turn_number})"
        )


@dataclass
class ProjectState:
    """The full persistent state for a project's memory.

    ProjectState is the root object that holds all memories, the complete
    transition log, and the current turn counter. It supports loading from
    and saving to JSON files, and provides query methods for active memories.

    Invariants:
        - active_memories only contains items with status == ACTIVE.
        - all_memories contains every item ever created (including removed/rejected).
        - transition_log is append-only and records every state change.
        - current_turn is monotonically increasing.

    Attributes:
        project_id: Unique identifier for this project.
        active_memories: Memory items with status == ACTIVE.
        all_memories: Every memory item (audit trail).
        transition_log: Append-only log of all state transitions.
        current_turn: The current conversation turn number.
    """

    project_id: str = field(default_factory=_new_id)
    active_memories: List[MemoryItem] = field(default_factory=list)
    all_memories: List[MemoryItem] = field(default_factory=list)
    transition_log: List[Transition] = field(default_factory=list)
    current_turn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "project_id": self.project_id,
            "active_memories": [m.to_dict() for m in self.active_memories],
            "all_memories": [m.to_dict() for m in self.all_memories],
            "transition_log": [t.to_dict() for t in self.transition_log],
            "current_turn": self.current_turn,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProjectState:
        """Deserialize from a dictionary.

        Raises ValueError if required fields are missing or malformed.
        """
        if not d:
            raise ValueError("Cannot deserialize ProjectState from empty dict")

        try:
            active = [MemoryItem.from_dict(m) for m in d.get("active_memories", [])]
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid active_memories: {e}")

        try:
            all_m = [MemoryItem.from_dict(m) for m in d.get("all_memories", [])]
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid all_memories: {e}")

        try:
            transitions = [
                Transition.from_dict(t) for t in d.get("transition_log", [])
            ]
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid transition_log: {e}")

        return cls(
            project_id=str(d.get("project_id", _new_id())),
            active_memories=active,
            all_memories=all_m,
            transition_log=transitions,
            current_turn=int(d.get("current_turn", 0)),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string with 2-space indentation."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> ProjectState:
        """Deserialize from a JSON string.

        Raises json.JSONDecodeError if the string is not valid JSON,
        or ValueError if the resulting dict is invalid.
        """
        if not isinstance(s, str) or not s.strip():
            raise ValueError("Cannot deserialize ProjectState from empty string")
        d = json.loads(s)
        return cls.from_dict(d)

    def save(self, path: str) -> None:
        """Write the current state to a JSON file.

        Creates parent directories if they don't exist.
        Uses atomic write (write to temp, then rename) for safety.

        Args:
            path: File path to write the JSON to.
        """
        import os
        import tempfile

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=dir_name or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(self.to_json())
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @classmethod
    def load(cls, path: str) -> ProjectState:
        """Load a project state from a JSON file.

        Args:
            path: File path to read the JSON from.

        Returns:
            A new ProjectState instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contents are invalid.
        """
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    def get_active_by_category(self, category: MemoryCategory) -> List[MemoryItem]:
        """Return all active memories in the given category.

        Args:
            category: The category to filter by.

        Returns:
            List of active MemoryItems in that category.
        """
        return [m for m in self.active_memories if m.category == category]

    def get_active_value(self, category: MemoryCategory) -> Optional[str]:
        """Return the single active value for a category, or None.

        If there is exactly one active memory in the category, return its value.
        If there are zero or more than one, return None (ambiguous).

        Args:
            category: The category to query.

        Returns:
            The canonical value string, or None if ambiguous/empty.
        """
        items = self.get_active_by_category(category)
        if len(items) == 1:
            return items[0].value
        return None

    def add_memory(self, item: MemoryItem) -> Transition:
        """Add a new memory item and record the transition.

        If an active memory with the same category and value already exists,
        the existing memory's updated_turn is refreshed and no duplicate is
        created (NO_CHANGE transition).

        Args:
            item: The MemoryItem to add.

        Returns:
            The Transition that was recorded.
        """
        for existing in self.active_memories:
            if (
                existing.category == item.category
                and existing.matches_value(item.value)
            ):
                existing.updated_turn = item.updated_turn
                transition = Transition(
                    transition_type=TransitionType.NO_CHANGE,
                    category=item.category,
                    value=item.value,
                    memory_id=existing.memory_id,
                    source_text=item.source_text,
                    source_start=item.source_start,
                    source_end=item.source_end,
                    prompt_text=item.prompt_text,
                    turn_number=item.updated_turn,
                    confidence=item.confidence,
                )
                self.transition_log.append(transition)
                return transition

        item.status = MemoryStatus.ACTIVE
        item.created_turn = item.updated_turn
        self.active_memories.append(item)
        self.all_memories.append(item)

        transition = Transition(
            transition_type=TransitionType.ADD,
            category=item.category,
            value=item.value,
            memory_id=item.memory_id,
            source_text=item.source_text,
            source_start=item.source_start,
            source_end=item.source_end,
            prompt_text=item.prompt_text,
            turn_number=item.updated_turn,
            confidence=item.confidence,
        )
        self.transition_log.append(transition)
        return transition

    def remove_memory(self, memory_id: str, turn: int = 0) -> Optional[Transition]:
        """Mark a memory as REMOVED and record the transition.

        Args:
            memory_id: The ID of the memory to remove.
            turn: The current conversation turn number.

        Returns:
            The Transition that was recorded, or None if memory_id not found.
        """
        target = None
        for item in self.active_memories:
            if item.memory_id == memory_id:
                target = item
                break

        if target is None:
            return None

        target.status = MemoryStatus.REMOVED
        target.updated_turn = turn
        self.active_memories = [m for m in self.active_memories if m.memory_id != memory_id]

        transition = Transition(
            transition_type=TransitionType.REMOVE,
            category=target.category,
            value=target.value,
            old_value=target.value,
            memory_id=memory_id,
            source_text=target.source_text,
            source_start=target.source_start,
            source_end=target.source_end,
            prompt_text=target.prompt_text,
            turn_number=turn,
        )
        self.transition_log.append(transition)
        return transition

    def reject_memory(self, memory_id: str, turn: int = 0) -> Optional[Transition]:
        """Mark a memory as REJECTED and record the transition.

        Rejection differs from removal in semantics: a rejected memory was
        extracted but determined to be incorrect or irrelevant, while a
        removed memory was previously valid but is no longer current.

        Args:
            memory_id: The ID of the memory to reject.
            turn: The current conversation turn number.

        Returns:
            The Transition that was recorded, or None if memory_id not found.
        """
        target = None
        for item in self.active_memories:
            if item.memory_id == memory_id:
                target = item
                break

        if target is None:
            return None

        target.status = MemoryStatus.REJECTED
        target.updated_turn = turn
        self.active_memories = [m for m in self.active_memories if m.memory_id != memory_id]

        transition = Transition(
            transition_type=TransitionType.REJECT,
            category=target.category,
            value=target.value,
            memory_id=memory_id,
            source_text=target.source_text,
            source_start=target.source_start,
            source_end=target.source_end,
            prompt_text=target.prompt_text,
            turn_number=turn,
        )
        self.transition_log.append(transition)
        return transition

    def modify_memory(
        self,
        memory_id: str,
        new_value: str,
        new_source: str = "",
        new_source_start: int = 0,
        new_source_end: int = 0,
        new_prompt: str = "",
        turn: int = 0,
        confidence: float = 1.0,
    ) -> Optional[Transition]:
        """Update an active memory's value and record the transition.

        Args:
            memory_id: The ID of the memory to modify.
            new_value: The new canonical value.
            new_source: The new source text (optional).
            new_source_start: The new source start offset (optional).
            new_source_end: The new source end offset (optional).
            new_prompt: The new prompt text (optional).
            turn: The current conversation turn number.
            confidence: The confidence for this modification.

        Returns:
            The Transition that was recorded, or None if memory_id not found.
        """
        target = None
        for item in self.active_memories:
            if item.memory_id == memory_id:
                target = item
                break

        if target is None:
            return None

        old_value = target.value
        target.value = new_value
        target.updated_turn = turn
        target.confidence = confidence
        if new_source:
            target.source_text = new_source
        if new_source_start or new_source_end:
            target.source_start = new_source_start
            target.source_end = new_source_end
        if new_prompt:
            target.prompt_text = new_prompt

        transition = Transition(
            transition_type=TransitionType.MODIFY,
            category=target.category,
            value=new_value,
            old_value=old_value,
            memory_id=memory_id,
            source_text=target.source_text,
            source_start=target.source_start,
            source_end=target.source_end,
            prompt_text=target.prompt_text,
            turn_number=turn,
            confidence=confidence,
        )
        self.transition_log.append(transition)
        return transition

    def __repr__(self) -> str:
        return (
            f"ProjectState(project={self.project_id[:8]}, "
            f"active={len(self.active_memories)}, "
            f"total={len(self.all_memories)}, "
            f"transitions={len(self.transition_log)}, "
            f"turn={self.current_turn})"
        )


@dataclass
class MemoryCandidate:
    """An extracted span that needs to be evaluated against the current state.

    MemoryCandidates are produced by the extraction model. They represent
    potential memory items that have not yet been compared to the existing
    ProjectState. The evaluation layer will determine whether each candidate
    should become an ADD, MODIFY, REMOVE, REJECT, or NO_CHANGE transition.

    Attributes:
        text: The extracted text span.
        category: Semantic category of the extraction.
        start: Character offset start in the original prompt.
        end: Character offset end in the original prompt.
        prompt_text: The full original prompt text.
        confidence: Model confidence for this extraction (0.0–1.0).
        turn_number: Conversation turn that produced this extraction.
    """

    text: str = ""
    category: MemoryCategory = MemoryCategory.PROJECT
    start: int = 0
    end: int = 0
    prompt_text: str = ""
    confidence: float = 1.0
    turn_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "text": self.text,
            "category": self.category.value,
            "start": self.start,
            "end": self.end,
            "prompt_text": self.prompt_text,
            "confidence": self.confidence,
            "turn_number": self.turn_number,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryCandidate:
        """Deserialize from a dictionary.

        Raises ValueError if required fields are missing or invalid.
        """
        if not d:
            raise ValueError("Cannot deserialize MemoryCandidate from empty dict")
        try:
            cat = d.get("category", MemoryCategory.PROJECT)
            if isinstance(cat, str):
                cat = MemoryCategory(cat)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid 'category': {e}")

        return cls(
            text=str(d.get("text", "")),
            category=cat,
            start=int(d.get("start", 0)),
            end=int(d.get("end", 0)),
            prompt_text=str(d.get("prompt_text", "")),
            confidence=float(d.get("confidence", 1.0)),
            turn_number=int(d.get("turn_number", 0)),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string with 2-space indentation."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> MemoryCandidate:
        """Deserialize from a JSON string.

        Raises json.JSONDecodeError if the string is not valid JSON,
        or ValueError if the resulting dict is invalid.
        """
        if not isinstance(s, str) or not s.strip():
            raise ValueError("Cannot deserialize MemoryCandidate from empty string")
        d = json.loads(s)
        return cls.from_dict(d)

    def matches_memory(self, item: MemoryItem) -> bool:
        """Check if this candidate matches an existing memory item.

        A candidate matches if it has the same category and a case-insensitive
        equivalent value (stripped, lowered).

        Args:
            item: The MemoryItem to compare against.

        Returns:
            True if the candidate matches the memory item.
        """
        return self.category == item.category and item.matches_value(self.text)

    def __repr__(self) -> str:
        return (
            f"MemoryCandidate(text={self.text!r}, "
            f"cat={self.category.value}, "
            f"conf={self.confidence:.2f})"
        )
