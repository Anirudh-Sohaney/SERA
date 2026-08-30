"""SERA deterministic project-memory state engine.

This package provides the canonical data types and state management for
persistent project knowledge. The memory engine maintains an auditable,
serializable state that tracks every fact extracted from conversation turns.

Core types:
    MemoryCategory  — Enum of memory categories (LANGUAGE, DATABASE, etc.)
    MemoryStatus    — Lifecycle status (active, removed, rejected)
    TransitionType  — State change types (ADD, MODIFY, REMOVE, REJECT, NO_CHANGE)
    MemoryItem      — A single fact with source provenance
    Transition      — An immutable record of a state change
    ProjectState    — Full persistent state with query and mutation methods
    MemoryCandidate — A candidate extraction awaiting evaluation

Usage::

    from src.memory import ProjectState, MemoryItem, MemoryCategory

    state = ProjectState()
    item = MemoryItem(
        category=MemoryCategory.DATABASE,
        value="PostgreSQL",
        source_text="PostgreSQL database",
        source_start=0,
        source_end=18,
        prompt_text="I need a PostgreSQL database for my project",
        created_turn=1,
        updated_turn=1,
    )
    transition = state.add_memory(item)
"""

from src.memory.schema import (
    MemoryCandidate,
    MemoryCategory,
    MemoryItem,
    MemoryStatus,
    ProjectState,
    Transition,
    TransitionType,
)

__all__ = [
    "MemoryCandidate",
    "MemoryCategory",
    "MemoryItem",
    "MemoryStatus",
    "ProjectState",
    "Transition",
    "TransitionType",
]
