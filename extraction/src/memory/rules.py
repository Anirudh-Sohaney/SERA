"""
Deterministic transition rules.

Examines extracted text, context, and existing state to determine
the correct transition type (ADD/MODIFY/REMOVE/REJECT/NO_CHANGE).

The rule engine is the "decision" layer that sits between the matcher
(which only finds matches) and the transition engine (which applies
changes).  It is designed to be fully deterministic: given the same
inputs it will always produce the same output.

Rules are stored as data structures so that new patterns can be added
without changing control flow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.memory.schema import (
    MemoryCandidate,
    ProjectState,
    TransitionType,
)
from src.memory.matcher import MatchResult


# ---------------------------------------------------------------------------
# Built-in conflict / replacement / negation patterns
# ---------------------------------------------------------------------------

# Each entry: {pattern: str, transition_type: TransitionType, description: str}
# These patterns are matched against the *full prompt text* to detect
# implicit conflict signals (e.g. "instead", "actually", "switch to").

DEFAULT_CONFLICT_PATTERNS: List[Dict[str, Any]] = [
    {
        "pattern": r"\binstead\b",
        "transition_type": TransitionType.MODIFY,
        "description": "Replacement signal: 'instead' implies a new value supersedes the old.",
    },
    {
        "pattern": r"\bactually\b",
        "transition_type": TransitionType.MODIFY,
        "description": "Correction signal: 'actually' implies the speaker is correcting themselves.",
    },
    {
        "pattern": r"\bswitch\s+to\b",
        "transition_type": TransitionType.MODIFY,
        "description": "Replacement signal: 'switch to X' implies leaving the current value.",
    },
    {
        "pattern": r"\breplace\b",
        "transition_type": TransitionType.MODIFY,
        "description": "Replacement signal: explicit replace instruction.",
    },
    {
        "pattern": r"\bchange\s+from\b",
        "transition_type": TransitionType.MODIFY,
        "description": "Modification signal: 'change from Y' implies altering existing state.",
    },
    {
        "pattern": r"\bno\s+longer\b",
        "transition_type": TransitionType.REMOVE,
        "description": "Removal signal: 'no longer using X' implies current value should be removed.",
    },
    {
        "pattern": r"\bremove\b",
        "transition_type": TransitionType.REMOVE,
        "description": "Removal signal: explicit remove instruction.",
    },
    {
        "pattern": r"\bdon'?t\s+use\b",
        "transition_type": TransitionType.REJECT,
        "description": "Rejection signal: 'don't use X' implies the value should not be adopted.",
    },
    {
        "pattern": r"\bdo\s+not\s+use\b",
        "transition_type": TransitionType.REJECT,
        "description": "Rejection signal: formal variant of don't use.",
    },
    {
        "pattern": r"\bavoid\b",
        "transition_type": TransitionType.REJECT,
        "description": "Rejection signal: 'avoid X' implies X should not be part of the state.",
    },
    {
        "pattern": r"\bdrop\b",
        "transition_type": TransitionType.REMOVE,
        "description": "Removal signal: 'drop X' implies removing from active state.",
    },
    {
        "pattern": r"\bdiscard\b",
        "transition_type": TransitionType.REMOVE,
        "description": "Removal signal: 'discard X' implies removal.",
    },
    {
        "pattern": r"\bstop\s+using\b",
        "transition_type": TransitionType.REMOVE,
        "description": "Removal signal: explicit stop instruction.",
    },
    {
        "pattern": r"\binstead\s+of\b",
        "transition_type": TransitionType.MODIFY,
        "description": "Replacement signal: 'instead of Y' implies Y is being replaced.",
    },
    {
        "pattern": r"\bnot\s+required\b",
        "transition_type": TransitionType.REJECT,
        "description": "Rejection signal: 'not required' implies value should not be stored.",
    },
    {
        "pattern": r"\bnot\s+needed\b",
        "transition_type": TransitionType.REMOVE,
        "description": "Removal signal: 'not needed' implies existing value is obsolete.",
    },
    {
        "pattern": r"\bcancelled?\b",
        "transition_type": TransitionType.REMOVE,
        "description": "Removal signal: 'cancelled' implies the value is no longer active.",
    },
    {
        "pattern": r"\bcanceled?\b",
        "transition_type": TransitionType.REMOVE,
        "description": "Removal signal: US spelling variant of cancelled.",
    },
]

# Negation detection patterns — applied to the full prompt with {value}
# substituted in.  Each pattern looks for a negation word in proximity to
# the value.  We use {{0,N}} quantifier to limit distance and avoid
# cross-clause matches (e.g. "do not use Flask. Use Django" should only
# match Flask, not Django).
DEFAULT_NEGATION_PATTERNS: List[str] = [
    r"\bdon'?t\b[^\.\!?\n]{{0,50}}{escaped_value}",
    r"\bdo\s+not\b[^\.\!?\n]{{0,50}}{escaped_value}",
    r"\bnever\b[^\.\!?\n]{{0,50}}{escaped_value}",
    r"\bavoid\b[^\.\!?\n]{{0,50}}{escaped_value}",
    r"\bwithout\b[^\.\!?\n]{{0,50}}{escaped_value}",
]

# Replacement detection patterns — capture groups: <old> and <new>.
DEFAULT_REPLACEMENT_PATTERNS: List[Dict[str, Any]] = [
    {
        "pattern": r"Use\s+(?P<new>.+?)\s+instead\s+of\s+(?P<old>.+)",
        "description": "Pattern: Use <new> instead of <old>",
    },
    {
        "pattern": r"Switch\s+(?:from\s+)?(?P<old>.+?)\s+to\s+(?P<new>.+)",
        "description": "Pattern: Switch [from] <old> to <new>",
    },
    {
        "pattern": r"Replace\s+(?P<old>.+?)\s+with\s+(?P<new>.+)",
        "description": "Pattern: Replace <old> with <new>",
    },
    {
        "pattern": r"use\s+(?P<new>.+?)\s+instead\s+of\s+(?P<old>.+)",
        "description": "Pattern (lowercase): use <new> instead of <old>",
    },
    {
        "pattern": r"switch\s+(?:from\s+)?(?P<old>.+?)\s+to\s+(?P<new>.+)",
        "description": "Pattern (lowercase): switch [from] <old> to <new>",
    },
    {
        "pattern": r"replace\s+(?P<old>.+?)\s+with\s+(?P<new>.+)",
        "description": "Pattern (lowercase): replace <old> with <new>",
    },
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RuleResult:
    """Outcome of applying the rule engine to a single candidate.

    Attributes:
        transition_type: The recommended transition type.
        confidence:      Confidence in this classification (1.0 for
                         deterministic rules, lower for heuristics).
        rule_id:         Identifier of the rule that fired (for auditing).
        reason:          Human-readable explanation.
    """

    transition_type: TransitionType
    confidence: float
    rule_id: str
    reason: str


@dataclass
class ConflictPattern:
    """A single conflict/replacement/negation pattern.

    Attributes:
        pattern:          Regex pattern string (may contain ``{value}``
                          placeholder for negation patterns).
        transition_type:  The transition type implied by this pattern.
        description:      Human-readable description.
    """

    pattern: str
    transition_type: TransitionType
    description: str


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _escape_for_regex(text: str) -> str:
    """Escape a literal string for safe use inside a regex pattern.

    Args:
        text: The literal string to escape.

    Returns:
        Regex-safe escaped string.
    """
    return re.escape(text)


def detect_negation_context(text: str, value: str) -> bool:
    """Check if *value* appears in a negation context within *text*.

    A negation context is any sentence or clause where a negation word
    (don't, do not, no, never, avoid, without) precedes the value.

    Special handling:
        - "no" is only treated as negation if it is followed by a space
          and then the value (to avoid false positives like "NoSQL").
        - "without" is only treated as negation if the value follows it
          directly or within a short span.

    Args:
        text:  The full prompt text.
        value: The value string to search for.

    Returns:
        ``True`` if a negation context is detected.
    """
    if not text or not value:
        return False

    escaped_value = _escape_for_regex(value)

    for pat_template in DEFAULT_NEGATION_PATTERNS:
        pat = pat_template.format(escaped_value=escaped_value)
        if re.search(pat, text, re.IGNORECASE):
            return True

    # Additional careful "no" check: require word boundary after "no"
    # and a space before the value to avoid "NoSQL" matching "no SQL"
    no_pat = r"\bno\b\s+" + escaped_value
    if re.search(no_pat, text, re.IGNORECASE):
        return True

    return False


def detect_replacement_context(text: str) -> Optional[Dict[str, str]]:
    """Detect a replacement pattern in the text.

    Looks for constructions like:
        - "Use X instead of Y"
        - "Switch from Y to X"
        - "Replace Y with X"

    Args:
        text: The full prompt text.

    Returns:
        A dict ``{"old_value": ..., "new_value": ...}`` if a replacement
        pattern is found, otherwise ``None``.
    """
    if not text:
        return None

    for pat_info in DEFAULT_REPLACEMENT_PATTERNS:
        match = re.search(pat_info["pattern"], text, re.IGNORECASE)
        if match:
            old_val = match.group("old").strip()
            new_val = match.group("new").strip()
            if old_val and new_val:
                return {"old_value": old_val, "new_value": new_val}

    return None


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

class TransitionRuleEngine:
    """Deterministic rule engine for transition classification.

    The engine examines a candidate, its match result, the current state,
    and the full prompt to decide which transition type should be applied.

    Rules are stored as lists of pattern dicts for extensibility.  New
    patterns can be appended to ``self.conflict_patterns``,
    ``self.negation_patterns``, or ``self.replacement_patterns`` at
    runtime.

    Args:
        conflict_patterns:   Optional override for conflict patterns.
                             Defaults to ``DEFAULT_CONFLICT_PATTERNS``.
        negation_patterns:   Optional override for negation patterns.
                             Defaults to ``DEFAULT_NEGATION_PATTERNS``.
        replacement_patterns: Optional override for replacement patterns.
                             Defaults to ``DEFAULT_REPLACEMENT_PATTERNS``.
    """

    def __init__(
        self,
        conflict_patterns: Optional[List[Dict[str, Any]]] = None,
        negation_patterns: Optional[List[str]] = None,
        replacement_patterns: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.conflict_patterns: List[Dict[str, Any]] = (
            list(conflict_patterns) if conflict_patterns is not None
            else list(DEFAULT_CONFLICT_PATTERNS)
        )
        self.negation_patterns: List[str] = (
            list(negation_patterns) if negation_patterns is not None
            else list(DEFAULT_NEGATION_PATTERNS)
        )
        self.replacement_patterns: List[Dict[str, Any]] = (
            list(replacement_patterns) if replacement_patterns is not None
            else list(DEFAULT_REPLACEMENT_PATTERNS)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        candidate: MemoryCandidate,
        match_result: MatchResult,
        state: ProjectState,
        full_prompt: str,
    ) -> RuleResult:
        """Classify the transition type for a single candidate.

        The classification logic is:

        1. **No match** → check negation → ADD or REJECT.
        2. **Exact / normalized match** → NO_CHANGE if identical,
           MODIFY if different, REMOVE if negation detected.
        3. **Category-only match** → check conflict patterns, then
           negation, then default to ADD.

        Args:
            candidate:    The extracted memory candidate.
            match_result: The match result from the StateMatcher.
            state:        The current project state.
            full_prompt:  The full prompt text for context analysis.

        Returns:
            A :class:`RuleResult` with the recommended transition.
        """
        match_type = match_result.match_type

        # ----- No match -----
        if match_type == "none":
            return self._classify_no_match(candidate, full_prompt)

        # ----- Exact or normalized match -----
        if match_type in ("exact", "normalized"):
            return self._classify_exact_or_normalized(
                candidate, match_result, full_prompt
            )

        # ----- Category-only (substring) match -----
        if match_type == "category_only":
            return self._classify_category_only(
                candidate, match_result, full_prompt
            )

        # Fallback (should not happen)
        return RuleResult(
            transition_type=TransitionType.ADD,
            confidence=1.0,
            rule_id="fallback_add",
            reason="No matching rule; defaulting to ADD.",
        )

    # ------------------------------------------------------------------
    # Internal classification branches
    # ------------------------------------------------------------------

    def _classify_no_match(
        self, candidate: MemoryCandidate, full_prompt: str
    ) -> RuleResult:
        """Classify a candidate with no match in the current state.

        If the prompt contains a negation for this value, the candidate
        is REJECTED.  If a conflict pattern is detected, treat as ADD
        (the transition engine will handle removing the old value).
        Otherwise it is a new ADD.

        Args:
            candidate:   The candidate to classify.
            full_prompt: The full prompt text.

        Returns:
            A :class:`RuleResult`.
        """
        if detect_negation_context(full_prompt, candidate.text):
            return RuleResult(
                transition_type=TransitionType.REJECT,
                confidence=1.0,
                rule_id="negation_reject",
                reason=(
                    f"Value '{candidate.text}' appears in a negation "
                    f"context in the prompt."
                ),
            )

        # Check for conflict patterns (e.g. "actually", "switch to")
        # If detected, this is a new value replacing an old one
        conflict = self._detect_conflict(full_prompt)
        if conflict is not None:
            return RuleResult(
                transition_type=TransitionType.ADD,
                confidence=0.9,
                rule_id="conflict_new_add",
                reason=(
                    f"Conflict pattern detected: {conflict['description']} "
                    f"Candidate '{candidate.text}' is a new value."
                ),
            )

        return RuleResult(
            transition_type=TransitionType.ADD,
            confidence=1.0,
            rule_id="new_add",
            reason=(
                f"No existing memory matches '{candidate.text}' in "
                f"category '{candidate.category.value}'."
            ),
        )

    def _classify_exact_or_normalized(
        self,
        candidate: MemoryCandidate,
        match_result: MatchResult,
        full_prompt: str,
    ) -> RuleResult:
        """Classify a candidate that has an exact or normalized match.

        If the prompt contains a negation for the existing value → REMOVE.
        If the prompt contains a conflict/removal signal → REMOVE.
        If the candidate text is identical to the existing value → NO_CHANGE.
        If different (but normalized-equal) → MODIFY.

        Args:
            candidate:   The candidate to classify.
            match_result: The match result (must have matched_memory).
            full_prompt: The full prompt text.

        Returns:
            A :class:`RuleResult`.
        """
        memory = match_result.matched_memory
        if memory is None:
            # Defensive: should not happen for exact/normalized matches
            return self._classify_no_match(candidate, full_prompt)

        # Check negation first — if user is removing an existing value
        if detect_negation_context(full_prompt, memory.value):
            return RuleResult(
                transition_type=TransitionType.REMOVE,
                confidence=1.0,
                rule_id="negation_remove_exact",
                reason=(
                    f"Existing memory '{memory.value}' (id={memory.memory_id}) "
                    f"appears in a negation context."
                ),
            )

        # Check conflict patterns (e.g. "Remove X", "Drop X") even for
        # exact matches — the user explicitly wants to remove/change.
        conflict = self._detect_conflict(full_prompt)
        if conflict is not None:
            return RuleResult(
                transition_type=conflict["transition_type"],
                confidence=1.0,
                rule_id=f"conflict_{conflict['pattern']}",
                reason=(
                    f"Conflict pattern '{conflict['pattern']}' detected in "
                    f"prompt. {conflict['description']}"
                ),
            )

        # Exact value match → no change needed
        if candidate.text.strip() == memory.value.strip():
            return RuleResult(
                transition_type=TransitionType.NO_CHANGE,
                confidence=1.0,
                rule_id="exact_no_change",
                reason=(
                    f"Candidate '{candidate.text}' is identical to existing "
                    f"memory '{memory.value}'."
                ),
            )

        # Normalized match but different surface form → modify
        return RuleResult(
            transition_type=TransitionType.MODIFY,
            confidence=0.9,
            rule_id="normalized_modify",
            reason=(
                f"Candidate '{candidate.text}' normalizes to the same form "
                f"as existing memory '{memory.value}' but differs in "
                f"surface text."
            ),
        )

    def _classify_category_only(
        self,
        candidate: MemoryCandidate,
        match_result: MatchResult,
        full_prompt: str,
    ) -> RuleResult:
        """Classify a candidate that matches only by category (substring).

        Logic:
            1. Check for explicit conflict patterns in the prompt.
            2. Check for negation patterns.
            3. If no signals → ADD (new value in same category).

        Args:
            candidate:   The candidate to classify.
            match_result: The match result (must have matched_memory).
            full_prompt: The full prompt text.

        Returns:
            A :class:`RuleResult`.
        """
        memory = match_result.matched_memory

        # Check negation first — might be removing an existing value
        if memory is not None and detect_negation_context(full_prompt, memory.value):
            return RuleResult(
                transition_type=TransitionType.REMOVE,
                confidence=1.0,
                rule_id="negation_remove_category",
                reason=(
                    f"Existing memory '{memory.value}' (id={memory.memory_id}) "
                    f"appears in a negation context."
                ),
            )

        # Check negation for the candidate itself
        if detect_negation_context(full_prompt, candidate.text):
            return RuleResult(
                transition_type=TransitionType.REJECT,
                confidence=1.0,
                rule_id="negation_reject_category",
                reason=(
                    f"Candidate '{candidate.text}' appears in a negation "
                    f"context."
                ),
            )

        # Check conflict patterns
        conflict = self._detect_conflict(full_prompt)
        if conflict is not None:
            return RuleResult(
                transition_type=conflict["transition_type"],
                confidence=1.0,
                rule_id=f"conflict_{conflict['pattern']}",
                reason=(
                    f"Conflict pattern '{conflict['pattern']}' detected in "
                    f"prompt. {conflict['description']}"
                ),
            )

        # No signals → new value in same category
        return RuleResult(
            transition_type=TransitionType.ADD,
            confidence=0.85,
            rule_id="category_new_add",
            reason=(
                f"Candidate '{candidate.text}' shares category "
                f"'{candidate.category.value}' with existing memory "
                f"'{memory.value}' but is a distinct value."
            ),
        )

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def _detect_conflict(self, text: str) -> Optional[Dict[str, Any]]:
        """Scan the prompt for conflict / replacement / removal signals.

        Iterates over ``self.conflict_patterns`` and returns the first
        match found.

        Args:
            text: The full prompt text.

        Returns:
            A dict with ``pattern``, ``transition_type``, and
            ``description`` keys, or ``None`` if no pattern matched.
        """
        for pattern_info in self.conflict_patterns:
            if re.search(pattern_info["pattern"], text, re.IGNORECASE):
                return pattern_info
        return None
