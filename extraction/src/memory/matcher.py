"""
Deterministic state matcher.

Compares extracted memory candidates against existing project state
to find matches. Does NOT decide transitions — only finds matches.

The matcher operates on a best-priority basis: the first matching
strategy that succeeds determines the match_type. Strategies are
evaluated in this order:

    1. Exact match — identical after case/whitespace normalization.
    2. Normalized match — identical after punctuation normalization.
    3. Category-only (substring) match — same category, one is a
       substring of the other.
    4. No match.

All match results carry a confidence score.  Exact and normalized
matches are scored 1.0; category-only matches are scored 0.7.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

from src.memory.schema import MemoryCandidate, MemoryItem, ProjectState


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for comparison purposes.

    Steps:
        1. Unicode NFKD normalization.
        2. Lowercase.
        3. Strip leading/trailing whitespace.
        4. Collapse runs of internal whitespace to a single space.
        5. Strip trailing periods and commas (common punctuation noise).

    Args:
        text: Raw input string.

    Returns:
        A normalized string suitable for deterministic comparison.
    """
    if not text:
        return ""
    # 1. Unicode normalize
    result = unicodedata.normalize("NFKD", text)
    # 2. Lowercase
    result = result.lower()
    # 3. Strip outer whitespace
    result = result.strip()
    # 4. Collapse internal whitespace
    result = re.sub(r"\s+", " ", result)
    # 5. Strip trailing punctuation that is purely cosmetic
    result = result.rstrip(".,;:!?")
    return result


def _normalize_for_punctuation(text: str) -> str:
    """Aggressively normalize text for punctuation-insensitive matching.

    Extends ``normalize_text`` by also stripping common punctuation
    characters that do not affect semantic meaning (hyphens, underscores,
    parentheses, etc.).

    Args:
        text: Raw input string.

    Returns:
        A heavily normalized string.
    """
    base = normalize_text(text)
    if not base:
        return ""
    # Remove common punctuation but keep alphanumerics and spaces
    base = re.sub(r"[\-_/\\(){}[\]<>@#$%^&*+=|~`\"']", "", base)
    # Re-collapse whitespace after removal
    base = re.sub(r"\s+", " ", base).strip()
    return base


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Outcome of comparing a single candidate against the current state.

    Attributes:
        candidate:        The candidate that was evaluated.
        matched_memory:   The best-matching MemoryItem, or ``None`` if no
                          match was found.
        match_type:       One of ``"exact"``, ``"normalized"``,
                          ``"category_only"``, or ``"none"``.
        confidence:       Confidence score for the match (0.0–1.0).
    """

    candidate: MemoryCandidate
    matched_memory: Optional[MemoryItem]
    match_type: str
    confidence: float


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class StateMatcher:
    """Deterministic state matcher.

    Compares a batch of :class:`MemoryCandidate` objects against the
    active memories in a :class:`ProjectState` and returns ranked
    :class:`MatchResult` objects.

    The matcher is stateless beyond the reference to the ``ProjectState``
    it was constructed with; it does not mutate the state.

    Args:
        state: The current project state to match against.
    """

    def __init__(self, state: ProjectState) -> None:
        self._state = state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_matches(self, candidates: List[MemoryCandidate]) -> List[MatchResult]:
        """Find the best match for each candidate against active memories.

        For every candidate the matcher iterates over all *active* memories
        and evaluates match strategies in priority order.  The first
        strategy that succeeds determines the ``match_type``.

        Args:
            candidates: Extracted memory candidates to evaluate.

        Returns:
            A list of :class:`MatchResult` objects, one per candidate,
            in the same order as the input.
        """
        results: List[MatchResult] = []
        for candidate in candidates:
            result = self._match_one(candidate)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal matching pipeline
    # ------------------------------------------------------------------

    def _match_one(self, candidate: MemoryCandidate) -> MatchResult:
        """Evaluate a single candidate against all active memories.

        Strategies are tried in priority order.  The first match wins.
        If no strategy matches, a ``"none"`` result is returned.

        Args:
            candidate: The candidate to evaluate.

        Returns:
            The best :class:`MatchResult` for this candidate.
        """
        best: Optional[MatchResult] = None

        for memory in self._state.active_memories:
            result = self._evaluate_pair(candidate, memory)
            if result is None:
                continue
            if best is None:
                best = result
            else:
                # Prefer higher-priority match (lower index = higher priority)
                if self._priority(result.match_type) < self._priority(best.match_type):
                    best = result
                elif (
                    self._priority(result.match_type) == self._priority(best.match_type)
                    and result.confidence > best.confidence
                ):
                    best = result

        if best is not None:
            return best

        return MatchResult(
            candidate=candidate,
            matched_memory=None,
            match_type="none",
            confidence=0.0,
        )

    def _evaluate_pair(
        self, candidate: MemoryCandidate, memory: MemoryItem
    ) -> Optional[MatchResult]:
        """Evaluate a single (candidate, memory) pair.

        Returns ``None`` if no match strategy succeeds.  Otherwise returns
        the best (highest-priority) match result.

        Args:
            candidate: The candidate to test.
            memory:    The existing memory item to test against.

        Returns:
            A :class:`MatchResult` or ``None``.
        """
        # Categories must match for any positive result
        if candidate.category != memory.category:
            return None

        candidate_text = candidate.text
        memory_value = memory.value

        # --- Strategy 1: Exact match ---
        norm_cand = normalize_text(candidate_text)
        norm_mem = normalize_text(memory_value)
        if norm_cand and norm_mem and norm_cand == norm_mem:
            return MatchResult(
                candidate=candidate,
                matched_memory=memory,
                match_type="exact",
                confidence=1.0,
            )

        # --- Strategy 2: Normalized (punctuation-insensitive) match ---
        punct_cand = _normalize_for_punctuation(candidate_text)
        punct_mem = _normalize_for_punctuation(memory_value)
        if punct_cand and punct_mem and punct_cand == punct_mem:
            return MatchResult(
                candidate=candidate,
                matched_memory=memory,
                match_type="normalized",
                confidence=0.95,
            )

        # --- Strategy 3: Category-only / substring containment ---
        if self._is_substring_match(norm_cand, norm_mem):
            return MatchResult(
                candidate=candidate,
                matched_memory=memory,
                match_type="category_only",
                confidence=0.7,
            )

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_substring_match(norm_candidate: str, norm_memory: str) -> bool:
        """Check if either text is a substring of the other.

        Only meaningful when both strings are non-empty and at least
        a minimum length (3 chars) to avoid trivial single-char matches.

        Args:
            norm_candidate: Normalized candidate text.
            norm_memory:    Normalized memory value.

        Returns:
            ``True`` if a substring relationship exists.
        """
        if not norm_candidate or not norm_memory:
            return False
        # Require minimum length to avoid trivial matches
        if len(norm_candidate) < 3 or len(norm_memory) < 3:
            return False
        return norm_candidate in norm_memory or norm_memory in norm_candidate

    @staticmethod
    def _priority(match_type: str) -> int:
        """Return a numeric priority for a match type (lower = better).

        Args:
            match_type: One of ``"exact"``, ``"normalized"``,
                        ``"category_only"``, ``"none"``.

        Returns:
            Integer priority (0 = best).
        """
        return {
            "exact": 0,
            "normalized": 1,
            "category_only": 2,
            "none": 3,
        }.get(match_type, 3)
