"""
Evidence-based admission system for SERA.

Combines multiple independent signals to decide whether a candidate
span should be LOCKED, PENDING, or DISCARDED.

Signals:
    1. Extractor confidence
    2. SpanFilter confidence
    3. Category validity
    4. Existing memory agreement
    5. Cross-turn repetition
    6. Negation/rejection detection
    7. Description-only language detection

Each signal is independently computable and ablatable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.memory.schema import MemoryCategory, MemoryItem, ProjectState
from src.memory.transitions import _infer_category_from_text
from src.memory.validator import _normalize_value


# ---------------------------------------------------------------------------
# Admission decision
# ---------------------------------------------------------------------------

class AdmissionDecision(str, Enum):
    """Possible admission decisions."""
    LOCK = "LOCK"
    PENDING = "PENDING"
    DISCARD = "DISCARD"


@dataclass
class AdmissionResult:
    """Result of evidence-based admission for a single candidate."""
    text: str
    decision: AdmissionDecision
    evidence_score: float
    signals: Dict[str, float]
    reasons: List[str]
    category: Optional[MemoryCategory] = None
    extractor_confidence: float = 0.0
    spanfilter_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Signal extractors (independently ablatable)
# ---------------------------------------------------------------------------

# Negation patterns
_NEGATION_PATTERNS = [
    r"\bnot?\s+use\b",
    r"\bdo\s+not\s+use\b",
    r"\bdon'?t\s+use\b",
    r"\bavoid\b",
    r"\binstead\s+of\b",
    r"\brather\s+than\b",
    r"\bno\s+longer\b",
    r"\bstop\s+using\b",
    r"\bwithout\b",
]

# Description-only patterns (fragments, not requirements)
_DESCRIPTION_PATTERNS = [
    r"^(a|an|the)\s+",
    r"\s+(that|which|who)\s+",
    r"\s+(to|for|of|in|on|at)\s+",
    r"^(create|build|make|write|implement|develop|set up)\b",
    r"^(how|what|where|when|why)\s+",
    r"\?$",
]

# Hypothetical/speculative language patterns
_HYPOTHETICAL_PATTERNS = [
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bcould\b",
    r"\bmight\b",
    r"\bwould\b",
    r"\bif\s+we\b",
    r"\bif\s+I\b",
    r"\bwhen\s+we\b",
    r"\bwhen\s+I\b",
    r"\blet'?s\s+try\b",
    r"\bwhat\s+if\b",
    r"\bsuppose\b",
    r"\bassume\b",
    r"\bimagine\b",
    r"\bthink\s+about\b",
    r"\bconsider\s+using\b",
    r"\bconsider\s+adding\b",
    r"\bmaybe\s+we\s+should\b",
    r"\bperhaps\s+we\s+should\b",
    r"\bwe\s+could\s+try\b",
    r"\bwe\s+might\s+want\b",
    r"\bwe\s+would\s+benefit\b",
]

# Requirement language patterns (strong signals)
_REQUIREMENT_PATTERNS = [
    r"\bmust\b",
    r"\bshould\b",
    r"\buse\s+\w+\b",
    r"\bwith\s+\w+\b",
    r"\binstead\b",
    r"\breplace\b",
]

# Technology categories (things that are usually project-relevant)
_TECH_CATEGORIES = {
    MemoryCategory.LANGUAGE,
    MemoryCategory.FRAMEWORK,
    MemoryCategory.LIBRARY,
    MemoryCategory.DATABASE,
    MemoryCategory.PLATFORM,
    MemoryCategory.DEPLOYMENT,
    MemoryCategory.TOOL,
    MemoryCategory.RUNTIME,
}


def signal_extractor_confidence(
    candidate: Dict[str, Any],
    confidence_threshold: float = 0.5,
) -> Tuple[float, str]:
    """Signal: extractor confidence level.

    Returns (score, reason). Score in [0, 1].
    """
    conf = candidate.get("confidence", 0.0)
    if conf >= 0.95:
        return 1.0, f"high extractor confidence ({conf:.3f})"
    elif conf >= 0.8:
        return 0.8, f"good extractor confidence ({conf:.3f})"
    elif conf >= 0.6:
        return 0.5, f"moderate extractor confidence ({conf:.3f})"
    elif conf >= confidence_threshold:
        return 0.3, f"low extractor confidence ({conf:.3f})"
    else:
        return 0.0, f"below threshold ({conf:.3f})"


def signal_spanfilter_confidence(
    candidate: Dict[str, Any],
) -> Tuple[float, str]:
    """Signal: SpanFilter probability.

    Returns (score, reason). Score in [0, 1].
    """
    prob = candidate.get("stage2_prob", None)
    if prob is None:
        return 0.5, "no SpanFilter score"
    if prob >= 0.9:
        return 1.0, f"high SpanFilter prob ({prob:.3f})"
    elif prob >= 0.7:
        return 0.8, f"good SpanFilter prob ({prob:.3f})"
    elif prob >= 0.5:
        return 0.5, f"moderate SpanFilter prob ({prob:.3f})"
    else:
        return 0.2, f"low SpanFilter prob ({prob:.3f})"


def signal_category_validity(
    candidate: Dict[str, Any],
) -> Tuple[float, str]:
    """Signal: Does the text naturally belong to a project-relevant category?

    Returns (score, reason). Score in [0, 1].
    """
    text = candidate.get("text", "").strip()
    if not text:
        return 0.0, "empty text"

    # Check if category was explicitly assigned
    category = candidate.get("category")
    if category:
        try:
            cat = MemoryCategory(category)
            if cat in _TECH_CATEGORIES:
                return 1.0, f"valid tech category ({cat.value})"
            elif cat in (MemoryCategory.REQUIREMENT, MemoryCategory.CONSTRAINT):
                return 0.7, f"generic category ({cat.value})"
            else:
                return 0.5, f"other category ({cat.value})"
        except (ValueError, KeyError):
            pass

    # Infer category from text
    inferred = _infer_category_from_text(text)
    if inferred in _TECH_CATEGORIES:
        return 0.9, f"inferred tech category ({inferred.value})"
    elif inferred == MemoryCategory.REQUIREMENT:
        return 0.4, "inferred as generic requirement"
    else:
        return 0.3, f"inferred non-tech category ({inferred.value})"


def signal_negation_detection(
    candidate: Dict[str, Any],
    prompt: str,
) -> Tuple[float, str]:
    """Signal: Is this text in a negation/rejection context?

    Returns (score, reason). Score in [0, 1].
    Lower score = more likely negated = should NOT be locked.
    """
    text = candidate.get("text", "").strip()
    if not text:
        return 0.5, "empty text"

    # Check surrounding context in prompt
    start = candidate.get("start", 0)
    end = candidate.get("end", len(prompt))

    # Get a window around the span
    context_start = max(0, start - 50)
    context_end = min(len(prompt), end + 50)
    context = prompt[context_start:context_end].lower()

    for pattern in _NEGATION_PATTERNS:
        if re.search(pattern, context):
            return 0.1, f"negation detected in context"

    # Also check if the text itself contains negation
    text_lower = text.lower()
    for pattern in _NEGATION_PATTERNS:
        if re.search(pattern, text_lower):
            return 0.1, "text itself contains negation"

    return 1.0, "no negation detected"


def signal_description_language(
    candidate: Dict[str, Any],
    prompt: str,
) -> Tuple[float, str]:
    """Signal: Is this text description language rather than a requirement?

    Returns (score, reason). Score in [0, 1].
    Lower score = more likely description = should NOT be locked.
    """
    text = candidate.get("text", "").strip()
    if not text:
        return 0.5, "empty text"

    text_lower = text.lower()

    # Check for description patterns
    for pattern in _DESCRIPTION_PATTERNS:
        if re.search(pattern, text_lower):
            return 0.3, f"description pattern detected"

    # Check for requirement patterns
    for pattern in _REQUIREMENT_PATTERNS:
        if re.search(pattern, text_lower):
            return 0.8, "requirement language detected"

    # Short phrases are more likely requirements
    word_count = len(text.split())
    if word_count <= 3:
        return 0.7, "short phrase (likely requirement)"
    elif word_count <= 6:
        return 0.5, "medium phrase"
    else:
        return 0.3, "long phrase (likely description)"


def signal_repetition(
    candidate: Dict[str, Any],
    memory_state: Optional[ProjectState],
) -> Tuple[float, str]:
    """Signal: Does this candidate match an existing memory?

    Returns (score, reason). Score in [0, 1].
    Higher score = already known = less likely to be false lock.
    """
    text = candidate.get("text", "").strip()
    if not text or not memory_state:
        return 0.5, "no memory state"

    norm_text = _normalize_value(text)

    for mem in memory_state.active_memories:
        if _normalize_value(mem.value) == norm_text:
            return 1.0, f"matches existing memory ({mem.category.value})"

    return 0.0, "no match in existing memory"


def signal_hypothetical_language(
    candidate: Dict[str, Any],
    prompt: str,
) -> Tuple[float, str]:
    """Signal: Is this text in a hypothetical/speculative context?

    Returns (score, reason). Score in [0, 1].
    Lower score = more likely hypothetical = should NOT be locked.
    """
    text = candidate.get("text", "").strip()
    if not text:
        return 0.5, "empty text"

    prompt_lower = prompt.lower()

    # Check for hypothetical patterns in the prompt
    for pattern in _HYPOTHETICAL_PATTERNS:
        if re.search(pattern, prompt_lower):
            return 0.1, f"hypothetical language detected: {pattern}"

    return 1.0, "no hypothetical language detected"


# ---------------------------------------------------------------------------
# Admission policy
# ---------------------------------------------------------------------------

@dataclass
class AdmissionPolicy:
    """Configuration for evidence-based admission.

    Each weight controls how much a signal contributes to the final score.
    Setting weight to 0.0 disables that signal (ablation).
    """
    # Signal weights (set to 0.0 to disable)
    weight_extractor_confidence: float = 1.0
    weight_spanfilter_confidence: float = 1.0
    weight_category_validity: float = 0.5
    weight_negation: float = 1.5
    weight_description: float = 1.0
    weight_repetition: float = 0.3
    weight_hypothetical: float = 2.0  # Strong penalty for hypothetical language

    # Decision thresholds
    lock_threshold: float = 0.7
    pending_threshold: float = 0.4

    # SpanFilter override: if SpanFilter prob < this, always discard
    spanfilter_min: float = 0.0

    # Minimum signals required to LOCK
    min_signals_for_lock: int = 2

    def policy_name(self) -> str:
        """Human-readable policy name."""
        active = []
        if self.weight_extractor_confidence > 0:
            active.append("extractor")
        if self.weight_spanfilter_confidence > 0:
            active.append("spanfilter")
        if self.weight_category_validity > 0:
            active.append("category")
        if self.weight_negation > 0:
            active.append("negation")
        if self.weight_description > 0:
            active.append("description")
        if self.weight_repetition > 0:
            active.append("repetition")
        if self.weight_hypothetical > 0:
            active.append("hypothetical")
        return "+".join(active) if active else "none"


# Predefined policies for ablation
# Policy A: extractor confidence only (lower min_signals since only 1 signal)
POLICY_A = AdmissionPolicy(
    weight_extractor_confidence=1.0,
    weight_spanfilter_confidence=0.0,
    weight_category_validity=0.0,
    weight_negation=0.0,
    weight_description=0.0,
    weight_repetition=0.0,
    min_signals_for_lock=1,
    lock_threshold=0.85,
)

# Policy B: extractor + SpanFilter
POLICY_B = AdmissionPolicy(
    weight_extractor_confidence=1.0,
    weight_spanfilter_confidence=1.0,
    weight_category_validity=0.0,
    weight_negation=0.0,
    weight_description=0.0,
    weight_repetition=0.0,
    min_signals_for_lock=2,
    lock_threshold=0.85,
)

# Policy C: extractor + SpanFilter + category + negation + description
POLICY_C = AdmissionPolicy(
    weight_extractor_confidence=1.0,
    weight_spanfilter_confidence=1.0,
    weight_category_validity=0.5,
    weight_negation=1.5,
    weight_description=1.0,
    weight_repetition=0.0,
    min_signals_for_lock=3,
    lock_threshold=0.80,
)

# Policy D: all signals
POLICY_D = AdmissionPolicy(
    weight_extractor_confidence=1.0,
    weight_spanfilter_confidence=1.0,
    weight_category_validity=0.5,
    weight_negation=1.5,
    weight_description=1.0,
    weight_repetition=0.3,
    min_signals_for_lock=3,
    lock_threshold=0.80,
)


# ---------------------------------------------------------------------------
# Admission engine
# ---------------------------------------------------------------------------

class EvidenceBasedAdmission:
    """Decides whether a candidate span should be LOCKED, PENDING, or DISCARDED.

    Combines multiple independent signals into a single evidence score,
    then applies decision thresholds.
    """

    def __init__(self, policy: Optional[AdmissionPolicy] = None) -> None:
        self._policy = policy or POLICY_D

    @property
    def policy(self) -> AdmissionPolicy:
        return self._policy

    def decide(
        self,
        candidate: Dict[str, Any],
        prompt: str,
        memory_state: Optional[ProjectState] = None,
    ) -> AdmissionResult:
        """Make an admission decision for a single candidate.

        Args:
            candidate:     Dict with text, confidence, start, end, stage2_prob, etc.
            prompt:        The full user prompt (for context analysis).
            memory_state:  Current project memory state (for repetition check).

        Returns:
            AdmissionResult with decision, score, signals, and reasons.
        """
        signals: Dict[str, float] = {}
        reasons: List[str] = []

        # 1. Extractor confidence
        score, reason = signal_extractor_confidence(candidate)
        signals["extractor_confidence"] = score
        reasons.append(reason)

        # 2. SpanFilter confidence
        score, reason = signal_spanfilter_confidence(candidate)
        signals["spanfilter_confidence"] = score
        reasons.append(reason)

        # SpanFilter minimum override
        sf_prob = candidate.get("stage2_prob", None)
        if sf_prob is not None and self._policy.spanfilter_min > 0:
            if sf_prob < self._policy.spanfilter_min:
                return AdmissionResult(
                    text=candidate.get("text", ""),
                    decision=AdmissionDecision.DISCARD,
                    evidence_score=0.0,
                    signals=signals,
                    reasons=[f"SpanFilter prob {sf_prob:.3f} < min {self._policy.spanfilter_min}"],
                    extractor_confidence=candidate.get("confidence", 0.0),
                    spanfilter_confidence=sf_prob,
                )

        # 3. Category validity
        score, reason = signal_category_validity(candidate)
        signals["category_validity"] = score
        reasons.append(reason)

        # 4. Negation detection
        score, reason = signal_negation_detection(candidate, prompt)
        signals["negation"] = score
        reasons.append(reason)

        # 5. Description language
        score, reason = signal_description_language(candidate, prompt)
        signals["description"] = score
        reasons.append(reason)

        # 6. Repetition
        score, reason = signal_repetition(candidate, memory_state)
        signals["repetition"] = score
        reasons.append(reason)

        # 7. Hypothetical language
        score, reason = signal_hypothetical_language(candidate, prompt)
        signals["hypothetical"] = score
        reasons.append(reason)

        # Compute weighted evidence score
        weights = {
            "extractor_confidence": self._policy.weight_extractor_confidence,
            "spanfilter_confidence": self._policy.weight_spanfilter_confidence,
            "category_validity": self._policy.weight_category_validity,
            "negation": self._policy.weight_negation,
            "description": self._policy.weight_description,
            "repetition": self._policy.weight_repetition,
            "hypothetical": self._policy.weight_hypothetical,
        }

        total_weight = 0.0
        weighted_sum = 0.0
        active_signals = 0
        for key, weight in weights.items():
            if weight > 0 and key in signals:
                weighted_sum += signals[key] * weight
                total_weight += weight
                active_signals += 1

        evidence_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Determine admission decision
        if (evidence_score >= self._policy.lock_threshold
                and active_signals >= self._policy.min_signals_for_lock):
            decision = AdmissionDecision.LOCK
        elif evidence_score >= self._policy.pending_threshold:
            decision = AdmissionDecision.PENDING
        else:
            decision = AdmissionDecision.DISCARD

        # Get category
        category = None
        cat_raw = candidate.get("category")
        if cat_raw:
            try:
                category = MemoryCategory(cat_raw)
            except (ValueError, KeyError):
                pass
        if category is None:
            category = _infer_category_from_text(candidate.get("text", ""))

        return AdmissionResult(
            text=candidate.get("text", ""),
            decision=decision,
            evidence_score=evidence_score,
            signals=signals,
            reasons=reasons,
            category=category,
            extractor_confidence=candidate.get("confidence", 0.0),
            spanfilter_confidence=candidate.get("stage2_prob", 0.0),
        )

    def decide_batch(
        self,
        candidates: List[Dict[str, Any]],
        prompt: str,
        memory_state: Optional[ProjectState] = None,
    ) -> List[AdmissionResult]:
        """Make admission decisions for a batch of candidates."""
        return [
            self.decide(c, prompt, memory_state)
            for c in candidates
        ]
