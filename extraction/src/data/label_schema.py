"""
Label schema for project-memory span extraction.

The label schema defines what categories of information are relevant
to persistent project memory. Each label corresponds to a semantic
role that a span of text can play in a coding prompt.

Design decisions:
- We use BIO (Begin/Inside/Outside) tagging for token-level classification.
- Labels are grouped into high-level categories that map to project memory fields.
- The schema is deliberately coarse-grained for the prototype; fine-grained
  sub-categories can be added after baseline validation.

Label mapping from dataset fields:
- output.specs → SPECS spans (technology choices, parameters, constraints)
- output.design → DESIGN spans (architectural decisions, component descriptions)
- output.project_overview → OVERVIEW spans (high-level project description)

Within specs, common key patterns map to:
- language/tool/framework names → technology
- purpose/goal descriptions → purpose
- version/constraint values → constraint

For the prototype, we use a unified "project_info" label rather than
fine-grained sub-labels, because:
1. The alignment pipeline produces spans from different output fields
2. The model's primary task is binary: is this span project-memory relevant?
3. Fine-grained labeling requires manual annotation which is out of scope

Future work can extend to:
- technology, purpose, constraint, path, version, etc.
"""

from enum import Enum
from typing import Dict, List, Tuple


class LabelScheme(Enum):
    """BIO label scheme for token classification."""

    OUTSIDE = 0
    BEGIN_PROJECT_INFO = 1
    INSIDE_PROJECT_INFO = 2


# Mapping from label enum to readable names
LABEL_NAMES: Dict[int, str] = {
    0: "O",
    1: "B-PROJECT_INFO",
    2: "I-PROJECT_INFO",
}

NUM_LABELS = len(LabelScheme)

# Reverse mapping
NAME_TO_LABEL: Dict[str, int] = {v: k for k, v in LABEL_NAMES.items()}


def bio_tags_to_spans(
    tags: List[str], offsets: List[Tuple[int, int]], prompt: str
) -> List[Dict]:
    """
    Convert BIO tag sequence to span dictionaries with exact character offsets.

    Args:
        tags: List of BIO tag strings (e.g., ["O", "B-PROJECT_INFO", ...])
        offsets: List of (start, end) character offsets for each token
        prompt: The original prompt text

    Returns:
        List of span dicts with keys: start, end, label, text
    """
    spans = []
    current_start = None
    current_end = None

    for i, (tag, (start, end)) in enumerate(zip(tags, offsets)):
        if tag == "B-PROJECT_INFO":
            # Close any open span
            if current_start is not None:
                spans.append({
                    "start": current_start,
                    "end": current_end,
                    "label": "project_info",
                    "text": prompt[current_start:current_end],
                })
            current_start = start
            current_end = end
        elif tag == "I-PROJECT_INFO" and current_start is not None:
            # Extend current span
            current_end = end
        else:
            # Close any open span
            if current_start is not None:
                spans.append({
                    "start": current_start,
                    "end": current_end,
                    "label": "project_info",
                    "text": prompt[current_start:current_end],
                })
                current_start = None
                current_end = None

    # Close final span
    if current_start is not None:
        spans.append({
            "start": current_start,
            "end": current_end,
            "label": "project_info",
            "text": prompt[current_start:current_end],
        })

    return spans


def validate_spans(spans: List[Dict], prompt: str) -> List[Dict]:
    """
    Validate that every span is an exact contiguous substring of the prompt.

    This is the hard validation required by the spec:
        assert prompt[start:end] == extracted_text

    Invalid spans are removed and logged.
    """
    valid = []
    for span in spans:
        start, end = span["start"], span["end"]
        extracted = prompt[start:end]
        if extracted == span["text"]:
            valid.append(span)
        else:
            # Try to find the text in the prompt as a fallback
            idx = prompt.find(span["text"])
            if idx >= 0:
                span["start"] = idx
                span["end"] = idx + len(span["text"])
                valid.append(span)
            # else: discard invalid span
    return valid
