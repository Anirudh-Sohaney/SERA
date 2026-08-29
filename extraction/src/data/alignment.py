"""
Dataset alignment pipeline.

Maps output fields back to exact character offsets in the user prompt.
Produces BIO labels for token-level classification.

Key design decisions:
1. We search for each output value as a substring in the user prompt.
2. Case-insensitive matching is used by default (configurable).
3. Non-extractive outputs (not found in prompt) are recorded but excluded
   from the BIO training signal.
4. Ambiguous matches (multiple occurrences) use the first occurrence.
5. Partial matches are recorded separately for analysis.

Alignment categories:
- exact_match: output value found verbatim in prompt
- partial_match: some overlap between output and prompt
- no_match: output value not found in prompt
- ambiguous_match: output value appears multiple times in prompt
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AlignmentResult:
    """Result of aligning a single output value to the prompt."""

    def __init__(
        self,
        field: str,
        value: str,
        category: str,
        spans: Optional[List[Dict]] = None,
        match_count: int = 0,
    ):
        self.field = field
        self.value = value
        self.category = category  # exact_match, partial_match, no_match, ambiguous_match
        self.spans = spans or []
        self.match_count = match_count


class AlignmentStats:
    """Aggregate statistics for alignment pipeline."""

    def __init__(self):
        self.total_records = 0
        self.total_output_values = 0
        self.exact_matches = 0
        self.partial_matches = 0
        self.no_matches = 0
        self.ambiguous_matches = 0
        self.total_spans = 0
        self.field_stats: Dict[str, Counter] = defaultdict(Counter)
        self.errors: List[str] = []

    def to_dict(self) -> Dict:
        return {
            "total_records": self.total_records,
            "total_output_values": self.total_output_values,
            "exact_matches": self.exact_matches,
            "partial_matches": self.partial_matches,
            "no_matches": self.no_matches,
            "ambiguous_matches": self.ambiguous_matches,
            "total_spans": self.total_spans,
            "field_stats": {k: dict(v) for k, v in self.field_stats.items()},
            "match_rate": (
                self.exact_matches / max(self.total_output_values, 1)
            ),
            "errors": self.errors[:50],  # Keep first 50 errors
        }


def find_substring_offsets(
    text: str,
    substring: str,
    case_sensitive: bool = False,
) -> List[Tuple[int, int]]:
    """
    Find all occurrences of substring in text, returning (start, end) offsets.

    Args:
        text: The prompt text to search in
        substring: The value to find
        case_sensitive: Whether matching is case-sensitive

    Returns:
        List of (start, end) character offset tuples
    """
    if not substring or not text:
        return []

    search_text = text if case_sensitive else text.lower()
    search_sub = substring if case_sensitive else substring.lower()

    offsets = []
    start = 0
    while True:
        idx = search_text.find(search_sub, start)
        if idx == -1:
            break
        offsets.append((idx, idx + len(substring)))
        start = idx + 1

    return offsets


def find_fuzzy_offsets(
    text: str,
    substring: str,
    case_sensitive: bool = False,
    min_overlap: float = 0.6,
) -> List[Tuple[int, int]]:
    """
    Find approximate matches when exact match fails.

    Uses character-level overlap (Jaccard on character n-grams).
    """
    if not substring or len(substring) < 3:
        return []

    # Try sliding window of similar length
    sub_len = len(substring)
    best_offsets = []
    best_score = 0.0

    search_text = text if case_sensitive else text.lower()
    search_sub = substring if case_sensitive else substring.lower()

    # Generate character bigrams for the substring
    sub_bigrams = set()
    for i in range(len(search_sub) - 1):
        sub_bigrams.add(search_sub[i : i + 2])

    for start in range(len(text) - sub_len + 1):
        window = search_text[start : start + sub_len]
        window_bigrams = set()
        for i in range(len(window) - 1):
            window_bigrams.add(window[i : i + 2])

        if not sub_bigrams:
            continue

        overlap = len(sub_bigrams & window_bigrams) / len(sub_bigrams)
        if overlap > best_score and overlap >= min_overlap:
            best_score = overlap
            best_offsets = [(start, start + sub_len)]

    return best_offsets


def extract_output_values(output: Dict) -> List[Tuple[str, str]]:
    """
    Flatten the output dict into (field_name, value_string) pairs.

    Handles:
    - output.project_overview → ("project_overview", value)
    - output.specs.key → ("specs.key", value)
    - output.design[i] → ("design[i]", value)
    """
    values = []

    # project_overview: string
    if output.get("project_overview") and isinstance(output["project_overview"], str):
        values.append(("project_overview", output["project_overview"]))

    # specs: dict of key-value pairs
    specs = output.get("specs")
    if isinstance(specs, dict):
        for key, val in specs.items():
            if val and isinstance(val, str):
                values.append((f"specs.{key}", val))
            elif val and isinstance(val, (int, float)):
                values.append((f"specs.{key}", str(val)))

    # design: list of strings
    design = output.get("design")
    if isinstance(design, list):
        for i, item in enumerate(design):
            if item and isinstance(item, str):
                values.append((f"design[{i}]", item))

    return values


def align_record(
    record: Dict,
    case_sensitive: bool = False,
    min_span_length: int = 2,
    max_span_length: int = 200,
) -> Tuple[List[AlignmentResult], AlignmentStats]:
    """
    Align all output values to the user prompt.

    Returns:
        List of AlignmentResult objects
        Updated AlignmentStats
    """
    stats = AlignmentStats()
    stats.total_records = 1

    prompt = record.get("input", {}).get("user_prompt", "")
    output = record.get("output", {})

    if not prompt:
        stats.errors.append("Empty user_prompt")
        return [], stats

    output_values = extract_output_values(output)
    stats.total_output_values = len(output_values)

    results = []
    for field, value in output_values:
        # Skip very short or very long values
        if len(value) < min_span_length:
            results.append(AlignmentResult(field, value, "too_short"))
            stats.field_stats[field]["too_short"] += 1
            continue

        if len(value) > max_span_length:
            results.append(AlignmentResult(field, value, "too_long"))
            stats.field_stats[field]["too_long"] += 1
            continue

        # Exact match
        offsets = find_substring_offsets(prompt, value, case_sensitive)

        if len(offsets) == 1:
            # Exact single match
            start, end = offsets[0]
            span = {
                "start": start,
                "end": end,
                "label": "project_info",
                "text": prompt[start:end],
                "field": field,
                "source_value": value,
            }
            results.append(AlignmentResult(field, value, "exact_match", [span], 1))
            stats.exact_matches += 1
            stats.total_spans += 1
            stats.field_stats[field]["exact_match"] += 1

        elif len(offsets) > 1:
            # Ambiguous: multiple matches — use first occurrence
            start, end = offsets[0]
            span = {
                "start": start,
                "end": end,
                "label": "project_info",
                "text": prompt[start:end],
                "field": field,
                "source_value": value,
                "occurrence": 1,
                "total_occurrences": len(offsets),
            }
            results.append(
                AlignmentResult(field, value, "ambiguous_match", [span], len(offsets))
            )
            stats.ambiguous_matches += 1
            stats.total_spans += 1
            stats.field_stats[field]["ambiguous_match"] += 1

        else:
            # No exact match — record as non-extractive
            results.append(AlignmentResult(field, value, "no_match"))
            stats.no_matches += 1
            stats.field_stats[field]["no_match"] += 1

    return results, stats


def build_bio_labels(
    prompt: str,
    spans: List[Dict],
    tokenizer,
    max_length: int = 512,
) -> Dict:
    """
    Convert character-level spans to token-level BIO labels.

    Uses the tokenizer to get character-to-token mappings, then assigns
    BIO tags based on span boundaries.

    Returns:
        Dict with keys: input_ids, attention_mask, labels, char_to_token_map
    """
    # Tokenize the prompt
    encoding = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"]

    # Initialize all labels to 0 (O)
    labels = [0] * len(input_ids)

    # Build character-to-token mapping
    # offset_mapping[i] = (char_start, char_end) for token i
    # Special tokens have (0, 0)
    char_to_token = {}
    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == 0 and char_end == 0:
            continue  # Special token
        for c in range(char_start, char_end):
            char_to_token[c] = token_idx

    # Assign BIO labels
    for span in spans:
        span_start = span["start"]
        span_end = span["end"]

        # Find tokens that overlap with this span
        first_token = None
        last_token = None

        for c in range(span_start, min(span_end, len(prompt))):
            if c in char_to_token:
                tok = char_to_token[c]
                if first_token is None:
                    first_token = tok
                last_token = tok

        if first_token is not None:
            labels[first_token] = 1  # B-PROJECT_INFO
            for tok in range(first_token + 1, last_token + 1):
                labels[tok] = 2  # I-PROJECT_INFO

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offset_mapping": offset_mapping,
    }


def build_bilo_labels(
    prompt: str,
    spans: List[Dict],
    tokenizer,
    max_length: int = 512,
) -> Dict:
    """
    Convert character-level spans to token-level BILOU labels.

    BILOU labels:
    - O: Outside any span (0)
    - B: Beginning of a multi-token span (1)
    - I: Inside (continuation) of a multi-token span (2)
    - L: Last token of a multi-token span (3)
    - U: Unit length span (single token) (4)

    Returns:
        Dict with keys: input_ids, attention_mask, labels, offset_mapping
    """
    # Tokenize the prompt
    encoding = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"]

    # Initialize all labels to 0 (O)
    labels = [0] * len(input_ids)

    # Build character-to-token mapping
    char_to_token = {}
    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == 0 and char_end == 0:
            continue  # Special token
        for c in range(char_start, char_end):
            char_to_token[c] = token_idx

    # Assign BILOU labels
    for span in spans:
        span_start = span["start"]
        span_end = span["end"]

        # Find tokens that overlap with this span
        first_token = None
        last_token = None

        for c in range(span_start, min(span_end, len(prompt))):
            if c in char_to_token:
                tok = char_to_token[c]
                if first_token is None:
                    first_token = tok
                last_token = tok

        if first_token is not None:
            if first_token == last_token:
                # Single token span -> U
                labels[first_token] = 4  # U-PROJECT_INFO
            else:
                # Multi-token span -> B ... I ... L
                labels[first_token] = 1  # B-PROJECT_INFO
                for tok in range(first_token + 1, last_token):
                    labels[tok] = 2  # I-PROJECT_INFO
                labels[last_token] = 3  # L-PROJECT_INFO

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offset_mapping": offset_mapping,
    }
