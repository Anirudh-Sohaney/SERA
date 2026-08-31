"""
Build targeted training set for SERA Stage-1.

Creates hard negatives, negation pairs, and description-vs-technology
examples to address the dominant error categories identified in
candidate analysis.

Usage:
    python -m src.data.build_targeted
"""

from __future__ import annotations

import json
import os
import re
import sys
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_offsets(text: str, start: int, end: int) -> bool:
    """Check that start/end are valid character offsets."""
    if start < 0 or end > len(text):
        return False
    if start >= end:
        return False
    return True


def _validate_exact_substring(text: str, span_text: str, start: int, end: int) -> bool:
    """Check that span text matches the prompt at the given offsets."""
    return text[start:end] == span_text


def _detect_template_family(text: str, seen_families: Counter) -> bool:
    """Check if text belongs to an over-represented template family."""
    # Simple template detection: normalize numbers and specific names
    normalized = re.sub(r'\d+', 'N', text.lower())
    normalized = re.sub(r'\b(python|java|javascript|rust|go|flask|django|postgres|mysql)\b', 'LANG', normalized)
    return seen_families[normalized] > 10


def _compute_hash(text: str, spans: List[Dict]) -> str:
    """Compute a hash for deduplication."""
    content = text.strip().lower() + json.dumps(sorted([s.get("text", "") for s in spans]))
    return hashlib.md5(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Hard negative extraction
# ---------------------------------------------------------------------------

def extract_hard_negatives(
    records: List[Dict],
    split_indices: List[int],
    max_examples: int = 1000,
) -> List[Dict]:
    """Extract hard negatives from existing data (0-span records).

    These are prompts where the gold standard says NOTHING should be
    extracted. The model currently extracts from these — they are
    valuable training signal.
    """
    hard_negatives = []

    for idx in split_indices:
        rec = records[idx]
        spans = rec.get("spans", [])
        if len(spans) > 0:
            continue  # Skip records with spans

        prompt = rec.get("record", {}).get("input", {}).get("user_prompt", "")
        if not prompt or len(prompt.strip()) < 10:
            continue

        hard_negatives.append({
            "prompt": prompt,
            "spans": [],
            "source": "existing_zero_span",
            "error_category": "description_language",
            "line_index": idx,
        })

        if len(hard_negatives) >= max_examples:
            break

    logger.info(f"Extracted {len(hard_negatives)} hard negatives from existing data")
    return hard_negatives


# ---------------------------------------------------------------------------
# Description vs Technology distinction
# ---------------------------------------------------------------------------

# Patterns that indicate technology names (should be extracted)
_TECH_PATTERNS = [
    r"\b(python|java|javascript|typescript|rust|go|ruby|php|c\+\+|c#|swift|kotlin)\b",
    r"\b(flask|django|fastapi|express|react|vue|angular|svelte|next\.?js)\b",
    r"\b(postgresql|mysql|sqlite|mongodb|redis|elasticsearch|cassandra)\b",
    r"\b(aws|gcp|azure|docker|kubernetes|terraform|ansible)\b",
    r"\b(node\.?js|deno|bun)\b",
    r"\b(html|css|sql|json|yaml|toml|xml|graphql)\b",
    r"\b(numpy|pandas|scikit-learn|tensorflow|pytorch)\b",
    r"\b(git|github|gitlab|bitbucket)\b",
    r"\b(nginx|apache|caddy)\b",
    r"\b(postman|insomnia|curl)\b",
]

# Patterns that indicate generic descriptions (should NOT be extracted)
_DESC_PATTERNS = [
    r"^(a|an|the)\s+",
    r"\s+(that|which|who)\s+",
    r"^(create|build|make|write|implement|develop|set up|design)\b",
    r"^(how|what|where|when|why)\s+",
    r"\?$",
    r"^(good|bad|nice|great|better|worse|fast|slow|simple|easy|complex)\b",
    r"^(web|mobile|desktop|server|client)\s+(app|application|service|backend|frontend)\b",
    r"^(the|my|our|your)\s+(project|app|application|service|system|code)\b",
]


def _contains_tech(text: str) -> Optional[str]:
    """Check if text contains a technology name. Returns the tech if found."""
    text_lower = text.lower()
    for pattern in _TECH_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    return None


def _is_description(text: str) -> bool:
    """Check if text is a generic description."""
    text_lower = text.lower()
    for pattern in _DESC_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def build_description_negatives(
    records: List[Dict],
    split_indices: List[int],
    max_examples: int = 500,
) -> List[Dict]:
    """Build examples where description language should NOT be extracted.

    Takes existing records and identifies cases where the prompt contains
    description language that the model incorrectly extracts.
    """
    examples = []
    seen_hashes = set()

    for idx in split_indices:
        rec = records[idx]
        prompt = rec.get("record", {}).get("input", {}).get("user_prompt", "")
        spans = rec.get("spans", [])

        if not prompt:
            continue

        # Check if prompt contains description language
        if not _is_description(prompt):
            continue

        # Check if there are spans that are description-like
        desc_spans = []
        for span in spans:
            text = span.get("text", "")
            if _is_description(text) and not _contains_tech(text):
                desc_spans.append(span)

        if not desc_spans:
            continue

        # Create a version with description spans REMOVED (hard negative)
        non_desc_spans = [s for s in spans if s not in desc_spans]

        h = _compute_hash(prompt, [])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        examples.append({
            "prompt": prompt,
            "spans": non_desc_spans,  # Only keep non-description spans
            "removed_spans": desc_spans,
            "source": "description_negative",
            "error_category": "description_language",
            "line_index": idx,
        })

        if len(examples) >= max_examples:
            break

    logger.info(f"Built {len(examples)} description negative examples")
    return examples


# ---------------------------------------------------------------------------
# Negation pairs
# ---------------------------------------------------------------------------

_NEGATION_TRANSFORMS = [
    (r"\buse\s+(\w+)", r"do not use \1"),
    (r"\buse\s+(\w+)\s+for\b", r"do not use \1 for"),
    (r"\bwith\s+(\w+)\b", r"without \1"),
]


def build_negation_pairs(
    records: List[Dict],
    split_indices: List[int],
    max_examples: int = 500,
) -> List[Dict]:
    """Build negation pairs from existing records.

    Takes records with technology spans and creates negated versions
    where those technologies should NOT be extracted.
    """
    examples = []

    for idx in split_indices:
        rec = records[idx]
        prompt = rec.get("record", {}).get("input", {}).get("user_prompt", "")
        spans = rec.get("spans", [])

        if not prompt or not spans:
            continue

        # Find technology spans
        tech_spans = []
        for span in spans:
            text = span.get("text", "")
            if _contains_tech(text):
                tech_spans.append(span)

        if not tech_spans:
            continue

        # Create negated prompt
        negated_prompt = prompt
        for span in tech_spans:
            text = span.get("text", "")
            # Try each negation transform
            for pattern, replacement in _NEGATION_TRANSFORMS:
                new_prompt = re.sub(pattern, replacement, negated_prompt, count=1, flags=re.IGNORECASE)
                if new_prompt != negated_prompt:
                    negated_prompt = new_prompt
                    break

        if negated_prompt == prompt:
            # Fallback: prepend "Do not use"
            negated_prompt = f"Do not use {prompt}"

        # The negated version should have NO extracted spans for the negated tech
        remaining_spans = [s for s in spans if s not in tech_spans]

        examples.append({
            "prompt": negated_prompt,
            "spans": remaining_spans,
            "negated_spans": tech_spans,
            "source": "negation_pair",
            "error_category": "negation_detection",
            "line_index": idx,
        })

        if len(examples) >= max_examples:
            break

    logger.info(f"Built {len(examples)} negation pair examples")
    return examples


# ---------------------------------------------------------------------------
# Boundary-sensitive examples
# ---------------------------------------------------------------------------

def build_boundary_examples(
    records: List[Dict],
    split_indices: List[int],
    max_examples: int = 300,
) -> List[Dict]:
    """Extract examples where exact span boundaries matter.

    Identifies cases where the gold span is a substring or superstring
    of what the model typically extracts.
    """
    examples = []

    for idx in split_indices:
        rec = records[idx]
        prompt = rec.get("record", {}).get("input", {}).get("user_prompt", "")
        spans = rec.get("spans", [])

        if not prompt or not spans:
            continue

        # Find spans with multi-word technology mentions
        for span in spans:
            text = span.get("text", "")
            words = text.split()

            # Look for "X Y" patterns where X is a technology
            if len(words) >= 2:
                first_word = words[0].lower()
                if _contains_tech(first_word):
                    examples.append({
                        "prompt": prompt,
                        "spans": spans,
                        "boundary_span": span,
                        "boundary_type": "multi_word_technology",
                        "source": "boundary_example",
                        "error_category": "boundary_detection",
                        "line_index": idx,
                    })
                    break  # One example per conversation

        if len(examples) >= max_examples:
            break

    logger.info(f"Built {len(examples)} boundary examples")
    return examples


# ---------------------------------------------------------------------------
# Main: Build targeted dataset
# ---------------------------------------------------------------------------

def build_targeted_dataset(
    output_dir: str,
    max_hard_negatives: int = 1000,
    max_desc_negatives: int = 500,
    max_negation_pairs: int = 500,
    max_boundary: int = 300,
) -> Dict[str, Any]:
    """Build the complete targeted training set."""
    os.makedirs(output_dir, exist_ok=True)

    data_dir = PROJECT_ROOT / "data" / "processed"
    aligned_path = str(data_dir / "aligned_records.jsonl")
    splits_path = str(data_dir / "splits.json")

    # Load data
    logger.info("Loading data...")
    with open(splits_path) as f:
        splits = json.load(f)
    with open(aligned_path) as f:
        all_records = [json.loads(line) for line in f]

    train_indices = splits["train"]
    logger.info(f"Training set: {len(train_indices)} records")

    # Build each category
    hard_negatives = extract_hard_negatives(all_records, train_indices, max_hard_negatives)
    desc_negatives = build_description_negatives(all_records, train_indices, max_desc_negatives)
    negation_pairs = build_negation_pairs(all_records, train_indices, max_negation_pairs)
    boundary_examples = build_boundary_examples(all_records, train_indices, max_boundary)

    # Validate all examples
    all_examples = []
    validation_errors = 0

    for example in hard_negatives + desc_negatives + negation_pairs + boundary_examples:
        prompt = example["prompt"]
        spans = example.get("spans", [])

        # Validate offsets
        valid = True
        for span in spans:
            start = span.get("start", 0)
            end = span.get("end", 0)
            text = span.get("text", "")

            if not _validate_offsets(prompt, start, end):
                valid = False
                break
            if not _validate_exact_substring(prompt, text, start, end):
                valid = False
                break

        if valid:
            all_examples.append(example)
        else:
            validation_errors += 1

    logger.info(f"Validation: {len(all_examples)} valid, {validation_errors} errors")

    # Check for duplicates
    seen_hashes = set()
    unique_examples = []
    duplicates = 0
    for ex in all_examples:
        h = _compute_hash(ex["prompt"], ex.get("spans", []))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_examples.append(ex)
        else:
            duplicates += 1

    logger.info(f"Deduplication: {len(unique_examples)} unique, {duplicates} duplicates")

    # Check template families
    family_counter = Counter()
    for ex in unique_examples:
        normalized = re.sub(r'\d+', 'N', ex["prompt"].lower())
        family_counter[normalized] += 1
    over_represented = sum(1 for v in family_counter.values() if v > 10)
    logger.info(f"Template families: {len(family_counter)} unique, {over_represented} over-represented (>10)")

    # Save
    output_path = os.path.join(output_dir, "targeted_training_data.jsonl")
    with open(output_path, "w") as f:
        for ex in unique_examples:
            f.write(json.dumps(ex) + "\n")

    # Save stats
    stats = {
        "total_examples": len(unique_examples),
        "hard_negatives": len(hard_negatives),
        "description_negatives": len(desc_negatives),
        "negation_pairs": len(negation_pairs),
        "boundary_examples": len(boundary_examples),
        "validation_errors": validation_errors,
        "duplicates_removed": duplicates,
        "category_distribution": dict(Counter(ex.get("source", "unknown") for ex in unique_examples)),
    }

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved {len(unique_examples)} examples to {output_path}")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build targeted training set")
    parser.add_argument("--output-dir", type=str, default="data/targeted")
    parser.add_argument("--max-hard-negatives", type=int, default=1000)
    parser.add_argument("--max-desc-negatives", type=int, default=500)
    parser.add_argument("--max-negation-pairs", type=int, default=500)
    parser.add_argument("--max-boundary", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = str(PROJECT_ROOT / args.output_dir)
    stats = build_targeted_dataset(
        output_dir,
        max_hard_negatives=args.max_hard_negatives,
        max_desc_negatives=args.max_desc_negatives,
        max_negation_pairs=args.max_negation_pairs,
        max_boundary=args.max_boundary,
    )

    print(f"\nTargeted dataset built:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Hard negatives: {stats['hard_negatives']}")
    print(f"  Description negatives: {stats['description_negatives']}")
    print(f"  Negation pairs: {stats['negation_pairs']}")
    print(f"  Boundary examples: {stats['boundary_examples']}")
    print(f"  Validation errors: {stats['validation_errors']}")
    print(f"  Duplicates removed: {stats['duplicates_removed']}")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    main()
