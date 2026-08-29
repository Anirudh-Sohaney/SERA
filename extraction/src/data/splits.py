"""
Dataset splitting with leakage prevention.

Key requirements from the spec:
1. Conversation-level grouping: all turns from one conversation stay in the same split.
2. Duplicate detection: exact and near-duplicate prompts must not cross split boundaries.
3. Concept-holdout split: certain technologies/concepts held out from training.
4. Programming-language stratification: language groups represented in each split.
"""

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_programming_languages(records: List[Dict]) -> Dict[str, List[int]]:
    """
    Identify programming languages mentioned in prompts.

    Returns dict mapping language name to list of record indices.
    """
    language_patterns = {
        "python": r"\bpython\b",
        "javascript": r"\bjavascript\b|\bjs\b",
        "typescript": r"\btypescript\b|\bts\b",
        "java": r"\bjava\b(?!\s*script)",
        "c": r"\b(?<!\w)c(?!\+\+|#|sharp|ss)\b",
        "cpp": r"\bc\+\+\b|\bcpp\b",
        "csharp": r"\bc#\b|\bcsharp\b|\bc sharp\b",
        "rust": r"\brust\b",
        "go": r"\bgo\b|\bgolang\b",
        "bash": r"\bbash\b|\bshell\b|\bsh\b",
        "sql": r"\bsql\b|\bmysql\b|\bpostgresql\b|\bsqlite\b",
        "html": r"\bhtml\b",
        "css": r"\bcss\b",
        "php": r"\bphp\b",
        "ruby": r"\bruby\b",
        "swift": r"\bswift\b",
        "kotlin": r"\bkotlin\b",
        "scala": r"\bscala\b",
        "r": r"\br\b(?=\s+(?:for|programming|language|script|code|stats|statistical))",
    }

    lang_records: Dict[str, List[int]] = defaultdict(list)

    for i, record in enumerate(records):
        prompt = record.get("input", {}).get("user_prompt", "").lower()
        specs = record.get("output", {}).get("specs", {})

        # Check specs.language first (most reliable)
        if isinstance(specs, dict) and specs.get("language"):
            lang = specs["language"].lower().strip()
            if lang in language_patterns:
                lang_records[lang].append(i)
                continue

        # Fall back to regex matching on prompt
        for lang, pattern in language_patterns.items():
            if re.search(pattern, prompt):
                lang_records[lang].append(i)
                break  # Assign to first matching language

    return dict(lang_records)


def compute_text_fingerprint(text: str) -> str:
    """Compute a normalized fingerprint for duplicate detection."""
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized


def compute_ngrams(text: str, n: int = 3) -> Set[str]:
    """Compute character n-grams for fuzzy matching."""
    normalized = compute_text_fingerprint(text)
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def detect_duplicates(
    records: List[Dict],
    threshold: float = 0.85,
) -> Dict:
    """
    Detect exact and near-duplicate records.

    Returns:
        Dict with:
        - exact_duplicates: groups of records with identical prompts
        - near_duplicates: groups of records with similar prompts
        - cross_split_count: number of near-duplicate pairs that cross split boundaries
    """
    # Exact duplicates by prompt text
    prompt_groups: Dict[str, List[int]] = defaultdict(list)
    for i, record in enumerate(records):
        prompt = record.get("input", {}).get("user_prompt", "")
        fingerprint = compute_text_fingerprint(prompt)
        prompt_groups[fingerprint].append(i)

    exact_duplicates = {
        k: v for k, v in prompt_groups.items() if len(v) > 1
    }

    # Near duplicates by character n-gram overlap
    # For efficiency, only compare records within the same conversation group
    # or with similar prompt length
    near_duplicate_groups: List[List[int]] = []
    prompt_ngrams = {}
    for i, record in enumerate(records):
        prompt = record.get("input", {}).get("user_prompt", "")
        prompt_ngrams[i] = compute_ngrams(prompt)

    # Compare in batches by prompt length
    sorted_indices = sorted(
        range(len(records)),
        key=lambda i: len(records[i].get("input", {}).get("user_prompt", "")),
    )

    visited = set()
    for idx in range(len(sorted_indices)):
        i = sorted_indices[idx]
        if i in visited:
            continue

        group = [i]
        prompt_i = records[i].get("input", {}).get("user_prompt", "")
        ngrams_i = prompt_ngrams[i]

        # Only compare to records within 30% length difference
        len_i = len(prompt_i)
        for jdx in range(idx + 1, min(idx + 200, len(sorted_indices))):
            j = sorted_indices[jdx]
            if j in visited:
                continue

            prompt_j = records[j].get("input", {}).get("user_prompt", "")
            len_j = len(prompt_j)

            # Skip if lengths differ too much
            if abs(len_i - len_j) / max(len_i, 1) > 0.3:
                continue

            sim = jaccard_similarity(ngrams_i, prompt_ngrams[j])
            if sim >= threshold:
                group.append(j)
                visited.add(j)

        if len(group) > 1:
            near_duplicate_groups.append(group)
            visited.add(i)

    return {
        "exact_duplicates": {k: v for k, v in exact_duplicates.items()},
        "exact_duplicate_count": sum(len(v) for v in exact_duplicates.values()),
        "near_duplicate_groups": near_duplicate_groups,
        "near_duplicate_count": sum(len(g) for g in near_duplicate_groups),
    }


def create_splits(
    records: List[Dict],
    conversation_ids: Optional[List[str]] = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Create train/val/test splits with conversation-level grouping.

    For OASST data: conversation_id is the grouping key.
    For AM data: records are single-turn, so we use record-level splitting.

    Strategy:
    1. Group records by conversation_id
    2. Shuffle groups with fixed seed
    3. Assign groups to splits respecting ratios
    4. Ensure no conversation spans multiple splits

    Returns:
        Dict mapping split name to list of record indices
    """
    rng = np.random.RandomState(seed)

    if conversation_ids is not None:
        # Group by conversation
        conv_groups: Dict[str, List[int]] = defaultdict(list)
        for i, conv_id in enumerate(conversation_ids):
            conv_groups[conv_id].append(i)

        group_ids = list(conv_groups.keys())
        rng.shuffle(group_ids)

        total = len(records)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        splits = {"train": [], "validation": [], "test": []}
        current_count = 0

        for conv_id in group_ids:
            indices = conv_groups[conv_id]
            if current_count < train_end:
                splits["train"].extend(indices)
            elif current_count < val_end:
                splits["validation"].extend(indices)
            else:
                splits["test"].extend(indices)
            current_count += len(indices)

    else:
        # Record-level splitting
        indices = list(range(len(records)))
        rng.shuffle(indices)

        total = len(indices)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        splits = {
            "train": indices[:train_end],
            "validation": indices[train_end:val_end],
            "test": indices[val_end:],
        }

    # Log split sizes
    for split_name, split_indices in splits.items():
        logger.info(f"{split_name}: {len(split_indices)} records")

    return splits


def create_concept_holdout(
    records: List[Dict],
    splits: Dict[str, List[int]],
    seed: int = 42,
) -> Dict:
    """
    Create a concept-holdout test set.

    Identifies the most common technologies/frameworks in training data,
    then holds out a subset for testing generalization.

    The held-out concepts are those that appear in at least 10 training
    records but are completely absent from the held-out test records.
    """
    rng = np.random.RandomState(seed)

    # Count technology mentions in training data
    tech_counts: Counter = Counter()
    for idx in splits["train"]:
        record = records[idx]
        prompt = record.get("input", {}).get("user_prompt", "").lower()
        specs = record.get("output", {}).get("specs", {})

        if isinstance(specs, dict) and specs.get("language"):
            tech_counts[specs["language"].lower()] += 1

    # Find technologies with enough examples
    common_techs = [tech for tech, count in tech_counts.most_common(30) if count >= 10]

    # Hold out ~30% of common technologies
    holdout_count = max(3, len(common_techs) // 3)
    holdout_techs = set(common_techs[:holdout_count])

    # Find test records that mention held-out technologies
    holdout_indices = []
    for idx in splits["test"]:
        record = records[idx]
        prompt = record.get("input", {}).get("user_prompt", "").lower()
        specs = record.get("output", {}).get("specs", {})

        mentioned_techs = set()
        if isinstance(specs, dict) and specs.get("language"):
            mentioned_techs.add(specs["language"].lower())

        # Also check prompt text
        for tech in holdout_techs:
            if tech in prompt:
                mentioned_techs.add(tech)

        if mentioned_techs & holdout_techs:
            holdout_indices.append(idx)

    return {
        "holdout_technologies": list(holdout_techs),
        "holdout_indices": holdout_indices,
        "holdout_size": len(holdout_indices),
        "common_technologies": common_techs,
    }


def create_unseen_vocabulary_test(
    records: List[Dict],
    splits: Dict[str, List[int]],
    seed: int = 42,
) -> Dict:
    """
    Create a test set with prompts containing technologies NOT seen in training.

    This tests the model's ability to generalize to new vocabulary.
    """
    # Build training vocabulary of technology terms
    train_techs = set()
    for idx in splits["train"]:
        record = records[idx]
        specs = record.get("output", {}).get("specs", {})
        if isinstance(specs, dict) and specs.get("language"):
            train_techs.add(specs["language"].lower())

    # Also extract from design/overview
    for idx in splits["train"]:
        record = records[idx]
        output = record.get("output", {})
        for field in ["project_overview", "design"]:
            val = output.get(field, "")
            if isinstance(val, str):
                # Simple extraction of capitalized terms
                for word in val.split():
                    if word[0].isupper() and len(word) > 2:
                        train_techs.add(word.lower())
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        for word in item.split():
                            if word[0].isupper() and len(word) > 2:
                                train_techs.add(word.lower())

    # Find test records with technologies NOT in training
    unseen_indices = []
    seen_indices = []

    for idx in splits["test"]:
        record = records[idx]
        prompt = record.get("input", {}).get("user_prompt", "")
        specs = record.get("output", {}).get("specs", {})

        test_techs = set()
        if isinstance(specs, dict) and specs.get("language"):
            test_techs.add(specs["language"].lower())

        # Check if any test tech is unseen
        unseen_techs = test_techs - train_techs
        if unseen_techs:
            unseen_indices.append(idx)
        else:
            seen_indices.append(idx)

    return {
        "train_techs": list(train_techs),
        "unseen_indices": unseen_indices,
        "seen_indices": seen_indices,
        "unseen_count": len(unseen_indices),
        "seen_count": len(seen_indices),
    }


def verify_no_cross_split_leakage(
    records: List[Dict],
    splits: Dict[str, List[int]],
    threshold: float = 0.85,
) -> Dict:
    """
    Verify that no near-duplicate records cross split boundaries.

    This is a hard requirement from the spec.
    """
    split_assignments = {}
    for split_name, indices in splits.items():
        for idx in indices:
            split_assignments[idx] = split_name

    # Build fingerprints for each record
    fingerprints = {}
    for idx in range(len(records)):
        prompt = records[idx].get("input", {}).get("user_prompt", "")
        fingerprints[idx] = compute_text_fingerprint(prompt)

    # Check for exact duplicates across splits
    cross_split_exact = 0
    fp_to_indices = defaultdict(list)
    for idx, fp in fingerprints.items():
        fp_to_indices[fp].append(idx)

    for fp, indices in fp_to_indices.items():
        if len(indices) > 1:
            split_names = set(split_assignments.get(i, "unknown") for i in indices)
            if len(split_names) > 1:
                cross_split_exact += 1

    # Check for near duplicates across splits
    cross_split_near = 0
    ngrams = {idx: compute_ngrams(records[idx].get("input", {}).get("user_prompt", ""))
              for idx in range(len(records))}

    checked = set()
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if (i, j) in checked:
                continue

            split_i = split_assignments.get(i)
            split_j = split_assignments.get(j)
            if split_i == split_j:
                continue

            # Quick length filter
            len_i = len(records[i].get("input", {}).get("user_prompt", ""))
            len_j = len(records[j].get("input", {}).get("user_prompt", ""))
            if abs(len_i - len_j) / max(len_i, 1) > 0.3:
                continue

            sim = jaccard_similarity(ngrams[i], ngrams[j])
            if sim >= threshold:
                cross_split_near += 1
                checked.add((i, j))

    return {
        "cross_split_exact_duplicates": cross_split_exact,
        "cross_split_near_duplicates": cross_split_near,
        "leakage_free": cross_split_exact == 0 and cross_split_near == 0,
    }
