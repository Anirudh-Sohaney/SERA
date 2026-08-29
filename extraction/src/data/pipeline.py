"""
Main data preprocessing pipeline — optimized version.

Loads raw datasets, aligns outputs to inputs, creates splits,
and produces evaluation sets. Optimized for speed on CPU.
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .alignment import (
    AlignmentStats,
    align_record,
    extract_output_values,
)
from .splits import (
    create_concept_holdout,
    create_splits,
    create_unseen_vocabulary_test,
    extract_programming_languages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_dataset(path: str) -> List[Dict]:
    """Load a JSONL dataset file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num}: {e}")
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def fast_duplicate_check(records: List[Dict], sample_size: int = 5000) -> Dict:
    """
    Fast duplicate detection using fingerprinting.
    Only checks a sample for near-duplicates (full check is O(n²)).
    """
    rng = np.random.RandomState(42)

    # Exact duplicates by prompt fingerprint
    fp_groups = defaultdict(list)
    for i, record in enumerate(records):
        prompt = record.get("input", {}).get("user_prompt", "")
        fp = prompt.lower().strip()[:200]  # Truncate for speed
        fp_groups[fp].append(i)

    exact_dupes = {k: v for k, v in fp_groups.items() if len(v) > 1}
    exact_count = sum(len(v) for v in exact_dupes.values())

    # Sample-based near-duplicate check
    sample_indices = rng.choice(len(records), min(sample_size, len(records)), replace=False)
    near_dup_count = 0

    # Build n-gram index for sample
    sample_fps = {}
    for i in sample_indices:
        prompt = records[i].get("input", {}).get("user_prompt", "")
        sample_fps[i] = set(prompt.lower().split())

    # Check for high-overlap pairs in sample
    checked = 0
    for i in range(len(sample_indices)):
        for j in range(i + 1, min(i + 50, len(sample_indices))):
            idx_a, idx_b = sample_indices[i], sample_indices[j]
            if idx_a not in sample_fps or idx_b not in sample_fps:
                continue
            a, b = sample_fps[idx_a], sample_fps[idx_b]
            if not a or not b:
                continue
            overlap = len(a & b) / len(a | b)
            if overlap > 0.85:
                near_dup_count += 1
            checked += 1

    return {
        "exact_duplicate_groups": len(exact_dupes),
        "exact_duplicate_count": exact_count,
        "near_duplicate_sample_count": near_dup_count,
        "near_duplicate_sample_checked": checked,
    }


def run_pipeline(
    oasst_path: str = "data/final/dataset_coding.jsonl",
    am_path: str = "data/final/am_dataset_coding.jsonl",
    output_dir: str = "data/processed",
    seed: int = 42,
) -> Dict:
    """Run the complete data preprocessing pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load datasets
    logger.info("=" * 60)
    logger.info("STEP 1: Loading raw datasets")
    logger.info("=" * 60)

    base_dir = Path(__file__).parent.parent.parent
    oasst_records = load_dataset(str(base_dir / oasst_path))
    am_records = load_dataset(str(base_dir / am_path))

    # Step 2: Filter to extractable types
    logger.info("=" * 60)
    logger.info("STEP 2: Filtering to extractable record types")
    logger.info("=" * 60)

    include_types = {"new", "update"}
    oasst_filtered = [r for r in oasst_records if r.get("type") in include_types]
    am_filtered = [r for r in am_records if r.get("type") in include_types]
    all_records = oasst_filtered + am_filtered
    total = len(all_records)
    logger.info(f"OASST: {len(oasst_filtered)}, AM: {len(am_filtered)}, Combined: {total}")

    # Step 3: Align outputs to inputs
    logger.info("=" * 60)
    logger.info("STEP 3: Aligning outputs to inputs")
    logger.info("=" * 60)

    aggregate_stats = AlignmentStats()
    aligned_records = []

    for i, record in enumerate(all_records):
        results, stats = align_record(record)
        aggregate_stats.total_records += stats.total_records
        aggregate_stats.total_output_values += stats.total_output_values
        aggregate_stats.exact_matches += stats.exact_matches
        aggregate_stats.partial_matches += stats.partial_matches
        aggregate_stats.no_matches += stats.no_matches
        aggregate_stats.ambiguous_matches += stats.ambiguous_matches
        aggregate_stats.total_spans += stats.total_spans
        for field, counts in stats.field_stats.items():
            for cat, count in counts.items():
                aggregate_stats.field_stats[field][cat] += count

        all_spans = []
        for result in results:
            all_spans.extend(result.spans)

        aligned_records.append({
            "record": record,
            "spans": all_spans,
            "alignment_results": [
                {"field": r.field, "value": r.value, "category": r.category}
                for r in results
            ],
        })

        if (i + 1) % 5000 == 0:
            logger.info(f"  Aligned {i + 1}/{total} records")

    alignment_dict = aggregate_stats.to_dict()
    logger.info(f"Alignment: {alignment_dict['exact_matches']} exact, "
                f"{alignment_dict['partial_matches']} partial, "
                f"{alignment_dict['no_matches']} no match, "
                f"{alignment_dict['total_spans']} total spans")

    # Step 4: Create splits
    logger.info("=" * 60)
    logger.info("STEP 4: Creating train/val/test splits")
    logger.info("=" * 60)

    conversation_ids = [r.get("conversation_id", str(i)) for i, r in enumerate(all_records)]
    splits = create_splits(all_records, conversation_ids=conversation_ids, seed=seed)

    # Step 5: Fast duplicate check
    logger.info("=" * 60)
    logger.info("STEP 5: Duplicate detection (fast mode)")
    logger.info("=" * 60)

    dup_info = fast_duplicate_check(all_records)
    logger.info(f"Exact duplicates: {dup_info['exact_duplicate_count']}")
    logger.info(f"Near duplicates (sampled): {dup_info['near_duplicate_sample_count']}")

    # Step 6: Create generalization test sets
    logger.info("=" * 60)
    logger.info("STEP 6: Creating generalization test sets")
    logger.info("=" * 60)

    concept_holdout = create_concept_holdout(all_records, splits, seed=seed)
    unseen_vocab = create_unseen_vocabulary_test(all_records, splits, seed=seed)
    logger.info(f"Concept holdout: {concept_holdout['holdout_size']} records")
    logger.info(f"Unseen vocabulary: {unseen_vocab['unseen_count']} records")

    # Step 7: Language distribution
    logger.info("=" * 60)
    logger.info("STEP 7: Programming language distribution")
    logger.info("=" * 60)

    lang_dist = extract_programming_languages(all_records)
    for lang, indices in sorted(lang_dist.items(), key=lambda x: -len(x[1]))[:15]:
        logger.info(f"  {lang}: {len(indices)} records")

    # Step 8: Save everything
    logger.info("=" * 60)
    logger.info("STEP 8: Saving processed data")
    logger.info("=" * 60)

    # Save aligned records
    with open(output_path / "aligned_records.jsonl", "w") as f:
        for ar in aligned_records:
            f.write(json.dumps(ar) + "\n")

    # Save splits
    with open(output_path / "splits.json", "w") as f:
        json.dump(splits, f)

    # Save concept holdout
    with open(output_path / "concept_holdout.json", "w") as f:
        json.dump(concept_holdout, f, indent=2)

    # Save unseen vocab
    with open(output_path / "unseen_vocabulary.json", "w") as f:
        json.dump(unseen_vocab, f, indent=2)

    # Save stats
    stats = {
        "alignment": alignment_dict,
        "duplicates": dup_info,
        "concept_holdout": {
            "holdout_technologies": concept_holdout["holdout_technologies"],
            "holdout_size": concept_holdout["holdout_size"],
        },
        "unseen_vocabulary": {
            "train_techs_count": len(unseen_vocab["train_techs"]),
            "unseen_count": unseen_vocab["unseen_count"],
        },
        "programming_languages": {
            lang: len(indices) for lang, indices in lang_dist.items()
        },
        "splits": {k: len(v) for k, v in splits.items()},
        "total_records": total,
        "seed": seed,
    }

    with open(output_path / "pipeline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(json.dumps({k: v for k, v in stats.items() if k != "alignment"}, indent=2))

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(seed=args.seed)


if __name__ == "__main__":
    main()
