"""
Build hard-negative training data from non-extractive outputs.

Analyzes why outputs are non-extractive and constructs supervised examples.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


TECH_LEXICON = {
    "python": "language", "javascript": "language", "typescript": "language",
    "java": "language", "c++": "language", "c#": "language", "rust": "language",
    "go": "language", "golang": "language", "ruby": "language", "php": "language",
    "swift": "language", "kotlin": "language", "scala": "language",
    "flask": "framework", "django": "framework", "fastapi": "framework",
    "litestar": "framework", "express": "framework", "react": "framework",
    "vue": "framework", "angular": "framework", "svelte": "framework",
    "spring": "framework", "rails": "framework", "laravel": "framework",
    "postgresql": "database", "postgres": "database", "mysql": "database",
    "sqlite": "database", "mongodb": "database", "redis": "database",
    "elasticsearch": "database", "cassandra": "database", "dynamodb": "database",
    "docker": "tool", "kubernetes": "tool", "terraform": "tool",
    "nginx": "tool", "webpack": "tool", "vite": "tool", "pytest": "tool",
}


def detect_tech(prompt: str) -> Dict[str, str]:
    """Detect technologies in prompt."""
    prompt_lower = prompt.lower()
    found = {}
    for tech in sorted(TECH_LEXICON.keys(), key=len, reverse=True):
        if len(tech) <= 2:
            if re.search(r'\b' + re.escape(tech) + r'\b', prompt_lower):
                found[tech] = TECH_LEXICON[tech]
        else:
            if tech in prompt_lower:
                found[tech] = TECH_LEXICON[tech]
    return found


def categorize_non_extractive(output: Dict, prompt: str) -> str:
    """Categorize why a non-extractive output was generated."""
    overview = output.get("project_overview", "")
    specs = output.get("specs", {})
    design = output.get("design", [])

    prompt_lower = prompt.lower()
    overview_lower = overview.lower() if isinstance(overview, str) else ""

    # Check if overview is a summary/paraphrase
    overview_words = set(overview_lower.split()) if overview_lower else set()
    prompt_words = set(prompt_lower.split())
    overlap = len(overview_words & prompt_words) / max(len(overview_words), 1)

    if overlap < 0.3:
        return "paraphrase"
    elif overlap < 0.6:
        return "partial_summary"

    # Check if specs add inferred requirements
    if specs and isinstance(specs, dict):
        for key, val in specs.items():
            if isinstance(val, str) and val.lower() not in prompt_lower:
                return "inferred_requirement"

    # Check if design adds new concepts
    if design and isinstance(design, list):
        for item in design:
            if isinstance(item, str):
                item_words = set(item.lower().split())
                if not (item_words & prompt_words):
                    return "generated_design"

    return "near_match"


def build_hard_negative_data(
    aligned_records: List[Dict],
    output_dir: str = "data/evaluation",
) -> Dict:
    """
    Analyze non-extractive outputs and build training augmentation data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    categories = Counter()
    non_extractive_records = []

    for ar in aligned_records:
        record = ar["record"]
        prompt = record.get("input", {}).get("user_prompt", "")
        output = record.get("output", {})
        spans = ar["spans"]

        # Skip if has extractive spans
        if spans:
            continue

        # Categorize
        cat = categorize_non_extractive(output, prompt)
        categories[cat] += 1

        # Get technologies in prompt
        techs = detect_tech(prompt)

        non_extractive_records.append({
            "prompt": prompt,
            "output_overview": output.get("project_overview", ""),
            "output_specs": output.get("specs", {}),
            "output_design": output.get("design", []),
            "category": cat,
            "technologies": techs,
        })

    # Build contextual negative examples
    # For each non-extractive record, create a "what NOT to extract" signal
    contextual_negatives = []

    rng = np.random.RandomState(42)
    for rec in non_extractive_records:
        prompt = rec["prompt"]
        techs = rec["technologies"]

        if not techs:
            continue

        # The output is a paraphrase/summary, not an exact span
        # So the correct behavior is: extract nothing (or extract what IS exact)
        # This is already handled by the existing BIO labels

        # But we can also generate explicit negatives:
        # "The prompt mentions X, but the output is a summary, not X itself"
        pass

    # Build augmentation examples: syntactic variations of existing prompts
    augmentation_rules = [
        (r"Use (\w+) for the", r"The backend should use \1"),
        (r"Use (\w+) for the", r"Implement the server with \1"),
        (r"Use (\w+) for the", r"Build the API using \1"),
        (r"Create a (\w+) app", r"Build a \1 application"),
        (r"Write a (\w+) script", r"Create a \1 script"),
    ]

    # Find prompts that match patterns
    augmentation_examples = []
    for ar in aligned_records:
        if not ar["spans"]:
            continue
        prompt = ar["record"].get("input", {}).get("user_prompt", "")

        for pattern, replacement in augmentation_rules:
            if re.search(pattern, prompt, re.IGNORECASE):
                try:
                    new_prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
                    augmentation_examples.append({
                        "original": prompt,
                        "augmented": new_prompt,
                        "spans": ar["spans"],
                        "pattern": pattern,
                    })
                except Exception:
                    pass
                break

    # Save analysis
    analysis = {
        "non_extractive_categories": dict(categories),
        "non_extractive_total": len(non_extractive_records),
        "extractive_total": sum(1 for ar in aligned_records if ar["spans"]),
        "augmentation_examples": len(augmentation_examples),
        "sample_non_extractive": non_extractive_records[:10],
    }

    with open(output_path / "non_extractive_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    with open(output_path / "augmentation_examples.json", "w") as f:
        json.dump(augmentation_examples, f, indent=2)

    logger.info(f"Non-extractive categories: {dict(categories)}")
    logger.info(f"Non-extractive total: {len(non_extractive_records)}")
    logger.info(f"Extractive total: {analysis['extractive_total']}")
    logger.info(f"Augmentation examples: {len(augmentation_examples)}")

    return analysis


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    base_dir = Path(__file__).parent.parent.parent
    aligned_records = []
    with open(base_dir / "data/processed/aligned_records.jsonl") as f:
        for line in f:
            aligned_records.append(json.loads(line))

    build_hard_negative_data(aligned_records, str(base_dir / "data/evaluation"))


if __name__ == "__main__":
    main()
