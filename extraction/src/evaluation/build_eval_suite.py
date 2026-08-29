"""
Improved evaluation suite builder — handles low-frequency tech holdout
and builds larger unseen-vocabulary sets.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

logger = logging.getLogger(__name__)

# ── Tech lexicons (more precise matching) ───────────────────────────

TECH_LEXICON = {
    # Languages
    "python": "language", "javascript": "language", "typescript": "language",
    "java": "language", "c++": "language", "c#": "language", "rust": "language",
    "go": "language", "golang": "language", "ruby": "language", "php": "language",
    "swift": "language", "kotlin": "language", "scala": "language",
    "haskell": "language", "elixir": "language", "clojure": "language",
    "julia": "language", "dart": "language", "perl": "language", "lua": "language",
    "r": "language", "sql": "language", "bash": "language", "shell": "language",
    # Frameworks
    "flask": "framework", "django": "framework", "fastapi": "framework",
    "litestar": "framework", "actix": "framework", "axum": "framework",
    "express": "framework", "nextjs": "framework", "nuxt": "framework",
    "react": "framework", "vue": "framework", "vue.js": "framework",
    "angular": "framework", "svelte": "framework", "spring": "framework",
    "rails": "framework", "laravel": "framework", "symfony": "framework",
    "gin": "framework", "echo": "framework", "fiber": "framework",
    "strapi": "framework", "hasura": "framework", "supabase": "framework",
    "firebase": "framework", "tailwind": "framework", "bootstrap": "framework",
    "jquery": "framework", "ember": "framework", "backbone": "framework",
    # Databases
    "postgresql": "database", "postgres": "database", "mysql": "database",
    "sqlite": "database", "mongodb": "database", "mongo": "database",
    "redis": "database", "elasticsearch": "database", "elastic": "database",
    "cassandra": "database", "dynamodb": "database", "mariadb": "database",
    "neo4j": "database", "influxdb": "database", "clickhouse": "database",
    "snowflake": "database", "bigquery": "database", "cockroachdb": "database",
    # Tools
    "docker": "tool", "kubernetes": "tool", "k8s": "tool", "terraform": "tool",
    "ansible": "tool", "jenkins": "tool", "nginx": "tool", "apache": "tool",
    "webpack": "tool", "vite": "tool", "esbuild": "tool", "rollup": "tool",
    "babel": "tool", "eslint": "tool", "pytest": "tool", "jest": "tool",
    "mocha": "tool", "playwright": "tool", "cypress": "tool", "selenium": "tool",
    "git": "tool", "github": "tool", "gitlab": "tool", "postman": "tool",
}


def detect_tech_with_categories(prompt: str) -> Dict[str, str]:
    """Detect technologies with their categories."""
    prompt_lower = prompt.lower()
    found = {}
    # Sort by length descending to match longer names first
    for tech in sorted(TECH_LEXICON.keys(), key=len, reverse=True):
        # Use word boundary matching for short techs
        if len(tech) <= 2:
            pattern = r'\b' + re.escape(tech) + r'\b'
            if re.search(pattern, prompt_lower):
                found[tech] = TECH_LEXICON[tech]
        else:
            if tech in prompt_lower:
                found[tech] = TECH_LEXICON[tech]
    return found


def load_aligned_records(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_splits(path: str) -> Dict[str, List[int]]:
    with open(path) as f:
        return json.load(f)


def build_random_test(aligned_records, splits) -> List[Dict]:
    test_indices = splits.get("test", [])
    test_records = []
    for idx in test_indices:
        if idx >= len(aligned_records):
            continue
        ar = aligned_records[idx]
        prompt = ar["record"].get("input", {}).get("user_prompt", "")
        spans = ar["spans"]
        techs = detect_tech_with_categories(prompt)
        test_records.append({
            "index": idx,
            "prompt": prompt,
            "spans": spans,
            "technologies": techs,
        })
    return test_records


def build_concept_holdout(aligned_records, splits) -> Dict:
    """
    Build concept-holdout: pick technologies with <=3 training mentions
    and test on test-set records containing them.
    """
    train_indices = set(splits.get("train", []))
    test_indices = splits.get("test", [])

    # Tech frequency in training
    train_tech_freq: Counter = Counter()
    for idx in train_indices:
        if idx >= len(aligned_records):
            continue
        ar = aligned_records[idx]
        prompt = ar["record"].get("input", {}).get("user_prompt", "")
        for tech in detect_tech_with_categories(prompt):
            train_tech_freq[tech] += 1

    # Find low-frequency techs that appear in test
    test_tech_freq: Counter = Counter()
    for idx in test_indices:
        if idx >= len(aligned_records):
            continue
        ar = aligned_records[idx]
        prompt = ar["record"].get("input", {}).get("user_prompt", "")
        for tech in detect_tech_with_categories(prompt):
            test_tech_freq[tech] += 1

    # Select holdout techs: appear in test, <=3 in train
    # Prefer techs with decent test representation
    holdout_candidates = []
    for tech, test_count in test_tech_freq.most_common(100):
        train_count = train_tech_freq.get(tech, 0)
        if train_count <= 3 and test_count >= 2:
            holdout_candidates.append((tech, train_count, test_count))

    # Take top candidates across categories
    holdout_techs = set()
    categories_seen = set()
    for tech, train_count, test_count in holdout_candidates:
        cat = TECH_LEXICON.get(tech, "unknown")
        if cat not in categories_seen or len(holdout_techs) < 8:
            holdout_techs.add(tech)
            categories_seen.add(cat)
        if len(holdout_techs) >= 10:
            break

    # Get test records with holdout techs
    holdout_records = []
    for idx in test_indices:
        if idx >= len(aligned_records):
            continue
        ar = aligned_records[idx]
        prompt = ar["record"].get("input", {}).get("user_prompt", "")
        spans = ar["spans"]
        techs = detect_tech_with_categories(prompt)
        matching = {t for t in techs if t in holdout_techs}
        if matching:
            holdout_records.append({
                "index": idx,
                "prompt": prompt,
                "spans": spans,
                "technologies": techs,
                "holdout_techs": list(matching),
            })

    # Get clean training indices (no holdout techs)
    clean_train = []
    excluded = 0
    for idx in train_indices:
        if idx >= len(aligned_records):
            continue
        ar = aligned_records[idx]
        prompt = ar["record"].get("input", {}).get("user_prompt", "")
        techs = detect_tech_with_categories(prompt)
        if set(techs.keys()) & holdout_techs:
            excluded += 1
        else:
            clean_train.append(idx)

    return {
        "holdout_technologies": list(holdout_techs),
        "holdout_test_records": holdout_records,
        "holdout_test_count": len(holdout_records),
        "clean_train_indices": clean_train,
        "clean_train_count": len(clean_train),
        "excluded_train_count": excluded,
    }


def build_context_reversal(aligned_records, splits, max_neg=200) -> Dict:
    test_indices = splits.get("test", [])
    positive_records = []

    for idx in test_indices:
        if idx >= len(aligned_records):
            continue
        ar = aligned_records[idx]
        prompt = ar["record"].get("input", {}).get("user_prompt", "")
        spans = ar["spans"]
        techs = detect_tech_with_categories(prompt)
        if not techs:
            continue
        positive_records.append({
            "index": idx,
            "prompt": prompt,
            "spans": spans,
            "technologies": techs,
            "is_positive": True,
        })

    # Build negative examples from real prompts with synthetic negation
    negative_templates = [
        "{tech} was considered but rejected.",
        "Do not use {tech}.",
        "{tech} is no longer required.",
        "{tech} was previously used.",
        "{tech} could work, but we chose {alt}.",
        "Replace {tech} with {alt}.",
        "We tried {tech} but switched to {alt}.",
        "{tech} didn't work out.",
        "Initially used {tech}, but moved to {alt}.",
    ]

    all_techs = sorted(set(TECH_LEXICON.keys()))
    rng = np.random.RandomState(42)
    negative_records = []

    for tech in all_techs:
        if len(negative_records) >= max_neg:
            break
        # Find alternative of same category
        cat = TECH_LEXICON.get(tech)
        alts = [t for t, c in TECH_LEXICON.items() if c == cat and t != tech]
        if not alts:
            continue
        alt = rng.choice(alts)

        for template in negative_templates[:3]:
            try:
                text = template.format(tech=tech, alt=alt)
            except KeyError:
                continue
            negative_records.append({
                "prompt": text,
                "spans": [],
                "technologies": {tech: cat},
                "is_positive": False,
                "target_tech": tech,
            })
            if len(negative_records) >= max_neg:
                break

    return {
        "positive_examples": positive_records,
        "negative_examples": negative_records,
        "positive_count": len(positive_records),
        "negative_count": len(negative_records),
    }


def build_cross_language(aligned_records, splits, min_samples=15):
    test_indices = splits.get("test", [])
    lang_groups = defaultdict(list)

    for idx in test_indices:
        if idx >= len(aligned_records):
            continue
        ar = aligned_records[idx]
        prompt = ar["record"].get("input", {}).get("user_prompt", "")
        spans = ar["spans"]
        specs = ar["record"].get("output", {}).get("specs", {})

        lang = None
        if isinstance(specs, dict) and specs.get("language"):
            lang = specs["language"].lower().strip()
        else:
            for l in ["python", "javascript", "typescript", "java", "c++",
                       "c#", "rust", "go", "golang", "ruby", "php", "swift",
                       "kotlin", "scala", "haskell", "elixir", "bash", "sql",
                       "html", "css", "r"]:
                if re.search(r'\b' + re.escape(l) + r'\b', prompt.lower()):
                    lang = l
                    break

        if lang:
            lang_groups[lang].append({
                "index": idx,
                "prompt": prompt,
                "spans": spans,
                "language": lang,
            })

    return {
        lang: recs for lang, recs in lang_groups.items()
        if len(recs) >= min_samples
    }


def build_unseen_vocabulary(aligned_records, splits, freq_threshold=3):
    train_indices = set(splits.get("train", []))
    val_indices = set(splits.get("validation", []))
    test_indices = splits.get("test", [])

    # Count in train+val
    tech_freq = Counter()
    for idx in train_indices | val_indices:
        if idx >= len(aligned_records):
            continue
        ar = aligned_records[idx]
        prompt = ar["record"].get("input", {}).get("user_prompt", "")
        for tech in detect_tech_with_categories(prompt):
            tech_freq[tech] += 1

    # Unseen = appears <= freq_threshold in train+val
    unseen_techs = {t for t, c in tech_freq.items() if c <= freq_threshold}

    unseen_records = []
    seen_records = []
    for idx in test_indices:
        if idx >= len(aligned_records):
            continue
        ar = aligned_records[idx]
        prompt = ar["record"].get("input", {}).get("user_prompt", "")
        spans = ar["spans"]
        techs = detect_tech_with_categories(prompt)
        matching = {t for t in techs if t in unseen_techs}

        entry = {
            "index": idx,
            "prompt": prompt,
            "spans": spans,
            "technologies": techs,
            "unseen_techs": list(matching),
            "is_unseen": bool(matching),
        }
        if matching:
            unseen_records.append(entry)
        else:
            seen_records.append(entry)

    return {
        "unseen_records": unseen_records,
        "seen_records": seen_records,
        "unseen_count": len(unseen_records),
        "seen_count": len(seen_records),
        "unseen_techs": list(unseen_techs),
        "unseen_tech_count": len(unseen_techs),
    }


def build_full_evaluation(aligned_records, splits, output_dir="data/evaluation"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Building random test set...")
    random_test = build_random_test(aligned_records, splits)

    logger.info("Building concept holdout...")
    concept_holdout = build_concept_holdout(aligned_records, splits)

    logger.info("Building context reversal...")
    context_reversal = build_context_reversal(aligned_records, splits)

    logger.info("Building cross-language...")
    cross_language = build_cross_language(aligned_records, splits)

    logger.info("Building unseen vocabulary...")
    unseen_vocab = build_unseen_vocabulary(aligned_records, splits)

    eval_data = {
        "random_test": {"count": len(random_test), "examples": random_test},
        "concept_holdout": {
            "holdout_technologies": concept_holdout["holdout_technologies"],
            "holdout_test_count": concept_holdout["holdout_test_count"],
            "clean_train_count": concept_holdout["clean_train_count"],
            "excluded_train_count": concept_holdout["excluded_train_count"],
            "clean_train_indices": concept_holdout["clean_train_indices"],
            "examples": concept_holdout["holdout_test_records"],
        },
        "context_reversal": {
            "positive_count": context_reversal["positive_count"],
            "negative_count": context_reversal["negative_count"],
            "positive_examples": context_reversal["positive_examples"],
            "negative_examples": context_reversal["negative_examples"],
        },
        "cross_language": {
            lang: {"count": len(recs), "examples": recs}
            for lang, recs in cross_language.items()
        },
        "unseen_vocabulary": {
            "unseen_count": unseen_vocab["unseen_count"],
            "seen_count": unseen_vocab["seen_count"],
            "unseen_tech_count": unseen_vocab["unseen_tech_count"],
            "unseen_techs": unseen_vocab["unseen_techs"],
            "unseen_examples": unseen_vocab["unseen_records"],
            "seen_examples": unseen_vocab["seen_records"],
        },
    }

    with open(output_path / "evaluation_suite.json", "w") as f:
        json.dump(eval_data, f, indent=2)

    with open(output_path / "concept_holdout_train_indices.json", "w") as f:
        json.dump(concept_holdout["clean_train_indices"], f)

    summary = {
        "random_test": len(random_test),
        "concept_holdout": {
            "holdout_techs": concept_holdout["holdout_technologies"],
            "test_count": concept_holdout["holdout_test_count"],
            "excluded_train": concept_holdout["excluded_train_count"],
            "clean_train": concept_holdout["clean_train_count"],
        },
        "context_reversal": {
            "positive": context_reversal["positive_count"],
            "negative": context_reversal["negative_count"],
        },
        "cross_language": {
            lang: len(recs) for lang, recs in cross_language.items()
        },
        "unseen_vocabulary": {
            "unseen_count": unseen_vocab["unseen_count"],
            "seen_count": unseen_vocab["seen_count"],
            "unseen_techs": len(unseen_vocab["unseen_techs"]),
        },
    }

    with open(output_path / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(json.dumps(summary, indent=2))
    return eval_data


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    base_dir = Path(__file__).parent.parent.parent
    aligned_records = load_aligned_records(str(base_dir / "data/processed/aligned_records.jsonl"))
    splits = load_splits(str(base_dir / "data/processed/splits.json"))
    build_full_evaluation(aligned_records, splits, str(base_dir / "data/evaluation"))


if __name__ == "__main__":
    main()
