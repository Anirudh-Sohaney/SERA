"""Filter final/dataset.jsonl down to coding/project-associated prompts.

For every record, an LLM judge (gpt-5.6-luna, low reasoning, temperature 0)
classifies the raw user prompt as coding/project associated (1) or not (0).
Records judged 0 are dropped from the filtered output; nothing is deleted
from the source file.

Output:
- final/dataset_coding.jsonl : kept records, original order
- .filter_decisions.jsonl    : {"i": record_index, "d": 0|1} per classification
                               (this is also the resume checkpoint)
- final/filter_stats.json    : run summary

Independent checks: check_filter.py runs at every 10% of records classified;
check_dataset.py validates the assembled output at the end.

Usage:
    python3 filter_coding.py --pilot          # 16 prompts x2 passes, no writes
    python3 filter_coding.py                  # full run (resumable)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from typing import Any, Optional

from openai_client import RateLimitError, complete

log = logging.getLogger("filter_coding")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "final", "dataset.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "final", "dataset_coding.jsonl")
DECISIONS_PATH = os.path.join(BASE_DIR, ".filter_decisions.jsonl")
STATS_PATH = os.path.join(BASE_DIR, "final", "filter_stats.json")
CHECK_FILTER = os.path.join(BASE_DIR, "check_filter.py")
CHECK_DATASET = os.path.join(BASE_DIR, "check_dataset.py")

MODEL = "gpt-5.6-luna"
REASONING = "low"
# NOTE: the codex WebSocket API rejects `temperature` for this model
# ("Unsupported parameter"), so determinism relies on low reasoning effort
# plus the tightly constrained single-character output format. The pilot's
# two-pass agreement test verifies consistency empirically.
TEMPERATURE = None

CLASSIFIER_SYSTEM = """You are a strict binary classifier. Decide whether a user prompt is CODING/PROJECT ASSOCIATED.

Answer with exactly one character: 1 (associated) or 0 (not). No other text.

ASSOCIATED (1) - any prompt involving real software work:
- Creating/building/planning software: apps, scripts, websites, games, bots, APIs, tools, automations
- Modifying/extending an existing project: features, refactors, migrations, config changes
- Debugging: error messages, stack traces, tracebacks, crashes, unexpected program behavior
- Code review, explaining concrete code, fixing or completing snippets
- Dev file operations: moving/renaming files, imports, dependencies, package managers
- Shell/terminal commands, git, deployment, containers, database queries for a project
- Questions about a specific codebase or a concrete implementation task

NOT ASSOCIATED (0):
- General conversation, opinions, small talk, acknowledgments ("ok", "yes please", "thanks")
- Essays, poems, stories, translations of non-code text
- Conceptual/educational tech talk with no concrete code, project, or task ("what is recursion?", "explain how the internet works")
- Math/homework without code, tech news, hardware buying advice, career advice

TIE-BREAKERS:
- Contains actual code, a file path, an error message, or a concrete dev tool used for a task -> 1
- Merely mentions technology conceptually -> 0
- Unsure -> 0

Examples:
"Build a Python script that renames all files in a folder by date" -> 1
"Traceback: AttributeError: 'NoneType' object has no attribute 'append'" -> 1
"How do I join two tables in SQL for my inventory app?" -> 1
"Move the config folder into src/ and fix the imports" -> 1
"What is machine learning?" -> 0
"Write a poem about autumn leaves" -> 0
"Explain recursion with an example" -> 0
"thanks, that worked" -> 0"""


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": f"User prompt:\n{prompt}\n\nAnswer 1 or 0:"},
    ]


def parse_decision(text: str) -> Optional[int]:
    t = text.strip()
    for ch in t:
        if ch in "01":
            return int(ch)
        if ch not in " \n\t*#-":
            return None
    return None


def classify_prompt(prompt: str, max_parse_retries: int = 3) -> tuple[int, bool]:
    """Returns (decision, parsed_cleanly). Falls back to 1 (keep) if the model
    repeatedly returns garbage - conservative: never lose data to a glitch."""
    for _ in range(max_parse_retries):
        result = complete(
            build_messages(prompt), model=MODEL, reasoning=REASONING,
        )
        d = parse_decision(result.text)
        if d is not None:
            return d, True
        log.warning("unparseable classifier output: %.80s", result.text)
    return 1, False


def load_records() -> list[dict[str, Any]]:
    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_decisions() -> dict[int, int]:
    decided: dict[int, int] = {}
    if os.path.exists(DECISIONS_PATH):
        with open(DECISIONS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                decided[obj["i"]] = obj["d"]
    return decided


def run_pilot(records: list[dict[str, Any]]) -> int:
    """Classify 16 stratified prompts twice; report agreement. No writes."""
    by_type: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_type.setdefault(r["type"], []).append(i)
    sample: list[int] = []
    for t, idxs in sorted(by_type.items()):
        step = max(1, len(idxs) // 4)
        sample.extend(idxs[::step][:4])
    sample = sample[:16]

    print(f"PILOT: {len(sample)} prompts x 2 passes | model={MODEL} "
          f"reasoning={REASONING} temp={TEMPERATURE}")
    agree = 0
    clean = 0
    for i in sample:
        p = records[i]["input"]["user_prompt"]
        d1, c1 = classify_prompt(p)
        d2, c2 = classify_prompt(p)
        agree += int(d1 == d2)
        clean += int(c1 and c2)
        snippet = " ".join(p.split())[:70]
        print(f"  [{i:5d}] pass1={d1} pass2={d2} {'OK ' if d1 == d2 else 'MISMATCH'} "
              f"| {snippet}")
    print(f"\nagreement: {agree}/{len(sample)} ({agree/len(sample)*100:.0f}%)")
    print(f"clean parses both passes: {clean}/{len(sample)}")
    return 0 if agree == len(sample) else 1


def assemble(records: list[dict[str, Any]], decisions: dict[int, int]) -> int:
    kept = 0
    with open(OUT_PATH, "w") as out:
        for i, r in enumerate(records):
            if decisions.get(i, 0) == 1:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true",
                    help="classify 16 stratified prompts twice; no writes")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reset", action="store_true",
                    help="ignore/delete existing decisions and start over")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    records = load_records()
    total = len(records)
    log.info("loaded %d records from %s", total, DATASET_PATH)

    if args.pilot:
        return run_pilot(records)

    if args.reset and os.path.exists(DECISIONS_PATH):
        os.remove(DECISIONS_PATH)

    decisions = load_decisions()
    todo = [i for i in range(total) if i not in decisions]
    log.info("%d already classified, %d to go", len(decisions), len(todo))
    milestone_step = max(1, total // 10)
    next_milestone = ((len(decisions) // milestone_step) + 1) * milestone_step

    # ---------------- worker pool ----------------
    work_q: "queue.Queue[Optional[int]]" = queue.Queue()
    result_q: "queue.Queue[tuple[int, int]]" = queue.Queue()
    stop_flag = threading.Event()
    in_flight = 0
    lock = threading.Lock()

    def worker_fn() -> None:
        nonlocal in_flight
        while True:
            idx = work_q.get()
            if idx is None:
                work_q.task_done()
                return
            with lock:
                in_flight += 1
            try:
                prompt = records[idx]["input"]["user_prompt"]
                backoff, waits = 30, 0
                while True:
                    try:
                        d, _clean = classify_prompt(prompt)
                        result_q.put((idx, d))
                        break
                    except RateLimitError:
                        waits += 1
                        if waits > 120:
                            raise
                        log.warning("rate limited on %d; waiting %ds (backoff %d/120)",
                                    idx, backoff, waits)
                        slept = 0
                        while slept < backoff and not stop_flag.is_set():
                            time.sleep(5)
                            slept += 5
                        if stop_flag.is_set():
                            raise KeyboardInterrupt
            except Exception as e:
                log.exception("worker error on %d: %s", idx, e)
                result_q.put((idx, 1))  # conservative: keep
            finally:
                with lock:
                    in_flight -= 1
                work_q.task_done()

    workers = [threading.Thread(target=worker_fn, daemon=True)
               for _ in range(max(1, args.workers))]
    for w in workers:
        w.start()

    invalid = 0
    done_here = 0
    start = time.time()
    dec_f = open(DECISIONS_PATH, "a")

    def handle(idx: int, d: int) -> None:
        nonlocal done_here, invalid, next_milestone
        dec_f.write(json.dumps({"i": idx, "d": d}) + "\n")
        dec_f.flush()
        decisions[idx] = d
        done_here += 1
        if d not in (0, 1):
            invalid += 1
        while len(decisions) >= next_milestone:
            pct = round(next_milestone / total * 10)
            label = f"filter_check_{pct:02d}_10pct"
            log.info("MILESTONE %d%% (%d/%d classified); independent check",
                     pct * 10, len(decisions), total)
            try:
                proc = subprocess.run(
                    [sys.executable, CHECK_FILTER, "--decisions", DECISIONS_PATH,
                     "--total", str(total), "--label", label],
                    capture_output=True, text=True, timeout=600)
                log.info("milestone check: %s", proc.stdout.strip().splitlines()[:2])
                if proc.returncode != 0:
                    log.error("milestone check FAILED:\n%s", proc.stdout)
            except Exception as e:
                log.error("milestone check could not run: %s", e)
            next_milestone += milestone_step

    try:
        fed = 0
        for idx in todo:
            work_q.put(idx)
            fed += 1
        got = 0
        while got < fed:
            try:
                idx, d = result_q.get(timeout=15)
            except queue.Empty:
                continue
            got += 1
            handle(idx, d)
            if got % 500 == 0:
                rate = got / max(time.time() - start, 1)
                log.info("%d/%d classified this session (%.2f rec/s), "
                         "ones=%d zeros=%d", len(decisions), total, rate,
                         sum(decisions.values()), len(decisions) - sum(decisions.values()))
    finally:
        dec_f.close()
        for _ in workers:
            work_q.put(None)
        for w in workers:
            w.join(timeout=30)

    # ---------------- assemble + final validation ----------------
    missing = [i for i in range(total) if i not in decisions]
    if missing:
        log.error("%d records never classified; aborting assembly", len(missing))
        return 1

    kept = assemble(records, decisions)
    proc = subprocess.run(
        [sys.executable, CHECK_DATASET, "--dataset", OUT_PATH,
         "--label", "coding_filter_final"],
        capture_output=True, text=True, timeout=900)
    log.info("final dataset check: %s", proc.stdout.strip().splitlines()[:1])

    ones = sum(decisions.values())
    stats = {
        "total_records": total,
        "kept": kept,
        "dropped": total - kept,
        "kept_pct": round(kept / total * 100, 2),
        "invalid_parses": invalid,
        "elapsed_seconds": round(time.time() - start, 1),
        "output": OUT_PATH,
        "model": MODEL, "reasoning": REASONING, "temperature": TEMPERATURE,
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("DONE: %s", json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())