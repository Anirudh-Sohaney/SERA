"""Main data generation script.

Iterates the processed conversations, and for every user prompt in every
conversation, calls gpt-5.6-luna (medium reasoning) to extract / update the
project memory (project_overview, specs, design) per guide.md.

Procedure:
- For each conversation: reset project state to empty.
- For each user prompt in the conversation:
    - input  = current project state + user prompt
    - output = model's updated project state
    - record {conversation_id, turn, type, input, output}
    - carry the output forward as the new current state.
- State resets after every conversation.

Concurrency: conversations are processed in parallel by --workers threads
(state carry-forward stays sequential within each conversation).

Milestone checks: at every 10% of the total output values, an independent
validation (check_dataset.py) is run against the output file so errors or
failures are caught early.

Rate-limit safety: the account's codex usage endpoint is probed periodically;
when the weekly limit is reached the run stops gracefully (checkpointed) so
the account is never blocked.

Usage:
    python3 generate_dataset.py --max-conversations 20
    python3 generate_dataset.py --project-only --max-conversations 200
    python3 generate_dataset.py --conversation-id 00a5b5a5-9a28-42bc-847d-65bb12b38255:2
    python3 generate_dataset.py --resume --project-only
    python3 generate_dataset.py --workers 6 --keep-no-project   # full run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from typing import Any, Optional

import extraction
from openai_client import RateLimitError, check_rate_limit, complete

log = logging.getLogger("generate_dataset")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERSATIONS_PATH = os.path.join(
    BASE_DIR, "processed", "conversations", "oasst2_english_conversations.jsonl"
)
FINAL_DIR = os.path.join(BASE_DIR, "final")
DATASET_PATH = os.path.join(FINAL_DIR, "dataset.jsonl")
STATS_PATH = os.path.join(FINAL_DIR, "dataset_stats.json")
CHECKPOINT_PATH = os.path.join(BASE_DIR, ".generation_checkpoint.json")
CHECK_SCRIPT = os.path.join(BASE_DIR, "check_dataset.py")

# Keywords that suggest a user message is about building/modifying software.
PROJECT_KEYWORDS = re.compile(
    r"\b(build|create|write|develop|make|implement|design|modify|update|fix|"
    r"add|extend|refactor|convert|migrate|deploy|package|test|debug)\b.*\b("
    r"app|application|script|program|software|tool|utility|website|web|site|"
    r"bot|game|api|service|system|tracker|backend|frontend|database|server|"
    r"client|module|library|framework|cli|command|plugin|extension|crawler|"
    r"scraper|parser|generator|editor|dashboard|python|javascript|typescript|"
    r"java|rust|golang|sql|docker|linux|windows|macos|code|function|class|"
    r"endpoint|interface|schema|query|script|shell|bash|batch|config|"
    r"automation|workflow|pipeline|monitor|notifier|scraper|analyzer|"
    r"compiler|interpreter|simulator|emulator|renderer|engine)\b",
    re.IGNORECASE,
)


def looks_project_related(text: str) -> bool:
    return bool(PROJECT_KEYWORDS.search(text))


def count_total_output_values() -> int:
    """Total user prompts across all conversations == total output values."""
    total = 0
    with open(CONVERSATIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                conv = json.loads(line)
                total += sum(
                    1 for m in conv.get("messages", []) if m.get("role") == "user"
                )
    return total


def load_checkpoint() -> dict[str, Any]:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"processed": [], "records_written": 0}


def save_checkpoint(cp: dict[str, Any]) -> None:
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cp, f, indent=2)
    os.replace(tmp, CHECKPOINT_PATH)


def iter_conversations():
    with open(CONVERSATIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def process_conversation(
    conv: dict[str, Any],
    keep_no_project: bool,
    max_turns: Optional[int],
    delay: float,
) -> tuple[list[dict[str, Any]], Counter, Counter]:
    """Process one conversation. Returns (records, type_counts, violation_counts)."""
    records: list[dict[str, Any]] = []
    type_counts: Counter = Counter()
    violation_counts: Counter = Counter()
    state: Optional[dict[str, Any]] = None
    turn = 0

    for msg in conv.get("messages", []):
        if msg.get("role") != "user":
            continue
        if max_turns is not None and turn >= max_turns:
            break
        prompt = (msg.get("content") or "").strip()
        if not prompt:
            turn += 1
            continue

        messages = extraction.build_messages(prompt, state)
        result = complete(messages)
        parsed = extraction.parse_model_output(result.text)
        if parsed is None:
            log.warning(
                "Unparseable output for %s turn %d; skipping record. Raw: %.200s",
                conv["conversation_id"], turn, result.text,
            )
            violation_counts["unparseable"] += 1
            turn += 1
            continue

        violations = extraction.validate_output(parsed, prompt=prompt)
        for v in violations:
            violation_counts[v] += 1

        ptype = extraction.classify_type(state, parsed)

        # Skip records whose output has null fields for project-bearing
        # types: the model returned a degenerate state (e.g. for a prompt
        # with no project content). Writing them would fail the independent
        # milestone check. Do not carry the degenerate state forward.
        if ptype in ("new", "update", "no_change") and any(
            parsed.get(k) is None for k in ("project_overview", "specs", "design")
        ):
            violation_counts["null_output_skipped"] += 1
            turn += 1
            continue

        if ptype != "no_project" or keep_no_project:
            records.append(
                {
                    "conversation_id": conv["conversation_id"],
                    "turn": turn,
                    "type": ptype,
                    "input": {
                        "user_prompt": prompt,
                        "project_overview": extraction.normalize_state(state)[
                            "project_overview"
                        ],
                        "specs": extraction.normalize_state(state)["specs"],
                        "design": extraction.normalize_state(state)["design"],
                    },
                    "output": parsed,
                }
            )
            type_counts[ptype] += 1

        # Carry the latest project state forward.
        state = parsed
        turn += 1
        if delay:
            time.sleep(delay)

    return records, type_counts, violation_counts


def run_milestone_check(dataset_path: str, label: str) -> bool:
    """Run the independent check_dataset.py; returns True if clean."""
    try:
        proc = subprocess.run(
            [sys.executable, CHECK_SCRIPT, "--dataset", dataset_path,
             "--label", label],
            capture_output=True, text=True, timeout=600,
        )
        log.info("milestone check %s: %s", label, proc.stdout.strip().splitlines()[:3])
        if proc.returncode != 0:
            log.error("milestone check %s FAILED:\n%s", label, proc.stdout)
        return proc.returncode == 0
    except Exception as e:
        log.error("milestone check %s could not run: %s", label, e)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-conversations", type=int, default=None,
                    help="stop after processing this many conversations")
    ap.add_argument("--conversation-id", action="append", default=[],
                    help="process only this conversation id (repeatable)")
    ap.add_argument("--project-only", action="store_true",
                    help="only process conversations whose first user message "
                         "looks software-project related")
    ap.add_argument("--all-branches", action="store_true",
                    help="process every branch of a source tree (default: only "
                         "branch :0 per tree, since sibling branches share the "
                         "same first user message and would duplicate records)")
    ap.add_argument("--keep-no-project", action="store_true",
                    help="also write records where no project exists and none "
                         "is introduced (default: skip them)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="max user prompts to process per conversation")
    ap.add_argument("--resume", action="store_true",
                    help="skip conversations already recorded in the checkpoint")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to sleep between model calls")
    ap.add_argument("--workers", type=int, default=1,
                    help="number of parallel conversation workers (default 1)")
    ap.add_argument("--total", type=int, default=None,
                    help="total output values expected (default: auto-count "
                         "user prompts across all conversations)")
    ap.add_argument("--rate-limit-check-every", type=int, default=50,
                    help="probe the account rate limit every N conversations "
                         "(0 disables)")
    ap.add_argument("--out", default=DATASET_PATH,
                    help="output JSONL path (default: final/dataset.jsonl)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(FINAL_DIR, exist_ok=True)
    cp = load_checkpoint()
    processed_set = set(cp.get("processed", []))
    # Rebuild the seen-tree set from the checkpoint so tree dedup stays
    # consistent across resume runs.
    seen_trees: set[str] = set()
    if not args.all_branches:
        seen_trees = {cid.split(":")[0] for cid in processed_set}

    total_expected = args.total or count_total_output_values()
    milestone_step = max(1, total_expected // 10)
    log.info(
        "Total output values expected: %d (10%% milestone = %d records)",
        total_expected, milestone_step,
    )

    # Build the list of conversations to process (respecting filters).
    todo: list[dict[str, Any]] = []
    for conv in iter_conversations():
        cid = conv.get("conversation_id", "")
        if args.conversation_id and cid not in args.conversation_id:
            continue
        if args.resume and cid in processed_set:
            continue
        if args.project_only:
            first_user = next(
                (m.get("content", "") for m in conv.get("messages", [])
                 if m.get("role") == "user"),
                "",
            )
            if not looks_project_related(first_user):
                continue
        if not args.all_branches:
            tree = cid.split(":")[0]
            if tree in seen_trees:
                continue
            seen_trees.add(tree)
        todo.append(conv)
    log.info("Conversations queued for processing: %d", len(todo))

    # ------------------------------------------------------------------
    # Queue-based worker pool with a feeder thread.
    # ------------------------------------------------------------------
    work_q: "queue.Queue[Optional[dict[str, Any]]]" = queue.Queue()
    result_q: "queue.Queue[tuple[str, list, Counter, Counter]]" = queue.Queue()
    stop_flag = threading.Event()
    feeding_done = threading.Event()
    fed_count = 0
    in_flight = 0
    count_lock = threading.Lock()

    def worker_fn() -> None:
        nonlocal in_flight
        while True:
            conv = work_q.get()
            if conv is None:
                work_q.task_done()
                return
            with count_lock:
                in_flight += 1
            cid = conv.get("conversation_id", "")
            try:
                records, t_counts, v_counts = process_conversation_with_backoff(
                    conv, cid
                )
                result_q.put((cid, records, t_counts, v_counts))
            except Exception as e:
                log.exception("worker error on %s: %s", cid, e)
                result_q.put((cid, [], Counter(), Counter({"worker_error": 1})))
            finally:
                with count_lock:
                    in_flight -= 1
                work_q.task_done()

    def process_conversation_with_backoff(
        conv: dict[str, Any], cid: str
    ) -> tuple[list[dict[str, Any]], Counter, Counter]:
        """Run process_conversation, retrying the whole conversation with a
        long backoff when the account rate limit (HTTP 403) is hit."""
        backoff = 30
        waits = 0
        max_waits = 120  # up to ~60 minutes of waiting out the limit
        while True:
            try:
                return process_conversation(
                    conv, args.keep_no_project, args.max_turns, args.delay
                )
            except RateLimitError as e:
                waits += 1
                if waits > max_waits:
                    raise
                log.warning(
                    "rate limited on %s (backoff %d/%d); waiting %ds",
                    cid, waits, max_waits, backoff,
                )
                # Sleep in small chunks so stop_flag can interrupt promptly.
                slept = 0
                while slept < backoff and not stop_flag.is_set():
                    time.sleep(5)
                    slept += 5
                if stop_flag.is_set():
                    raise e

    def feeder() -> None:
        nonlocal fed_count
        for conv in todo:
            if stop_flag.is_set():
                break
            work_q.put(conv)
            with count_lock:
                fed_count += 1
        feeding_done.set()

    workers = [threading.Thread(target=worker_fn, daemon=True)
               for _ in range(max(1, args.workers))]
    for w in workers:
        w.start()
    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    total_type_counts: Counter = Counter()
    total_violation_counts: Counter = Counter()
    total_records = cp.get("records_written", 0)
    session_records = 0
    conversations_processed = 0
    start_time = time.time()
    next_milestone = milestone_step
    milestone_index = 1
    pending_cids: list[str] = []
    rate_limit_hit = False

    def handle_result(
        cid: str,
        records: list[dict[str, Any]],
        t_counts: Counter,
        v_counts: Counter,
        out_f,
    ) -> None:
        nonlocal total_records, session_records, conversations_processed
        nonlocal next_milestone, milestone_index
        pending_cids.append(cid)
        for r in records:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_f.flush()
        total_type_counts.update(t_counts)
        total_violation_counts.update(v_counts)
        total_records += len(records)
        session_records += len(records)
        conversations_processed += 1

        # Milestone checks at every 10% of total output values.
        while total_records >= next_milestone:
            label = f"check_{milestone_index:02d}_10pct"
            log.info(
                "MILESTONE %d%% reached (%d/%d records); "
                "running independent check",
                milestone_index * 10, total_records, total_expected,
            )
            run_milestone_check(args.out, label)
            milestone_index += 1
            next_milestone = milestone_index * milestone_step

        # Checkpoint (throttled to every 25 conversations).
        if conversations_processed % 25 == 0:
            cp["processed"].extend(pending_cids)
            pending_cids.clear()
            cp["records_written"] = total_records
            save_checkpoint(cp)

        if conversations_processed % 10 == 0:
            elapsed = time.time() - start_time
            rate = conversations_processed / elapsed if elapsed else 0
            log.info(
                "processed %d conversations, %d/%d records, "
                "%.2f conv/s, types=%s",
                conversations_processed, total_records, total_expected,
                rate, dict(total_type_counts),
            )

    try:
        with open(args.out, "a") as out_f:
            done = 0
            results_since_probe = 0
            while True:
                # Normal completion: feeding done and all results received.
                if not stop_flag.is_set() and feeding_done.is_set() and done >= fed_count:
                    break

                # Stop requested (rate limit / max conversations): drop
                # queued-but-not-started work, then wait for in-flight results.
                if stop_flag.is_set():
                    skipped = 0
                    while True:
                        try:
                            work_q.get_nowait()
                            skipped += 1
                        except queue.Empty:
                            break
                    if skipped:
                        with count_lock:
                            fed_count -= skipped
                        log.info("Dropped %d queued conversations on stop", skipped)
                    # Wait until every started conversation has finished.
                    # (put happens before the in_flight decrement, so once
                    # in_flight hits 0 all results are already in result_q.)
                    while True:
                        with count_lock:
                            remaining = in_flight
                        if remaining == 0:
                            break
                        try:
                            cid, records, t_counts, v_counts = result_q.get(timeout=10)
                        except queue.Empty:
                            continue
                        done += 1
                        handle_result(cid, records, t_counts, v_counts, out_f)
                    # Final drain: collect any results that arrived just
                    # before the last in_flight decrement.
                    while True:
                        try:
                            cid, records, t_counts, v_counts = result_q.get_nowait()
                        except queue.Empty:
                            break
                        done += 1
                        handle_result(cid, records, t_counts, v_counts, out_f)
                    break

                try:
                    cid, records, t_counts, v_counts = result_q.get(timeout=10)
                except queue.Empty:
                    continue
                done += 1
                handle_result(cid, records, t_counts, v_counts, out_f)

                # Periodic rate-limit probe while draining (catches limits hit
                # mid-run, after the feeder has finished).
                results_since_probe += 1
                if (args.rate_limit_check_every
                        and results_since_probe >= args.rate_limit_check_every):
                    results_since_probe = 0
                    rl = check_rate_limit()
                    if rl.get("limit_reached"):
                        log.warning(
                            "ACCOUNT RATE LIMIT REACHED (used %s%%); stopping "
                            "gracefully at %d records",
                            rl.get("used_percent"), total_records,
                        )
                        rate_limit_hit = True
                        stop_flag.set()

                if args.max_conversations and conversations_processed >= args.max_conversations:
                    log.info("Reached --max-conversations=%d", args.max_conversations)
                    stop_flag.set()
    finally:
        # Stop workers and flush the final checkpoint.
        for _ in workers:
            work_q.put(None)
        for w in workers:
            w.join(timeout=30)
        cp["processed"].extend(pending_cids)
        cp["records_written"] = total_records
        save_checkpoint(cp)

    elapsed = time.time() - start_time
    stats = {
        "conversations_processed": conversations_processed,
        "records_written": total_records,
        "session_records": session_records,
        "total_expected": total_expected,
        "milestone_step": milestone_step,
        "milestones_completed": milestone_index - 1,
        "rate_limit_hit": rate_limit_hit,
        "type_counts": dict(total_type_counts),
        "violation_counts": dict(total_violation_counts),
        "elapsed_seconds": round(elapsed, 1),
        "output": args.out,
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("DONE: %s", json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())