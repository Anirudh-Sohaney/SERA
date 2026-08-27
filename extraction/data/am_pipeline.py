"""AM-DeepSeek two-stage pipeline: classify coding relevance, then extract.

Stage 1: gpt-5.6-luna (low reasoning) binary-classifies each user prompt as
         coding/project-associated (1) or not (0).
Stage 2: if classified 1, gpt-5.6-luna (medium reasoning) extracts the
         project-memory state (project_overview / specs / design, guide.md
         contract). Single-turn records: state starts empty, so types are
         `new` or `no_project`.

Outputs:
- final/am_dataset_coding.jsonl : kept records, original order, same record
                                  schema as dataset_coding.jsonl
- .am_decisions.jsonl           : {"i", "cls", "rec"} per prompt (checkpoint)
- final/am_stats.json           : run summary

Independent checks run at every 10% via check_am.py.

Usage:
    python3 am_pipeline.py --pilot     # 16 prompts, consistency test, no writes
    python3 am_pipeline.py             # full run (resumable)
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

log = logging.getLogger("am_pipeline")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(BASE_DIR, "processed", "am_prompts.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "final", "am_dataset_coding.jsonl")
DECISIONS_PATH = os.path.join(BASE_DIR, ".am_decisions.jsonl")
STATS_PATH = os.path.join(BASE_DIR, "final", "am_stats.json")
CHECK_AM = os.path.join(BASE_DIR, "check_am.py")

MODEL = "gpt-5.6-luna"
CLS_REASONING = "low"
EXT_REASONING = "medium"

# ---------------------------------------------------------------------------
# Stage 1: classifier (verbatim ruleset from the validated oasst2 filter)
# ---------------------------------------------------------------------------

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


def parse_decision(text: str) -> Optional[int]:
    t = text.strip()
    for ch in t:
        if ch in "01":
            return int(ch)
        if ch not in " \n\t*#-":
            return None
    return None


def classify_prompt(prompt: str, max_parse_retries: int = 3) -> tuple[int, bool]:
    """Returns (decision, parsed_cleanly). Falls back to 1 (keep) on garbage."""
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": f"User prompt:\n{prompt}\n\nAnswer 1 or 0:"},
    ]
    for _ in range(max_parse_retries):
        result = complete(messages, model=MODEL, reasoning=CLS_REASONING)
        d = parse_decision(result.text)
        if d is not None:
            return d, True
        log.warning("unparseable classifier output: %.80s", result.text)
    return 1, False


# ---------------------------------------------------------------------------
# Stage 2: extraction (verbatim contract from the validated oasst2 pipeline)
# ---------------------------------------------------------------------------

GUIDELINES = """# Extraction Guidelines

## Project Overview

- Output exactly 10-13 words.
- Use only words present in the user prompt.
- Represent the project's primary purpose and type.
- Exclude implementation details when possible.
- Exclude pronouns.
- Exclude filler words.
- Avoid new terminology.
- Preserve important nouns and quantifiable project characteristics.
- Produce one direct declarative phrase.

## Specs

- Output 2-12 entries.
- Use `"key": "value"` format.
- Keys and values should use prompt terminology.
- Preserve numbers, units, limits, platforms, technologies, interfaces, and prohibitions.
- Prefer measurable or categorical values.
- Minimize wording.
- Do not infer unspecified values.
- Do not convert qualitative statements into unsupported measurements.
- Combine closely related constraints when necessary.
- Use `"none"` only when the prompt explicitly prohibits or excludes something.

## Design Statements

- Output 2-4 statements.
- Each statement must contain fewer than 6 words.
- Use only words present in the user prompt.
- Represent implementation structure, component relationships, or required behavior.
- Avoid generic software terminology absent from the prompt.
- Exclude unsupported implementation decisions.
- Preserve technical nouns and relationships.
- Do not duplicate specs unless the relationship or architecture is represented.
- Prefer noun-based or action-based structures."""

EXTRACTION_SYSTEM = f"""You are a project memory extraction engine. You maintain a software project's memory in exactly three fields: project_overview, specs, and design.

You will receive:
1. EXTRACTION GUIDELINES
2. CURRENT PROJECT STATE (a JSON object, or the literal word "none" when no project exists yet)
3. A USER PROMPT

Determine what the user prompt does to the project, then return the UPDATED project state.

DECISION RULES:
- NEW PROJECT: No current project state exists AND the prompt introduces a software project. Extract the full project state from the prompt.
- UPDATE: A current project state exists AND the prompt changes project requirements. Update the state: add new specs or designs, modify changed values, remove deleted entries, and update the overview if the project's purpose changed.
- NO CHANGE: A current project state exists AND the prompt only asks a question or discusses the project without changing any requirement. Return the current state unchanged.
- NO PROJECT: No current project state exists AND the prompt does not introduce a software project. Return null for all three fields.

OUTPUT FORMAT: Return ONLY a single JSON object with exactly these keys:
{{"project_overview": <string or null>, "specs": <object or null>, "design": <array of strings or null>}}

No markdown, no code fences, no explanations, no extra text before or after the JSON.

HARD LIMITS (never exceed these):
- project_overview: exactly 10-13 words. If the user prompt is too short to reach 10 words without repeating words, prefer a shorter overview over repetition.
- specs: 2-12 entries maximum. Combine closely related constraints into one entry (e.g. "views": "3" instead of one entry per view). Do not invent keys for every sentence.
- design: 2-4 statements, each fewer than 6 words. Do not duplicate spec content; represent structure, relationships, or required behavior.

EXAMPLES (from the extraction guide):

Example 1 - prompt: "Build a desktop inventory tracker for a small warehouse. Python only. The interface needs three views: current inventory, incoming shipments, and outgoing shipments. Store everything locally using SQLite; no cloud services. Product records need SKU, name, quantity, supplier, and reorder threshold. Search should work by SKU or product name. When quantity reaches the reorder threshold, display a warning beside that product. The application should launch without internet access and remain usable on Windows 11 machines with less than 6GB RAM. Keep the database below 400MB. Use a black background throughout the interface."
Expected output:
{{"project_overview": "Desktop inventory tracker for warehouse using Python and SQLite", "specs": {{"language": "Python", "views": "3", "storage": "SQLite", "cloud": "none", "RAM": "under 6GB", "database": "under 400MB", "platform": "Windows 11", "background": "black"}}, "design": ["current inventory view", "incoming shipments view", "outgoing shipments view", "local SQLite storage"]}}

Example 2 - prompt: "I need a backend service, not a frontend. It should receive temperature readings from remote sensors through HTTP POST requests and expose historical readings through a REST API. Each reading contains sensor ID, timestamp, Celsius temperature, and battery percentage. PostgreSQL is required. Reject temperatures below -80 or above 120 Celsius. Reject battery percentages outside 0 through 100. The service must process at least 500 requests per second and return HTTP 400 for invalid readings. Authentication is unnecessary for this prototype. Package it with Docker and expose port 8080."
Expected output:
{{"project_overview": "Backend service receiving temperature readings through HTTP POST requests", "specs": {{"interface": "HTTP POST", "API": "REST", "storage": "PostgreSQL", "throughput": "500 requests/second", "temperature": "-80 to 120 Celsius", "battery": "0 to 100", "invalid": "HTTP 400", "authentication": "none", "port": "8080", "packaging": "Docker"}}, "design": ["sensor ID and timestamp", "Celsius temperature readings", "battery percentage readings", "historical REST API"]}}

EXTRACTION GUIDELINES:
{GUIDELINES}"""

EMPTY_STATE: dict[str, Any] = {"project_overview": None, "specs": None, "design": None}


def _clean_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    return value


def normalize_state(state: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not state:
        return dict(EMPTY_STATE)
    out = {
        "project_overview": _clean_none(state.get("project_overview")),
        "specs": _clean_none(state.get("specs")),
        "design": _clean_none(state.get("design")),
    }
    if isinstance(out["specs"], dict) and not out["specs"]:
        out["specs"] = None
    if isinstance(out["design"], list) and not out["design"]:
        out["design"] = None
    return out


def is_empty_state(state: Optional[dict[str, Any]]) -> bool:
    if not state:
        return True
    return all(v is None for v in normalize_state(state).values())


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    import re
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return None
    return None


def validate_output(state: dict[str, Any], prompt: str) -> list[str]:
    violations: list[str] = []
    s = normalize_state(state)
    if is_empty_state(s):
        return violations
    prompt_words = len(prompt.split())
    overview = s.get("project_overview")
    if not isinstance(overview, str) or not overview.strip():
        violations.append("project_overview missing or not a string")
    else:
        words = overview.split()
        min_words = 9 if prompt_words >= 12 else 4
        if not (min_words <= len(words) <= 13):
            violations.append(f"project_overview has {len(words)} words "
                              f"(need {min_words}-13)")
    specs = s.get("specs")
    if not isinstance(specs, dict) or not specs:
        violations.append("specs missing or not a non-empty object")
    else:
        if not (2 <= len(specs) <= 12):
            violations.append(f"specs has {len(specs)} entries (need 2-12)")
        for k, v in specs.items():
            if not isinstance(k, str) or not isinstance(v, str):
                violations.append(f"spec entry not string->string: {k!r}: {v!r}")
    design = s.get("design")
    if not isinstance(design, list) or not design:
        violations.append("design missing or not a non-empty array")
    else:
        if not (2 <= len(design) <= 4):
            violations.append(f"design has {len(design)} statements (need 2-4)")
        for stmt in design:
            if not isinstance(stmt, str):
                violations.append(f"design entry not a string: {stmt!r}")
            elif len(stmt.split()) >= 6:
                violations.append(f"design statement has >=6 words: {stmt!r}")
    return violations


def extract_project(prompt: str, retries: int = 2) -> Optional[dict[str, Any]]:
    """Single-turn extraction from empty state. Returns normalized state."""
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user",
         "content": f'CURRENT PROJECT STATE:\nnone\n\nUSER PROMPT:\n{prompt}'},
    ]
    for attempt in range(retries + 1):
        result = complete(messages, model=MODEL, reasoning=EXT_REASONING)
        obj = extract_json_object(result.text)
        if obj is not None:
            return normalize_state(obj)
        log.warning("unparseable extraction output (attempt %d): %.80s",
                    attempt + 1, result.text)
    return None


def process_record(rec: dict[str, Any]) -> tuple[int, Optional[dict[str, Any]]]:
    """Classify then extract. Returns (cls, record_or_None)."""
    prompt = rec["prompt"]
    cls, _clean = classify_prompt(prompt)
    if cls != 1:
        return 0, None
    state = extract_project(prompt)
    if state is None:
        return 1, None  # classified coding but extraction failed; drop record
    if is_empty_state(state):
        ptype = "no_project"
    else:
        ptype = "new"
        # degenerate output guard (null fields on a project-bearing type)
        norm = normalize_state(state)
        if any(norm.get(k) is None for k in ("project_overview", "specs", "design")):
            return 1, None
    record = {
        "conversation_id": rec["record_id"],
        "turn": 0,
        "type": ptype,
        "input": {"user_prompt": prompt,
                  "project_overview": None, "specs": None, "design": None},
        "output": normalize_state(state),
    }
    return 1, record


# ---------------------------------------------------------------------------
# Pilot / main
# ---------------------------------------------------------------------------

def load_prompts() -> list[dict[str, Any]]:
    records = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_decisions() -> dict[int, dict[str, Any]]:
    decided: dict[int, dict[str, Any]] = {}
    if os.path.exists(DECISIONS_PATH):
        with open(DECISIONS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    decided[obj["i"]] = obj
    return decided


def run_pilot(prompts: list[dict[str, Any]]) -> int:
    by_domain: dict[str, list[int]] = {}
    for i, r in enumerate(prompts):
        by_domain.setdefault(r["domain"], []).append(i)
    sample: list[int] = []
    for dom, idxs in sorted(by_domain.items()):
        step = max(1, len(idxs) // 6)
        sample.extend(idxs[::step][:6])
    sample = sample[:16]

    print(f"PILOT: {len(sample)} prompts | classify x2 + extract | "
          f"model={MODEL} cls_reasoning={CLS_REASONING} ext_reasoning={EXT_REASONING}")
    agree = 0
    extracted = 0
    for i in sample:
        p = prompts[i]["prompt"]
        d1, _ = classify_prompt(p)
        d2, _ = classify_prompt(p)
        ok = d1 == d2
        agree += int(ok)
        line = f"[{i:5d}] {prompts[i]['domain']:5s} cls={d1}/{d2} " \
               f"{'OK ' if ok else 'MISMATCH'}"
        if d1 == 1:
            state = extract_project(p)
            if state and not is_empty_state(state):
                extracted += 1
                v = validate_output(state, p)
                ov = state.get("project_overview") or ""
                line += f" | EXTRACT new ({len(v)} viol) :: {ov[:60]}"
            elif state:
                line += " | EXTRACT no_project"
            else:
                line += " | EXTRACT FAILED"
        else:
            snippet = " ".join(p.split())[:50]
            line += f" | skip :: {snippet}"
        print(line, flush=True)
    print(f"\nagreement: {agree}/{len(sample)} ({agree / len(sample) * 100:.0f}%)")
    print(f"extractions produced: {extracted}")
    return 0 if agree == len(sample) else 1


def assemble(prompts: list[dict[str, Any]],
             decisions: dict[int, dict[str, Any]]) -> int:
    kept = 0
    with open(OUT_PATH, "w") as out:
        for i in range(len(prompts)):
            rec = decisions[i].get("rec")
            if rec is not None:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    prompts = load_prompts()
    total = len(prompts)
    log.info("loaded %d prompts from %s", total, PROMPTS_PATH)

    if args.pilot:
        return run_pilot(prompts)

    if args.reset and os.path.exists(DECISIONS_PATH):
        os.remove(DECISIONS_PATH)

    decisions = load_decisions()
    todo = [i for i in range(total) if i not in decisions]
    log.info("%d already done, %d to go", len(decisions), len(todo))
    milestone_step = max(1, total // 10)
    next_milestone = ((len(decisions) // milestone_step) + 1) * milestone_step

    work_q: "queue.Queue[Optional[int]]" = queue.Queue()
    result_q: "queue.Queue[tuple[int, dict[str, Any]]]" = queue.Queue()
    stop_flag = threading.Event()

    def worker_fn() -> None:
        while True:
            idx = work_q.get()
            if idx is None:
                work_q.task_done()
                return
            try:
                backoff, waits = 30, 0
                while True:
                    try:
                        cls, rec = process_record(prompts[idx])
                        result_q.put((idx, {"i": idx, "cls": cls, "rec": rec}))
                        break
                    except RateLimitError:
                        waits += 1
                        if waits > 120:
                            raise
                        log.warning("rate limited on %d; waiting %ds (%d/120)",
                                    idx, backoff, waits)
                        slept = 0
                        while slept < backoff and not stop_flag.is_set():
                            time.sleep(5)
                            slept += 5
                        if stop_flag.is_set():
                            raise KeyboardInterrupt
            except Exception as e:
                log.exception("worker error on %d: %s", idx, e)
                result_q.put((idx, {"i": idx, "cls": 1, "rec": None}))
            finally:
                work_q.task_done()

    workers = [threading.Thread(target=worker_fn, daemon=True)
               for _ in range(max(1, args.workers))]
    for w in workers:
        w.start()

    start = time.time()
    dec_f = open(DECISIONS_PATH, "a")

    def handle(obj: dict[str, Any]) -> None:
        nonlocal next_milestone
        dec_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        dec_f.flush()
        decisions[obj["i"]] = obj
        while len(decisions) >= next_milestone:
            pct = next_milestone * 10 // total
            label = f"am_check_{pct:02d}_10pct"
            log.info("MILESTONE %d%% (%d/%d); independent check",
                     pct * 10, len(decisions), total)
            try:
                proc = subprocess.run(
                    [sys.executable, CHECK_AM, "--decisions", DECISIONS_PATH,
                     "--total", str(total), "--label", label],
                    capture_output=True, text=True, timeout=600)
                log.info("milestone check: %s",
                         proc.stdout.strip().splitlines()[:2])
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
                idx, obj = result_q.get(timeout=15)
            except queue.Empty:
                continue
            got += 1
            handle(obj)
            if got % 500 == 0:
                rate = got / max(time.time() - start, 1)
                n_cls = sum(1 for d in decisions.values() if d["cls"] == 1)
                log.info("%d/%d done (%.2f rec/s) classified_coding=%d "
                         "extracted=%d", len(decisions), total, rate, n_cls,
                         sum(1 for d in decisions.values() if d.get("rec")))
    finally:
        dec_f.close()
        for _ in workers:
            work_q.put(None)
        for w in workers:
            w.join(timeout=30)

    missing = [i for i in range(total) if i not in decisions]
    if missing:
        log.error("%d prompts never processed; aborting assembly", len(missing))
        return 1

    kept = assemble(prompts, decisions)
    proc = subprocess.run(
        [sys.executable, CHECK_AM, "--decisions", DECISIONS_PATH,
         "--total", str(total), "--label", "am_final",
         "--dataset", OUT_PATH],
        capture_output=True, text=True, timeout=900)
    log.info("final check: %s", proc.stdout.strip().splitlines()[:1])

    n_cls = sum(1 for d in decisions.values() if d["cls"] == 1)
    types = Counter(d["rec"]["type"] for d in decisions.values() if d.get("rec"))
    stats = {
        "total_prompts": total,
        "classified_coding": n_cls,
        "classified_pct": round(n_cls / total * 100, 2),
        "records_kept": kept,
        "record_types": dict(types),
        "elapsed_seconds": round(time.time() - start, 1),
        "output": OUT_PATH,
        "model": MODEL,
        "cls_reasoning": CLS_REASONING,
        "ext_reasoning": EXT_REASONING,
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("DONE: %s", json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())