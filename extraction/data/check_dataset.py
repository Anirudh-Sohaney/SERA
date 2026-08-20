"""Independent dataset validation script.

Reads a dataset JSONL file directly (no dependency on the generator's internal
state) and validates every record. Used as the independent check at each 10%
milestone of the full extraction run.

Checks performed per record:
- valid JSON
- exact key set: conversation_id, turn, type, input, output
- input keys: user_prompt, project_overview, specs, design
- output keys: project_overview, specs, design
- non-empty user_prompt
- type in {new, update, no_change, no_project}
- type/input consistency (new -> empty input; update/no_change -> non-empty)
- non-null output fields for new/update/no_change
- no duplicate (conversation_id, turn) pairs
- per-conversation turn sequences are strictly increasing

Writes a JSON report to --report and exits 0 when clean, 1 when errors found.

Usage:
    python3 check_dataset.py --dataset final/dataset.jsonl --label check_01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Optional

VALID_TYPES = {"new", "update", "no_change", "no_project"}


def validate_dataset(path: str) -> dict[str, Any]:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    line_errors: list[dict[str, Any]] = []

    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                line_errors.append({"line": lineno, "error": f"JSON parse: {e}"})
                continue
            records.append(rec)
            rerr = _validate_record(rec, lineno)
            if rerr:
                errors.extend(rerr)

    # Duplicate (conversation_id, turn) check.
    seen: Counter = Counter((r["conversation_id"], r["turn"]) for r in records)
    dups = [k for k, v in seen.items() if v > 1]
    if dups:
        errors.append(f"duplicate (conversation_id, turn) pairs: {dups[:10]}")

    # Turn sequence check per conversation.
    turns: dict[str, list[int]] = defaultdict(list)
    for r in records:
        turns[r["conversation_id"]].append(r["turn"])
    for cid, ts in turns.items():
        if ts != sorted(ts) or len(set(ts)) != len(ts):
            errors.append(f"bad turn sequence in {cid}: {ts}")

    type_counts = Counter(r["type"] for r in records)
    return {
        "ok": not errors and not line_errors,
        "total_records": len(records),
        "type_counts": dict(type_counts),
        "unique_conversations": len(set(r["conversation_id"] for r in records)),
        "error_count": len(errors) + len(line_errors),
        "errors": errors[:50],
        "line_errors": line_errors[:20],
    }


def _validate_record(rec: dict[str, Any], lineno: int) -> list[str]:
    errs: list[str] = []
    tag = f"line {lineno}"
    if set(rec.keys()) != {"conversation_id", "turn", "type", "input", "output"}:
        errs.append(f"{tag}: bad top-level keys {sorted(rec.keys())}")
        return errs
    if set(rec["input"].keys()) != {"user_prompt", "project_overview", "specs", "design"}:
        errs.append(f"{tag}: bad input keys {sorted(rec['input'].keys())}")
    if set(rec["output"].keys()) != {"project_overview", "specs", "design"}:
        errs.append(f"{tag}: bad output keys {sorted(rec['output'].keys())}")
    if not isinstance(rec["input"]["user_prompt"], str) or not rec["input"]["user_prompt"].strip():
        errs.append(f"{tag}: empty user_prompt")
    if rec["type"] not in VALID_TYPES:
        errs.append(f"{tag}: invalid type {rec['type']!r}")

    inp = rec["input"]
    in_empty = all(
        v is None for v in (inp["project_overview"], inp["specs"], inp["design"])
    )
    if rec["type"] == "new" and not in_empty:
        errs.append(f"{tag}: type=new but input state non-empty")
    if rec["type"] in ("update", "no_change") and in_empty:
        errs.append(f"{tag}: type={rec['type']} but input state empty")
    if rec["type"] in ("new", "update", "no_change"):
        out = rec["output"]
        if any(out[k] is None for k in ("project_overview", "specs", "design")):
            errs.append(f"{tag}: null output field for type={rec['type']}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="path to dataset JSONL")
    ap.add_argument("--label", default="check", help="check label for the report")
    ap.add_argument("--report-dir", default=None,
                    help="directory for the report (default: alongside dataset)")
    args = ap.parse_args()

    result = validate_dataset(args.dataset)

    report_dir = args.report_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.dataset)), "checks"
    )
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{args.label}.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "CLEAN" if result["ok"] else "ERRORS FOUND"
    print(f"[{args.label}] {status} | records={result['total_records']} "
          f"errors={result['error_count']} | types={result['type_counts']}")
    for e in result["errors"][:10]:
        print(f"  ERROR: {e}")
    for e in result["line_errors"][:5]:
        print(f"  LINE ERROR: {e}")
    print(f"  report: {report_path}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())