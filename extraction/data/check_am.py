"""Independent validator for the AM pipeline decisions log and output.

Verifies decision lines are well-formed (indices unique/in-range, cls is
0/1, rec present iff cls==1, record schema correct) and optionally
validates the assembled dataset file. Exit 0 = clean.

Usage:
    python3 check_am.py --decisions .am_decisions.jsonl --total 38000 \
        --label am_check_01 [--dataset final/am_dataset_coding.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REQUIRED_KEYS = {"conversation_id", "turn", "type", "input", "output"}
VALID_TYPES = {"new", "update", "no_change", "no_project"}


def check_record(rec: dict, errors: list[str], where: str) -> None:
    if set(rec.keys()) != REQUIRED_KEYS:
        errors.append(f"{where}: bad keys {sorted(rec.keys())}")
        return
    if rec["type"] not in VALID_TYPES:
        errors.append(f"{where}: bad type {rec['type']!r}")
    if not isinstance(rec["turn"], int):
        errors.append(f"{where}: turn not int")
    inp = rec.get("input") or {}
    if "user_prompt" not in inp:
        errors.append(f"{where}: input missing user_prompt")
    out = rec.get("output") or {}
    if set(out.keys()) != {"project_overview", "specs", "design"}:
        errors.append(f"{where}: output bad keys {sorted(out.keys())}")
        return
    if rec["type"] == "new" and any(
            out.get(k) is None for k in ("project_overview", "specs", "design")):
        errors.append(f"{where}: null output field for type=new")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--label", default="am_check")
    ap.add_argument("--dataset", default=None,
                    help="also validate the assembled output file")
    args = ap.parse_args()

    errors: list[str] = []
    seen: set[int] = set()
    n_cls = n_rec = 0

    if not os.path.exists(args.decisions):
        print(f"[{args.label}] no decisions file yet | processed=0/{args.total}")
        return 0

    with open(args.decisions) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {n}: invalid JSON ({e})")
                continue
            if not isinstance(obj.get("i"), int) or not (
                    0 <= obj["i"] < args.total):
                errors.append(f"line {n}: index out of range: {obj.get('i')!r}")
                continue
            if obj["i"] in seen:
                errors.append(f"line {n}: duplicate index {obj['i']}")
                continue
            seen.add(obj["i"])
            if obj.get("cls") not in (0, 1):
                errors.append(f"line {n}: cls not 0/1: {obj.get('cls')!r}")
                continue
            if obj["cls"] == 1:
                n_cls += 1
            rec = obj.get("rec")
            if obj["cls"] == 0 and rec is not None:
                errors.append(f"line {n}: cls=0 but rec present")
            if rec is not None:
                n_rec += 1
                check_record(rec, errors, f"line {n}")

    n_out = 0
    if args.dataset and os.path.exists(args.dataset):
        with open(args.dataset) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"dataset line {ln}: invalid JSON ({e})")
                    continue
                n_out += 1
                check_record(rec, errors, f"dataset line {ln}")

    ok = not errors
    report = {
        "ok": ok,
        "label": args.label,
        "processed": len(seen),
        "total": args.total,
        "classified_coding": n_cls,
        "records_in_decisions": n_rec,
        "records_in_dataset": n_out,
        "error_count": len(errors),
        "errors": errors[:20],
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.decisions)),
                            "final", "checks", f"{args.label}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    status = "CLEAN" if ok else "ERRORS FOUND"
    print(f"[{args.label}] {status} | processed={len(seen)}/{args.total} "
          f"coding={n_cls} records={n_rec} dataset={n_out} errors={len(errors)}")
    for e in errors[:5]:
        print(f"  ERROR: {e}")
    print(f"  report: {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())