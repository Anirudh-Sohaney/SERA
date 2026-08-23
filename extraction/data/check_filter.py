"""Independent validator for the coding-filter decisions log.

Verifies every decision line is well-formed, indices are unique and in
range, and reports progress stats. Exit 0 = clean, 1 = problems.

Usage:
    python3 check_filter.py --decisions .filter_decisions.jsonl \
        --total 65377 --label filter_check_01
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--label", default="filter_check")
    args = ap.parse_args()

    errors: list[str] = []
    seen: set[int] = set()
    ones = zeros = bad_lines = 0

    if not os.path.exists(args.decisions):
        print(f"[{args.label}] no decisions file yet | classified=0/{args.total}")
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
                bad_lines += 1
                continue
            if set(obj.keys()) != {"i", "d"}:
                errors.append(f"line {n}: bad keys {sorted(obj.keys())}")
                bad_lines += 1
                continue
            i, d = obj["i"], obj["d"]
            if not isinstance(i, int) or not (0 <= i < args.total):
                errors.append(f"line {n}: index out of range: {i!r}")
                continue
            if i in seen:
                errors.append(f"line {n}: duplicate index {i}")
                continue
            seen.add(i)
            if d == 1:
                ones += 1
            elif d == 0:
                zeros += 1
            else:
                errors.append(f"line {n}: decision not 0/1: {d!r}")

    pct = len(seen) / args.total * 100
    ok = not errors
    report = {
        "ok": ok,
        "label": args.label,
        "classified": len(seen),
        "total": args.total,
        "progress_pct": round(pct, 2),
        "kept_ones": ones,
        "dropped_zeros": zeros,
        "malformed_lines": bad_lines,
        "error_count": len(errors),
        "errors": errors[:20],
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.decisions)),
                            "final", "checks", f"{args.label}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    status = "CLEAN" if ok else "ERRORS FOUND"
    print(f"[{args.label}] {status} | classified={len(seen)}/{args.total} "
          f"({pct:.1f}%) kept={ones} dropped={zeros} errors={len(errors)}")
    for e in errors[:5]:
        print(f"  ERROR: {e}")
    print(f"  report: {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())