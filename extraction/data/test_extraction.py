"""Test the extraction pipeline against guide.md before running full generation.

Tests the four required input->output process types:
  1. new        - no existing project, prompt introduces one
  2. update     - existing project, prompt makes a small change
  3. no_change  - existing project, prompt is only a question
  4. no_project - no existing project, prompt introduces none

Plus extra update variants (spec deletion, design change, overview change).

Usage:
    python3 test_extraction.py            # run all cases
    python3 test_extraction.py --case new # run one case
"""

from __future__ import annotations

import argparse
import json
import sys

import extraction
from openai_client import complete

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# State from guide.md Sample 1 (inventory tracker)
STATE_INVENTORY = {
    "project_overview": "Desktop inventory tracker for warehouse using Python and SQLite",
    "specs": {
        "language": "Python",
        "views": "3",
        "storage": "SQLite",
        "cloud": "none",
        "RAM": "under 6GB",
        "database": "under 400MB",
        "platform": "Windows 11",
        "background": "black",
    },
    "design": [
        "current inventory view",
        "incoming shipments view",
        "outgoing shipments view",
        "local SQLite storage",
    ],
}

# State from guide.md Sample 2 (temperature backend)
STATE_TEMPERATURE = {
    "project_overview": "Backend service receiving temperature readings through HTTP POST requests",
    "specs": {
        "interface": "HTTP POST",
        "API": "REST",
        "storage": "PostgreSQL",
        "throughput": "500 requests/second",
        "temperature": "-80 to 120 Celsius",
        "battery": "0 to 100",
        "invalid": "HTTP 400",
        "authentication": "none",
        "port": "8080",
        "packaging": "Docker",
    },
    "design": [
        "sensor ID and timestamp",
        "Celsius temperature readings",
        "battery percentage readings",
        "historical REST API",
    ],
}

CASES: list[dict] = [
    {
        "name": "new",
        "expected_type": "new",
        "state": None,
        "prompt": (
            "Build a desktop inventory tracker for a small warehouse. Python only. "
            "The interface needs three views: current inventory, incoming shipments, "
            "and outgoing shipments. Store everything locally using SQLite; no cloud "
            "services. Product records need SKU, name, quantity, supplier, and reorder "
            "threshold. Search should work by SKU or product name. When quantity reaches "
            "the reorder threshold, display a warning beside that product. The application "
            "should launch without internet access and remain usable on Windows 11 machines "
            "with less than 6GB RAM. Keep the database below 400MB. Use a black background "
            "throughout the interface."
        ),
    },
    {
        "name": "update",
        "expected_type": "update",
        "state": STATE_TEMPERATURE,
        "prompt": (
            "Actually, switch the storage from PostgreSQL to MySQL, and add basic "
            "authentication with API keys. Also change the port to 9090."
        ),
    },
    {
        "name": "no_change",
        "expected_type": "no_change",
        "state": STATE_INVENTORY,
        "prompt": (
            "How should I structure the SQLite schema for the product records so that "
            "searching by SKU stays fast?"
        ),
    },
    {
        "name": "no_project",
        "expected_type": "no_project",
        "state": None,
        "prompt": (
            "What is the best way to learn Python for a complete beginner? I have about "
            "an hour a day to practice."
        ),
    },
    {
        "name": "update_delete_spec",
        "expected_type": "update",
        "state": STATE_INVENTORY,
        "prompt": (
            "Remove the black background requirement, we will use the system default "
            "theme instead."
        ),
    },
    {
        "name": "update_design",
        "expected_type": "update",
        "state": STATE_INVENTORY,
        "prompt": (
            "Add a fourth view for supplier management, and make the reorder warning "
            "appear as a popup dialog instead of an inline label."
        ),
    },
    {
        "name": "update_overview",
        "expected_type": "update",
        "state": STATE_TEMPERATURE,
        "prompt": (
            "This is no longer just a temperature service. Extend it to also accept "
            "humidity readings from the same sensors through the same HTTP endpoint."
        ),
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(case: dict, verbose: bool = True) -> dict:
    messages = extraction.build_messages(case["prompt"], case["state"])
    result = complete(messages)
    parsed = extraction.parse_model_output(result.text)
    violations = extraction.validate_output(parsed, prompt=case["prompt"]) if parsed else ["unparseable output"]
    actual_type = extraction.classify_type(case["state"], parsed)

    record = {
        "case": case["name"],
        "expected_type": case["expected_type"],
        "actual_type": actual_type,
        "type_match": actual_type == case["expected_type"],
        "input_state": extraction.normalize_state(case["state"]),
        "user_prompt": case["prompt"],
        "output_state": parsed,
        "violations": violations,
        "raw_output": result.text,
    }
    if verbose:
        print("=" * 78)
        print(f"CASE: {case['name']}  (expected type: {case['expected_type']})")
        print("-" * 78)
        print("INPUT STATE:")
        print(extraction.state_to_text(case["state"]))
        print("-" * 78)
        print("USER PROMPT:")
        print(case["prompt"])
        print("-" * 78)
        print("OUTPUT STATE:")
        print(extraction.format_state(parsed))
        print("-" * 78)
        print(f"CLASSIFIED TYPE: {actual_type}  "
              f"{'OK' if actual_type == case['expected_type'] else 'MISMATCH'}")
        if violations:
            print("VIOLATIONS:")
            for v in violations:
                print(f"  - {v}")
        else:
            print("VIOLATIONS: none")
        print()
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="run only this case name")
    ap.add_argument("--json", action="store_true", help="dump results as JSON")
    args = ap.parse_args()

    cases = CASES
    if args.case:
        cases = [c for c in CASES if c["name"] == args.case]
        if not cases:
            print(f"Unknown case {args.case!r}. Available: "
                  f"{[c['name'] for c in CASES]}")
            return 2

    records = [run_case(c) for c in cases]

    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False))
        return 0

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    ok = True
    for r in records:
        status = "PASS" if (r["type_match"] and not r["violations"]) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {r['case']:<22} type={r['actual_type']:<10} "
              f"violations={len(r['violations'])}")
    print()
    print("ALL PASS" if ok else "SOME CASES FAILED - review output above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())