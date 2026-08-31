"""
Product acceptance test suite for SERA.

Tests the system against specific scenarios that must pass for
production readiness. Each test has an expected outcome.

Usage:
    python -m src.evaluation.acceptance_test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.engine import ProjectMemoryEngine
from src.memory.schema import MemoryCategory, MemoryStatus, ProjectState


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

class AcceptanceTest:
    """A single acceptance test case."""

    def __init__(
        self,
        name: str,
        category: str,
        turns: List[Dict[str, Any]],
        expected_state: Dict[str, Any],
        description: str = "",
    ) -> None:
        self.name = name
        self.category = category
        self.turns = turns
        self.expected_state = expected_state
        self.description = description

    def run(self, engine: ProjectMemoryEngine, admission=None) -> Dict[str, Any]:
        """Run the test and return results.
        
        Args:
            engine: ProjectMemoryEngine instance.
            admission: Optional EvidenceBasedAdmission instance for filtering.
        """
        results = []
        for i, turn in enumerate(self.turns):
            prompt = turn["prompt"]
            spans = turn.get("spans", [])
            turn_number = i + 1

            try:
                # Apply admission policy if provided
                if admission is not None and spans:
                    from src.memory.schema import MemoryCategory
                    filtered_spans = []
                    for span in spans:
                        candidate = {
                            "text": span["text"],
                            "confidence": span.get("confidence", 0.95),
                            "start": span.get("start", 0),
                            "end": span.get("end", len(span["text"])),
                            "category": span.get("category"),
                        }
                        result = admission.decide(candidate, prompt, engine._state)
                        if result.decision.value in ("LOCK", "PENDING"):
                            filtered_spans.append(span)
                    spans = filtered_spans
                
                result = engine.process_turn(
                    prompt=prompt,
                    extracted_spans=spans,
                    turn_number=turn_number,
                )
                transitions = result["transitions"]
                state = result["state_snapshot"]
                results.append({
                    "turn": turn_number,
                    "transitions": [
                        {"type": t.transition_type.value, "category": t.category.value, "value": t.value}
                        for t in transitions
                    ],
                    "active_memories": len(state["active_memories"]),
                })
            except Exception as e:
                results.append({
                    "turn": turn_number,
                    "error": str(e),
                })

        # Check final state
        final_state = engine.get_project_state()
        actual_active = {
            (m.category.value, m.value.lower().strip())
            for m in final_state.active_memories
        }
        expected_active = {
            (v["category"], v["value"].lower().strip())
            for v in self.expected_state.get("active", [])
        }

        matching = actual_active & expected_active
        missing = expected_active - actual_active
        extra = actual_active - expected_active

        passed = len(missing) == 0 and len(extra) == 0

        return {
            "name": self.name,
            "category": self.category,
            "passed": passed,
            "turns": results,
            "actual_memories": [
                {"category": m.category.value, "value": m.value}
                for m in final_state.active_memories
            ],
            "expected_memories": self.expected_state.get("active", []),
            "matching": [{"category": c, "value": v} for c, v in matching],
            "missing": [{"category": c, "value": v} for c, v in missing],
            "extra": [{"category": c, "value": v} for c, v in extra],
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def build_acceptance_suite() -> List[AcceptanceTest]:
    """Build the complete acceptance test suite."""
    tests = []

    # --- Explicit Requirements ---
    tests.append(AcceptanceTest(
        name="explicit_python_requirement",
        category="explicit_requirements",
        turns=[{
            "prompt": "Use Python for the backend.",
            "spans": [{"text": "Python", "start": 4, "end": 10, "confidence": 0.95, "label": "project_info"}],
        }],
        expected_state={"active": [{"category": "language", "value": "Python"}]},
        description="System must extract explicit technology requirement",
    ))

    tests.append(AcceptanceTest(
        name="explicit_framework_requirement",
        category="explicit_requirements",
        turns=[{
            "prompt": "Build the API with FastAPI.",
            "spans": [{"text": "FastAPI", "start": 23, "end": 30, "confidence": 0.95, "label": "project_info"}],
        }],
        expected_state={"active": [{"category": "framework", "value": "FastAPI"}]},
        description="System must extract framework requirement",
    ))

    # --- Technology Rejection ---
    tests.append(AcceptanceTest(
        name="technology_rejection",
        category="technology_rejection",
        turns=[{
            "prompt": "Use Flask for the backend.",
            "spans": [{"text": "Flask", "start": 4, "end": 9, "confidence": 0.95, "label": "project_info"}],
        }, {
            "prompt": "Actually, do not use Flask. Use Django instead.",
            "spans": [
                {"text": "Flask", "start": 24, "end": 29, "confidence": 0.95, "label": "project_info"},
                {"text": "Django", "start": 39, "end": 45, "confidence": 0.95, "label": "project_info"},
            ],
        }],
        expected_state={"active": [{"category": "framework", "value": "Django"}]},
        description="System must handle technology replacement across turns",
    ))

    # --- Technology Replacement ---
    tests.append(AcceptanceTest(
        name="technology_replacement",
        category="technology_replacement",
        turns=[{
            "prompt": "Use SQLite for the database.",
            "spans": [{"text": "SQLite", "start": 8, "end": 14, "confidence": 0.95, "label": "project_info"}],
        }, {
            "prompt": "Actually, use PostgreSQL instead of SQLite.",
            "spans": [
                {"text": "PostgreSQL", "start": 15, "end": 25, "confidence": 0.95, "label": "project_info"},
                {"text": "SQLite", "start": 36, "end": 42, "confidence": 0.95, "label": "project_info"},
            ],
        }],
        expected_state={"active": [{"category": "database", "value": "PostgreSQL"}]},
        description="System must replace SQLite with PostgreSQL",
    ))

    # --- Multi-turn Accumulation ---
    tests.append(AcceptanceTest(
        name="multi_turn_accumulation",
        category="multi_turn_updates",
        turns=[{
            "prompt": "Use Python for the backend.",
            "spans": [{"text": "Python", "start": 4, "end": 10, "confidence": 0.95, "label": "project_info"}],
        }, {
            "prompt": "Use FastAPI as the framework.",
            "spans": [{"text": "FastAPI", "start": 8, "end": 15, "confidence": 0.95, "label": "project_info"}],
        }, {
            "prompt": "Use PostgreSQL for the database.",
            "spans": [{"text": "PostgreSQL", "start": 8, "end": 18, "confidence": 0.95, "label": "project_info"}],
        }],
        expected_state={"active": [
            {"category": "language", "value": "Python"},
            {"category": "framework", "value": "FastAPI"},
            {"category": "database", "value": "PostgreSQL"},
        ]},
        description="System must accumulate facts across turns",
    ))

    # --- No Memory Change ---
    tests.append(AcceptanceTest(
        name="no_memory_change",
        category="no_memory_change",
        turns=[{
            "prompt": "Use Python for the backend.",
            "spans": [{"text": "Python", "start": 4, "end": 10, "confidence": 0.95, "label": "project_info"}],
        }, {
            "prompt": "Can you show me how to use list comprehensions?",
            "spans": [],
        }],
        expected_state={"active": [{"category": "language", "value": "Python"}]},
        description="System must not change memory for non-project prompts",
    ))

    # --- Hypothetical Statement ---
    tests.append(AcceptanceTest(
        name="hypothetical_no_lock",
        category="hypothetical_statements",
        turns=[{
            "prompt": "Maybe we should use Rust for performance.",
            "spans": [{"text": "Rust", "start": 27, "end": 31, "confidence": 0.95, "label": "project_info"}],
        }],
        expected_state={"active": []},
        description="System should NOT lock hypothetical decisions (spans present but admission should reject)",
    ))

    # --- Casual Conversation ---
    tests.append(AcceptanceTest(
        name="casual_no_extraction",
        category="casual_conversation",
        turns=[{
            "prompt": "Thanks for the help! That worked perfectly.",
            "spans": [],
        }],
        expected_state={"active": []},
        description="System must not extract from casual conversation",
    ))

    # --- Non-project Coding ---
    tests.append(AcceptanceTest(
        name="non_project_coding",
        category="non_project_coding",
        turns=[{
            "prompt": "How do I reverse a list in Python?",
            "spans": [],
        }],
        expected_state={"active": []},
        description="System must not extract from non-project coding questions",
    ))

    # --- Project Name ---
    tests.append(AcceptanceTest(
        name="project_name_extraction",
        category="project_names",
        turns=[{
            "prompt": "The project is called DataPipeline. Use Python and PostgreSQL.",
            "spans": [
                {"text": "DataPipeline", "start": 22, "end": 34, "confidence": 0.95, "label": "project_info"},
                {"text": "Python", "start": 44, "end": 50, "confidence": 0.95, "label": "project_info"},
                {"text": "PostgreSQL", "start": 55, "end": 65, "confidence": 0.95, "label": "project_info"},
            ],
        }],
        expected_state={"active": [
            {"category": "project", "value": "DataPipeline"},
            {"category": "language", "value": "Python"},
            {"category": "database", "value": "PostgreSQL"},
        ]},
        description="System must extract project name alongside technologies",
    ))

    # --- Directory Structure ---
    tests.append(AcceptanceTest(
        name="directory_structure",
        category="directory_structures",
        turns=[{
            "prompt": "Put the source code in src/ and tests in tests/.",
            "spans": [
                {"text": "src/", "start": 30, "end": 34, "confidence": 0.95, "label": "project_info"},
                {"text": "tests/", "start": 43, "end": 49, "confidence": 0.95, "label": "project_info"},
            ],
        }],
        expected_state={"active": [
            {"category": "directory", "value": "src/"},
            {"category": "directory", "value": "tests/"},
        ]},
        description="System must extract directory structure",
    ))

    # --- Multiple Languages ---
    tests.append(AcceptanceTest(
        name="multiple_languages",
        category="language_selection",
        turns=[{
            "prompt": "Use Python for the backend and JavaScript for the frontend.",
            "spans": [
                {"text": "Python", "start": 8, "end": 14, "confidence": 0.95, "label": "project_info"},
                {"text": "JavaScript", "start": 35, "end": 45, "confidence": 0.95, "label": "project_info"},
            ],
        }],
        expected_state={"active": [
            {"category": "language", "value": "Python"},
            {"category": "language", "value": "JavaScript"},
        ]},
        description="System must handle multiple languages in same prompt",
    ))

    # --- Deployment Decision ---
    tests.append(AcceptanceTest(
        name="deployment_decision",
        category="deployment_decisions",
        turns=[{
            "prompt": "Deploy to AWS using Docker and Kubernetes.",
            "spans": [
                {"text": "AWS", "start": 11, "end": 14, "confidence": 0.95, "label": "project_info"},
                {"text": "Docker", "start": 25, "end": 31, "confidence": 0.95, "label": "project_info"},
                {"text": "Kubernetes", "start": 36, "end": 46, "confidence": 0.95, "label": "project_info"},
            ],
        }],
        expected_state={"active": [
            {"category": "platform", "value": "AWS"},
            {"category": "tool", "value": "Docker"},
            {"category": "tool", "value": "Kubernetes"},
        ]},
        description="System must extract deployment stack",
    ))

    # --- Contradiction Handling ---
    tests.append(AcceptanceTest(
        name="contradiction_handling",
        category="contradictions",
        turns=[{
            "prompt": "Use React for the frontend.",
            "spans": [{"text": "React", "start": 8, "end": 13, "confidence": 0.95, "label": "project_info"}],
        }, {
            "prompt": "Wait, I changed my mind. Use Vue instead of React.",
            "spans": [
                {"text": "Vue", "start": 30, "end": 33, "confidence": 0.95, "label": "project_info"},
                {"text": "React", "start": 43, "end": 48, "confidence": 0.95, "label": "project_info"},
            ],
        }],
        expected_state={"active": [{"category": "framework", "value": "Vue"}]},
        description="System must handle explicit contradiction",
    ))

    return tests


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_acceptance_tests(
    tests: List[AcceptanceTest],
    high_confidence: float = 0.0,
    medium_confidence: float = 0.0,
    use_admission: bool = True,
) -> Dict[str, Any]:
    """Run all acceptance tests and return results.
    
    Args:
        tests: List of AcceptanceTest instances.
        high_confidence: High confidence threshold for engine.
        medium_confidence: Medium confidence threshold for engine.
        use_admission: Whether to use evidence-based admission for filtering.
    """
    from src.memory.admission import EvidenceBasedAdmission, AdmissionPolicy
    
    results = []
    passed = 0
    failed = 0
    
    # Create admission instance if needed
    admission = None
    if use_admission:
        policy = AdmissionPolicy(
            weight_hypothetical=10.0,  # Very strong penalty for hypothetical
            lock_threshold=0.7,
            pending_threshold=0.4,
        )
        admission = EvidenceBasedAdmission(policy=policy)

    for test in tests:
        engine = ProjectMemoryEngine(
            project_id=f"test_{test.name}",
            high_confidence_threshold=high_confidence,
            medium_confidence_threshold=medium_confidence,
        )

        result = test.run(engine, admission=admission)
        results.append(result)

        if result["passed"]:
            passed += 1
        else:
            failed += 1

    # Category summary
    category_results = {}
    for r in results:
        cat = r["category"]
        if cat not in category_results:
            category_results[cat] = {"passed": 0, "failed": 0, "total": 0}
        category_results[cat]["total"] += 1
        if r["passed"]:
            category_results[cat]["passed"] += 1
        else:
            category_results[cat]["failed"] += 1

    return {
        "total": len(tests),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / max(len(tests), 1),
        "category_results": category_results,
        "details": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Building acceptance test suite...")
    tests = build_acceptance_suite()
    logger.info(f"Built {len(tests)} acceptance tests")

    logger.info("Running acceptance tests (no admission policy)...")
    results = run_acceptance_tests(tests, high_confidence=0.0, medium_confidence=0.0)

    # Print results
    print("\n" + "=" * 60)
    print("PRODUCT ACCEPTANCE TEST RESULTS")
    print("=" * 60)
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass rate: {results['pass_rate']:.1%}")
    print()

    for cat, cat_result in results["category_results"].items():
        status = "PASS" if cat_result["failed"] == 0 else "FAIL"
        print(f"  [{status}] {cat}: {cat_result['passed']}/{cat_result['total']}")

    print()
    for detail in results["details"]:
        if not detail["passed"]:
            print(f"  FAILED: {detail['name']}")
            print(f"    Category: {detail['category']}")
            print(f"    Description: {detail['description']}")
            if detail.get("missing"):
                print(f"    Missing: {detail['missing']}")
            if detail.get("extra"):
                print(f"    Extra: {detail['extra']}")
    print("=" * 60)

    # Save results
    import os
    output_dir = "logs/acceptance_test"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write(f"# SERA Product Acceptance Test Results\n\n")
        f.write(f"**Total:** {results['total']}\n")
        f.write(f"**Passed:** {results['passed']}\n")
        f.write(f"**Failed:** {results['failed']}\n")
        f.write(f"**Pass rate:** {results['pass_rate']:.1%}\n\n")
        f.write("## By Category\n\n")
        f.write("| Category | Passed | Total | Status |\n")
        f.write("|----------|--------|-------|--------|\n")
        for cat, cat_result in results["category_results"].items():
            status = "PASS" if cat_result["failed"] == 0 else "FAIL"
            f.write(f"| {cat} | {cat_result['passed']} | {cat_result['total']} | {status} |\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
