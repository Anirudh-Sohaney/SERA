# Experiment 001: Deterministic State Engine Baseline

## Date
2026-08-30

## Description
Initial implementation of the deterministic project-memory state engine.

## What was built
- Memory schema (MemoryItem, Transition, ProjectState, MemoryCandidate)
- State matcher (exact, normalized, category-only matching)
- Transition rule engine (18 conflict patterns, negation detection, replacement detection)
- Transition engine (processes candidates, applies transitions in priority order)
- State validator (schema, source text, transition, and state validation)
- Audit logging (append-only audit log, experiment logger)
- Top-level engine (ProjectMemoryEngine orchestrator)
- Metrics computation (23 metrics including false lock rate)
- Integration layer (SERA extractor → state engine)

## Test results
- 264 unit tests: ALL PASS
- 135 fixture tests: ALL PASS
- 111 multi-turn conversation fixtures

## Modules
1. schema.py - Data types and serialization
2. matcher.py - State matching
3. rules.py - Transition classification
4. transitions.py - Transition engine
5. validator.py - State validation
6. audit.py - Audit logging
7. engine.py - Top-level orchestrator
8. metrics.py - Metrics computation
9. integration.py - Extractor integration

## Status
Baseline complete. Ready for integration testing with actual SERA extractor.
