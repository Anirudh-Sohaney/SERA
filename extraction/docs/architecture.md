# Architecture Document

## System Overview

The SERA deterministic project-memory state engine is a pipeline that extracts project knowledge from conversation turns, maintains a persistent state, and provides full auditability.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SERA Pipeline                            │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │   Extractor   │───▶│  State Engine    │───▶│  Persistent   │  │
│  │  (E6-A Model) │    │  (Deterministic) │    │  Memory       │  │
│  └──────────────┘    └──────────────────┘    └──────────────┘  │
│                                                                 │
│  Input:              Processing:              Output:           │
│  - User prompt       - Match finding          - state.json      │
│  - Extracted spans   - Rule classification    - transitions.json│
│                      - Transition application - audit_log.jsonl │
│                      - Validation                                  │
│                      - Audit logging                              │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Extraction**: The SERA extractor (E6-A model or custom callable) processes a prompt and produces raw spans with `text`, `start`, `end`, `confidence`, and optional `category`.

2. **Candidate Building**: `build_memory_candidates()` converts raw spans into typed `MemoryCandidate` objects. Category is resolved via: field name lookup → explicit category → text heuristic.

3. **State Matching**: `StateMatcher.find_matches()` compares candidates against active memories using priority-ordered strategies (exact → normalized → category_only → none).

4. **Rule Classification**: `TransitionRuleEngine.classify()` determines the transition type (ADD/MODIFY/REMOVE/REJECT/NO_CHANGE) based on match result, conflict patterns, negation patterns, and replacement patterns.

5. **Transition Application**: `TransitionEngine.process_candidates()` applies transitions in priority order (REMOVE/REJECT → ADD/MODIFY → NO_CHANGE), mutating the `ProjectState` in-place.

6. **Validation**: `StateValidator.validate_transition()` validates each transition against invariants. Invalid transitions produce errors that block commit.

7. **Audit Logging**: `AuditLog` records every transition with full provenance (state before, state after, validation result).

8. **Persistence**: `ProjectState.save()` writes state to JSON with atomic file operations.

## Module Structure

```
src/memory/
├── __init__.py         52 lines   — Public API re-exports
├── schema.py          810 lines   — Core data types
├── matcher.py         297 lines   — State matching strategies
├── rules.py           599 lines   — Transition classification rules
├── transitions.py     663 lines   — Transition engine + candidate builder
├── validator.py       883 lines   — State and transition validation
├── audit.py           517 lines   — Audit logging + experiment logger
├── engine.py          244 lines   — Top-level orchestrator
├── metrics.py         510 lines   — Quality metrics
└── integration.py     592 lines   — Extractor integration + evaluation
```

### Module Responsibilities

| Module | Responsibility | Key Classes |
|--------|---------------|-------------|
| `schema.py` | Core data types, serialization, state mutations | `MemoryItem`, `Transition`, `ProjectState`, `MemoryCandidate` |
| `matcher.py` | Deterministic matching of candidates to existing memories | `StateMatcher`, `MatchResult`, `normalize_text` |
| `rules.py` | Classify transition type from match + context signals | `TransitionRuleEngine`, `RuleResult`, `detect_negation_context`, `detect_replacement_context` |
| `transitions.py` | Orchestrate matching + classification + state mutation | `TransitionEngine`, `build_memory_candidates` |
| `validator.py` | Validate transitions and state consistency | `StateValidator`, `Result`, `validate_no_duplicates`, `validate_state_consistency` |
| `audit.py` | Append-only audit logging | `AuditLog`, `AuditRecord`, `ExperimentLogger` |
| `engine.py` | Top-level orchestrator connecting all modules | `ProjectMemoryEngine` |
| `metrics.py` | Compute extraction/transition/state quality metrics | `StateMetrics`, `compute_metrics`, `compare_states` |
| `integration.py` | Connect SERA extractor to state engine, run evaluations | `SERAIntegration`, `load_sera_extractor`, `run_full_evaluation` |

## Dependencies

### Internal Dependencies

```
engine.py
├── audit.py (AuditLog, AuditRecord)
├── schema.py (MemoryCandidate, ProjectState, Transition, TransitionType)
├── transitions.py (TransitionEngine, build_memory_candidates)
└── validator.py (StateValidator, Result)

transitions.py
├── schema.py (MemoryCandidate, MemoryCategory, MemoryItem, MemoryStatus, ProjectState, Transition, TransitionType)
├── matcher.py (MatchResult, StateMatcher)
└── rules.py (TransitionRuleEngine)

matcher.py
└── schema.py (MemoryCandidate, MemoryItem, ProjectState)

rules.py
├── schema.py (MemoryCandidate, ProjectState, TransitionType)
└── matcher.py (MatchResult)

validator.py
└── schema.py (MemoryCategory, MemoryItem, MemoryStatus, ProjectState, Transition, TransitionType)

metrics.py
├── schema.py (MemoryCategory, MemoryItem, MemoryStatus, ProjectState, Transition, TransitionType)
└── validator.py (_normalize_value)

integration.py
├── audit.py (AuditLog, ExperimentLogger)
├── engine.py (ProjectMemoryEngine)
├── metrics.py (StateMetrics, compare_states, compute_metrics, format_metrics)
└── schema.py (MemoryCategory, ProjectState, Transition, TransitionType)
```

### External Dependencies

| Package | Used By | Purpose |
|---------|---------|---------|
| `json` | All modules | Serialization |
| `uuid` | `schema.py`, `audit.py` | ID generation |
| `datetime` | `schema.py`, `audit.py` | Timestamps |
| `re` | `matcher.py`, `rules.py`, `transitions.py` | Pattern matching |
| `unicodedata` | `matcher.py`, `validator.py` | Text normalization |
| `os` | `schema.py`, `engine.py`, `audit.py`, `integration.py` | File I/O |
| `tempfile` | `schema.py`, `audit.py`, `engine.py`, `integration.py` | Atomic writes |
| `torch` | `integration.py` | ML model inference (optional) |
| `transformers` | `integration.py` | Token classification model (optional) |

## Configuration

### Pattern Configuration

The `TransitionRuleEngine` accepts optional pattern overrides:

```python
engine = TransitionRuleEngine(
    conflict_patterns=[...],      # Override DEFAULT_CONFLICT_PATTERNS
    negation_patterns=[...],      # Override DEFAULT_NEGATION_PATTERNS
    replacement_patterns=[...],   # Override DEFAULT_REPLACEMENT_PATTERNS
)
```

New patterns can be appended at runtime:

```python
engine.conflict_patterns.append({
    "pattern": r"\brefactor\b",
    "transition_type": TransitionType.MODIFY,
    "description": "Refactoring signal.",
})
```

### Category Inference Configuration

Category resolution order:
1. Field name → `FIELD_CATEGORY_MAP` lookup
2. Explicit `category` key in span
3. Heuristic text-based inference via `_infer_category_from_text()`

Known technology lookup tables:
- `KNOWN_LANGUAGES` — 50+ programming languages
- `KNOWN_FRAMEWORKS` — 90+ frameworks and libraries
- `KNOWN_DATABASES` — 30+ database systems
- `KNOWN_PLATFORMS` — 40+ cloud platforms and services

### Matcher Configuration

Matching strategies are fixed priority:
1. `exact` (confidence: 1.0) — case/whitespace normalization
2. `normalized` (confidence: 0.95) — punctuation normalization
3. `category_only` (confidence: 0.7) — substring containment (min 3 chars)
4. `none` (confidence: 0.0) — no match

### Validation Configuration

Validation is strict-by-default. The `StateValidator` has no configuration options — all rules are enforced. Warnings are non-blocking; errors block commit.

## Deployment Considerations

### File Layout

```
project_directory/
├── state.json              — Full ProjectState (atomic writes)
├── transitions.json        — Transition log
├── audit_log.jsonl         — Audit log (JSONL format)
└── experiment_output/      — Evaluation results
    ├── config.json
    ├── metrics.json
    ├── per_fixture.json
    ├── failures.jsonl
    ├── summary.md
    └── audit_log.jsonl
```

### Atomicity

All file writes use atomic operations:
- `ProjectState.save()` writes to a temp file, then `os.replace()`
- `ExperimentLogger` uses `tempfile.mkstemp()` + `os.replace()` for all outputs
- On failure, temp files are cleaned up

### Serialization Format

- `state.json`: Full `ProjectState` as JSON (2-space indentation, Unicode preserved)
- `transitions.json`: List of `Transition` dicts
- `audit_log.jsonl`: One `AuditRecord` per line (compact JSON)

### Recovery

To restore state:
```python
engine = ProjectMemoryEngine.load("path/to/directory")
```

This loads `state.json` and `audit_log.jsonl`, reconstructing the full state and audit history.

### Evaluation

To run evaluation:
```python
from src.memory.integration import run_full_evaluation

result = run_full_evaluation(
    extractor=load_sera_extractor("checkpoints/"),
    fixtures=test_fixtures,
    output_dir="logs/experiment_001/",
)
```

Produces:
- `config.json` — extractor and fixture configuration
- `metrics.json` — aggregate metrics
- `per_fixture.json` — per-fixture results
- `audit_log.jsonl` — combined audit log
- `summary.md` — human-readable summary
