# SERA Deterministic Project-Memory State Engine

## Purpose

The SERA state engine maintains a deterministic, auditable record of project knowledge extracted from conversation turns. It tracks what technologies, tools, requirements, and constraints a project uses — and provides a structured way to add, modify, remove, or reject those facts as the conversation evolves.

The engine guarantees:
- **Determinism**: same inputs always produce the same state.
- **Auditability**: every state change is recorded with full provenance.
- **Recoverability**: state is serializable to JSON and restorable across sessions.
- **No silent mutations**: every change produces an explicit transition record.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Conversation Turn                      │
│  prompt + extracted spans                                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  1. Candidate Builder (build_memory_candidates)          │
│     Raw spans → typed MemoryCandidate objects            │
│     Category resolution: field > explicit > heuristic    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  2. State Matcher (StateMatcher.find_matches)            │
│     Compare candidates against active memories           │
│     Strategies: exact → normalized → category_only → none│
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  3. Rule Engine (TransitionRuleEngine.classify)           │
│     Determine transition type from match + context       │
│     Detects: negation, replacement, conflict patterns    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  4. Transition Engine (TransitionEngine.process)         │
│     Apply transitions in priority order:                 │
│     REMOVE/REJECT → ADD/MODIFY → NO_CHANGE              │
│     Mutates ProjectState in-place                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  5. Validator (StateValidator.validate_transition)       │
│     Validate invariants before/after commit              │
│     Fails loudly on invalid transitions                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  6. Audit Log (AuditLog + AuditRecord)                   │
│     Record: transition, state_before, state_after        │
│     Append-only, serializable to JSONL                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  7. Persistent State (ProjectState.save/load)            │
│     JSON serialization with atomic writes                │
└──────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/memory/
├── __init__.py         — Public API re-exports
├── schema.py           — Core data types (MemoryItem, Transition, ProjectState, MemoryCandidate)
├── matcher.py          — State matching (exact/normalized/substring strategies)
├── rules.py            — Transition classification rules
├── transitions.py      — Transition engine + candidate builder
├── validator.py        — State and transition validation
├── audit.py            — Audit logging + experiment logger
├── engine.py           — Top-level ProjectMemoryEngine orchestrator
├── metrics.py          — Extraction, transition, and state quality metrics
└── integration.py      — SERA extractor integration + evaluation runner
```

## Memory Schema

### MemoryItem

A single piece of persistent project memory representing a canonical fact.

| Field | Type | Description |
|-------|------|-------------|
| `memory_id` | `str` | Globally unique identifier (UUID4, auto-generated) |
| `category` | `MemoryCategory` | Semantic category (LANGUAGE, DATABASE, etc.) |
| `value` | `str` | Canonical normalized value (e.g. "PostgreSQL") |
| `source_text` | `str` | Exact substring from the original prompt |
| `source_start` | `int` | Character offset start in the original prompt |
| `source_end` | `int` | Character offset end in the original prompt |
| `prompt_text` | `str` | Full original prompt text |
| `status` | `MemoryStatus` | Lifecycle status (active, removed, rejected) |
| `created_turn` | `int` | Conversation turn when first added |
| `updated_turn` | `int` | Conversation turn when last modified |
| `confidence` | `float` | Model confidence (0.0–1.0) |
| `metadata` | `dict` | Arbitrary key-value metadata |

### MemoryCategory

Canonical categories for project knowledge:

| Category | Example |
|----------|---------|
| `PROJECT` | "SERA extraction pipeline" |
| `LANGUAGE` | "Python", "TypeScript" |
| `FRAMEWORK` | "FastAPI", "React" |
| `LIBRARY` | "Redis", "Tailwind CSS" |
| `DATABASE` | "PostgreSQL", "MongoDB" |
| `PLATFORM` | "AWS", "Vercel" |
| `DEPLOYMENT` | "Docker", "Kubernetes" |
| `ARCHITECTURE` | "microservices" |
| `DIRECTORY` | "src/components/" |
| `FILE` | ".env" |
| `INTERFACE` | "REST", "gRPC" |
| `INPUT` | "CSV", "JSON" |
| `OUTPUT` | "JSON", "PDF" |
| `CONSTRAINT` | "must be FERPA compliant" |
| `REQUIREMENT` | "authentication" |
| `DESIGN` | "CLI tool" |
| `TOOL` | "Webpack", "pytest" |
| `RUNTIME` | "Node.js 20" |
| `TESTING` | "Jest", "pytest" |
| `CONFIGURATION` | "eslint.config.js" |

### MemoryStatus

| Status | Description |
|--------|-------------|
| `ACTIVE` | Currently valid and tracked |
| `REMOVED` | Was previously active, now superseded or deleted |
| `REJECTED` | Explicitly determined to be incorrect or unwanted |

### Transition

An immutable record of a state change. Transitions are append-only and serve as the source of truth.

| Field | Type | Description |
|-------|------|-------------|
| `transition_id` | `str` | UUID4 unique identifier |
| `transition_type` | `TransitionType` | ADD, MODIFY, REMOVE, REJECT, NO_CHANGE |
| `category` | `MemoryCategory` | Category of the affected memory |
| `value` | `str` | New value (ADD/MODIFY) or value being removed/rejected |
| `old_value` | `Optional[str]` | Previous value (MODIFY/REMOVE) |
| `memory_id` | `Optional[str]` | Affected memory item's ID (MODIFY/REMOVE/REJECT) |
| `source_text` | `str` | Exact substring from original prompt |
| `source_start` | `int` | Character offset start |
| `source_end` | `int` | Character offset end |
| `prompt_text` | `str` | Full original prompt text |
| `turn_number` | `int` | Conversation turn that produced this transition |
| `timestamp` | `str` | ISO 8601 UTC timestamp |
| `confidence` | `float` | Model confidence (0.0–1.0) |
| `metadata` | `dict` | Arbitrary metadata |

### ProjectState

The full persistent state for a project's memory.

| Field | Type | Description |
|-------|------|-------------|
| `project_id` | `str` | Unique project identifier |
| `active_memories` | `List[MemoryItem]` | Memories with status == ACTIVE |
| `all_memories` | `List[MemoryItem]` | Every memory ever created (audit trail) |
| `transition_log` | `List[Transition]` | Append-only log of all transitions |
| `current_turn` | `int` | Current conversation turn number |

### MemoryCandidate

An extracted span awaiting evaluation against current state.

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Extracted text span |
| `category` | `MemoryCategory` | Semantic category |
| `start` | `int` | Character offset start |
| `end` | `int` | Character offset end |
| `prompt_text` | `str` | Full original prompt text |
| `confidence` | `float` | Model confidence (0.0–1.0) |
| `turn_number` | `int` | Conversation turn number |

## Transition Types

### ADD

Creates a new active memory item. Triggers when a candidate has no match in the current state and is not negated.

```
Prompt: "Use Python."
Result: ADD(LANGUAGE, "Python")
State:  active_memories = [Python]
```

### MODIFY

Updates an existing active memory's value. Triggers when:
- An exact/normalized match exists AND a conflict pattern is detected (e.g. "instead", "switch to")
- A normalized match exists but the surface text differs

```
State:  active_memories = [Python]
Prompt: "Use Python 3.12 instead."
Result: MODIFY(LANGUAGE, "Python" → "Python 3.12")
State:  active_memories = [Python 3.12]
```

### REMOVE

Deactivates an existing active memory. Triggers when:
- A negation pattern is detected for an existing memory value (e.g. "don't use Docker")
- A removal conflict pattern is detected (e.g. "Remove Docker", "stop using Docker")

```
State:  active_memories = [Docker]
Prompt: "We don't use Docker."
Result: REMOVE(TOOL, "Docker")
State:  active_memories = []
```

### REJECT

Records that a value was explicitly rejected. Triggers when:
- A negation pattern is detected for a candidate with no existing match

```
State:  active_memories = []
Prompt: "Do not use MongoDB."
Result: REJECT(DATABASE, "MongoDB")
State:  all_memories = [MongoDB (rejected)]
```

### NO_CHANGE

Records that no state change occurred. Triggers when:
- A candidate exactly matches an existing active memory value

```
State:  active_memories = [Python]
Prompt: "Python is still the language."
Result: NO_CHANGE(LANGUAGE, "Python")
State:  active_memories = [Python]  (unchanged)
```

## State Transitions with Examples

### Single-Turn Lifecycle

```python
from src.memory import ProjectState, MemoryItem, MemoryCategory

state = ProjectState()

# Turn 1: Add Python
item = MemoryItem(
    category=MemoryCategory.LANGUAGE,
    value="Python",
    source_text="Python",
    source_start=8, source_end=14,
    prompt_text="Use Python for backend.",
    created_turn=1, updated_turn=1,
)
transition = state.add_memory(item)
# transition.transition_type == TransitionType.ADD
# state.active_memories == [MemoryItem(LANGUAGE, "Python")]

# Turn 1 again: Reaffirm Python (duplicate)
item2 = MemoryItem(
    category=MemoryCategory.LANGUAGE,
    value="Python",
    source_text="Python",
    source_start=0, source_end=6,
    prompt_text="Python is the language.",
    created_turn=1, updated_turn=1,
)
transition = state.add_memory(item2)
# transition.transition_type == TransitionType.NO_CHANGE
# state.active_memories still has 1 item
```

### Multi-Turn Lifecycle

```python
state = ProjectState()

# Turn 1: Add Python
item = _make_item(category=LANGUAGE, value="Python", turn=1)
state.add_memory(item)

# Turn 2: Modify to Python 3.12
state.modify_memory(
    memory_id=item.memory_id,
    new_value="Python 3.12",
    turn=2,
)
# state.active_memories[0].value == "Python 3.12"

# Turn 3: Remove Python
state.remove_memory(item.memory_id, turn=3)
# state.active_memories == []
# state.all_memories[0].status == MemoryStatus.REMOVED
```

## Conflict Handling Rules

The rule engine detects conflict patterns in the full prompt text. These patterns are regex-based and checked against the prompt before transition classification.

### Replacement Signals (→ MODIFY)

| Pattern | Example |
|---------|---------|
| `\binstead\b` | "Use Python instead" |
| `\bactually\b` | "Actually, use Python 3.12" |
| `\bswitch\s+to\b` | "Switch to Python 3.12" |
| `\breplace\b` | "Replace Flask with FastAPI" |
| `\bchange\s+from\b` | "Change from Flask to FastAPI" |
| `\binstead\s+of\b` | "Use Python instead of Ruby" |

### Removal Signals (→ REMOVE)

| Pattern | Example |
|---------|---------|
| `\bno\s+longer\b` | "No longer using Docker" |
| `\bremove\b` | "Remove Docker" |
| `\bdrop\b` | "Drop the auth requirement" |
| `\bdiscard\b` | "Discard Jest" |
| `\bstop\s+using\b` | "Stop using Kubernetes" |
| `\bnot\s+needed\b` | "PostgreSQL not needed" |
| `\bcancelled?\b` | "Cancelled the deployment" |

### Rejection Signals (→ REJECT)

| Pattern | Example |
|---------|---------|
| `\bdon'?t\s+use\b` | "Don't use Django" |
| `\bdo\s+not\s+use\b` | "Do not use MongoDB" |
| `\bavoid\b` | "Avoid Redux" |
| `\bnot\s+required\b` | "GraphQL not required" |

### Negation Detection

Negation patterns are applied to the full prompt with the value substituted in:

```
\bdon'?t\b.*{value}     →  "Don't use Docker"
\bdo\s+not\b.*{value}   →  "Do not use Docker"
\bnever\b.*{value}      →  "Never use MySQL"
\bavoid\b.*{value}      →  "Avoid SQLite"
\bwithout\b.*{value}    →  "without PostgreSQL"
\bno\b\s+{value}        →  "no Docker"  (with word boundary)
```

**Special case**: "NoSQL" does NOT match "no" + "SQL" because the "no" check requires a word boundary and a space before the value.

### Replacement Pattern Detection

Replacement patterns extract both the old and new values:

| Pattern | Captures |
|---------|----------|
| `Use\s+{new}\s+instead\s+of\s+{old}` | old, new |
| `Switch\s+(?:from\s+)?{old}\s+to\s+{new}` | old, new |
| `Replace\s+{old}\s+with\s+{new}` | old, new |
| Case-insensitive variants of above | old, new |

## Source Provenance

Every memory item preserves the full source context:

1. **source_text**: The exact substring extracted from the prompt.
2. **source_start / source_end**: Character offsets into the prompt.
3. **prompt_text**: The complete original prompt.

This enables:
- **Auditability**: Verify exactly what text was extracted and from where.
- **Reproducibility**: Re-extract the same spans from the original prompt.
- **Validation**: Confirm `prompt_text[source_start:source_end] == source_text`.

## Matching Strategies

The `StateMatcher` evaluates candidates against active memories using a priority-ordered strategy:

| Priority | Strategy | Confidence | Description |
|----------|----------|------------|-------------|
| 0 | `exact` | 1.0 | Identical after case/whitespace normalization |
| 1 | `normalized` | 0.95 | Identical after punctuation normalization |
| 2 | `category_only` | 0.7 | Same category, one is a substring of the other (min 3 chars) |
| 3 | `none` | 0.0 | No match found |

**Normalization steps** (for `exact`):
1. Unicode NFKD normalization
2. Lowercase
3. Strip leading/trailing whitespace
4. Collapse internal whitespace to single space
5. Strip trailing punctuation (`.`, `,`, `;`, `:`, `!`, `?`)

**Punctuation normalization** (for `normalized`):
- Additionally strips: `-`, `_`, `/`, `\`, `(`, `)`, `{`, `}`, `[`, `]`, `<`, `>`, `@`, `#`, `$`, `%`, `^`, `&`, `*`, `+`, `=`, `|`, `~`, `` ` ``, `"`, `'`

## Validation Rules

The `StateValidator` enforces these invariants:

### Transition Validation

- `turn_number` must be > 0
- `source_text` consistency: if provided, `source_start >= 0`, `source_end > source_start`, offsets within `prompt_text` bounds, and `prompt_text[source_start:source_end] == source_text`

### Type-Specific Rules

| Type | Rules |
|------|-------|
| ADD | `memory_id` must be None; `value` non-empty |
| MODIFY | `memory_id` must exist and be ACTIVE; `old_value` matches current; `value` non-empty |
| REMOVE | `memory_id` must exist and be ACTIVE |
| REJECT | `memory_id` must exist and be ACTIVE, or None (rejecting never-added value); if None, `value` non-empty |
| NO_CHANGE | `memory_id` must exist and be ACTIVE; `value` matches current |

### Post-Transition Checks

- No duplicate active memories for the same category + normalized value after the transition.

### State Validation

- No duplicate active memories (same category + normalized value)
- All active memories have valid schema
- Transition log is ordered by turn_number (monotonic)
- Every MODIFY/REMOVE/REJECT references a valid memory_id in all_memories
- No memory is both active and removed/rejected
- Every active memory is also in all_memories
- Every active memory has status == ACTIVE
- No orphaned transition references

## Failure Modes

### Validation Failures

The validator returns a `Result` with `valid=False` and a list of `ValidationError` objects. Each error has:
- `field`: Dot-notation path (e.g. `"transition.memory_id"`)
- `message`: Human-readable description
- `severity`: `"error"` (blocks commit) or `"warning"` (informational)

### Common Failure Scenarios

| Scenario | Error |
|----------|-------|
| ADD with non-None memory_id | "ADD transition must have memory_id=None" |
| MODIFY with missing memory_id | "MODIFY transition must have a non-null memory_id" |
| MODIFY referencing inactive memory | "MODIFY references memory_id with status removed, expected ACTIVE" |
| MODIFY with wrong old_value | "old_value does not match current value" |
| REMOVE referencing missing memory | "REMOVE references memory_id not found in state" |
| turn_number <= 0 | "turn_number must be > 0" |
| source_end <= source_start | "source_end must be > source_start" |
| source offsets exceed prompt length | "source_start exceeds prompt_text length" |
| Source text mismatch | "prompt_text[source_start:source_end] does not match source_text" |

### Atomicity Guarantees

- `ProjectState.save()` uses atomic write (write to temp file, then `os.replace`)
- `ExperimentLogger` uses atomic writes for all output files
- On failure, temp files are cleaned up

## Metrics

Metrics are computed separately at three levels:

### Extraction Metrics (Span-Level)

| Metric | Description |
|--------|-------------|
| `span_precision` | Fraction of predicted spans that are correct |
| `span_recall` | Fraction of gold spans that were predicted |
| `span_f1` | Harmonic mean of precision and recall |

### Transition Metrics

| Metric | Description |
|--------|-------------|
| `transition_accuracy` | Fraction of gold transitions that were predicted |
| `transition_precision` | Fraction of predicted transitions that are correct |
| `transition_recall` | Fraction of gold transitions that were predicted |

### State Metrics

| Metric | Description |
|--------|-------------|
| `state_precision` | Fraction of predicted active memories that are correct |
| `state_recall` | Fraction of gold active memories that were predicted |
| `state_f1` | Harmonic mean of precision and recall |

### Error Rates

| Rate | Description |
|------|-------------|
| `false_lock_rate` | (false ADDs) / (total predicted ADDs) |
| `false_update_rate` | (incorrect MODIFYs) / (total predicted MODIFYs) |
| `false_removal_rate` | (incorrect REMOVEs) / (total predicted REMOVEs) |
| `false_rejection_rate` | (incorrect REJECTs) / (total predicted REJECTs) |
| `stale_memory_rate` | Active memories that should be removed/rejected |
| `duplicate_memory_rate` | Active memories with duplicate category+value |
| `contradiction_rate` | Transitions that contradict each other in the same turn |

## Testing Strategy

### Unit Tests (`test_state_engine.py`)

- Schema types: creation, serialization roundtrips, equality, hashing
- Matcher: exact, normalized, category_only, no_match, edge cases
- Rules: ADD/MODIFY/REMOVE/REJECT classification, negation detection, replacement detection
- Transitions: single operations, priority ordering, candidate building
- Validator: all transition types, schema validation, source text validation, state consistency
- Audit: record creation, serialization, query by turn/type/category
- Integration: full multi-turn flows (add→modify, add→remove, add→reject, multi-category)

### Fixture Tests (`test_fixtures.py`)

100+ multi-turn conversation fixtures covering:
- Simple ADD (15+ fixtures across all categories)
- MODIFY (15+ fixtures with replacement signals)
- REMOVE (10+ fixtures with negation/removal signals)
- REJECT (10+ fixtures with negation signals)
- NO_CHANGE (10+ fixtures with reaffirmation)
- Multi-category scenarios
- Complex multi-turn flows
- Ambiguity handling
- Edge cases

### Running Tests

```bash
cd /root/projects/sera_models/extraction
python -m pytest tests/memory/ -v
```

## Future Learned-Model Interface

The architecture is designed for extensibility:

1. **Rule extensibility**: New conflict, negation, and replacement patterns can be added at runtime by appending to `conflict_patterns`, `negation_patterns`, or `replacement_patterns` on the `TransitionRuleEngine`.

2. **Matcher extensibility**: The `StateMatcher` can be subclassed or replaced to add fuzzy matching, embedding-based similarity, or learned matching strategies.

3. **Category inference**: The `_infer_category_from_text` function and `FIELD_CATEGORY_MAP` can be extended with new mappings or replaced with a learned classifier.

4. **Integration point**: `load_sera_extractor` loads a HuggingFace token classification model (E6-A) that produces raw spans. This can be swapped for any extraction model that produces spans with `text`, `start`, `end`, `confidence`, and optionally `category`.

5. **Evaluation**: `run_full_evaluation` accepts any extractor callable, making it easy to evaluate different extraction models against the same fixtures and metrics.

6. **Metrics**: `compute_metrics` compares predicted vs. expected states and can be extended with new metric functions by adding to the `StateMetrics` dataclass.
