# State Transition Specification

Formal behavioral specification for the SERA deterministic project-memory state engine. Every rule maps to a documented requirement.

## Requirements

### REQ-001: Every memory item must have a unique ID

**Description**: Every `MemoryItem` must have a globally unique `memory_id` (UUID4). No two memory items may share the same ID.

**Rule**: `schema.py:75-77` — `_new_id()` generates UUID4 strings. `schema.py:108` — `memory_id` defaults to `_new_id()`.

**Test**: `test_state_engine.py::TestMemoryItemCreation::test_memory_item_creation` — verifies `memory_id` is a non-empty string.

---

### REQ-002: Every memory item must have a category from the approved list

**Description**: Every `MemoryItem.category` must be a valid `MemoryCategory` enum value. The approved list is defined in `schema.py:32-54`.

**Rule**: `schema.py:32-54` — `MemoryCategory` enum definition. `schema.py:141-147` — `from_dict` validates category type.

**Test**: `test_state_engine.py::TestMemoryItemCreation::test_memory_item_from_invalid_category_raises` — verifies `ValueError` for invalid category.

---

### REQ-003: Every memory item must retain source provenance

**Description**: Every `MemoryItem` must preserve the original source context: `source_text` (exact substring), `source_start`/`source_end` (character offsets), and `prompt_text` (full prompt). This enables auditability and traceability.

**Rule**: `schema.py:86-119` — `MemoryItem` dataclass includes all source fields. `validator.py:420-505` — `validate_source_text` verifies `prompt_text[source_start:source_end] == source_text`.

**Test**: `test_state_engine.py::TestValidator::test_validate_source_text` — verifies source text consistency.

---

### REQ-004: ADD creates a new active memory item

**Description**: An ADD transition must create a new `MemoryItem` with `status=ACTIVE` and append it to both `active_memories` and `all_memories`.

**Rule**: `schema.py:505-557` — `ProjectState.add_memory()` appends new items to both lists.

**Test**: `test_state_engine.py::TestProjectStateCreation::test_project_state_add_memory` — verifies `active_memories` and `all_memories` grow by 1.

---

### REQ-005: MODIFY changes an existing active memory item

**Description**: A MODIFY transition must update the value of an existing ACTIVE memory item. The `old_value` must match the current value before modification.

**Rule**: `schema.py:638-699` — `ProjectState.modify_memory()` updates value and records old_value.

**Test**: `test_state_engine.py::TestProjectStateCreation::test_project_state_modify_memory` — verifies value is updated and old_value is recorded.

---

### REQ-006: REMOVE deactivates an existing active memory

**Description**: A REMOVE transition must set an ACTIVE memory's status to REMOVED and remove it from `active_memories`. The item remains in `all_memories`.

**Rule**: `schema.py:559-595` — `ProjectState.remove_memory()` sets status to REMOVED, filters from `active_memories`.

**Test**: `test_state_engine.py::TestProjectStateCreation::test_project_state_remove_memory` — verifies `active_memories` shrinks by 1, `all_memories` unchanged.

---

### REQ-007: REJECT records that a value was explicitly rejected

**Description**: A REJECT transition must set an ACTIVE memory's status to REJECTED. If no matching memory exists, a new REJECTED item is created in `all_memories` (for audit trail).

**Rule**: `schema.py:597-636` — `ProjectState.reject_memory()`. `transitions.py:523-592` — `_apply_reject()` handles both cases.

**Test**: `test_state_engine.py::TestProjectStateCreation::test_project_state_reject_memory` — verifies status is REJECTED.

---

### REQ-008: NO_CHANGE means no state change occurred

**Description**: A NO_CHANGE transition must not modify any memory item's value or status. It records an audit entry and refreshes the matched memory's `updated_turn`.

**Rule**: `transitions.py:594-638` — `_apply_no_change()` appends transition log, refreshes `updated_turn`.

**Test**: `test_state_engine.py::TestTransitions::test_process_no_change` — verifies `NO_CHANGE` transition is produced.

---

### REQ-009: Transitions must be processed in order: REMOVE/REJECT first, then ADD/MODIFY

**Description**: When processing a batch of candidates, REMOVE and REJECT transitions must be applied before ADD and MODIFY transitions. This prevents conflicts when replacing values.

**Rule**: `transitions.py:353-356` — `process_candidates()` sorts classified candidates by `_transition_priority()`. `transitions.py:644-663` — priority map: REMOVE=0, REJECT=1, ADD=2, MODIFY=3, NO_CHANGE=4.

**Test**: `test_state_engine.py::TestTransitions::test_remove_before_add` — verifies processing order.

---

### REQ-010: No duplicate active memories for same category+value

**Description**: At most one memory item may be ACTIVE for any given (category, normalized_value) pair. Adding a duplicate must produce a NO_CHANGE transition instead of a new item.

**Rule**: `schema.py:518-523` — `add_memory()` checks for existing match before adding. `validator.py:773-807` — `validate_no_duplicates()` checks for duplicates.

**Test**: `test_state_engine.py::TestProjectStateCreation::test_project_state_add_duplicate_returns_no_change` — verifies duplicate ADD produces NO_CHANGE.

---

### REQ-011: Every state change must produce an audit record

**Description**: Every transition (ADD, MODIFY, REMOVE, REJECT, NO_CHANGE) must produce an `AuditRecord` capturing the transition, state before, state after, and validation result.

**Rule**: `engine.py:83-114` — `process_turn()` creates an `AuditRecord` for every transition.

**Test**: `test_state_engine.py::TestAuditLog::test_audit_log_add_record` — verifies records are appended.

---

### REQ-012: State must be serializable and recoverable

**Description**: `ProjectState` must be serializable to JSON and restorable from JSON. All enum values must roundtrip correctly. Atomic writes must be used to prevent corruption.

**Rule**: `schema.py:121-179` — `MemoryItem.to_dict()`/`from_dict()`, `to_json()`/`from_json()`. `schema.py:434-476` — `ProjectState.save()`/`load()` with atomic write.

**Test**: `test_state_engine.py::TestMemoryItemCreation::test_memory_item_serialization_roundtrip`, `test_project_state_save_load_roundtrip`.

---

### REQ-013: Negation patterns must be detected deterministically

**Description**: Negation patterns (don't, do not, never, avoid, without, no) must be detected deterministically in the prompt text relative to a value. "NoSQL" must not match "no" + "SQL".

**Rule**: `rules.py:134-140` — `DEFAULT_NEGATION_PATTERNS`. `rules.py:225-260` — `detect_negation_context()` with special "no" word-boundary check.

**Test**: `test_state_engine.py::TestRules::test_negation_detection`, `test_negation_detection_nosql_false_positive`.

---

### REQ-014: Replacement patterns must be detected deterministically

**Description**: Replacement patterns ("Use X instead of Y", "Switch from Y to X", "Replace Y with X") must be detected deterministically with case-insensitive matching.

**Rule**: `rules.py:143-168` — `DEFAULT_REPLACEMENT_PATTERNS`. `rules.py:263-289` — `detect_replacement_context()`.

**Test**: `test_state_engine.py::TestRules::test_replacement_detection`, `test_replacement_detection_switch_to`, `test_replacement_detection_replace_with`.

---

### REQ-015: Ambiguous statements must not auto-add to memory

**Description**: When a candidate matches an existing memory only by category (substring match), and no conflict/negation signal is present, the engine must default to ADD (new value in same category) rather than MODIFY. The category_only match is lower confidence (0.7).

**Rule**: `rules.py:507-577` — `_classify_category_only()` defaults to ADD when no signals are detected.

**Test**: `test_state_engine.py::TestRules::test_category_only_add`.

---

### REQ-016: Source text must be recoverable from character offsets

**Description**: Given `source_start`, `source_end`, and `prompt_text`, the original `source_text` must be exactly recoverable via `prompt_text[source_start:source_end]`.

**Rule**: `validator.py:215-225` — validates `prompt_text[source_start:source_end] == source_text`. `validator.py:420-505` — `validate_source_text()` performs full verification.

**Test**: `test_state_engine.py::TestValidator::test_validate_source_text`, `test_validate_source_text_mismatch`.

---

### REQ-017: Validation must fail loudly on invalid transitions

**Description**: Invalid transitions must produce validation errors, never silent repairs. The caller must check `result.valid` before proceeding.

**Rule**: `validator.py:122-242` — `StateValidator.validate_transition()` returns `Result` with `valid=False` on any error.

**Test**: `test_state_engine.py::TestValidator::test_validate_invalid_add`, `test_validate_invalid_modify`, `test_validate_invalid_modify_nonexistent_memory`.

---

### REQ-018: The engine must be extensible with new patterns

**Description**: New conflict, negation, and replacement patterns must be addable at runtime without modifying control flow. Patterns are stored as data structures.

**Rule**: `rules.py:296-333` — `TransitionRuleEngine.__init__()` accepts optional pattern overrides. Patterns are lists of dicts that can be appended to at runtime.

**Test**: `test_state_engine.py::TestRules::test_conflict_patterns`, `test_conflict_patterns_remove`, `test_conflict_patterns_none`.

---

### REQ-019: Every test must have a unique name

**Description**: Every test method and test fixture must have a unique, descriptive name that identifies the scenario being tested.

**Rule**: Enforced by pytest collection. Test names in `test_state_engine.py` follow `test_<verb>_<scenario>` convention. Fixture names in `test_fixtures.py` follow `<action>_<category>_<value>` convention.

**Test**: Verified by pytest's test collection (no duplicate test IDs allowed).

---

### REQ-020: Metrics must be computed separately for extraction, transitions, and state

**Description**: Quality metrics must be computed at three independent levels: extraction (span-level), transition-level, and state-level. Each level has its own precision, recall, and F1 metrics.

**Rule**: `metrics.py:103-196` — `compute_metrics()` computes all three levels independently. `metrics.py:30-82` — `StateMetrics` dataclass holds all metrics.

**Test**: `test_state_engine.py::TestIntegration::test_simple_add_flow` — validates state after operations (metrics validated via `StateMetrics` fields).

---

### REQ-021: False lock rate must be tracked

**Description**: The fraction of predicted ADDs that are not in gold ADDs must be tracked as `false_lock_rate`. This measures the engine's tendency to incorrectly lock in values.

**Rule**: `metrics.py:363-380` — `_compute_false_lock_rate()`.

**Test**: `test_state_engine.py::TestIntegration::test_multi_category_flow` — full flow validation.
