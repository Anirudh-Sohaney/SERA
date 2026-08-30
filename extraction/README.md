# SERA Extraction

Training data and deterministic state engine for the SERA project-memory extraction pipeline.

## State Engine

The deterministic project-memory state engine (`src/memory/`) maintains an auditable, serializable record of project knowledge extracted from conversation turns.

### Quick Start

```python
from src.memory import ProjectState, MemoryItem, MemoryCategory
from src.memory.engine import ProjectMemoryEngine

# Create engine
engine = ProjectMemoryEngine(project_id="my-project")

# Process a conversation turn
result = engine.process_turn(
    prompt="Use Python with PostgreSQL.",
    extracted_spans=[
        {"text": "Python", "category": "language"},
        {"text": "PostgreSQL", "category": "database"},
    ],
    turn_number=1,
)

# Query state
state = engine.get_state()
print(f"Active memories: {len(state['active_memories'])}")

# Save to disk
engine.save("project_memory/")
```

### Core Concepts

| Concept | Description |
|---------|-------------|
| **MemoryItem** | A single fact (e.g. "PostgreSQL") with category, source provenance, and lifecycle status |
| **Transition** | An immutable record of a state change (ADD/MODIFY/REMOVE/REJECT/NO_CHANGE) |
| **ProjectState** | Full persistent state: all memories + transition log |
| **StateMatcher** | Finds matches between candidates and existing memories |
| **TransitionRuleEngine** | Classifies transition types from matches and context signals |
| **StateValidator** | Validates transitions and state consistency |
| **AuditLog** | Append-only log of every state change with full provenance |

### Documentation

- [Memory State Reference](docs/memory-state.md) — comprehensive guide covering schema, transitions, matching, validation, metrics, and examples
- [State Transition Specification](docs/state-transition-spec.md) — formal behavioral spec with 21 requirements
- [Architecture Document](docs/architecture.md) — system overview, module structure, dependencies, deployment

### Running Tests

```bash
python -m pytest tests/memory/ -v
```

## Data

### What's Here

- `data/final/` — the coding-project dataset, reports, validation checks
- `data/docs/` — schemas, citations, and processing manifest
- `data/sources/oasst2/README.md` — preserved upstream Hugging Face dataset card
- `data/guide.md` — the project-memory schema used for extraction

The final deliverable is [`data/final/dataset_coding.jsonl`](data/final/README.md).

### Sources

Individual prompts originated from [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), pinned to `7d6490e462285cf85d91eabea0f9a954fbddcd1f`. Full conversations originated from [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2), pinned to `179dd21fc55192153d94adb0e0ce8f69e222bf75`. A Kaggle candidate was reviewed but not downloaded because its description states that its dialogues were programmatically generated; see [`data/docs/sources.md`](data/docs/sources.md).

Human authorship is based on source descriptions and role labels and cannot be independently proven from message text.

### Pipeline (removed)

The raw data was processed in two LLM stages (both `gpt-5.6-luna` via the
Codex API): (1) per-prompt project-memory state tracking over all 65,389 user
prompts → 65,377 validated records; (2) binary coding-relevance filtering →
7,251 kept records. Procedure, rulesets, and validation evidence:
[`data/final/filter_report.md`](data/final/filter_report.md) and
[`data/final/README.md`](data/final/README.md).
