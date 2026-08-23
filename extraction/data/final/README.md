# Coding-Project Conversation Dataset

**Final deliverable: `dataset_coding.jsonl`** — 7,251 coding/project-associated
records extracted from human-authored conversations.

## Provenance & citation

| Stage | Detail |
|---|---|
| Source data | [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) (English conversation trees), source revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`; 32,687 conversations / 65,389 user prompts |
| Extraction | `gpt-5.6-luna` (medium reasoning) via the Codex Responses WebSocket API, following the schema in [`../guide.md`](../guide.md); produced 65,377 validated state-tracking records over ~22.9 h (2026-08-19 → 2026-08-20) |
| Filtering | Every record's user prompt binary-classified (coding/project-associated vs not) by `gpt-5.6-luna` (low reasoning), 2026-08-22 → 2026-08-23; 100% two-pass pilot agreement, 0 invalid parses across all 65,377 calls; kept 7,251 (11.09%) |
| Validation | Independent checks at every 10% of both runs (`checks/`), final assembly check CLEAN — see [`filter_report.md`](filter_report.md) |

Raw source data and intermediate artifacts were removed after completion;
the pipeline and decisions are documented in `filter_report.md`,
`filter_stats.json`, and `../docs/`.

## Files

- `dataset_coding.jsonl` — **the dataset**: one JSON record per line
- `filter_report.md` — filtering procedure, timeline, and results
- `filter_stats.json` — filter run summary
- `supervision.log`, `filter_supervision.log` — autonomous monitor logs for both runs
- `checks/` — independent validation reports (10% milestones + finals)
- `dataset_stats.json` — full-extraction run summary

## Record format

Each line is a JSON object with exactly these fields:

| field             | type   | description                                                        |
|-------------------|--------|--------------------------------------------------------------------|
| `conversation_id` | string | source conversation id (`<tree>:<branch>`)                          |
| `turn`            | int    | index of the user prompt within the conversation                    |
| `type`            | string | `new`, `update`, `no_change`, or `no_project`                       |
| `input`           | object | `user_prompt` + the project state that existed *before* the prompt  |
| `output`          | object | the project state *after* the prompt                                |

### `input` / `output`

```json
{
  "user_prompt": "string",
  "project_overview": "string | null",
  "specs": {"key": "value", ...} | null,
  "design": ["statement", ...] | null
}
```

## Process types

- `new` — no project existed; the prompt introduced one.
- `update` — a project existed; the prompt changed requirements.
- `no_change` — a project existed; the prompt discussed it without changes.
- `no_project` — no project on either side; the prompt is still
  coding-associated (debugging, code questions, snippets, tooling).

## Kept-dataset composition

| type         | count | share |
|--------------|-------|-------|
| `no_project` | 3,722 | 51.3% |
| `new`        | 2,419 | 33.4% |
| `update`     |   570 |  7.9% |
| `no_change`  |   540 |  7.4% |

Across 5,261 unique source conversations.

## Filtering criteria

A prompt was **kept** if it involves real software work in any manner:
building/planning software, project introductions or large-scale updates,
debugging (errors, stack traces, crashes), code review/explanation of concrete
code, dev file operations, shell/git/deployment/database work, or questions
about a specific codebase. Dropped: general chat, essays/poems/stories,
conceptual tech talk with no concrete code or task, homework without code,
acknowledgments. Full ruleset in `filter_report.md`.
