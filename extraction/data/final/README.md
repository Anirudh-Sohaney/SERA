# Project Memory Extraction Dataset

Generated from human-authored conversations (OpenAssistant/oasst2, English
branches) by gpt-5.6-luna (medium reasoning) following `../guide.md`.

## Files

- `dataset.jsonl` — one JSON record per line (the dataset)
- `dataset_stats.json` — generation statistics
- `generation.log` — generation log

## Record format

Each line is a JSON object with exactly these fields:

| field            | type   | description                                              |
|------------------|--------|----------------------------------------------------------|
| `conversation_id`| string | source conversation id (`<tree>:<branch>`)               |
| `turn`           | int    | index of the user prompt within the conversation         |
| `type`           | string | `new`, `update`, or `no_change`                          |
| `input`          | object | `user_prompt` + the project state that existed *before* the prompt |
| `output`         | object | the project state *after* the prompt                     |

### `input`

```json
{
  "user_prompt": "string",
  "project_overview": "string | null",
  "specs": {"key": "value", ...} | null,
  "design": ["statement", ...] | null
}
```

`project_overview`, `specs`, and `design` are `null` when no project existed
yet (the `new` type).

### `output`

```json
{
  "project_overview": "string | null",
  "specs": {"key": "value", ...} | null,
  "design": ["statement", ...] | null
}
```

`null` values appear only for the `no_project` case, which is excluded from
this dataset (see `generate_dataset.py --keep-no-project`).

## Process types

- `new` — no project existed; the prompt introduced one.
- `update` — a project existed; the prompt changed requirements (specs,
  design, or overview added / modified / removed).
- `no_change` — a project existed; the prompt only asked a question or
  discussed the project without changing requirements.

## Generation procedure

1. Iterate conversations; reset project state at the start of each.
2. For each user prompt: send current state + prompt to the model.
3. Record `input` (state before) and `output` (state after).
4. Carry the output forward as the state for the next prompt.
5. Only the first branch (`:0`) of each source tree is used, and only
   conversations whose first user message is software-project related.