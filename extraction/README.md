# English prompts and conversations

Raw source data has been removed after the extraction and filtering pipeline
completed. The final deliverable is [`data/final/dataset_coding.jsonl`](data/final/README.md).

## What remains

- `data/final/` — the coding-project dataset, reports, validation checks
- `data/docs/` — schemas, citations, and processing manifest
- `data/sources/oasst2/README.md` — preserved upstream Hugging Face dataset card
- `data/guide.md` — the project-memory schema used for extraction

## Sources

Individual prompts originated from [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), pinned to `7d6490e462285cf85d91eabea0f9a954fbddcd1f`. Full conversations originated from [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2), pinned to `179dd21fc55192153d94adb0e0ce8f69e222bf75`. A Kaggle candidate was reviewed but not downloaded because its description states that its dialogues were programmatically generated; see [`data/docs/sources.md`](data/docs/sources.md).

Human authorship is based on source descriptions and role labels and cannot be independently proven from message text.

## Pipeline (removed)

The raw data was processed in two LLM stages (both `gpt-5.6-luna` via the
Codex API): (1) per-prompt project-memory state tracking over all 65,389 user
prompts → 65,377 validated records; (2) binary coding-relevance filtering →
7,251 kept records. Procedure, rulesets, and validation evidence:
[`data/final/filter_report.md`](data/final/filter_report.md) and
[`data/final/README.md`](data/final/README.md).
