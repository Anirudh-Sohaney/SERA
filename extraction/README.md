# English prompts and conversations

The dataset is organized under `data/`.

## Contents

- `data/processed/prompts/wildchat_english_user_messages.jsonl` — individual English user prompts only. The filter uses WildChat's exact `language: "English"` metadata and does not independently detect mixed-language or mislabeled text.
- `data/processed/conversations/oasst2_english_conversations.jsonl` — full English root-to-leaf conversations containing alternating user and assistant messages. Each branch has a unique conversation ID.
- `data/docs/` — schemas, citations, and processing manifest.
- `data/sources/oasst2/README.md` — preserved upstream Hugging Face dataset card.

## Sources

Individual prompts originate from [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), pinned to `7d6490e462285cf85d91eabea0f9a954fbddcd1f`. Full conversations originate from [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2), pinned to `179dd21fc55192153d94adb0e0ce8f69e222bf75`. A Kaggle candidate was reviewed but not downloaded because its description states that its dialogues were programmatically generated; see [`data/docs/sources.md`](data/docs/sources.md).

Human authorship is based on source descriptions and role labels and cannot be independently proven from message text.
