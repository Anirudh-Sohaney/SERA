# English full-conversation schema

Each JSONL line in `../processed/conversations/oasst2_english_conversations.jsonl` is one full English root-to-leaf branch from OpenAssistant/oasst2.

Fields:

- `conversation_id`: unique `<source_tree_id>:<branch_index>` identifier; branches from one source tree are distinct records
- `source_dataset`: `OpenAssistant/oasst2`
- `source_revision`: immutable Hugging Face revision
- `language`: literal `en`; every retained node also has source language `en`
- `messages`: ordered array of message objects with `role` (`user` or `assistant`) and non-empty `content`
- `message_count`: number of messages in `messages`

Only alternating paths containing both roles are retained. Any node marked `synthetic: true` is excluded, whether it is a prompter/user or assistant node. Moderation, user IDs, labels, and other unnecessary metadata are omitted. See [`sources.md`](sources.md) for citations.
