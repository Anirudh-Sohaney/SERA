# English individual prompts schema

Each JSONL line in `../processed/prompts/wildchat_english_user_messages.jsonl` is one JSON object copied from the previously processed WildChat user-message record. Only records with `language: "English"`, `role: "user"`, and non-empty `text` are retained.

Fields: `record_id`, `source_dataset`, `source_revision`, `source_shard`, `source_row`, `message_index`, `turn_identifier`, `role`, `text`, `language`, `timestamp`, `redacted`, and `toxic`.

Assistant responses and conversation arrays are not present. See [`sources.md`](sources.md) for citations.
