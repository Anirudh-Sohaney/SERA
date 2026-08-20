# Data sources and citations

## Hugging Face: WildChat-1M (individual prompts)

- Dataset: <https://huggingface.co/datasets/allenai/WildChat-1M>
- Revision: `7d6490e462285cf85d91eabea0f9a954fbddcd1f`
- Paper: [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470)
- License: [ODC-BY](https://opendatacommons.org/licenses/by/1-0/)

The prompt output contains individual source-labeled user messages filtered to the source language metadata value exactly `English`. This does not detect mixed-language text or correct a mislabeled source record.

## Hugging Face: OpenAssistant/oasst2 (full conversations)

- Dataset: <https://huggingface.co/datasets/OpenAssistant/oasst2>
- Pinned revision: `179dd21fc55192153d94adb0e0ce8f69e222bf75`
- Upstream card: [`../sources/oasst2/README.md`](../sources/oasst2/README.md)
- License: Apache-2.0, per upstream dataset metadata

The conversation output is built from the upstream `ready.trees` file by flattening valid English root-to-leaf branches. `prompter` is represented as `user`; `assistant` remains `assistant`. Every node must have language `en`, roles must alternate, both roles must be present, and any node marked `synthetic: true` causes the branch to be excluded. Each branch gets a unique tree-and-branch identifier.

## Kaggle review (not downloaded)

- Reviewed candidate: <https://www.kaggle.com/datasets/abhayayare/multi-turn-chatbot-conversation-dataset>
- Status: reviewed but not downloaded
- Decision: excluded because its description attributes dialogues to programmatic generation from randomized queries, intent-based responses, and generated conversation trees. It does not meet the real-user-oriented collection requirement.

## Authorship and privacy caveat

Dataset role labels and collection descriptions are not proof that every prompt was independently written by a human. User text may contain pasted or automated material. Outputs omit unnecessary source metadata, but message text can still contain personal or third-party information.
