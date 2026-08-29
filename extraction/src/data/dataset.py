"""
PyTorch Dataset for token classification.

Converts aligned records into model-ready format with:
- Tokenized input_ids
- Attention masks
- BIO label sequences
- Character offset mappings for exact span recovery
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .alignment import build_bio_labels
from .label_schema import NUM_LABELS

logger = logging.getLogger(__name__)


class ExtractionDataset(Dataset):
    """
    Dataset for project-memory span extraction.

    Each example is a user prompt with BIO labels indicating which
    tokens are part of a project-memory span.
    """

    def __init__(
        self,
        aligned_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        label_all_tokens: bool = True,
    ):
        """
        Args:
            aligned_records: List of aligned record dicts from pipeline
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            label_all_tokens: If True, label all sub-tokens of a span;
                             if False, only label the first sub-token
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        self.examples = []

        skipped = 0
        included_negative = 0
        for record in aligned_records:
            prompt = record["record"]["input"].get("user_prompt", "")
            spans = record["spans"]

            if not prompt:
                skipped += 1
                continue

            # Build BIO labels
            encoded = build_bio_labels(prompt, spans, tokenizer, max_length)

            # Include examples with at least one positive label OR
            # include negative examples (all O) if they come from augmentation
            alignment_results = record.get("alignment_results", {})
            if isinstance(alignment_results, dict):
                is_augmented = alignment_results.get("source", "").startswith("e6_")
            elif isinstance(alignment_results, list) and len(alignment_results) > 0:
                is_augmented = False
            else:
                is_augmented = False
            has_positive = any(l != 0 for l in encoded["labels"])
            
            if has_positive or (is_augmented and len(spans) == 0):
                self.examples.append({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": encoded["labels"],
                    "prompt": prompt,
                    "spans": spans,
                    "offset_mapping": encoded["offset_mapping"],
                })
                if not has_positive:
                    included_negative += 1
            else:
                skipped += 1

        logger.info(
            f"Created dataset with {len(self.examples)} examples "
            f"(skipped {skipped}, included {included_negative} negative examples)"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.long),
        }

    def get_metadata(self, idx: int) -> Dict:
        """Get non-tensor metadata for a given index."""
        example = self.examples[idx]
        return {
            "prompt": example["prompt"],
            "spans": example["spans"],
            "offset_mapping": example["offset_mapping"],
        }


def create_datasets(
    aligned_records: List[Dict],
    splits: Dict[str, List[int]],
    model_name: str = "microsoft/deberta-v3-small",
    max_length: int = 512,
) -> Tuple[ExtractionDataset, ExtractionDataset, ExtractionDataset]:
    """
    Create train/val/test datasets from aligned records and split indices.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_records = [aligned_records[i] for i in splits["train"]]
    val_records = [aligned_records[i] for i in splits["validation"]]
    test_records = [aligned_records[i] for i in splits["test"]]

    train_dataset = ExtractionDataset(train_records, tokenizer, max_length)
    val_dataset = ExtractionDataset(val_records, tokenizer, max_length)
    test_dataset = ExtractionDataset(test_records, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset


def load_datasets_from_disk(
    processed_dir: str,
    model_name: str = "microsoft/deberta-v3-small",
    max_length: int = 512,
) -> Tuple[ExtractionDataset, ExtractionDataset, ExtractionDataset]:
    """
    Load preprocessed data from disk and create datasets.
    """
    processed_path = Path(processed_dir)

    # Load aligned records
    aligned_records = []
    with open(processed_path / "aligned_records.jsonl") as f:
        for line in f:
            aligned_records.append(json.loads(line))

    # Load splits
    with open(processed_path / "splits.json") as f:
        splits = json.load(f)

    return create_datasets(aligned_records, splits, model_name, max_length)
