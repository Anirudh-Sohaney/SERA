"""
E7 Stage-2 Span Filter Dataset.

Creates training examples for the span filter from:
- Gold spans (positive examples)
- E6-A false-positive candidates (negative examples)

Input format:
    [CLS] prompt tokens [SEP] candidate span tokens [SEP]

Plus candidate position information via token_type_ids:
    0 = prompt region
    1 = candidate region

Labels:
    1 = KEEP (true project-memory span)
    0 = REJECT (false-positive extraction)
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class SpanFilterDataset(Dataset):
    """
    Dataset for training the E7 span filter.

    Each example is a (prompt, candidate, label) triple where:
    - prompt is the full user prompt
    - candidate is a text span with character offsets
    - label is 1 (KEEP) or 0 (REJECT)
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        prompt_max_length: int = 200,
        candidate_max_length: int = 56,
    ):
        """
        Args:
            examples: List of dicts with keys:
                - prompt: str
                - candidate_text: str
                - candidate_start: int
                - candidate_end: int
                - label: int (0 or 1)
                - category: str (for negatives)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum total sequence length
            prompt_max_length: Maximum tokens for prompt portion
            candidate_max_length: Maximum tokens for candidate portion
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_max_length = prompt_max_length
        self.candidate_max_length = candidate_max_length
        self.examples = []

        skipped = 0
        for ex in examples:
            prompt = ex["prompt"]
            candidate_text = ex["candidate_text"]

            if not prompt or not candidate_text:
                skipped += 1
                continue

            # Tokenize as sequence pair: [CLS] prompt [SEP] candidate [SEP]
            encoding = tokenizer(
                prompt,
                candidate_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            self.examples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "token_type_ids": encoding["token_type_ids"].squeeze(0),
                "label": torch.tensor(ex["label"], dtype=torch.float),
                "prompt": prompt,
                "candidate_text": candidate_text,
                "candidate_start": ex.get("candidate_start", -1),
                "candidate_end": ex.get("candidate_end", -1),
                "category": ex.get("category", "unknown"),
            })

        logger.info(
            f"Created SpanFilterDataset with {len(self.examples)} examples "
            f"(skipped {skipped})"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "token_type_ids": example["token_type_ids"],
            "labels": example["label"],
        }

    def get_metadata(self, idx: int) -> Dict:
        """Get non-tensor metadata for a given index."""
        example = self.examples[idx]
        return {
            "prompt": example["prompt"],
            "candidate_text": example["candidate_text"],
            "candidate_start": example["candidate_start"],
            "candidate_end": example["candidate_end"],
            "label": example["label"].item(),
            "category": example["category"],
        }

    def get_label_distribution(self) -> Dict[str, int]:
        """Get the distribution of labels in the dataset."""
        positive = sum(1 for e in self.examples if e["label"].item() == 1.0)
        negative = sum(1 for e in self.examples if e["label"].item() == 0.0)
        return {"positive": positive, "negative": negative}

    def get_category_distribution(self) -> Dict[str, int]:
        """Get the distribution of negative categories."""
        categories = {}
        for e in self.examples:
            cat = e["category"]
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        return categories


def generate_stage2_training_data(
    aligned_records_path: str,
    splits_path: str,
    stage1_model_path: str,
    output_dir: str,
    model_name: str = "google/bert_uncased_L-6_H-512_A-8",
    max_length: int = 128,
    seed: int = 42,
) -> Dict:
    """
    Generate Stage-2 training data by running E6-A inference on training split.

    Pipeline:
        1. Load aligned records and splits
        2. Run E6-A on training prompts to get candidate spans
        3. Compare with gold spans to create positive/negative labels
        4. Balance positive/negative examples
        5. Save to disk

    Returns:
        Dict with statistics
    """
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from ..models.token_classifier import ExtractionClassifier

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load splits
    with open(splits_path) as f:
        splits = json.load(f)

    train_indices = set(splits["train"])
    val_indices = set(splits["validation"])

    # Load aligned records
    train_records = []
    val_records = []
    with open(aligned_records_path) as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            if i in train_indices:
                train_records.append((i, record))
            elif i in val_indices:
                val_records.append((i, record))

    logger.info(f"Loaded {len(train_records)} train, {len(val_records)} val records")

    # Load Stage 1 model
    model = ExtractionClassifier(model_name=model_name)
    model_path = Path(stage1_model_path)
    if (model_path / "model.pt").exists():
        model_file = model_path / "model.pt"
    elif (model_path / "best" / "model.pt").exists():
        model_file = model_path / "best" / "model.pt"
    else:
        raise FileNotFoundError(f"No model.pt found in {stage1_model_path}")
    state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generate Stage-2 examples for training split
    train_examples = _generate_examples(
        train_records, model, tokenizer, max_length, is_train=True
    )

    # Generate Stage-2 examples for validation split
    val_examples = _generate_examples(
        val_records, model, tokenizer, max_length, is_train=False
    )

    # Balance training data
    train_examples = _balance_examples(train_examples, seed=seed)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "stage2_train.jsonl"
    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    val_path = output_path / "stage2_val.jsonl"
    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    # Compute statistics
    stats = {
        "train_total": len(train_examples),
        "train_positive": sum(1 for e in train_examples if e["label"] == 1),
        "train_negative": sum(1 for e in train_examples if e["label"] == 0),
        "val_total": len(val_examples),
        "val_positive": sum(1 for e in val_examples if e["label"] == 1),
        "val_negative": sum(1 for e in val_examples if e["label"] == 0),
        "negative_categories": {},
    }

    for ex in train_examples:
        if ex["label"] == 0:
            cat = ex.get("category", "unknown")
            if cat not in stats["negative_categories"]:
                stats["negative_categories"][cat] = 0
            stats["negative_categories"][cat] += 1

    # Save stats
    stats_path = output_path / "stage2_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Generated Stage-2 training data:")
    logger.info(f"  Train: {stats['train_total']} ({stats['train_positive']} pos, {stats['train_negative']} neg)")
    logger.info(f"  Val: {stats['val_total']} ({stats['val_positive']} pos, {stats['val_negative']} neg)")

    return stats


def _generate_examples(
    records: List[Tuple[int, Dict]],
    model: ExtractionClassifier,
    tokenizer: AutoTokenizer,
    max_length: int,
    is_train: bool,
) -> List[Dict]:
    """
    Generate Stage-2 examples from records using batched inference.

    For each record:
    1. Run Stage 1 to get candidate spans
    2. Compare with gold spans
    3. Exact gold match → POSITIVE
    4. Non-gold prediction → NEGATIVE
    5. Gold span not predicted → POSITIVE (added manually)
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    examples = []

    # Batch inference for efficiency
    batch_size = 64
    all_prompts = []
    all_input_ids = []
    all_attention_masks = []
    all_offset_mappings = []
    all_gold_spans = []
    all_record_indices = []

    for idx, record in records:
        prompt = record.get("record", {}).get("input", {}).get("user_prompt", "")
        gold_spans = record.get("spans", [])

        if not prompt:
            continue

        # Tokenize for Stage 1
        encoding = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
        )

        all_prompts.append(prompt)
        all_input_ids.append(encoding["input_ids"])
        all_attention_masks.append(encoding["attention_mask"])
        all_offset_mappings.append(encoding["offset_mapping"])
        all_gold_spans.append(gold_spans)
        all_record_indices.append(idx)

    logger.info(f"Processing {len(all_prompts)} records in batches...")

    # Process in batches
    for batch_start in range(0, len(all_prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(all_prompts))

        batch_input_ids = torch.tensor(all_input_ids[batch_start:batch_end], dtype=torch.long)
        batch_attention_mask = torch.tensor(all_attention_masks[batch_start:batch_end], dtype=torch.long)

        # Run Stage 1
        with torch.no_grad():
            outputs = model(batch_input_ids, batch_attention_mask)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            confidences = probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)

        # Process each example in the batch
        for i in range(batch_end - batch_start):
            global_idx = batch_start + i
            prompt = all_prompts[global_idx]
            gold_spans = all_gold_spans[global_idx]
            offset_mapping = all_offset_mappings[global_idx]

            # Decode Stage 1 candidates
            candidates = _decode_candidates(
                predictions[i].numpy(),
                confidences[i].numpy(),
                batch_attention_mask[i].numpy(),
                offset_mapping,
                prompt,
            )

            # Create gold span set for exact matching
            gold_set = set()
            for span in gold_spans:
                gold_set.add((span["start"], span["end"]))

            # Create positive examples from gold spans
            gold_matched = set()
            for candidate in candidates:
                c_start = candidate["start"]
                c_end = candidate["end"]

                if (c_start, c_end) in gold_set:
                    # Exact match → POSITIVE
                    examples.append({
                        "prompt": prompt,
                        "candidate_text": candidate["text"],
                        "candidate_start": c_start,
                        "candidate_end": c_end,
                        "label": 1,
                        "category": "gold_match",
                    })
                    gold_matched.add((c_start, c_end))
                else:
                    # Non-gold prediction → NEGATIVE
                    examples.append({
                        "prompt": prompt,
                        "candidate_text": candidate["text"],
                        "candidate_start": c_start,
                        "candidate_end": c_end,
                        "label": 0,
                        "category": _classify_candidate(candidate["text"], prompt),
                    })

            # Add gold spans that were NOT predicted as POSITIVE
            for span in gold_spans:
                s, e = span["start"], span["end"]
                if (s, e) not in gold_matched:
                    text = prompt[s:e]
                    if text.strip():
                        examples.append({
                            "prompt": prompt,
                            "candidate_text": text,
                            "candidate_start": s,
                            "candidate_end": e,
                            "label": 1,
                            "category": "gold_missed",
                        })

        if batch_start % 1000 == 0:
            logger.info(f"  Processed {batch_start}/{len(all_prompts)} records, {len(examples)} examples so far")

    logger.info(f"Total examples generated: {len(examples)}")
    return examples


def _decode_candidates(
    predictions: np.ndarray,
    confidences: np.ndarray,
    attention_mask: np.ndarray,
    offset_mapping: List,
    prompt: str,
    confidence_threshold: float = 0.3,
) -> List[Dict]:
    """Decode BIO predictions to candidate spans."""
    candidates = []
    active_mask = attention_mask == 1
    pred_seq = predictions[active_mask]
    conf_seq = confidences[active_mask]
    offsets = offset_mapping[: len(pred_seq)]

    in_span = False
    span_start = None
    span_conf = []

    for i, (tag, conf, (char_start, char_end)) in enumerate(
        zip(pred_seq, conf_seq, offsets)
    ):
        if tag == 1:  # B-PROJECT_INFO
            if in_span and span_start is not None:
                avg_conf = float(np.mean(span_conf))
                if avg_conf >= confidence_threshold:
                    text = prompt[span_start:char_start]
                    if text.strip():
                        candidates.append({
                            "start": span_start,
                            "end": char_start,
                            "text": text,
                            "confidence": avg_conf,
                        })

            span_start = char_start
            span_conf = [float(conf)]
            in_span = True

        elif tag == 2 and in_span:  # I-PROJECT_INFO
            span_conf.append(float(conf))

        else:  # O
            if in_span and span_start is not None:
                avg_conf = float(np.mean(span_conf))
                if avg_conf >= confidence_threshold:
                    text = prompt[span_start:char_start]
                    if text.strip():
                        candidates.append({
                            "start": span_start,
                            "end": char_start,
                            "text": text,
                            "confidence": avg_conf,
                        })

            in_span = False
            span_start = None
            span_conf = []

    # Close final span
    if in_span and span_start is not None:
        last_offset = offsets[-1] if offsets else (0, 0)
        avg_conf = float(np.mean(span_conf))
        if avg_conf >= confidence_threshold:
            text = prompt[span_start:last_offset[1]]
            if text.strip():
                candidates.append({
                    "start": span_start,
                    "end": last_offset[1],
                    "text": text,
                    "confidence": avg_conf,
                })

    return candidates


def _classify_candidate(text: str, prompt: str) -> str:
    """
    Classify a false-positive candidate into a category.
    Uses the existing E6 false-positive categories.
    """
    text_lower = text.lower().strip()

    # Common words that are not project memory
    common_words = {
        "given", "each", "takes", "all", "only", "two", "no", "first", "any",
        "write", "using", "handle", "find", "check", "new", "original", "empty",
        "input", "output", "number", "numbers", "sum", "length", "difference",
        "maximum", "minimum", "largest", "smallest", "return", "returns",
    }

    # Data types
    data_types = {
        "string", "integer", "int", "float", "bool", "list", "array", "dict",
        "dictionary", "tuple", "set", "vector", "matrix", "json", "boolean",
    }

    # Function patterns
    function_patterns = [
        "function", "method", "class", "def ", "return ", "import ",
        "def ", "print", "lambda",
    ]

    # I/O specs
    io_specs = [
        "list of", "number of", "return an", "ascending order",
        "return a", "return the", "empty list", "new list",
        "input string", "given string", "sorted in",
    ]

    # Code syntax
    code_syntax = ["return", "def", "import", "print", "from"]

    # Check common words
    if text_lower in common_words:
        return "COMMON_WORD"

    # Check data types
    if text_lower in data_types:
        return "DATA_TYPE"

    # Check function patterns
    for pattern in function_patterns:
        if pattern in text_lower:
            return "FUNCTION_SIGNATURE"

    # Check I/O specs
    for spec in io_specs:
        if spec in text_lower:
            return "INPUT_OUTPUT_SPEC"

    # Check code syntax
    if text_lower in code_syntax:
        return "CODE_SYNTAX"

    # Check if it looks like a description fragment
    if any(word in text_lower for word in ["should", "must", "will", "can", "need"]):
        return "DESCRIPTION_FRAGMENT"

    # Check for action verbs
    action_verbs = {"find", "create", "calculate", "test", "check", "sort",
                    "remove", "add", "insert", "update", "delete", "modify"}
    if text_lower in action_verbs:
        return "ACTION_VERB"

    return "OTHER"


def _balance_examples(
    examples: List[Dict],
    seed: int = 42,
    target_ratio: float = 1.0,
) -> List[Dict]:
    """
    Balance positive and negative examples.

    Target: equal number of positive and negative examples.
    If insufficient negatives, use all available.
    """
    random.seed(seed)

    positives = [e for e in examples if e["label"] == 1]
    negatives = [e for e in examples if e["label"] == 0]

    # Determine target count
    target_neg = min(len(negatives), int(len(positives) * target_ratio))

    if target_neg < len(negatives):
        # Sample negatives
        negatives = random.sample(negatives, target_neg)

    balanced = positives + negatives
    random.shuffle(balanced)

    return balanced
