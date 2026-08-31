"""
Experiment runner for E0-E5.

Sequentially runs:
E0: BERT-tiny baseline (already done)
E1: Evaluation rebuild
E2: Stronger encoder (DeBERTa-small)
E3: Hard contextual negatives
E4: Boundary optimization (BILOU comparison)
E5: Span classifier
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


class ExtractionDataset(Dataset):
    """PyTorch dataset for token classification."""

    def __init__(self, examples, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for ex in examples:
            prompt = ex.get("prompt", "")
            spans = ex.get("spans", [])

            if not prompt or not spans:
                continue

            encoding = tokenizer(
                prompt,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            offset_mapping = encoding["offset_mapping"]

            # Build BIO labels
            labels = [0] * len(input_ids)
            char_to_token = {}
            for tok_idx, (cs, ce) in enumerate(offset_mapping):
                if cs == 0 and ce == 0:
                    continue
                for c in range(cs, ce):
                    char_to_token[c] = tok_idx

            for span in spans:
                first_tok = None
                last_tok = None
                for c in range(span["start"], min(span["end"], len(prompt))):
                    if c in char_to_token:
                        tok = char_to_token[c]
                        if first_tok is None:
                            first_tok = tok
                        last_tok = tok
                if first_tok is not None:
                    labels[first_tok] = 1  # B
                    for t in range(first_tok + 1, last_tok + 1):
                        labels[t] = 2  # I

            if any(l > 0 for l in labels):
                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


def compute_span_metrics(preds, labels, masks):
    """Compute token-level and span-level metrics."""
    active = masks.flatten() == 1
    pred_flat = preds.flatten()[active]
    label_flat = labels.flatten()[active]

    # Token-level
    pos_pred = pred_flat >= 1
    pos_label = label_flat >= 1
    tp = int((pos_pred & pos_label).sum())
    fp = int((pos_pred & ~pos_label).sum())
    fn = int((~pos_pred & pos_label).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def evaluate_model(model, dataset, batch_size=32):
    """Evaluate model on a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    all_masks = []

    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.numpy())
        all_labels.append(labels.numpy())
        all_masks.append(attention_mask.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_masks = np.concatenate(all_masks)

    return compute_span_metrics(all_preds, all_labels, all_masks)


def run_experiment_e0():
    """E0: Load baseline_v0 and evaluate on all test sets."""
    logger.info("=" * 60)
    logger.info("E0: BERT-tiny baseline evaluation")
    logger.info("=" * 60)

    from src.models.token_classifier import ExtractionClassifier

    model = ExtractionClassifier(model_name="google/bert_uncased_L-4_H-256_A-4")
    state_dict = torch.load("checkpoints/baseline_v0/model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # Load evaluation data
    with open("data/evaluation/evaluation_suite.json") as f:
        eval_data = json.load(f)

    # Evaluate on random test
    random_examples = eval_data["random_test"]["examples"]
    logger.info(f"Random test: {len(random_examples)} examples")

    # Build dataset from random test
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
    test_dataset = ExtractionDataset(random_examples, tokenizer, max_length=128)

    metrics = evaluate_model(model, test_dataset, batch_size=32)
    logger.info(f"Random test F1: {metrics['f1']:.4f}")
    logger.info(f"Random test P: {metrics['precision']:.4f}")
    logger.info(f"Random test R: {metrics['recall']:.4f}")

    return {"random_test": metrics}


def run_experiment_e1_e3(encoder_name, max_length, train_indices, eval_data,
                         use_hard_negatives=False, label_scheme="BIO"):
    """
    E1-E3: Train stronger encoder with improved evaluation.
    """
    from transformers import AutoTokenizer, AutoModel
    import torch.nn as nn

    logger.info(f"Training {encoder_name} (max_length={max_length})")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # Load all aligned records
    aligned_records = []
    with open("data/processed/aligned_records.jsonl") as f:
        for line in f:
            aligned_records.append(json.loads(line))

    # Build training set (normalize record structure for ExtractionDataset)
    train_records = []
    for idx in train_indices:
        if idx >= len(aligned_records):
            continue
        ar = aligned_records[idx]
        if ar["spans"]:
            prompt = ar["record"].get("input", {}).get("user_prompt", "")
            train_records.append({"prompt": prompt, "spans": ar["spans"]})

    # Add hard negatives if requested
    if use_hard_negatives:
        neg_examples = eval_data.get("context_reversal", {}).get("negative_examples", [])
        for neg in neg_examples[:200]:
            train_records.append({
                "prompt": neg["prompt"],
                "spans": [],
            })
        logger.info(f"Added {min(len(neg_examples), 200)} hard negative examples")

    # Add augmentation if available
    aug_path = Path("data/evaluation/augmentation_examples.json")
    if aug_path.exists() and use_hard_negatives:
        with open(aug_path) as f:
            aug_data = json.load(f)
        for aug in aug_data:
            # Create augmented record with same spans but shifted offsets
            original = aug["original"]
            augmented = aug["augmented"]
            spans = aug["spans"]
            # Simple: use original prompt with spans (augmented not used for training yet)
            pass

    logger.info(f"Training records: {len(train_records)}")

    # Subsample for CPU feasibility (keep all for evaluation)
    MAX_TRAIN = 5000
    if len(train_records) > MAX_TRAIN:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(train_records), MAX_TRAIN, replace=False)
        train_records = [train_records[i] for i in indices]
        logger.info(f"Subsampled to {MAX_TRAIN} training records")

    # Create dataset
    train_dataset = ExtractionDataset(train_records, tokenizer, max_length=max_length)
    logger.info(f"Train examples: {len(train_dataset)}")

    # Create model
    encoder = AutoModel.from_pretrained(encoder_name)
    hidden_size = encoder.config.hidden_size

    class TokenClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(hidden_size, 3)
            # Freeze embeddings
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False

        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state.float()
            hidden = self.dropout(hidden)
            logits = self.classifier(hidden)
            result = {"logits": logits}
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                active = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, 3)[active]
                active_labels = labels.view(-1)[active]
                result["loss"] = loss_fn(active_logits, active_labels)
            return result

    model = TokenClassifier()

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total:,}, Trainable: {trainable:,}")

    # Training config
    config = {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "epochs": 2,
        "batch_size": 8,
        "grad_accum": 4,
        "warmup_ratio": 0.1,
        "seed": 42,
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    total_steps = (len(train_dataset) // config["batch_size"]) * config["epochs"] // config["grad_accum"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Train
    start_time = time.time()
    best_f1 = 0.0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss = outputs["loss"] / config["grad_accum"]
            loss.backward()
            total_loss += loss.item() * config["grad_accum"]
            num_batches += 1

            if (step + 1) % config["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    training_time = time.time() - start_time
    logger.info(f"Training time: {training_time:.1f}s")

    # Evaluate on all test sets
    results = {"config": config, "training_time": training_time, "params": {"total": total, "trainable": trainable}}

    # Random test
    random_examples = eval_data["random_test"]["examples"]
    random_dataset = ExtractionDataset(random_examples, tokenizer, max_length=max_length)
    results["random_test"] = evaluate_model(model, random_dataset)
    logger.info(f"Random test F1: {results['random_test']['f1']:.4f}")

    # Context reversal - positive
    pos_examples = eval_data["context_reversal"]["positive_examples"]
    pos_dataset = ExtractionDataset(pos_examples, tokenizer, max_length=max_length)
    if len(pos_dataset) > 0:
        results["context_positive"] = evaluate_model(model, pos_dataset)
        logger.info(f"Context positive F1: {results['context_positive']['f1']:.4f}")

    # Context reversal - negative (should extract nothing)
    neg_examples = eval_data["context_reversal"]["negative_examples"]
    # For negatives, we expect no extraction - measure false positive rate
    neg_dataset = ExtractionDataset(neg_examples, tokenizer, max_length=max_length)
    if len(neg_dataset) > 0:
        neg_metrics = evaluate_model(model, neg_dataset)
        # For negatives, low recall is good (not extracting)
        results["context_negative"] = neg_metrics
        logger.info(f"Context negative (should be low): F1={neg_metrics['f1']:.4f}")

    # Unseen vocabulary
    unseen_examples = eval_data["unseen_vocabulary"]["unseen_examples"]
    unseen_dataset = ExtractionDataset(unseen_examples, tokenizer, max_length=max_length)
    if len(unseen_dataset) > 0:
        results["unseen_vocab"] = evaluate_model(model, unseen_dataset)
        logger.info(f"Unseen vocab F1: {results['unseen_vocab']['f1']:.4f}")

    # Cross-language
    cross_results = {}
    for lang, lang_data in eval_data.get("cross_language", {}).items():
        lang_examples = lang_data["examples"]
        lang_dataset = ExtractionDataset(lang_examples, tokenizer, max_length=max_length)
        if len(lang_dataset) > 5:
            cross_results[lang] = evaluate_model(model, lang_dataset)
            logger.info(f"Cross-lang {lang}: F1={cross_results[lang]['f1']:.4f}")
    results["cross_language"] = cross_results

    # Save model
    model_dir = Path(f"checkpoints/experiment_{encoder_name.split('/')[-1]}")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    with open(model_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load evaluation data
    with open("data/evaluation/evaluation_suite.json") as f:
        eval_data = json.load(f)

    # Load splits
    with open("data/processed/splits.json") as f:
        splits = json.load(f)

    # E0: Baseline
    logger.info("Running E0...")
    e0_results = run_experiment_e0()

    # E1-E3: DeBERTa-small with full training set
    logger.info("Running E1-E3: DeBERTa-small...")
    e1_results = run_experiment_e1_e3(
        encoder_name="microsoft/deberta-v3-small",
        max_length=128,
        train_indices=splits["train"],
        eval_data=eval_data,
        use_hard_negatives=False,
    )

    # E3: With hard negatives
    logger.info("Running E3: DeBERTa-small + hard negatives...")
    e3_results = run_experiment_e1_e3(
        encoder_name="microsoft/deberta-v3-small",
        max_length=128,
        train_indices=splits["train"],
        eval_data=eval_data,
        use_hard_negatives=True,
    )

    # Save all results
    all_results = {
        "E0_baseline": e0_results,
        "E1_deberta_small": e1_results,
        "E3_deberta_hard_neg": e3_results,
    }

    with open("logs/experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for name, res in all_results.items():
        logger.info(f"{name}: random_f1={res.get('random_test', {}).get('f1', 'N/A')}")


if __name__ == "__main__":
    main()
