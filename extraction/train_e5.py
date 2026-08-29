"""
E5: BERT-medium + BIO + CRF

Adds a linear-chain CRF layer on top of BERT-medium for sequence-level
transition modeling. Tests whether CRF improves BIO consistency and
span boundary detection.

Architecture:
    INPUT TOKENS → BERT-MEDIUM → emission layer → CRF → VITERBI → BIO

Label set: O=0, B-PROJECT_INFO=1, I-PROJECT_INFO=2 (same as E1)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from src.config import ModelConfig, TrainingConfig
from src.data.alignment import build_bio_labels
from src.models.token_classifier_crf import create_crf_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ExtractionDataset(Dataset):
    """Dataset for BIO token classification."""

    def __init__(self, records, tokenizer, max_length=128):
        self.examples = []
        skipped = 0

        for record in records:
            # Handle both aligned record format and flat format
            if "record" in record:
                prompt = record["record"]["input"].get("user_prompt", "")
                spans = record["spans"]
            else:
                prompt = record.get("prompt", "")
                spans = record.get("spans", [])

            if not prompt or not spans:
                skipped += 1
                continue

            encoded = build_bio_labels(prompt, spans, tokenizer, max_length)

            # Only include examples with at least one positive label
            if any(l != 0 for l in encoded["labels"]):
                self.examples.append({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": encoded["labels"],
                })
            else:
                skipped += 1

        logger.info(
            f"Created dataset with {len(self.examples)} examples "
            f"(skipped {skipped} with no extractive spans)"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


def load_aligned_records(aligned_path, splits_path, split):
    """Load aligned records for a specific split."""
    with open(splits_path) as f:
        splits = json.load(f)

    indices = splits[split]
    records = []
    with open(aligned_path) as f:
        for i, line in enumerate(f):
            if i in indices:
                records.append(json.loads(line))

    return records


def compute_span_level_metrics(predictions, labels, attention_mask):
    """
    Compute span-level precision, recall, F1.

    A span is a contiguous sequence of B I I ... tokens.
    Two spans match if they have the same start and end position.
    """
    tp = fp = fn = 0

    for pred_seq, label_seq, mask_seq in zip(predictions, labels, attention_mask):
        # Get active positions
        active = mask_seq == 1
        pred_active = pred_seq[active]
        label_active = label_seq[active]

        # Extract spans from predictions
        pred_spans = extract_spans(pred_active.tolist())
        label_spans = extract_spans(label_active.tolist())

        # Match spans
        pred_set = set(pred_spans)
        label_set = set(label_spans)

        tp += len(pred_set & label_set)
        fp += len(pred_set - label_set)
        fn += len(label_set - pred_set)

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


def extract_spans(tag_sequence):
    """Extract (start, end) spans from BIO tag sequence."""
    spans = []
    start = None
    for i, tag in enumerate(tag_sequence):
        if tag == 1:  # B
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == 0:  # O
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(tag_sequence)))
    return spans


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Train for one epoch with CRF loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    all_masks = []

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass (CRF computes loss internally)
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs["loss"]

        # Scale loss for gradient accumulation
        loss = loss / 2  # gradient_accumulation_steps = 2
        loss.backward()

        total_loss += loss.item() * 2
        num_batches += 1

        # Gradient accumulation
        if (step + 1) % 2 == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Collect predictions for metrics
        # CRF returns variable-length lists, pad to seq_length
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        padded_preds = torch.zeros(batch_size, seq_length, dtype=torch.long)
        for i, pred in enumerate(outputs["predictions"]):
            pred_len = min(len(pred), seq_length)
            padded_preds[i, :pred_len] = torch.tensor(pred[:pred_len], dtype=torch.long)

        all_preds.append(padded_preds.cpu())
        all_labels.append(labels.cpu())
        all_masks.append(attention_mask.cpu())

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch}: step {step+1}, loss={avg_loss:.4f}")

    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    avg_loss = total_loss / max(num_batches, 1)

    # Token-level F1
    active = all_masks == 1
    pred_flat = all_preds[active].numpy()
    label_flat = all_labels[active].numpy()
    positive_pred = pred_flat >= 1
    positive_label = label_flat >= 1
    tp = int((positive_pred & positive_label).sum())
    fp = int((positive_pred & ~positive_label).sum())
    fn = int((~positive_pred & positive_label).sum())
    token_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

    # Span-level F1
    span_metrics = compute_span_level_metrics(all_preds, all_labels, all_masks)

    return {
        "loss": avg_loss,
        "token_f1": token_f1,
        "span_f1": span_metrics["f1"],
        "span_precision": span_metrics["precision"],
        "span_recall": span_metrics["recall"],
    }


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    all_masks = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs["loss"]

        total_loss += loss.item()
        num_batches += 1

        # Collect predictions
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        padded_preds = torch.zeros(batch_size, seq_length, dtype=torch.long)
        for i, pred in enumerate(outputs["predictions"]):
            pred_len = min(len(pred), seq_length)
            padded_preds[i, :pred_len] = torch.tensor(pred[:pred_len], dtype=torch.long)

        all_preds.append(padded_preds.cpu())
        all_labels.append(labels.cpu())
        all_masks.append(attention_mask.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    avg_loss = total_loss / max(num_batches, 1)

    # Token-level F1
    active = all_masks == 1
    pred_flat = all_preds[active].numpy()
    label_flat = all_labels[active].numpy()
    positive_pred = pred_flat >= 1
    positive_label = label_flat >= 1
    tp = int((positive_pred & positive_label).sum())
    fp = int((positive_pred & ~positive_label).sum())
    fn = int((~positive_pred & positive_label).sum())
    token_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # Span-level F1
    span_metrics = compute_span_level_metrics(all_preds, all_labels, all_masks)

    return {
        "loss": avg_loss,
        "token_f1": token_f1,
        "span_f1": span_metrics["f1"],
        "span_precision": span_metrics["precision"],
        "span_recall": span_metrics["recall"],
        "precision": precision,
        "recall": recall,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/bert_uncased_L-6_H-512_A-8")
    parser.add_argument("--output-dir", default="checkpoints/experiment_e5_crf")
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()

    model_name = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    logger.info("Loading datasets...")
    train_records = load_aligned_records(
        "data/processed/aligned_records.jsonl",
        "data/processed/splits.json",
        "train"
    )
    val_records = load_aligned_records(
        "data/processed/aligned_records.jsonl",
        "data/processed/splits.json",
        "validation"
    )

    train_dataset = ExtractionDataset(train_records, tokenizer, max_length=128)
    val_dataset = ExtractionDataset(val_records, tokenizer, max_length=128)

    # Subsample for CPU feasibility
    if len(train_dataset) > args.max_train:
        indices = np.random.RandomState(42).choice(
            len(train_dataset), args.max_train, replace=False
        )
        train_dataset = Subset(train_dataset, indices)
        logger.info(f"Subsampled train to {args.max_train} examples")

    if len(val_dataset) > args.max_val:
        indices = np.random.RandomState(42).choice(
            len(val_dataset), args.max_val, replace=False
        )
        val_dataset = Subset(val_dataset, indices)
        logger.info(f"Subsampled val to {args.max_val} examples")

    logger.info(f"Train: {len(train_dataset)} examples")
    logger.info(f"Val: {len(val_dataset)} examples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    # Create CRF model
    logger.info("Creating CRF model...")
    model = create_crf_model(
        model_name=model_name,
        num_labels=3,  # BIO
        dropout=0.1,
        freeze_embeddings=True,
    )
    model.to(device)

    # Optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
        eps=1e-6,
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}")

    # Training loop
    logger.info("Starting E5 training (CRF)...")
    start_time = time.time()

    best_val_f1 = 0.0
    best_epoch = 0
    training_history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        logger.info(
            f"Train — loss: {train_metrics['loss']:.4f}, "
            f"token_f1: {train_metrics['token_f1']:.4f}, "
            f"span_f1: {train_metrics['span_f1']:.4f}"
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"Val   — loss: {val_metrics['loss']:.4f}, "
            f"token_f1: {val_metrics['token_f1']:.4f}, "
            f"span_f1: {val_metrics['span_f1']:.4f}, "
            f"P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}"
        )

        # Save best model
        if val_metrics["span_f1"] > best_val_f1:
            best_val_f1 = val_metrics["span_f1"]
            best_epoch = epoch
            (output_dir / "best").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / "best" / "model.pt")
            logger.info(f"Saved best model (span_f1={best_val_f1:.4f})")

        training_history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

    elapsed = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"Best validation span F1: {best_val_f1:.4f} (epoch {best_epoch})")

    # Save training results
    params = model.count_parameters()
    with open(output_dir / "training_results.json", "w") as f:
        json.dump({
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "epochs": args.epochs,
            "elapsed_seconds": elapsed,
            "model_params": params,
            "model_name": model_name,
            "max_seq_length": 128,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "num_labels": 3,
            "label_scheme": "BIO + CRF",
            "config": {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "epochs": args.epochs,
                "seed": 42,
            },
        }, f, indent=2, default=str)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2, default=str)

    return {"best_val_f1": best_val_f1, "best_epoch": best_epoch}


if __name__ == "__main__":
    main()
