"""
Training loop with all required controls.

Implements:
- Learning rate warmup + linear decay
- Gradient accumulation for effective batch sizing
- Early stopping on validation F1
- Class-weighted loss for imbalanced BIO labels
- Gradient clipping
- Comprehensive metric tracking
- Checkpoint saving
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..data.label_schema import NUM_LABELS, LABEL_NAMES

logger = logging.getLogger(__name__)


def compute_class_weights(labels: List[List[int]], num_labels: int = NUM_LABELS) -> torch.Tensor:
    """
    Compute class weights for imbalanced BIO labels.

    The O label typically dominates (90%+ of tokens).
    We weight B and I labels higher to compensate.
    """
    label_counts = [0] * num_labels
    for seq in labels:
        for label in seq:
            if label >= 0:
                label_counts[label] += 1

    total = sum(label_counts)
    weights = []
    for count in label_counts:
        if count > 0:
            weights.append(total / (num_labels * count))
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    attention_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute token-level and span-level metrics.

    Returns:
        Dict with precision, recall, F1, exact_span_accuracy
    """
    # Flatten, ignoring padding
    active_mask = attention_mask.flatten() == 1
    pred_flat = predictions.flatten()[active_mask]
    label_flat = labels.flatten()[active_mask]

    # Token-level metrics (excluding O label)
    # B-PROJECT_INFO = 1, I-PROJECT_INFO = 2
    positive_pred = pred_flat >= 1
    positive_label = label_flat >= 1

    tp = (positive_pred & positive_label).sum()
    fp = (positive_pred & ~positive_label).sum()
    fn = (~positive_pred & positive_label).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Exact match: all tokens in a sequence must match exactly
    # Group by sequence
    batch_size = predictions.shape[0]
    seq_length = predictions.shape[1]
    exact_matches = 0
    total_seqs = 0

    for b in range(batch_size):
        mask = attention_mask[b] == 1
        pred_seq = predictions[b][mask]
        label_seq = labels[b][mask]

        if len(label_seq) > 0 and (label_seq >= 1).any():
            total_seqs += 1
            if np.array_equal(pred_seq, label_seq):
                exact_matches += 1

    exact_match_rate = exact_matches / max(total_seqs, 1)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_match_rate": exact_match_rate,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


class Trainer:
    """
    Training orchestrator with all required controls.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config,
        output_dir: str = "checkpoints",
        num_labels: int = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_labels = num_labels or NUM_LABELS

        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Optimizer — only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon,
        )

        # Compute class weights
        all_labels = []
        # Handle both ExtractionDataset and Subset
        if hasattr(train_dataset, 'examples'):
            examples = train_dataset.examples
        elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'examples'):
            # Subset: access the underlying dataset
            examples = [train_dataset.dataset.examples[i] for i in train_dataset.indices]
        else:
            examples = []
            for i in range(len(train_dataset)):
                item = train_dataset[i]
                all_labels.append(item["labels"].tolist() if hasattr(item["labels"], 'tolist') else item["labels"])

        if not all_labels:
            for ex in examples:
                labels = ex["labels"]
                if hasattr(labels, 'tolist'):
                    all_labels.append(labels.tolist())
                else:
                    all_labels.append(labels)

        class_weights = compute_class_weights(all_labels, num_labels=self.num_labels)
        self.class_weights = class_weights.to(self.device)
        logger.info(f"Class weights: {class_weights.tolist()}")

        # Scheduler
        total_steps = (
            len(self.train_loader)
            * config.num_epochs
            // config.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * config.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Loss function with class weights
        self.loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=-100,
        )

        # Tracking
        self.training_history = []
        self.best_f1 = 0.0
        self.epochs_without_improvement = 0

        # Log total steps
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        all_masks = []

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        self.optimizer.zero_grad()

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Replace -100 with a valid index for loss computation
            labels_for_loss = labels.clone()
            labels_for_loss[labels_for_loss == -100] = 0

            # Forward pass
            outputs = self.model(input_ids, attention_mask, labels=labels_for_loss)
            logits = outputs["logits"]

            # Compute loss manually with class weights
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels_for_loss.view(-1)[active_loss]
            loss = self.loss_fn(active_logits, active_labels)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Collect predictions
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels.cpu().numpy())
            all_masks.append(attention_mask.cpu().numpy())

            progress.set_postfix({"loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}"})

        # Compute epoch metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        metrics = compute_metrics(all_predictions, all_labels, all_masks)
        metrics["loss"] = total_loss / num_batches
        metrics["epoch"] = epoch + 1
        metrics["lr"] = self.scheduler.get_last_lr()[0]

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        all_masks = []

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            labels_for_loss = labels.clone()
            labels_for_loss[labels_for_loss == -100] = 0

            outputs = self.model(input_ids, attention_mask, labels=labels_for_loss)
            logits = outputs["logits"]

            loss = self.loss_fn(
                logits.view(-1, self.num_labels)[attention_mask.view(-1) == 1],
                labels_for_loss.view(-1)[attention_mask.view(-1) == 1],
            )
            total_loss += loss.item()
            num_batches += 1

            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels.cpu().numpy())
            all_masks.append(attention_mask.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        metrics = compute_metrics(all_predictions, all_labels, all_masks)
        metrics["loss"] = total_loss / num_batches

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"epoch_{epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")

        # Save config
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save training config
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=2, default=str)

        if is_best:
            best_dir = self.output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), best_dir / "model.pt")
            with open(best_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            with open(best_dir / "config.json", "w") as f:
                json.dump(self.config.model_dump(), f, indent=2, default=str)
            logger.info(f"Saved best model (F1={metrics['f1']:.4f})")

    def train(self) -> Dict:
        """Run full training loop."""
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(
                f"Train — loss: {train_metrics['loss']:.4f}, "
                f"f1: {train_metrics['f1']:.4f}, "
                f"precision: {train_metrics['precision']:.4f}, "
                f"recall: {train_metrics['recall']:.4f}"
            )

            # Evaluate
            val_metrics = self.evaluate()
            logger.info(
                f"Val   — loss: {val_metrics['loss']:.4f}, "
                f"f1: {val_metrics['f1']:.4f}, "
                f"precision: {val_metrics['precision']:.4f}, "
                f"recall: {val_metrics['recall']:.4f}"
            )

            # Record history
            self.training_history.append({
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            })

            # Check for improvement
            is_best = val_metrics["f1"] > self.best_f1
            if is_best:
                self.best_f1 = val_metrics["f1"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {self.config.early_stopping_patience} epochs)"
                )
                break

        # Final summary
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        logger.info(f"Best validation F1: {self.best_f1:.4f}")
        logger.info(f"Epochs completed: {len(self.training_history)}")

        # Save training history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2, default=str)

        return {
            "best_f1": self.best_f1,
            "epochs": len(self.training_history),
            "training_history": self.training_history,
            "elapsed_seconds": elapsed,
        }
