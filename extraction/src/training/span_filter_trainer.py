"""
E7 Stage-2 Span Filter Trainer.

Training loop for the binary span filter classifier.
Implements:
- Learning rate warmup + linear decay
- Gradient accumulation
- Early stopping on validation F1
- Threshold optimization on validation set
- Comprehensive metric tracking
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

logger = logging.getLogger(__name__)


class SpanFilterTrainer:
    """
    Training loop for the E7 span filter.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        output_dir: str,
        learning_rate: float = 3e-5,
        num_epochs: int = 3,
        per_device_train_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        gradient_accumulation_steps: int = 2,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-6,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 3,
        seed: int = 42,
        logging_steps: int = 25,
        eval_steps: int = 100,
        save_steps: int = 100,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Compute effective batch size
        self.effective_batch_size = (
            per_device_train_batch_size * gradient_accumulation_steps
        )

        # Compute total training steps
        self.train_steps_per_epoch = (
            len(train_dataset) // self.effective_batch_size
        )
        self.total_training_steps = self.train_steps_per_epoch * num_epochs

        # Warmup steps
        self.warmup_steps = int(self.total_training_steps * warmup_ratio)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_epsilon,
        )

        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_training_steps,
        )

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Tracking
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history = []

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Log config
        logger.info(f"SpanFilterTrainer config:")
        logger.info(f"  Effective batch size: {self.effective_batch_size}")
        logger.info(f"  Total training steps: {self.total_training_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")
        logger.info(f"  Train size: {len(train_dataset)}")
        logger.info(f"  Val size: {len(val_dataset)}")

    def train(self) -> Dict:
        """Run the full training loop."""
        logger.info("=" * 60)
        logger.info("STARTING SPAN FILTER TRAINING")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_metrics = self._train_epoch(epoch)

            # Evaluate
            val_metrics = self._evaluate()

            # Log
            logger.info(
                f"Train — loss: {train_metrics['loss']:.4f}, "
                f"f1: {train_metrics['f1']:.4f}, "
                f"p: {train_metrics['precision']:.4f}, "
                f"r: {train_metrics['recall']:.4f}"
            )
            logger.info(
                f"Val   — loss: {val_metrics['loss']:.4f}, "
                f"f1: {val_metrics['f1']:.4f}, "
                f"p: {val_metrics['precision']:.4f}, "
                f"r: {val_metrics['recall']:.4f}"
            )

            # Save history
            self.training_history.append({
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            })

            # Check for improvement
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_epoch = epoch + 1
                self.patience_counter = 0

                # Save best model
                self._save_checkpoint("best", val_metrics)
                logger.info(f"  Saved best model (F1={self.best_val_f1:.4f})")
            else:
                self.patience_counter += 1
                logger.info(
                    f"  No improvement ({self.patience_counter}/{self.early_stopping_patience})"
                )

                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("  Early stopping triggered")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % 1 == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}", val_metrics)

        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info("=" * 60)

        # Save training history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        return {
            "best_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "epochs": epoch + 1,
            "elapsed_seconds": elapsed,
        }

    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            num_workers=0,
        )

        total_loss = 0.0
        all_preds = []
        all_labels = []
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"] / self.gradient_accumulation_steps
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Accumulate gradients
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Collect predictions
            probs = outputs["probs"].detach().cpu()
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics["loss"] = total_loss / max(len(dataloader), 1)

        return metrics

    def _evaluate(self) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs["loss"].item()

                probs = outputs["probs"].cpu()
                preds = (probs >= 0.5).long()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = self._compute_metrics(all_preds, all_labels)
        metrics["loss"] = total_loss / max(len(dataloader), 1)

        return metrics

    def _compute_metrics(
        self, preds: List[int], labels: List[int]
    ) -> Dict[str, float]:
        """Compute precision, recall, F1."""
        preds = np.array(preds)
        labels = np.array(labels)

        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())

        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def _save_checkpoint(self, name: str, metrics: Dict):
        """Save model checkpoint."""
        ckpt_dir = self.output_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")

        with open(ckpt_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save config
        config = {
            "model_name": self.model.model_name,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "effective_batch_size": self.effective_batch_size,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "seed": self.seed,
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
        }

        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)


def optimize_threshold(
    model: nn.Module,
    val_dataset,
    batch_size: int = 64,
    thresholds: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Find optimal threshold on validation set.

    Objective: maximize precision subject to recall >= 0.90

    Returns:
        Dict with threshold results
    """
    if thresholds is None:
        thresholds = [
            0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
            0.80, 0.85, 0.90,
        ]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Collect all probabilities and labels
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Threshold search"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)
            probs = outputs["probs"].cpu()

            all_probs.extend(probs.numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Evaluate each threshold
    results = []
    best_result = None
    best_f1 = 0.0

    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)

        tp = int(((preds == 1) & (all_labels == 1)).sum())
        fp = int(((preds == 1) & (all_labels == 0)).sum())
        fn = int(((preds == 0) & (all_labels == 1)).sum())

        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        result = {
            "threshold": threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "meets_recall_constraint": recall >= 0.90,
        }

        results.append(result)

        # Track best F1 among those meeting recall constraint
        if recall >= 0.90 and f1 > best_f1:
            best_f1 = f1
            best_result = result

    # If no threshold meets recall constraint, find best F1 overall
    if best_result is None:
        for result in results:
            if result["f1"] > best_f1:
                best_f1 = result["f1"]
                best_result = result

    return {
        "thresholds": results,
        "best": best_result,
        "total_examples": len(all_labels),
        "positive_examples": int(all_labels.sum()),
        "negative_examples": int((all_labels == 0).sum()),
    }
