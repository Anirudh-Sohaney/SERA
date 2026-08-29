"""
FULL-DATA BIO: Train with all eligible extractive training data.

Uses BERT-medium + BIO with the complete training set (~16,591 examples)
instead of subsampled 5,000. Tests whether more data improves performance.

Architecture: same as E1 (BERT-medium + BIO)
No CRF, no BILOU, no synthetic negatives.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from src.config import ModelConfig, TrainingConfig
from src.data.dataset import ExtractionDataset
from src.models.token_classifier import create_model
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/bert_uncased_L-6_H-512_A-8")
    parser.add_argument("--output-dir", default="checkpoints/experiment_full_bio")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-val", type=int, default=500)
    args = parser.parse_args()

    model_name = args.model

    model_config = ModelConfig(
        model_name=model_name,
        max_seq_length=128,
        dropout=0.1,
        freeze_embeddings=True,
    )

    train_config = TrainingConfig(
        learning_rate=3e-5,
        weight_decay=0.01,
        num_epochs=args.epochs,
        warmup_ratio=0.1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        early_stopping_patience=3,
        seed=42,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        output_dir=args.output_dir,
    )

    # Set seeds
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Load datasets — NO subsampling for train
    logger.info("Loading datasets (FULL DATA)...")
    train_dataset, val_dataset, test_dataset = ExtractionDataset(
        load_aligned_records("data/processed/aligned_records.jsonl", "data/processed/splits.json", "train"),
        AutoTokenizer.from_pretrained(model_name),
        max_length=model_config.max_seq_length,
    ), ExtractionDataset(
        load_aligned_records("data/processed/aligned_records.jsonl", "data/processed/splits.json", "validation"),
        AutoTokenizer.from_pretrained(model_name),
        max_length=model_config.max_seq_length,
    ), None

    # Only subsample val for speed
    from torch.utils.data import Subset
    if len(val_dataset) > args.max_val:
        indices = np.random.RandomState(42).choice(
            len(val_dataset), args.max_val, replace=False
        )
        val_dataset = Subset(val_dataset, indices)
        logger.info(f"Subsampled val to {args.max_val} examples")

    logger.info(f"Train: {len(train_dataset)} examples (FULL DATA)")
    logger.info(f"Val: {len(val_dataset)} examples")

    # Create model
    logger.info("Creating model...")
    model = create_model(
        model_name=model_name,
        num_labels=model_config.num_labels,
        dropout=model_config.dropout,
        freeze_embeddings=model_config.freeze_embeddings,
    )

    # Log parameter counts
    params = model.count_parameters()
    logger.info(f"Model parameters: {json.dumps(params, indent=2)}")

    # Train
    logger.info("Starting FULL-DATA BIO training...")
    start_time = time.time()

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        output_dir=args.output_dir,
    )

    results = trainer.train()

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"Best validation F1: {results['best_f1']:.4f}")

    # Save final results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output_dir}/training_results.json", "w") as f:
        json.dump({
            "best_f1": results["best_f1"],
            "epochs": results["epochs"],
            "elapsed_seconds": elapsed,
            "model_params": params,
            "model_name": model_name,
            "max_seq_length": model_config.max_seq_length,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "config": train_config.model_dump(),
        }, f, indent=2, default=str)

    return results


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


if __name__ == "__main__":
    main()
