"""
E6-A Training: BERT-medium + BIO + Targeted Augmentation

Combines:
- Full original training data (~16,591 extractive examples)
- E6 targeted augmentation (2,289 examples)
- Total: ~18,880 examples

Architecture: same as E1/FULL-DATA (BERT-medium + BIO)
No CRF, no BILOU.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
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


def load_augmented_records(augmented_path):
    """Load augmented records and convert to aligned format."""
    records = []
    with open(augmented_path) as f:
        for line in f:
            aug = json.loads(line)
            # Convert to aligned format
            aligned_record = {
                "record": {
                    "conversation_id": f"augmented_{len(records)}",
                    "turn": 0,
                    "type": "new",
                    "input": {
                        "user_prompt": aug["prompt"],
                        "project_overview": None,
                        "specs": None,
                        "design": None,
                    },
                    "output": {
                        "project_overview": None,
                        "specs": None,
                        "design": None,
                    },
                },
                "spans": aug["spans"],
                "alignment_results": {
                    "source": aug.get("source", "e6_augmentation"),
                    "category": aug.get("category", "UNKNOWN"),
                },
            }
            records.append(aligned_record)
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/bert_uncased_L-6_H-512_A-8")
    parser.add_argument("--output-dir", default="checkpoints/experiment_e6_targeted_fp")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--augmented-data", default="data/processed/e6_targeted_augmented.jsonl")
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

    # Load datasets
    logger.info("Loading datasets...")
    
    # Load original training data
    train_records = load_aligned_records(
        "data/processed/aligned_records.jsonl",
        "data/processed/splits.json",
        "train"
    )
    logger.info(f"Original training records: {len(train_records)}")
    
    # Load augmented data
    augmented_records = load_augmented_records(args.augmented_data)
    logger.info(f"Augmented records: {len(augmented_records)}")
    
    # Combine datasets
    combined_records = train_records + augmented_records
    logger.info(f"Combined training records: {len(combined_records)}")
    
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = ExtractionDataset(combined_records, tokenizer, max_length=model_config.max_seq_length)
    
    # Load validation data
    val_records = load_aligned_records(
        "data/processed/aligned_records.jsonl",
        "data/processed/splits.json",
        "validation"
    )
    val_dataset = ExtractionDataset(val_records, tokenizer, max_length=model_config.max_seq_length)
    
    # Subsample val for speed
    from torch.utils.data import Subset
    if len(val_dataset) > args.max_val:
        indices = np.random.RandomState(42).choice(
            len(val_dataset), args.max_val, replace=False
        )
        val_dataset = Subset(val_dataset, indices)
        logger.info(f"Subsampled val to {args.max_val} examples")
    
    logger.info(f"Train: {len(train_dataset)} examples")
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
    logger.info("Starting E6-A training...")
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
            "original_train_size": len(train_records),
            "augmented_size": len(augmented_records),
            "config": train_config.model_dump(),
        }, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
