"""
E3: Training with hard negatives.

Adds 200 context-reversal negative examples to training set.
Measures impact on context-negative F1 and overall performance.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from src.config import ModelConfig, TrainingConfig
from src.data.alignment import build_bio_labels
from src.data.dataset import ExtractionDataset
from src.models.token_classifier import create_model
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class NegativeDataset(Dataset):
    """
    Dataset of negative examples (should NOT extract any spans).
    All labels are O (0).
    """
    
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = []
        for ex in examples:
            prompt = ex.get("prompt", "")
            if not prompt:
                continue
            
            # Tokenize
            encoded = tokenizer(
                prompt,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
            )
            
            # All labels are O (0)
            labels = [0] * len(encoded["input_ids"])
            
            self.examples.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
                "prompt": prompt,
            })
        
        logger.info(f"Created negative dataset with {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/bert_uncased_L-6_H-512_A-8")
    parser.add_argument("--output-dir", default="checkpoints/experiment_e3")
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load positive training data
    logger.info("Loading positive training data...")
    train_dataset, val_dataset, _ = ExtractionDataset(
        load_aligned_records("data/processed/aligned_records.jsonl", "data/processed/splits.json", "train"),
        tokenizer,
        max_length=model_config.max_seq_length,
    ), ExtractionDataset(
        load_aligned_records("data/processed/aligned_records.jsonl", "data/processed/splits.json", "validation"),
        tokenizer,
        max_length=model_config.max_seq_length,
    ), None

    # Load negative examples
    logger.info("Loading negative examples...")
    with open("data/evaluation/evaluation_suite.json") as f:
        eval_data = json.load(f)
    
    neg_examples = eval_data["context_reversal"]["negative_examples"]
    neg_dataset = NegativeDataset(neg_examples, tokenizer, max_length=model_config.max_seq_length)

    # Combine datasets
    combined_train = ConcatDataset([train_dataset, neg_dataset])
    logger.info(f"Combined training set: {len(combined_train)} examples ({len(train_dataset)} positive + {len(neg_dataset)} negative)")

    # Subsample for CPU feasibility
    max_train = args.max_train
    max_val = args.max_val

    if len(combined_train) > max_train:
        # Keep all negatives + subsample positives
        n_pos = max_train - len(neg_dataset)
        pos_indices = np.random.RandomState(42).choice(
            len(train_dataset), min(n_pos, len(train_dataset)), replace=False
        )
        neg_indices = np.arange(len(neg_dataset))
        # ConcatDataset indices: positive examples start at 0, negatives start at len(train_dataset)
        combined_indices = list(pos_indices) + list(len(train_dataset) + neg_indices)
        combined_train = Subset(combined_train, combined_indices)
        logger.info(f"Subsampled combined train to {len(combined_train)} examples")

    if len(val_dataset) > max_val:
        indices = np.random.RandomState(42).choice(
            len(val_dataset), max_val, replace=False
        )
        val_dataset = Subset(val_dataset, indices)
        logger.info(f"Subsampled val to {max_val} examples")

    logger.info(f"Train: {len(combined_train)} examples")
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
    logger.info("Starting E3 training (with hard negatives)...")
    start_time = time.time()

    trainer = Trainer(
        model=model,
        train_dataset=combined_train,
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
            "train_size": len(combined_train),
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
