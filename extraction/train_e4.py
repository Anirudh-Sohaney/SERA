"""
E4: BILOU labeling scheme.

BILOU (Beginning, Inside, Last, Outside, Unit) provides better boundary
detection than BIO by explicitly marking span endpoints.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from src.config import ModelConfig, TrainingConfig
from src.data.alignment import build_bilo_labels
from src.models.token_classifier import create_model
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# BILOU labels: O=0, B=1, I=2, L=3, U=4
NUM_BILOU_LABELS = 5


class BiloDataset(Dataset):
    """Dataset with BILOU labeling."""
    
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
            
            encoded = build_bilo_labels(prompt, spans, tokenizer, max_length)
            
            # Only include examples with at least one non-O label
            if any(l != 0 for l in encoded["labels"]):
                self.examples.append({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": encoded["labels"],
                })
            else:
                skipped += 1
        
        logger.info(f"Created BILOU dataset with {len(self.examples)} examples (skipped {skipped})")
    
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


def bilo_to_bio(labels):
    """Convert BILOU labels to BIO for evaluation comparison."""
    # BILOU: O=0, B=1, I=2, L=3, U=4
    # BIO: O=0, B=1, I=2
    bio = []
    for l in labels:
        if l == 0:
            bio.append(0)  # O
        elif l == 1:
            bio.append(1)  # B
        elif l == 2:
            bio.append(2)  # I
        elif l == 3:
            bio.append(2)  # L -> I (end of span)
        elif l == 4:
            bio.append(1)  # U -> B (single token)
        else:
            bio.append(0)
    return bio


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/bert_uncased_L-6_H-512_A-8")
    parser.add_argument("--output-dir", default="checkpoints/experiment_e4_bilo")
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

    train_dataset = BiloDataset(train_records, tokenizer, max_length=model_config.max_seq_length)
    val_dataset = BiloDataset(val_records, tokenizer, max_length=model_config.max_seq_length)

    # Subsample for CPU feasibility
    max_train = args.max_train
    max_val = args.max_val

    if len(train_dataset) > max_train:
        indices = np.random.RandomState(42).choice(
            len(train_dataset), max_train, replace=False
        )
        train_dataset = Subset(train_dataset, indices)
        logger.info(f"Subsampled train to {max_train} examples")

    if len(val_dataset) > max_val:
        indices = np.random.RandomState(42).choice(
            len(val_dataset), max_val, replace=False
        )
        val_dataset = Subset(val_dataset, indices)
        logger.info(f"Subsampled val to {max_val} examples")

    logger.info(f"Train: {len(train_dataset)} examples")
    logger.info(f"Val: {len(val_dataset)} examples")

    # Create model with 5 BILOU labels
    logger.info("Creating model with BILOU labels...")
    model = create_model(
        model_name=model_name,
        num_labels=NUM_BILOU_LABELS,
        dropout=model_config.dropout,
        freeze_embeddings=model_config.freeze_embeddings,
    )

    # Log parameter counts
    params = model.count_parameters()
    logger.info(f"Model parameters: {json.dumps(params, indent=2)}")

    # Train
    logger.info("Starting E4 training (BILOU)...")
    start_time = time.time()

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        output_dir=args.output_dir,
        num_labels=NUM_BILOU_LABELS,
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
            "num_labels": NUM_BILOU_LABELS,
            "label_scheme": "BILOU",
            "config": train_config.model_dump(),
        }, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
