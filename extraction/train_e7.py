"""
E7 Training Script — Stage-2 Span Filter.

This script:
1. Generates Stage-2 training data by running E6-A inference on training split
2. Trains the span filter binary classifier
3. Optimizes threshold on validation set
4. Saves checkpoint and results

Usage:
    # Step 1: Generate Stage-2 data
    python train_e7.py --generate-data

    # Step 2: Train span filter
    python train_e7.py --train

    # Step 3: Optimize threshold
    python train_e7.py --optimize-threshold

    # Or run all steps
    python train_e7.py --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data.span_filter_dataset import (
    SpanFilterDataset,
    generate_stage2_training_data,
)
from src.models.span_filter import SpanFilter, create_span_filter
from src.training.span_filter_trainer import SpanFilterTrainer, optimize_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="E7 Stage-2 Span Filter Training")

    # Modes
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate Stage-2 training data")
    parser.add_argument("--train", action="store_true",
                        help="Train span filter model")
    parser.add_argument("--optimize-threshold", action="store_true",
                        help="Optimize threshold on validation set")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps: generate + train + threshold")

    # Paths
    parser.add_argument("--stage1-checkpoint", default="checkpoints/oracle_e6a/best",
                        help="Stage 1 (E6-A) checkpoint directory")
    parser.add_argument("--aligned-data", default="data/processed/aligned_records.jsonl",
                        help="Path to aligned records")
    parser.add_argument("--splits", default="data/processed/splits.json",
                        help="Path to splits file")
    parser.add_argument("--stage2-data-dir", default="data/processed/e7_stage2",
                        help="Directory for Stage-2 training data")
    parser.add_argument("--output-dir", default="checkpoints/experiment_e7_span_filter",
                        help="Output directory for Stage-2 model")

    # Model
    parser.add_argument("--model-name", default="google/bert_uncased_L-6_H-512_A-8",
                        help="Encoder model name")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max sequence length")

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.all:
        args.generate_data = True
        args.train = True
        args.optimize_threshold = True

    if not (args.generate_data or args.train or args.optimize_threshold):
        parser.print_help()
        return

    # Step 1: Generate Stage-2 data
    if args.generate_data:
        logger.info("=" * 60)
        logger.info("STEP 1: Generating Stage-2 training data")
        logger.info("=" * 60)

        stats = generate_stage2_training_data(
            aligned_records_path=args.aligned_data,
            splits_path=args.splits,
            stage1_model_path=args.stage1_checkpoint,
            output_dir=args.stage2_data_dir,
            model_name=args.model_name,
            max_length=128,  # Stage 1 max length
            seed=args.seed,
        )

        logger.info(f"Stage-2 data generated:")
        logger.info(f"  Train: {stats['train_total']} ({stats['train_positive']} pos, {stats['train_negative']} neg)")
        logger.info(f"  Val: {stats['val_total']} ({stats['val_positive']} pos, {stats['val_negative']} neg)")
        logger.info(f"  Negative categories: {stats['negative_categories']}")

    # Step 2: Train span filter
    if args.train:
        logger.info("=" * 60)
        logger.info("STEP 2: Training Stage-2 span filter")
        logger.info("=" * 60)

        # Load Stage-2 data (use subsampled if available)
        train_path = Path(args.stage2_data_dir) / "stage2_train_subsample.jsonl"
        if not train_path.exists():
            train_path = Path(args.stage2_data_dir) / "stage2_train.jsonl"
        val_path = Path(args.stage2_data_dir) / "stage2_val.jsonl"

        if not train_path.exists():
            logger.error(f"Stage-2 training data not found at {train_path}")
            logger.error("Run with --generate-data first")
            return

        # Load examples
        train_examples = []
        with open(train_path) as f:
            for line in f:
                train_examples.append(json.loads(line))

        val_examples = []
        with open(val_path) as f:
            for line in f:
                val_examples.append(json.loads(line))

        logger.info(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples")

        # Create tokenizer and datasets
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        train_dataset = SpanFilterDataset(
            train_examples, tokenizer, max_length=args.max_length
        )
        val_dataset = SpanFilterDataset(
            val_examples, tokenizer, max_length=args.max_length
        )

        # Log distributions
        train_dist = train_dataset.get_label_distribution()
        val_dist = val_dataset.get_label_distribution()
        logger.info(f"Train distribution: {train_dist}")
        logger.info(f"Val distribution: {val_dist}")

        # Create model
        model = create_span_filter(
            model_name=args.model_name,
            dropout=args.dropout,
            freeze_embeddings=True,
        )

        params = model.count_parameters()
        logger.info(f"Model parameters: {json.dumps(params, indent=2)}")

        # Create trainer
        trainer = SpanFilterTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=args.gradient_accumulation,
            warmup_ratio=0.1,
            weight_decay=0.01,
            adam_epsilon=1e-6,
            max_grad_norm=1.0,
            early_stopping_patience=3,
            seed=args.seed,
            logging_steps=25,
            eval_steps=100,
            save_steps=100,
        )

        # Train
        results = trainer.train()

        # Save results
        results_path = Path(args.output_dir) / "training_results.json"
        with open(results_path, "w") as f:
            json.dump({
                **results,
                "model_params": params,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "train_dist": train_dist,
                "val_dist": val_dist,
                "config": {
                    "learning_rate": args.learning_rate,
                    "num_epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "dropout": args.dropout,
                    "max_length": args.max_length,
                    "seed": args.seed,
                },
            }, f, indent=2)

        logger.info(f"Results saved to {results_path}")

    # Step 3: Optimize threshold
    if args.optimize_threshold:
        logger.info("=" * 60)
        logger.info("STEP 3: Optimizing threshold on validation set")
        logger.info("=" * 60)

        # Load model
        best_model_path = Path(args.output_dir) / "best" / "model.pt"
        if not best_model_path.exists():
            logger.error(f"Best model not found at {best_model_path}")
            logger.error("Run with --train first")
            return

        model = create_span_filter(
            model_name=args.model_name,
            dropout=args.dropout,
            freeze_embeddings=True,
        )
        state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        # Load val dataset
        val_path = Path(args.stage2_data_dir) / "stage2_val.jsonl"
        val_examples = []
        with open(val_path) as f:
            for line in f:
                val_examples.append(json.loads(line))

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        val_dataset = SpanFilterDataset(
            val_examples, tokenizer, max_length=args.max_length
        )

        # Optimize threshold
        threshold_results = optimize_threshold(
            model=model,
            val_dataset=val_dataset,
            batch_size=64,
        )

        # Save results
        results_path = Path(args.output_dir) / "threshold_results.json"
        with open(results_path, "w") as f:
            json.dump(threshold_results, f, indent=2)

        logger.info(f"Best threshold: {threshold_results['best']['threshold']}")
        logger.info(f"Best F1: {threshold_results['best']['f1']}")
        logger.info(f"Best Precision: {threshold_results['best']['precision']}")
        logger.info(f"Best Recall: {threshold_results['best']['recall']}")
        logger.info(f"Results saved to {results_path}")

    logger.info("E7 training complete!")


if __name__ == "__main__":
    main()
