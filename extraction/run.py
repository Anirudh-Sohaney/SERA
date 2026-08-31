"""
Main run script for the extraction SLM prototype.

Orchestrates:
1. Data preprocessing pipeline
2. Lexical baseline evaluation
3. Model training
4. Comprehensive evaluation
5. Inference demo

Usage:
    # Full pipeline
    python run.py

    # Data preprocessing only
    python run.py --stage preprocess

    # Training only
    python run.py --stage train

    # Evaluation only
    python run.py --stage evaluate

    # Inference demo
    python run.py --stage infer
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DataConfig, EvalConfig, ModelConfig, TrainingConfig
from src.data.pipeline import run_pipeline
from src.data.dataset import load_datasets_from_disk
from src.data.splits import extract_programming_languages
from src.models.token_classifier import create_model
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.evaluation.baselines import run_lexical_baseline
from src.inference.extractor import ExtractionSLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_preprocess(data_config: DataConfig, seed: int = 42) -> dict:
    """Run data preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE: DATA PREPROCESSING")
    logger.info("=" * 60)

    stats = run_pipeline(
        oasst_path=data_config.oasst_path,
        am_path=data_config.am_path,
        output_dir=data_config.processed_dir,
        seed=seed,
    )

    return stats


def run_baseline(aligned_records_path: str, splits_path: str) -> dict:
    """Run lexical baseline evaluation."""
    logger.info("=" * 60)
    logger.info("STAGE: LEXICAL BASELINE")
    logger.info("=" * 60)

    import json

    aligned_records = []
    with open(aligned_records_path) as f:
        for line in f:
            aligned_records.append(json.loads(line))

    with open(splits_path) as f:
        splits = json.load(f)

    results = run_lexical_baseline(aligned_records, splits)
    return results


def run_train(
    model_config: ModelConfig,
    train_config: TrainingConfig,
    processed_dir: str,
) -> dict:
    """Train the extraction model."""
    logger.info("=" * 60)
    logger.info("STAGE: MODEL TRAINING")
    logger.info("=" * 60)

    set_seeds(train_config.seed)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_datasets_from_disk(
        processed_dir,
        model_name=model_config.model_name,
        max_length=model_config.max_seq_length,
    )

    logger.info(f"Train: {len(train_dataset)} examples")
    logger.info(f"Val: {len(val_dataset)} examples")
    logger.info(f"Test: {len(test_dataset)} examples")

    # Create model
    logger.info("Creating model...")
    model = create_model(
        model_name=model_config.model_name,
        num_labels=model_config.num_labels,
        dropout=model_config.dropout,
        freeze_embeddings=model_config.freeze_embeddings,
    )

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        output_dir=train_config.output_dir,
    )

    training_results = trainer.train()

    return training_results


def run_eval(
    model_config: ModelConfig,
    eval_config: EvalConfig,
    train_config: TrainingConfig,
    processed_dir: str,
    checkpoint_dir: str = "checkpoints/best",
) -> dict:
    """Run comprehensive evaluation."""
    logger.info("=" * 60)
    logger.info("STAGE: COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)

    set_seeds(train_config.seed)

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets_from_disk(
        processed_dir,
        model_name=model_config.model_name,
        max_length=model_config.max_seq_length,
    )

    # Load model
    logger.info("Loading model from checkpoint...")
    extraction_slm = ExtractionSLM.from_checkpoint(
        checkpoint_dir,
        model_name=model_config.model_name,
        confidence_threshold=eval_config.confidence_threshold,
    )

    evaluator = Evaluator(
        model=extraction_slm.model,
        tokenizer=extraction_slm.tokenizer,
        config=model_config,
        device=extraction_slm.device,
    )

    results = {}

    # Random test evaluation
    if eval_config.random_test:
        logger.info("Evaluating on random test set...")
        results["random_test"] = evaluator.evaluate_dataset(test_dataset, "random_test")

    # Load additional data for generalization tests
    import json

    # Load concept holdout indices
    concept_holdout_path = Path(processed_dir) / "concept_holdout.json"
    if concept_holdout_path.exists() and eval_config.concept_holdout:
        with open(concept_holdout_path) as f:
            concept_data = json.load(f)

        holdout_indices = concept_data.get("holdout_indices", [])
        if holdout_indices:
            # Filter to valid indices
            valid_indices = [i for i in holdout_indices if i < len(test_dataset)]
            if valid_indices:
                subset = torch.utils.data.Subset(test_dataset, valid_indices)
                results["concept_holdout"] = evaluator.evaluate_dataset(
                    subset, "concept_holdout"
                )

    # Cross-language evaluation
    if eval_config.cross_language:
        logger.info("Evaluating cross-language performance...")
        # Load language groups from full dataset
        aligned_records_path = Path(processed_dir) / "aligned_records.jsonl"
        splits_path = Path(processed_dir) / "splits.json"

        if aligned_records_path.exists() and splits_path.exists():
            all_records = []
            with open(aligned_records_path) as f:
                for line in f:
                    all_records.append(json.loads(line))

            with open(splits_path) as f:
                splits = json.load(f)

            # Get test indices from aligned records
            test_indices = splits.get("test", [])
            lang_groups = extract_programming_languages(
                [all_records[i]["record"] for i in test_indices if i < len(all_records)]
            )

            # Evaluate per language
            cross_lang_results = {}
            for lang, indices in lang_groups.items():
                if len(indices) >= 10:
                    valid_test_indices = [
                        i for i in indices if i < len(test_dataset)
                    ]
                    if valid_test_indices:
                        subset = torch.utils.data.Subset(test_dataset, valid_test_indices)
                        cross_lang_results[lang] = evaluator.evaluate_dataset(
                            subset, f"lang_{lang}"
                        )

            results["cross_language"] = cross_lang_results

    # Generate report
    evaluator.generate_report(results, "logs/evaluation")

    return results


def run_inference_demo(checkpoint_dir: str = "checkpoints/best"):
    """Run inference demonstration."""
    logger.info("=" * 60)
    logger.info("STAGE: INFERENCE DEMO")
    logger.info("=" * 60)

    # Load model
    extraction_slm = ExtractionSLM.from_checkpoint(checkpoint_dir)

    # Demo prompts
    demo_prompts = [
        "Use FastAPI for the backend with PostgreSQL as the database",
        "Create a React frontend with TypeScript",
        "Build a CLI tool in Rust that parses TOML config files",
        "I previously considered Django, but use Flask instead",
        "The server should be implemented with Express.js",
        "Use a relational database for the data layer",
        "Write a Python script to process CSV files",
        "Implement the API using GraphQL with Apollo Server",
        "Do not use MongoDB — use PostgreSQL",
        "Build a VS Code extension for syntax highlighting",
    ]

    logger.info("Running inference on demo prompts:")
    for prompt in demo_prompts:
        result = extraction_slm.extract(prompt)
        logger.info(f"\nPrompt: {prompt}")
        if result.has_spans:
            for span in result.spans:
                logger.info(
                    f"  → [{span['start']}:{span['end']}] "
                    f"\"{span['text']}\" "
                    f"(confidence: {span['confidence']:.3f})"
                )
        else:
            logger.info("  → No spans extracted")

    return {
        "num_prompts": len(demo_prompts),
        "prompts_with_spans": sum(
            1 for p in demo_prompts if extraction_slm.extract(p).has_spans
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Extraction SLM Pipeline")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "baseline", "train", "evaluate", "infer", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", default="checkpoints/best")
    args = parser.parse_args()

    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainingConfig(seed=args.seed)
    eval_config = EvalConfig()

    set_seeds(args.seed)

    if args.stage in ("preprocess", "all"):
        stats = run_preprocess(data_config, args.seed)
        logger.info(f"Preprocessing complete: {json.dumps(stats, indent=2)}")

    if args.stage in ("baseline", "all"):
        baseline_results = run_baseline(
            f"{data_config.processed_dir}/aligned_records.jsonl",
            f"{data_config.processed_dir}/splits.json",
        )
        logger.info(f"Baseline results: {json.dumps(baseline_results, indent=2)}")

    if args.stage in ("train", "all"):
        training_results = run_train(model_config, train_config, data_config.processed_dir)
        logger.info(f"Training complete: best F1 = {training_results['best_f1']:.4f}")

    if args.stage in ("evaluate", "all"):
        eval_results = run_eval(
            model_config, eval_config, train_config,
            data_config.processed_dir, args.checkpoint_dir,
        )

    if args.stage in ("infer", "all"):
        demo_results = run_inference_demo(args.checkpoint_dir)


if __name__ == "__main__":
    main()
