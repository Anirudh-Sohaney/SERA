"""
Configuration for the extraction SLM prototype.

All parameters are documented with rationale. See docs/training.md
for the full parameter justification.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Dataset and preprocessing configuration."""

    # Source data paths
    oasst_path: str = "data/final/dataset_coding.jsonl"
    am_path: str = "data/final/am_dataset_coding.jsonl"

    # Output paths
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    adversarial_dir: str = "data/adversarial"

    # Split ratios
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Alignment
    min_span_length: int = 2  # Minimum characters for a valid span
    max_span_length: int = 200  # Maximum characters for a valid span
    case_sensitive: bool = False  # Case-insensitive matching for alignment

    # Duplicate detection
    near_duplicate_threshold: float = 0.85  # Jaccard similarity threshold

    # Filtering
    include_types: list = Field(default_factory=lambda: ["new", "update"])
    # "no_change" and "no_project" records are excluded from extractive training
    # because they don't introduce new project information


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    # Encoder selection
    # DeBERTa-v3-small: 44M params, 512 context, good for token classification
    # Rationale: Small enough for CPU training, strong contextual representations,
    # pretrained on diverse text including code-adjacent content.
    model_name: str = "microsoft/deberta-v3-small"
    num_labels: int = 3  # O, B-PROJECT_INFO, I-PROJECT_INFO
    max_seq_length: int = 512
    dropout: float = 0.1

    # Freeze early layers to prevent overfitting on small dataset
    freeze_embeddings: bool = True
    freeze_encoder_layers: int = 0  # Number of encoder layers to freeze from bottom


class TrainingConfig(BaseModel):
    """Training hyperparameters — all documented in docs/training.md."""

    # Optimization
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-6
    max_grad_norm: float = 1.0

    # Schedule
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"

    # Batch size — small for CPU training with gradient accumulation
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    # Effective batch size = 8 * 4 = 32

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_metric: str = "eval_f1"

    # Reproducibility
    seed: int = 42

    # Mixed precision — use fp32 on CPU
    fp16: bool = False
    bf16: bool = False

    # Class weighting for imbalanced BIO labels
    # O label is vastly more common; weight B/I labels higher
    class_weights: Optional[list] = None  # Will be computed from data

    # Logging
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    report_to: str = "none"  # Set to "wandb" for experiment tracking

    # Output
    output_dir: str = "checkpoints"
    logging_dir: str = "logs"


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    # Test sets
    random_test: bool = True
    concept_holdout: bool = True
    unseen_vocabulary: bool = True
    adversarial: bool = True
    cross_language: bool = True

    # Error analysis
    error_examples_per_category: int = 10
    manual_review_sample: int = 100

    # Confidence threshold for span extraction
    confidence_threshold: float = 0.5

    # Inference
    overlap_strategy: str = "keep_all"  # keep_all, keep_longest, keep_highest_conf
