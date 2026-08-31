"""
Run E1/E2 experiments by calling the existing train.py with different models.
"""

import subprocess
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = [
    ("google/bert_uncased_L-6_H-512_A-8", "experiment_bert_L6_H512"),
    ("google/electra-small-discriminator", "experiment_electra_small"),
]

def run_with_model(model_name, output_dir):
    """Run train.py with a specific model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {model_name} -> checkpoints/{output_dir}")
    logger.info(f"{'='*60}")

    # Patch train.py by creating a temporary version
    script = f'''
import sys
sys.path.insert(0, ".")

from pathlib import Path
import json
import logging
import time
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from src.config import ModelConfig, TrainingConfig
from src.data.dataset import load_datasets_from_disk
from src.models.token_classifier import create_model
from src.training.trainer import Trainer

model_name = "{model_name}"
output_dir = "checkpoints/{output_dir}"

model_config = ModelConfig(
    model_name=model_name,
    max_seq_length=128,
    dropout=0.1,
    freeze_embeddings=True,
)

train_config = TrainingConfig(
    learning_rate=3e-5,
    weight_decay=0.01,
    num_epochs=3,
    warmup_ratio=0.1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    early_stopping_patience=3,
    seed=42,
    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    output_dir=output_dir,
)

torch.manual_seed(train_config.seed)
np.random.seed(train_config.seed)

logger.info("Loading datasets...")
train_dataset, val_dataset, test_dataset = load_datasets_from_disk(
    "data/processed",
    model_name=model_name,
    max_length=model_config.max_seq_length,
)

max_train = 5000
max_val = 500
if len(train_dataset) > max_train:
    indices = np.random.RandomState(42).choice(len(train_dataset), max_train, replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, indices)
    logger.info(f"Subsampled train to {{max_train}}")
if len(val_dataset) > max_val:
    indices = np.random.RandomState(42).choice(len(val_dataset), max_val, replace=False)
    val_dataset = torch.utils.data.Subset(val_dataset, indices)
    logger.info(f"Subsampled val to {{max_val}}")

logger.info(f"Train: {{len(train_dataset)}}, Val: {{len(val_dataset)}}, Test: {{len(test_dataset)}}")

model = create_model(
    model_name=model_name,
    num_labels=model_config.num_labels,
    dropout=model_config.dropout,
    freeze_embeddings=model_config.freeze_embeddings,
)

params = model.count_parameters()
logger.info(f"Parameters: {{json.dumps(params, indent=2)}}")

Path(output_dir).mkdir(parents=True, exist_ok=True)

start = time.time()
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=train_config,
    output_dir=output_dir,
)
results = trainer.train()
elapsed = time.time() - start

logger.info(f"Done: {{elapsed:.1f}}s, best_f1={{results['best_f1']:.4f}}")

with open(f"{{output_dir}}/experiment_results.json", "w") as f:
    json.dump({{
        "model_name": model_name,
        "params": params,
        "training_time": elapsed,
        "best_val_f1": results["best_f1"],
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }}, f, indent=2, default=str)
'''
    return script


def main():
    for model_name, output_dir in MODELS:
        script = run_with_model(model_name, output_dir)
        script_path = Path(f"/tmp/run_{output_dir}.py")
        script_path.write_text(script)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            cwd=str(Path(__file__).parent),
        )
        if result.returncode != 0:
            logger.error(f"FAILED: {model_name}")
        else:
            logger.info(f"SUCCESS: {model_name}")


if __name__ == "__main__":
    main()
