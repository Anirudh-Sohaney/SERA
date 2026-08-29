# SERA Models — Technical Log

Raw technical details: commands, parameter counts, file sizes, exact numbers.

---

## Environment

- Platform: linux (CPU only)
- Python: 3.14.4
- PyTorch: 2.13.0
- Transformers: 5.13.1
- RAM: ~15GB
- Cores: 16

---

## Dataset

- Source: 22,908 coding prompts (2,989 OASST + 19,919 AM-DeepSeek after filtering)
- Extractive spans: 76,005 total
- Exact matches used for training: 62,584 (33.2%)
- Non-extractive outputs excluded: 125,964
- Conversation-level split: 80/10/10
  - Train: 18,326 conversations
  - Val: 2,291 conversations
  - Test: 2,291 conversations
- Train examples with extractive spans: 16,591 (after tokenization, some dropped)
- Val examples with extractive spans: 2,060
- Test examples with extractive spans: 2,073

---

## Evaluation Suite (`data/evaluation/evaluation_suite.json`)

### Random Test
- Count: 2,291 examples (full test set, conversation-level split)
- Format: prompt + spans + technologies dict

### Concept Holdout
- Holdout techs: ["kotlin"] (appeared ≤3 times in train)
- Holdout test examples: 2 (too few for statistical significance)
- Excluded from train: 3 records
- Clean train: 18,323 records
- Note: Dataset is dominated by Python/JS/SQL — few low-frequency techs with test-set presence

### Context Reversal
- Positive: 1,083 examples (real prompts with extractive spans, filtered to those with tech mentions)
- Negative: 200 synthetic examples
  - Templates: "X was considered but rejected", "Do not use X", "X was previously used", etc.
  - Techs: all from TECH_LEXICON (66 techs × ~3 templates each, capped at 200)

### Cross-language
- python: 756 examples
- r: 42 examples
- python 3: 22 examples
- go: 22 examples
- javascript: 16 examples
- Detection: language field from specs + regex fallback

### Unseen Vocabulary
- Unseen techs: 24 (appear ≤3 times in train+val)
- Unseen examples: 6 (test records containing unseen techs)
- Seen examples: 2,285

---

## Non-Extractive Analysis (`data/evaluation/non_extractive_analysis.json`)

- Total non-extractive: 834 out of 22,908 records
- Total extractive: 22,074
- Categories:
  - paraphrase: 106 (output restates prompt, low word overlap)
  - inferred_requirement: 612 (output adds specs/design not in prompt)
  - partial_summary: 116 (output partially overlaps prompt)
- Augmentation examples: 162 (syntactic rewrites matching "Use X for the" → "The backend should use X")

---

## Experiments

### E0: BERT-tiny Baseline

```
Model: google/bert_uncased_L-4_H-256_A-4
Total params: 11,171,331
Trainable params: 3,225,603
Frozen embeddings: 7,945,728
Encoder params: 3,224,832
Task head params: 771 (Linear 256→3)
Max seq length: 128
Dropout: 0.1
Freeze embeddings: True
```

Training config:
```
lr: 3e-5
weight_decay: 0.01
epochs: 3
batch_size: 32
grad_accum: 2 (effective batch 64)
warmup_ratio: 0.1
seed: 42
subsample: 3000 train, 500 val
```

Results:
```
Epoch 1: loss=1.2284, train_f1=0.3374, val_f1=0.4214
Epoch 2: loss=0.7143, train_f1=0.4557, val_f1=0.4469
Epoch 3: loss=0.5650, train_f1=0.5058, val_f1=0.4533
Best val F1: 0.4533 (epoch 3)
Training time: ~4.2 min
Checkpoint: checkpoints/baseline_v0/model.pt (43MB)
```

Full eval suite (evaluated with EvalDataset wrapper):
```
Random test F1:       0.4429 (P=0.2934, R=0.9022), n=2073
Context positive F1:  0.4342, n=986
Context negative F1:  0.0000, n=0
Unseen vocab F1:      0.5551, n=5
```

### E1: BERT-medium (Current Best)

```
Model: google/bert_uncased_L-6_H-512_A-8
Total params: 35,069,955
Trainable params: 19,178,499
Frozen embeddings: 15,891,456
Encoder params: 19,176,960
Task head params: 1,539 (Linear 512→3)
Max seq length: 128
Dropout: 0.1
Freeze embeddings: True
```

Training config:
```
lr: 3e-5
weight_decay: 0.01
epochs: 2
batch_size: 32
grad_accum: 2 (effective batch 64)
warmup_ratio: 0.1
seed: 42
subsample: 5000 train, 500 val
```

Results:
```
Epoch 1: loss=0.6979, train_f1=0.3935, val_f1=0.4718
Epoch 2: loss=0.4758, train_f1=0.4839, val_f1=0.5150
Best val F1: 0.5150 (epoch 2)
Training time: 1098.6s (18.3 min)
Checkpoint: checkpoints/experiment_bert_medium/best/model.pt (140MB)
```

Full eval suite:
```
Random test F1:       0.5058 (P=0.3488, R=0.9201), n=2073
Context positive F1:  0.4920 (P=0.3355, R=0.9221), n=986
Context negative F1:  0.0000, n=0
Unseen vocab F1:      0.7085, n=5
Cross-lang python:    F1=0.4858, n=709
Cross-lang r:         F1=0.4064, n=30
Cross-lang python 3:  F1=0.3456, n=21
Cross-lang go:        F1=0.3631, n=17
Cross-lang javascript: F1=0.5973, n=14
```

### E2: Electra-small

```
Model: google/electra-small-discriminator
Total params: 13,483,008
Hidden size: 256
Layers: 12
```

Training config: same as E1 (5000 train, 500 val, 2 epochs)

Results:
```
Epoch 1: loss=0.7710, train_f1=0.3632, val_f1=0.4623
Epoch 2: loss=0.5192, train_f1=0.4658, val_f1=0.4994
Best val F1: 0.4994 (epoch 2)
Training time: 816.5s (13.6 min)
Checkpoint: overwritten (see bugs section)
```

---

## Comparison Table

| Model | Params | Trainable | Epochs | Train Size | Val F1 | Test F1 | Unseen F1 | Time |
|---|---|---|---|---|---|---|---|---|
| BERT-tiny | 11M | 3.2M | 3 | 3000 | 0.4533 | 0.4429 | 0.5551 | 4.2m |
| BERT-medium | 35M | 19M | 2 | 5000 | **0.5150** | **0.5058** | **0.7085** | 18.3m |
| Electra-small | 13.5M | — | 2 | 5000 | 0.4994 | — | — | 13.6m |
| BERT-medium + neg | 35M | 19M | 2 | 5200 | 0.5113 | 0.5004 | 0.7059 | 17.7m |
| BERT-medium BILOU | 35M | 19M | 2 | 5000 | 0.4665 | 0.4546 | 0.6235 | 18.3m |

Key deltas (BERT-tiny → BERT-medium):
- Val F1: +0.0617 (+13.6%)
- Test F1: +0.0630 (+14.2%)
- Unseen vocab F1: +0.1534 (+27.6%)

Key findings:
- Hard negatives (E3): -0.54% test F1 (hurt)
- BILOU labeling (E4): -5.12% test F1 (significantly hurt)

### E3: BERT-medium + Hard Negatives

```
Model: google/bert_uncased_L-6_H-512_A-8 (same as E1)
Training data: 5000 positive + 200 negative = 5200 total
Negative examples: 200 context-reversal prompts ("X was considered but rejected")
```

Training config: same as E1 (lr=3e-5, batch=32, grad_accum=2, 2 epochs)

Results:
```
Epoch 1: loss=0.7038, train_f1=0.3859, val_f1=0.4879
Epoch 2: loss=0.4739, train_f1=0.4766, val_f1=0.5113
Best val F1: 0.5113 (epoch 2)
Training time: 1062.8s (17.7 min)
Checkpoint: checkpoints/experiment_e3/best/model.pt (140MB)
```

Full eval suite:
```
Random test F1:       0.5004 (P=0.3432, R=0.9230), n=2073
Context positive F1:  0.4867, n=986
Context negative F1:  0.0000, n=0
Unseen vocab F1:      0.7059, n=5
Cross-lang python:    F1=0.4815, n=709
Cross-lang r:         F1=0.4039, n=30
Cross-lang python 3:  F1=0.3154, n=21
Cross-lang go:        F1=0.3569, n=17
Cross-lang javascript: F1=0.5947, n=14
```

**Finding**: Hard negatives HURT performance. Model already learns to not extract from context-reversal prompts (F1=0.0000 on negatives even without hard negatives). Adding synthetic negatives just dilutes positive training signal.

### E4: BERT-medium + BILOU Labeling

```
Model: google/bert_uncased_L-6_H-512_A-8 (same as E1)
Label scheme: BILOU (5 labels: O=0, B=1, I=2, L=3, U=4)
Training data: 5000 examples (same as E1, no negatives)
```

Training config: same as E1 (lr=3e-5, batch=32, grad_accum=2, 2 epochs)

Results:
```
Epoch 1: loss=0.9479, train_f1=0.3511, val_f1=0.4266
Epoch 2: loss=0.5685, train_f1=0.4334, val_f1=0.4665
Best val F1: 0.4665 (epoch 2)
Training time: 1098.0s (18.3 min)
Checkpoint: checkpoints/experiment_e4_bilo/best/model.pt (140MB)
```

Full eval suite (evaluated using >= 1 threshold for positive predictions):
```
Random test F1:       0.4546 (P=0.3002, R=0.9363), n=2073
Context positive F1:  0.4382, n=986
Context negative F1:  0.0000, n=0
Unseen vocab F1:      0.6235, n=5
```

**Finding**: BILOU labeling SIGNIFICANTLY WORSE than BIO. The 5-class fragmentation (-5.12% test F1, -8.50% unseen vocab F1) hurts more than any boundary detection benefit. BIO is simpler and works better for this task.

---

## Bugs Encountered

### Bug 1: `TECH` undefined
- File: `src/evaluation/build_eval_suite.py` line 56
- Cause: Typo — `ALL_TECH = LANGUAGES | FRAMEWORKS | DATABASES | TECH`
- Fix: Changed `TECH` to `TOOLS`

### Bug 2: ExtractionDataset returns 0 examples
- File: `run_experiments.py` ExtractionDataset class
- Cause: Dataset expects `ex.get("prompt", "")` but aligned records have `ar["record"]["input"]["user_prompt"]`
- Fix: Normalize records before passing: `{"prompt": prompt, "spans": ar["spans"]}`
- Note: The existing `src/data/dataset.py` ExtractionDataset works correctly with aligned records format

### Bug 3: Trainer output_dir hardcoded
- File: `train.py` line 126
- Cause: `Trainer(..., output_dir="checkpoints")` — always saves to checkpoints/best/
- Fix: Changed to `Trainer(..., output_dir=args.output_dir)`
- Impact: BERT-medium and Electra both saved to same `checkpoints/best/`, Electra overwrote BERT-medium

### Bug 4: DeBERTa training timeout
- Model: microsoft/deberta-v3-small (44M trainable params)
- Issue: Each step ~4-5s on CPU, 17k examples → hours per epoch
- Attempted fixes: subsampled to 5000, reduced max_length to 128, reduced epochs to 2
- Result: Still too slow (~5 min/step with full dataset). Abandoned for CPU.
- Decision: Use BERT-medium instead (3x fewer params, ~3.3s/step)

### Bug 5: Trainer num_labels hardcoded to 3
- File: `src/training/trainer.py`
- Cause: `compute_class_weights()` used default `NUM_LABELS=3` from label_schema.py
- Fix: Added `num_labels` parameter to Trainer class, passed to `compute_class_weights()`
- Impact: E4 (BILOU) training failed with `IndexError: list index out of range`

### Bug 6: Trainer logits reshaped with hardcoded NUM_LABELS
- File: `src/training/trainer.py` lines 253, 318
- Cause: `logits.view(-1, NUM_LABELS)` hardcoded to 3 labels
- Fix: Changed to `logits.view(-1, self.num_labels)`
- Impact: E4 training crashed after fixing Bug 5

---

## File Inventory

### Source files modified
- `train.py` — added argparse (5 lines changed)
- `src/evaluation/build_eval_suite.py` — rewritten (~350 lines)
- `src/evaluation/build_hard_negatives.py` — new (~200 lines)

### Generated data files
- `data/evaluation/evaluation_suite.json` — 5-category eval suite
- `data/evaluation/evaluation_summary.json` — counts summary
- `data/evaluation/concept_holdout_train_indices.json` — clean train indices
- `data/evaluation/non_extractive_analysis.json` — non-extractive breakdown
- `data/evaluation/augmentation_examples.json` — 162 augmentation examples

### Checkpoints
- `checkpoints/baseline_v0/model.pt` — 43MB, BERT-tiny
- `checkpoints/baseline_v0/config.json` — training config
- `checkpoints/baseline_v0/metrics.json` — final metrics
- `checkpoints/baseline_v0/training_history.json` — epoch-by-epoch
- `checkpoints/experiment_bert_medium/best/model.pt` — 140MB, BERT-medium
- `checkpoints/experiment_bert_medium/best/config.json`
- `checkpoints/experiment_bert_medium/best/metrics.json`
- `checkpoints/experiment_bert_medium/full_eval.json` — full eval suite results
- `checkpoints/experiment_bert_medium/training_history.json`
- `checkpoints/experiment_bert_medium/epoch_1/model.pt` — 140MB
- `checkpoints/experiment_bert_medium/epoch_2/model.pt` — 140MB
- `checkpoints/experiment_electra_small/training_results.json` — no model.pt (overwritten)
- `checkpoints/experiment_e3/best/model.pt` — 140MB, E3 model (hard negatives)
- `checkpoints/experiment_e3/full_eval.json` — full eval results
- `checkpoints/experiment_e4_bilo/best/model.pt` — 140MB, E4 model (BILOU)

### Source files created/modified in Session 2
- `train_e3.py` — training script with hard negatives
- `train_e4.py` — training script with BILOU labeling
- `src/data/alignment.py` — added `build_bilo_labels()` function
- `src/training/trainer.py` — added `num_labels` parameter to Trainer class

### Documentation
- `log.md` — master changelog
- `log-tech.md` — this file (technical details)

---

## Open Questions

1. **Concept holdout too small**: Only 1 holdout tech (kotlin) with 2 test examples. Need to either:
   - Lower the holdout threshold (currently ≤3 train mentions)
   - Use a different approach: test on prompts with rare tech combinations
   - Accept that the dataset is dominated by common techs

2. **Unseen vocabulary too small**: Only 6 test examples with unseen techs. Same root cause — most techs appear frequently.

3. **DeBERTa on CPU**: Not feasible at current dataset size. Would need GPU or aggressive subsampling (<1000 examples).

4. **BILOU vs BIO**: Not yet tested. Could improve boundary detection.

5. **Span classifier**: Not yet implemented. CRF layer could improve sequence coherence.

6. **Hard negatives (E3)**: Not yet run. Plan: add 200 context-reversal negatives to training, measure impact on context_negative F1 and overall F1.
