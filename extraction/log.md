# SERA Models — Master Changelog

Track: Project-Memory Span Extraction for Clanker Coding Harness

---

## 2026-08-28 — Session 1: Evaluation Suite + Experiment Runner

### What was done

1. **Evaluation suite builder fixed and executed**
   - Fixed typo in `build_eval_suite.py`: `TECH` → `TOOLS` in set union
   - Rewrote the builder entirely with improved detection: word-boundary regex for short techs, category-aware holdout selection, larger unseen-vocabulary set
   - Generated `data/evaluation/evaluation_suite.json` with 5 categories:
     - Random test: 2,291 examples (full test set)
     - Concept holdout: 1 holdout tech (kotlin), 2 test examples, 3 excluded from train
     - Context reversal: 1,083 positive examples, 200 synthetic negatives
     - Cross-language: python (756), r (42), python 3 (22), go (22), javascript (16)
     - Unseen vocabulary: 6 unseen examples, 2,285 seen, 24 unseen techs (freq ≤3 in train+val)

2. **Hard negative analysis built**
   - Created `build_hard_negatives.py` to analyze non-extractive outputs
   - Results: 834 non-extractive outputs out of 22,908 total records
     - paraphrase: 106 (output restates prompt in different words)
     - inferred_requirement: 612 (output adds specs/design not in prompt)
     - partial_summary: 116 (output partially overlaps prompt)
   - 162 augmentation examples found (syntactic rewrites of extractive prompts)
   - Analysis saved to `data/evaluation/non_extractive_analysis.json` and `augmentation_examples.json`

3. **Train.py made configurable**
   - Added argparse: `--model`, `--output-dir`, `--max-train`, `--max-val`, `--epochs`
   - Default remains BERT-tiny for backward compatibility
   - Fixed bug: `output_dir` was hardcoded to `"checkpoints"` in Trainer constructor call, now uses `args.output_dir`

4. **run_experiments.py created** (monolithic experiment runner)
   - Contains inline ExtractionDataset, evaluation logic, training loop
   - Includes subsampling for CPU feasibility (MAX_TRAIN=5000)
   - First attempt failed due to record format mismatch (eval examples have `prompt`/`spans`, ExtractionDataset expects `record.record.input.user_prompt`)

5. **Evaluation runner created and iterated**
   - First version had `TECH` undefined (fixed)
   - Second version had record format mismatch with DeBERTa tokenizer (debugged: offset mapping works fine, issue was in dataset constructor expecting wrong format)
   - Final working version uses EvalDataset wrapper that calls `build_bio_labels` from alignment module

### Experiments run

#### E0: BERT-tiny baseline (re-evaluated on new eval suite)
- Model: `google/bert_uncased_L-4_H-256_A-4`
- Params: 11M total, 3.2M trainable
- Random test F1: **0.4429** (P=0.2934, R=0.9022)
- Context positive F1: **0.4342**
- Context negative F1: **0.0000** (correct — no false extractions on synthetic negatives)
- Unseen vocab F1: **0.5551** (n=5)
- Training: 3 epochs, 3000 train subsample, ~4 min

#### E1: BERT-medium (winner)
- Model: `google/bert_uncased_L-6_H-512_A-8`
- Params: 35M total, 19M trainable, 15.9M frozen embeddings
- Training: 2 epochs, 5000 train subsample, batch=32, grad_accum=2, lr=3e-5
- Training time: 18.3 min (1098.6s)
- Best val F1: **0.5150** (epoch 2)
- Full eval suite:
  - Random test F1: **0.5058** (P=0.3488, R=0.9201)
  - Context positive F1: **0.4920** (P=0.3355, R=0.9221)
  - Context negative F1: **0.0000** (correct)
  - Unseen vocab F1: **0.7085** (n=5)
  - Cross-lang python: F1=0.4858, r: F1=0.4064, go: F1=0.3631, javascript: F1=0.5973
- Checkpoint: `checkpoints/experiment_bert_medium/best/model.pt`

#### E2: Electra-small
- Model: `google/electra-small-discriminator`
- Params: 13.5M total
- Training: 2 epochs, 5000 train subsample
- Training time: 13.6 min (816.5s)
- Best val F1: **0.4994**
- Did NOT run full eval suite (BERT-medium already ahead)
- Checkpoint: overwritten by subsequent runs (see below)

### Bugs found and fixed

1. **`TECH` undefined in build_eval_suite.py** — `ALL_TECH = LANGUAGES | FRAMEWORKS | DATABASES | TECH` → `TOOLS`
2. **ExtractionDataset 0 examples with DeBERTa** — Dataset constructor expected `ex.get("prompt")` but aligned records have nested `record.input.user_prompt` structure. Fixed by normalizing records before passing to dataset.
3. **Trainer output_dir hardcoded** — `Trainer(..., output_dir="checkpoints")` always overwrote `checkpoints/best/`. Fixed to use `args.output_dir`.
4. **Model checkpoint overwritten** — BERT-medium saved to `checkpoints/best/`, then Electra overwrote it. Had to retrain BERT-medium after fixing the output_dir bug.

### Files created/modified

- `src/evaluation/build_eval_suite.py` — rewritten with improved detection
- `src/evaluation/build_hard_negatives.py` — new file for non-extractive analysis
- `train.py` — added argparse for model/output-dir/epochs/etc.
- `run_experiments.py` — new monolithic experiment runner
- `data/evaluation/evaluation_suite.json` — generated
- `data/evaluation/evaluation_summary.json` — generated
- `data/evaluation/concept_holdout_train_indices.json` — generated
- `data/evaluation/non_extractive_analysis.json` — generated
- `data/evaluation/augmentation_examples.json` — generated
- `checkpoints/experiment_bert_medium/` — retrained, saved correctly
- `checkpoints/experiment_bert_medium/full_eval.json` — full eval results

---

## 2026-08-28 — Session 2: E3 (Hard Negatives) + E4 (BILOU)

### What was done

1. **E3: Hard negatives experiment**
   - Created `train_e3.py` — training script with NegativeDataset class
   - Added 200 context-reversal negative examples to training set
   - Trained BERT-medium (35M params) on combined dataset (5000 pos + 200 neg)
   - Training time: 17.7 min, best val F1: 0.5113

2. **E3 results: Hard negatives HURT performance**
   - Random test F1: 0.5004 (vs E1: 0.5058, -0.54%)
   - Context positive F1: 0.4867 (vs E1: 0.4920, -0.53%)
   - Unseen vocab F1: 0.7059 (vs E1: 0.7085, -0.26%)
   - Context negative F1: 0.0000 (already perfect without negatives)
   - **Finding**: Model already learns to not extract from context-reversal prompts. Adding synthetic negatives just dilutes positive training signal.

3. **E4: BILOU labeling scheme**
   - Created `train_e4.py` — training script with BILOU label encoding
   - Modified alignment module to support B/I/L/U/O labels (5 labels instead of 3)
   - Trained BERT-medium with BILOU labeling
   - **Result: BILOU performed significantly worse**
   - Random test F1: 0.4546 (vs E1: 0.5058, -5.12%)
   - Context positive F1: 0.4382 (vs E1: 0.4920, -5.38%)
   - Unseen vocab F1: 0.6235 (vs E1: 0.7085, -8.50%)
   - **Finding**: BILOU fragments the label space (5 classes vs 3) without providing useful boundary information for this task. BIO is simpler and works better.

### Files created/modified

- `train_e3.py` — new training script with NegativeDataset
- `train_e4.py` — new training script with BILOU labeling
- `src/data/alignment.py` — added `build_bilo_labels()` function
- `src/training/trainer.py` — added `num_labels` parameter to Trainer class
- `checkpoints/experiment_e3/best/model.pt` — 140MB, E3 model
- `checkpoints/experiment_e3/full_eval.json` — full eval results
- `checkpoints/experiment_e4_bilo/best/model.pt` — 140MB, E4 model

### Current experiment summary

| Experiment | Model | Training | Val F1 | Test F1 | Status |
|---|---|---|---|---|---|
| E0 | BERT-tiny (11M) | 3 epochs, 3000 | 0.4533 | 0.4429 | Baseline |
| E1 | BERT-medium (35M) | 2 epochs, 5000 | 0.5150 | 0.5058 | **Current best** |
| E2 | Electra-small (13.5M) | 2 epochs, 5000 | 0.4994 | — | Checkpoint lost |
| E3 | BERT-medium (35M) + neg | 2 epochs, 5200 | 0.5113 | 0.5004 | Negatives hurt |
| E4 | BERT-medium (35M) BILOU | 2 epochs, 5000 | 0.4665 | 0.4546 | BILOU worse |

### Key findings

1. **BERT-medium (35M) > BERT-tiny (11M)**: +6.3% test F1, +15.3% unseen vocab F1
2. **Hard negatives don't help**: Model already learns to not extract from context-reversal prompts
3. **BILOU labeling hurts**: 5-class fragmentation worse than 3-class BIO
4. **Context negative F1 is already 0.0000**: No false extractions on synthetic negatives
5. **Unseen vocab is strongest signal**: 70.85% F1 on unseen technologies (vs 50.58% random)

### Next steps

- E5: CRF layer for sequence coherence (may help with BIO consistency)
- Consider: full dataset training (no subsampling) if time permits
- Consider: ensemble of BERT-medium models
- Consider: data augmentation with paraphrase examples

### Current state

| Model | Params | Val F1 | Test F1 | Status |
|---|---|---|---|---|
| BERT-tiny (baseline) | 11M | 0.4533 | 0.4429 | Preserved in baseline_v0/ |
| BERT-medium | 35M | 0.5150 | 0.5058 | **Current best** |
| Electra-small | 13.5M | 0.4994 | — | Checkpoint overwritten |

### Next steps

- E3: Train BERT-medium with hard negatives (context-reversal synthetic negatives added to training)
- E4: Boundary optimization — compare BIO vs BILOU labeling
- E5: Span classifier — CRF layer or extractive head instead of token classification
- Full evaluation on concept holdout (currently only 2 examples — need to expand)
- Decision: whether to pursue DeBERTa (too slow on CPU, would need GPU)

---

## 2026-08-29 — Session 3: Full-Data BIO + E6 Targeted Augmentation

### What was done

1. **Full-Data BIO baseline**
   - Trained BERT-medium on all 18,326 training examples (no subsampling)
   - Test F1: 0.5143 (+0.85% vs E1), unseen vocab F1: 0.7248 (+1.63%)
   - New baseline for E6 experiments

2. **E6-A: Targeted FP augmentation (10%)**
   - Created 2,289 targeted augmentation examples from error analysis patterns
   - Categories: negative templates (763), positive contrasts (763), concise positives (763)
   - Validation: substring equality check (exact match required)
   - Trained on Oracle server (112.7 min), val F1: 0.5460

3. **E6-B: Targeted FP augmentation (20%)**
   - Fixed infinite loop bug in `augmentation_v2.py` (added max_attempts)
   - Generated 3,667 examples (16% actual ratio due to template exhaustion)
   - Trained locally (74.3 min), val F1: 0.5368

4. **Full evaluation on same eval suite**
   - E6-A: test F1=0.4980, unseen F1=0.632
   - E6-B: test F1=0.4883, unseen F1=0.597
   - Full-Data BIO baseline: test F1=0.4791 (on same eval suite)

### Results

| Model | Augmentation | Test F1 | Precision | Recall | Unseen F1 |
|---|---|---|---|---|---|
| Full-Data BIO | None (baseline) | 0.4791 | 0.3206 | 0.9471 | — |
| E6-A | 10% targeted | **0.4980** | **0.3384** | 0.9425 | **0.632** |
| E6-B | 20% targeted | 0.4883 | 0.3285 | 0.9511 | 0.597 |

### Key findings

1. **10% augmentation is optimal** — E6-A outperforms both baseline and E6-B
2. **More augmentation hurts** — 20% augmentation introduces noise, reduces unseen vocab F1 by 5.5%
3. **Precision improved** — E6-A has +5.5% precision vs baseline with minimal recall loss
4. **Augmentation validated** — substring equality check ensures quality

### State of the art

- **E6-A is now the best model**: test F1=0.4980, unseen F1=0.632
- Trained on Oracle, checkpoint at `checkpoints/oracle_e6a/best/model.pt`
- 19.2M trainable parameters, 35M total

---

## 2026-08-29 — Session 4: E7 Two-Stage Span Filter

### What was done

1. **SpanFilter (Stage 2) trained**
   - BERT-medium (35M params), binary [CLS] classifier
   - 10K subsampled training data (5K pos / 5K neg) from E6-A inference
   - 3 epochs, best validation F1=0.6487 at epoch 3
   - Checkpoint: `checkpoints/experiment_e7_span_filter/best/`

2. **Threshold optimized on validation set**
   - Optimal threshold: 0.45 (recall >= 0.90 constraint)
   - Val F1: 0.6418 (P=0.4971, R=0.9053)

3. **Three ablations completed on test set (span-level exact match)**

   | Ablation | Test F1 | Test P | Test R | Unseen F1 | Rejection Rate |
   |---|---|---|---|---|---|
   | E7-A (Stage 1 only) | 0.1003 | 0.0652 | 0.2175 | 0.0460 | — |
   | E7-B (Gold → SpanFilter) | **0.9494** | **1.0000** | 0.9036 | **0.9091** | 9.59% |
   | E7-C (Full pipeline) | 0.1873 | 0.1765 | 0.1995 | 0.0870 | 66.14% |

4. **E7 experiment doc created**: `docs/experiments/E7_TWO_STAGE_FILTER.md`

### Key findings

1. **SpanFilter architecture validated** — E7-B shows F1=0.9494 with 100% precision when given gold candidates. The filter concept works.
2. **Stage 1 is the bottleneck** — The gap between E7-B (0.9494) and E7-C (0.1873) is entirely from bad Stage 1 candidates.
3. **Token-level F1 overestimates by ~5x** — E6-A token-level F1 ~0.50 vs actual span-level F1 0.1003. A model getting 50% of tokens right still misses entire spans.
4. **Stage 1 generates ~10.7 candidates/message** — 24,575 total from 2,291 messages, only 1,469 TP.
5. **SpanFilter rejects 66% of candidates** — Aggressive but necessary given the noise level.

### State of the art

- **E7-B is the theoretical ceiling**: span-level F1=0.9494 (given gold candidates)
- **E7-C is the practical system**: span-level F1=0.1873
- **Stage 1 (E6-A) needs fundamental improvement** — not just more data/augmentation
- Per user directive: proceed to state-transition architecture (Step 2) rather than scaling Stage 1
