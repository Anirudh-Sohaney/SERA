"""
E7 Ablation Evaluation Script.

Runs three ablations on the test set + unseen-vocab split:
- E7-A: Stage 1 (E6-A) only — token-level BIO evaluation
- E7-B: Gold candidates → SpanFilter — theoretical filtering upper bound
- E7-C: Full pipeline — E6-A candidates → SpanFilter → evaluate end-to-end

Reports: precision, recall, F1, unseen-vocab F1, candidate count/message,
filter rejection rate, inference latency, RAM, model parameters.

Usage:
    python3 eval_e7.py [--ablation A|B|C|all]
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from src.data.alignment import build_bio_labels
from src.data.span_filter_dataset import SpanFilterDataset
from src.models.span_filter import SpanFilter, create_span_filter
from src.models.token_classifier import ExtractionClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
SPLIT_PATH = BASE / "data/processed/splits.json"
ALIGNED_PATH = BASE / "data/processed/aligned_records.jsonl"
UNSEEN_PATH = BASE / "data/processed/unseen_vocabulary.json"
STAGE1_CKPT = BASE / "checkpoints/oracle_e6a/best"
STAGE2_CKPT = BASE / "checkpoints/experiment_e7_span_filter/best"
STAGE1_MODEL_NAME = "google/bert_uncased_L-6_H-512_A-8"
STAGE2_MODEL_NAME = "google/bert_uncased_L-6_H-512_A-8"
THRESHOLD_PATH = BASE / "checkpoints/experiment_e7_span_filter/threshold_results.json"

MAX_LENGTH = 128
BATCH_SIZE = 64


def get_memory_mb():
    """Get current RSS memory usage in MB via /proc."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * 4 / 1024  # 4 KB pages → MB
    except Exception:
        return 0.0


def load_aligned_records():
    """Load all aligned records."""
    records = []
    with open(ALIGNED_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_test_examples(all_records):
    """Load test split from aligned records, returning [{prompt, spans}]."""
    with open(SPLIT_PATH) as f:
        splits = json.load(f)
    test_indices = splits["test"]

    examples = []
    for idx in test_indices:
        rec = all_records[idx]
        inp = rec["record"]["input"]
        prompt = inp.get("user_prompt", "") or ""
        spans = rec.get("spans", [])
        examples.append({"prompt": prompt, "spans": spans})
    logger.info(f"Loaded {len(examples)} test examples")
    return examples


def load_unseen_examples(all_records):
    """Load unseen-vocabulary split (prompts containing never-seen tech terms)."""
    with open(UNSEEN_PATH) as f:
        unseen = json.load(f)
    indices = unseen["unseen_indices"]

    examples = []
    for idx in indices:
        rec = all_records[idx]
        inp = rec["record"]["input"]
        prompt = inp.get("user_prompt", "") or ""
        spans = rec.get("spans", [])
        examples.append({"prompt": prompt, "spans": spans})
    logger.info(f"Loaded {len(examples)} unseen-vocab examples")
    return examples


# ── BIO Decoding ────────────────────────────────────────────────────────
def decode_bio(predictions, confidences, attention_mask, offset_mapping, prompt):
    """Decode BIO tag sequence to character-offset spans."""
    candidates = []
    active_mask = attention_mask == 1
    pred_seq = predictions[active_mask]
    conf_seq = confidences[active_mask]
    offsets = offset_mapping[: len(pred_seq)]

    in_span = False
    span_start = None
    span_conf = []

    for tag, conf, (cs, ce) in zip(pred_seq, conf_seq, offsets):
        if tag == 1:  # B
            if in_span and span_start is not None:
                avg_conf = float(np.mean(span_conf))
                text = prompt[span_start:cs]
                if text.strip():
                    candidates.append({"start": span_start, "end": cs,
                                       "text": text, "confidence": avg_conf})
            span_start = cs
            span_conf = [float(conf)]
            in_span = True
        elif tag == 2 and in_span:  # I
            span_conf.append(float(conf))
        else:  # O
            if in_span and span_start is not None:
                avg_conf = float(np.mean(span_conf))
                text = prompt[span_start:cs]
                if text.strip():
                    candidates.append({"start": span_start, "end": cs,
                                       "text": text, "confidence": avg_conf})
            in_span = False
            span_start = None
            span_conf = []

    if in_span and span_start is not None:
        last_ce = offsets[-1][1] if offsets else 0
        avg_conf = float(np.mean(span_conf))
        text = prompt[span_start:last_ce]
        if text.strip():
            candidates.append({"start": span_start, "end": last_ce,
                               "text": text, "confidence": avg_conf})
    return candidates


# ── E7-A: Stage 1 Only ────────────────────────────────────────────────
def run_e7a(examples, label=""):
    """Evaluate Stage 1 (E6-A) token-level BIO."""
    logger.info(f"E7-A: Stage 1 Only — {label} ({len(examples)} examples)")

    model = ExtractionClassifier(model_name=STAGE1_MODEL_NAME)
    state = torch.load(STAGE1_CKPT / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(STAGE1_MODEL_NAME)
    model.eval()

    mem_before = get_memory_mb()
    t0 = time.time()
    tp = fp = fn = 0
    latencies = []

    for ex in examples:
        prompt = ex["prompt"]
        gold_spans = ex["spans"]
        if not prompt:
            continue

        t_batch = time.time()
        enc = tokenizer(prompt, max_length=MAX_LENGTH, truncation=True,
                        padding="max_length", return_offsets_mapping=True)
        input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long)

        with torch.no_grad():
            out = model(input_ids, attention_mask)
            preds = torch.argmax(out["logits"], dim=-1)
            probs = torch.softmax(out["logits"], dim=-1)
            confs = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)

        latencies.append(time.time() - t_batch)

        pred_spans = decode_bio(preds[0].numpy(), confs[0].numpy(),
                                attention_mask[0].numpy(), enc["offset_mapping"], prompt)
        pred_set = set((s["start"], s["end"]) for s in pred_spans)
        gold_set = set((s["start"], s["end"]) for s in gold_spans)

        for pred in pred_set:
            if pred in gold_set:
                tp += 1
            else:
                fp += 1
        for gold in gold_set:
            if gold not in pred_set:
                fn += 1

    total_time = time.time() - t0
    mem_after = get_memory_mb()
    del model, tokenizer
    gc.collect()

    p = tp / max(tp + fp, 1e-8)
    r = tp / max(tp + fn, 1e-8)
    f1 = 2 * p * r / max(p + r, 1e-8)

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "avg_latency_ms": round(np.mean(latencies) * 1000, 2) if latencies else 0,
        "total_time_s": round(total_time, 2),
        "memory_mb": round(mem_after - mem_before, 1),
        "n": len(examples),
    }


# ── E7-B: Gold Candidates → SpanFilter ─────────────────────────────────
def run_e7b(examples, threshold, label=""):
    """Evaluate SpanFilter with gold candidate spans (theoretical upper bound)."""
    logger.info(f"E7-B: Gold → SpanFilter — {label} ({len(examples)} examples)")

    model = create_span_filter(model_name=STAGE2_MODEL_NAME, dropout=0.1, freeze_embeddings=True)
    state = torch.load(STAGE2_CKPT / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(STAGE2_MODEL_NAME)
    model.eval()

    mem_before = get_memory_mb()
    t0 = time.time()
    tp = fp = fn = 0
    total_candidates = 0
    total_accepted = 0
    latencies = []

    for ex in examples:
        prompt = ex["prompt"]
        gold_spans = ex["spans"]
        if not prompt:
            continue

        # Gold candidates
        gold_candidates = []
        for s in gold_spans:
            start, end = s["start"], s["end"]
            text = prompt[start:end]
            if text.strip():
                gold_candidates.append({"start": start, "end": end, "text": text})
        total_candidates += len(gold_candidates)

        # Classify each gold candidate
        accepted = []
        for cand in gold_candidates:
            t_batch = time.time()
            enc = tokenizer(prompt, cand["text"], max_length=256, truncation=True,
                            padding="max_length", return_tensors="pt")
            with torch.no_grad():
                out = model(enc["input_ids"], enc["attention_mask"])
                prob = out["probs"].item()
            latencies.append(time.time() - t_batch)
            if prob >= threshold:
                accepted.append(cand)
        total_accepted += len(accepted)

        # Gold = all gold candidates are "relevant"; rejected gold = FN; accepted = TP
        accepted_set = set((s["start"], s["end"]) for s in accepted)
        gold_set = set((s["start"], s["end"]) for s in gold_candidates)

        # All gold candidates ARE the ground truth; accepted = TP
        for a in accepted_set:
            if a in gold_set:
                tp += 1
            else:
                fp += 1
        for g in gold_set:
            if g not in accepted_set:
                fn += 1

    total_time = time.time() - t0
    mem_after = get_memory_mb()
    del model, tokenizer
    gc.collect()

    p = tp / max(tp + fp, 1e-8)
    r = tp / max(tp + fn, 1e-8)
    f1 = 2 * p * r / max(p + r, 1e-8)

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "total_candidates": total_candidates,
        "total_accepted": total_accepted,
        "rejection_rate": round(1 - (total_accepted / max(total_candidates, 1)), 4),
        "avg_latency_ms": round(np.mean(latencies) * 1000, 2) if latencies else 0,
        "total_time_s": round(total_time, 2),
        "memory_mb": round(mem_after - mem_before, 1),
        "n": len(examples),
        "threshold": threshold,
    }


# ── E7-C: Full Pipeline ────────────────────────────────────────────────
def run_e7c(examples, threshold, label=""):
    """Evaluate full pipeline: E6-A → SpanFilter."""
    logger.info(f"E7-C: Full Pipeline — {label} ({len(examples)} examples)")

    # Load Stage 1
    s1 = ExtractionClassifier(model_name=STAGE1_MODEL_NAME)
    s1.load_state_dict(torch.load(STAGE1_CKPT / "model.pt", map_location="cpu", weights_only=True))
    s1_tok = __import__("transformers").AutoTokenizer.from_pretrained(STAGE1_MODEL_NAME)
    s1.eval()

    # Load Stage 2
    s2 = create_span_filter(model_name=STAGE2_MODEL_NAME, dropout=0.1, freeze_embeddings=True)
    s2.load_state_dict(torch.load(STAGE2_CKPT / "model.pt", map_location="cpu", weights_only=True))
    s2_tok = __import__("transformers").AutoTokenizer.from_pretrained(STAGE2_MODEL_NAME)
    s2.eval()

    mem_before = get_memory_mb()
    t0 = time.time()
    tp = fp = fn = 0
    total_candidates = 0
    total_accepted = 0
    latencies = []

    for ex in examples:
        prompt = ex["prompt"]
        gold_spans = ex["spans"]
        if not prompt:
            continue

        t_batch = time.time()

        # Stage 1
        s1_enc = s1_tok(prompt, max_length=MAX_LENGTH, truncation=True,
                        padding="max_length", return_offsets_mapping=True)
        input_ids = torch.tensor([s1_enc["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([s1_enc["attention_mask"]], dtype=torch.long)
        with torch.no_grad():
            s1_out = s1(input_ids, attention_mask)
            s1_preds = torch.argmax(s1_out["logits"], dim=-1)
            s1_probs = torch.softmax(s1_out["logits"], dim=-1)
            s1_confs = s1_probs.gather(-1, s1_preds.unsqueeze(-1)).squeeze(-1)

        candidates = decode_bio(s1_preds[0].numpy(), s1_confs[0].numpy(),
                                attention_mask[0].numpy(), s1_enc["offset_mapping"], prompt)
        total_candidates += len(candidates)

        # Stage 2
        accepted = []
        for cand in candidates:
            s2_enc = s2_tok(prompt, cand["text"], max_length=256, truncation=True,
                            padding="max_length", return_tensors="pt")
            with torch.no_grad():
                s2_out = s2(s2_enc["input_ids"], s2_enc["attention_mask"])
                prob = s2_out["probs"].item()
            if prob >= threshold:
                accepted.append(cand)

        latencies.append(time.time() - t_batch)
        total_accepted += len(accepted)

        # Validate offsets
        validated = []
        for s in accepted:
            extracted = prompt[s["start"]:s["end"]]
            if extracted == s["text"]:
                validated.append(s)
            else:
                idx = prompt.find(s["text"])
                if idx >= 0:
                    s["start"] = idx
                    s["end"] = idx + len(s["text"])
                    validated.append(s)

        pred_set = set((s["start"], s["end"]) for s in validated)
        gold_set = set((s["start"], s["end"]) for s in gold_spans)
        for pred in pred_set:
            if pred in gold_set:
                tp += 1
            else:
                fp += 1
        for gold in gold_set:
            if gold not in pred_set:
                fn += 1

    total_time = time.time() - t0
    mem_after = get_memory_mb()
    del s1, s2, s1_tok, s2_tok
    gc.collect()

    p = tp / max(tp + fp, 1e-8)
    r = tp / max(tp + fn, 1e-8)
    f1 = 2 * p * r / max(p + r, 1e-8)

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "total_candidates": total_candidates,
        "total_accepted": total_accepted,
        "rejection_rate": round(1 - (total_accepted / max(total_candidates, 1)), 4),
        "avg_latency_ms": round(np.mean(latencies) * 1000, 2) if latencies else 0,
        "total_time_s": round(total_time, 2),
        "memory_mb": round(mem_after - mem_before, 1),
        "n": len(examples),
        "threshold": threshold,
    }


# ── Model param counter ─────────────────────────────────────────────────
def count_model_params(model_name, ckpt_path, model_cls, create_fn=None):
    """Count total, trainable, and model params."""
    if create_fn:
        model = create_fn()
    else:
        model = model_cls(model_name=model_name)
    state = torch.load(ckpt_path / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    gc.collect()
    return {"total": total, "trainable": trainable}


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="E7 Ablation Evaluation")
    parser.add_argument("--ablation", choices=["A", "B", "C", "all"], default="all")
    args = parser.parse_args()

    logger.info("Loading aligned records...")
    all_records = load_aligned_records()

    test_examples = load_test_examples(all_records)
    unseen_examples = load_unseen_examples(all_records)

    # Load threshold
    with open(THRESHOLD_PATH) as f:
        thresh_data = json.load(f)
    threshold = thresh_data["best"]["threshold"]
    logger.info(f"Stage-2 threshold: {threshold}")

    # Count params
    s1_params = count_model_params(STAGE1_MODEL_NAME, STAGE1_CKPT, ExtractionClassifier)
    s2_params = count_model_params(STAGE2_MODEL_NAME, STAGE2_CKPT, SpanFilter,
                                   create_fn=lambda: create_span_filter(
                                       model_name=STAGE2_MODEL_NAME, dropout=0.1, freeze_embeddings=True))
    logger.info(f"Stage-1 params: {s1_params}")
    logger.info(f"Stage-2 params: {s2_params}")

    all_results = {
        "threshold": threshold,
        "stage1_params": s1_params,
        "stage2_params": s2_params,
    }

    # ── E7-A ──
    if args.ablation in ("A", "all"):
        gc.collect()
        test_res = run_e7a(test_examples, label="test")
        gc.collect()
        unseen_res = run_e7a(unseen_examples, label="unseen-vocab")
        all_results["E7-A"] = {
            "test": test_res,
            "unseen_vocab": unseen_res,
        }

    # ── E7-B ──
    if args.ablation in ("B", "all"):
        gc.collect()
        test_res = run_e7b(test_examples, threshold, label="test")
        gc.collect()
        unseen_res = run_e7b(unseen_examples, threshold, label="unseen-vocab")
        all_results["E7-B"] = {
            "test": test_res,
            "unseen_vocab": unseen_res,
        }

    # ── E7-C ──
    if args.ablation in ("C", "all"):
        gc.collect()
        test_res = run_e7c(test_examples, threshold, label="test")
        gc.collect()
        unseen_res = run_e7c(unseen_examples, threshold, label="unseen-vocab")
        all_results["E7-C"] = {
            "test": test_res,
            "unseen_vocab": unseen_res,
        }

    # Save results
    out_path = BASE / "checkpoints/experiment_e7_span_filter/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("E7 ABLATION RESULTS SUMMARY")
    print("=" * 80)
    for ablation_name in ["E7-A", "E7-B", "E7-C"]:
        if ablation_name not in all_results:
            continue
        res = all_results[ablation_name]
        print(f"\n{'━' * 70}")
        label_map = {
            "E7-A": "E7-A: Stage 1 (E6-A) Only — Token-Level BIO",
            "E7-B": "E7-B: Gold Candidates → SpanFilter (Upper Bound)",
            "E7-C": "E7-C: Full Pipeline (E6-A → SpanFilter)",
        }
        print(f"  {label_map[ablation_name]}")
        print(f"{'━' * 70}")
        for split in ["test", "unseen_vocab"]:
            if split not in res:
                continue
            sr = res[split]
            print(f"\n  [{split}]")
            for k, v in sr.items():
                print(f"    {k:30s}: {v}")
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
