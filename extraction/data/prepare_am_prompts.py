"""Build the AM-DeepSeek prompts-only sample for the coding-extraction pipeline.

Source: a-m-team/AM-DeepSeek-Distilled-40M (released subset). Each unique
question appears 12x in the full dataset (3 models x 4 passes); we read one
pass (r1_1pass) per domain via the HF datasets-server rows API, so every
question is seen exactly once and no answer bytes are kept beyond the API
response.

Output: processed/am_prompts.jsonl
    {"record_id", "domain", "category", "question_source", "prompt"}

Filters: English-leaning (ASCII ratio >= 0.85), dedup by question hash.
Sampling: contiguous blocks at evenly spaced offsets (avoids source-ordering
bias while keeping API requests low).

Usage: python3 prepare_am_prompts.py [--full]
    default samples 20k code / 12k math / 6k if; --full takes everything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://datasets-server.huggingface.co/rows"
DATASET = "a-m-team/AM-DeepSeek-Distilled-40M"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "processed", "am_prompts.jsonl")

# domain, config, total_rows, target_sample
DOMAINS = [
    ("code", "code_r1_1pass", 200_866, 20_000),
    ("math", "math_r1_1pass", 413_062, 12_000),
    ("if", "if_r1_1pass", 76_520, 6_000),
]
BLOCK = 500          # rows per contiguous block
PAGE = 100           # rows API max length
OVERSAMPLE = 1.35    # to compensate for English/dedup filtering
ASCII_MIN = 0.85


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text)


def fetch_page(config: str, offset: int, length: int, retries: int = 6) -> list[dict]:
    url = (f"{BASE}?dataset={urllib.parse.quote(DATASET, safe='')}"
           f"&config={config}&split=train&offset={offset}&length={length}")
    wait = 10
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sera-extraction/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            if "rows" in data:
                return [r["row"] for r in data["rows"]]
            raise RuntimeError(str(data)[:200])
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}/{retries} at offset {offset}: {e}; "
                  f"waiting {wait}s", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 120)
    return []


def sample_domain(domain: str, config: str, total: int, target: int) -> list[dict]:
    need = int(target * OVERSAMPLE)
    n_blocks = max(1, need // BLOCK)
    out: list[dict] = []
    print(f"[{domain}] {config}: total={total:,} target={target:,} "
          f"({n_blocks} blocks x {BLOCK})", flush=True)
    for b in range(n_blocks):
        offset = (b * (total - BLOCK)) // max(1, n_blocks - 1) if n_blocks > 1 else 0
        got = 0
        for p in range(BLOCK // PAGE):
            rows = fetch_page(config, offset + p * PAGE, PAGE)
            got += len(rows)
            for r in rows:
                q = (r.get("question") or "").strip()
                if not q or ascii_ratio(q) < ASCII_MIN:
                    continue
                out.append({
                    "domain": domain,
                    "category": r.get("category"),
                    "question_source": r.get("question_source"),
                    "prompt": q,
                })
        print(f"  block {b + 1}/{n_blocks} offset={offset:>8,} rows={got} "
              f"kept_so_far={len(out)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="sample every English question instead of a subset")
    args = ap.parse_args()

    random.seed(42)
    all_records: list[dict] = []
    seen: set[str] = set()

    for domain, config, total, target in DOMAINS:
        tgt = total if args.full else target
        rows = sample_domain(domain, config, total, tgt)
        # dedup within domain, trim to target
        uniq, dropped_dup = [], 0
        for r in rows:
            h = hashlib.md5(r["prompt"].encode()).hexdigest()
            if h in seen:
                dropped_dup += 1
                continue
            seen.add(h)
            uniq.append(r)
        random.shuffle(uniq)
        uniq = uniq[:tgt]
        for k, r in enumerate(uniq):
            r["record_id"] = f"am-{domain}-{k:06d}"
        all_records.extend(uniq)
        print(f"[{domain}] kept {len(uniq):,} (dropped {dropped_dup} dups)",
              flush=True)

    random.shuffle(all_records)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_domain = Counter(r["domain"] for r in all_records)
    print(f"\nWROTE {len(all_records):,} prompts -> {OUT_PATH}")
    print("by domain:", dict(by_domain))
    return 0


if __name__ == "__main__":
    sys.exit(main())