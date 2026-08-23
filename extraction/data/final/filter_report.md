# Coding-Prompt Filtering — Final Report

**Date:** 2026-08-22 → 2026-08-23
**Machine:** ani-cloud-server
**Task:** Classify all records in `final/dataset.jsonl` as coding/project-associated (1) or not (0) using `gpt-5.6-luna` (low reasoning); assemble kept records into `final/dataset_coding.jsonl`.

## Procedure

1. Launched `python3 filter_coding.py` via `setsid nohup` (PID **1027739**) on 2026-08-22 at 02:17. The script loads all 65,377 records, classifies each with a two-pass gpt-5.6-luna call, appends decisions incrementally to `.filter_decisions.jsonl` (resume-safe), and writes milestone checks every 10%.
2. Supervised in ~30-minute cycles: process liveness (`pgrep`/`ps`), progress (`wc -l .filter_decisions.jsonl`, log tail), milestone check verification (`final/checks/filter_check_*.json`), and crash scan (`grep -c Traceback`). Each cycle appended a status line to `final/filter_supervision.log`.
3. Rate-limit warnings ("rate limited on N; waiting 30s") appeared continuously throughout; these are the script's built-in account-throttle backoff and were not treated as anomalies.
4. On completion (log shows `DONE`, `final/filter_stats.json` written), ran `check_dataset.py --dataset final/dataset_coding.jsonl --label dataset_coding_final` to validate the assembled output.

## Timeline

| Event | Time |
|---|---|
| Launch (PID 1027739) | 2026-08-22 02:17 |
| First milestone check (10%) | early Aug 22 |
| Final milestone check (99.99%) | Aug 23 ~00:45 |
| DONE + stats written | 2026-08-23 01:02 |
| Total elapsed | 81,906 s ≈ 22.75 h |

## Results

| Metric | Value |
|---|---|
| Total classified | **65,377 / 65,377** (100%) |
| Kept (coding=1) | **7,251** (11.09%) |
| Dropped (coding=0) | **58,126** (88.91%) |
| Invalid parses | **0** |
| Malformed decision lines | **0** |
| Throughput | ~0.80 records/s |

## Milestone checks

10/10 checks passed (`filter_check_01_10pct.json` … `filter_check_10_10pct.json`), all with `ok=true`, `error_count=0`, `malformed_lines=0`. Final check at 65,370/65,377 confirmed kept=7,251 / dropped=58,119.

Post-assembly validation: `[dataset_coding_final] CLEAN | records=7251 errors=0` with type breakdown `{'new': 2419, 'no_change': 540, 'update': 570, 'no_project': 3722}`.

## Interventions

**None.** Zero tracebacks, zero crashes, no restarts, no stalled cycles. Process ran start-to-finish on the original PID. Supervision was purely observational (42 logged cycles).

Note: supervision log has a gap between cycle24 (Aug 22 14:05) and cycle25 (Aug 22 16:06) due to prior supervisor idling; the filter run itself was unaffected and healthy throughout.

## Artifacts

| Artifact | Path |
|---|---|
| Source dataset | `~/sera_models/extraction/data/final/dataset.jsonl` |
| Raw decisions (resume ledger) | `~/sera_models/extraction/data/.filter_decisions.jsonl` (65,377 lines) |
| Filtered output | `~/sera_models/extraction/data/final/dataset_coding.jsonl` (7,251 records) |
| Run stats | `~/sera_models/extraction/data/final/filter_stats.json` |
| Run log | `~/sera_models/extraction/data/final/filter_run.log` |
| Milestone checks | `~/sera_models/extraction/data/final/checks/filter_check_{01..10}_10pct.json` |
| Final dataset validation | `~/sera_models/extraction/data/final/checks/dataset_coding_final.json` |
| Supervision log | `~/sera_models/extraction/data/final/filter_supervision.log` |
