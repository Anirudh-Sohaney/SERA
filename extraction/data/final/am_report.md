# AM-DeepSeek Pipeline — Final Report

**Date:** 2026-08-24 · **Host:** ani-cloud-server · **Status:** COMPLETE ✅

## Procedure
Two-stage pipeline (`am_pipeline.py`) over 38,000 prompts from a-m-team/AM-DeepSeek-Distilled-40M (`processed/am_prompts.jsonl`):
1. **Stage 1 — classification:** gpt-5.6-luna (low reasoning) → coding-associated vs not.
2. **Stage 2 — extraction:** for coding prompts, project-memory state extraction (medium reasoning).

Resume-safe via `.am_decisions.jsonl` checkpoint. Independent quality checks every 10% milestone (`check_am.py`), sampled against decisions log.

## Timeline
- Pipeline found already running at supervision start (PID 1515681, ~10h37m elapsed); launch skipped per runbook.
- Supervision cycles 10–19 (30-min intervals): 30,000 → 38,000 processed.
- **DONE logged 10:36:06.** Total elapsed: 71,138s ≈ **19h46m**.

## Results
| Metric | Value |
|---|---|
| Total processed | 38,000 / 38,000 |
| Classified coding | 21,127 (**55.6%**) |
| Records kept | 21,127 |
| — type `new` | 19,919 |
| — type `no_project` | 1,208 |
| Throughput | steady 0.53–0.55 rec/s |

## Milestone checks
am_check_01 … am_check_10 + `am_final`: **all CLEAN, errors=0**. Final check confirms dataset=21127 written.

## Interventions
None required. Transient conditions auto-recovered by built-in retry/backoff:
- account rate-limit backoff ("waiting 30s")
- transport retries: TimeoutError handshake, ConnectionClosedError ×2, HTTP 503/522 WebSocket rejects, one API "servers overloaded"
Zero Tracebacks across entire run.

## Artifacts
- Dataset: `final/am_dataset_coding.jsonl` (32.6 MB, 21,127 lines)
- Stats: `final/am_stats.json`
- Run log: `final/am_run.log`
- Check reports: `final/checks/am_check_{01..10}_10pct.json`
- Decisions checkpoint: `.am_decisions.jsonl` (38,000 lines)
- Supervision log: `final/am_supervision.log`

Monitoring stopped.
