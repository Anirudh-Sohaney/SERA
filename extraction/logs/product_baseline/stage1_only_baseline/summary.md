# SERA Product Baseline Evaluation

**Timestamp:** 2026-08-30T19:32:36.105967+00:00
**Conversations evaluated:** 2238
**Conversations failed:** 0

## Extraction Metrics

- **Span Precision:** 0.1306
- **Span Recall:** 0.4259
- **Span F1:** 0.1860
- **Total Spans:** 32164

## State Metrics

- **State Precision:** 0.0469
- **State Recall:** 0.2382
- **State F1:** 0.0723

## Error Rates (PRIMARY)

- **FALSE LOCK RATE:** 0.9514
- **False Update Rate:** 0.0022
- **False Removal Rate:** 0.0000
- **False Rejection Rate:** 0.0970
- **Stale Memory Rate:** 0.9531
- **Duplicate Memory Rate:** 0.0001
- **Contradiction Rate:** 0.0000

## Transition Counts

- **ADD:** 29164
- **MODIFY:** 11
- **REMOVE:** 0
- **REJECT:** 887
- **NO_CHANGE:** 2102

## False Lock Analysis

- **Total ADDs:** 29164
- **False Locks:** 27986
- **False Lock Rate:** 0.9596

## Pipeline Statistics

- **Total Candidates:** 32164
- **Stage 1 Candidates:** 32164
- **Stage 2 Accepted:** 32164
- **Locked:** 4756
- **Pending:** 14684
- **Discarded:** 12724

## Error Attribution (A-G)

| Code | Category | Count |
|------|----------|-------|
| A | Extraction failure | 31026 |
| B | Candidate typing failure | 1068 |
| C | Matching failure | 0 |
| D | Transition-rule failure | 6418 |
| E | Validation failure | 0 |
| F | Persistence/state failure | 29798 |
| G | Evaluation ambiguity | 0 |