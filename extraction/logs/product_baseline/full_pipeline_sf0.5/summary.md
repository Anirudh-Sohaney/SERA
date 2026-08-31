# SERA Product Baseline Evaluation

**Timestamp:** 2026-08-30T20:15:23.615970+00:00
**Conversations evaluated:** 2238
**Conversations failed:** 0

## Extraction Metrics

- **Span Precision:** 0.3130
- **Span Recall:** 0.3325
- **Span F1:** 0.2902
- **Total Spans:** 8322

## State Metrics

- **State Precision:** 0.1115
- **State Recall:** 0.1760
- **State F1:** 0.1213

## Error Rates (PRIMARY)

- **FALSE LOCK RATE:** 0.8443
- **False Update Rate:** 0.0009
- **False Removal Rate:** 0.0004
- **False Rejection Rate:** 0.0326
- **Stale Memory Rate:** 0.8648
- **Duplicate Memory Rate:** 0.0000
- **Contradiction Rate:** 0.0000

## Transition Counts

- **ADD:** 7951
- **MODIFY:** 2
- **REMOVE:** 1
- **REJECT:** 124
- **NO_CHANGE:** 244

## False Lock Analysis

- **Total ADDs:** 7951
- **False Locks:** 6952
- **False Lock Rate:** 0.8744

## Pipeline Statistics

- **Total Candidates:** 8322
- **Stage 1 Candidates:** 24575
- **Stage 2 Accepted:** 8322
- **Locked:** 2580
- **Pending:** 4235
- **Discarded:** 1507

## Error Attribution (A-G)

| Code | Category | Count |
|------|----------|-------|
| A | Extraction failure | 10610 |
| B | Candidate typing failure | 854 |
| C | Matching failure | 0 |
| D | Transition-rule failure | 6597 |
| E | Validation failure | 0 |
| F | Persistence/state failure | 9518 |
| G | Evaluation ambiguity | 0 |