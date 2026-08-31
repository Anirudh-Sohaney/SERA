# SERA Admission Policy Comparison

**Timestamp:** 2026-08-30T22:55:37.512949+00:00
**Policies tested:** 4

## Results by Policy

| Policy | FLR | State P | State R | State F1 | Span F1 | Locks | Pending | Discard | Time (s) |
|--------|-----|---------|---------|----------|---------|-------|---------|---------|----------|
| Policy_A | 0.6844 | 0.1555 | 0.0943 | 0.1087 | 0.1747 | 1931 | 23994 | 6239 | 3106.1 |
| Policy_B | 0.7755 | 0.1580 | 0.1232 | 0.1250 | 0.2259 | 3321 | 15661 | 13182 | 8.3 |
| Policy_C | 0.8030 | 0.1589 | 0.1412 | 0.1328 | 0.2377 | 4341 | 27508 | 315 | 9.6 |
| Policy_D | 0.4401 | 0.1482 | 0.0634 | 0.0850 | 0.0920 | 745 | 31008 | 411 | 5.0 |

## Best Policy (Lowest False Lock Rate)

**Policy_D**
- False Lock Rate: 0.4401
- State Precision: 0.1482
- State Recall: 0.0634
- State F1: 0.0850