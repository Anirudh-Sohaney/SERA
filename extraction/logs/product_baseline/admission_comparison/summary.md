# SERA Admission Policy Comparison

**Timestamp:** 2026-08-30T22:02:43.958720+00:00
**Policies tested:** 4

## Results by Policy

| Policy | FLR | State P | State R | State F1 | Span F1 | Locks | Pending | Discard | Time (s) |
|--------|-----|---------|---------|----------|---------|-------|---------|---------|----------|
| Policy_A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 25925 | 6239 | 2905.8 |
| Policy_B | 0.8730 | 0.1265 | 0.1722 | 0.1300 | 0.2700 | 7709 | 11273 | 13182 | 12.5 |
| Policy_C | 0.9041 | 0.1047 | 0.2047 | 0.1239 | 0.2734 | 11501 | 20348 | 315 | 16.6 |
| Policy_D | 0.8535 | 0.1448 | 0.1658 | 0.1373 | 0.2654 | 6554 | 25199 | 411 | 10.5 |

## Best Policy (Lowest False Lock Rate)

**Policy_A**
- False Lock Rate: 0.0000
- State Precision: 0.0000
- State Recall: 0.0000
- State F1: 0.0000