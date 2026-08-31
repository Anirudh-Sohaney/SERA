# Stage-1 Candidate Error Analysis

**Timestamp:** 2026-08-30T23:05:07.264292+00:00

## Distribution

- **Total candidates:** 32164
- **Total gold spans:** 7598
- **True positives:** 3349
- **False positives:** 28815
- **False negatives:** 4306
- **Precision:** 0.1041
- **Recall:** 0.4375

## False Positive Analysis

**Count:** 28815

### Confidence Distribution
- Mean: 0.7244
- Median: 0.7279
- P25: 0.6198
- P75: 0.8311

### Category Distribution
- requirement: 28424 (98.6%)
- language: 254 (0.9%)
- directory: 65 (0.2%)
- framework: 59 (0.2%)
- platform: 9 (0.0%)
- database: 4 (0.0%)

### Type Analysis
- technology: 646 (2.2%)
- description: 3318 (11.5%)
- code_syntax: 2460 (8.5%)
- negated: 205 (0.7%)

### Length Distribution
- single_word: 14163 (49.2%)
- short_2_3: 10827 (37.6%)
- medium_4_6: 3482 (12.1%)
- long_7_plus: 343 (1.2%)

### Position Distribution
- start: 14698 (51.0%)
- middle: 8213 (28.5%)
- end: 5904 (20.5%)

## False Negative Analysis

**Count:** 4306

### Category Distribution
- requirement: 4035 (93.7%)
- language: 210 (4.9%)
- directory: 32 (0.7%)
- framework: 19 (0.4%)
- platform: 7 (0.2%)
- database: 2 (0.0%)
- file: 1 (0.0%)

### Type Analysis
- technology: 304 (7.1%)
- description: 852 (19.8%)