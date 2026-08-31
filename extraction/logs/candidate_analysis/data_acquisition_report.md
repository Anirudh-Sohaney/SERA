# Data Acquisition Report — SERA Stage-1 Error Analysis

**Timestamp:** 2026-08-30
**Analysis basis:** 2,238 test conversations, 32,164 Stage-1 candidates

---

## Error Summary

| Metric | Count |
|--------|-------|
| Total candidates | 32,164 |
| True positives | 3,349 |
| False positives | 28,815 |
| False negatives | 4,306 |
| Precision | 10.4% |
| Recall | 43.8% |

The extractor produces ~9 false positives for every true positive.

---

## False Positive Breakdown

### By Category
| Category | Count | % of FP |
|----------|-------|---------|
| requirement | 28,424 | 98.6% |
| language | 254 | 0.9% |
| directory | 65 | 0.2% |
| framework | 59 | 0.2% |
| platform | 9 | 0.0% |
| database | 4 | 0.0% |

**Key finding:** 98.6% of false positives are categorized as generic "requirement". The model extracts almost any short phrase from the prompt as project information.

### By Type
| Type | Count | % of FP |
|------|-------|---------|
| description | 3,318 | 11.5% |
| code_syntax | 2,460 | 8.5% |
| technology | 646 | 2.2% |
| negated | 205 | 0.7% |

### By Length
| Length | Count | % of FP |
|--------|-------|---------|
| single word | 14,163 | 49.2% |
| 2-3 words | 10,827 | 37.6% |
| 4-6 words | 3,482 | 12.1% |
| 7+ words | 343 | 1.2% |

### By Confidence
- Mean: 0.724
- Median: 0.728
- P25: 0.620
- P75: 0.831

False positives have high confidence. Raising the threshold alone will not solve this — the model is confidently wrong.

---

## False Negative Breakdown

### By Category
| Category | Count | % of FN |
|----------|-------|---------|
| requirement | 4,035 | 93.7% |
| language | 210 | 4.9% |
| directory | 32 | 0.7% |
| framework | 19 | 0.4% |

---

## Error Category Analysis

### A. Description vs Requirement Distinction (HIGHEST PRIORITY)

The dominant error: the model cannot distinguish between:

- "Create a **bash** ping script" → SHOULD extract "bash"
- "I want to create a **web application**" → SHOULD NOT extract "web application"
- "Build a **backend** that handles requests" → SHOULD NOT extract "backend"
- "Write **Python code** for data processing" → SHOULD extract "Python"

The model treats all noun phrases as extractable. It needs to learn:
- Technology names (Python, Flask, PostgreSQL) → EXTRACT
- Generic descriptions (web application, backend, code) → DO NOT EXTRACT
- Function signatures (create a bash script) → EXTRACT only the technology

**Training data needed:** Pairs of prompts where one contains a technology and one contains a description, with different gold labels.

### B. Negation Detection (HIGH PRIORITY)

205 false positives are negated. Examples:
- "Do not use **PostgreSQL**" → model extracts "PostgreSQL" as positive
- "Instead of **React**, use Vue" → model extracts "React"

**Training data needed:** Negation pairs: "Use X" vs "Do not use X" with opposite labels.

### C. Code Syntax Extraction (MEDIUM PRIORITY)

2,460 false positives are code syntax fragments. The model extracts:
- Function signatures: "def create_app()"
- Import statements: "from flask import"
- Variable names: "const db_connection"

**Training data needed:** Prompts containing code snippets where code elements should NOT be extracted as project memory.

### D. Technology Misclassification (LOW PRIORITY)

646 false positives are actual technology mentions but wrong. These may be:
- Technologies mentioned but rejected: "We considered Rust but chose Go"
- Technologies in examples: "Like how Django handles ORM"
- Technologies in comparisons: "Flask vs Django"

**Training data needed:** Rejection examples, comparison examples, hypothetical examples.

---

## Recommendation: Whether Additional Data Is Required

### Primary bottleneck: NOT data quantity

The model has 22,908 training examples and achieves 43.8% recall. The problem is NOT insufficient data — it is insufficient **semantic discrimination**. The model cannot tell the difference between:

1. A technology name that should be extracted
2. A generic description that should not be extracted
3. A negated mention that should be rejected

### Primary bottleneck: Training label quality

The current gold labels mark almost every noun phrase in coding prompts as "project_info". This includes:
- "bash" (correct)
- "web application" (debatable)
- "check uptime" (debatable)
- "ping script" (debatable)

The model learns to extract everything because the training data labels everything.

### What would actually help

1. **Hard negative examples** (estimated value: HIGH)
   - 500-1000 prompts where the correct answer is NO_EXTRACTION
   - Prompts with descriptions, generic terms, code snippets
   - Currently the model never sees "don't extract" examples

2. **Negation pairs** (estimated value: MEDIUM)
   - 200-500 prompts with negated technology mentions
   - "Use Flask" vs "Don't use Flask"

3. **Boundary refinement** (estimated value: MEDIUM)
   - Examples where the exact span boundary matters
   - "PostgreSQL database" vs "PostgreSQL"
   - "React frontend" vs "React"

4. **NOT recommended: Random additional prompts**
   - Adding more random coding prompts will not help
   - The model already has sufficient vocabulary coverage
   - The problem is semantic, not lexical

---

## Decision: TARGETED DATA REQUISITION — JUSTIFIED

Additional data IS justified, but ONLY if it is:
- Targeted at specific error categories
- Balanced (not just more positives)
- Validated for correctness
- Small in quantity (500-2000 examples)

The data should focus on:
1. Hard negatives (NO_EXTRACTION prompts)
2. Negation pairs
3. Description vs technology distinction
4. Boundary-sensitive examples

**Do NOT:**
- Collect random additional prompts
- Augment with template-based synthetic data
- Expand beyond 2000 targeted examples
- Repeat the E6-B mistake of over-augmentation
