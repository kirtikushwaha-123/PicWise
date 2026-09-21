# PicWise ML Evaluation & Validation Report (Phase 3 Part 1)

## 1. Executive Summary

This report evaluates the **generalization performance** of PicWise's ingredient-name machine learning models on **unseen canonical ingredients**.

To guarantee evaluation integrity, all alternate/packaging names belonging to a canonical ingredient were grouped together, ensuring zero data leakage between training and testing sets. Furthermore, the character TF-IDF vectorizer was fitted strictly on the training partition.

---

## 2. Evaluation Strategy & Data Split

- **Split Strategy**: Canonical Ingredient Group-Aware Split (`GroupShuffleSplit`)
- **Test Set Ratio**: 20%
- **Random Seed**: 42
- **Leakage Prevention**: All alternate/packaging names share the canonical ingredient group ID. TF-IDF vectorization fitted on training names only.

### Dataset Overview

| Metric | Count |
| :--- | :--- |
| **Total Representations** | 2232 |
| **Food Representations** | 1449 |
| **Personal Care Representations** | 783 |
| **Total Unique Canonical Ingredients** | 1245 |
| **Train Canonical Ingredients** | 996 |
| **Test Canonical Ingredients** | 249 |
| **Train Representations** | 1773 |
| **Test Representations** | 459 |

---

## 3. Class Distributions

### Safety Level Distribution

| Class | Overall Count (%) | Train Count (%) | Test Count (%) |
| :--- | :--- | :--- | :--- |
| **Very Safe** | 542 (24.28%) | 401 (22.62%) | 141 (30.72%) |
| **Safe** | 1282 (57.44%) | 1045 (58.94%) | 237 (51.63%) |
| **Moderate Risk** | 314 (14.07%) | 247 (13.93%) | 67 (14.6%) |
| **High Risk** | 94 (4.21%) | 80 (4.51%) | 14 (3.05%) |

### Allergy Risk Distribution

| Class | Overall Count (%) | Train Count (%) | Test Count (%) |
| :--- | :--- | :--- | :--- |
| **None** | 1048 (46.95%) | 844 (47.6%) | 204 (44.44%) |
| **Low** | 814 (36.47%) | 644 (36.32%) | 170 (37.04%) |
| **Medium** | 310 (13.89%) | 235 (13.25%) | 75 (16.34%) |
| **High** | 60 (2.69%) | 50 (2.82%) | 10 (2.18%) |

---

## 4. Safety Level Classifier Evaluation

### Baseline vs. XGBoost Performance

| Metric | Majority Baseline (Safe) | XGBoost Model | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.5163 | **0.6340** | +0.1177 |
| **Macro Precision** | 0.1291 | **0.5892** | +0.4601 |
| **Macro Recall** | 0.2500 | **0.4653** | +0.2153 |
| **Macro F1** | 0.1703 | **0.4964** | +0.3261 |
| **Weighted F1** | 0.3516 | **0.6091** | +0.2575 |

### Safety Level Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **High Risk** | 0.5000 | 0.2857 | 0.3636 | 14 |
| **Moderate Risk** | 0.5172 | 0.2239 | 0.3125 | 67 |
| **Safe** | 0.6223 | 0.8481 | 0.7179 | 237 |
| **Very Safe** | 0.7172 | 0.5035 | 0.5917 | 141 |

### Safety Level Confusion Matrix

Rows represent true labels; columns represent predictions:
`['High Risk', 'Moderate Risk', 'Safe', 'Very Safe']`

```text
[[  4   0   8   2]
 [  0  15  45   7]
 [  4  13 201  19]
 [  0   1  69  71]]
```

### Safety Level Confidence Analysis

- **Overall Test Confidence**: Mean = 0.7235, Median = 0.7313, Range = [0.3711, 0.9925]
- **Correct Predictions (291)**: Mean Confidence = 0.7594, Median = 0.7830
- **Incorrect Predictions (168)**: Mean Confidence = 0.6613, Median = 0.6259

---

## 5. Allergy Risk Classifier Evaluation

### Baseline vs. XGBoost Performance

| Metric | Majority Baseline (None) | XGBoost Model | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.4444 | **0.6797** | +0.2353 |
| **Macro Precision** | 0.1111 | **0.6418** | +0.5307 |
| **Macro Recall** | 0.2500 | **0.4937** | +0.2437 |
| **Macro F1** | 0.1538 | **0.5197** | +0.3659 |
| **Weighted F1** | 0.2735 | **0.6640** | +0.3905 |

### Allergy Risk Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **High** | 0.5000 | 0.1000 | 0.1667 | 10 |
| **Low** | 0.6879 | 0.6353 | 0.6606 | 170 |
| **Medium** | 0.7073 | 0.3867 | 0.5000 | 75 |
| **None** | 0.6718 | 0.8529 | 0.7516 | 204 |

### Allergy Risk Confusion Matrix

Rows represent true labels; columns represent predictions:
`['High', 'Low', 'Medium', 'None']`

```text
[[  1   0   1   8]
 [  0 108   8  54]
 [  0  23  29  23]
 [  1  26   3 174]]
```

### Allergy Risk Confidence Analysis

- **Overall Test Confidence**: Mean = 0.7195, Median = 0.7291, Range = [0.3498, 0.9745]
- **Correct Predictions (312)**: Mean Confidence = 0.7634, Median = 0.7811
- **Incorrect Predictions (147)**: Mean Confidence = 0.6263, Median = 0.6103

---

## 6. Interpretation & Findings

1. **Baseline Comparison**:
   - **Safety Model**: Accuracy is **0.6340** (Macro F1: **0.4964**) vs Majority Baseline **0.5163** (Macro F1: **0.1703**).
   - **Allergy Model**: Accuracy is **0.6797** (Macro F1: **0.5197**) vs Majority Baseline **0.4444** (Macro F1: **0.1538**).

2. **Class Imbalance & Hardest Classes**:
   - Safety Level hardest class: **Moderate Risk** (F1: 0.3125, support: 67).
   - Allergy Risk hardest class: **High** (F1: 0.1667, support: 10).
   - Both datasets exhibit heavy class imbalance (e.g. `Low` dominates Allergy Risk while `Safe` / `Very Safe` dominates Safety Level).

3. **Confidence Calibration Observation**:
   - Safety Model Mean Confidence: Correct predictions (0.7594) vs Incorrect predictions (0.6613).
   - Allergy Model Mean Confidence: Correct predictions (0.7634) vs Incorrect predictions (0.6263).
   - Higher confidence on correct predictions indicates useful probability separation, which can be leveraged for confidence thresholds/abstention in Phase 3 Part 2.

4. **Conclusion & Recommendation**:
   - Character n-gram TF-IDF on ingredient names alone provides modest predictive signal beyond majority class guessing on unseen canonical ingredients.
   - For high-stakes safety and allergy classifications, deterministic knowledge base lookup should remain the primary authority, with ML used as a secondary fallback accompanied by confidence thresholds.
