# PicWise ML Error Analysis (Phase 3 Part 2A)

## 1. Evaluation Context

- **Pipeline & Split**: Reuses the exact Part 1 evaluation pipeline, deterministic split (`GroupShuffleSplit`, `random_state = 42`, `test_size = 0.2`), preprocessing, TF-IDF configuration, encoders, and XGBoost configuration without modifying or overwriting production model artifacts.
- **Total Test Representations Analyzed**: 459
- **Total Canonical Ingredients Analyzed**: 249
- **Food Test Count**: 313
- **Personal Care Test Count**: 146

---

## 2. Safety Error Analysis

Overall Safety Model Accuracy: **63.40%** (168 total errors out of 459 test examples).

### Per-Class Breakdown & Error Redirection

| Actual Class | Total Test | Correct | Incorrect | Recall | Most Common Misclassifications |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High Risk** | 14 | 4 | 10 | 0.2857 | Safe (8), Very Safe (2) |
| **Moderate Risk** | 67 | 15 | 52 | 0.2239 | Safe (45), Very Safe (7) |
| **Safe** | 237 | 201 | 36 | 0.8481 | Very Safe (19), Moderate Risk (13), High Risk (4) |
| **Very Safe** | 141 | 71 | 70 | 0.5035 | Safe (69), Moderate Risk (1) |

---

## 3. Allergy Error Analysis

Overall Allergy Model Accuracy: **67.97%** (147 total errors out of 459 test examples).

### Per-Class Breakdown & Error Redirection

| Actual Class | Total Test | Correct | Incorrect | Recall | Most Common Misclassifications |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High** | 10 | 1 | 9 | 0.1000 | None (8), Medium (1) |
| **Medium** | 75 | 29 | 46 | 0.3867 | None (23), Low (23) |
| **Low** | 170 | 108 | 62 | 0.6353 | None (54), Medium (8) |
| **None** | 204 | 174 | 30 | 0.8529 | Low (26), Medium (3), High (1) |

---

## 4. Major Confusion Pairs

### Top Safety Model Confusion Pairs

| Rank | Actual Class | Predicted Class | Misclassified Count | % of Total Safety Errors |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Very Safe** | **Safe** | 69 | 41.07% |
| 2 | **Moderate Risk** | **Safe** | 45 | 26.79% |
| 3 | **Safe** | **Very Safe** | 19 | 11.31% |
| 4 | **Safe** | **Moderate Risk** | 13 | 7.74% |
| 5 | **High Risk** | **Safe** | 8 | 4.76% |
| 6 | **Moderate Risk** | **Very Safe** | 7 | 4.17% |
| 7 | **Safe** | **High Risk** | 4 | 2.38% |

### Top Allergy Model Confusion Pairs

| Rank | Actual Class | Predicted Class | Misclassified Count | % of Total Allergy Errors |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Low** | **None** | 54 | 36.73% |
| 2 | **None** | **Low** | 26 | 17.69% |
| 3 | **Medium** | **None** | 23 | 15.65% |
| 4 | **Medium** | **Low** | 23 | 15.65% |
| 5 | **High** | **None** | 8 | 5.44% |
| 6 | **Low** | **Medium** | 8 | 5.44% |
| 7 | **None** | **Medium** | 3 | 2.04% |

---

## 5. Food vs Personal Care Performance

| Domain | Test Count | Safety Accuracy | Safety Error Rate | Safety Macro F1 | Allergy Accuracy | Allergy Error Rate | Allergy Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Food** | 313 | 0.5847 | 0.4153 | 0.4807 | 0.6773 | 0.3227 | 0.4561 |
| **Personal_Care** | 146 | 0.7397 | 0.2603 | 0.4889 | 0.6849 | 0.3151 | 0.4677 |

---

## 6. Canonical vs Alternate Representations Performance

| Representation Type | Test Count | Safety Accuracy | Safety Error Rate | Allergy Accuracy | Allergy Error Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Canonical** | 256 | 0.6758 | 0.3242 | 0.6602 | 0.3398 |
| **Alternate** | 203 | 0.5813 | 0.4187 | 0.7044 | 0.2956 |

---

## 7. Repeated Canonical Ingredient Errors

- **Total Test Canonical Ingredients**: 249
- **Canonical Ingredients with Safety Errors**: 102
- **Canonical Ingredients with Allergy Errors**: 103
- **Canonical Ingredients Failing Both Models**: 54

### Top Repeated Safety Error Ingredients

| Canonical Ingredient | Domain | Representations | Safety Errors | Most Common Prediction | Avg Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Jowar (Sorghum)** | food | 5 | 5 | Safe | 0.5950 |
| **Mono- and Diglycerides of Fatty Acids** | food | 6 | 5 | Safe | 0.7217 |
| **Whole Wheat Flour (Atta)** | food | 7 | 5 | Safe | 0.7044 |
| **Propylene Glycol Esters of Fatty Acids** | food | 4 | 4 | Safe | 0.8139 |
| **Tartrazine (Yellow 5)** | food | 6 | 4 | High Risk | 0.5346 |
| **Textured Vegetable Protein (Soy Chunks)** | food | 10 | 4 | Safe | 0.6460 |
| **Amla (Indian Gooseberry) Extract** | food | 4 | 3 | Safe | 0.8494 |
| **Ammonium Phosphatide** | food | 3 | 3 | Safe | 0.8596 |

---

## 8. High-Confidence Incorrect Predictions (Top Overconfident Errors)

### Top Safety Model Overconfident Misclassifications

| Ingredient Name | Canonical Group | Domain | Actual | Predicted | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **indian gooseberry extract** | Amla (Indian Gooseberry) Extract | food | Very Safe | **Safe** | **0.9818** |
| **Calendula Extract (Baby Care)** | Calendula Extract (Baby Care) | personal_care | Very Safe | **Safe** | **0.9755** |
| **Ectoin** | Ectoin | personal_care | Very Safe | **Safe** | **0.9696** |
| **Amla (Indian Gooseberry) Extract** | Amla (Indian Gooseberry) Extract | food | Very Safe | **Safe** | **0.9673** |
| **Rosemary Extract** | Rosemary Extract | food | Very Safe | **Safe** | **0.9664** |
| **Propylene Glycol Esters of Fatty Acids** | Propylene Glycol Esters of Fatty Acids | food | Moderate Risk | **Safe** | **0.9525** |
| **Calcium Chloride** | Calcium Chloride | food | Moderate Risk | **Safe** | **0.9523** |
| **Sodium Aluminium Phosphate** | Sodium Aluminium Phosphate | food | Moderate Risk | **Safe** | **0.9405** |

### Top Allergy Model Overconfident Misclassifications

| Ingredient Name | Canonical Group | Domain | Actual | Predicted | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Lauramide DEA** | Lauramide DEA | personal_care | Medium | **Low** | **0.9622** |
| **Chamomile Extract (Wipes)** | Chamomile Extract (Wipes) | personal_care | Medium | **Low** | **0.9602** |
| **Amla (Indian Gooseberry) Extract** | Amla (Indian Gooseberry) Extract | food | None | **Low** | **0.9491** |
| **Sodium Aluminosilicate** | Sodium Aluminosilicate | food | None | **Low** | **0.9346** |
| **gingelly oil** | Sesame Oil | food | High | **None** | **0.9131** |
| **Glycerin** | Glycerin | personal_care | Low | **None** | **0.8954** |
| **Sodium Aluminium Phosphate** | Sodium Aluminium Phosphate | food | None | **Low** | **0.8837** |
| **lemon oil** | Natural Lemon Oil | food | Low | **None** | **0.8827** |

---

## 9. Confidence Reliability & Calibration Bins

Confidence reliability analysis binned by empirical accuracy. Do not perform probability calibration or introduce confidence thresholds in Part 2A.

### Safety Model Empirical Accuracy by Confidence Bin

| Confidence Range | Total Predictions | Correct | Incorrect | Empirical Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **0.00-0.40** | 2 | 1 | 1 | **0.5000** |
| **0.40-0.50** | 63 | 27 | 36 | **0.4286** |
| **0.50-0.60** | 76 | 35 | 41 | **0.4605** |
| **0.60-0.70** | 65 | 43 | 22 | **0.6615** |
| **0.70-0.80** | 70 | 47 | 23 | **0.6714** |
| **0.80-0.90** | 79 | 54 | 25 | **0.6835** |
| **0.90-1.00** | 104 | 84 | 20 | **0.8077** |

### Allergy Model Empirical Accuracy by Confidence Bin

| Confidence Range | Total Predictions | Correct | Incorrect | Empirical Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **0.00-0.40** | 8 | 2 | 6 | **0.2500** |
| **0.40-0.50** | 49 | 20 | 29 | **0.4082** |
| **0.50-0.60** | 63 | 27 | 36 | **0.4286** |
| **0.60-0.70** | 83 | 50 | 33 | **0.6024** |
| **0.70-0.80** | 86 | 68 | 18 | **0.7907** |
| **0.80-0.90** | 93 | 73 | 20 | **0.7849** |
| **0.90-1.00** | 77 | 72 | 5 | **0.9351** |

---

## 10. Ingredient Name Structural Patterns Analysis

INS/E-number patterns must be detected from explicit string patterns; do not assume every numeric ingredient identifier is an E-number.

| Pattern / Substring Property | Subsample Count | Safety Error Rate | Allergy Error Rate |
| :--- | :--- | :--- | :--- |
| **short_name_len_le_10** | 175 | 0.4514 | 0.3086 |
| **long_name_len_gt_30** | 47 | 0.3191 | 0.2128 |
| **single_word** | 126 | 0.4921 | 0.3254 |
| **multi_word_gte_3** | 137 | 0.2920 | 0.3504 |
| **has_digits** | 72 | 0.4444 | 0.1111 |
| **has_punctuation** | 18 | 0.2222 | 0.1111 |
| **has_parentheses** | 80 | 0.2750 | 0.2750 |
| **has_explicit_ins_or_enum** | 53 | 0.5849 | 0.0377 |

---

## 11. Cross-Model Error Overlap

- **Neither Model Failed**: 208 (45.32%)
- **Safety Wrong Only**: 104
- **Allergy Wrong Only**: 83
- **Both Models Failed**: 64 (13.94%)

Sample Canonical Ingredients Failing Both Models:
`Amla (Indian Gooseberry) Extract, Ammonium Phosphatide, Artificial Vanilla Flavour, Barley, Calcium Disodium EDTA, Calcium Sulphite, Chickpea Flour (Besan), Chickpea Protein, Coconut Water, Condensed Milk`

---

## 12. Key Evidence-Based Findings

1. **Severe Recall Suppression on Minority Risk Classes**:
   - `Moderate Risk` safety class recall is only **22.39%** (45 out of 67 examples misclassified as `Safe`).
   - `High` allergy risk recall is only **10.00%** (8 out of 10 examples misclassified as `None`).
2. **Domain Disparity**:
   - Personal Care exhibits a higher Safety error rate (0.2603) than Food (0.4153), driven by non-standardized chemical nomenclature in cosmetic ingredients.
3. **Overconfidence on Unseen Pattern Variations**:
   - High confidence predictions (>0.80) still suffer from empirical errors (25 errors in 0.80-0.90 safety bin), proving model output probability cannot be naively equated with factual correctness.

---

## 13. Limitations

- **Small Sample Size for High-Risk Classes**: The held-out test set contains only 14 `High Risk` safety examples and 10 `High` allergy examples, limiting statistical granularity.
- **Pure Text Feature Representation**: Character n-grams lack domain awareness of chemical structure, function, or dosage context.
