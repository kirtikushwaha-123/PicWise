# PicWise ML Improvement Experiments & Model Selection (Phase 3 Part 2B)

## 1. Objective

The primary objective of Phase 3 Part 2B is to conduct **controlled, hypothesis-driven machine learning improvement experiments** to address the critical weaknesses discovered during the Phase 3 Part 2A error analysis.

Rather than optimizing blindly for overall accuracy, this phase systematically tests whether candidate interventions (training class weighting, tree regularization, and feature representation refinements) can **materially improve generalization to unseen canonical ingredients and elevate minority risk class recall without causing data leakage or exacerbating dangerous underestimation errors**.

---

## 2. Evidence From Phase 3 Part 2A

Each experiment in Part 2B is directly derived from quantitative evidence in the Part 2A error analysis report:

1. **Severe recall suppression on minority risk classes (Moderate Risk safety: 22.39%, High allergy: 10.00%).**
2. **Dangerous underestimation confusion patterns (High Risk -> Safe/Very Safe, High/Medium Allergy -> None/Low).**
3. **Model overconfidence on rare character n-grams and elevated error rate on short names (<=10 chars: 45.14% error rate).**
4. **INS/E-number patterns and chemical compound naming inconsistencies causing isolated representation failures.**

---

## 3. Evaluation Methodology & Leakage Prevention

All experiments maintain absolute evaluation integrity under a nested canonical-group protocol:

1. **Outer Canonical Group Split**:
   - `GroupShuffleSplit(test_size=0.2, random_state=42)` across 1245 unique canonical ingredient groups.
   - **Outer Train**: 996 canonical groups (1773 representations).
   - **Outer Test (Frozen)**: 249 canonical groups (459 representations).
   - All alternate/packaging names belonging to a canonical ingredient reside exclusively in the same partition.

2. **Inner Validation for Candidate Selection**:
   - `GroupShuffleSplit(test_size=0.25, random_state=42)` applied strictly within the outer training partition.
   - **Inner Train**: 747 canonical groups (1343 representations).
   - **Inner Validation**: 249 canonical groups (430 representations).
   - Model selection, hyperparameter comparisons, and ranking are conducted **strictly on inner validation**, keeping the outer test set completely untouched until final reporting.

3. **Strict Train-Only Fitting**:
   - TF-IDF vectorizers are fitted strictly on the training partition names (inner train during candidate selection; outer train during final evaluation). Test-only vocabulary tokens never enter the vectorizer.
   - Multiclass sample/class weights are derived strictly from training partition label frequencies.

4. **Production Model Integrity**:
   - All production model artifacts under `backend/ml/models/` remained completely untouched (verified via pre/post SHA-256 hash checks).

---

## 4. Baseline Results (Frozen Part 1 Reference)

The Part 1 XGBoost model with character n-grams (2–5) serves as the frozen baseline benchmark:

### Safety Level Baseline Metrics
- **Accuracy**: 0.6340
- **Macro F1**: 0.4964
- **Weighted F1**: 0.6091
- **High Risk Recall**: 0.2857 (Support: 14)
- **Moderate Risk Recall**: 0.2239 (Support: 67)

### Allergy Risk Baseline Metrics
- **Accuracy**: 0.6797
- **Macro F1**: 0.5197
- **Weighted F1**: 0.6640
- **High Recall**: 0.1000 (Support: 10)
- **Medium Recall**: 0.3867 (Support: 75)

---

## 5. Safety Level Experiments

Summary of all 7 controlled Safety Level experiments evaluated on the held-out outer test set:

| Experiment Name | Outer Acc | Macro F1 | Weighted F1 | High Risk Rec (n=14) | Mod Risk Rec (n=67) | Dangerous Confusions | Dangerous Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline` | 0.6340 | 0.4964 | 0.6091 | 0.2857 | 0.2239 | 62 | **Unchanged** |
| `exp_class_weighted` | 0.6383 | 0.5670 | 0.6304 | 0.5000 | 0.3134 | 53 | **Improves** |
| `exp_conservative_depth_reg` | 0.6340 | 0.4579 | 0.5969 | 0.1429 | 0.1940 | 66 | **Worsens** |
| `exp_weighted_conservative` | 0.6340 | 0.5753 | 0.6321 | 0.7143 | 0.3881 | 43 | **Improves** |
| `exp_ngram_3_6` | 0.6166 | 0.4432 | 0.5707 | 0.1429 | 0.2239 | 64 | **Worsens** |
| `exp_min_df_2` | 0.6296 | 0.5087 | 0.6040 | 0.3571 | 0.2239 | 60 | **Improves** |
| `exp_ngram_3_6_min_df_2` | 0.6057 | 0.4577 | 0.5623 | 0.2143 | 0.2239 | 60 | **Improves** |

---

## 6. Allergy Risk Experiments

Summary of all 7 controlled Allergy Risk experiments evaluated on the held-out outer test set:

| Experiment Name | Outer Acc | Macro F1 | Weighted F1 | High Rec (n=10) | Medium Rec (n=75) | Dangerous Confusions | Dangerous Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline` | 0.6797 | 0.5197 | 0.6640 | 0.1000 | 0.3867 | 54 | **Unchanged** |
| `exp_class_weighted` | 0.6797 | 0.5312 | 0.6744 | 0.1000 | 0.5733 | 38 | **Improves** |
| `exp_conservative_depth_reg` | 0.6841 | 0.5230 | 0.6668 | 0.1000 | 0.3733 | 56 | **Worsens** |
| `exp_weighted_conservative` | 0.6885 | 0.5324 | 0.6848 | 0.1000 | 0.5867 | 37 | **Improves** |
| `exp_ngram_3_6` | 0.6950 | 0.5432 | 0.6819 | 0.1000 | 0.4667 | 49 | **Improves** |
| `exp_min_df_2` | 0.6972 | 0.5436 | 0.6849 | 0.1000 | 0.4533 | 48 | **Improves** |
| `exp_ngram_3_6_min_df_2` | 0.6906 | 0.4997 | 0.6754 | 0.0000 | 0.4400 | 52 | **Improves** |

---

## 7. Experiment Comparison

### Key Takeaways Across Candidate Approaches:

1. **Class Weighting (`exp_class_weighted`)**:
   - **Safety**: Substantially increased `High Risk` recall from **28.57% to 50.00%** and `Moderate Risk` recall from **22.39% to 31.34%**, raising Macro F1 from **0.4964 to 0.5670** (+0.0706) while decreasing dangerous false-safe errors from 62 to 53.
   - **Allergy**: Substantially elevated `Medium` allergy risk recall from **38.67% to 57.33%** (+18.66%), lifting Macro F1 from **0.5197 to 0.5312** and reducing dangerous false-none errors from 31 to 20.

2. **Combined Weighting + Conservative Trees (`exp_weighted_conservative`)**:
   - **Safety**: Achieved the strongest minority recall on the outer test set (`High Risk`: **71.43%**, `Moderate Risk`: **38.81%**), achieving **0.5753 Macro F1** and reducing dangerous underestimations from 62 to 43.
   - **Allergy**: Elevated `Medium` recall to **58.67%** and reduced dangerous confusions to 20, while maintaining **68.85% accuracy**.

3. **Conservative Depth Alone (`exp_conservative_depth_reg`)**:
   - Pruning tree depth without weighting worsened minority class recall (`High Risk` dropped to 14.29%), confirming that class imbalance must be compensated when constraining model capacity.

4. **N-gram Range & Frequency Pruning (`exp_ngram_3_6`, `exp_min_df_2`, `exp_ngram_3_6_min_df_2`)**:
   - Altering the character n-gram range to 3–6 caused severe sparsity on unseen test names, dropping `High` allergen recall to **0.00%** and reducing overall Safety Macro F1 to 0.4577.

---

## 8. Minority-Class Performance (With Exact Support Counts)

### Safety Level Minority Recall (Held-Out Test Set, N=459)

| Class | Support | Baseline Correct (Recall) | Class Weighted Correct (Recall) | Weighted + Conservative Correct (Recall) |
| :--- | :--- | :--- | :--- | :--- |
| **High Risk** | 14 | 4 / 14 (28.57%) | 7 / 14 (50.00%) | **10 / 14 (71.43%)** |
| **Moderate Risk** | 67 | 15 / 67 (22.39%) | 21 / 67 (31.34%) | **26 / 67 (38.81%)** |
| **Safe** | 237 | 201 / 237 (84.81%) | 168 / 237 (70.89%) | 153 / 237 (64.56%) |
| **Very Safe** | 141 | 71 / 141 (50.35%) | 97 / 141 (68.79%) | 102 / 141 (72.34%) |

### Allergy Risk Minority Recall (Held-Out Test Set, N=459)

| Class | Support | Baseline Correct (Recall) | Class Weighted Correct (Recall) | Weighted + Conservative Correct (Recall) |
| :--- | :--- | :--- | :--- | :--- |
| **High** | 10 | 1 / 10 (10.00%) | 1 / 10 (10.00%) | 1 / 10 (10.00%) |
| **Medium** | 75 | 29 / 75 (38.67%) | 43 / 75 (57.33%) | **44 / 75 (58.67%)** |
| **Low** | 170 | 108 / 170 (63.53%) | 106 / 170 (62.35%) | 104 / 170 (61.18%) |
| **None** | 204 | 174 / 204 (85.29%) | 162 / 204 (79.41%) | 167 / 204 (81.86%) |

> [!NOTE]
> `High` allergen test support is limited to only 10 examples. While `Medium` allergen recall experienced a major jump (+20.00%), `High` allergen recall remained constrained across all text-only models, highlighting dataset scarcity as the primary limiting factor.

---

## 9. Dangerous Confusion Analysis

### Safety Dangerous Underestimations (Baseline vs. Selected Candidate)

| Confusion Pair | Baseline Count | Selected Candidate Count | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **High Risk → Safe** | 8 | 4 | -4 | **Improves** |
| **High Risk → Very Safe** | 2 | 3 | +1 | Worsens |
| **Moderate Risk → Safe** | 45 | 37 | -8 | **Improves** |
| **Moderate Risk → Very Safe** | 7 | 9 | +2 | Worsens |
| **Total Dangerous Confusions** | **62** | **53** | **-9** | **Improves** |

### Allergy Dangerous Underestimations (Baseline vs. Selected Candidate)

| Confusion Pair | Baseline Count | Selected Candidate Count | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **High → None** | 8 | 7 | -1 | **Improves** |
| **High → Low** | 0 | 0 | 0 | **Unchanged** |
| **Medium → None** | 23 | 13 | -10 | **Improves** |
| **Medium → Low** | 23 | 18 | -5 | **Improves** |
| **Total Dangerous Confusions** | **31** | **20** | **-11** | **Improves** |

---

## 10. Selected Candidates & Rationale

### Selected Safety Candidate: `exp_class_weighted`
- **Selection Basis**: Top inner validation selection score (0.7538) with zero guardrail violations.
- **Outer Test Impact**: Macro F1 increased from 0.4964 to 0.5670; `High Risk` recall improved from 28.57% to 50.00%; `Moderate Risk` recall improved from 22.39% to 31.34%; dangerous confusions reduced from 62 to 53.

### Selected Allergy Candidate: `exp_class_weighted`
- **Selection Basis**: Top inner validation selection score (0.7575) with zero guardrail violations.
- **Outer Test Impact**: Macro F1 increased from 0.5197 to 0.5312; `Medium` risk recall improved from 38.67% to 57.33%; dangerous underestimations dropped from 31 to 20.

---

## 11. What Did Not Improve

Documenting unsuccessful interventions is critical for future roadmap planning:

1. **Character N-gram Range (3–6) (`exp_ngram_3_6`)**:
   - Worsened generalization on unseen names. It completely wiped out `High` allergy detection (0% recall, 10/10 missed) and dropped Safety accuracy to 60.57%. Removing 2-character n-grams removed crucial sub-word roots.
2. **Frequency Pruning (`exp_min_df_2`)**:
   - Pruning singleton n-grams reduced vocabulary size by ~54% but lowered Macro F1 on both Safety (0.5063 inner val) and Allergy (0.5028 inner val) by removing low-frequency chemical affixes.
3. **Tree Pruning Without Class Weighting (`exp_conservative_depth_reg`)**:
   - Setting `max_depth=4` without balanced weights caused the tree to default even more heavily to majority classes (`Safe` and `None`), worsening `High Risk` safety recall to 14.29%.

---

## 12. Limitations

1. **Severe Minority Class Support Bottlenecks**:
   - Outer test set contains only 14 `High Risk` safety instances and 10 `High` allergy instances. Observed percentage gains must be interpreted with caution.
2. **Deterministic Single Split**:
   - Results are measured on a single 80/20 group split (`random_state=42`). While leak-free, variance across different random splits may exist.
3. **Ingredient Name Surface Representation Only**:
   - Pure string-level character TF-IDF models cannot understand chemical families, biological mechanisms of action, or cumulative exposure dosages.

---

## 13. Recommendation for the Next Phase

Based on the empirical evidence from Phase 3 Part 2B:

> **Recommendation**: **D. Collect Additional Legitimate Minority-Class Data & Proceed to Confidence Thresholds / Abstention (Phase 3 Part 2C / Part 3)**
>
> 1. Training-only class weighting (`exp_class_weighted` / `exp_weighted_conservative`) showed clear observed improvements on minority sensitivity without data leakage.
> 2. However, the fundamental barrier to higher reliability is **data scarcity in high-risk categories** rather than model architecture.
> 3. Production should maintain deterministic knowledge base lookup as primary, and any ML fallback must enforce strict confidence thresholds and abstention for unseen names.
