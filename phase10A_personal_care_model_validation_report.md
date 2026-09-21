# Phase 10A Final Report: Personal Care Model Validation & Optimization

## 1. Objective

Phase 10A establishes a rigorous, reproducible, leakage-free validation and optimization pipeline for the already-selected Personal Care production configuration:

$$\text{Ingredient Name (character-level TF-IDF)} + \text{Structured Semantics} \longrightarrow \text{Logistic Regression}$$

The primary goals of this phase:
1. Conduct a thorough reproducibility audit across the repository to determine whether the historical benchmark metrics ($\text{Safety: } 0.7410, \text{ Allergy: } 0.7432, \text{ Irritation: } 0.7210, \text{ Mean: } 0.7351$) can be traced to documented codebase configurations.
2. Establish the authoritative Personal Care dataset (`final personal care dataset.csv`, 926 rows, 10 columns, 0 nulls) within `data/personal_care/`.
3. Design and verify a deterministic canonical-grouping protocol that prevents cross-fold data leakage across all textual representations and application-context variants ($\text{Train groups} \cap \text{Validation groups} = \emptyset$).
4. Implement a leak-free 5-fold `StratifiedGroupKFold` cross-validation protocol.
5. Evaluate all three orthogonal targets (`Safety_Level`, `Allergy_Risk`, `Irritation_Risk`) independently under a controlled Logistic Regression hyperparameter grid ($C \in \{0.1, 0.5, 1.0, 2.0, 5.0, 10.0\}$, $\text{class\_weight} \in \{\text{None}, \text{"balanced"}\}$).
6. Perform in-depth error, confidence, and dangerous confusion analysis on out-of-fold (OOF) predictions.
7. Deliver automated unit and regression tests, verifying 100% reproducibility and backward compatibility.

---

## 2. Dataset Source

The repository previously tracked only truncated versions of the Personal Care dataset in `data/personal_care/`:
- `personal_care_ingredients_dataset_cleaned.xlsx` (786 rows, 9 columns, lacking alternate names).
- `personal_care_ingredients_dataset_csv.xlsx` (786 rows, 9 columns, lacking alternate names).

The authoritative, finalized Personal Care dataset was located on the user system at:
`C:\Users\velzyaa\Desktop\PicWise DataSet\final personal care dataset.csv`
and explicitly provided in the user prompt.

In Phase 10A, this authoritative dataset was ingested into the repository at:
`data/personal_care/final_personal_care_dataset.csv`

---

## 3. Dataset Audit

A comprehensive audit of `data/personal_care/final_personal_care_dataset.csv` yielded the following verified properties:

| Metric | Value | Audit Finding |
| :--- | :--- | :--- |
| **Row Count** | 926 | Exactly 926 ingredient entries |
| **Column Count** | 10 | 10 feature and target columns |
| **Missing / Null Values** | **0** | Clean, complete dataset; zero null values across all columns |
| **Duplicate `Ingredient_Name`** | **0** | Every single row has a unique primary ingredient name |
| **Unique Canonical Groups** | 881 | Normalized for parenthetical application contexts |
| **Rows with Alternate Names** | 256 | Alternate names delimited by semicolon (`;`) |
| **Rows without Alternate Names** | 670 | Explicitly marked as `"No Alternate Names"` |

### Column Schema
1. `Ingredient_Name` (string): Primary chemical, INCI, or botanical name.
2. `Primary_Function` (string, 71 unique values): Formulatory role (e.g., *Emollient, Botanical Extract, Colorant, Surfactant*).
3. `Ingredient_Category` (string, 72 unique values): Material class (e.g., *Botanical Extract, Polymer, Surfactant, Mineral, Preservative*).
4. `Product_Categories` (string, 519 unique combinations): Multi-label comma-separated target products (e.g., *"Cream, Lotion"*, *"Shampoo, Body Wash"*).
5. `Origin` (string, 6 unique values): Sourcing (*Synthetic, Natural, Naturally Derived, Mineral, Biotechnology Derived, Animal Derived*).
6. `Safety_Level` (string, target): Ingredient safety classification.
7. `Allergy_Risk` (string, target): Allergenic sensitization potential.
8. `Irritation_Risk` (string, target): Topical irritation potential.
9. `Regulatory_Status` (string, 3 unique values): Legal compliance status (*Approved, Restricted, Country Restricted*).
10. `Packaging Names / Alternate Names` (string): Common consumer or packaging aliases.

---

## 4. Target Definitions & Class Distributions

Each of the three targets contains exactly four discrete classes with notable class imbalance:

```text
Safety_Level:
  Safe             651 (70.30%)
  Very Safe        118 (12.74%)
  Moderate Risk    112 (12.10%)
  High Risk         45  (4.86%)
  Total: 926

Allergy_Risk:
  Low              662 (71.49%)
  Medium           143 (15.44%)
  No Risk           80  (8.64%)
  High              41  (4.43%)
  Total: 926

Irritation_Risk:
  No Risk          446 (48.16%)
  Low              297 (32.07%)
  Medium           156 (16.85%)
  High              27  (2.92%)
  Total: 926
```

Minority classes (`High Risk` in Safety at 4.86%, `High` in Allergy at 4.43%, and `High` in Irritation at 2.92%) represent critical safety concerns where false-negative classifications could cause consumer harm.

---

## 5. Historical Experiment Discovery

A deep codebase inspection across commit history, repository files, documentation, and notebooks revealed:
- **No historical personal care model files or experiment scripts existed in the repository.**
- Historical commits (`phase9A_complete_codebase_audit.md`) explicitly noted:
  > *"Feature Representation: None. Model Architecture: None. Validation Methodology: None. Final Selected Model: NOT FOUND."*
- The historical metrics reported in the prompt ($\text{Safety: } 0.7410, \text{ Allergy: } 0.7432, \text{ Irritation: } 0.7210, \text{ Mean: } 0.7351$) originate from preliminary offline explorations conducted prior to or outside the active repository.

---

## 6. Historical TF-IDF Configuration

Because no configuration files or code artifacts for Personal Care TF-IDF were present in the repository, the **historical TF-IDF configuration could not be reproduced exactly from available repository evidence.**

To adhere strictly to reproducibility standards without silently inventing parameters, Phase 10A locks the established repository-wide character n-gram standard:
```python
TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    sublinear_tf=True,
    min_df=1
)
```
This is explicitly designated as the **Phase 10A Reproducibility Baseline Configuration**.

---

## 7. Historical Structured Feature Configuration

No historical encoder definitions existed for the structured fields. Phase 10A establishes a deterministic, leakage-safe pipeline:
- `Primary_Function`, `Ingredient_Category`, `Origin`, `Regulatory_Status` &rarr; `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`.
- `Product_Categories` &rarr; `CountVectorizer(tokenizer=lambda x: [s.strip().lower() for s in x.split(',')], token_pattern=None, binary=True)` (extracts individual product types such as *shampoo, serum, cream, sunscreen* into distinct binary features).
- Combined with Name TF-IDF via `ColumnTransformer`.

---

## 8. Historical Canonical Grouping

Historically, no canonical grouping script existed for Personal Care in the repository (Food alone had a grouping script in `dataset.py`).

Phase 10A designs an authoritative grouping methodology:
- Primary ingredient names frequently incorporate parenthetical application-context suffixes (e.g. `(Baby Care)`, `(Foot Care)`, `(Oral Care)`, `(Lip Plumper)`, `(Aftershave)`, `(Hair Spray)`, `(Hair Wax)`, `(Makeup)`, `(Sanitary Products)`, `(Deodorant)`).
- Normalizing these suffixes links 45 variant representations back to their base chemical ingredient (e.g., `Menthol (Lip Plumper)`, `Menthol (Aftershave)`, `Menthol (Foot Care)` map to canonical group `Menthol`).
- Core biological identities (e.g., `Aqua (Water)`, `Cocos Nucifera Oil (Coconut Oil)`) retain their complete compound identity.
- This mapping groups the 926 rows into **881 unique canonical groups**.

### Investigation of 881 vs. 882 Canonical Groups
- **Expected Preliminary Count:** 882
- **Actual Verified Count:** 881
- **Exact Grouping Relationship Responsible for Difference:**
  During preliminary exploration and initial planning (`scratch/test_canonical_grouping.py`), the candidate application-context suffix list included `"deodorant grade"`, but omitted the standalone suffix `"deodorant"`. As a result, row 549 `Sodium Bicarbonate (Deodorant)` was not normalized and formed its own separate group from `Sodium Bicarbonate`:
  - `Sodium Bicarbonate`: 1 row
  - `Sodium Bicarbonate (Deodorant)`: 1 row
  This generated 882 preliminary groups.
  When implementing the finalized deterministic grouping module in `backend/ml/preprocessing/personal_care_dataset.py`, `"deodorant"` was explicitly added to `APPLICATION_CONTEXT_SUFFIXES` (alongside `"deodorant grade"`). This merged row 549 `Sodium Bicarbonate (Deodorant)` into the base canonical group `Sodium Bicarbonate`:
  - `Sodium Bicarbonate`: 2 rows (`['Sodium Bicarbonate', 'Sodium Bicarbonate (Deodorant)']`)
- **Why 881 is the Correct Count under Implemented Methodology:**
  Unifying `Sodium Bicarbonate (Deodorant)` with `Sodium Bicarbonate` is necessary to prevent cross-fold canonical data leakage ($\text{Train} \cap \text{Val} = \emptyset$) between formulation/application variants and their parent chemical ingredient. The implementation is verified by unit tests ([`tests/test_personal_care_validation.py`](file:///C:/Users/velzyaa/Desktop/PicWise/tests/test_personal_care_validation.py)) to have zero leakage and exactly 881 disjoint canonical groups.

---

## 9. Historical Validation Method

The exact historical validation protocol (whether standard $K$-fold, random train/test split, or group-aware split) was not documented in repository files.

Phase 10A finalizes the strict standard:
```python
StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```
Grouping strictly on `canonical_group_id`.

---

## 10. Reproducibility Result

The reproducibility comparison verdict is: **Approximately Reproduced (Independent Pipeline)**.

| Metric | Historical Benchmark | Phase 10A Baseline (`C=1.0, cw=None`) | Phase 10A Optimized (`C=10.0, cw=balanced`) |
| :--- | :---: | :---: | :---: |
| **Safety Macro F1** | 0.7410 | **0.7611** (+0.0201) | 0.7549 (+0.0139) |
| **Allergy Macro F1** | 0.7432 | 0.6997 (-0.0435) | 0.7159 (-0.0273) |
| **Irritation Macro F1** | 0.7210 | 0.6486 (-0.0724) | 0.7110 (-0.0100) |
| **Mean Macro F1** | **0.7351** | **0.7031** (-0.0320) | **0.7273** (-0.0078) |

### Explanation of Variance
1. **Strict Group-Leakage Elimination:** By enforcing disjoint canonical groups across folds, the model cannot memorize text n-grams between application variants of the same compound (e.g., `Menthol` vs. `Menthol (Foot Care)`), slightly reducing overly optimistic validation scores.
2. **True Out-of-Fold Cross Validation:** Every single observation among all 926 rows was evaluated out-of-fold across 5 folds.
3. **Severe Minority Imbalance:** High Allergy (4.4%) and High Irritation (2.9%) suffered under unweighted baseline training; controlled optimization recovered the Mean Macro F1 to **0.7273**, within 0.78% of the historical benchmark.

---

## 11. Final Canonical Grouping & Leakage Audit

Across all 5 folds and for all three classification targets, the group leakage audit confirmed:
$$\text{Train Groups} \cap \text{Validation Groups} = \emptyset$$
- Group leakage detected: **False** across all 15 fold evaluations (5 folds $\times$ 3 targets).
- Target leakage audit: Target columns (`Safety_Level`, `Allergy_Risk`, `Irritation_Risk`) and post-outcome data were strictly excluded from input feature pipelines.
- Fold-local fitting: Vectorizers and encoders were fit exclusively on training fold indices `train_idx`.

---

## 12. Feature Representation Dimensions

For each training fold:
- `Ingredient_Name` character n-grams $(2, 5)$ yield $\sim 4,100$ features.
- Categorical OneHot features yield $\sim 150$ sparse binary features.
- Product Category multi-label features yield $\sim 65$ binary features.
- Total feature space: $\sim 4,315$ dimensions per sample.

---

## 13. Baseline Evaluation Results

Evaluated with `C=1.0`, `class_weight=None`, `solver="lbfgs"`, `max_iter=1000`, `random_state=42`:

### Aggregate Metrics
- **Safety_Level:** Macro F1 = **0.7611** | Weighted F1 = **0.8485** | Accuracy = **0.8585** | Balanced Acc = **0.7231**
- **Allergy_Risk:** Macro F1 = **0.6997** | Weighted F1 = **0.8346** | Accuracy = **0.8456** | Balanced Acc = **0.6406**
- **Irritation_Risk:** Macro F1 = **0.6486** | Weighted F1 = **0.7751** | Accuracy = **0.7840** | Balanced Acc = **0.6188**
- **Mean Macro F1:** **0.7031**

### Baseline Per-Class Detailed Performance

#### Safety Level
```text
               precision    recall  f1-score   support
    High Risk     0.8000    0.8000    0.8000        45
Moderate Risk     0.8415    0.6161    0.7113       112
         Safe     0.8726    0.9677    0.9177       651
    Very Safe     0.7792    0.5085    0.6154       118
```

#### Allergy Risk
```text
               precision    recall  f1-score   support
         High     0.7143    0.4878    0.5797        41
          Low     0.8657    0.9637    0.9121       662
       Medium     0.7387    0.5734    0.6457       143
      No Risk     0.8600    0.5375    0.6615        80
```

#### Irritation Risk
```text
               precision    recall  f1-score   support
         High     0.7143    0.1852    0.2941        27
          Low     0.7228    0.6936    0.7079       297
       Medium     0.7737    0.6795    0.7235       156
      No Risk     0.8164    0.9170    0.8638       446
```

---

## 14. Controlled Logistic Regression Optimization

Grid Search across all 12 hyperparameter configurations ($C \in \{0.1, 0.5, 1.0, 2.0, 5.0, 10.0\}$, $\text{class\_weight} \in \{\text{None}, \text{"balanced"}\}$):

| Configuration | C | Class Weight | Safety Macro F1 | Allergy Macro F1 | Irritation Macro F1 | **Mean Macro F1** | Safety High Recall | Allergy High Recall | Irritation High Recall | Dangerous Confusions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C=10.0_cw=balanced`** | **10.0** | **balanced** | **0.7549** | **0.7159** | **0.7110** | **0.7273** | **0.8444** | **0.6341** | **0.5556** | **17** |
| `C=10.0_cw=None` | 10.0 | None | 0.7625 | 0.7257 | 0.6937 | 0.7273 | 0.8444 | 0.5854 | 0.3704 | 22 |
| `C=1.0_cw=balanced` | 1.0 | balanced | 0.7654 | 0.7063 | 0.7048 | 0.7255 | 0.8444 | 0.6585 | 0.5556 | 22 |
| `C=5.0_cw=None` | 5.0 | None | 0.7642 | 0.7175 | 0.6931 | 0.7249 | 0.8444 | 0.5366 | 0.3704 | 24 |
| `C=5.0_cw=balanced` | 5.0 | balanced | 0.7553 | 0.7140 | 0.7052 | 0.7248 | 0.8444 | 0.6341 | 0.5185 | 19 |
| `C=2.0_cw=balanced` | 2.0 | balanced | 0.7561 | 0.7121 | 0.7001 | 0.7228 | 0.8444 | 0.6341 | 0.5185 | 20 |
| `C=2.0_cw=None` | 2.0 | None | 0.7651 | 0.7107 | 0.6918 | 0.7225 | 0.8222 | 0.5366 | 0.3333 | 22 |
| `C=0.5_cw=balanced` | 0.5 | balanced | 0.7656 | 0.6912 | 0.6971 | 0.7180 | 0.8444 | 0.6585 | 0.5556 | 23 |
| `C=1.0_cw=None` (Base) | 1.0 | None | 0.7611 | 0.6997 | 0.6486 | 0.7031 | 0.8000 | 0.4878 | 0.1852 | 27 |
| `C=0.1_cw=balanced` | 0.1 | balanced | 0.7526 | 0.6295 | 0.6636 | 0.6819 | 0.8444 | 0.6341 | 0.5185 | 31 |
| `C=0.5_cw=None` | 0.5 | None | 0.7348 | 0.6907 | 0.6087 | 0.6781 | 0.7556 | 0.4634 | 0.1111 | 29 |
| `C=0.1_cw=None` | 0.1 | None | 0.6728 | 0.5984 | 0.5360 | 0.6024 | 0.6667 | 0.3171 | 0.0000 | 37 |

---

## 15. Class Imbalance Analysis & Model Selection

### Class Imbalance Impact
Unweighted Logistic Regression (`class_weight=None`) exhibits severe recall degradation on the minority risk classes in this dataset:
- High Irritation recall at $C=1.0$: **18.52%** (only 5 of 27 identified).
- High Irritation recall at $C=10.0$: **37.04%** (10 of 27 identified).
- High Allergy recall at $C=1.0$: **48.78%** (20 of 41 identified).

Enabling `class_weight="balanced"`:
- High Irritation recall increases to **55.56%** (+37.04 percentage points over baseline).
- High Allergy recall increases to **63.41%** (+14.63 percentage points over baseline).
- High Safety recall reaches **84.44%**.
- Dangerous under-prediction confusions decrease from 27 (baseline) and 22 ($C=10.0$ unweighted) down to **17** (lowest in the entire grid).

*Dataset-Specific Empirical Finding:* In this specific Personal Care dataset, inverse-frequency balanced class weighting substantially improves minority High-risk sensitivity across all three risk targets and reduces severe under-prediction errors without degrading overall Mean Macro F1.

### Formalized Deterministic Model Selection Rule
1. **Primary Criterion:** Maximize Mean Macro F1 across the three targets.
2. **Secondary Criterion (Tie-Breaker):** If Mean Macro F1 is tied, minimize total dangerous confusion count (severe under-predictions of high/moderate risks as safe/no risk).
3. **Tertiary Criterion (Tie-Breaker):** If still tied, prefer higher aggregate minority High-risk recall ($\text{Recall}_{\text{Safety High}} + \text{Recall}_{\text{Allergy High}} + \text{Recall}_{\text{Irritation High}}$).

### Selection Application
Comparing top contenders:
- `C=10.0, class_weight=None`: Mean Macro F1 = **0.7273**, Dangerous Confusions = **22**, Aggregate High Recall = **1.8002**
- `C=10.0, class_weight="balanced"`: Mean Macro F1 = **0.7273**, Dangerous Confusions = **17**, Aggregate High Recall = **2.0341**

Under Rule 2, `class_weight="balanced"` cleanly resolves the tie by achieving 5 fewer dangerous confusions (17 vs 22) and superior minority recall across targets (e.g., 55.56% vs 37.04% on High Irritation).

**Final Selected Configuration:**
```python
LogisticRegression(
    C=10.0,
    class_weight="balanced",
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)
```

---

## 16. Detailed Error & Confidence Analysis

### 1. Dangerous Confusions Identified (Best Model)
- **Safety Level:** 0 High Risk predicted as Safe or Very Safe. Exactly 1 Moderate Risk predicted as Very Safe:
  - `Menthol (Lip Plumper)`: True Moderate Risk, predicted Very Safe (confidence: 0.81). *Cause:* Other Menthol entries are Safe; strong general emollient/cooling weights pushed the prediction.
- **Allergy Risk:** 7 High Allergy predicted as Low:
  - `Lanolin`, `Lanolin (Hair Pomade)`, `Lanolin Alcohol (Wool Alcohols)`: Lanolin derivatives are categorized as `Wax` / `Animal Derived`, which generally have low allergy rates in cosmetics.
  - `Thimerosal`, `Colophonium (Rosin)`, `p-tert-Butylphenol Formaldehyde Resin`.
- **Irritation Risk:** 5 High Irritation predicted as Low/No Risk:
  - `Sodium Lauryl Sulfate`, `Ammonium Thioglycolate`, `Glyceryl Thioglycolate`.

### 2. Error Clustering by Ingredient Family
- **Safety Errors:** Concentrated in *Polymers* (16 errors) and *Minerals* (11 errors).
- **Allergy Errors:** Concentrated in *Botanical Extracts* (19 errors) and *Preservatives* (15 errors).
- **Irritation Errors:** Concentrated in *Surfactants* (21 errors) and *Botanical Extracts* (21 errors).

### 3. Prediction Confidence Analysis
- **Correct Predictions:** Average confidence = **0.93** across all targets. High-confidence ($\ge 0.80$) correct predictions exceed $75\%$ of the dataset.
- **Incorrect Predictions:** Average confidence = **0.77** &ndash; **0.81**.
- **Low-Confidence Predictions ($< 0.50$):** Extremely rare ($< 2\%$ of dataset), indicating decisive probability boundaries.

---

## 17. Automated Test Suite & Regression Verification

### Personal Care Test Suite ([`tests/test_personal_care_validation.py`](file:///C:/Users/velzyaa/Desktop/PicWise/tests/test_personal_care_validation.py))
10 targeted unit and integration tests passed:
- `test_dataset_shape_and_columns`: Confirmed 926 rows, 10 columns, valid schema.
- `test_dataset_zero_nulls`: Confirmed 0 null values.
- `test_target_class_domains`: Confirmed discrete 4-class domains for Safety, Allergy, and Irritation.
- `test_canonical_group_context_suffix_normalization`: Confirmed context stripping.
- `test_canonical_group_identity_preservation`: Confirmed core biological name preservation.
- `test_canonical_group_count`: Confirmed exactly 881 groups.
- `test_zero_group_leakage_across_all_folds`: Confirmed 0 overlap across all 5 folds.
- `test_feature_pipeline_excludes_targets`: Confirmed targets never enter feature space.
- `test_model_probabilities_and_oof_completeness`: Confirmed 926 OOF predictions, probabilities sum to 1.0.
- `test_reproducibility_deterministic_outputs`: Confirmed identical bitwise predictions on re-runs.

**Result:** `10 passed in 4.40s (OK)`.

### Full Existing Test Suite Regression
Executed the full required regression suite across all Food analysis subsystems:
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_food_status_mapping.py tests/test_food_backend_hardening.py tests/test_food_analysis_pipeline.py tests/test_food_frontend_integration.py
```
- `tests/test_food_status_mapping.py`
- `tests/test_food_backend_hardening.py`
- `tests/test_food_analysis_pipeline.py`
- `tests/test_food_frontend_integration.py`

**Result:** `Ran 57 tests in 376.796s — OK`.
Zero errors, zero failures, zero regressions across the frozen Food pipeline.

---

## 18. Reproducibility Verification

Running the entire cross-validation suite twice with `random_state=42`:
- **Out-of-Fold Predictions:** 100% identical.
- **Out-of-Fold Probabilities:** Max absolute difference $< 10^{-7}$.
- **Metric Tables:** Bitwise reproducible.

---

## 19. Recommendation for Phase 10B

1. **Model Ready for Feature Freeze:** The `LogisticRegression(C=10.0, class_weight="balanced")` architecture achieves strong, reliable performance (Mean Macro F1 = 0.7273), robust minority recall, and zero group leakage.
2. **Phase 10B Scope:**
   - Formalize final model training and freeze production model artifacts in `backend/ml/models/personal_care/`.
   - Implement inference service (`backend/ml/inference/personal_care_service.py`).
   - Define status presentation mapping for Personal Care (Safety, Allergy, Irritation).
   - Integrate with the unified API.

---
