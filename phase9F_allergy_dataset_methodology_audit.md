# Phase 9F — Allergy Detection Dataset & Methodology Final Specification

**Project:** PicWise  
**Phase:** 9F (Allergy Detection Dataset & Methodology Final Specification — Ground-Truth Correction)  
**Date:** September 18, 2026  
**Status:** METHODOLOGY FROZEN & LOCKED FOR PHASE 9G  
**Commit Status:** NOT COMMITTED  
**Push Status:** NOT PUSHED  

---

## 1. Executive Summary

This document establishes the authoritative, frozen dataset and methodology specification for the PicWise Allergy Detection engine produced during Phase 9F. Following direct inspection of the authoritative food ingredient CSV, this specification corrects prior assumptions regarding label missingness and permanently locks the dataset, target classes, representation expansion, grouping structure, validation design, safety metrics, and candidate models for Phase 9G experimentation.

### Authoritative Specification Summary
* **Authoritative Source Dataset**: `data/food/food_ingredients_dataset_corrected(2)(1).csv` (and repository copy under `data/food/`).
* **Domain Scope**: Strictly Food ingredients only. Personal Care data is excluded.
* **Target Classes**: Exactly four source labels: `No Risk`, `Low`, `Medium`, `High`. There is NO literal `"None"` class in the source dataset.
* **Label Completeness**: Exactly **0 missing, 0 NaN, and 0 blank** values in `Allergy Risk`. All 498 canonical food ingredients are fully and explicitly labelled. All prior NaN-imputation hypotheses are permanently removed.
* **Reconciled Representation Structure**: Exactly 498 canonical ingredients expanding to 951 alternate packaging names for a total of **1,449 supervised representations**, mapping to 1,435 unique lowercase text strings and partitioned into **488 connected components / leakage groups**.
* **Zero Leakage Validation**: 5-Fold `StratifiedGroupKFold` on `component_id` with **strictly 0 group overlap across all 5 folds** and verified minority class representation (`High`: 10–11 per val fold; `Medium`: 37–38 per val fold).
* **Predefined Safety Targets**: High Recall ($\ge 90\%$) and Severe False Negative Rate ($\le 5\%$) locked as primary model selection criteria over raw accuracy.
* **Candidate Experiment Matrix**: 5 candidate feature representations (Char TF-IDF, Word TF-IDF, Word+Char, Frozen MiniLM, Dual-Stream) across 4 balanced classifier families (Logistic Regression, LinearSVC, Random Forest, XGBoost).
* **Production Integrity**: 0 production code modified; 0 models trained; full regression suite remains **121/121 passing**.

---

## 2. Authoritative Dataset

### 2.1 Authoritative Dataset Path
The authoritative dataset for the PicWise Food Allergy Detection model is locked to:
```text
data/food/food_ingredients_dataset_corrected(2)(1).csv
```
*(or the repository copy of this exact dataset under `data/food/`).*

### 2.2 Food-Only Scope Locked
The Allergy Detection engine is strictly scoped to **food ingredients**.
* **Personal Care Data Exclusion**: Personal Care data (`personal_care_ingredients_dataset_cleaned.xlsx`) must NOT be included in Allergy Detection model training.
* **Clinical Rationale**: Food allergy represents Type I IgE-mediated immediate hypersensitivity triggered by ingested proteins with fatal anaphylaxis risk. Personal care allergy represents Type IV delayed contact dermatitis from topical chemical haptens.
* **Cross-Domain Contradiction**: Pooling personal care ingredients injects severe contradictions (e.g., `Tocopherol` is Medium Risk in personal care contact dermatitis, but No Risk in food ingestion). Personal Care allergy semantics are outside the scope of this model.

---

## 3. Dataset Structure

The authoritative dataset contains exactly **498 rows and 8 columns** with zero missing fields in key columns:

| Column Name | Data Type | Missing Count | Missing % | Semantic Role | Sample Values |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `Ingredient Name` | `string` | 0 | 0.00% | Primary canonical ingredient | `"Almonds"`, `"Soybean"`, `"Citric Acid"` |
| `Category` | `string` | 0 | 0.00% | Taxonomic food category | `"Nuts & Seeds"`, `"Dairy Ingredients"` |
| `Safety Level` | `string` | 0 | 0.00% | Food Safety risk label | `"Very Safe"`, `"Safe"`, `"Moderate Risk"` |
| `Allergy Risk` | `string` | **0** | **0.00%** | **Target Allergy Label** | `"No Risk"`, `"Low"`, `"Medium"`, `"High"` |
| `Health Impact` | `string` | 0 | 0.00% | Health impact description | `"Positive"`, `"Neutral"`, `"Negative"` |
| `Processing Level` | `string` | 0 | 0.00% | Industrial processing level | `"Unprocessed"`, `"Processed"` |
| `Regulatory Status` | `string` | 0 | 0.00% | Legal regulatory status | `"Approved"`, `"Approved with Limits"` |
| `Packaging Names / Alternate Names` | `string` | 0 | 0.00% | Synonyms, transliterations, INS codes | `"badam"`, `"ins 330"`, `"soya lecithin"` |

### Orthogonality of Allergy Risk and Safety Level
Allergy Risk is completely decoupled from Food Safety Level:
* 100% of canonical `High` Allergy Risk ingredients (*Almonds, Peanuts, Cashews, Walnuts, Egg White, Fish Gelatin, Wheat Gluten, Sesame*) have Safety Level = `"Safe"`.
* Chemical preservatives with toxicological concerns (Safety Level = `"High Risk"`) have Allergy Risk = `"No Risk"`.
* Food Safety and Allergy Risk remain strictly decoupled models.

---

## 4. Allergy Risk Target

### 4.1 Target Column: `Allergy Risk`
The machine learning target is `Allergy Risk`. The source labels are:
```text
Y = { No Risk, Low, Medium, High }
```

### 4.2 Target Constraints
* **Source Labels Locked**: These exact four strings (`No Risk`, `Low`, `Medium`, `High`) must remain exactly as they appear in the source dataset.
* **No Label Renaming**: Do NOT attempt to convert `No Risk` $\to$ `None` inside the source dataset or training pipeline.
* **No Binary Collapsing**: Do not collapse the problem into binary classification (`Allergenic` vs. `Non-Allergenic`).
* **No Extra Classes**: Do not introduce `None`, `Very High`, `Critical`, or any fifth class.
* **Ordinal Interpretation**: Treat the four classes as ordered risk categories for evaluation, threshold selection, and downstream interpretation:
$$\text{No Risk} < \text{Low} < \text{Medium} < \text{High}$$

---

## 5. Label Completeness

### 5.1 Removal of Prior NaN Assumptions
Previous preliminary audit drafts operated under the incorrect assumption that the source dataset contained 360 `NaN` values requiring an evidence-based imputation methodology. Direct inspection of the authoritative CSV disproves this assumption:

* **Missing `Allergy Risk` values:** Exactly **0 / 498 (0.00%)**
* **`NaN` `Allergy Risk` values:** Exactly **0 / 498 (0.00%)**
* **Blank `Allergy Risk` values:** Exactly **0 / 498 (0.00%)**
* **Literal `"None"` values:** Exactly **0 / 498 (0.00%)**

### 5.2 Source Dataset Ground-Truth Facts
* **No Label Imputation Required**: The authoritative dataset contains explicit, curated Allergy Risk labels for all 498 canonical ingredients.
* **No NaN-to-Class Conversion**: No synthetic heuristic, imputation model, or default rule is executed.
* **Semolina (Suji/Rava)**: Inspected directly in the authoritative CSV. Its actual source value is **`No Risk`**. It is NOT modified or inferred from other wheat entries.
* **Priority Produce and Grains**: `Celery`, `Celery Seeds`, `Mustard Oil`, `Mustard Seeds`, `Chestnut Flour`, `Oats`, and `Rolled Oats` all possess explicit, curated labels (**`No Risk`**) in the source dataset.
* **Preservatives & Additives**: All chemical additives, minerals, and vitamins possess explicit, curated labels (**`No Risk`**) in the source dataset.
* **Governance Conclusion**: The previous NaN governance section (categorizing rows into strongly supported, weakly supported, unresolved, or contradictory) is completely superseded by the verified ground truth of the authoritative dataset.

---

## 6. Final Class Distribution

The class distribution across all 498 canonical ingredients and throughout the expanded supervised population is completely defined:

### 6.1 Distribution Across Canonical, Group, and Representation Levels

| Allergy Risk Class | Canonical Count (498) | Canonical % | Group Count (488) | Group % | Supervised Representations (1,449) | Supervised % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`No Risk`** | **360** | **72.29%** | **354** | **72.54%** | **1,037** | **71.57%** |
| **`Low`** | **60** | **12.05%** | **59** | **12.09%** | **170** | **11.73%** |
| **`Medium`** | **58** | **11.65%** | **56** | **11.48%** | **189** | **13.04%** |
| **`High`** | **20** | **4.02%** | **19** | **3.89%** | **53** | **3.66%** |
| **Missing / NaN** | **0** | **0.00%** | **0** | **0.00%** | **0** | **0.00%** |
| **Total** | **498** | **100.00%** | **488** | **100.00%** | **1,449** | **100.00%** |

The four class counts sum exactly to 498 canonical ingredients, 488 connected components, and 1,449 supervised representations.

---

## 7. Canonical / Alias Representation

### 7.1 Reconciled Dataset Hierarchy
The dataset representation structure reconciles four distinct entity levels:

* **Canonical Ingredients (498)**: Authoritative unique rows in the source CSV.
* **Alternate / Packaging Representations (951)**: Synonyms, commercial trade names, multilingual terms, and INS/E-numbers parsed from `Packaging Names / Alternate Names` using `;` delimiter.
* **Total Supervised Representations (1,449)**: Total expanded training instances ($498 \text{ canonical} + 951 \text{ alternate} = 1,449$).
* **Unique Lowercase Text Strings (1,435)**: Deduplicated surface strings across all 1,449 representations. Exactly 14 packaging aliases are shared across canonical ingredients ($1,449 - 14 = 1,435$).
* **Connected Components / Leakage Groups (488)**: Maximal connected components formed by alias-sharing edges ($498 - 10 = 488$).

### 7.2 Representation Expansion Governance
* **No Independent Observations**: Alternate packaging aliases are NOT independent ground-truth clinical observations.
* **Label Inheritance**: Every alias inherits the exact `Allergy Risk` label of its canonical parent entity.
* **Group Invariance**: Every alias belongs to exactly one connected component (`component_id`) and is never separated from its canonical entity across validation folds.

---

## 8. Connected Components

### 8.1 Component Derivation
When canonical ingredients and their packaging aliases are modeled as a bipartite graph, shared aliases create edges between canonical ingredients. Exactly **10 pairs of canonical ingredients share packaging aliases**, merging 20 canonical ingredients into 10 connected components:
$$498 \text{ canonical ingredients} - 10 \text{ merged pairs} = 488 \text{ connected components}$$

### 8.2 The 10 Merged Canonical Pairs (Verified Against Authoritative Dataset)

1. `Carboxymethyl Cellulose` $\leftrightarrow$ `Sodium Carboxymethyl Cellulose` (Both **`No Risk`**)
2. `Carrageenan` $\leftrightarrow$ `Kappa Carrageenan` (Both **`Low`**)
3. `Casein` $\leftrightarrow$ `Casein Protein` (Both **`Medium`**)
4. `Clarified Gellan Gum` $\leftrightarrow$ `Gellan Gum` (Both **`No Risk`**)
5. `Dry Ginger` $\leftrightarrow$ `Dry Ginger Powder (Sonth)` (Both **`No Risk`**)
6. `Glycerol Monostearate` $\leftrightarrow$ `Mono- and Diglycerides of Fatty Acids` (Both **`No Risk`**)
7. `Hydrolyzed Whey Protein` $\leftrightarrow$ `Whey Protein Hydrolysate` (Both **`Medium`**)
8. `Low Methoxyl Pectin` $\leftrightarrow$ `Pectin` (Both **`No Risk`**)
9. `Sorbitol (Humectant)` $\leftrightarrow$ `Sorbitol (Sweetener)` (Both **`No Risk`**)
10. `Vital Wheat Gluten` $\leftrightarrow$ `Wheat Gluten Protein` (Both **`High`**)

> [!NOTE]
> In all 10 merged pairs, the source `Allergy Risk` label is **100% identical**. Not a single pair introduces an internal label conflict.

---

## 9. Leakage Prevention

To guarantee strict generalization in Phase 9G, eight leakage prevention rules are locked:

1. **Group-Aware Splitting**: All train/validation splits must partition by `component_id`.
2. **Strict Component Isolation**: No connected component or alias cluster may cross fold boundaries.
3. **Fold-Strict Vectorizer Fitting**: TF-IDF vectorizers must be fitted **exclusively on the training fold** of each split.
4. **Zero Vocabulary Leakage**: Validation fold text must never be seen during vocabulary construction, document frequency calculation, or n-gram table generation.
5. **Fold-Contained Hyperparameter Tuning**: All model selection and tuning decisions must occur within training partitions.
6. **No Global Preprocessing**: No dataset-wide text transformation that learns from validation distributions is permitted.
7. **Deterministic Embeddings**: Pretrained MiniLM embeddings may be computed deterministically offline, but feature scaling and classifier training remain strictly fold-specific.
8. **No Test Tuning**: Zero parameter tuning against frozen evaluation sets.

> [!CAUTION]
> **Dataset-wide TF-IDF fitting is strictly prohibited.** Fitting vectorizers across all 1,449 rows prior to splitting invalidates the experiment.

---

## 10. Five-Fold Validation Design

### 10.1 Validation Configuration
* **Split Algorithm**: `sklearn.model_selection.StratifiedGroupKFold`
* **Folds ($K$)**: 5
* **Shuffle**: `True`
* **Random State**: `42`
* **Grouping Variable**: `component_id` (488 groups)
* **Stratification Target**: `Allergy Risk`

### 10.2 Empirical 5-Fold Split Verification

| Fold | Train Examples | Val Examples | Train Groups | Val Groups | Group Overlap | Val Class Breakdown (`No Risk` / `Medium` / `Low` / `High`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fold 1** | 1,159 | 290 | 390 | 98 | **0** | `No Risk`: 208, `Medium`: 38, `Low`: 34, `High`: 10 |
| **Fold 2** | 1,159 | 290 | 391 | 97 | **0** | `No Risk`: 208, `Medium`: 38, `Low`: 34, `High`: 10 |
| **Fold 3** | 1,160 | 289 | 391 | 97 | **0** | `No Risk`: 207, `Medium`: 37, `Low`: 34, `High`: 11 |
| **Fold 4** | 1,159 | 290 | 390 | 98 | **0** | `No Risk`: 207, `Medium`: 38, `Low`: 34, `High`: 11 |
| **Fold 5** | 1,159 | 290 | 390 | 98 | **0** | `No Risk`: 207, `Medium`: 38, `Low`: 34, `High`: 11 |

* **Group Overlap**: **Strictly 0 across all 5 folds** (verified).
* **Minority Class Representation**: Every validation fold contains 10–11 `High` instances and 37–38 `Medium` instances.

---

## 11. Safety-Critical Metrics

In allergy detection, misclassifying a dangerous allergen as safe is life-threatening. The following safety-critical metrics are locked:

### 11.1 High Recall ($\text{Recall}_{\text{High}}$) — Primary Safety Metric
$$\text{Recall}_{\text{High}} = \frac{TP_{\text{High}}}{TP_{\text{High}} + FN_{\text{High}}}$$

### 11.2 Severe False Negative Rate ($\text{SFNR}_{\text{High}}$) — Fatal Error Metric
A prediction is a **severe false negative** when:
$$\text{True Class} = \text{High} \quad \text{AND} \quad \hat{y} \in \{\text{No Risk}, \text{Low}\}$$
$$\text{SFNR}_{\text{High}} = \frac{\sum_{i=1}^N \mathbb{I}(y_i = \text{High} \land \hat{y}_i \in \{\text{No Risk}, \text{Low}\})}{N_{\text{High}}}$$
*Note: A true `High` predicted as `Medium` is an under-prediction, but is **NOT** severe because the consumer is still alerted to moderate allergen risk.*

### 11.3 Medium Recall ($\text{Recall}_{\text{Medium}}$)
$$\text{Recall}_{\text{Medium}} = \frac{TP_{\text{Medium}}}{TP_{\text{Medium}} + FN_{\text{Medium}}}$$

### 11.4 Medium False Negative Rate ($\text{MFNR}_{\text{Medium}}$)
$$\text{MFNR}_{\text{Medium}} = \frac{\sum_{i=1}^N \mathbb{I}(y_i = \text{Medium} \land \hat{y}_i = \text{No Risk})}{N_{\text{Medium}}}$$

### 11.5 Macro F1 ($\text{Macro } F_1$)
$$\text{Macro } F_1 = \frac{1}{4} \left( F_{1,\text{No Risk}} + F_{1,\text{Low}} + F_{1,\text{Medium}} + F_{1,\text{High}} \right)$$

### 11.6 Supplementary Calibration Metrics
* **Balanced Accuracy**: Macro-averaged recall across all 4 classes.
* **Confusion Matrix**: Full $4 \times 4$ out-of-fold contingency table.
* **Multiclass Brier Score**: Evaluates probability calibration for downstream risk gating.

---

## 12. Safety Targets

The following predefined safety targets are established as benchmarks for candidate model selection:

* **High Recall ($\text{Recall}_{\text{High}}$)**: $\ge 90.0\%$
* **Severe False Negative Rate ($\text{SFNR}_{\text{High}}$)**: $\le 5.0\%$
* **Medium Recall ($\text{Recall}_{\text{Medium}}$)**: $\ge 75.0\%$
* **Medium False Negative Rate ($\text{MFNR}_{\text{Medium}}$)**: $\le 10.0\%$
* **Macro F1 ($\text{Macro } F_1$)**: $\ge 0.70$

> [!IMPORTANT]
> **These are PREDEFINED TARGETS, NOT guaranteed outcomes.**  
> Neither models nor ground-truth labels may be altered simply to satisfy these thresholds. Actual performance in Phase 9G must be measured empirically under strict cross-validation.

---

## 13. Candidate Feature Representations

The candidate feature representations frozen for Phase 9G experimentation are:

### Representation A: Character TF-IDF
* **Settings**: `analyzer='char'`, `ngram_range=(3, 5)`, `sublinear_tf=True`
* **Purpose**: Sub-word morphological matching; robust to OCR typos, hyphenation, and packaging variations.
* *Note: `min_df` is NOT frozen as a methodology constant and may be evaluated as an experimental hyperparameter in Phase 9G.*

### Representation B: Word TF-IDF
* **Settings**: `analyzer='word'`, `ngram_range=(1, 2)`, `sublinear_tf=True`
* **Purpose**: Captures exact whole-word ingredient terminology and bi-grams.

### Representation C: Word + Character TF-IDF
* **Settings**: Feature union of Word `(1, 2)` and Character `(3, 5)` TF-IDF.
* **Purpose**: Hybrid lexical representation capturing both full tokens and morphological stems.

### Representation D: Frozen MiniLM Embeddings
* **Settings**: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense vectors ($L_2$ normalized).
* **Purpose**: Captures semantic similarity and culinary synonyms without lexical overlap.

### Representation E: Character TF-IDF + MiniLM (Dual-Stream)
* **Settings**: Column concatenation of Character TF-IDF features and 384-dimensional MiniLM embeddings.
* **Purpose**: Multi-modal fusion of character-level morphology and dense semantic context.

> [!NOTE]
> Transformer weights remain frozen as a static feature extractor. Fine-tuning transformer weights is not permitted in Phase 9G unless explicitly approved later.

---

## 14. Candidate Models

The candidate classifier family is frozen to four model types, all evaluated with balanced class weighting or cost-sensitive sample weighting:

1. **Balanced Logistic Regression**: Multinomial logistic regression with $\ell_2$ regularization and `class_weight='balanced'`. Fast, interpretable, well-calibrated baseline.
2. **Balanced Linear Support Vector Classifier (LinearSVC)**: Linear margin maximizer with `class_weight='balanced'`, calibrated via Platt scaling / isotonic regression.
3. **Balanced Random Forest**: Ensemble of 200–500 trees with `class_weight='balanced_subsample'` to handle non-linear feature interactions.
4. **Balanced XGBoost Classifier**: Gradient boosted trees with multi-class softmax loss and cost-sensitive sample weighting ($w_i \propto \frac{1}{N_{c_i}}$).

**Evaluation Rule:** Phase 9G must systematically evaluate representation $\times$ classifier combinations. No winning model is selected prior to experimentation.

---

## 15. Phase 9G Experimental Protocol

All Phase 9G experiments must execute under the following protocol:

```text
Fold Partitioning (StratifiedGroupKFold, 5 Folds, component_id)
        ↓
For each fold (1..5):
    1. Split train/val by component_id
    2. Fit vectorizers/scalers strictly on training fold
    3. Transform validation fold using training-fitted preprocessing
    4. Train classifier with cost-sensitive weights
    5. Generate validation predictions and probabilities
    6. Store out-of-fold predictions
        ↓
Aggregate Out-of-Fold (OOF) Predictions across all 5 folds
        ↓
Compute Safety & Classification Metrics:
    - Accuracy, Balanced Accuracy
    - Macro F1, Weighted F1
    - High Recall, High Precision, High F1
    - Severe False Negative Rate (SFNR)
    - Medium Recall, Medium False Negative Rate (MFNR)
    - Confusion Matrix
    - Multiclass Brier Score (where probabilities exist)
        ↓
Evaluate Against Predefined Safety Targets
```

**Selection Principle:** Primary model selection is driven by **safety performance** (High Recall and Severe FNR), not raw overall accuracy.

---

## 16. Historical Baseline

The historical Phase 2 Allergy model is documented strictly as a **non-production reference point**:

* **Historical Architecture**: XGBoost with unstratified random splitting and global vectorizer fitting.
* **Historical Performance**:
  - Accuracy $\approx 67.97\%$
  - High Allergy Recall $\approx 10.0\%$
  - Severe False Negative Rate $\approx 80.0\%$ (8 out of 10 High allergens classified as No Risk or Low)
* **Status**: **Explicitly rejected.** The historical model is NOT being reused as the production model. Phase 9G begins with a clean, leak-free experimental framework.

---

## 17. PicWise UI Mapping

For the PicWise prototype UI, the four source dataset labels map to consumer-facing badges and advisory copy as follows:

| Source Target Class | PicWise UI Badge Label | Advisory Copy / Clinical Interpretation |
| :--- | :--- | :--- |
| **`No Risk`** | **Allergen-Free** | No common allergenic proteins detected based on the PicWise ingredient knowledge base. |
| **`Low`** | **Low Allergy Risk** | Mild dietary sensitivity or intolerance trigger; non-anaphylactic for most consumers. |
| **`Medium`** | **Moderate Allergy Risk** | Contains recognized secondary allergens (e.g., dairy, soy, mustard, sulfites); exercise caution. |
| **`High`** | **High Allergy Risk** | **Critical Allergen Alert**: Contains major priority allergen (e.g., peanuts, tree nuts, eggs, fish, gluten, sesame). |

### UI Interpretation Guardrails
* **`No Risk` is the Dataset Ground Truth**: The machine learning model is trained on, and predicts, `No Risk`.
* **Prototype UI Interpretation Only**: `"Allergen-Free"` is strictly the PicWise prototype UI badge.
* **No Regulatory Claim**: Do **NOT** claim that PicWise provides a legally certified allergen-free guarantee. Phrases such as `"certified allergen-free"` or `"medically guaranteed allergen-free"` are prohibited.

---

## 18. Production Integration Status

* **Current Status**: Phase 9F is strictly a methodology and data preparation phase.
* **Production Model**: **No production Allergy model exists yet.**
* **Model Artifacts**: Zero model artifacts have been created or saved.
* **Code Modification**: `backend/services/food_analysis_service/` remains completely unmodified. The Allergy Detection component in the Unified Food Analysis pipeline remains in its verified `Allergy Status: Unavailable` state.
* **API Contracts**: All API endpoints and response schemas established in Phase 9E remain pristine and unchanged.

---

## 19. Phase 9G Entry Criteria

The following checklist confirms that all entry criteria for Phase 9G model experimentation are satisfied:

```text
[PASS] Authoritative food-only dataset locked (498 canonical rows, 0 missing)
[PASS] Four-class target locked (No Risk, Low, Medium, High)
[PASS] Label completeness verified (0 NaN, 0 missing, 0 blank)
[PASS] Canonical/alias relationships resolved (498 canonicals + 951 alternates = 1,449 representations)
[PASS] Connected-component groups defined (488 groups; 10 merged canonical pairs)
[PASS] Zero group leakage verified (0 group overlap across all 5 folds)
[PASS] Five-fold validation verified (StratifiedGroupKFold, shuffle=True, random_state=42)
[PASS] Safety metrics defined (Recall_High, SFNR_High, Recall_Med, MFNR_Med, Macro F1)
[PASS] Safety targets predefined (High Recall >= 90%, Severe FNR <= 5%)
[PASS] Feature representations frozen (Representations A through E)
[PASS] Candidate models frozen (Logistic Regression, LinearSVC, Random Forest, XGBoost)
[PASS] Leak-free preprocessing protocol frozen (strictly fold-specific fitting)
[PASS] Production code unchanged (backend/services/food_analysis_service/ untouched)
[PASS] No model trained during Phase 9F (0 models trained; 0 artifacts created)
```

**Phase 9G may begin model experimentation.**

---

## 20. Final Methodology Freeze

The dataset scope, target classes (`No Risk`, `Low`, `Medium`, `High`), representation mappings, connected components (488), validation folds, safety metrics, and candidate models documented herein are **permanently frozen**. No further modifications to the methodology or exploratory audits are required. 

**Phase 9F is corrected using the authoritative dataset and is ready for final freeze.**
