# PHASE 9A — PICWISE COMPLETE CODEBASE & INTEGRATION AUDIT REPORT

**Date:** September 14, 2026  
**Auditor:** Antigravity AI Engineering Team  
**Scope:** Strict Read-Only Audit of Codebase, Datasets, ML Pipelines, OCR Implementation, Frontend/Backend Architecture, and Git State  
**Target Repository:** `PicWise` (`https://github.com/raiamisha25/PicWise.git`)  
**External OCR Repository:** `final-ocr` (`https://github.com/kirtikushwaha-123/final-ocr.git`)  

---

## 1. Executive Summary

PicWise is transitioning from the machine learning experimentation and feasibility phase into production integration. The objective of PicWise is to provide consumers with an intuitive, trustworthy mobile/web tool that analyzes product label images across two distinct domains—**Food** and **Personal Care**—and renders an actionable assessment categorized into **Green (Safe)**, **Yellow (Moderate / Caution)**, or **Red (High Risk / Avoid)**.

This Phase 9A audit was conducted under strict read-only constraints to assess the architectural alignment, codebase health, data assets, and integration readiness of all system components.

### Key Audit Findings

1. **Intended vs. Actual Routing Architecture:**
   * **Intended:** The user explicitly selects the product category (`Food` vs. `Personal Care`) *before* uploading an image. This user decision is the sole routing mechanism. No automated product classification model is permitted.
   * **Actual State in Codebase:** The current frontend (`templates/upload.html`, `static/app.js`) does **not** allow the user to select a category. The backend route (`/api/analyze`) does not accept a category parameter. Instead, `backend/services/analysis_service/analyzer.py` attempts to **heuristically infer** the domain by counting matched ingredients (`_detect_domain()`), directly contradicting the core architectural principle.

2. **OCR Subsystem Status:**
   * In the main PicWise repository, OCR is represented only by a minimal 7-line placeholder (`backend/services/ocr_service/stub.py`) that returns hardcoded text.
   * However, the external `final-ocr` repository (`https://github.com/kirtikushwaha-123/final-ocr`) contains a mature, production-grade OCR pipeline utilizing **PaddleOCR, OpenCV, document layout analysis, region reconciliation, multi-variant enhancement, and parsed nutrition/ingredient extractors**.
   * While `final-ocr` is highly performant and tested, its interface produces structured dictionaries (bounding boxes, polygons, nutrient value-unit pairs), whereas the PicWise backend currently expects a monolithic single string.

3. **Food Safety Model Status:**
   * The Food Safety experimentation phase is **complete and frozen**: **Character TF-IDF (3–5 n-grams) + Frozen MiniLM Embeddings (`sentence-transformers/all-MiniLM-L6-v2`) + Balanced Logistic Regression** (Macro F1: 0.7437, High-Risk Recall: 73.91%, Severe Error Rate: 2.01%).
   * However, the physical model serialized in the active repository (`backend/ml/models/safety_model.joblib`) is the outdated Phase 2 **XGBoost** model. The frozen MiniLM + Balanced Logistic Regression model has **not yet been serialized or integrated** into the production service path.

4. **Personal Care Model Status:**
   * `Final Personal Care model specification: NOT FOUND / INCOMPLETE`
   * Personal care ingredients were historically pooled with food ingredients to train a shared Phase 2 XGBoost model. There is **no independent Personal Care model architecture, hyperparameters, validation split, or standalone model artifact** in the repository.
   * In the git-tracked dataset, `High Risk` personal care items are critically sparse ($N=1$). A separate untracked dataset exists on Desktop with 45 High Risk examples, but has not been integrated.

5. **Nutrition / Health Scoring Status:**
   * The repository contains a 47-entry lookup table (`data/nutrition/nutrition_knowledge_dataset.csv`) providing qualitative nutritional properties (`Health Role`, `Health Impact`, `Decision Priority`, `Better Direction`).
   * There is **no quantitative scoring formula or calculation engine** (e.g., Nutri-Score, NOVA score, or nutrient threshold algorithm) implemented in Python.

6. **Product Analysis Layer & Green/Yellow/Red UI:**
   * A dedicated **Product Analysis Layer** that aggregates ingredient safety predictions, nutrition scores, and allergen findings into an overall product-level result **does not exist**.
   * The Green/Yellow/Red status pills exist only as static HTML/CSS mockups on `templates/home.html`. The live results screen (`upload.html`) renders unranked, raw key-value cards without any overall product rating or color badge.

7. **Git & Repository Structure Risk:**
   * The local git repository root was initialized directly at `C:\Users\velzyaa\Desktop` rather than in an isolated project subdirectory. As a result, personal desktop files, shortcuts, credentials, and unrelated projects appear as untracked files in git.

---

## 2. Intended PicWise Architecture

The intended PicWise architecture strictly enforces **user-directed pipeline routing**. The user chooses whether they are analyzing a food product or a personal care item.

```text
                 USER
                   │
                   ▼
          Select Product Type
            ┌──────┴──────┐
            │             │
          FOOD      PERSONAL CARE
            │             │
            └──────┬──────┘
                   │
                   ▼
              Image Upload
                   │
                   ▼
              Existing OCR (PaddleOCR + Layout Engine)
                   │
                   ▼
             Extracted Data
            ┌──────┴──────┐
            │             │
          FOOD      PERSONAL CARE
            │             │
       ┌────┼────┐        │
       ▼    ▼    ▼        ▼
    Safety Nutrition Allergy  Personal
     Model   Score    Detection Care Model
       │      │       │        │
       └──────┴───────┘        │
               │               │
               ▼               ▼
          Food Product    Personal Care
            Analysis        Analysis
               │               │
               └───────┬───────┘
                       ▼
                 PicWise Result
                       │
                 ┌─────┼─────┐
                 ▼     ▼     ▼
               GREEN YELLOW RED
```

### Architectural Axioms
1. **Zero Automated Product Classification:** There is no model inferring whether a product is Food or Personal Care. Product packaging can be ambiguous (e.g., edible oils vs. hair oils). The user's explicit selection determines which pipeline executes.
2. **Domain Separation:** Food and Personal Care safety profiles, risks, regulatory standards, and feature spaces are distinct. Food models evaluate ingestion hazards and nutritional profiles; Personal Care models evaluate topical toxicity, skin sensitization, and irritation.
3. **Component Pipeline Independence:** Within each domain, analysis sub-modules execute independently before funneling into the Product Analysis Layer.
4. **Hierarchical Aggregation:** Ingredient-level ML predictions, nutritional evaluations, and allergen alerts must be aggregated through an explicit decision policy into an overall product-level classification (**Green / Yellow / Red**).

---

## 3. Current Project Structure

The project currently spans two separate repositories/directories:

### A. PicWise Application Repository (`C:\Users\velzyaa\Desktop`)

```text
PicWise/
├── .env.example
├── .gitignore
├── app.py                                  # Flask application entry point
├── requirements.txt                        # Application Python dependencies
├── README.md
├── index.html                              # Standalone static dashboard mockup
│
├── backend/
│   ├── __init__.py                         # Flask app factory (create_app)
│   ├── routes/
│   │   ├── __init__.py
│   │   └── api.py                          # POST /api/analyze endpoint
│   ├── services/
│   │   ├── __init__.py
│   │   ├── knowledge_base.py               # In-memory KB loader & search index
│   │   ├── analysis_service/
│   │   │   ├── __init__.py
│   │   │   └── analyzer.py                 # Core analysis orchestration
│   │   ├── ingredient_matching/
│   │   │   ├── __init__.py
│   │   │   └── matcher.py                  # Multi-stage deterministic ingredient matcher
│   │   ├── nutrition_service/
│   │   │   ├── __init__.py
│   │   │   └── lookup.py                   # Nutrition knowledge base lookup
│   │   └── ocr_service/
│   │       ├── __init__.py
│   │       └── stub.py                     # Hardcoded 7-line OCR dummy
│   └── ml/
│       ├── __init__.py
│       ├── models/
│       │   ├── vectorizer.joblib           # Fitted character n-gram (2,5) TF-IDF (Phase 2)
│       │   ├── safety_model.joblib         # XGBoost safety classifier (Phase 2)
│       │   ├── safety_label_encoder.joblib # Safety LabelEncoder
│       │   ├── allergy_model.joblib        # XGBoost allergy classifier (Phase 2)
│       │   └── allergy_label_encoder.joblib# Allergy LabelEncoder
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predictor.py                # Standalone predict_ingredient_risk()
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── dataset.py                  # Unified dataset loader & group split
│       │   ├── audit_dataset.py            # Dataset auditing script
│       │   └── conflicting_alt_names.csv
│       ├── training/
│       │   ├── __init__.py
│       │   └── train.py                    # Training script for Phase 2 XGBoost models
│       └── evaluation/
│           ├── __init__.py
│           ├── evaluate.py                 # Phase 3 Part 1 baseline evaluator
│           ├── error_analysis.py           # Phase 3 Part 2A error analysis runner
│           ├── error_analysis.json / .md
│           ├── improvement_experiments.py  # Phase 3 Part 2B experiment runner
│           └── improvement_experiments.json / .md
│
├── data/
│   ├── food/
│   │   ├── food_ingredients_dataset_final.csv
│   │   ├── ingredient_knowledge_base_500_cleaned.csv
│   │   └── ingredient_knowledge_base_500_with_alternate_names.csv
│   ├── nutrition/
│   │   └── nutrition_knowledge_dataset.csv
│   └── personal_care/
│       ├── personal_care_ingredients_dataset_cleaned.xlsx
│       └── personal_care_ingredients_dataset_csv.xlsx
│
├── static/
│   ├── app.js                              # Frontend client-side controller
│   ├── styles.css                          # Application styling
│   └── uploads/.gitkeep
│
├── templates/
│   ├── base.html                           # Base layout template
│   ├── home.html                           # Dashboard page (with mockup pills)
│   ├── login.html                          # Login placeholder
│   └── upload.html                         # Scan/Upload page
│
└── tests/
    ├── fixtures/test_product.jpg
    ├── test_analysis.py                    # Pipeline & API endpoint test
    ├── test_knowledge_base.py              # KnowledgeBase loader tests
    ├── test_matcher.py                     # Deterministic matcher tests
    ├── test_nutrition.py                   # Nutrition lookup tests
    ├── test_ml_evaluation.py               # Evaluation verification tests
    ├── test_ml_error_analysis.py           # Error analysis tests
    └── test_ml_improvement_experiments.py # Phase 3 Part 2B experiments test
```

### B. External OCR Repository (`final-ocr`)

Located in scratch directory: `C:\Users\velzyaa\.gemini\antigravity\scratch\final-ocr`

```text
final-ocr/
├── config.py                               # Central hyperparameters, thresholds & vocabularies
├── main.py                                 # End-to-end OCR & layout analysis pipeline runner
├── ground_truth.json                       # Benchmark ground truth annotations
├── requirements.txt                        # PaddleOCR, OpenCV, RapidFuzz dependencies
├── detection/
│   ├── packet_region.py                    # OpenCV contour-based packet isolation
│   ├── ocr_detector.py                     # Full-image PaddleOCR wrapper
│   ├── document_layout.py                  # Unified Layout Analysis (lines, columns, blocks)
│   ├── line_builder.py                     # Spatial line reconstruction & gap grouping
│   ├── line_classifier.py                  # Multi-class semantic line scoring
│   ├── block_detector.py                   # Column-constrained visual block clustering
│   ├── ingredient_region.py                # Ingredient section detector (anchor + layout)
│   ├── nutrition_region.py                 # Nutrition table detector (grid / row clustering)
│   ├── region_reconciliation.py            # Overlap resolution & line ownership assignment
│   ├── section_signals.py                  # Stop-word & signal keyword regexes
│   ├── tight_roi.py                        # Minimal bounding polygon computation
│   └── geometry.py                         # Bounding box & IoU utilities
├── ocr/
│   ├── paddle_engine.py                    # Lazy-loaded PaddleOCR singleton
│   ├── ensemble.py                         # Multi-variant image preprocessing ensemble
│   └── detector_interface.py               # OCR abstraction interface
├── preprocessing/
│   ├── image_utils.py                      # Load, resize, normalize, safe crop
│   ├── enhancement.py                      # CLAHE, thresholding, sharpening variants
│   ├── deskew.py                           # Hough line / minAreaRect deskewing
│   └── perspective.py                      # 4-point quadrilateral perspective warp
├── parsing/
│   ├── ingredient_parser.py                # Comma/delimiter splitting, INS/E stripping
│   └── nutrition_parser.py                 # Multi-column nutrient table parser
├── nlp/
│   └── ingredient_corrector.py             # Fuzzy KB correction layer
├── matching/
│   └── knowledge_base.py                   # RapidFuzz-backed knowledge base indexer
└── visualization/
    ├── draw_regions.py                     # Final detection bounding box visualizer
    └── draw_layout_debug.py                # Detailed line/column layout visualizer
```

---

## 4. User Category Selection Audit

### Audit Questions & Findings

| Audit Question | Finding | Source / Evidence |
| :--- | :--- | :--- |
| **Where is the category selection stored?** | **Nowhere.** | There is no variable, cookie, session storage, or form input for product category in `upload.html` or `app.js`. |
| **How is it passed to the backend?** | **It is NOT passed.** | `static/app.js` builds `FormData` containing only `formData.append("image", selectedFile);`. |
| **Does the backend receive category?** | **No.** | `backend/routes/api.py` checks only `request.files.get("image")`. |
| **Does category-specific routing exist?** | **No.** | `backend/services/analysis_service/analyzer.py` invokes a single path: `analyze_product_image()`. |
| **How does the system currently determine domain?** | **Heuristic inference post-OCR.** | `analyzer.py` (lines 25–34): `_detect_domain()` counts whether more food or personal care ingredients were matched in the text. |

### Source Inspection: `analyzer.py` (Domain Inference Anti-Pattern)
```python
def _detect_domain(ingredients):
    matched_items = [item for item in ingredients if item.get("matched")]
    food_count = sum(1 for item in matched_items if item["domain"] == "food")
    personal_care_count = sum(1 for item in matched_items if item["domain"] == "personal_care")

    if food_count > personal_care_count:
        return "food"
    if personal_care_count > food_count:
        return "personal_care"
    return "unknown"
```

### Architectural Verdict
This is a **major architectural gap**. The system currently attempts automated domain inference based on ingredient counts, which:
1. Fails when ingredients are shared across domains (e.g., Glycerin, Water, Citric Acid, Xanthan Gum).
2. Violates the fundamental design rule: **The user's selection is the routing decision.**

---

## 5. OCR Audit

### PicWise Repository State
* Located in `backend/services/ocr_service/stub.py`:
```python
def extract_text(image_bytes):
    """Stubbed OCR boundary. Replace this function with real OCR/vision later."""
    if not image_bytes:
        return ""
    return "Ingredients: whole wheat flour, sugar, salt. Nutrition: energy, protein."
```
* Status: **100% Mock / Placeholder**.

### External `final-ocr` Repository Audit
The `final-ocr` repository represents the accepted OCR implementation. It has been built and tested on real-world packaging images.

#### Pipeline Architecture & Execution Flow
1. **Image Ingestion & Quality Assessment (`preprocessing/image_utils.py`):**
   * Computes blur score (Laplacian variance), contrast (standard deviation of gray values), and average brightness.
   * Dynamically resizes images: minimum height 1000px (upscaled via bicubic interpolation), maximum dimension capped at 2800px.
2. **Packet Isolation (`detection/packet_region.py`):**
   * Uses OpenCV Canny edge detection, dilation, and contour hierarchy to find product boundaries.
   * If candidate contour area $\ge 15\%$ of image and confidence $\ge 0.55$, crops to packet; otherwise gracefully retains normalized full image.
3. **Full-Image OCR (`detection/ocr_detector.py`):**
   * Executes PaddleOCR (`en`, angle classification enabled).
   * Generates bounding polygons, oriented rectangles, center points, dimensions, and normalized text strings.
4. **Unified Document Layout Analysis (`detection/document_layout.py`):**
   * **Line Reconstruction (`line_builder.py`):** Merges horizontally adjacent, vertically aligned text fragments using adaptive vertical tolerance ($0.45 \times \text{median line height}$) and horizontal gap limits ($3.5 \times \text{median line height}$).
   * **Column Clustering:** Identifies distinct column bands separated by white-space gutters ($\ge 8\%$ image width).
   * **Block Detection (`block_detector.py`):** Groups lines into coherent reading blocks.
   * **Line Semantic Classification (`line_classifier.py`):** Computes 11 semantic scores per line: `ingredient`, `nutrition`, `allergen`, `instruction`, `manufacturer`, `contact`, `storage`, `mrp`, `legal`, `marketing`, `other`.
5. **Region Detection & Reconciliation (`detection/ingredient_region.py`, `nutrition_region.py`, `region_reconciliation.py`):**
   * Identifies candidate starting anchors (e.g., "Ingredients:", "Nutrition Facts").
   * Expands vertically and horizontally along column boundaries while rejecting stop-words and competing section lines.
   * Resolves overlaps via IoU thresholding ($0.15$) and reassigns contested lines to the highest-scoring section owner.
   * Generates tight polygon ROIs (`tight_roi.py`).
6. **Region Re-OCR & Multi-Variant Ensemble (`ocr/ensemble.py`, `preprocessing/enhancement.py`):**
   * Crops detected regions, deskews using Hough transform / minimum area bounding box, and corrects perspective warp.
   * Produces multiple image variants: grayscale, CLAHE, adaptive threshold, bilateral filter, unsharp mask.
   * Re-runs OCR across variants and selects the best text output using scoring based on vocabulary density and OCR confidence.
7. **Structured Text Parsing:**
   * **Ingredients (`parsing/ingredient_parser.py`):** Strips leading headers ("Ingredients:"), removes parenthetical additive codes ("INS 330", "E 330"), strips percentage qualifiers ("65%"), splits on commas/semicolons without breaking multi-word chemical names ("Sodium Benzoate").
   * **Nutrition (`parsing/nutrition_parser.py`):** Parses multi-column tables, identifies headers ("per 100g", "per serving"), maps nutrient rows to 14 canonical keys (`energy`, `protein`, `total_carbohydrate`, `total_sugars`, `total_fat`, `saturated_fat`, `trans_fat`, `dietary_fibre`, `sodium`, `salt`, `cholesterol`, `calcium`, `iron`, `vitamin`), and extracts numeric values and units.
8. **Fuzzy KB Correction (`nlp/ingredient_corrector.py`, `matching/knowledge_base.py`):**
   * Performs fuzzy matching of extracted ingredient tokens against knowledge base vocabularies using RapidFuzz `WRatio` (acceptance threshold $\ge 85\%$).

---

## 6. Food Safety Model Audit

### Frozen Final Model Specification
The Food Safety model experimentation phase is complete and frozen.

| Parameter | Specification |
| :--- | :--- |
| **Pipeline Architecture** | Dual-stream: Character TF-IDF + Pretrained MiniLM Embeddings + Balanced Logistic Regression |
| **Character TF-IDF** | `TfidfVectorizer(analyzer="char", ngram_range=(3,5), sublinear_tf=True, min_df=1)` |
| **Semantic Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Embedding Properties** | Frozen pretrained, 384 dimensions, L2 normalized |
| **Classifier** | `LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)` |
| **Target Classes (4)** | `Very Safe`, `Safe`, `Moderate Risk`, `High Risk` |
| **Decision Policy** | Argmax on predicted class probabilities (or cost-sensitive safety margin) |

### Frozen Benchmark Performance Reference
* **Macro F1:** $0.7437$
* **False Safe Rate (FSR):** $32.98\%$
* **High-Risk Recall:** $73.91\%$
* **High-Risk Precision:** $80.95\%$
* **High-Risk F1:** $0.7727$
* **Overall Accuracy:** $76.31\%$
* **Balanced Accuracy:** $73.63\%$
* **ROC-AUC:** $0.9050$
* **Ordinal MAE:** $0.2651$
* **Severe Error Rate:** $2.01\%$

### Active Repository Audit (Discrepancy Analysis)
1. **Model in `backend/ml/models/safety_model.joblib`:**
   * Contains an `xgboost.sklearn.XGBClassifier` instance trained during Phase 2.
   * Features: Character n-grams $(2, 5)$ only.
   * Dataset: Trained on **pooled** food and personal care data.
   * Performance: Macro F1: $0.4964$, High Risk Recall: $28.57\%$, Moderate Risk Recall: $22.39\%$, Dangerous false-safe confusions: $62$.
2. **Inference Script (`backend/ml/inference/predictor.py`):**
   * Hardcoded to load the old Phase 2 XGBoost model and character-only vectorizer.
   * Completely lacks code to load `sentence-transformers` or concatenate 384-d semantic embeddings.
   * Not imported or called by any active backend service.

### Integration Verdict
The frozen Food Safety model represents an outstanding improvement (+0.2473 Macro F1, +45.34% High Risk Recall over baseline). However, it exists **only as an experimental specification**. The actual production serialization, inference wrapper, and service integration remain to be created in the integration phase.

---

## 7. Personal Care Model Audit

### Status Declaration
```text
Final Personal Care model specification: NOT FOUND / INCOMPLETE
```

### Detailed Component Audit

| Attribute | Audit Finding |
| :--- | :--- |
| **Exact Dataset** | In repository: `data/personal_care/personal_care_ingredients_dataset_cleaned.xlsx` (and `..._csv.xlsx`).<br>Untracked on Desktop: `C:\Users\velzyaa\Desktop\PicWise DataSet\final personal care dataset.csv`. |
| **Dataset Size** | Tracked: **786 rows**, 9 columns.<br>Untracked: **926 rows**, 10 columns. |
| **Columns (Tracked)** | `Ingredient_Name`, `Primary_Function`, `Ingredient_Category`, `Product_Categories`, `Origin`, `Safety_Level`, `Allergy_Risk`, `Irritation_Risk`, `Regulatory_Status`. |
| **Target Variables** | Intended: `Safety_Level` (and optionally `Allergy_Risk`, `Irritation_Risk`). |
| **Class Distribution (Tracked)** | `Safe`: 633 (80.5%), `Moderate Risk`: 96 (12.2%), `Very Safe`: 56 (7.1%), `High Risk`: **1 (0.13%)**. |
| **Class Distribution (Untracked)** | `Safe`: 651 (70.3%), `Very Safe`: 118 (12.7%), `Moderate Risk`: 112 (12.1%), `High Risk`: **45 (4.9%)**. |
| **Preprocessing** | Basic string trimming and categorical normalization in `dataset.py`. `Irritation_Risk` has 446 missing/null values (56.7%). |
| **Feature Representation** | **None.** No dedicated vectorizer, embeddings, or chemical features defined for Personal Care. |
| **Model Architecture** | **None.** No dedicated Personal Care model architecture exists. In Phase 2, personal care items were merely appended to food data to train a generic model. |
| **Hyperparameters** | **None.** |
| **Validation Methodology** | **None.** No separate group-aware split or cross-validation exists for personal care. |
| **Final Selected Model** | **NOT FOUND.** |
| **Inference Implementation** | **None.** |
| **Model Artifacts** | **None.** |
| **Production Readiness** | **0% (Not Production Ready).** |

### Missing Information Required Before Implementation
1. Selection of authoritative Personal Care dataset (tracked 786-row vs. untracked 926-row with 45 High Risk examples).
2. Definition of target variables (Safety Level only, or multi-target including Irritation Risk).
3. Handling of missing `Irritation_Risk` entries (56.7% null).
4. Feature representation methodology (Character TF-IDF, domain-specific embeddings, or INCI chemical name handling).
5. Evaluation benchmark and model selection experiments specifically on the Personal Care distribution.

---

## 8. Nutrition / Health Audit

### Knowledge Dataset
* **File:** `data/nutrition/nutrition_knowledge_dataset.csv`
* **Size:** 47 rows, 6 columns.
* **Columns:** `Nutrient`, `Health Role`, `Health Impact`, `Decision Priority`, `Better Direction`, `Alternative / Packaging Names`.
* **Example Row:**
  * `Nutrient`: "Saturated Fat"
  * `Health Role`: "Risk Factor"
  * `Health Impact`: "Negative in Excess"
  * `Decision Priority`: "Very High"
  * `Better Direction`: "Lower is Better"
  * `Alternative / Packaging Names`: "saturated fat; sat fat; sat. fat; saturated fatty acids; SFA; saturates"

### Methodology & Implementation
* Implemented in `backend/services/nutrition_service/lookup.py`.
* Functions:
  * `lookup_nutrient_term(term, knowledge_base)`: Look up an individual nutrient name.
  * `find_relevant_nutrition(extracted_text, knowledge_base)`: Scans extracted text for mentions of canonical or alternate nutrient names.
* **Nature of Output:** Qualitative lookup only.
```python
{
    "nutrient": "Saturated Fat",
    "canonicalNutrient": "Saturated Fat",
    "originalInput": "saturated fat",
    "matched": True,
    "matchType": "exact_canonical",
    "confidence": 1.0,
    "healthRole": "Risk Factor",
    "role": "Risk Factor",
    "healthImpact": "Negative in Excess",
    "decisionPriority": "Very High",
    "betterDirection": "Lower is Better"
}
```

### Critical Gaps in Nutrition / Health
1. **No Quantitative Scoring Engine:** There is no formula or algorithm that computes an overall nutritional health score (e.g., 0–100, Nutri-Score A–E, or High/Medium/Low rating) from nutritional values.
2. **Interface Incompatibility with OCR:**
   * `final-ocr` extracts **numeric values, units, and column frames**:
     `"total_sugars": {"value": 24.5, "unit": "g", "per_100g": {"value": 24.5, "unit": "g"}}`
   * `lookup.py` accepts **only raw string text** and matches nutrient names using string inclusion (`_mentions_key`). It completely discards numeric values, serving sizes, and units.

---

## 9. Allergy Component Audit

### Component Breakdown

| Component Layer | Current Implementation | Source File | Status |
| :--- | :--- | :--- | :--- |
| **ML Allergy Model** | XGBoost 4-class classifier (`None`, `Low`, `Medium`, `High`). Evaluated in Phase 3 Part 2B. Baseline achieved only **10.00% High allergy recall** ($1/10$) due to extreme class sparsity. | `backend/ml/models/allergy_model.joblib` | Dormant. Not imported by backend services. High recall bottlenecked by training data scarcity. |
| **KB Ingredient Allergy** | Deterministic lookup table returning the `Allergy Risk` column from the food/personal care datasets. | `backend/services/ingredient_matching/matcher.py` | Operational for exact matches in the 500-ingredient KB. |
| **OCR Section Filtering** | Detects allergen advice statements ("Contains: Milk, Soy", "May contain tree nuts") to isolate them from ingredient paragraphs. | `final-ocr/detection/section_signals.py`, `line_classifier.py` | Active in OCR layout analysis. |
| **User Profile Matching** | Hardcoded visual mockup of user allergens (Peanuts, Tree Nuts, Milk, Shellfish, Soy). | `templates/home.html` | UI mockup only. No user allergen state or matching logic in backend. |

### Integration Verdict
There is currently **no unified Allergy Detection Engine**. An end-to-end engine must:
1. Extract explicit packaging allergen statements ("Contains: Milk") from OCR.
2. Cross-reference individual ingredients against known allergen databases or ML predictions.
3. Compare findings against the user's personal allergen profile.

---

## 10. OCR → Analysis Interface

### Current Interface vs. Production Requirement

```text
CURRENT PICWISE FLOW:
[Image File] ──► api.py ──► analyzer.py ──► stub.py ("Ingredients: flour, sugar...") ──► regex split ──► KB lookup

REQUIRED INTEGRATION FLOW:
[Image File + Category] ──► api.py ──► Unified Controller
                                             │
                                             ▼
                                  final-ocr Engine (PaddleOCR)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             Structured Ingredients                     Structured Nutrition
        [ {ocr_text, matched_name, ...} ]          {energy: {val, unit}, sugars: ...}
                       │                                           │
                       ▼                                           ▼
             Category ML Pipeline                       Nutrition Scoring Engine
          (Food Safety / Personal Care)                 (Thresholds & RDA analysis)
```

### Detailed Schema Comparison

#### 1. Ingredients Output
* **`final-ocr` output schema (`nlp/ingredient_corrector.py`):**
```json
[
  {
    "ocr_text": "citric acid (ins 330)",
    "normalized_text": "citric acid",
    "corrected_ingredient": "Citric Acid",
    "match_confidence": 0.95,
    "matched_name": "Citric Acid",
    "confidence": 0.95
  }
]
```
* **Current PicWise backend expectation (`matcher.py`):**
  Expects a raw string containing `"ingredients: ..."` and executes `re.split(r"[,.;\n]", cleaned)`. It completely ignores OCR bounding boxes, pre-cleaned tokens, and OCR confidence scores.

#### 2. Nutrition Output
* **`final-ocr` output schema (`parsing/nutrition_parser.py`):**
```json
{
  "energy": { "value": 450.0, "unit": "kcal", "per_100g": { "value": 450.0, "unit": "kcal" } },
  "protein": { "value": 7.5, "unit": "g", "per_100g": { "value": 7.5, "unit": "g" } },
  "total_carbohydrate": { "value": 68.0, "unit": "g", "per_100g": { "value": 68.0, "unit": "g" } },
  "total_sugars": { "value": 24.0, "unit": "g", "per_100g": { "value": 24.0, "unit": "g" } },
  "total_fat": { "value": 16.0, "unit": "g", "per_100g": { "value": 16.0, "unit": "g" } },
  "saturated_fat": { "value": 8.0, "unit": "g", "per_100g": { "value": 8.0, "unit": "g" } },
  "sodium": { "value": 380.0, "unit": "mg", "per_100g": { "value": 380.0, "unit": "mg" } }
}
```
* **Current PicWise backend expectation (`lookup.py`):**
  Expects a raw string containing `"nutrition: ..."` and looks for substring occurrences of "energy", "protein", etc. It has no structure to accept or store numerical values.

---

## 11. Food Pipeline Audit

Tracing the required Food pipeline:

$$\text{User Selects FOOD} \longrightarrow \text{Image Upload} \longrightarrow \text{OCR} \longrightarrow \text{Ingredients} \longrightarrow \text{Food Safety ML} \longrightarrow \text{Nutrition Score} \longrightarrow \text{Allergy} \longrightarrow \text{Product Analysis} \longrightarrow \text{PicWise UI}$$

| Stage | Intended Role | Current Repository Status | Gap / Issue |
| :--- | :--- | :--- | :--- |
| **Category Selection** | User chooses "Food" | **Missing** | UI has no selector. |
| **Image Upload** | Multipart upload | **Implemented** | `upload.html` / `api.py` works for JPG/PNG/WEBP. |
| **OCR** | PaddleOCR + Layout Analysis | **Ready in `final-ocr`**; Stubbed in main repo | Must integrate `final-ocr` codebase into `backend/services/ocr_service/`. |
| **Ingredient Extraction** | Token parsing & normalization | **Ready in `final-ocr`** | Generates clean tokens; needs routing to Food models. |
| **Food Safety Model** | Predict ingredient risk | **Frozen spec ready**; Artifacts missing | Need to serialize frozen MiniLM + LogisticRegression and invoke on extracted tokens. |
| **Nutrition / Health** | Quantitative health score | **Incomplete** | Only qualitative name lookup exists. Numeric scoring calculation missing. |
| **Allergy** | Ingestion allergy detection | **Incomplete** | KB lookup exists; user profile matching missing; ML model has low minority recall. |
| **Food Product Analysis** | Aggregate components | **Missing** | No aggregation layer exists. |
| **PicWise UI** | Render Green/Yellow/Red | **Missing** | Upload results screen displays unranked plain text list. |

---

## 12. Personal Care Pipeline Audit

Tracing the required Personal Care pipeline:

$$\text{User Selects PERSONAL CARE} \longrightarrow \text{Image Upload} \longrightarrow \text{OCR} \longrightarrow \text{Ingredients} \longrightarrow \text{Personal Care Model} \longrightarrow \text{Product Analysis} \longrightarrow \text{PicWise UI}$$

| Stage | Intended Role | Current Repository Status | Gap / Issue |
| :--- | :--- | :--- | :--- |
| **Category Selection** | User chooses "Personal Care" | **Missing** | UI has no selector. |
| **Image Upload** | Multipart upload | **Implemented** | Functional in `upload.html`. |
| **OCR** | PaddleOCR + Layout Analysis | **Ready in `final-ocr`** | `final-ocr` supports `--domain personal_care` (skips nutrition region). |
| **Text Extraction** | Ingredient list parsing | **Ready in `final-ocr`** | Handles cosmetic ingredient delimiters and INCI naming. |
| **Personal Care Model** | Predict cosmetic ingredient risks | **Missing / Incomplete** | No standalone model, architecture, or artifacts exist. |
| **Domain Analysis** | Function, irritation, regulation | **Partial** | KB lookup returns `Primary_Function`, `Irritation_Risk`, `Origin`. |
| **PC Product Analysis** | Aggregate ingredients | **Missing** | No aggregation layer exists. |
| **PicWise UI** | Render Green/Yellow/Red | **Missing** | No domain-tailored color status card. |

---

## 13. Product Analysis Layer Audit

### Core Architectural Purpose
The Product Analysis Layer sits between raw component outputs and the UI. It translates raw multi-ingredient predictions, nutritional facts, and allergen warnings into a cohesive, consumer-facing product verdict.

### Audit Findings
1. **Current Codebase:** No Product Analysis Layer exists. `backend/services/analysis_service/analyzer.py` merely packages whatever lists were returned by the matcher and nutrition lookup into a raw dictionary.
2. **Missing Decision Logic:**
   * What threshold of "High Risk" or "Moderate Risk" ingredients triggers a product-level **Red** vs. **Yellow**?
   * How do high sodium/sugar levels alter an otherwise "Safe" ingredient list?
   * How does a confirmed user allergen interact with general safety (e.g., does an allergen automatically override safety to Red)?
3. **Verdict:** This entire layer is an **unimplemented architectural gap**.

---

## 14. Red / Yellow / Green UI Audit

### Audit Findings

1. **Dashboard Mockup (`templates/home.html`):**
   * Contains three hardcoded status pill classes in CSS (`static/styles.css`):
     * `.status-pill.safe`: `#16A34A` (Green) with checkmark icon ("Safe - No Allergens Found").
     * `.status-pill.uncertain`: `#D97706` (Amber/Yellow) with question mark ("Uncertain - May Contain: Tree Nuts").
     * `.status-pill.not-safe`: `#DC2626` (Red) with exclamation icon ("Not Safe - Contains: Milk, Soy").
2. **Live Upload / Result Screen (`templates/upload.html`, `static/app.js`):**
   * The live results screen (`#resultsPanel`) has **zero Red / Yellow / Green status badges**.
   * It renders a plain summary grid (`Name`, `Brand`, `Domain`) followed by unordered text cards (`resultItem`).
   * The backend does not send any product-level color or tier.
3. **Verdict:** The Red/Yellow/Green UI is currently a **visual mockup only** and is not connected to any backend data or decision engine.

---

## 15. Backend Audit

### Architecture Details
* **Framework:** Flask 3.0.3 (WSGI)
* **Application Factory:** `backend/__init__.py:create_app()`
* **Entry Point:** `app.py` (`app.run(debug=True)`)
* **Routing:** Blueprint `api_bp` registered with prefix `/api` in `backend/routes/api.py`.
* **Endpoints:**
  * `GET /`: Renders dashboard (`home.html`).
  * `GET /upload`: Renders scan page (`upload.html`).
  * `GET /login`: Renders placeholder login (`login.html`).
  * `POST /api/analyze`: Accepts `multipart/form-data` with `image`.
* **State & Configuration:**
  * `app.config["KNOWLEDGE_BASE"]`: Loaded once at application startup from CSV/XLSX files.
* **Error Handling:** Returns `400` with JSON `{"error": ...}` on missing image or invalid extension.

### Proposed Integration Point
The cleanest future integration point is refactoring `backend/routes/api.py` and `analyzer.py`:
1. `POST /api/analyze` accepts `image` (file) and `category` (form string: `"food"` or `"personal_care"`).
2. Validate `category in {"food", "personal_care"}` upfront (return 400 if missing or invalid).
3. Pass `(image_bytes, category)` into `analyze_product()`.
4. Route to `run_food_pipeline()` or `run_personal_care_pipeline()`.

---

## 16. Frontend Audit

### Architecture Details
* **Technology:** Vanilla HTML5 + Jinja2 + CSS3 + Vanilla JavaScript (ES6+).
* **Upload Interaction:** Drag-and-drop zone and hidden file input (`#imageInput`).
* **State Management:** Simple DOM class manipulation (`hidden` toggles for `#loadingState`, `#previewWrap`, `#resultsPanel`).
* **API Communication:** `fetch("/api/analyze", { method: "POST", body: formData })`.

### Missing Frontend Pieces
1. **Category Selection UI:** Must add a prominent, mandatory selector (e.g., dual toggle cards: 🍏 Food vs. 🧴 Personal Care) before the file dropzone.
2. **Form Data Serialization:** Include `formData.append("category", selectedCategory)`.
3. **Hero Result Banner:** Implement a prominent Green / Yellow / Red product verdict card on the results screen.
4. **Domain-Specific Result Sections:** Food should display Nutrition Facts and Allergen warnings; Personal Care should display Function and Irritation warnings.

---

## 17. Model Artifact Audit

### Current vs. Required Artifact Inventory

| Domain | Artifact Name | Current Status in Repo | Required Production Artifact |
| :--- | :--- | :--- | :--- |
| **Food Safety** | Classifier | `backend/ml/models/safety_model.joblib` *(Outdated XGBoost)* | `models/food_safety/classifier.joblib` *(Balanced LogisticRegression)* |
| **Food Safety** | Vectorizer | `backend/ml/models/vectorizer.joblib` *(Char 2–5)* | `models/food_safety/vectorizer.joblib` *(Char 3–5, sublinear_tf)* |
| **Food Safety** | Semantic Model | None | `sentence-transformers/all-MiniLM-L6-v2` *(Local cache or HuggingFace ID)* |
| **Food Safety** | Label Encoder | `backend/ml/models/safety_label_encoder.joblib` | `models/food_safety/label_encoder.joblib` |
| **Food Safety** | Metadata | None | `models/food_safety/metadata.json` *(Version, hash, metrics)* |
| **Personal Care** | Classifier | None | `models/personal_care/classifier.joblib` *(TBD)* |
| **Personal Care** | Feature Extractor | None | `models/personal_care/vectorizer.joblib` *(TBD)* |
| **Personal Care** | Metadata | None | `models/personal_care/metadata.json` *(TBD)* |
| **Allergy** | Model / Rules | `backend/ml/models/allergy_model.joblib` *(Dormant XGBoost)* | Dedicated Allergen Rule Engine / Model |

---

## 18. Dependency Audit

### Environment Inspection

#### 1. PicWise Repository `requirements.txt`
```text
Flask==3.0.3
openpyxl==3.1.5
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
```

#### 2. `final-ocr` Repository `requirements.txt`
```text
opencv-python>=4.8,<5.0
numpy>=1.24,<2.0
pandas>=2.0
paddleocr>=2.7,<3.0
paddlepaddle>=2.6,<2.7
rapidfuzz>=3.0
openpyxl>=3.1
Pillow>=10.0
```

#### 3. Active Python Environment State (`pip list`)
* `Flask 3.0.3` installed.
* `scikit-learn 1.9.0` installed.
* `sentence-transformers 6.0.1` installed.
* `torch 2.14.0` installed.
* `xgboost 3.4.1` installed.
* `pandas 3.0.5` installed.
* `numpy 2.3.4` installed.
* **Missing in active environment:** `paddleocr`, `paddlepaddle`, `opencv-python`, `rapidfuzz`.

### Critical Compatibility Risk: NumPy 2.x vs. PaddlePaddle
> [!WARNING]
> **CRITICAL DEPENDENCY CONFLICT RISK**  
> `final-ocr` requires `paddlepaddle>=2.6,<2.7` and specifies `numpy>=1.24,<2.0`.  
> PaddlePaddle 2.6 C-extensions break when run under NumPy 2.0+ due to ABI changes. The active environment has `numpy 2.3.4`.  
> In Phase 9B, dependency alignment must pin `numpy<2.0.0` inside the dedicated virtual environment before installing `paddlepaddle`.

---

## 19. Git / GitHub Audit

### Repository State
* **Working Directory:** `C:\Users\velzyaa\Desktop`
* **Remote Origin:** `https://github.com/raiamisha25/PicWise.git`
* **Current Branch:** `main`
* **Current Commit:** `dbc9d68b0eeb046d87c9063de2c8fc7d281cb042`
* **Branch Sync Status:** `ahead of 'origin/main' by 7 commits`
* **Modified Tracked Files:** **0** (clean working tree).
* **Git Status:** Clean with respect to tracked files; numerous untracked files.

### Critical Git Root Misconfiguration
> [!CAUTION]
> **CRITICAL REPOSITORY LOCATION RISK**  
> The Git repository root is set to `C:\Users\velzyaa\Desktop` instead of a dedicated folder like `C:\Users\velzyaa\Desktop\PicWise`.  
> Consequently, **every document, shortcut, and personal file on the user's desktop** is currently an untracked file in git:
> * Personal documents (`AI.docx`, `IWT_524110045.docx`, `524110045.docx`, `-format.docx`)
> * Financial/employment forms (`Recruitment of Probationary Officers...pdf`, `CRP PO_MT-XVI.pdf`)
> * High-risk PII files (`masked addhar.jpg`, `passport size photo/`)
> * Shortcuts (`Discord.lnk`, `Opera Browser.lnk`, `MongoDB Compass.lnk`)
> * Other local code projects (`ai-learnmate`, `Mess management`)
>
> **Hazard:** Any unintentional `git add .` or `git add -A` will stage sensitive personal data and push it to the public GitHub repository.

### Tracked Datasets & Large Files in Git
* **Tracked Datasets:**
  * `data/food/food_ingredients_dataset_final.csv`
  * `data/food/ingredient_knowledge_base_500_cleaned.csv`
  * `data/food/ingredient_knowledge_base_500_with_alternate_names.csv`
  * `data/nutrition/nutrition_knowledge_dataset.csv`
  * `data/personal_care/personal_care_ingredients_dataset_cleaned.xlsx`
  * `data/personal_care/personal_care_ingredients_dataset_csv.xlsx`
* **Tracked Binary Model Files:**
  * `backend/ml/models/safety_model.joblib` (862 KB)
  * `backend/ml/models/allergy_model.joblib` (870 KB)
  * `backend/ml/models/vectorizer.joblib` (391 KB)
  * Total tracked binaries: ~2.1 MB.
* **Tracked Generated JSON Reports:**
  * `backend/ml/evaluation/improvement_experiments.json` (190 KB)
  * `backend/ml/evaluation/error_analysis.json` (362 KB)

---

## 20. Current Architecture Diagram

```text
CURRENT PICWISE IMPLEMENTATION (AS-IS)

[User]
   │
   ▼  Uploads image (NO category selection)
[Browser UI: upload.html / app.js]
   │
   ▼  POST /api/analyze (FormData: image only)
[Flask Route: api.py]
   │
   ▼  image_bytes
[backend.services.analysis_service.analyzer.py]
   │
   ├──► [ocr_service.stub.py]
   │       └─► Returns: "Ingredients: whole wheat flour, sugar... Nutrition: energy..."
   │
   ├──► [ingredient_matching.matcher.py]
   │       └─► Regex string split -> Exact KB dict lookup (500 items)
   │
   ├──► [nutrition_service.lookup.py]
   │       └─► Substring match on raw text -> Returns qualitative role/impact
   │
   ├──► _detect_domain(ingredients)  <── [ANTI-PATTERN: Infers domain by counting matches]
   │
   ▼
[JSON Response: { product: {domain}, ingredients, nutrition, personalCare, warnings }]
   │
   ▼
[Browser UI renders raw unranked text cards] (NO Green/Yellow/Red verdict)

[DORMANT CODE / UNUSED IN PRODUCTION PATH]:
- backend/ml/models/*.joblib (Phase 2 XGBoost models, not called by analyzer)
- backend/ml/inference/predictor.py (Never imported by services)
- final-ocr repo (PaddleOCR + layout analysis, completely unintegrated)
```

---

## 21. Proposed Production Architecture

```text
PROPOSED PRODUCTION ARCHITECTURE (TO-BE)

                          USER
                            │
                            ▼
                   [Select Product Category]
                     ┌──────┴──────┐
                     │             │
                   FOOD      PERSONAL CARE
                     │             │
                     └──────┬──────┘
                            │
                            ▼
                    [Image Upload (JPG/PNG/WEBP)]
                            │
                            ▼
             POST /api/analyze (image + category)
                            │
                            ▼
                 [PicWise Analysis Controller]
                            │
                            ▼
         [Production OCR Engine: Integrated final-ocr]
         (PaddleOCR + Document Layout + Bounding Polygons)
                            │
                            ▼
                     Extracted Data
                     ┌──────┴──────┐
                     │             │
                 FOOD PATH   PERSONAL CARE PATH
                     │             │
       ┌─────────────┼─────────────┤
       ▼             ▼             ▼
  [Food Safety] [Nutrition]    [Allergy]       [Personal Care]
  Frozen Model   Scoring       Detection          Safety &
 (MiniLM+LR)   (RDA/Limits)   (Pack+User)        Irritation
       │             │             │                 │
       └─────────────┼─────────────┘                 │
                     │                               │
                     ▼                               ▼
            [Food Product Analysis]       [Personal Care Analysis]
          (Rule-based Risk Aggregator)  (Rule-based Risk Aggregator)
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
                        [Overall Product Assessment]
                                     │
                             ┌───────┼───────┐
                             ▼       ▼       ▼
                           GREEN   YELLOW   RED
                             │       │       │
                             └───────┼───────┘
                                     ▼
                      [PicWise UI Results Dashboard]
```

---

## 22. Production Readiness Table

| Component | Current State | Production Ready? | Missing Pieces | Risk | Next Action |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Category Selection** | Absent from UI and API | **NO** | Frontend radio/toggle; `FormData` field; API request validation. | **HIGH** (Violates architecture; triggers wrong pipeline) | Add explicit UI selection and enforce backend parameter in Phase 9B. |
| **OCR** | Stubbed in PicWise; Fully mature in `final-ocr` | **NO** *(in PicWise)*<br>**YES** *(in `final-ocr`)* | Integration of `final-ocr` modules into `backend/services/ocr_service/`; handling in-memory bytes. | **MEDIUM** (NumPy 2.x vs PaddlePaddle dependency conflict) | Vendor/integrate `final-ocr` into `backend/services/ocr_service/` and adapt for byte streams. |
| **Food Safety Model** | Frozen specification verified; old XGBoost in repo | **NO** | Serialization of frozen MiniLM + LogisticRegression; inference service wrapper. | **MEDIUM** (Latency of MiniLM on CPU; memory footprint) | Implement script to serialize frozen model and build `FoodSafetyPredictor`. |
| **Nutrition / Health** | Qualitative KB lookup only | **NO** | Quantitative scoring calculation; interface for structured OCR nutrition dict. | **LOW** (Mathematical rules are deterministic) | Define and implement nutrition scoring function taking OCR nutrition dict. |
| **Allergy** | Fragmented between KB lookup and OCR filters | **NO** | Allergen detection engine; user allergen profile matching. | **MEDIUM** (False negatives on severe allergens) | Connect OCR allergen text and ingredient warnings to an allergy service. |
| **Personal Care Model** | Incomplete / Not found | **NO** | Model architecture, training, validation, artifacts, inference wrapper. | **HIGH** (Domain completely unsupported by ML) | Establish dedicated Personal Care model using expanded dataset. |
| **Product Analysis** | Missing entirely | **NO** | Aggregation logic mapping component outputs to Green/Yellow/Red. | **HIGH** (Cannot produce final consumer verdict) | Design and implement deterministic product aggregation rules. |
| **Red/Yellow/Green UI** | Static mockup on `home.html` only | **NO** | Dynamic product status card on `upload.html` driven by API response. | **LOW** (UI rendering straightforward once API sends status) | Implement status badge and color theme rendering in `app.js`. |
| **Backend** | Solid Flask foundation | **PARTIAL** | Pipeline routing; controller refactor; error handling for OCR/ML. | **LOW** (Framework is clean and modular) | Refactor `api.py` and `analyzer.py` to route based on category. |
| **Frontend** | Clean responsive UI | **PARTIAL** | Category selector; results card redesign. | **LOW** (Clean CSS and vanilla JS foundation) | Update `upload.html` and `app.js` with category selection and status banner. |
| **Model Artifacts** | Only outdated Phase 2 models exist | **NO** | Serialized Frozen Food Safety model artifacts; version metadata. | **MEDIUM** (Git LFS vs file tracking) | Create `models/food_safety/` and serialize frozen weights. |
| **Dependencies** | Split across two repositories; NumPy conflict | **NO** | Consolidated `requirements.txt`; resolution of NumPy 1.x requirement for Paddle. | **HIGH** (PaddlePaddle crashes if NumPy 2.x is present) | Create unified `requirements.txt` with `numpy<2.0.0`, `paddleocr`, `sentence-transformers`. |

---

## 23. Gaps

### Architectural & Functional Gaps
1. **Missing Category Routing:** User category selection does not exist in UI or API. Backend attempts automatic domain guessing via ingredient counts.
2. **Missing Product Analysis Layer:** No logic exists to combine ingredient risks, nutrition scores, and allergens into an overall product-level score or rating.
3. **Missing Red/Yellow/Green Verdict:** UI results screen only renders a plain list; no aggregated color badge is produced.
4. **Disconnected OCR Engine:** Real OCR lives in a separate repository (`final-ocr`); main repository uses a 7-line stub.
5. **OCR-Backend Interface Mismatch:** Real OCR outputs rich dictionaries with bounding boxes and value-unit pairs; PicWise backend expects a single raw string.
6. **Unserialized Frozen Food Safety Model:** The final MiniLM + LogisticRegression model exists only as an experimental configuration, not as a deployed artifact.
7. **Missing Personal Care Model:** No dedicated Personal Care ML architecture, hyperparameters, or model artifacts exist.
8. **Missing Quantitative Nutrition Scoring:** Nutrition component only performs qualitative lookups; does not calculate health scores from OCR numbers.
9. **Fragmented Allergy Detection:** No unified module combining OCR packaging allergen warnings, ingredient risks, and user profiles.

### Data & Environment Gaps
10. **Personal Care Dataset Discrepancy:** Tracked dataset has only 1 High Risk instance; untracked dataset has 45 High Risk instances but is not in git.
11. **NumPy 2.x vs. PaddlePaddle Conflict:** Active Python environment runs NumPy 2.3.4, which breaks PaddlePaddle 2.6.
12. **Desktop Git Root:** Git root initialized at `C:\Users\velzyaa\Desktop`, exposing sensitive personal files to accidental commits.

---

## 24. Risks

### Technical & System Risks
1. **Dependency Incompatibility (NumPy / PaddlePaddle):** PaddlePaddle 2.6 binary wheels require NumPy $< 2.0$. If installed in an environment with NumPy 2.x, import failures or segmentation faults will occur.
2. **CPU Inference Latency:** Running PaddleOCR (detection + recognition + angle classification + ensemble) alongside Sentence-Transformers (`all-MiniLM-L6-v2`) on a standard CPU could result in an end-to-end response time of 5–15 seconds per image.
3. **Memory Consumption:** Loading PaddleOCR models, PyTorch (`sentence-transformers`), and scikit-learn models in a single worker process requires ~1.5 GB to 2.5 GB of RAM.
4. **Data Privacy & Git Exposure:** The desktop git repository contains PII (government ID images, personal photos, academic files) as untracked files. Any rogue git command poses an immediate privacy and security risk.

### Domain & Safety Risks
5. **False-Safe Underestimation in Food Safety:** The frozen Food Safety model has a False Safe Rate of $32.98\%$ on minority risk classes. Without a conservative decision threshold or human-in-the-loop warning, risky ingredients could be labeled safe.
6. **Allergen Omission Risk:** Missing an allergen statement due to OCR crop truncation or poor lighting is a critical user safety hazard. Allergen lines must have high-recall priority.

---

## 25. Recommended Next Phase (Phase 9B Action Plan)

Moving into Phase 9B, execution must proceed through disciplined, sequential stages:

### Step 1: Environment & Repository Hygiene
1. Isolate the PicWise codebase into a clean, dedicated project directory (`C:\Users\velzyaa\Desktop\PicWise`) so Desktop files are completely detached from git.
2. Update `.gitignore` to explicitly ignore virtual environments, cache files, and local personal files.
3. Establish a dedicated virtual environment with `numpy<2.0.0`, `paddlepaddle==2.6.2`, `paddleocr>=2.7`, `sentence-transformers`, `torch`, `Flask`, `scikit-learn`, `rapidfuzz`, and `openpyxl`.

### Step 2: Integrate Accepted OCR Subsystem
1. Transfer the proven `final-ocr` pipeline modules into `backend/services/ocr_service/`.
2. Adapt `load_image` to handle in-memory image bytes (`io.BytesIO`) directly from Flask's `request.files["image"]`.
3. Expose a unified OCR interface: `run_ocr(image_bytes, domain)` returning structured ingredients and nutrition dictionaries.

### Step 3: Serialize & Integrate Frozen Food Safety Model
1. Run a clean, isolated script to fit and serialize the frozen **Character TF-IDF (3–5 n-grams) + MiniLM + Balanced Logistic Regression** pipeline using the canonical group split.
2. Save production artifacts to `models/food_safety/` (`classifier.joblib`, `vectorizer.joblib`, `label_encoder.joblib`, `metadata.json`).
3. Implement `FoodSafetyPredictor` in `backend/ml/inference/food_safety_predictor.py`.

### Step 4: Implement Category Selection & Backend Routing
1. Update `templates/upload.html` with explicit Category Selection cards (**Food** vs. **Personal Care**).
2. Update `static/app.js` to validate category selection and append it to `FormData`.
3. Update `backend/routes/api.py` to validate `category` and route strictly to the domain pipeline.
4. Eliminate `_detect_domain()` automated guessing from `analyzer.py`.

### Step 5: Implement Product Analysis Layer & Green/Yellow/Red Engine
1. Implement quantitative Nutrition Scoring based on extracted nutrient values per 100g.
2. Implement Allergy Matching cross-referencing OCR allergen statements and ingredient risks.
3. Build the Product Analysis Layer with explicit aggregation rules:
   * **🔴 Red:** Any High Risk ingredient, severe allergen match, or critical nutritional excess.
   * **🟡 Yellow:** Multiple Moderate Risk ingredients, missing critical label sections, or moderate nutritional concerns.
   * **🟢 Green:** All ingredients Safe/Very Safe, zero allergens detected, balanced nutritional profile.

### Step 6: Update Results UI & End-to-End Verification
1. Add the Green/Yellow/Red hero verdict card to `templates/upload.html` and `static/app.js`.
2. Write automated unit and integration tests covering the complete pipeline with real product fixtures.
3. Validate end-to-end performance and latency.

---
*Report completed under Phase 9A strict read-only audit guidelines. Awaiting user review and authorization before initiating Phase 9B implementation.*
