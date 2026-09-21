# PHASE 9B — REPOSITORY FOUNDATION & OCR INTEGRATION REPORT

**Project:** PicWise  
**Phase:** 9B (Repository Hygiene, Environment Foundation, OCR Integration & Category Routing)  
**Date:** September 14, 2026  
**Status:** COMPLETED & VERIFIED  

---

## 1. Executive Summary

Phase 9B transitions PicWise from the ML experimentation audit (Phase 9A) into the first production integration milestone. All objectives set for Phase 9B have been fully realized:

1. **Repository Hygiene:** The Git repository was safely disentangled and isolated from `C:\Users\velzyaa\Desktop` into its dedicated root at `C:\Users\velzyaa\Desktop\PicWise`. Desktop personal files remain untouched and completely invisible to Git.
2. **Dedicated Environment:** A standalone virtual environment (`.venv`) was built using Python 3.13.2 with PyTorch, PaddlePaddle, PaddleOCR, Sentence-Transformers, XGBoost, and OpenCV properly provisioned.
3. **Full In-Memory OCR Integration:** The `final-ocr` subsystem was integrated into `backend/services/ocr_service/`. A pure in-memory pipeline (`run_ocr(image_bytes, category)`) was engineered, eliminating all disk-write latency and supporting both Food and Personal Care analysis.
4. **Strict Category Routing:** Heuristic domain guessing (`_detect_domain()`) has been completely eradicated. The user's explicit selection (`food` vs. `personal_care`) is now strictly validated at both UI and API boundaries, directly dictating vocabulary lookup and nutrition extraction behavior.
5. **Zero Out-of-Scope Leakage:** No ML models were trained, no nutrition scoring algorithms were introduced, no decision rules were altered, and no Git commits or remote pushes were executed.
6. **100% Test Passing Rate:** All 49 test cases (18 core unit tests, 25 ML evaluation/grouping tests, and 6 new OCR integration tests) passed cleanly.

---

## 2. Repository Hygiene Verification

### 2.1 Separation from Desktop
Previously, Git tracking resided at `C:\Users\velzyaa\Desktop\.git`, placing the user's personal desktop environment under version control. During Phase 9B:
- The `.git` repository, working tree, and configuration files were safely relocated into `C:\Users\velzyaa\Desktop\PicWise`.
- Verification command confirmed repository root:
  ```powershell
  git rev-parse --show-toplevel
  # Output: C:/Users/velzyaa/Desktop/PicWise
  ```
- Verification command confirmed Desktop is no longer a repository:
  ```powershell
  git -C "C:\Users\velzyaa\Desktop" status
  # Output: fatal: not a git repository (or any of the parent directories): .git
  ```
- All personal files, shortcuts, pictures, and folders on Desktop are 100% intact and unmonitored.
- Commit history remains preserved on branch `main` at commit `dbc9d68`.

---

## 3. Environment Foundation & Dependencies

A dedicated virtual environment was constructed at `C:\Users\velzyaa\Desktop\PicWise\.venv`.

### 3.1 Key Runtime Packages
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `python` | `3.13.2` | Core runtime |
| `paddleocr` | `3.7.0` | OCR text detection and recognition engine |
| `paddlepaddle` | `3.3.1` | Deep learning backend for OCR models |
| `torch` | `2.14.0` | Embedding model acceleration |
| `sentence-transformers` | `6.0.1` | Semantic ingredient similarity matching |
| `opencv-python` | `5.0.0` | Image preprocessing and packet contour detection |
| `rapidPuzz` | `3.14.6` | High-speed fuzzy string matching |
| `scikit-learn` | `1.9.1` | Baseline ML metrics and evaluation splits |
| `xgboost` | `3.4.1` | Ingredient risk assessment classifiers |
| `pandas` | `3.0.5` | Knowledge base indexing and tabular data handling |
| `flask` | `3.1.3` | Backend web service API |

### 3.2 Platform Resolution (Windows DLL & oneDNN PIR Fixes)
1. **PyTorch / PaddlePaddle DLL Collision:** On Windows, PyTorch DLL directories (`torch/lib`) must be added to the DLL search path via `os.add_dll_directory()` before loading native C++ extensions. This was implemented lazily in `paddle_engine.py`.
2. **PaddlePaddle 3.3.1 oneDNN PIR Bug on Windows:** The default CPU runner in PaddlePaddle 3.3.1 / PaddleOCR 3.7.0 triggered a `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]` inside `onednn_instruction.cc:118`. This was resolved by passing `enable_mkldnn=False` during PaddleOCR initialization, allowing smooth execution on standard CPU instructions.

---

## 4. OCR Integration Architecture

The OCR subsystem in `backend/services/ocr_service/` provides an in-memory pipeline:

```text
Uploaded Image Bytes (in-memory)
             │
             ▼
    [load_image_from_bytes]
    (Decodes via OpenCV, orientation normalized)
             │
             ├──────────────────────────┐
             ▼                          ▼
   [Preprocessing & Packet]    [PaddleEngine OCR]
   - Packet border detection   - Text detection (PP-OCRv6)
   - Quality / blur check      - Text recognition
   - Region crop heuristics    - Standardized bounding boxes
             │                          │
             └──────────┬───────────────┘
                        ▼
            [Hierarchical Region Parser]
            - Separates Ingredients text block
            - Separates Nutrition table block (Food only)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
 [NLP Ingredient Parser]      [Nutrition Parser]
 - Tokenizer / delimiter split - Regex / tabular parser
 - Matcher vs. Knowledge Base - Extracts Energy, Fats,
   (Food or Personal Care)      Carbs, Sugars, Sodium, etc.
         │                             │
         └──────────────┬──────────────┘
                        ▼
            Unified Structured Output
```

### 4.1 In-Memory Byte Adapter
In `backend/services/ocr_service/preprocessing/image_utils.py`:
- `load_image_from_bytes(image_bytes)` decodes images directly from byte arrays using `cv2.imdecode` without writing any intermediate files to disk.
- `pipeline.run_ocr(image_bytes, category="food")` accepts raw bytes and returns structured Python dictionaries containing:
  - `domain`: `"food"` or `"personal_care"`
  - `ingredients`: Extracted ingredients with canonical name, match confidence, match type, and metadata.
  - `nutrition`: Extracted nutrient values, units, and per-100g metrics (Food only; `None` for Personal Care).
  - `packet_detection`: Bounding polygons and packet detection confidence.
  - `raw_text`: Dictionary with `all_text`, `ingredients_text`, and `nutrition_text`.
  - `image_quality`: Brightness, blur scores, and quality flags.

---

## 5. Category Routing & Backend Wiring

### 5.1 Elimination of Heuristic Domain Guessing
Previously, `analyzer.py` inspected extracted ingredients and used `_detect_domain(ingredients)` to guess whether a product was food or personal care based on count tallies. This violated the PicWise architecture requirement:
> **"The user explicitly selects the product category BEFORE OCR. There is NO automatic Food-vs-Personal-Care classification model."**

- `_detect_domain()` was **completely removed**.
- The `category` is now passed directly from the user's selection into `analyze_product_image(image_bytes, knowledge_base, category=category)`.

### 5.2 API Validation (`backend/routes/api.py`)
The `/api/analyze` endpoint strictly enforces:
1. **Category Required:** Missing or blank `category` field returns `400 Bad Request` (`{"error": "Product category is required in form field 'category'."}`).
2. **Category Values:** Values other than `"food"` or `"personal_care"` return `400 Bad Request` (`{"error": "Invalid category '...'. Allowed values are 'food' or 'personal_care'."}`).
3. **Image Presence:** Missing image or empty filename returns `400 Bad Request`.
4. **Empty Payload:** 0-byte uploaded files return `400 Bad Request` (`{"error": "Uploaded image file is empty."}`).
5. **Corrupt Files:** Corrupt or non-image payloads verified via PIL return `400 Bad Request` (`{"error": "Invalid or corrupt image file."}`).

---

## 6. Frontend UI Enhancements

### 6.1 Category Selection Radio Cards
In `templates/upload.html`:
- Added interactive toggle cards allowing users to choose between:
  - 🍏 **Food & Beverages:** Extracts ingredients & nutrition facts.
  - 🧴 **Personal Care:** Extracts cosmetics & skin safety ingredients.
- Radio input is checked by default on Food and accessible via keyboard navigation.
- Status badge updated from "Stubbed" to `Integrated (PaddleOCR)`.

### 6.2 Modern CSS Styling (`static/styles.css`)
- Styled `.category-card` and `.category-card-inner` to match PicWise's design system tokens (`--primary-blue`, `--blue-tint`, `--dark-navy`).
- Active selection displays crisp blue focus borders and subtle tint elevation.

### 6.3 Client-Side Logic (`static/app.js`)
- Validates category selection before submitting.
- Appends `category` to `FormData`.
- Updated `renderNutrition()` to handle modern structured nutrition maps (`{energy: {value, unit, per_100g}, ...}`).
- Displays `"Not applicable for personal care products."` when personal care is analyzed, cleanly omitting irrelevant nutritional tables.

---

## 7. Test Suite Execution & Verification

### 7.1 New OCR Integration Test Suite (`tests/test_ocr_integration.py`)
Six targeted integration tests verify end-to-end functionality against the `/api/analyze` endpoint:

| Test Case | Description | Result |
| :--- | :--- | :--- |
| `test_food_analysis_success` | Food image + `category="food"` returns structured ingredients, nutrition dictionary, and empty personal care list. | **PASSED** |
| `test_personal_care_analysis_success` | Personal care image + `category="personal_care"` returns ingredients, populates personal care details, and strictly sets `nutrition=None`. | **PASSED** |
| `test_missing_category_returns_400` | Request omitting `category` form field returns HTTP 400 Bad Request. | **PASSED** |
| `test_invalid_category_returns_400` | Request with `category="automotive"` returns HTTP 400 Bad Request. | **PASSED** |
| `test_corrupt_image_returns_400` | Corrupt image byte payload returns HTTP 400 Bad Request. | **PASSED** |
| `test_empty_image_returns_400` | 0-byte file payload returns HTTP 400 Bad Request. | **PASSED** |

### 7.2 Full Repository Test Suite
Running all 49 tests across the repository:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
**Output:**
```text
Ran 49 tests in 559.067s

OK
```
Zero failures, zero regressions.

---

## 8. Explicit Scope Boundaries (Phase 9B Discipline)

To preserve architectural boundaries, the following were **deliberately NOT modified or introduced**:
- ❌ **No Model Retraining or Re-serialization:** Food safety XGBoost models and personal care candidate models remain in their evaluation state; no production model pickle files were generated.
- ❌ **No Nutrition Scoring:** Algorithms calculating final numeric nutrition health scores were not implemented.
- ❌ **No Product-Level Decision Engine:** Green/Yellow/Red product rating rules remain untouched for future phases.
- ❌ **No Git Pushes:** No commits were pushed to remote GitHub repositories.

---

## 9. Readiness for Phase 9C

With Phase 9B complete:
1. The repository is hygienically isolated and safely managed.
2. The OCR engine reliably converts raw image bytes into structured text, regions, and parsed vocabulary matches.
3. The category selection pipeline provides deterministic domain routing.

The codebase is now ready for **Phase 9C: Production ML Model Training & Artifact Serialization**.
