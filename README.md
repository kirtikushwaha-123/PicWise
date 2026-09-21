# PicWise Phase 1

PicWise is a Flask-based product label analysis skeleton. Users upload a food or
personal-care label image, the Flask app calls `POST /api/analyze`, and the
backend runs a swappable OCR boundary, ingredient matching, knowledge-base
lookups, and response shaping.

Phase 1 does not include ML, image classification, authentication, accounts, a
database server, payments, or deployment infrastructure.

## Tech Choice

This project uses Flask for both the website and the REST API because the project
is intended to stay entirely within the Flask framework. Flask keeps the Phase 1
app compact: Jinja templates render the pages, static JavaScript handles upload
interactions, and Python service modules own the backend pipeline.

## Project Structure

```text
PicWise/
|-- app.py
|-- backend/
|   |-- routes/
|   |-- services/
|   |   |-- ocr_service/
|   |   |-- ingredient_matching/
|   |   |-- nutrition_service/
|   |   `-- analysis_service/
|   `-- ml/
|       |-- preprocessing/
|       |-- training/
|       |-- inference/
|       `-- models/
|-- data/
|   |-- food/
|   |-- nutrition/
|   `-- personal_care/
|-- templates/
|-- static/
`-- README.md
```

## Knowledge Base Files

The current knowledge base is file-backed and loaded into memory at app startup.
The backend reads these files only; the frontend never reads them directly.

Default paths:

```text
FOOD_DATA_PATH=data/food/ingredient_knowledge_base_500_with_alternate_names.csv
NUTRITION_DATA_PATH=data/nutrition/nutrition_knowledge_dataset.csv
PERSONAL_CARE_DATA_PATH=data/personal_care/personal_care_ingredients_dataset_csv.xlsx
```

To swap datasets later, set those environment variables to different CSV/XLSX
paths without changing the API contract.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

Useful local pages:

- `http://127.0.0.1:5000/` - dashboard-style home page
- `http://127.0.0.1:5000/upload` - product scan/upload flow
- `http://127.0.0.1:5000/login` - Phase 1 login UI placeholder

## API

`POST /api/analyze`

Request:

- `multipart/form-data`
- file field: `image`
- supported types: JPG, JPEG, PNG, WEBP

Response:

```json
{
  "product": {
    "name": null,
    "brand": null,
    "domain": "food"
  },
  "ingredients": [],
  "nutrition": [],
  "personalCare": [],
  "warnings": []
}
```

Missing values are returned as `null` by the API and rendered by the UI as
`Information not available`.

## Stubbed vs Real

Stubbed:

- OCR in `backend/services/ocr_service/stub.py`. It currently returns placeholder
  extracted text so the rest of the pipeline can be exercised end to end.
- Login page UI. It is a visual placeholder only; no account system is active.

Real:

- Flask route and multipart image validation.
- Knowledge-base loading from env-configurable file paths.
- CSV loading for food and nutrition.
- XLSX loading for personal-care ingredients via `openpyxl`.
- Simple ingredient normalization and lookup against primary and alternate food
  ingredient names.
- Response shaping for the agreed Phase 1 API contract.

## Future Extension Points

- Replace `extract_text(image_bytes)` with Tesseract or a vision model without
  touching the route or analysis service.
- Swap data files by changing env vars.
- Add later ML code under `backend/ml/` without restructuring the API.

## ML Training (Phase 2)

PicWise includes a multi-class XGBoost text classification pipeline under `backend/ml/` for predicting `Safety Level` and `Allergy Risk` from ingredient names using character n-gram TF-IDF features.

### How to Run Training

Run the training pipeline from the root directory:

```powershell
python -m backend.ml.training.train
```

The pipeline automatically:
1. Loads food and personal-care datasets (configured via `FOOD_DATA_PATH` and `PERSONAL_CARE_DATA_PATH`).
2. Normalizes safety level labels (`Very Safe`, `Safe`, `Moderate Risk`, `High Risk`) and maps missing/blank allergy risk values to `"None"`.
3. Engineers character TF-IDF n-gram features (including alternate and packaging names).
4. Trains two multi-class XGBoost models for Safety Level and Allergy Risk.
5. Saves fitted artifacts to `backend/ml/models/`.

### Model Artifacts

Artifacts are saved in `backend/ml/models/`:
- `vectorizer.joblib`: Fitted character n-gram `TfidfVectorizer`
- `safety_model.joblib`: Trained XGBoost Safety Level classifier
- `allergy_model.joblib`: Trained XGBoost Allergy Risk classifier
- `safety_label_encoder.joblib`: LabelEncoder for Safety Level
- `allergy_label_encoder.joblib`: LabelEncoder for Allergy Risk

### Inference Usage

Import `predict_ingredient_risk` in service modules:

```python
from backend.ml.inference.predictor import predict_ingredient_risk

result = predict_ingredient_risk("Sodium Lauryl Sulfate")
# Returns: {"safetyLevel": "Moderate Risk", "allergyRisk": "Low", "confidence": 0.88}
```
