"""
PicWise ML Inference Service Layer

NOTE: In Phase 9C, the production Food Safety inference path uses the frozen
Dual-stream Character TF-IDF (3-5 n-grams) + MiniLM-L6-v2 + Balanced Logistic Regression pipeline.
The legacy Phase 2 XGBoost model (backend/ml/models/safety_model.joblib) is preserved
strictly as a historical experiment artifact and is NOT used for production food safety predictions.
"""

import os

import joblib
from backend.ml.inference.food_safety_service import predict_food_safety

DEFAULT_LEGACY_MODEL_DIR = "backend/ml/models"

_LEGACY_CACHE = {
    "vectorizer": None,
    "allergy_model": None,
    "allergy_encoder": None,
    "loaded": False,
}


def _load_legacy_artifacts(model_dir=DEFAULT_LEGACY_MODEL_DIR):
    if _LEGACY_CACHE["loaded"]:
        return

    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
    allergy_model_path = os.path.join(model_dir, "allergy_model.joblib")
    allergy_encoder_path = os.path.join(model_dir, "allergy_label_encoder.joblib")

    if (
        os.path.exists(vectorizer_path)
        and os.path.exists(allergy_model_path)
        and os.path.exists(allergy_encoder_path)
    ):
        _LEGACY_CACHE["vectorizer"] = joblib.load(vectorizer_path)
        _LEGACY_CACHE["allergy_model"] = joblib.load(allergy_model_path)
        _LEGACY_CACHE["allergy_encoder"] = joblib.load(allergy_encoder_path)
        _LEGACY_CACHE["loaded"] = True


def predict_ingredient_risk(ingredient_name: str) -> dict:
    """
    Predicts Food Safety Level (using frozen production Logistic Regression pipeline)
    and legacy Allergy Risk for a given ingredient name.

    Returns:
        dict: {
            "safetyLevel": str or None,
            "allergyRisk": str or None,
            "confidence": float,
            "foodSafety": dict (full structured food safety prediction)
        }
    """
    if not ingredient_name or not str(ingredient_name).strip():
        return {
            "safetyLevel": None,
            "allergyRisk": None,
            "confidence": 0.0,
            "foodSafety": None,
        }

    cleaned_name = str(ingredient_name).strip()

    # 1. Production Food Safety Prediction (Frozen Character TF-IDF + MiniLM + Balanced Logistic Regression)
    food_safety_pred = predict_food_safety(cleaned_name)
    predicted_safety = food_safety_pred.get("risk_class")
    safety_conf = food_safety_pred.get("confidence", 0.0)

    # 2. Legacy Allergy Prediction (preserved for backward compatibility until Phase 9 Allergy Engine)
    _load_legacy_artifacts()
    predicted_allergy = None
    allergy_conf = 0.0

    if _LEGACY_CACHE["loaded"]:
        try:
            vec = _LEGACY_CACHE["vectorizer"]
            allergy_model = _LEGACY_CACHE["allergy_model"]
            allergy_encoder = _LEGACY_CACHE["allergy_encoder"]

            X_vec = vec.transform([cleaned_name])
            allergy_probs = allergy_model.predict_proba(X_vec)[0]
            allergy_class_idx = allergy_probs.argmax()
            predicted_allergy = str(allergy_encoder.inverse_transform([allergy_class_idx])[0])
            allergy_conf = float(allergy_probs[allergy_class_idx])
        except Exception:
            predicted_allergy = None
            allergy_conf = 0.0

    overall_confidence = round(float((safety_conf + allergy_conf) / 2.0), 4) if predicted_allergy else safety_conf

    return {
        "safetyLevel": predicted_safety,
        "allergyRisk": predicted_allergy,
        "confidence": overall_confidence,
        "foodSafety": food_safety_pred,
    }


__all__ = ["predict_food_safety", "predict_ingredient_risk"]
