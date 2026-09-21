"""
backend/ml/inference/personal_care_service.py

Production inference service for the PicWise Personal Care models.
Dual-stream: Character TF-IDF (2-5 n-grams, sublinear_tf=True) + Structured Semantics
+ Balanced Logistic Regression (C=10.0, max_iter=1000).

Provides independent inference across three orthogonal targets:
- Safety_Level
- Allergy_Risk
- Irritation_Risk

Guarantees thread-safety, component isolation, and zero target leakage.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "personal_care"


class TargetPredictor:
    """Wraps an isolated target pipeline (safety, allergy, or irritation)."""

    def __init__(self, target_dir: Path, target_name: str):
        self.target_dir = target_dir
        self.target_name = target_name
        self.pipeline = None
        self.metadata = None
        self.classes = []
        self._loaded = False
        self._load()

    def _load(self):
        pipeline_path = self.target_dir / "pipeline.joblib"
        metadata_path = self.target_dir / "model_metadata.json"

        if not pipeline_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Production artifact missing in {self.target_dir}. "
                f"Run backend/ml/training/train_personal_care.py first."
            )

        self.pipeline = joblib.load(pipeline_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.classes = list(self.pipeline.named_steps["classifier"].classes_)
        self._loaded = True

    def predict(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes inference for this specific target on a 1-row DataFrame.
        """
        try:
            pred_class = self.pipeline.predict(input_df)[0]
            probs = self.pipeline.predict_proba(input_df)[0]
            conf = float(probs[self.classes.index(pred_class)])
            prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.classes, probs)}

            return {
                "risk_class": pred_class,
                "confidence": round(conf, 4),
                "probabilities": prob_dict,
                "status": "success",
            }
        except Exception as exc:
            logger.error("Inference failure for %s: %s", self.target_name, exc, exc_info=True)
            return {
                "risk_class": None,
                "confidence": 0.0,
                "probabilities": {},
                "status": "model_prediction_failure",
                "error": str(exc),
            }


class PersonalCarePredictor:
    """
    Thread-safe production orchestrator for Personal Care inference across all 3 targets.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        self.safety_predictor = TargetPredictor(self.models_dir / "safety", "Safety_Level")
        self.allergy_predictor = TargetPredictor(self.models_dir / "allergy", "Allergy_Risk")
        self.irritation_predictor = TargetPredictor(self.models_dir / "irritation", "Irritation_Risk")
        self.targets = {
            "safety": self.safety_predictor,
            "allergy": self.allergy_predictor,
            "irritation": self.irritation_predictor,
        }
        self._loaded = True

    @classmethod
    def get_instance(cls, models_dir: Optional[Path] = None) -> "PersonalCarePredictor":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(models_dir=models_dir)
        return cls._instance

    def predict(
        self,
        ingredient_name: Optional[str] = None,
        primary_function: Optional[str] = None,
        ingredient_category: Optional[str] = None,
        product_categories: Optional[str] = None,
        origin: Optional[str] = None,
        regulatory_status: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Predicts Safety, Allergy, and Irritation for a personal care ingredient.
        Guarantees component failure isolation across all 3 targets.
        """
        name = str(ingredient_name or kwargs.get("Ingredient_Name") or "").strip()
        func = str(primary_function or kwargs.get("Primary_Function") or "").strip()
        cat = str(ingredient_category or kwargs.get("Ingredient_Category") or "").strip()
        prod_cats = str(product_categories or kwargs.get("Product_Categories") or "").strip()
        orig = str(origin or kwargs.get("Origin") or "").strip()
        reg = str(regulatory_status or kwargs.get("Regulatory_Status") or "").strip()

        row_dict = {
            "Ingredient_Name": [name],
            "Primary_Function": [func],
            "Ingredient_Category": [cat],
            "Product_Categories": [prod_cats],
            "Origin": [orig],
            "Regulatory_Status": [reg],
        }
        input_df = pd.DataFrame(row_dict)

        # Execute predictions with component isolation
        try:
            safety_res = self.safety_predictor.predict(input_df)
        except Exception as exc:
            logger.error("Safety predictor exception: %s", exc, exc_info=True)
            safety_res = {
                "risk_class": None,
                "confidence": 0.0,
                "probabilities": {},
                "status": "model_prediction_failure",
                "error": str(exc),
            }

        try:
            allergy_res = self.allergy_predictor.predict(input_df)
        except Exception as exc:
            logger.error("Allergy predictor exception: %s", exc, exc_info=True)
            allergy_res = {
                "risk_class": None,
                "confidence": 0.0,
                "probabilities": {},
                "status": "model_prediction_failure",
                "error": str(exc),
            }

        try:
            irritation_res = self.irritation_predictor.predict(input_df)
        except Exception as exc:
            logger.error("Irritation predictor exception: %s", exc, exc_info=True)
            irritation_res = {
                "risk_class": None,
                "confidence": 0.0,
                "probabilities": {},
                "status": "model_prediction_failure",
                "error": str(exc),
            }

        return {
            "ingredient": name,
            "safety": safety_res,
            "allergy": allergy_res,
            "irritation": irritation_res,
        }


def predict_personal_care(
    ingredient_name: str,
    primary_function: str,
    ingredient_category: str,
    product_categories: str,
    origin: str,
    regulatory_status: str,
) -> Dict[str, Any]:
    """Convenience functional interface for Personal Care predictions."""
    predictor = PersonalCarePredictor.get_instance()
    return predictor.predict(
        ingredient_name=ingredient_name,
        primary_function=primary_function,
        ingredient_category=ingredient_category,
        product_categories=product_categories,
        origin=origin,
        regulatory_status=regulatory_status,
    )


get_personal_care_predictor = PersonalCarePredictor.get_instance
