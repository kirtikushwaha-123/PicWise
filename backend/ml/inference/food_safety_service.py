import json
import os
import sys
import threading
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

DEFAULT_FOOD_SAFETY_MODEL_DIR = "backend/ml/models/food_safety"
CANONICAL_CLASSES = ["Very Safe", "Safe", "Moderate Risk", "High Risk"]


class FoodSafetyPredictor:
    """
    Production inference service for the PicWise Food Safety model.
    Dual-stream: Character TF-IDF (3-5 n-grams) + MiniLM-L6-v2 (384-d L2 normalized)
    + Balanced Logistic Regression.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_dir=DEFAULT_FOOD_SAFETY_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.vectorizer = None
        self.classifier = None
        self.metadata = None
        self.embedder = None
        self.class_order = CANONICAL_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
        self.idx_to_class = {i: c for i, c in enumerate(CANONICAL_CLASSES)}
        self._loaded = False
        self._load_artifacts()

    @classmethod
    def get_instance(cls, model_dir=DEFAULT_FOOD_SAFETY_MODEL_DIR):
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_dir=model_dir)
        return cls._instance

    def _load_artifacts(self):
        """Loads serialized production model artifacts and verifies configuration integrity."""
        vectorizer_path = self.model_dir / "vectorizer.joblib"
        classifier_path = self.model_dir / "classifier.joblib"
        metadata_path = self.model_dir / "model_metadata.json"

        if not (vectorizer_path.exists() and classifier_path.exists() and metadata_path.exists()):
            raise FileNotFoundError(
                f"Production Food Safety artifacts not found under {self.model_dir}. "
                f"Ensure train_food_safety.py has been executed."
            )

        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Validate classifier classes mapping against canonical classes
        classifier_classes = list(self.classifier.classes_)
        expected_classes = [0, 1, 2, 3]
        if classifier_classes != expected_classes:
            raise ValueError(
                f"Classifier classes mismatch! Found: {classifier_classes}, expected: {expected_classes}"
            )

        # Load frozen semantic embedding model
        model_name = self.metadata.get("semantic_embeddings", {}).get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedder = SentenceTransformer(model_name)
        self._loaded = True

    def predict(self, ingredient_name: str) -> dict:
        """
        Executes Food Safety inference on an extracted or canonical ingredient name.

        Returns:
            dict: {
                "ingredient": str,
                "risk_class": str,
                "confidence": float,
                "probabilities": {
                    "Very Safe": float,
                    "Safe": float,
                    "Moderate Risk": float,
                    "High Risk": float
                }
            }
        """
        if not ingredient_name or not str(ingredient_name).strip():
            return {
                "ingredient": str(ingredient_name or ""),
                "risk_class": None,
                "confidence": 0.0,
                "probabilities": {c: 0.0 for c in self.class_order},
            }

        cleaned_name = str(ingredient_name).strip()

        # 1. Feature Stream 1: Character TF-IDF (3, 5 n-grams)
        X_tfidf = self.vectorizer.transform([cleaned_name])

        # 2. Feature Stream 2: MiniLM Semantic Embeddings (384-d, L2 normalized)
        X_emb = self.embedder.encode(
            [cleaned_name],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # 3. Combine Streams
        X_combined = hstack([X_tfidf, csr_matrix(X_emb)])

        # 4. Logistic Regression Prediction
        probs = self.classifier.predict_proba(X_combined)[0]

        # 5. Build strict, validated class-to-probability mapping
        probabilities = {}
        for cls_idx, prob in zip(self.classifier.classes_, probs):
            class_name = self.idx_to_class[int(cls_idx)]
            probabilities[class_name] = round(float(prob), 4)

        # Argmax prediction and confidence
        top_idx = int(np.argmax(probs))
        predicted_class = self.idx_to_class[int(self.classifier.classes_[top_idx])]
        confidence = round(float(probs[top_idx]), 4)

        return {
            "ingredient": cleaned_name,
            "risk_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
        }


def predict_food_safety(ingredient_name: str, model_dir=DEFAULT_FOOD_SAFETY_MODEL_DIR) -> dict:
    """
    Public functional interface for Food Safety prediction.
    """
    predictor = FoodSafetyPredictor.get_instance(model_dir=model_dir)
    return predictor.predict(ingredient_name)
