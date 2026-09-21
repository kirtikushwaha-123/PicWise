import datetime
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.ml.preprocessing.dataset import (
    DEFAULT_FOOD_DATA_PATH,
    FALLBACK_FOOD_DATA_PATH,
    normalize_safety_level,
)

CANONICAL_CLASSES = ["Very Safe", "Safe", "Moderate Risk", "High Risk"]
PRODUCTION_MODEL_DIR = "backend/ml/models/food_safety"

REFERENCE_BENCHMARK_METRICS = {
    "benchmark_type": "Previously established leak-free outer evaluation benchmark",
    "macro_f1": 0.7437,
    "accuracy": 0.7631,
    "balanced_accuracy": 0.7363,
    "roc_auc": 0.9050,
    "high_risk_recall": 0.7391,
    "high_risk_precision": 0.8095,
    "high_risk_f1": 0.7727,
    "ordinal_mae": 0.2651,
    "severe_error_rate": 0.0201,
}


def load_food_safety_dataset(data_path=None):
    """
    Loads and normalizes the approved Food Safety dataset with representations
    (canonical names and alternate / packaging names).
    """
    if data_path is None:
        data_path = os.getenv("FOOD_DATA_PATH", DEFAULT_FOOD_DATA_PATH)
        if not Path(data_path).exists():
            data_path = FALLBACK_FOOD_DATA_PATH

    df_raw = pd.read_csv(data_path)
    rows = []

    for _, row in df_raw.iterrows():
        primary_name = str(row.get("Ingredient Name", "")).strip()
        safety = normalize_safety_level(row.get("Safety Level"))

        if primary_name and safety:
            rows.append({
                "ingredient_name": primary_name,
                "canonical_group": primary_name,
                "is_alternate": False,
                "safety_level": safety,
            })

        alt_names = str(row.get("Packaging Names / Alternate Names", "") or "")
        if alt_names and alt_names.lower() != "nan":
            for alt in alt_names.split(";"):
                alt_clean = alt.strip()
                if alt_clean and alt_clean.lower() != primary_name.lower() and safety:
                    rows.append({
                        "ingredient_name": alt_clean,
                        "canonical_group": primary_name,
                        "is_alternate": True,
                        "safety_level": safety,
                    })

    df = pd.DataFrame(rows)
    df["ingredient_name_lower"] = df["ingredient_name"].str.strip().str.lower()
    df = df.drop_duplicates(
        subset=["ingredient_name_lower", "canonical_group", "safety_level"]
    ).copy().reset_index(drop=True)

    return df


def train_and_serialize_food_safety_model(
    data_path=None,
    output_dir=PRODUCTION_MODEL_DIR,
    random_state=42,
):
    """
    Trains and serializes the frozen Food Safety pipeline:
      - Character TF-IDF (3, 5 n-grams, sublinear_tf=True, min_df=1)
      - Pretrained all-MiniLM-L6-v2 embeddings (384-d, L2 normalized)
      - Balanced Logistic Regression (lbfgs, max_iter=1000, random_state=42)
      - Canonical classes: Very Safe, Safe, Moderate Risk, High Risk
    """
    print(f"[Food Safety Reproduction] Loading dataset from: {data_path or DEFAULT_FOOD_DATA_PATH}")
    df = load_food_safety_dataset(data_path)
    print(f"[Food Safety Reproduction] Total representations: {len(df)}, Canonical groups: {df['canonical_group'].nunique()}")

    class_to_idx = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
    y = df["safety_level"].map(class_to_idx).values

    # 1. Fit Character TF-IDF Vectorizer
    print("[Food Safety Reproduction] Fitting Character TF-IDF (3, 5) vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
    )
    X_tfidf = vectorizer.fit_transform(df["ingredient_name"])
    print(f"[Food Safety Reproduction] TF-IDF feature matrix shape: {X_tfidf.shape}")

    # 2. Extract MiniLM Semantic Embeddings
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[Food Safety Reproduction] Generating embeddings with {embedding_model_name}...")
    embedder = SentenceTransformer(embedding_model_name)
    X_emb = embedder.encode(
        df["ingredient_name"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"[Food Safety Reproduction] Embedding matrix shape: {X_emb.shape}")

    # 3. Combine Feature Streams
    X_combined = hstack([X_tfidf, csr_matrix(X_emb)])
    print(f"[Food Safety Reproduction] Combined feature matrix shape: {X_combined.shape}")

    # 4. Train Balanced Logistic Regression Classifier
    print("[Food Safety Reproduction] Fitting Balanced Logistic Regression classifier...")
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )
    classifier.fit(X_combined, y)

    # Verify class ordering matches canonical classes
    assert list(classifier.classes_) == [0, 1, 2, 3], f"Unexpected classifier classes: {classifier.classes_}"

    # 5. Serialize Production Artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    vectorizer_path = out_path / "vectorizer.joblib"
    classifier_path = out_path / "classifier.joblib"
    metadata_path = out_path / "model_metadata.json"

    print(f"[Food Safety Reproduction] Saving vectorizer to: {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)

    print(f"[Food Safety Reproduction] Saving classifier to: {classifier_path}")
    joblib.dump(classifier, classifier_path)

    metadata = {
        "model_name": "PicWise Food Safety Classifier",
        "model_type": "Dual-stream Character TF-IDF + Pretrained MiniLM + Balanced Logistic Regression",
        "version": "1.0.0",
        "serialization_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": random_state,
        "dataset": {
            "source_path": str(data_path or DEFAULT_FOOD_DATA_PATH),
            "total_representations": int(len(df)),
            "canonical_ingredients": int(df["canonical_group"].nunique()),
            "class_distribution": {
                str(k): int(v) for k, v in df["safety_level"].value_counts().items()
            },
        },
        "character_tfidf": {
            "analyzer": "char",
            "ngram_range": [3, 5],
            "sublinear_tf": True,
            "min_df": 1,
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "semantic_embeddings": {
            "model_name": embedding_model_name,
            "dimension": 384,
            "normalize_embeddings": True,
            "frozen": True,
        },
        "classifier": {
            "type": "LogisticRegression",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": random_state,
        },
        "canonical_classes": CANONICAL_CLASSES,
        "class_index_mapping": class_to_idx,
        "reference_validation_metrics": REFERENCE_BENCHMARK_METRICS,
    }

    print(f"[Food Safety Reproduction] Saving model metadata to: {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[Food Safety Reproduction] Production serialization complete.")
    return {
        "vectorizer_path": str(vectorizer_path),
        "classifier_path": str(classifier_path),
        "metadata_path": str(metadata_path),
        "total_representations": len(df),
        "canonical_ingredients": df["canonical_group"].nunique(),
    }


if __name__ == "__main__":
    train_and_serialize_food_safety_model()
