"""
backend/ml/training/train_personal_care.py

Trains and serializes the frozen Personal Care production models for:
- Safety_Level
- Allergy_Risk
- Irritation_Risk

Uses the authoritative 926-row dataset:
data/personal_care/final_personal_care_dataset.csv

Configuration frozen in Phase 10A:
- Character-level TF-IDF (char_wb, n-grams 2-5, sublinear_tf=True, min_df=1)
- Structured Semantics (OneHotEncoder for 4 categoricals + CountVectorizer for Product_Categories)
- LogisticRegression(C=10.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)
"""

import datetime
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.ml.preprocessing.personal_care_dataset import (
    DEFAULT_PERSONAL_CARE_DATA_PATH,
    build_feature_pipeline,
    load_personal_care_dataset,
)

PRODUCTION_MODELS_DIR = PROJECT_ROOT / "backend" / "ml" / "models" / "personal_care"

TARGETS_CONFIG = {
    "Safety_Level": {
        "dir_name": "safety",
        "canonical_classes": ["High Risk", "Moderate Risk", "Safe", "Very Safe"],
    },
    "Allergy_Risk": {
        "dir_name": "allergy",
        "canonical_classes": ["High", "Low", "Medium", "No Risk"],
    },
    "Irritation_Risk": {
        "dir_name": "irritation",
        "canonical_classes": ["High", "Low", "Medium", "No Risk"],
    },
}


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def train_and_serialize_personal_care_models(
    data_path: str = DEFAULT_PERSONAL_CARE_DATA_PATH,
    output_base_dir: Path = PRODUCTION_MODELS_DIR,
    random_state: int = 42,
) -> dict:
    """
    Fits and serializes the frozen production pipelines for all 3 Personal Care targets.
    """
    data_file = PROJECT_ROOT / data_path if not Path(data_path).is_absolute() else Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Authoritative dataset not found at {data_file}")

    dataset_hash = compute_file_sha256(data_file)
    df = load_personal_care_dataset(str(data_file))

    total_rows = len(df)
    canonical_groups = int(df["canonical_group_id"].nunique())
    alt_count = int((df["Packaging Names / Alternate Names"] != "No Alternate Names").sum())

    results = {}

    for target, conf in TARGETS_CONFIG.items():
        dir_name = conf["dir_name"]
        target_dir = output_base_dir / dir_name
        target_dir.mkdir(parents=True, exist_ok=True)

        y = df[target].values
        classes_present = sorted(list(set(y)))
        expected_classes = sorted(conf["canonical_classes"])
        if classes_present != expected_classes:
            raise ValueError(
                f"Target {target} classes mismatch! Found {classes_present}, expected {expected_classes}"
            )

        # Frozen feature pipeline and classifier
        pipeline = Pipeline([
            ("preprocessor", build_feature_pipeline()),
            (
                "classifier",
                LogisticRegression(
                    C=10.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ])

        FEATURE_COLUMNS = [
            "Ingredient_Name",
            "Primary_Function",
            "Ingredient_Category",
            "Product_Categories",
            "Origin",
            "Regulatory_Status",
        ]
        pipeline.fit(df[FEATURE_COLUMNS], y)

        # Compute feature dimensions
        preprocessor = pipeline.named_steps["preprocessor"]
        name_feats = len(preprocessor.named_transformers_["name_tfidf"].get_feature_names_out())
        cat_feats = len(preprocessor.named_transformers_["cat_ohe"].get_feature_names_out())
        prod_feats = len(preprocessor.named_transformers_["prod_cat_bow"].get_feature_names_out())
        total_feats = name_feats + cat_feats + prod_feats

        # Save pipeline artifact
        pipeline_path = target_dir / "pipeline.joblib"
        joblib.dump(pipeline, pipeline_path, compress=3)

        # Save metadata
        metadata = {
            "domain": "personal_care",
            "target": target,
            "target_column": target,
            "target_dir": dir_name,
            "model_type": "LogisticRegression",
            "C": 10.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": random_state,
            "hyperparameters": {
                "C": 10.0,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": random_state,
            },
            "text_analyzer": "char_wb",
            "ngram_range": [2, 5],
            "sublinear_tf": True,
            "min_df": 1,
            "dataset_rows": total_rows,
            "canonical_groups": canonical_groups,
            "alternate_representations": alt_count,
            "classes": list(pipeline.named_steps["classifier"].classes_),
            "feature_dimensions": {
                "name_tfidf": name_feats,
                "cat_ohe": cat_feats,
                "prod_cat_bow": prod_feats,
                "total": total_feats,
            },
            "dataset_sha256": dataset_hash,
            "source_dataset_path": str(data_path),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "phase": "10B",
            "phase10A_experiment_reference": "C=10.0_cw=balanced (Mean Macro F1: 0.7273, Dangerous Confusions: 17)",
        }

        metadata_path = target_dir / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        results[target] = {
            "pipeline_path": str(pipeline_path),
            "metadata_path": str(metadata_path),
            "total_features": total_feats,
            "classes": metadata["classes"],
        }
        print(f"[SUCCESS] Serialized {target} model to {pipeline_path} ({total_feats} features)")

    return results


if __name__ == "__main__":
    train_and_serialize_personal_care_models()
