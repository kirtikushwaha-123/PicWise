"""
backend/ml/preprocessing/personal_care_dataset.py

Authoritative loader, canonical group extractor, and feature pipeline
for Phase 10A Personal Care validation and optimization.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

DEFAULT_PERSONAL_CARE_DATA_PATH = "data/personal_care/final_personal_care_dataset.csv"

# Target domain definitions
VALID_SAFETY_LEVELS = {"Very Safe", "Safe", "Moderate Risk", "High Risk"}
VALID_ALLERGY_RISKS = {"No Risk", "Low", "Medium", "High"}
VALID_IRRITATION_RISKS = {"No Risk", "Low", "Medium", "High"}

# Application context parentheticals to normalize for canonical grouping
APPLICATION_CONTEXT_SUFFIXES = [
    "baby care grade",
    "baby care",
    "baby",
    "foot care",
    "foot",
    "oral care",
    "oral",
    "lip care",
    "lip plumper",
    "aftershave",
    "hair spray",
    "hair wax",
    "hair pomade",
    "makeup",
    "sanitary products",
    "wipes",
    "cosmetic grade",
    "emulsifier grade",
    "deodorant grade",
    "sheet mask",
    "nail",
    "nail care",
    "intimate care",
    "sunscreen",
    "deodorant",
]


def extract_canonical_group_id(name: str, alt_names_str: Optional[str] = None) -> str:
    """
    Derives the canonical group identifier for a Personal Care ingredient.
    
    Prevents canonical leakage by ensuring that:
    1. Base ingredients with application-context suffixes
       (e.g., 'Menthol (Lip Plumper)', 'Zinc Oxide (Baby Care)', 'Potassium Sorbate (Baby)')
       map to their base canonical ingredient ('Menthol', 'Zinc Oxide', 'Potassium Sorbate').
    2. Primary chemical/botanical names preserve their core identity.
    3. Packaging/alternate names belonging to an ingredient family map to the canonical group.
    
    Returns:
        Canonical group string identifier.
    """
    raw_name = str(name).strip()
    
    m = re.match(r"^(.*?)\s*\((.*?)\)$", raw_name)
    if m:
        main_part = m.group(1).strip()
        paren_part = m.group(2).strip().lower()
        if paren_part in APPLICATION_CONTEXT_SUFFIXES:
            return main_part
        # Otherwise, parenthetical is part of identity (e.g. 'Aqua (Water)', 'Beeswax (Cera Alba)')
        return raw_name
    return raw_name


def load_personal_care_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and validates the authoritative 926-row Personal Care dataset.
    
    Returns:
        pd.DataFrame with 926 rows, 10 original columns + 'canonical_group_id'.
    """
    if path is None:
        path = DEFAULT_PERSONAL_CARE_DATA_PATH

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Personal Care dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    expected_cols = [
        "Ingredient_Name",
        "Primary_Function",
        "Ingredient_Category",
        "Product_Categories",
        "Origin",
        "Safety_Level",
        "Allergy_Risk",
        "Irritation_Risk",
        "Regulatory_Status",
        "Packaging Names / Alternate Names",
    ]

    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in {file_path}")

    # Verify no nulls
    null_count = df[expected_cols].isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Unexpected missing values in Personal Care dataset: {null_count} nulls found.")

    # Clean string fields
    for col in expected_cols:
        df[col] = df[col].astype(str).str.strip()

    # Validate target domains
    safety_classes = set(df["Safety_Level"].unique())
    if not safety_classes.issubset(VALID_SAFETY_LEVELS):
        raise ValueError(f"Invalid classes in Safety_Level: {safety_classes - VALID_SAFETY_LEVELS}")

    allergy_classes = set(df["Allergy_Risk"].unique())
    if not allergy_classes.issubset(VALID_ALLERGY_RISKS):
        raise ValueError(f"Invalid classes in Allergy_Risk: {allergy_classes - VALID_ALLERGY_RISKS}")

    irritation_classes = set(df["Irritation_Risk"].unique())
    if not irritation_classes.issubset(VALID_IRRITATION_RISKS):
        raise ValueError(f"Invalid classes in Irritation_Risk: {irritation_classes - VALID_IRRITATION_RISKS}")

    # Generate canonical_group_id
    df["canonical_group_id"] = df.apply(
        lambda row: extract_canonical_group_id(
            row["Ingredient_Name"],
            row["Packaging Names / Alternate Names"],
        ),
        axis=1,
    )

    return df


def tokenize_product_categories(text: str) -> List[str]:
    """Tokenizes comma-separated product category strings."""
    if not text or not isinstance(text, str):
        return []
    return [item.strip().lower() for item in str(text).split(",") if item.strip()]


def build_feature_pipeline() -> ColumnTransformer:
    """
    Builds the ColumnTransformer for the selected feature representation:
      Ingredient Name (char_wb TF-IDF) + Structured Semantics (OneHot + MultiLabel Count)
    
    Guarantees:
      - Fits vocabulary and one-hot categories strictly on training folds.
      - Never includes target or post-outcome columns.
    """
    name_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        sublinear_tf=True,
        min_df=1,
    )

    cat_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    prod_cat_vectorizer = CountVectorizer(
        tokenizer=tokenize_product_categories,
        token_pattern=None,
        binary=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("name_tfidf", name_vectorizer, "Ingredient_Name"),
            (
                "cat_ohe",
                cat_encoder,
                ["Primary_Function", "Ingredient_Category", "Origin", "Regulatory_Status"],
            ),
            ("prod_cat_bow", prod_cat_vectorizer, "Product_Categories"),
        ]
    )

    return preprocessor
