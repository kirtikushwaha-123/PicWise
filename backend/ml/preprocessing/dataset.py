import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

DEFAULT_FOOD_DATA_PATH = "data/food/food_ingredients_dataset_final.csv"
FALLBACK_FOOD_DATA_PATH = "data/food/ingredient_knowledge_base_500_cleaned.csv"

DEFAULT_PERSONAL_CARE_DATA_PATH = "data/personal_care/personal_care_ingredients_dataset_cleaned.xlsx"
FALLBACK_PERSONAL_CARE_DATA_PATH = "data/personal_care/personal_care_ingredients_dataset_csv.xlsx"

DEFAULT_MODEL_DIR = "backend/ml/models"


def normalize_safety_level(val):
    if pd.isna(val) or not str(val).strip():
        return None
    s = str(val).strip().lower()
    if s in ["very safe"]:
        return "Very Safe"
    elif s in ["safe"]:
        return "Safe"
    elif s in ["moderate", "moderate risk"]:
        return "Moderate Risk"
    elif s in ["risky", "high risk"]:
        return "High Risk"
    return str(val).strip()


def normalize_allergy_risk(val):
    if pd.isna(val) or not str(val).strip():
        return "None"
    s = str(val).strip().lower()
    if s in ["none", "no risk", "n/a", "no"]:
        return "None"
    elif s in ["low"]:
        return "Low"
    elif s in ["medium", "moderate"]:
        return "Medium"
    elif s in ["high"]:
        return "High"
    return str(val).strip().title()


def load_raw_datasets(food_path=None, personal_care_path=None):
    if food_path is None:
        food_path = os.getenv("FOOD_DATA_PATH", DEFAULT_FOOD_DATA_PATH)
        if not Path(food_path).exists():
            food_path = FALLBACK_FOOD_DATA_PATH

    if personal_care_path is None:
        personal_care_path = os.getenv("PERSONAL_CARE_DATA_PATH", DEFAULT_PERSONAL_CARE_DATA_PATH)
        if not Path(personal_care_path).exists():
            personal_care_path = FALLBACK_PERSONAL_CARE_DATA_PATH

    rows = []

    # 1. Load Food Dataset
    food_p = Path(food_path)
    if food_p.exists():
        food_df = pd.read_csv(food_p)
        for _, row in food_df.iterrows():
            primary_name = str(row.get("Ingredient Name", "")).strip()
            safety = row.get("Safety Level")
            allergy = row.get("Allergy Risk")
            category = row.get("Category", "")

            if primary_name:
                rows.append({
                    "ingredient_name": primary_name,
                    "canonical_group": primary_name,
                    "is_alternate": False,
                    "domain": "food",
                    "safety_level_raw": safety,
                    "allergy_risk_raw": allergy,
                    "category": category
                })

            # Process alternate names / packaging names
            alt_names = str(row.get("Packaging Names / Alternate Names", "") or "")
            if alt_names and alt_names.lower() != "nan":
                for alt in alt_names.split(";"):
                    alt_clean = alt.strip()
                    if alt_clean and alt_clean.lower() != primary_name.lower():
                        rows.append({
                            "ingredient_name": alt_clean,
                            "canonical_group": primary_name,
                            "is_alternate": True,
                            "domain": "food",
                            "safety_level_raw": safety,
                            "allergy_risk_raw": allergy,
                            "category": category
                        })

    # 2. Load Personal Care Dataset
    pc_p = Path(personal_care_path)
    if pc_p.exists():
        if pc_p.suffix.lower() in [".xlsx", ".xls"]:
            pc_df = pd.read_excel(pc_p)
        else:
            pc_df = pd.read_csv(pc_p)

        for _, row in pc_df.iterrows():
            primary_name = str(row.get("Ingredient_Name", "")).strip()
            safety = row.get("Safety_Level")
            allergy = row.get("Allergy_Risk")
            category = row.get("Ingredient_Category", "")

            if primary_name:
                rows.append({
                    "ingredient_name": primary_name,
                    "canonical_group": primary_name,
                    "is_alternate": False,
                    "domain": "personal_care",
                    "safety_level_raw": safety,
                    "allergy_risk_raw": allergy,
                    "category": category
                })

    df = pd.DataFrame(rows)
    return df


def prepare_unified_dataset(food_path=None, personal_care_path=None):
    df = load_raw_datasets(food_path, personal_care_path)

    if df.empty:
        raise ValueError("No data loaded from datasets!")

    # Apply normalizations
    df["safety_level"] = df["safety_level_raw"].apply(normalize_safety_level)
    df["allergy_risk"] = df["allergy_risk_raw"].apply(normalize_allergy_risk)

    # Filter out rows missing safety_level if any
    df = df.dropna(subset=["safety_level"]).copy()

    # Deduplicate exact duplicate text and targets within same canonical group
    df["ingredient_name_lower"] = df["ingredient_name"].str.strip().str.lower()
    df = df.drop_duplicates(subset=["ingredient_name_lower", "canonical_group", "safety_level", "allergy_risk"]).copy()

    return df


def build_and_group_split_data(food_path=None, personal_care_path=None, test_size=0.2, random_state=42):
    """
    Performs canonical-ingredient group-aware train/test split.
    All representations (canonical and alternate names) belonging to the same canonical ingredient
    are placed exclusively into train or test set.

    TF-IDF Vectorizer is fitted STRICTLY on training set names after splitting.
    """
    df = prepare_unified_dataset(food_path, personal_care_path)

    groups = df["canonical_group"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    # Verify zero group leakage
    train_groups = set(df_train["canonical_group"])
    test_groups = set(df_test["canonical_group"])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise ValueError(f"Data leakage detected! Overlapping canonical groups: {overlap}")

    # Fit TF-IDF Vectorizer ONLY on training names
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(df_train["ingredient_name"])
    X_test = vectorizer.transform(df_test["ingredient_name"])

    # Fit Label Encoders across all unique valid target classes
    safety_classes = ["Very Safe", "Safe", "Moderate Risk", "High Risk"]
    safety_encoder = LabelEncoder()
    safety_encoder.fit(safety_classes)
    y_safety_train = safety_encoder.transform(df_train["safety_level"])
    y_safety_test = safety_encoder.transform(df_test["safety_level"])

    allergy_classes = ["None", "Low", "Medium", "High"]
    allergy_encoder = LabelEncoder()
    allergy_encoder.fit(allergy_classes)
    y_allergy_train = allergy_encoder.transform(df_train["allergy_risk"])
    y_allergy_test = allergy_encoder.transform(df_test["allergy_risk"])

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_safety_train": y_safety_train,
        "y_safety_test": y_safety_test,
        "y_allergy_train": y_allergy_train,
        "y_allergy_test": y_allergy_test,
        "vectorizer": vectorizer,
        "safety_encoder": safety_encoder,
        "allergy_encoder": allergy_encoder,
        "df": df,
        "df_train": df_train,
        "df_test": df_test,
        "train_groups": train_groups,
        "test_groups": test_groups
    }


def build_and_split_data(food_path=None, personal_care_path=None, test_size=0.2, random_state=42, model_dir=DEFAULT_MODEL_DIR):
    """
    Backward-compatible wrapper using group-aware splitting.
    """
    return build_and_group_split_data(
        food_path=food_path,
        personal_care_path=personal_care_path,
        test_size=test_size,
        random_state=random_state
    )
