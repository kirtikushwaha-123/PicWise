import csv
import os
import re
from pathlib import Path

from backend import config

DEFAULT_FOOD_DATA_PATH = config.DEFAULT_FOOD_DATA_PATH
FALLBACK_FOOD_DATA_PATH = config.FALLBACK_FOOD_DATA_PATH

DEFAULT_PERSONAL_CARE_DATA_PATH = config.DEFAULT_PERSONAL_CARE_DATA_PATH
FALLBACK_PERSONAL_CARE_DATA_PATH = config.FALLBACK_PERSONAL_CARE_DATA_PATH

DEFAULT_NUTRITION_DATA_PATH = config.DEFAULT_NUTRITION_DATA_PATH

REQUIRED_FOOD_COLUMNS = [
    "Ingredient Name",
    "Category",
    "Safety Level",
    "Allergy Risk",
    "Health Impact",
    "Processing Level",
    "Regulatory Status",
    "Packaging Names / Alternate Names",
]

REQUIRED_PERSONAL_CARE_COLUMNS = [
    "Ingredient_Name",
    "Primary_Function",
    "Ingredient_Category",
    "Product_Categories",
    "Origin",
    "Safety_Level",
    "Allergy_Risk",
    "Irritation_Risk",
    "Regulatory_Status",
]

REQUIRED_NUTRITION_COLUMNS = [
    "Nutrient",
    "Health Role",
    "Health Impact",
    "Decision Priority",
    "Better Direction",
    "Alternative / Packaging Names",
]


class KnowledgeBase:
    def __init__(self, food, nutrition, personal_care, warnings=None):
        self.food = food
        self.nutrition = nutrition
        self.personal_care = personal_care
        self.warnings = warnings or []

        self.food_canonical_index, self.food_alternate_index = _build_split_indexes(
            food, "Ingredient Name", "Packaging Names / Alternate Names"
        )
        self.personal_care_canonical_index, self.personal_care_alternate_index = _build_split_indexes(
            personal_care, "Ingredient_Name", "Packaging Names / Alternate Names"
        )
        self.nutrition_canonical_index, self.nutrition_alternate_index = _build_split_indexes(
            nutrition, "Nutrient", "Alternative / Packaging Names"
        )

        # Unified indexes for fast fallback
        self.food_index = {**self.food_alternate_index, **self.food_canonical_index}
        self.personal_care_index = {**self.personal_care_alternate_index, **self.personal_care_canonical_index}
        self.nutrition_index = {**self.nutrition_alternate_index, **self.nutrition_canonical_index}

    @classmethod
    def from_env(cls):
        food_path = _resolve_path("FOOD_DATA_PATH", DEFAULT_FOOD_DATA_PATH, FALLBACK_FOOD_DATA_PATH)
        personal_care_path = _resolve_path("PERSONAL_CARE_DATA_PATH", DEFAULT_PERSONAL_CARE_DATA_PATH, FALLBACK_PERSONAL_CARE_DATA_PATH)
        nutrition_path = os.getenv("NUTRITION_DATA_PATH", DEFAULT_NUTRITION_DATA_PATH)

        warnings = []
        food = _load_table(food_path, warnings, "food")
        nutrition = _load_table(nutrition_path, warnings, "nutrition")
        personal_care = _load_table(personal_care_path, warnings, "personal care")

        return cls(food, nutrition, personal_care, warnings)

    def generate_quality_report(self):
        """
        Generates a structured dataset quality & validation report for all three datasets.
        """
        report = {}
        report["food"] = _audit_records(self.food, "Food", "Ingredient Name", "Packaging Names / Alternate Names", REQUIRED_FOOD_COLUMNS)
        report["personal_care"] = _audit_records(self.personal_care, "Personal Care", "Ingredient_Name", "Packaging Names / Alternate Names", REQUIRED_PERSONAL_CARE_COLUMNS)
        report["nutrition"] = _audit_records(self.nutrition, "Nutrition", "Nutrient", "Alternative / Packaging Names", REQUIRED_NUTRITION_COLUMNS)
        report["warnings"] = self.warnings
        return report

    def get_food_ingredient(self, query):
        """
        Retrieves canonical food ingredient record matching canonical name or alias.
        Returns the full row dict if found, else None.
        """
        if not query or not isinstance(query, str):
            return None
        norm_key = normalize_value(query)
        return self.food_index.get(norm_key)

    def lookup_food_allergy_risk(self, query):
        """
        Returns the raw Allergy Risk ('No Risk', 'Low', 'Medium', 'High')
        for the given food ingredient query (canonical name or alias), or None if not found.
        """
        row = self.get_food_ingredient(query)
        if row:
            val = row.get("Allergy Risk")
            if val is not None and str(val).strip():
                clean_val = str(val).strip()
                if clean_val.lower() == "none":
                    return "No Risk"
                return clean_val
        return None


def normalize_value(value):
    """
    Safely normalizes input text for lookup matching.
    Strips whitespace, converts to lowercase, handles trailing punctuation/commas/semicolons.
    Does NOT strip chemical names or numbers inside words.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    # Replace trailing punctuation or quotes
    text = re.sub(r"^[,\.\;\"\']+|[,\.\;\"\']+$", "", text).strip()
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Remove characters other than alnum and spaces for key normalization
    return "".join(c for c in text if c.isalnum() or c.isspace()).strip()


def _resolve_path(env_var, primary_default, fallback):
    val = os.getenv(env_var)
    if val and Path(val).exists():
        return val
    if Path(primary_default).exists():
        return primary_default
    if Path(fallback).exists():
        return fallback
    return primary_default


def _build_split_indexes(rows, primary_field, alternate_field):
    canonical_index = {}
    alternate_index = {}

    for row in rows:
        primary_val = row.get(primary_field)
        if primary_val:
            key = normalize_value(primary_val)
            if key and key not in canonical_index:
                canonical_index[key] = row

        if alternate_field:
            alt_val = row.get(alternate_field)
            if alt_val:
                for alt_item in str(alt_val).split(";"):
                    alt_key = normalize_value(alt_item)
                    if alt_key and alt_key not in canonical_index and alt_key not in alternate_index:
                        alternate_index[alt_key] = row

    return canonical_index, alternate_index


def _load_table(path_value, warnings, label):
    path = Path(path_value)
    if not path.exists():
        warnings.append(f"{label.title()} knowledge base not loaded: file not found at {path}")
        return []

    if path.suffix.lower() == ".csv":
        return _load_csv(path, warnings, label)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return _load_xlsx(path, warnings, label)

    warnings.append(f"{label.title()} knowledge base has unsupported format: {path}")
    return []


def _load_csv(path, warnings, label):
    try:
        records = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                clean_row = {}
                for key, val in row.items():
                    if key is None:
                        continue
                    k_clean = str(key).strip()
                    if val is None or str(val).strip() == "":
                        clean_row[k_clean] = None  # Genuine missing value is Python None
                    else:
                        clean_row[k_clean] = str(val).strip()  # Explicit value preserved
                records.append(clean_row)
        return records
    except Exception as e:
        warnings.append(f"Failed to load CSV {path.name}: {str(e)}")
        return []


def _load_xlsx(path, warnings, label):
    try:
        from openpyxl import load_workbook
    except ImportError:
        warnings.append(f"{label.title()} knowledge base requires openpyxl to read {path.name}.")
        return []

    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        rows = sheet.iter_rows(values_only=True)
        headers = [str(value).strip() if value is not None else "" for value in next(rows)]
        records = []
        for row in rows:
            clean_row = {}
            for idx, val in enumerate(row):
                if idx < len(headers) and headers[idx]:
                    k_clean = headers[idx]
                    if val is None or str(val).strip() == "":
                        clean_row[k_clean] = None  # Genuine missing value is Python None
                    else:
                        clean_row[k_clean] = str(val).strip()  # Explicit value preserved
            if any(v is not None for v in clean_row.values()):
                records.append(clean_row)
        workbook.close()
        return records
    except Exception as e:
        warnings.append(f"Failed to load XLSX {path.name}: {str(e)}")
        return []


def _audit_records(records, dataset_name, primary_field, alternate_field, required_columns):
    if not records:
        return {
            "name": dataset_name,
            "row_count": 0,
            "column_count": 0,
            "required_columns_present": False,
            "missing_required_columns": required_columns,
            "empty_canonical_names": 0,
            "duplicate_canonical_names": 0,
            "duplicate_normalized_names": 0,
            "allergy_risk_distribution": {},
            "true_missing_values_count": 0
        }

    actual_columns = list(records[0].keys()) if records else []
    missing_req = [c for c in required_columns if c not in actual_columns]

    row_count = len(records)
    col_count = len(actual_columns)
    empty_canonical = 0
    canonical_names = []
    normalized_names = []
    true_missing_count = 0
    allergy_dist = {}

    for row in records:
        primary_val = row.get(primary_field)
        if not primary_val:
            empty_canonical += 1
        else:
            canonical_names.append(primary_val)
            normalized_names.append(normalize_value(primary_val))

        # Check allergy risk distribution if column exists
        for col, val in row.items():
            if val is None:
                true_missing_count += 1
            if col.lower() in ["allergy risk", "allergy_risk"]:
                val_key = str(val) if val is not None else "Missing (Null)"
                allergy_dist[val_key] = allergy_dist.get(val_key, 0) + 1

    dup_canonical = len(canonical_names) - len(set(canonical_names))
    dup_normalized = len(normalized_names) - len(set(normalized_names))

    return {
        "name": dataset_name,
        "row_count": row_count,
        "column_count": col_count,
        "required_columns_present": len(missing_req) == 0,
        "missing_required_columns": missing_req,
        "empty_canonical_names": empty_canonical,
        "duplicate_canonical_names": dup_canonical,
        "duplicate_normalized_names": dup_normalized,
        "allergy_risk_distribution": allergy_dist,
        "true_missing_values_count": true_missing_count
    }
