"""
backend/services/nutrition_service/normalization.py

Normalization and unit conversion for the PicWise Nutrition Scoring Engine.
Consumes raw or structured OCR nutrition tables, standardizes units to per-100g,
and strictly separates explicit zero from missing data.
"""

import re
from typing import Dict, Any, Optional, Tuple
from backend.services.nutrition_service.constants import (
    SALT_TO_SODIUM_FACTOR,
    KJ_TO_KCAL_DIVISOR,
)
from backend.services.nutrition_service.models import NormalizedNutrient

# Canonical key mappings for common OCR variations
KEY_MAPPINGS = {
    # Energy
    "energy": "energy",
    "calories": "energy",
    "energy_kcal": "energy",
    "energy_kj": "energy_kj",
    # Protein
    "protein": "protein",
    "proteins": "protein",
    "total_protein": "protein",
    # Carbohydrate & Sugars
    "total_carbohydrate": "total_carbohydrate",
    "carbohydrate": "total_carbohydrate",
    "carbohydrates": "total_carbohydrate",
    "carbs": "total_carbohydrate",
    "total_carb": "total_carbohydrate",
    "total_sugars": "total_sugars",
    "total_sugar": "total_sugars",
    "sugars": "total_sugars",
    "sugar": "total_sugars",
    "added_sugars": "added_sugars",
    "added_sugar": "added_sugars",
    # Fats
    "total_fat": "total_fat",
    "fat": "total_fat",
    "total_fats": "total_fat",
    "fats": "total_fat",
    "saturated_fat": "saturated_fat",
    "sat_fat": "saturated_fat",
    "saturates": "saturated_fat",
    "trans_fat": "trans_fat",
    "trans_fats": "trans_fat",
    "monounsaturated_fat": "monounsaturated_fat",
    "polyunsaturated_fat": "polyunsaturated_fat",
    "mufa": "monounsaturated_fat",
    "pufa": "polyunsaturated_fat",
    # Fiber
    "dietary_fibre": "dietary_fibre",
    "dietary_fiber": "dietary_fibre",
    "fibre": "dietary_fibre",
    "fiber": "dietary_fibre",
    # Sodium & Salt
    "sodium": "sodium",
    "salt": "salt",
    # Other & Micronutrients
    "cholesterol": "cholesterol",
    "calcium": "calcium",
    "iron": "iron",
    "potassium": "potassium",
    "zinc": "zinc",
    "magnesium": "magnesium",
    "vitamin_a": "vitamin_a",
    "vit_a": "vitamin_a",
    "vitamin_c": "vitamin_c",
    "vit_c": "vitamin_c",
    "vitamin_d": "vitamin_d",
    "vit_d": "vitamin_d",
}

BEVERAGE_INDICATORS = [
    r"\bml\b",
    r"\bliter\b",
    r"\blitre\b",
    r"\bfl\s*oz\b",
    r"\bbeverage\b",
    r"\bdrink\b",
    r"\bjuice\b",
    r"\bsoda\b",
    r"\bcola\b",
    r"\btea\b",
    r"\bcoffee\b",
]

CULINARY_FAT_INDICATORS = [
    r"\boil\b",
    r"\bbutter\b",
    r"\bghee\b",
    r"\bshortening\b",
    r"\blard\b",
    r"\bmargarine\b",
]

WATER_INDICATORS = [
    r"\bdrinking\s+water\b",
    r"\bmineral\s+water\b",
    r"\bspring\s+water\b",
    r"\bpurified\s+water\b",
    r"\bpure\s+water\b",
    r"\bdistilled\s+water\b",
]


def detect_product_context(
    product_text: str = "",
    serving_unit: str = "",
) -> Tuple[bool, bool, bool]:
    """
    Detects product context from text and units:
    Returns (is_beverage, is_culinary_fat, is_pure_water).
    """
    text_lower = (product_text or "").lower()
    unit_lower = (serving_unit or "").lower()

    # Beverage detection
    is_beverage = any(re.search(pat, text_lower) for pat in BEVERAGE_INDICATORS) or unit_lower in ("ml", "l", "fl oz")
    
    # Culinary fat detection
    is_culinary_fat = any(re.search(pat, text_lower) for pat in CULINARY_FAT_INDICATORS)
    
    # Pure water detection
    is_pure_water = any(re.search(pat, text_lower) for pat in WATER_INDICATORS)
    if not is_pure_water and text_lower.strip() == "water":
        is_pure_water = True

    return is_beverage, is_culinary_fat, is_pure_water


def normalize_nutrition_data(
    raw_nutrition: Dict[str, Any],
    serving_size_grams: Optional[float] = None,
    product_text: str = "",
) -> Tuple[Dict[str, NormalizedNutrient], Dict[str, Any]]:
    """
    Normalizes raw or structured OCR nutrition data into a dictionary of
    NormalizedNutrient objects keyed by canonical nutrient name.
    
    Returns:
        (normalized_dict, metadata_dict)
    """
    normalized: Dict[str, NormalizedNutrient] = {}
    warnings = []

    if not raw_nutrition or not isinstance(raw_nutrition, dict):
        return normalized, {"warnings": ["No nutrition dictionary provided"]}

    # First pass: extract per_100g or per_serving values for each raw item
    raw_extracted: Dict[str, Dict[str, Any]] = {}
    
    for raw_key, raw_val in raw_nutrition.items():
        canonical_key = _map_to_canonical_key(raw_key)
        if not canonical_key:
            continue
            
        parsed_item = _extract_raw_value_and_unit(raw_val, serving_size_grams)
        if parsed_item is not None:
            # If canonical key already exists, only overwrite if new one has per_100g and existing didn't
            if canonical_key in raw_extracted:
                existing = raw_extracted[canonical_key]
                if existing.get("source") != "per_100g" and parsed_item.get("source") == "per_100g":
                    raw_extracted[canonical_key] = parsed_item
            else:
                raw_extracted[canonical_key] = parsed_item

    # Second pass: Standardize units and apply unit conversion rules
    
    # 1. Energy: handle kJ -> kcal if kcal is missing
    if "energy" in raw_extracted:
        item = raw_extracted["energy"]
        val = item["value"]
        unit = (item["unit"] or "kcal").lower()
        if unit == "kj":
            val = val / KJ_TO_KCAL_DIVISOR
            unit = "kcal"
        normalized["energy"] = NormalizedNutrient(
            name="energy",
            amount_per_100g=round(val, 2) if val is not None else None,
            unit="kcal",
            is_explicit_zero=item["is_explicit_zero"],
            is_missing=item["is_missing"],
            original_value=item["original_value"],
            original_unit=item["original_unit"],
            source=item["source"],
        )
    elif "energy_kj" in raw_extracted:
        item = raw_extracted["energy_kj"]
        val = item["value"] / KJ_TO_KCAL_DIVISOR if item["value"] is not None else None
        normalized["energy"] = NormalizedNutrient(
            name="energy",
            amount_per_100g=round(val, 2) if val is not None else None,
            unit="kcal",
            is_explicit_zero=item["is_explicit_zero"],
            is_missing=item["is_missing"],
            original_value=item["original_value"],
            original_unit=item["original_unit"],
            source="derived_from_kj",
        )

    # 2. Sodium & Salt conversion:
    # Sodium is in mg. Salt is in g. Sodium (mg) = Salt (g) * 400.0
    if "sodium" in raw_extracted:
        item = raw_extracted["sodium"]
        val = item["value"]
        unit = (item["unit"] or "mg").lower()
        if unit == "g":
            val = val * 1000.0
            unit = "mg"
        normalized["sodium"] = NormalizedNutrient(
            name="sodium",
            amount_per_100g=round(val, 2) if val is not None else None,
            unit="mg",
            is_explicit_zero=item["is_explicit_zero"],
            is_missing=item["is_missing"],
            original_value=item["original_value"],
            original_unit=item["original_unit"],
            source=item["source"],
        )
    elif "salt" in raw_extracted:
        item = raw_extracted["salt"]
        val = item["value"]
        unit = (item["unit"] or "g").lower()
        salt_g = val if unit == "g" else (val / 1000.0 if val is not None else None)
        sodium_mg = (salt_g * SALT_TO_SODIUM_FACTOR) if salt_g is not None else None
        normalized["sodium"] = NormalizedNutrient(
            name="sodium",
            amount_per_100g=round(sodium_mg, 2) if sodium_mg is not None else None,
            unit="mg",
            is_explicit_zero=item["is_explicit_zero"],
            is_missing=item["is_missing"],
            original_value=item["original_value"],
            original_unit=item["original_unit"],
            source="derived_from_salt",
        )

    # 3. Standard Macronutrients (in grams)
    standard_g_nutrients = [
        "protein",
        "total_carbohydrate",
        "total_sugars",
        "added_sugars",
        "total_fat",
        "saturated_fat",
        "trans_fat",
        "dietary_fibre",
        "monounsaturated_fat",
        "polyunsaturated_fat",
    ]

    for key in standard_g_nutrients:
        if key in raw_extracted:
            item = raw_extracted[key]
            val = item["value"]
            unit = (item["unit"] or "g").lower()
            if unit == "mg" and val is not None:
                val = val / 1000.0
                unit = "g"
            elif unit == "mcg" and val is not None:
                val = val / 1000000.0
                unit = "g"
                
            normalized[key] = NormalizedNutrient(
                name=key,
                amount_per_100g=round(val, 4) if val is not None else None,
                unit="g",
                is_explicit_zero=item["is_explicit_zero"],
                is_missing=item["is_missing"],
                original_value=item["original_value"],
                original_unit=item["original_unit"],
                source=item["source"],
            )

    # 4. Micronutrients & Minerals/Vitamins
    micronutrient_keys = [
        "calcium", "iron", "potassium", "zinc", "magnesium",
        "vitamin_a", "vitamin_c", "vitamin_d", "cholesterol"
    ]
    for key in micronutrient_keys:
        if key in raw_extracted:
            item = raw_extracted[key]
            val = item["value"]
            unit = item["unit"] or "mg"
            normalized[key] = NormalizedNutrient(
                name=key,
                amount_per_100g=round(val, 4) if val is not None else None,
                unit=unit,
                is_explicit_zero=item["is_explicit_zero"],
                is_missing=item["is_missing"],
                original_value=item["original_value"],
                original_unit=item["original_unit"],
                source=item["source"],
            )

    # Context detection
    is_bev, is_fat, is_water = detect_product_context(product_text)

    metadata = {
        "is_beverage": is_bev,
        "is_culinary_fat": is_fat,
        "is_pure_water": is_water,
        "serving_size_grams": serving_size_grams,
        "warnings": warnings,
    }

    return normalized, metadata


def _map_to_canonical_key(raw_key: str) -> Optional[str]:
    """Normalizes key string to canonical key."""
    k = raw_key.strip().lower().replace(" ", "_").replace("-", "_")
    return KEY_MAPPINGS.get(k) or KEY_MAPPINGS.get(raw_key.strip().lower())


def _extract_raw_value_and_unit(
    raw_val: Any,
    serving_size_grams: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extracts numerical value, unit, and explicit-zero flag from diverse OCR representations.
    Prioritizes 'per_100g' sub-dict if present.
    """
    if raw_val is None:
        return None

    # Case 1: Structured dict from OCR
    if isinstance(raw_val, dict):
        # Prioritize per_100g
        if "per_100g" in raw_val and isinstance(raw_val["per_100g"], dict):
            sub = raw_val["per_100g"]
            val, unit, is_zero, is_missing = _parse_val_unit(sub.get("value"), sub.get("unit"))
            return {
                "value": val,
                "unit": unit,
                "is_explicit_zero": is_zero,
                "is_missing": is_missing,
                "original_value": sub.get("value"),
                "original_unit": sub.get("unit"),
                "source": "per_100g",
            }
        
        # Fallback to top-level value if not serving-specific
        if "value" in raw_val:
            val, unit, is_zero, is_missing = _parse_val_unit(raw_val.get("value"), raw_val.get("unit"))
            return {
                "value": val,
                "unit": unit,
                "is_explicit_zero": is_zero,
                "is_missing": is_missing,
                "original_value": raw_val.get("value"),
                "original_unit": raw_val.get("unit"),
                "source": "top_level",
            }
            
        # Fallback to per_serving if serving_size_grams is known
        if "per_serving" in raw_val and isinstance(raw_val["per_serving"], dict):
            sub = raw_val["per_serving"]
            s_val, s_unit, is_zero, is_missing = _parse_val_unit(sub.get("value"), sub.get("unit"))
            if is_zero:
                return {
                    "value": 0.0,
                    "unit": s_unit,
                    "is_explicit_zero": True,
                    "is_missing": False,
                    "original_value": sub.get("value"),
                    "original_unit": s_unit,
                    "source": "per_serving_zero",
                }
            if s_val is not None and serving_size_grams and serving_size_grams > 0:
                scaled = s_val * (100.0 / serving_size_grams)
                return {
                    "value": scaled,
                    "unit": s_unit,
                    "is_explicit_zero": False,
                    "is_missing": False,
                    "original_value": sub.get("value"),
                    "original_unit": s_unit,
                    "source": "scaled_from_serving",
                }
            return {
                "value": None,
                "unit": s_unit,
                "is_explicit_zero": False,
                "is_missing": True,
                "original_value": sub.get("value"),
                "original_unit": s_unit,
                "source": "per_serving_unscaled",
            }

    # Case 2: Direct float or int
    if isinstance(raw_val, (int, float)):
        val = float(raw_val)
        return {
            "value": val,
            "unit": "",
            "is_explicit_zero": (val == 0.0),
            "is_missing": False,
            "original_value": raw_val,
            "original_unit": "",
            "source": "direct_number",
        }

    # Case 3: Direct string like "5g" or "0g" or "12.5 kcal"
    if isinstance(raw_val, str):
        val, unit, is_zero, is_missing = _parse_val_unit(raw_val, None)
        return {
            "value": val,
            "unit": unit,
            "is_explicit_zero": is_zero,
            "is_missing": is_missing,
            "original_value": raw_val,
            "original_unit": unit,
            "source": "parsed_string",
        }

    return None


def _parse_val_unit(raw_val: Any, explicit_unit: Optional[str] = None) -> Tuple[Optional[float], str, bool, bool]:
    """
    Parses a value and unit pair.
    Returns: (float_value_or_none, unit_str, is_explicit_zero, is_missing)
    """
    if raw_val is None:
        return None, explicit_unit or "", False, True

    if isinstance(raw_val, (int, float)):
        val = float(raw_val)
        if val < 0:
            # Negative values are malformed
            return None, explicit_unit or "", False, True
        return val, explicit_unit or "", (val == 0.0), False

    val_str = str(raw_val).strip().lower()
    if not val_str or val_str in ("-", "na", "n/a", "nil", "none", "unknown"):
        return None, explicit_unit or "", False, True

    # Check for explicit 0
    if val_str in ("0", "0.0", "0g", "0mg", "0.0g", "0.0mg", "0kcal", "0kj", "nil", "0.00"):
        unit = explicit_unit or re.sub(r"[0-9\.\s]", "", val_str)
        return 0.0, unit, True, False

    # Check for dual energy units (kcal and kj), e.g. "110 kcal / 460 kj", "460 kj / 110 kcal", "110 kcal (460 kj)"
    kcal_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*kcal\b", val_str)
    if kcal_match:
        try:
            num = float(kcal_match.group(1))
            return num, "kcal", (num == 0.0), False
        except ValueError:
            pass

    kj_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*kj\b", val_str)
    if kj_match:
        try:
            num = float(kj_match.group(1))
            return num, "kj", (num == 0.0), False
        except ValueError:
            pass

    # Extract numeric and unit
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z%]*)$", val_str)
    if m:
        try:
            num = float(m.group(1))
            unit = explicit_unit or m.group(2)
            if num < 0:
                return None, unit, False, True
            return num, unit, (num == 0.0), False
        except ValueError:
            pass

    # Fallback: slash-separated values without unit, e.g. '110 / 460' or '110/460'
    m_slash = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z%]*)$", val_str)
    if m_slash:
        try:
            num = float(m_slash.group(1))
            unit = explicit_unit or m_slash.group(3)
            return num, unit, (num == 0.0), False
        except ValueError:
            pass

    return None, explicit_unit or "", False, True
