"""
backend/services/comparison_service/comparator.py

Core implementation for the PicWise Food-Only Product Comparison Service.
Pure functional comparison:
- Does NOT rerun OCR or ML inference.
- Does NOT declare an overall winner.
- Does NOT create a combined score.
- Does NOT rank products from best to worst.
- Preserves independent dimensions (Food Safety, Allergy Risk, Nutrition).
- Keeps unavailable values unavailable.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

from backend.services.comparison_service.models import (
    ComparedProduct,
    ComparisonRow,
    FoodComparisonResult,
)
from backend.services.food_status_service import (
    map_food_safety_status,
    map_allergy_status,
    map_nutrition_status,
    VALID_STATUSES,
)


def validate_comparison_input(data: Any) -> Tuple[bool, Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Validates input payload for food product comparison.

    Parameters:
        data: Parsed JSON request body.

    Returns:
        (is_valid, error_message, validated_products_list)
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object.", None

    if "products" not in data:
        return False, "Missing 'products' field in request body.", None

    products = data.get("products")
    if not isinstance(products, list):
        return False, "'products' must be a list of product objects.", None

    if len(products) < 2:
        return False, f"At least 2 products are required for comparison (received {len(products)}).", None

    if len(products) > 4:
        return False, f"At most 4 products can be compared at once (received {len(products)}).", None

    validated: List[Dict[str, Any]] = []
    for idx, item in enumerate(products):
        if not isinstance(item, dict):
            return False, f"Product at index {idx} must be an object with 'name' and 'analysis'.", None

        name = item.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            return False, f"Product at index {idx} must have a non-empty 'name'.", None

        analysis = item.get("analysis")
        if not isinstance(analysis, dict):
            return False, f"Product '{name.strip()}' must have an 'analysis' object.", None

        # Category validation: strictly food products only
        cat = analysis.get("category")
        if cat is not None and isinstance(cat, str) and cat.strip().lower() != "food":
            return (
                False,
                f"Product comparison only supports food products. Product '{name.strip()}' has category '{cat}'.",
                None,
            )

        if "personal_care" in analysis and "food_safety" not in analysis:
            return (
                False,
                f"Product comparison only supports food products. Product '{name.strip()}' appears to be a personal care product.",
                None,
            )

        validated.append({
            "name": name.strip(),
            "analysis": analysis,
        })

    return True, None, validated


def compare_food_products(products_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares 2–4 already analyzed food products across independent dimensions.

    Parameters:
        products_data: Validated list of dicts with 'name' and 'analysis'.

    Returns:
        Dict[str, Any]: Serialized FoodComparisonResult with products and rows.
    """
    compared_products: List[ComparedProduct] = []

    for item in products_data:
        name = item["name"]
        # Deep copy analysis to ensure caller data is never mutated
        analysis = copy.deepcopy(item["analysis"])

        # 1. Food Safety Dimension Extraction
        fs_pres = (analysis.get("presentation") or {}).get("food_safety")
        if isinstance(fs_pres, dict) and fs_pres.get("status") in VALID_STATUSES:
            fs_status = fs_pres["status"]
            fs_label = fs_pres.get("label")
            fs_risk = fs_pres.get("risk_class")
        else:
            fs_data = analysis.get("food_safety") or {}
            raw_rc = fs_data.get("risk_class") if isinstance(fs_data, dict) else None
            mapped = map_food_safety_status(raw_rc)
            fs_status = mapped.status
            fs_label = mapped.label
            fs_risk = mapped.risk_class

        food_safety_dict = {
            "status": fs_status,
            "label": fs_label,
        }
        if fs_risk is not None:
            food_safety_dict["risk_class"] = fs_risk

        # 2. Allergy Risk Dimension Extraction
        al_pres = (analysis.get("presentation") or {}).get("allergy")
        if isinstance(al_pres, dict) and al_pres.get("status") in VALID_STATUSES:
            al_status = al_pres["status"]
            al_label = al_pres.get("label")
            al_risk = al_pres.get("risk_level")
        else:
            al_data = analysis.get("allergy") or {}
            raw_risk = (
                al_data.get("product_risk_level") or al_data.get("risk_level")
                if isinstance(al_data, dict)
                else None
            )
            raw_st = al_data.get("status") if isinstance(al_data, dict) else None
            mapped_al = map_allergy_status(raw_risk, raw_status=raw_st)
            al_status = mapped_al.status
            al_label = mapped_al.label
            al_risk = mapped_al.risk_level

        allergy_dict = {
            "status": al_status,
            "label": al_label,
        }
        if al_risk is not None:
            allergy_dict["risk_level"] = al_risk

        # 3. Nutrition Dimension Extraction
        nut_pres = (analysis.get("presentation") or {}).get("nutrition")
        if isinstance(nut_pres, dict) and nut_pres.get("status") in VALID_STATUSES:
            nut_status = nut_pres["status"]
            nut_label = nut_pres.get("label")
        else:
            nut_data = analysis.get("nutrition") or {}
            raw_score = nut_data.get("nutrition_score") if isinstance(nut_data, dict) else None
            raw_st = nut_data.get("status") if isinstance(nut_data, dict) else None
            mapped_nut = map_nutrition_status(raw_score, raw_status=raw_st)
            nut_status = mapped_nut.status
            nut_label = mapped_nut.label

        nutrition_dict = {
            "status": nut_status,
            "label": nut_label,
        }

        # 4. Nutrition Score Extraction (Numerical 0..100 or None)
        score_val = None
        if isinstance(nut_pres, dict) and nut_pres.get("score") is not None:
            score_val = nut_pres.get("score")
        elif isinstance(analysis.get("nutrition"), dict):
            score_val = analysis["nutrition"].get("nutrition_score")
            if score_val is None:
                score_val = analysis["nutrition"].get("score")

        if (
            score_val is not None
            and not isinstance(score_val, bool)
            and isinstance(score_val, (int, float))
        ):
            nutrition_score = round(float(score_val), 1)
        else:
            nutrition_score = None

        compared_products.append(
            ComparedProduct(
                name=name,
                food_safety=food_safety_dict,
                allergy=allergy_dict,
                nutrition=nutrition_dict,
                nutrition_score=nutrition_score,
                analysis=analysis,
            )
        )

    # Build row-oriented matrix for convenience
    rows = [
        ComparisonRow(
            dimension="Food Safety",
            key="food_safety",
            values=[
                {
                    "name": p.name,
                    "status": p.food_safety["status"],
                    "label": p.food_safety.get("label"),
                }
                for p in compared_products
            ],
        ),
        ComparisonRow(
            dimension="Allergy Risk",
            key="allergy",
            values=[
                {
                    "name": p.name,
                    "status": p.allergy["status"],
                    "label": p.allergy.get("label"),
                }
                for p in compared_products
            ],
        ),
        ComparisonRow(
            dimension="Nutrition",
            key="nutrition",
            values=[
                {
                    "name": p.name,
                    "status": p.nutrition["status"],
                    "label": p.nutrition.get("label"),
                }
                for p in compared_products
            ],
        ),
        ComparisonRow(
            dimension="Nutrition Score",
            key="nutrition_score",
            values=[
                {
                    "name": p.name,
                    "score": p.nutrition_score,
                    "status": p.nutrition["status"],
                }
                for p in compared_products
            ],
        ),
    ]

    result = FoodComparisonResult(
        success=True,
        products=compared_products,
        rows=rows,
    )
    return result.to_dict()
