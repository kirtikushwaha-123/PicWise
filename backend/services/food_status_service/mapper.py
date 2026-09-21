"""
backend/services/food_status_service/mapper.py

Deterministic status mapping functions for the PicWise Food Analysis presentation layer (Phase 9H).

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. No overall product health score, no overall product color, and no weighting/averaging across dimensions.
2. No new Food Safety product-level aggregation methodology (no determine_product_safety_risk).
3. Exact mappings:
   - Food Safety: Very Safe -> green, Safe -> yellow (NEVER green), Moderate Risk -> orange, High Risk -> red
   - Allergy Risk: No Risk -> green, Low -> yellow, Medium -> orange, High -> red
   - Nutrition: 0-25 -> red (Low Nutrition), 26-50 -> orange (Slightly Better Nutrition),
                51-75 -> yellow (Better Nutrition), 76-100 -> green (Good Nutrition)
4. Missing != Zero: Missing nutrition score produces status='unavailable', never red or 0.
5. Unknown != No Risk: Missing or insufficient allergy data produces status='unavailable', never green.
"""

from typing import Any, Dict, Optional, Union

from backend.services.food_status_service.constants import (
    ALLERGY_STATUS_MAP,
    ALLERGY_UI_LABEL_MAP,
    FOOD_SAFETY_LABEL_MAP,
    FOOD_SAFETY_STATUS_MAP,
    STATUS_GREEN,
    STATUS_ORANGE,
    STATUS_RED,
    STATUS_UNAVAILABLE,
    STATUS_YELLOW,
)
from backend.services.food_status_service.models import (
    AllergyPresentation,
    FoodAnalysisPresentation,
    FoodSafetyPresentation,
    NutritionPresentation,
)


def map_food_safety_status(risk_class: Optional[str]) -> FoodSafetyPresentation:
    """
    Maps a Food Safety ML risk class to presentation status.

    Parameters:
        risk_class: "Very Safe" | "Safe" | "Moderate Risk" | "High Risk" | None

    Returns:
        FoodSafetyPresentation:
            - Very Safe       -> green  (label: "Very Safe")
            - Safe            -> yellow (label: "Safe") -- strictly yellow, never green!
            - Moderate Risk   -> orange (label: "Moderate Risk")
            - High Risk       -> red    (label: "High Risk")
            - None/Invalid    -> unavailable (label: None)
    """
    if not risk_class or not isinstance(risk_class, str):
        return FoodSafetyPresentation(
            status=STATUS_UNAVAILABLE,
            label=None,
            risk_class=None,
        )

    cleaned = risk_class.strip()
    if cleaned in FOOD_SAFETY_STATUS_MAP:
        return FoodSafetyPresentation(
            status=FOOD_SAFETY_STATUS_MAP[cleaned],
            label=FOOD_SAFETY_LABEL_MAP[cleaned],
            risk_class=cleaned,
        )

    return FoodSafetyPresentation(
        status=STATUS_UNAVAILABLE,
        label=None,
        risk_class=None,
    )


def map_allergy_status(
    risk_level: Optional[str],
    raw_status: Optional[str] = None,
) -> AllergyPresentation:
    """
    Maps an Allergy Risk level to presentation status.

    Parameters:
        risk_level: "No Risk" | "Low" | "Medium" | "High" | None
        raw_status: Execution status of allergy service (e.g. "unavailable", "insufficient_data")

    Returns:
        AllergyPresentation:
            - No Risk                 -> green  (label: "Allergen-Free")
            - Low                     -> yellow (label: "Low Allergy Risk")
            - Medium                  -> orange (label: "Moderate Allergy Risk")
            - High                    -> red    (label: "High Allergy Risk")
            - unavailable/insufficient -> unavailable (label: None) -- never green!
    """
    # Guard: if pipeline explicitly reported unavailable or insufficient data
    if raw_status in ("unavailable", "insufficient_data", "skipped", "error"):
        return AllergyPresentation(
            status=STATUS_UNAVAILABLE,
            label=None,
            risk_level=None,
        )

    if not risk_level or not isinstance(risk_level, str):
        return AllergyPresentation(
            status=STATUS_UNAVAILABLE,
            label=None,
            risk_level=None,
        )

    cleaned = risk_level.strip()
    if cleaned in ALLERGY_STATUS_MAP:
        return AllergyPresentation(
            status=ALLERGY_STATUS_MAP[cleaned],
            label=ALLERGY_UI_LABEL_MAP[cleaned],
            risk_level=cleaned,
        )

    return AllergyPresentation(
        status=STATUS_UNAVAILABLE,
        label=None,
        risk_level=None,
    )


def map_nutrition_status(
    score: Optional[Union[int, float]],
    raw_status: Optional[str] = None,
) -> NutritionPresentation:
    """
    Maps a Nutrition score (0..100) to presentation status.

    Parameters:
        score: Numerical score between 0.0 and 100.0, or None if missing.
        raw_status: Scorer status string (e.g. "Insufficient Nutrition Data", "error")

    Returns:
        NutritionPresentation:
            - 0–25   -> red    (label: "Low Nutrition")
            - 26–50  -> orange (label: "Slightly Better Nutrition")
            - 51–75  -> yellow (label: "Better Nutrition")
            - 76–100 -> green  (label: "Good Nutrition")
            - None/Invalid -> unavailable (label: None, score: None)
    """
    if raw_status in ("error", "Insufficient Nutrition Data", "unavailable"):
        return NutritionPresentation(
            status=STATUS_UNAVAILABLE,
            label=None,
            score=None,
        )

    if score is None or isinstance(score, bool) or not isinstance(score, (int, float)):
        return NutritionPresentation(
            status=STATUS_UNAVAILABLE,
            label=None,
            score=None,
        )

    num_score = float(score)
    # Range validation: score must be in [0.0, 100.0]
    if num_score < 0.0 or num_score > 100.0:
        return NutritionPresentation(
            status=STATUS_UNAVAILABLE,
            label=None,
            score=None,
        )

    if num_score <= 25.0:
        return NutritionPresentation(
            status=STATUS_RED,
            label="Low Nutrition",
            score=num_score,
        )
    elif num_score <= 50.0:
        return NutritionPresentation(
            status=STATUS_ORANGE,
            label="Slightly Better Nutrition",
            score=num_score,
        )
    elif num_score <= 75.0:
        return NutritionPresentation(
            status=STATUS_YELLOW,
            label="Better Nutrition",
            score=num_score,
        )
    else:
        return NutritionPresentation(
            status=STATUS_GREEN,
            label="Good Nutrition",
            score=num_score,
        )


def map_food_analysis_presentation(
    food_safety: Any,
    allergy: Any,
    nutrition: Any,
) -> FoodAnalysisPresentation:
    """
    Computes presentation statuses for the three dimensions independently.

    CRITICAL: Does NOT create an overall score, color, or blended verdict.
    """
    # 1. Food Safety
    if isinstance(food_safety, str):
        fs_pres = map_food_safety_status(food_safety)
    elif isinstance(food_safety, dict):
        fs_raw_status = food_safety.get("status")
        if fs_raw_status in ("error", "no_ingredients", "unavailable"):
            fs_pres = FoodSafetyPresentation(status=STATUS_UNAVAILABLE, label=None, risk_class=None)
        else:
            fs_pres = map_food_safety_status(food_safety.get("risk_class"))
    else:
        fs_pres = FoodSafetyPresentation(status=STATUS_UNAVAILABLE, label=None, risk_class=None)

    # 2. Allergy Risk
    if isinstance(allergy, str):
        al_pres = map_allergy_status(allergy)
    elif isinstance(allergy, dict):
        al_risk = allergy.get("product_risk_level") or allergy.get("risk_level")
        al_pres = map_allergy_status(
            risk_level=al_risk,
            raw_status=allergy.get("status"),
        )
    elif hasattr(allergy, "status"):
        al_dict = allergy.to_dict() if hasattr(allergy, "to_dict") else allergy.__dict__
        al_risk = al_dict.get("product_risk_level") or al_dict.get("risk_level")
        al_pres = map_allergy_status(
            risk_level=al_risk,
            raw_status=al_dict.get("status"),
        )
    else:
        al_pres = AllergyPresentation(status=STATUS_UNAVAILABLE, label=None, risk_level=None)

    # 3. Nutrition
    if isinstance(nutrition, (int, float)) and not isinstance(nutrition, bool):
        nut_pres = map_nutrition_status(nutrition)
    elif isinstance(nutrition, dict):
        nut_score = nutrition.get("nutrition_score")
        if nut_score is None:
            nut_score = nutrition.get("score")
        nut_pres = map_nutrition_status(
            score=nut_score,
            raw_status=nutrition.get("status"),
        )
    else:
        nut_pres = NutritionPresentation(status=STATUS_UNAVAILABLE, label=None, score=None)

    return FoodAnalysisPresentation(
        food_safety=fs_pres,
        allergy=al_pres,
        nutrition=nut_pres,
    )
