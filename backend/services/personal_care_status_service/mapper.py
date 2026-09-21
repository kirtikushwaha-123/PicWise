"""
backend/services/personal_care_status_service/mapper.py

Deterministic status mapping and conservative aggregation functions for Personal Care.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. No overall personal care composite score or overall color.
2. Independent dimensions: Safety, Allergy, and Irritation remain orthogonal.
3. Risk rankings:
   - Safety: Very Safe < Safe < Moderate Risk < High Risk
   - Allergy: No Risk < Low < Medium < High
   - Irritation: No Risk < Low < Medium < High
4. Conservative product aggregation:
   - Product result = highest valid predicted risk among recognized ingredients.
   - Unknown ingredients do NOT invalidate recognized ingredients (they appear in warnings).
   - If no recognized ingredients have valid predictions -> status = 'unavailable'.
"""

from typing import Any, Dict, List, Optional, Tuple

from backend.services.personal_care_status_service.constants import (
    ALLERGY_LABEL_MAP,
    ALLERGY_RISK_RANKS,
    ALLERGY_STATUS_MAP,
    IRRITATION_LABEL_MAP,
    IRRITATION_RISK_RANKS,
    IRRITATION_STATUS_MAP,
    SAFETY_LABEL_MAP,
    SAFETY_RISK_RANKS,
    SAFETY_STATUS_MAP,
    STATUS_UNAVAILABLE,
)
from backend.services.personal_care_status_service.models import (
    DimensionPresentation,
    PersonalCareAnalysisPresentation,
)


def map_personal_care_safety_status(
    risk_class: Optional[str],
    raw_status: Optional[str] = None,
) -> DimensionPresentation:
    """
    Maps Personal Care Safety risk class to presentation status.
    Very Safe -> green
    Safe -> yellow (strictly yellow, never green)
    Moderate Risk -> orange
    High Risk -> red
    unavailable -> unavailable
    """
    if raw_status and raw_status != "success":
        return DimensionPresentation(
            status=STATUS_UNAVAILABLE,
            label="Unavailable",
            color=STATUS_UNAVAILABLE,
            risk_class=None,
        )

    if not risk_class or not isinstance(risk_class, str):
        return DimensionPresentation(
            status=STATUS_UNAVAILABLE,
            label="Unavailable",
            color=STATUS_UNAVAILABLE,
            risk_class=None,
        )

    cleaned = risk_class.strip()
    if cleaned in SAFETY_STATUS_MAP:
        color = SAFETY_STATUS_MAP[cleaned]
        label = SAFETY_LABEL_MAP[cleaned]
        return DimensionPresentation(
            status=color,
            label=label,
            color=color,
            risk_class=cleaned,
        )

    return DimensionPresentation(
        status=STATUS_UNAVAILABLE,
        label="Unavailable",
        color=STATUS_UNAVAILABLE,
        risk_class=None,
    )


def map_personal_care_allergy_status(
    risk_class: Optional[str],
    raw_status: Optional[str] = None,
) -> DimensionPresentation:
    """
    Maps Personal Care Allergy risk class to presentation status.
    No Risk -> green
    Low -> yellow
    Medium -> orange
    High -> red
    unavailable -> unavailable
    """
    if raw_status and raw_status != "success":
        return DimensionPresentation(
            status=STATUS_UNAVAILABLE,
            label="Unavailable",
            color=STATUS_UNAVAILABLE,
            risk_class=None,
        )

    if not risk_class or not isinstance(risk_class, str):
        return DimensionPresentation(
            status=STATUS_UNAVAILABLE,
            label="Unavailable",
            color=STATUS_UNAVAILABLE,
            risk_class=None,
        )

    cleaned = risk_class.strip()
    if cleaned in ALLERGY_STATUS_MAP:
        color = ALLERGY_STATUS_MAP[cleaned]
        label = ALLERGY_LABEL_MAP[cleaned]
        return DimensionPresentation(
            status=color,
            label=label,
            color=color,
            risk_class=cleaned,
        )

    return DimensionPresentation(
        status=STATUS_UNAVAILABLE,
        label="Unavailable",
        color=STATUS_UNAVAILABLE,
        risk_class=None,
    )


def map_personal_care_irritation_status(
    risk_class: Optional[str],
    raw_status: Optional[str] = None,
) -> DimensionPresentation:
    """
    Maps Personal Care Irritation risk class to presentation status.
    No Risk -> green
    Low -> yellow
    Medium -> orange
    High -> red
    unavailable -> unavailable
    """
    if raw_status and raw_status != "success":
        return DimensionPresentation(
            status=STATUS_UNAVAILABLE,
            label="Unavailable",
            color=STATUS_UNAVAILABLE,
            risk_class=None,
        )

    if not risk_class or not isinstance(risk_class, str):
        return DimensionPresentation(
            status=STATUS_UNAVAILABLE,
            label="Unavailable",
            color=STATUS_UNAVAILABLE,
            risk_class=None,
        )

    cleaned = risk_class.strip()
    if cleaned in IRRITATION_STATUS_MAP:
        color = IRRITATION_STATUS_MAP[cleaned]
        label = IRRITATION_LABEL_MAP[cleaned]
        return DimensionPresentation(
            status=color,
            label=label,
            color=color,
            risk_class=cleaned,
        )

    return DimensionPresentation(
        status=STATUS_UNAVAILABLE,
        label="Unavailable",
        color=STATUS_UNAVAILABLE,
        risk_class=None,
    )


def aggregate_product_dimension(
    ingredient_results: List[Dict[str, Any]],
    dimension_key: str,
) -> Tuple[Optional[str], str]:
    """
    Deterministically computes the product-level status for a dimension using
    conservative worst-case risk ordering over all recognized ingredients.

    Args:
        ingredient_results: List of ingredient analysis dicts.
        dimension_key: "safety", "allergy", or "irritation".

    Returns:
        (worst_risk_class or None, status: "success" or "unavailable")
    """
    if dimension_key == "safety":
        rank_map = SAFETY_RISK_RANKS
    elif dimension_key == "allergy":
        rank_map = ALLERGY_RISK_RANKS
    elif dimension_key == "irritation":
        rank_map = IRRITATION_RISK_RANKS
    else:
        raise ValueError(f"Unknown dimension key '{dimension_key}'")

    valid_predictions = []
    for ing in ingredient_results:
        # Only consider ingredients that were successfully enriched and predicted
        if ing.get("status") != "success":
            continue
        dim_data = ing.get(dimension_key, {})
        if dim_data.get("status") == "success" and dim_data.get("risk_class") in rank_map:
            valid_predictions.append(dim_data["risk_class"])

    if not valid_predictions:
        return None, STATUS_UNAVAILABLE

    # Worst-case: highest risk rank
    worst_class = max(valid_predictions, key=lambda rc: rank_map[rc])
    return worst_class, "success"


def map_personal_care_presentation(
    safety_risk: Optional[str] = None,
    safety_status: Optional[str] = None,
    allergy_risk: Optional[str] = None,
    allergy_status: Optional[str] = None,
    irritation_risk: Optional[str] = None,
    irritation_status: Optional[str] = None,
) -> PersonalCareAnalysisPresentation:
    """
    Constructs top-level PersonalCareAnalysisPresentation across the 3 independent dimensions.
    """
    return PersonalCareAnalysisPresentation(
        personal_care_safety=map_personal_care_safety_status(safety_risk, safety_status),
        allergy=map_personal_care_allergy_status(allergy_risk, allergy_status),
        irritation=map_personal_care_irritation_status(irritation_risk, irritation_status),
    )
