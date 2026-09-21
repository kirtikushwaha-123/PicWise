"""
backend/services/allergy_service/__init__.py

PicWise Food Allergy Risk Service.
Deterministic ingredient matching and knowledge-base lookup engine.
"""

from backend.services.allergy_service.constants import (
    INSUFFICIENT_DATA_LABEL,
    REASON_NOT_FOUND_IN_KB,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_NO_RISK,
    RISK_RANKS,
    STATUS_ERROR,
    STATUS_INSUFFICIENT_DATA,
    STATUS_NO_INGREDIENTS,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_UNAVAILABLE,
    UI_RISK_LABELS,
    VALID_ALLERGY_RISKS,
)
from backend.services.allergy_service.engine import calculate_allergy_risk
from backend.services.allergy_service.models import (
    AllergyIngredientResult,
    AllergyResult,
)

__all__ = [
    "calculate_allergy_risk",
    "AllergyResult",
    "AllergyIngredientResult",
    "RISK_NO_RISK",
    "RISK_LOW",
    "RISK_MEDIUM",
    "RISK_HIGH",
    "VALID_ALLERGY_RISKS",
    "RISK_RANKS",
    "UI_RISK_LABELS",
    "INSUFFICIENT_DATA_LABEL",
    "STATUS_SUCCESS",
    "STATUS_UNAVAILABLE",
    "STATUS_INSUFFICIENT_DATA",
    "STATUS_SKIPPED",
    "STATUS_NO_INGREDIENTS",
    "STATUS_ERROR",
    "REASON_NOT_FOUND_IN_KB",
]
