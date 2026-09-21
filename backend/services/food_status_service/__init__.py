"""
backend/services/food_status_service/__init__.py

Public package exports for PicWise Food Status Service (Phase 9H).
"""

from backend.services.food_status_service.constants import (
    ALLERGY_STATUS_MAP,
    ALLERGY_UI_LABEL_MAP,
    FOOD_SAFETY_LABEL_MAP,
    FOOD_SAFETY_STATUS_MAP,
    NUTRITION_SCORE_THRESHOLDS,
    STATUS_GREEN,
    STATUS_ORANGE,
    STATUS_RED,
    STATUS_UNAVAILABLE,
    STATUS_YELLOW,
    VALID_STATUSES,
)
from backend.services.food_status_service.mapper import (
    map_allergy_status,
    map_food_analysis_presentation,
    map_food_safety_status,
    map_nutrition_status,
)
from backend.services.food_status_service.models import (
    AllergyPresentation,
    FoodAnalysisPresentation,
    FoodSafetyPresentation,
    NutritionPresentation,
)

__all__ = [
    "STATUS_GREEN",
    "STATUS_YELLOW",
    "STATUS_ORANGE",
    "STATUS_RED",
    "STATUS_UNAVAILABLE",
    "VALID_STATUSES",
    "FOOD_SAFETY_STATUS_MAP",
    "FOOD_SAFETY_LABEL_MAP",
    "ALLERGY_STATUS_MAP",
    "ALLERGY_UI_LABEL_MAP",
    "NUTRITION_SCORE_THRESHOLDS",
    "FoodSafetyPresentation",
    "AllergyPresentation",
    "NutritionPresentation",
    "FoodAnalysisPresentation",
    "map_food_safety_status",
    "map_allergy_status",
    "map_nutrition_status",
    "map_food_analysis_presentation",
]
