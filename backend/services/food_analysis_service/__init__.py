"""
backend/services/food_analysis_service/__init__.py

Public package exports for PicWise Unified Food Analysis Service.
"""

from .analyzer import analyze_food, analyze_food_from_text
from .errors import (
    FoodAnalysisError,
    InvalidCategoryError,
    ImageProcessingError,
    OCRError,
)
from .models import (
    FoodAnalysisResult,
    FoodSafetyResult,
    FoodSafetyIngredientResult,
    AllergyResult,
)

__all__ = [
    "analyze_food",
    "analyze_food_from_text",
    "FoodAnalysisResult",
    "FoodSafetyResult",
    "FoodSafetyIngredientResult",
    "AllergyResult",
    "FoodAnalysisError",
    "InvalidCategoryError",
    "ImageProcessingError",
    "OCRError",
]
