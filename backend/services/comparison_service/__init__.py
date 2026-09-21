"""
backend/services/comparison_service/__init__.py

Package entry point for the PicWise Food-Only Product Comparison Service.
"""

from backend.services.comparison_service.comparator import (
    compare_food_products,
    validate_comparison_input,
)
from backend.services.comparison_service.models import (
    ComparedProduct,
    ComparisonRow,
    FoodComparisonResult,
)

__all__ = [
    "compare_food_products",
    "validate_comparison_input",
    "ComparedProduct",
    "ComparisonRow",
    "FoodComparisonResult",
]
