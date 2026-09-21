"""
backend/services/food_analysis_service/errors.py

Exception classes for the PicWise Unified Food Analysis Service.
"""


class FoodAnalysisError(Exception):
    """Base exception for all food analysis errors."""
    pass


class InvalidCategoryError(FoodAnalysisError):
    """Raised when category is missing, not 'food', or unsupported."""
    pass


class ImageProcessingError(FoodAnalysisError):
    """Raised when image bytes are empty, malformed, or unreadable."""
    pass


class OCRError(FoodAnalysisError):
    """Raised when OCR extraction fails fatally."""
    pass
