"""
backend/services/personal_care_analysis_service/errors.py

Exception hierarchy for the Personal Care analysis service.
"""


class PersonalCareAnalysisError(Exception):
    """Base exception for all Personal Care analysis errors."""
    pass


class InvalidCategoryError(PersonalCareAnalysisError):
    """Raised when category is missing or not strictly 'personal_care'."""
    pass


class ImageProcessingError(PersonalCareAnalysisError):
    """Raised when image payload is empty, corrupted, or unsupported."""
    pass


class OCRError(PersonalCareAnalysisError):
    """Raised when the underlying shared OCR engine fails critically."""
    pass


# Convenience aliases
CategoryValidationError = InvalidCategoryError
ImageValidationError = ImageProcessingError
