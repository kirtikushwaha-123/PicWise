"""
backend/services/personal_care_analysis_service/__init__.py

Public interface for the Personal Care analysis service.
"""

from backend.services.personal_care_analysis_service.analyzer import analyze_personal_care
from backend.services.personal_care_analysis_service.enrichment import (
    PersonalCareKnowledgeBase,
    PersonalCareSemanticFeatures,
    get_personal_care_knowledge_base,
)
from backend.services.personal_care_analysis_service.errors import (
    CategoryValidationError,
    ImageProcessingError,
    ImageValidationError,
    InvalidCategoryError,
    OCRError,
    PersonalCareAnalysisError,
)
from backend.services.personal_care_analysis_service.models import (
    PersonalCareAnalysisResult,
    PersonalCareIngredientAnalysis,
)

__all__ = [
    "analyze_personal_care",
    "PersonalCareKnowledgeBase",
    "PersonalCareSemanticFeatures",
    "get_personal_care_knowledge_base",
    "PersonalCareAnalysisError",
    "InvalidCategoryError",
    "CategoryValidationError",
    "ImageProcessingError",
    "ImageValidationError",
    "OCRError",
    "PersonalCareAnalysisResult",
    "PersonalCareIngredientAnalysis",
]
