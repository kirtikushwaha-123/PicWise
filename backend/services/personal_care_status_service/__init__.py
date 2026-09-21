"""
backend/services/personal_care_status_service/__init__.py

Public interface for Personal Care status mapping and aggregation.
"""

from backend.services.personal_care_status_service.constants import (
    ALLERGY_LABEL_MAP,
    ALLERGY_STATUS_MAP,
    IRRITATION_LABEL_MAP,
    IRRITATION_STATUS_MAP,
    SAFETY_LABEL_MAP,
    SAFETY_STATUS_MAP,
    STATUS_GREEN,
    STATUS_ORANGE,
    STATUS_RED,
    STATUS_UNAVAILABLE,
    STATUS_YELLOW,
)
from backend.services.personal_care_status_service.mapper import (
    aggregate_product_dimension,
    map_personal_care_allergy_status,
    map_personal_care_irritation_status,
    map_personal_care_presentation,
    map_personal_care_safety_status,
)
from backend.services.personal_care_status_service.models import (
    DimensionPresentation,
    PersonalCareAnalysisPresentation,
)

__all__ = [
    "STATUS_GREEN",
    "STATUS_YELLOW",
    "STATUS_ORANGE",
    "STATUS_RED",
    "STATUS_UNAVAILABLE",
    "SAFETY_STATUS_MAP",
    "SAFETY_LABEL_MAP",
    "ALLERGY_STATUS_MAP",
    "ALLERGY_LABEL_MAP",
    "IRRITATION_STATUS_MAP",
    "IRRITATION_LABEL_MAP",
    "DimensionPresentation",
    "PersonalCareAnalysisPresentation",
    "map_personal_care_safety_status",
    "map_personal_care_allergy_status",
    "map_personal_care_irritation_status",
    "aggregate_product_dimension",
    "map_personal_care_presentation",
]
