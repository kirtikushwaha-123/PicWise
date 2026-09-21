"""
backend/services/personal_care_analysis_service/models.py

Data models and serialization structures for Personal Care analysis.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PersonalCareIngredientAnalysis:
    """Detailed analysis for a single extracted or recognized ingredient."""
    raw_text: str
    matched_name: Optional[str]
    status: str  # "success" | "ingredient_not_recognized" | "semantic_enrichment_unavailable" | "model_prediction_failure"
    reason: Optional[str] = None
    features: Optional[Dict[str, str]] = None
    safety: Dict[str, Any] = field(default_factory=dict)
    allergy: Dict[str, Any] = field(default_factory=dict)
    irritation: Dict[str, Any] = field(default_factory=dict)
    presentation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "matched_name": self.matched_name,
            "status": self.status,
            "reason": self.reason,
            "features": self.features,
            "safety": self.safety,
            "allergy": self.allergy,
            "irritation": self.irritation,
            "presentation": self.presentation,
        }


@dataclass
class PersonalCareAnalysisResult:
    """Unified top-level analysis result returned by the Personal Care orchestrator."""
    category: str = "personal_care"
    success: bool = True
    ocr: Optional[Dict[str, Any]] = None
    personal_care: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    presentation: Dict[str, Any] = field(default_factory=dict)
    ocr_quality_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "category": self.category,
            "success": self.success,
            "ocr": self.ocr,
            "personal_care": self.personal_care,
            "errors": self.errors,
            "warnings": self.warnings,
            "presentation": self.presentation,
        }
        if self.ocr_quality_warning is not None:
            d["ocr_quality_warning"] = self.ocr_quality_warning
        return d
