"""
backend/services/food_analysis_service/models.py

Data models and contracts for the PicWise Unified Food Analysis Service.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class FoodSafetyIngredientResult:
    """Individual ingredient food safety assessment."""
    ingredient: str
    risk_class: Optional[str]
    confidence: float
    probabilities: Dict[str, float]
    raw_text: str = ""
    matched_name: Optional[str] = None
    match_type: str = "unmatched"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FoodSafetyResult:
    """Component result for production Food Safety ML inference."""
    status: str  # "success" | "no_ingredients" | "error"
    ingredients: List[Dict[str, Any]] = field(default_factory=list)
    total_ingredients: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    risk_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


from backend.services.allergy_service.models import (
    AllergyIngredientResult,
    AllergyResult,
)


def _sanitize_for_serialization(obj: Any) -> Any:
    """
    Recursively sanitizes values for JSON serialization without mutating
    any semantic values.

    Strict value preservation guarantees:
      - None remains None (never coerced to 0, false, or empty string)
      - 0 remains 0 (integer type preserved)
      - 0.0 remains 0.0 (float type preserved)
      - False / True remain bool
    """
    if obj is None:
        return None

    # Check bool before int because bool inherits from int in Python
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, str):
        return obj

    # Handle NumPy scalar and array types if present
    try:
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return [_sanitize_for_serialization(item) for item in obj.tolist()]
    except ImportError:
        pass

    # Handle dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        return _sanitize_for_serialization(asdict(obj))

    # Handle objects with to_dict()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _sanitize_for_serialization(obj.to_dict())

    # Handle Enums
    if hasattr(obj, "value") and hasattr(obj, "name"):
        return _sanitize_for_serialization(obj.value)

    # Handle dictionaries
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_serialization(v) for k, v in obj.items()}

    # Handle sequences
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_serialization(item) for item in obj]
    if isinstance(obj, set):
        return [_sanitize_for_serialization(item) for item in sorted(obj, key=lambda x: str(x))]

    # Fallback for Path or other printable objects
    return str(obj)


@dataclass
class FoodAnalysisResult:
    """
    Unified result of the PicWise Food Analysis Pipeline.
    Combines OCR, Food Safety ML, Nutrition Scoring, and Allergy status.
    """
    category: str
    success: bool
    ocr: Optional[Dict[str, Any]] = None
    food_safety: Optional[Dict[str, Any]] = None
    nutrition: Optional[Dict[str, Any]] = None
    allergy: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    presentation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        fs_val = self.food_safety
        if hasattr(fs_val, "to_dict"):
            fs_val = fs_val.to_dict()

        al_val = self.allergy
        if hasattr(al_val, "to_dict"):
            al_val = al_val.to_dict()

        nut_val = self.nutrition
        if hasattr(nut_val, "to_dict"):
            nut_val = nut_val.to_dict()

        pres_val = self.presentation
        if hasattr(pres_val, "to_dict"):
            pres_val = pres_val.to_dict()

        d = {
            "category": self.category,
            "success": self.success,
            "ocr": self.ocr,
            "food_safety": fs_val,
            "nutrition": nut_val,
            "allergy": al_val,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }
        if pres_val is not None:
            d["presentation"] = pres_val

        return _sanitize_for_serialization(d)
