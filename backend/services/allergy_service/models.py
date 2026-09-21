"""
backend/services/allergy_service/models.py

Typed data models for PicWise Allergy Risk analysis.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class AllergyIngredientResult:
    """Allergy risk lookup result for a single ingredient."""
    ingredient: str
    allergy_risk: Optional[str] = None
    ui_label: Optional[str] = None
    status: str = "unavailable"
    reason: Optional[str] = None
    matched_name: Optional[str] = None
    raw_text: str = ""
    match_type: str = "unmatched"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AllergyResult:
    """Component result for Allergy detection."""
    status: str
    product_risk_level: Optional[str] = None
    product_ui_label: Optional[str] = None
    allergens_detected: List[str] = field(default_factory=list)
    ingredients: List[Dict[str, Any]] = field(default_factory=list)
    total_ingredients: int = 0
    known_ingredients: int = 0
    unknown_ingredients: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def presentation(self) -> Any:
        from backend.services.food_status_service.mapper import map_allergy_status
        return map_allergy_status(self.product_risk_level, self.status)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
