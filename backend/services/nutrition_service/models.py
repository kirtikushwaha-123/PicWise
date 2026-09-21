"""
backend/services/nutrition_service/models.py

Data models and type definitions for the PicWise Nutrition Scoring Engine.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class NormalizedNutrient:
    """Represents a normalized nutrient value per 100g."""
    name: str
    amount_per_100g: Optional[float] = None
    unit: str = "g"
    is_explicit_zero: bool = False
    is_missing: bool = True
    original_value: Optional[float] = None
    original_unit: Optional[str] = None
    source: str = "unspecified"  # "per_100g", "per_serving", "derived", "fallback"


@dataclass
class EvaluatedNutrient:
    """Represents the evaluation detail of a single nutrient."""
    nutrient: str
    amount_per_100g: Optional[float]
    unit: str
    threshold: Optional[float] = None
    penalty: Optional[float] = None
    positive_score: Optional[float] = None
    direction: Optional[str] = None
    status: Optional[str] = None
    is_explicit_zero: bool = False
    is_missing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class NutritionScoreResult:
    """Deterministic result of the PicWise Nutrition Scoring Engine."""
    nutrition_score: Optional[float] = None
    status: str = "scored"  # "scored", "insufficient_data", "water_override", "not_applicable"
    nutrition_completeness: float = 0.0
    components: Dict[str, float] = field(default_factory=lambda: {
        "negative_risk": 0.0,
        "positive_nutrition": 0.0,
        "micronutrient_contribution": 0.0,
    })
    risk_details: Dict[str, Any] = field(default_factory=dict)
    energy_penalty: float = 0.0
    guardrails: Dict[str, bool] = field(default_factory=lambda: {
        "sugar_guardrail": False,
        "catastrophic_risk": False,
        "catastrophic_ceiling_applied": False,
        "water_override": False,
    })
    nutrients_evaluated: List[Dict[str, Any]] = field(default_factory=list)
    nutrients_missing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
