"""
backend/services/comparison_service/models.py

Data models for the PicWise Food-Only Product Comparison Service (Step 5).
Adheres to strict architectural constraints:
- Food products only
- No overall winner
- No total comparison score
- No ranking from best to worst
- Independent dimension preservation across Food Safety, Allergy Risk, and Nutrition
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class DimensionValue:
    """Status and presentation for a single product dimension."""
    status: str
    label: Optional[str] = None
    risk_class: Optional[str] = None
    risk_level: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ComparedProduct:
    """Summary of a single product within the comparison."""
    name: str
    food_safety: Dict[str, Any]
    allergy: Dict[str, Any]
    nutrition: Dict[str, Any]
    nutrition_score: Optional[float] = None
    analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "food_safety": self.food_safety,
            "allergy": self.allergy,
            "nutrition": self.nutrition,
            "nutrition_score": self.nutrition_score,
        }
        if self.analysis:
            d["analysis"] = self.analysis
        return d


@dataclass
class ComparisonRow:
    """Row-oriented representation of a dimension across all compared products."""
    dimension: str
    key: str
    values: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FoodComparisonResult:
    """Unified result of food product comparison."""
    success: bool
    products: List[ComparedProduct] = field(default_factory=list)
    rows: List[ComparisonRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "products": [p.to_dict() for p in self.products],
            "rows": [r.to_dict() for r in self.rows],
        }
