"""
backend/services/food_status_service/models.py

Typed data models for the PicWise Food Analysis presentation status layer (Phase 9H).
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class FoodSafetyPresentation:
    """Presentation status for Food Safety."""
    status: str                                  # "green" | "yellow" | "orange" | "red" | "unavailable"
    label: Optional[str] = None                  # "Very Safe" | "Safe" | "Moderate Risk" | "High Risk" | None
    risk_class: Optional[str] = None             # Raw ML classification

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AllergyPresentation:
    """Presentation status for Allergy Risk."""
    status: str                                  # "green" | "yellow" | "orange" | "red" | "unavailable"
    label: Optional[str] = None                  # "Allergen-Free" | "Low Allergy Risk" | "Moderate Allergy Risk" | "High Allergy Risk" | None
    risk_level: Optional[str] = None             # Raw KB risk level: "No Risk" | "Low" | "Medium" | "High" | None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class NutritionPresentation:
    """Presentation status for Nutrition."""
    status: str                                  # "green" | "yellow" | "orange" | "red" | "unavailable"
    label: Optional[str] = None                  # "Good Nutrition" | "Better Nutrition" | "Slightly Better Nutrition" | "Low Nutrition" | None
    score: Optional[float] = None                # Numerical nutrition score: 0..100 | None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class FoodAnalysisPresentation:
    """
    Combined presentation container for the three independent dimensions.
    CRITICAL: There is NO overall product score, color, or blended verdict.
    """
    food_safety: FoodSafetyPresentation
    allergy: AllergyPresentation
    nutrition: NutritionPresentation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "food_safety": self.food_safety.to_dict(),
            "allergy": self.allergy.to_dict(),
            "nutrition": self.nutrition.to_dict(),
        }
