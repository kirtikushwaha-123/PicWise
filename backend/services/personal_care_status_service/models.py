"""
backend/services/personal_care_status_service/models.py

Presentation dataclasses for Personal Care status cards.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DimensionPresentation:
    """Presentation parameters for a single dimension card (Safety, Allergy, or Irritation)."""
    status: str          # "green" | "yellow" | "orange" | "red" | "unavailable"
    label: Optional[str] # Human-readable label or None
    color: str           # Same as status, for frontend convenience
    risk_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "label": self.label,
            "color": self.color,
            "risk_class": self.risk_class,
        }


@dataclass(frozen=True)
class PersonalCareAnalysisPresentation:
    """Top-level presentation object consumed by the Personal Care frontend."""
    personal_care_safety: DimensionPresentation
    allergy: DimensionPresentation
    irritation: DimensionPresentation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "personal_care_safety": self.personal_care_safety.to_dict(),
            "allergy": self.allergy.to_dict(),
            "irritation": self.irritation.to_dict(),
        }
