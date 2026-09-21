"""
backend/services/food_status_service/constants.py

Constants, status color definitions, label mappings, and boundary thresholds
for the PicWise Food Analysis presentation-status layer (Phase 9H).
"""

from typing import Dict, Tuple

# Presentation status colors
STATUS_GREEN = "green"
STATUS_YELLOW = "yellow"
STATUS_ORANGE = "orange"
STATUS_RED = "red"
STATUS_UNAVAILABLE = "unavailable"

VALID_STATUSES = {
    STATUS_GREEN,
    STATUS_YELLOW,
    STATUS_ORANGE,
    STATUS_RED,
    STATUS_UNAVAILABLE,
}

# Food Safety ML class -> presentation status color
# Very Safe     -> green
# Safe          -> yellow (Strictly yellow, never green!)
# Moderate Risk -> orange
# High Risk     -> red
FOOD_SAFETY_STATUS_MAP: Dict[str, str] = {
    "Very Safe": STATUS_GREEN,
    "Safe": STATUS_YELLOW,
    "Moderate Risk": STATUS_ORANGE,
    "High Risk": STATUS_RED,
}

# Food Safety presentation labels (preserves canonical ML class names)
FOOD_SAFETY_LABEL_MAP: Dict[str, str] = {
    "Very Safe": "Very Safe",
    "Safe": "Safe",
    "Moderate Risk": "Moderate Risk",
    "High Risk": "High Risk",
}

# Allergy Knowledge Base raw level -> presentation status color
# No Risk -> green
# Low     -> yellow
# Medium  -> orange
# High    -> red
ALLERGY_STATUS_MAP: Dict[str, str] = {
    "No Risk": STATUS_GREEN,
    "Low": STATUS_YELLOW,
    "Medium": STATUS_ORANGE,
    "High": STATUS_RED,
}

# Allergy presentation labels
ALLERGY_UI_LABEL_MAP: Dict[str, str] = {
    "No Risk": "Allergen-Free",
    "Low": "Low Allergy Risk",
    "Medium": "Moderate Allergy Risk",
    "High": "High Allergy Risk",
}

# Nutrition score boundaries:
# 0–25   -> red    -> Low Nutrition
# 26–50  -> orange -> Slightly Better Nutrition
# 51–75  -> yellow -> Better Nutrition
# 76–100 -> green  -> Good Nutrition
# Exact inclusive boundaries:
# 0 <= score <= 25   -> red
# 26 <= score <= 50  -> orange
# 51 <= score <= 75  -> yellow
# 76 <= score <= 100 -> green
NUTRITION_SCORE_THRESHOLDS: Tuple[Tuple[float, float, str, str], ...] = (
    (0.0, 25.0, STATUS_RED, "Low Nutrition"),
    (26.0, 50.0, STATUS_ORANGE, "Slightly Better Nutrition"),
    (51.0, 75.0, STATUS_YELLOW, "Better Nutrition"),
    (76.0, 100.0, STATUS_GREEN, "Good Nutrition"),
)
