"""
backend/services/personal_care_status_service/constants.py

Domain-specific constants and color mappings for the Personal Care presentation layer.

CRITICAL PRESENTATION CONSTRAINTS:
1. Safety: Very Safe -> green, Safe -> yellow (strictly yellow), Moderate Risk -> orange, High Risk -> red
2. Allergy: No Risk -> green, Low -> yellow, Medium -> orange, High -> red
3. Irritation: No Risk -> green, Low -> yellow, Medium -> orange, High -> red
4. Missing / Unknown: status='unavailable', color='unavailable', label='Unavailable'
5. Independent Dimensions: No composite score, no overall color, no cross-dimension averaging.
"""

# Color tokens aligned with CSS and Food presentation layer
STATUS_GREEN = "green"
STATUS_YELLOW = "yellow"
STATUS_ORANGE = "orange"
STATUS_RED = "red"
STATUS_UNAVAILABLE = "unavailable"

# --------------------------------------------------------------------------
# SAFETY LEVEL
# --------------------------------------------------------------------------
SAFETY_VERY_SAFE = "Very Safe"
SAFETY_SAFE = "Safe"
SAFETY_MODERATE_RISK = "Moderate Risk"
SAFETY_HIGH_RISK = "High Risk"

SAFETY_STATUS_MAP = {
    SAFETY_VERY_SAFE: STATUS_GREEN,
    SAFETY_SAFE: STATUS_YELLOW,
    SAFETY_MODERATE_RISK: STATUS_ORANGE,
    SAFETY_HIGH_RISK: STATUS_RED,
}

SAFETY_LABEL_MAP = {
    SAFETY_VERY_SAFE: "Very Safe",
    SAFETY_SAFE: "Safe",
    SAFETY_MODERATE_RISK: "Moderate Risk",
    SAFETY_HIGH_RISK: "High Risk",
}

# Risk ranking: lower rank = safer
SAFETY_RISK_RANKS = {
    SAFETY_VERY_SAFE: 0,
    SAFETY_SAFE: 1,
    SAFETY_MODERATE_RISK: 2,
    SAFETY_HIGH_RISK: 3,
}

# --------------------------------------------------------------------------
# ALLERGY RISK
# --------------------------------------------------------------------------
ALLERGY_NO_RISK = "No Risk"
ALLERGY_LOW = "Low"
ALLERGY_MEDIUM = "Medium"
ALLERGY_HIGH = "High"

ALLERGY_STATUS_MAP = {
    ALLERGY_NO_RISK: STATUS_GREEN,
    ALLERGY_LOW: STATUS_YELLOW,
    ALLERGY_MEDIUM: STATUS_ORANGE,
    ALLERGY_HIGH: STATUS_RED,
}

ALLERGY_LABEL_MAP = {
    ALLERGY_NO_RISK: "No Allergy Risk",
    ALLERGY_LOW: "Low Allergy Risk",
    ALLERGY_MEDIUM: "Moderate Allergy Risk",
    ALLERGY_HIGH: "High Allergy Risk",
}

ALLERGY_RISK_RANKS = {
    ALLERGY_NO_RISK: 0,
    ALLERGY_LOW: 1,
    ALLERGY_MEDIUM: 2,
    ALLERGY_HIGH: 3,
}

# --------------------------------------------------------------------------
# IRRITATION RISK
# --------------------------------------------------------------------------
IRRITATION_NO_RISK = "No Risk"
IRRITATION_LOW = "Low"
IRRITATION_MEDIUM = "Medium"
IRRITATION_HIGH = "High"

IRRITATION_STATUS_MAP = {
    IRRITATION_NO_RISK: STATUS_GREEN,
    IRRITATION_LOW: STATUS_YELLOW,
    IRRITATION_MEDIUM: STATUS_ORANGE,
    IRRITATION_HIGH: STATUS_RED,
}

IRRITATION_LABEL_MAP = {
    IRRITATION_NO_RISK: "No Irritation Risk",
    IRRITATION_LOW: "Low Irritation Risk",
    IRRITATION_MEDIUM: "Moderate Irritation Risk",
    IRRITATION_HIGH: "High Irritation Risk",
}

IRRITATION_RISK_RANKS = {
    IRRITATION_NO_RISK: 0,
    IRRITATION_LOW: 1,
    IRRITATION_MEDIUM: 2,
    IRRITATION_HIGH: 3,
}
