"""
backend/services/allergy_service/constants.py

Constants and mappings for deterministic food allergy risk calculation.
"""

from typing import Dict

# Valid raw risk values present in the authoritative food dataset
RISK_NO_RISK = "No Risk"
RISK_LOW = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH = "High"

VALID_ALLERGY_RISKS = {RISK_NO_RISK, RISK_LOW, RISK_MEDIUM, RISK_HIGH}

# Numeric ranks for product-level severity aggregation (higher rank = higher risk)
RISK_RANKS: Dict[str, int] = {
    RISK_NO_RISK: 0,
    RISK_LOW: 1,
    RISK_MEDIUM: 2,
    RISK_HIGH: 3,
}

# Presentation UI labels
UI_RISK_LABELS: Dict[str, str] = {
    RISK_NO_RISK: "Allergen-Free",
    RISK_LOW: "Low Allergy Risk",
    RISK_MEDIUM: "Moderate Allergy Risk",
    RISK_HIGH: "High Allergy Risk",
}

# Presentation status for products with insufficient or unresolvable allergy data
INSUFFICIENT_DATA_LABEL = "Insufficient Allergy Data"

# Pipeline and component status strings
STATUS_SUCCESS = "success"
STATUS_UNAVAILABLE = "unavailable"
STATUS_INSUFFICIENT_DATA = "insufficient_data"
STATUS_SKIPPED = "skipped"
STATUS_NO_INGREDIENTS = "no_ingredients"
STATUS_ERROR = "error"

# Default reason for unknown/unmatched ingredients
REASON_NOT_FOUND_IN_KB = "Ingredient not found in knowledge base"
