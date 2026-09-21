"""
backend/services/nutrition_service/constants.py

Locked constants and parameters for the PicWise Deterministic Nutrition Scoring Engine.
Authoritative source of truth: phase9D_nutrition_methodology_spec.md
"""

# Overall Score Range
SCORE_MIN = 0.0
SCORE_MAX = 100.0

# -------------------------------------------------------------------------
# Primary Risk Factors & Thresholds (per 100g)
# -------------------------------------------------------------------------
# Saturated Fat
THRESHOLD_SATURATED_FAT_FOOD = 4.0        # g / 100g
THRESHOLD_SATURATED_FAT_CULINARY = 20.0    # g / 100g (20% for oils, butter, etc.)

# Added Sugar
THRESHOLD_ADDED_SUGAR_FOOD = 10.0         # g / 100g
THRESHOLD_ADDED_SUGAR_BEVERAGE = 5.0      # g / 100g (for liquid/beverages)
# Fallback when Added Sugar is missing but Total Sugar is available
THRESHOLD_TOTAL_SUGAR_FALLBACK_FOOD = 12.5     # g / 100g
THRESHOLD_TOTAL_SUGAR_FALLBACK_BEVERAGE = 6.25 # g / 100g

# Sodium
THRESHOLD_SODIUM = 400.0                  # mg / 100g

# Trans Fat
THRESHOLD_TRANS_FAT = 0.3                 # g / 100g

# Logistic Penalty Parameters
LOGISTIC_K = 3.0                          # Steepness constant
EXP_CLAMP_MIN = -50.0                     # For numerical stability in exp()
EXP_CLAMP_MAX = 50.0

# Equal internal risk weights for Minkowski L2 aggregation
PRIMARY_RISK_WEIGHTS = {
    "saturated_fat": 1.0,
    "added_sugars": 1.0,
    "sodium": 1.0,
    "trans_fat": 1.0,
}

# Missing Trans Fat baseline penalty when no hydrogenated oils found
MISSING_TRANS_FAT_PENALTY = 10.0

# Partially hydrogenated keywords in ingredients indicating trans fat risk
HYDROGENATED_OIL_KEYWORDS = [
    "partially hydrogenated",
    "hydrogenated vegetable oil",
    "hydrogenated fat",
    "hydrogenated oil",
    "partially hydrogenated oil",
]

# -------------------------------------------------------------------------
# Energy / Calorie Stepwise Penalties (per 100g)
# -------------------------------------------------------------------------
ENERGY_LOW_THRESHOLD = 250.0              # kcal / 100g
ENERGY_HIGH_THRESHOLD = 400.0             # kcal / 100g

ENERGY_PENALTY_LOW = 0.0                  # Energy < 250 kcal
ENERGY_PENALTY_MEDIUM = 15.0               # 250 <= Energy <= 400 kcal
ENERGY_PENALTY_HIGH = 30.0                # Energy > 400 kcal

# -------------------------------------------------------------------------
# Positive Macronutrient Component Parameters
# -------------------------------------------------------------------------
WEIGHT_FIBER = 0.45
TAU_FIBER = 4.0                           # g / 100g

WEIGHT_PROTEIN = 0.35
TAU_PROTEIN = 8.0                         # g / 100g

WEIGHT_UNSATURATED_FAT = 0.20
TAU_UNSATURATED_FAT = 12.0                # g / 100g

# -------------------------------------------------------------------------
# Top-Level Synthesis Weights
# -------------------------------------------------------------------------
WEIGHT_SNEG = 0.60                        # Negative risk component weight
WEIGHT_SPOS = 0.28                        # Positive nutrition component weight
WEIGHT_SMICRO = 0.12                      # Beta: Micronutrient adequacy weight

# Anti-fortification discount factor parameters: gamma = max(0.0, 1.0 - Prisk / 50.0)
GAMMA_PRISK_DIVISOR = 50.0

# -------------------------------------------------------------------------
# Guardrails
# -------------------------------------------------------------------------
# Water override
WATER_SCORE = 100.0

# Sugar Guardrail: triggers when Added Sugar >= 2.5 * T_sugar
SUGAR_GUARDRAIL_MULTIPLIER = 2.5

# Catastrophic Risk Guardrail: triggers when any primary risk >= 2.5 * T_i
CATASTROPHIC_MULTIPLIER = 2.5
CATASTROPHIC_CEILING = 35.0

# -------------------------------------------------------------------------
# Completeness & Conversions
# -------------------------------------------------------------------------
CORE_NUTRIENTS = {
    "energy",
    "protein",
    "carbohydrate",
    "fat",
    "sodium",
}
MIN_CORE_NUTRIENTS = 3

SALT_TO_SODIUM_FACTOR = 400.0             # Sodium (mg) = Salt (g) * 400.0
KJ_TO_KCAL_DIVISOR = 4.184                # Energy (kcal) = Energy (kJ) / 4.184
