"""
backend/config.py

Central application configuration for PicWise (STEP 7).
Defines single source of truth for:
- Upload size limits and allowed image types
- Knowledge base paths and environment variable resolution
- ML model directories
- Thread pool and runtime environment defaults
"""

import os
from pathlib import Path

# Base Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# --------------------------------------------------------------------------
# UPLOAD LIMITS & IMAGE VALIDATION
# --------------------------------------------------------------------------
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB ceiling
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_MIMETYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}
# Pillow's default decompression bomb limit
MAX_IMAGE_PIXELS = 89478485

# --------------------------------------------------------------------------
# KNOWLEDGE BASE PATHS
# --------------------------------------------------------------------------
DEFAULT_FOOD_DATA_PATH = "data/food/food_ingredients_dataset_corrected(2)(1).csv"
FALLBACK_FOOD_DATA_PATH = "data/food/ingredient_knowledge_base_500_cleaned.csv"
FALLBACK_FOOD_ALT_DATA_PATH = "data/food/ingredient_knowledge_base_500_with_alternate_names.csv"

DEFAULT_PERSONAL_CARE_DATA_PATH = "data/personal_care/personal_care_ingredients_dataset_cleaned.xlsx"
FALLBACK_PERSONAL_CARE_DATA_PATH = "data/personal_care/personal_care_ingredients_dataset_csv.xlsx"

DEFAULT_NUTRITION_DATA_PATH = "data/nutrition/nutrition_knowledge_dataset.csv"

# --------------------------------------------------------------------------
# MODEL PATHS
# --------------------------------------------------------------------------
DEFAULT_FOOD_SAFETY_MODEL_DIR = "backend/ml/models/food_safety"
DEFAULT_PERSONAL_CARE_MODEL_DIR = "backend/ml/models/personal_care"
DEFAULT_LEGACY_MODEL_DIR = "backend/ml/models"

# --------------------------------------------------------------------------
# RUNTIME ENVIRONMENT SETUP & PATH RESOLVERS
# --------------------------------------------------------------------------
def setup_runtime_environment():
    """Configures native thread pool environment variables for local runtime stability."""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")


def resolve_food_kb_path() -> str:
    """
    Resolves the canonical food knowledge base CSV path:
    1. FOOD_DATA_PATH environment variable (if set and exists)
    2. DEFAULT_FOOD_DATA_PATH
    3. FALLBACK_FOOD_DATA_PATH
    4. FALLBACK_FOOD_ALT_DATA_PATH
    """
    env_path = os.getenv("FOOD_DATA_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists(DEFAULT_FOOD_DATA_PATH):
        return DEFAULT_FOOD_DATA_PATH
    if os.path.exists(FALLBACK_FOOD_DATA_PATH):
        return FALLBACK_FOOD_DATA_PATH
    if os.path.exists(FALLBACK_FOOD_ALT_DATA_PATH):
        return FALLBACK_FOOD_ALT_DATA_PATH
    return DEFAULT_FOOD_DATA_PATH


def resolve_personal_care_kb_path() -> str:
    """
    Resolves the canonical personal care knowledge base XLSX path:
    1. PERSONAL_CARE_DATA_PATH environment variable (if set and exists)
    2. DEFAULT_PERSONAL_CARE_DATA_PATH
    3. FALLBACK_PERSONAL_CARE_DATA_PATH
    """
    env_path = os.getenv("PERSONAL_CARE_DATA_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists(DEFAULT_PERSONAL_CARE_DATA_PATH):
        return DEFAULT_PERSONAL_CARE_DATA_PATH
    if os.path.exists(FALLBACK_PERSONAL_CARE_DATA_PATH):
        return FALLBACK_PERSONAL_CARE_DATA_PATH
    return DEFAULT_PERSONAL_CARE_DATA_PATH


def resolve_nutrition_kb_path() -> str:
    """
    Resolves the canonical nutrition knowledge base CSV path:
    1. NUTRITION_DATA_PATH environment variable (if set and exists)
    2. DEFAULT_NUTRITION_DATA_PATH
    """
    env_path = os.getenv("NUTRITION_DATA_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    return DEFAULT_NUTRITION_DATA_PATH
