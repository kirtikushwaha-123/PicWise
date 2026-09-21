"""
tests/test_real_world_defect_remediation.py

Phase 14: Evidence-Based Real-World Defect Remediation Regression Suite.
Covers:
  - DEF-01: False ingredient recognition from non-ingredient marketing text (pc_15)
  - DEF-02: Food Safety top-level badge product risk_class aggregation
  - DEF-03: Small/dense Personal Care text extraction (pc_09)
  - DEF-04: Dense Food ingredient phrase splitting (dots, bullets, parens)
  - DEF-05: Dual-unit nutrition parsing and normalization (food_12)
"""

import os
import pytest
from backend.services.food_analysis_service.models import FoodSafetyResult
from backend.services.food_analysis_service import analyze_food
from backend.services.personal_care_analysis_service import analyze_personal_care
from backend.services.food_status_service.mapper import map_food_safety_status
from backend.services.ocr_service.nlp.ingredient_corrector import IngredientCorrector
from backend.services.ocr_service.matching.knowledge_base import KnowledgeBase
from backend.services.ocr_service.parsing.nutrition_parser import parse_nutrition
from backend.services.nutrition_service.normalization import (
    _parse_val_unit,
    normalize_nutrition_data,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "validation_set")


# ==============================================================================
# DEF-01: False ingredient recognition from non-ingredient marketing text
# ==============================================================================

def test_def01_marketing_text_not_matched_to_ingredient():
    """Verify that KnowledgeBase.match_ingredient rejects long marketing sentences."""
    kb = KnowledgeBase()
    marketing_text = "maison de parfum rose & oud eau de parfum 100 ml - 3.4 fl. oz"
    res = kb.match_ingredient(marketing_text, domain="personal_care")
    assert res["matched_name"] is None, f"Expected None but got {res['matched_name']}"


def test_def01_blank_perfume_box_has_no_active_risk():
    """Verify pc_15_perfume_box_blank yields 0 ingredients and unavailable presentation status."""
    pc15_path = os.path.join(FIXTURE_DIR, "pc_15_perfume_box_blank.png")
    if not os.path.exists(pc15_path):
        pytest.skip("Fixture pc_15_perfume_box_blank.png not found")

    with open(pc15_path, "rb") as f:
        img_bytes = f.read()

    res = analyze_personal_care(img_bytes, category="personal_care")
    assert res.success
    assert res.category == "personal_care"

    # Must extract 0 ingredients
    ocr_ingredients = (res.ocr or {}).get("ingredients", [])
    assert len(ocr_ingredients) == 0, f"Expected 0 ingredients, got {len(ocr_ingredients)}"

    # Presentation status must be unavailable, never active safe/moderate/high
    pres = res.presentation or {}
    safety_pres = pres.get("personal_care_safety", {}).get("status")
    allergy_pres = pres.get("allergy", {}).get("status")
    irritation_pres = pres.get("irritation", {}).get("status")

    assert safety_pres == "unavailable", f"Expected unavailable, got {safety_pres}"
    assert allergy_pres == "unavailable", f"Expected unavailable, got {allergy_pres}"
    assert irritation_pres == "unavailable", f"Expected unavailable, got {irritation_pres}"


# ==============================================================================
# DEF-02: Food Safety top-level badge product risk_class aggregation
# ==============================================================================

def test_def02_food_safety_result_dataclass_has_risk_class():
    """Verify FoodSafetyResult supports risk_class attribute and serialization."""
    res = FoodSafetyResult(
        status="success",
        ingredients=[{"ingredient": "Sugar", "risk_class": "Safe"}],
        total_ingredients=1,
        risk_class="Safe",
    )
    d = res.to_dict()
    assert d.get("risk_class") == "Safe"


def test_def02_food_safety_product_risk_class_aggregation():
    """Verify worst-case aggregation logic in food analysis."""
    food01_path = os.path.join(FIXTURE_DIR, "food_01_biscuit_oreo_clean.png")
    if not os.path.exists(food01_path):
        pytest.skip("Fixture food_01_biscuit_oreo_clean.png not found")

    with open(food01_path, "rb") as f:
        img_bytes = f.read()

    res = analyze_food(img_bytes, category="food")
    assert res.success
    fs = res.food_safety
    assert fs is not None
    assert fs.get("status") == "success"
    # When ingredients are analyzed, product-level risk_class must not be None
    assert fs.get("risk_class") in ("Very Safe", "Safe", "Moderate Risk", "High Risk")
    # Presentation status must match the risk_class, not unavailable
    pres = res.presentation or {}
    assert pres.get("food_safety", {}).get("status") in ("green", "yellow", "orange", "red")


def test_def02_food_safety_no_ingredients_unavailable():
    """Verify that when no ingredients are present, risk_class is None and status is unavailable."""
    pres = map_food_safety_status(None)
    assert pres.status == "unavailable"
    assert pres.risk_class is None

    food15_path = os.path.join(FIXTURE_DIR, "food_15_food_front_blank.png")
    if os.path.exists(food15_path):
        with open(food15_path, "rb") as f:
            img_bytes = f.read()
        res = analyze_food(img_bytes, category="food")
        assert res.success
        fs = res.food_safety
        assert fs.get("risk_class") is None
        pres = res.presentation or {}
        assert pres.get("food_safety", {}).get("status") == "unavailable"


# ==============================================================================
# DEF-03: Small/dense Personal Care text extraction (pc_09)
# ==============================================================================

def test_def03_pc09_small_text_extracted():
    """Verify pc_09_serum_ordinary_small_text extracts multiple ingredient tokens."""
    pc09_path = os.path.join(FIXTURE_DIR, "pc_09_serum_ordinary_small_text.png")
    if not os.path.exists(pc09_path):
        pytest.skip("Fixture pc_09_serum_ordinary_small_text.png not found")

    with open(pc09_path, "rb") as f:
        img_bytes = f.read()

    res = analyze_personal_care(img_bytes, category="personal_care")
    assert res.success
    ocr_ingredients = (res.ocr or {}).get("ingredients", [])
    assert len(ocr_ingredients) >= 5, f"Expected >= 5 ingredients, got {len(ocr_ingredients)}"

    # Check that key ingredients are present
    names = [(i.get("matched_name") or i.get("ocr_text") or "").lower() for i in ocr_ingredients]
    assert any("aqua" in n or "water" in n for n in names)
    assert any("niacinamide" in n for n in names)
    assert any("glycol" in n for n in names)


# ==============================================================================
# DEF-04: Dense Food ingredient phrase splitting (dots, bullets, parens)
# ==============================================================================

def test_def04_split_phrases_dot_separated():
    """Verify split_phrases separates dot-delimited ingredients."""
    ic = IngredientCorrector()
    text = "Edible Vegetable Oil. Wheat Flour. Iodised Salt. Spices and Condiments."
    tokens = ic.split_phrases(text)
    assert len(tokens) == 4
    assert tokens == ["edible vegetable oil", "wheat flour", "iodised salt", "spices and condiments"]


def test_def04_split_phrases_bullet_separated():
    """Verify split_phrases separates bullet-delimited ingredients."""
    ic = IngredientCorrector()
    text = "Wheat Flour (65%) • Palm Oil • Salt • Sugar"
    tokens = ic.split_phrases(text)
    assert len(tokens) == 4
    assert "wheat flour (65%)" in tokens
    assert "palm oil" in tokens
    assert "salt" in tokens
    assert "sugar" in tokens


def test_def04_split_phrases_dense_compound_with_dots():
    """Verify split_phrases separates dot without space between words."""
    ic = IngredientCorrector()
    text = "Edible Starch.Noodle Powder.Flavour Enhancer (INS 635)"
    tokens = ic.split_phrases(text)
    assert len(tokens) == 3
    assert tokens == ["edible starch", "noodle powder", "flavour enhancer (ins 635)"]


def test_def04_split_phrases_parenthesis_touching_next_word():
    """Verify split_phrases separates closing paren touching next word."""
    ic = IngredientCorrector()
    text = "Aqua(Water)Niacinamide,Pentylene Glycol"
    tokens = ic.split_phrases(text)
    assert len(tokens) == 3
    assert tokens == ["aqua(water)", "niacinamide", "pentylene glycol"]


def test_def04_split_phrases_preserves_decimals():
    """Verify split_phrases does NOT split on decimal points in percentages or amounts."""
    ic = IngredientCorrector()
    text = "0.5% Preservatives, Citric Acid 0.1%, Salt"
    tokens = ic.split_phrases(text)
    assert len(tokens) == 3
    assert tokens[0] == "0.5% preservatives"
    assert tokens[1] == "citric acid 0.1%"
    assert tokens[2] == "salt"


# ==============================================================================
# DEF-05: Dual-unit nutrition parsing and normalization (food_12)
# ==============================================================================

def test_def05_parse_val_unit_dual_units():
    """Verify _parse_val_unit extracts kcal correctly from dual units."""
    cases = [
        ("110 kcal / 460 kj", 110.0, "kcal"),
        ("460 kj / 110 kcal", 110.0, "kcal"),
        ("110 kcal (460 kj)", 110.0, "kcal"),
        ("460 kj (110 kcal)", 110.0, "kcal"),
        ("110 kcal", 110.0, "kcal"),
        ("460 kj", 460.0, "kj"),
        ("110 / 460", 110.0, ""),
    ]
    for val_str, expected_num, expected_unit in cases:
        val, unit, is_zero, is_missing = _parse_val_unit(val_str)
        assert val == expected_num, f"For '{val_str}': expected val {expected_num}, got {val}"
        assert unit == expected_unit, f"For '{val_str}': expected unit '{expected_unit}', got '{unit}'"
        assert not is_missing


def test_def05_nutrition_normalization_dual_units():
    """Verify normalize_nutrition_data correctly normalizes dual-unit energy."""
    raw = {
        "energy": "110 kcal / 460 kJ",
        "total_fat": "0g",
        "total_carbohydrate": "27g",
        "protein": "1g",
        "sodium": "105mg",
    }
    normalized, meta = normalize_nutrition_data(raw)
    assert "energy" in normalized
    assert normalized["energy"].amount_per_100g == 110.0
    assert normalized["energy"].unit == "kcal"
    assert not normalized["energy"].is_missing


def test_def05_food12_energy_drink_analysis():
    """Verify food_12_energy_drink_redbull_units analyzes without nutrition failure."""
    food12_path = os.path.join(FIXTURE_DIR, "food_12_energy_drink_redbull_units.png")
    if not os.path.exists(food12_path):
        pytest.skip("Fixture food_12_energy_drink_redbull_units.png not found")

    with open(food12_path, "rb") as f:
        img_bytes = f.read()

    res = analyze_food(img_bytes, category="food")
    assert res.success
    ocr_nut = (res.ocr or {}).get("nutrition", {})
    # Verify energy value is 110.0 kcal
    assert ocr_nut.get("energy", {}).get("value") == 110.0
    assert ocr_nut.get("energy", {}).get("unit") == "kcal"
    nut = res.nutrition
    assert nut is not None
    assert nut.get("status") in ("scored", "clean", "warning", "partial")
    pres = res.presentation or {}
    nut_pres = pres.get("nutrition", {})
    assert nut_pres.get("status") in ("green", "yellow", "orange", "red")
