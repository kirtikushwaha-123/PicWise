"""
tests/test_personal_care_remediation.py

Unit and integration tests validating the four Phase 10D remediations:
1. Personal Care parenthetical INCI alias coverage.
2. OCR line-boundary preservation for dense labels.
3. Low-contrast redundant OCR-pass optimization (conservative short-circuit).
4. Conservative image-quality advisory/handling.
"""

from pathlib import Path
import pytest

from backend.services.personal_care_analysis_service.enrichment import (
    get_personal_care_knowledge_base,
)
from backend.services.personal_care_analysis_service.analyzer import analyze_personal_care
from backend.services.ocr_service.ocr.ensemble import (
    join_ocr_items_with_line_boundaries,
    is_result_sufficiently_complete,
)
from backend.services.ocr_service.nlp.ingredient_corrector import IngredientCorrector


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ==============================================================================
# REMEDIATION 1: PARENTHETICAL INCI ALIAS TESTS
# ==============================================================================

def test_inci_alias_resolution_aqua():
    kb = get_personal_care_knowledge_base()

    res_aqua = kb.lookup("Aqua")
    res_water = kb.lookup("Water")
    res_full = kb.lookup("Aqua (Water)")

    assert res_aqua is not None, "Lookup for 'Aqua' must succeed"
    assert res_water is not None, "Lookup for 'Water' must succeed"
    assert res_full is not None, "Lookup for 'Aqua (Water)' must succeed"

    assert res_aqua.ingredient_name == "Aqua (Water)"
    assert res_water.ingredient_name == "Aqua (Water)"
    assert res_full.ingredient_name == "Aqua (Water)"

    # Features must match identically
    assert res_aqua.primary_function == res_full.primary_function
    assert res_aqua.ingredient_category == res_full.ingredient_category
    assert res_water.primary_function == res_full.primary_function


def test_inci_alias_resolution_tocopherol():
    kb = get_personal_care_knowledge_base()

    res_toco = kb.lookup("Tocopherol")
    res_vite = kb.lookup("Vitamin E")
    res_full = kb.lookup("Tocopherol (Vitamin E)")

    assert res_toco is not None, "Lookup for 'Tocopherol' must succeed"
    assert res_vite is not None, "Lookup for 'Vitamin E' must succeed"
    assert res_full is not None, "Lookup for 'Tocopherol (Vitamin E)' must succeed"

    assert res_toco.ingredient_name == "Tocopherol (Vitamin E)"
    assert res_vite.ingredient_name == "Tocopherol (Vitamin E)"
    assert res_full.ingredient_name == "Tocopherol (Vitamin E)"


def test_inci_alias_unrelated_lookup():
    kb = get_personal_care_knowledge_base()

    # Completely non-existent ingredient should remain None
    assert kb.lookup("PhonyNonExistentChemical999") is None

    # Standard non-parenthetical ingredient remains intact
    res_glyc = kb.lookup("Glycerin")
    assert res_glyc is not None
    assert res_glyc.ingredient_name == "Glycerin"


# ==============================================================================
# REMEDIATION 2: OCR LINE BOUNDARY PRESERVATION TESTS
# ==============================================================================

def test_ocr_line_boundaries_multiline_without_commas():
    # Items on separate vertical lines without trailing commas
    items = [
        {"text": "Cetearyl Alcohol", "bbox": [[10, 20], [200, 20], [200, 40], [10, 40]]},
        {"text": "Dimethicone", "bbox": [[10, 60], [150, 60], [150, 80], [10, 80]]},
        {"text": "Tocopherol", "bbox": [[10, 100], [140, 100], [140, 120], [10, 120]]},
    ]

    joined = join_ocr_items_with_line_boundaries(items)
    assert "\n" in joined
    lines = joined.split("\n")
    assert len(lines) == 3
    assert lines[0] == "Cetearyl Alcohol"
    assert lines[1] == "Dimethicone"
    assert lines[2] == "Tocopherol"

    corrector = IngredientCorrector()
    tokens = corrector.split_phrases(joined)
    assert len(tokens) == 3
    assert "cetearyl alcohol" in tokens
    assert "dimethicone" in tokens
    assert "tocopherol" in tokens


def test_ocr_line_boundaries_same_line_preserved():
    # Items on the SAME visual line (e.g. split word bounding boxes)
    items = [
        {"text": "Sodium", "bbox": [[10, 100], [50, 100], [50, 120], [10, 120]]},
        {"text": "Benzoate,", "bbox": [[60, 102], [130, 102], [130, 122], [60, 122]]},
        {"text": "Citric", "bbox": [[10, 140], [50, 140], [50, 160], [10, 160]]},
        {"text": "Acid", "bbox": [[60, 140], [100, 140], [100, 160], [60, 160]]},
    ]

    joined = join_ocr_items_with_line_boundaries(items)
    lines = joined.split("\n")
    assert len(lines) == 2
    assert lines[0] == "Sodium Benzoate,"
    assert lines[1] == "Citric Acid"

    corrector = IngredientCorrector()
    tokens = corrector.split_phrases(joined)
    assert "sodium benzoate" in tokens
    assert "citric acid" in tokens
    assert "sodium" not in tokens
    assert "benzoate" not in tokens


def test_ocr_line_boundaries_comma_separated_preserved():
    single_line_csv = "Aqua, Glycerin, Dimethicone, Citric Acid"
    corrector = IngredientCorrector()
    tokens = corrector.split_phrases(single_line_csv)
    assert len(tokens) == 4
    assert tokens == ["aqua", "glycerin", "dimethicone", "citric acid"]


# ==============================================================================
# REMEDIATION 3: LOW-CONTRAST CONSERVATIVE SHORT-CIRCUIT TESTS
# ==============================================================================

def test_low_contrast_short_circuit_safe_condition():
    # Valid strong result with delimiters and multiple items
    items = [
        {"text": "Aqua, Glycerin, Cetearyl Alcohol,", "confidence": 0.95},
        {"text": "Dimethicone, Tocopherol, Parfum.", "confidence": 0.93},
    ]
    text = "Aqua, Glycerin, Cetearyl Alcohol,\nDimethicone, Tocopherol, Parfum."
    conf = 0.94

    is_complete = is_result_sufficiently_complete(items, text, conf, mode="ingredient", min_conf=0.88)
    assert is_complete is True, "Strong multi-line ingredient result must short-circuit"


def test_low_contrast_short_circuit_weak_condition_continues():
    # Single tiny item with high confidence should NOT short-circuit
    items = [{"text": "Lotion", "confidence": 0.99}]
    text = "Lotion"
    conf = 0.99

    is_complete = is_result_sufficiently_complete(items, text, conf, mode="ingredient", min_conf=0.88)
    assert is_complete is False, "Single word without delimiters/anchor must NOT short-circuit"

    # Low confidence should NOT short-circuit
    low_conf_items = [
        {"text": "Aqua, Glycerin, Cetearyl Alcohol,", "confidence": 0.65},
        {"text": "Dimethicone, Tocopherol.", "confidence": 0.70},
    ]
    is_complete_low = is_result_sufficiently_complete(
        low_conf_items, "Aqua, Glycerin\nDimethicone", 0.675, mode="ingredient", min_conf=0.88
    )
    assert is_complete_low is False, "Low confidence result must NOT short-circuit"


# ==============================================================================
# REMEDIATION 4: CONSERVATIVE IMAGE QUALITY ADVISORY TESTS
# ==============================================================================

def test_image_quality_advisory_severe_blur():
    blurred_path = FIXTURES_DIR / "product_pc_blurred.png"
    if not blurred_path.exists():
        pytest.skip("Fixture product_pc_blurred.png not found")

    with open(blurred_path, "rb") as f:
        img_bytes = f.read()

    result = analyze_personal_care(img_bytes, category="personal_care")
    assert result.success is True
    # Must contain quality advisory warning
    assert result.ocr_quality_warning is not None
    assert "OCR quality may be unreliable" in result.ocr_quality_warning
    assert result.ocr_quality_warning in result.warnings

    # Must remain unavailable, never safe or low-risk
    assert result.personal_care["safety"]["status"] == "unavailable"
    assert result.personal_care["allergy"]["status"] == "unavailable"
    assert result.personal_care["irritation"]["status"] == "unavailable"


def test_image_quality_advisory_clean_image_no_false_warning():
    clean_path = FIXTURES_DIR / "product_personal_care.png"
    if not clean_path.exists():
        pytest.skip("Fixture product_personal_care.png not found")

    with open(clean_path, "rb") as f:
        img_bytes = f.read()

    result = analyze_personal_care(img_bytes, category="personal_care")
    assert result.success is True
    # Clean image must NOT trigger false quality advisory
    assert result.ocr_quality_warning is None
    for w in result.warnings:
        assert "OCR quality may be unreliable" not in w


def test_dense_label_ingredient_separation():
    dense_path = FIXTURES_DIR / "product_pc_dense.png"
    if not dense_path.exists():
        pytest.skip("Fixture product_pc_dense.png not found")

    with open(dense_path, "rb") as f:
        img_bytes = f.read()

    result = analyze_personal_care(img_bytes, category="personal_care")
    assert result.success is True

    recognized_names = [
        ing.get("matched_name")
        for ing in result.personal_care.get("ingredients", [])
        if ing.get("status") == "success"
    ]

    # Dimethicone and Cetearyl Alcohol must be recognized separately
    assert any("Dimethicone" in (name or "") for name in recognized_names), (
        f"Dimethicone must be recognized separately in {recognized_names}"
    )
    assert any("Cetearyl Alcohol" in (name or "") for name in recognized_names), (
        f"Cetearyl Alcohol must be recognized in {recognized_names}"
    )
    assert any("Tocopherol" in (name or "") for name in recognized_names), (
        f"Tocopherol must be recognized in {recognized_names}"
    )
    assert any("Aqua" in (name or "") for name in recognized_names), (
        f"Aqua must be recognized in {recognized_names}"
    )
    assert result.personal_care["recognized_ingredients"] >= 8
