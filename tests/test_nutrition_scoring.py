"""
tests/test_nutrition_scoring.py

Comprehensive unit test suite for the PicWise Deterministic Nutrition Scoring Engine.
Tests all formulas, thresholds, guardrails, conversions, and edge cases in accordance
with phase9D_nutrition_methodology_spec.md and Phase 9D requirements.
"""

import unittest
import math
from backend.services.nutrition_service.scorer import (
    calculate_nutrition_score,
    _calc_logistic_penalty,
    _calc_minkowski_l2_risk,
    _calc_energy_penalty,
    _calc_saturation,
    _calc_unsaturated_fat,
)
from backend.services.nutrition_service.normalization import (
    normalize_nutrition_data,
    detect_product_context,
)
from backend.services.nutrition_service.constants import (
    THRESHOLD_SATURATED_FAT_FOOD,
    THRESHOLD_SATURATED_FAT_CULINARY,
    THRESHOLD_ADDED_SUGAR_FOOD,
    THRESHOLD_ADDED_SUGAR_BEVERAGE,
    THRESHOLD_SODIUM,
    THRESHOLD_TRANS_FAT,
    CATASTROPHIC_CEILING,
)


class TestLogisticRiskPenalty(unittest.TestCase):
    """Tests for the locked logistic risk penalty function: p(x) = 100 / (1 + exp(-3.0 * (x - T)/T))."""

    def test_threshold_value_gives_exactly_fifty(self):
        """Verify p(T) = 50.0 exactly."""
        for t in [4.0, 10.0, 400.0, 0.3]:
            p = _calc_logistic_penalty(t, t)
            self.assertAlmostEqual(p, 50.0, places=4)

    def test_zero_risk_penalty(self):
        """Verify p(0) = 100 / (1 + e^3) ≈ 4.7426."""
        expected_zero_p = 100.0 / (1.0 + math.exp(3.0))
        for t in [4.0, 10.0, 400.0, 0.3]:
            p = _calc_logistic_penalty(0.0, t)
            self.assertAlmostEqual(p, expected_zero_p, places=3)
            self.assertAlmostEqual(p, 4.74, places=1)

    def test_above_threshold_behavior(self):
        """Verify p(x) increases strictly monotonically above threshold."""
        t = 10.0
        p_at_t = _calc_logistic_penalty(10.0, t)
        p_above = _calc_logistic_penalty(15.0, t)
        p_double = _calc_logistic_penalty(20.0, t)
        self.assertGreater(p_above, p_at_t)
        self.assertGreater(p_double, p_above)

    def test_asymptotic_saturation(self):
        """Verify p(x) saturates at 100.0 as x becomes very large."""
        t = 10.0
        p_huge = _calc_logistic_penalty(500.0, t)
        self.assertAlmostEqual(p_huge, 100.0, places=2)
        # Even with astronomical values, no overflow error occurs
        p_astronomical = _calc_logistic_penalty(1e9, t)
        self.assertAlmostEqual(p_astronomical, 100.0, places=2)


class TestMinkowskiL2Aggregation(unittest.TestCase):
    """Tests for the weighted normalized Minkowski L2 / RMS aggregation."""

    def test_l2_rms_formulation_enforces_square_root(self):
        """Verify P_risk = sqrt(mean(p_i^2)) and does NOT omit square root."""
        # 4 penalties of 50.0 should aggregate to exactly 50.0
        penalties = {
            "saturated_fat": 50.0,
            "added_sugars": 50.0,
            "sodium": 50.0,
            "trans_fat": 50.0,
        }
        p_risk = _calc_minkowski_l2_risk(penalties)
        self.assertAlmostEqual(p_risk, 50.0, places=4)

        # Without square root, mean(50^2) = 2500, which would fail this test
        self.assertLessEqual(p_risk, 100.0)

    def test_l2_punishes_single_extreme_outlier(self):
        """Verify L2 RMS weights extreme spikes much higher than arithmetic mean."""
        # 3 zeroes and one 100:
        # Arithmetic mean = (0 + 0 + 0 + 100) / 4 = 25.0
        # L2 RMS = sqrt((0^2 + 0^2 + 0^2 + 100^2)/4) = sqrt(2500) = 50.0
        penalties = {
            "saturated_fat": 0.0,
            "added_sugars": 0.0,
            "sodium": 100.0,
            "trans_fat": 0.0,
        }
        p_risk = _calc_minkowski_l2_risk(penalties)
        self.assertAlmostEqual(p_risk, 50.0, places=4)


class TestPositiveSaturation(unittest.TestCase):
    """Tests for the exponential saturation function f(x, tau) = 100 * (1 - e^(-x/tau))."""

    def test_zero_gives_zero(self):
        self.assertEqual(_calc_saturation(0.0, 4.0), 0.0)
        self.assertEqual(_calc_saturation(-5.0, 4.0), 0.0)

    def test_saturation_at_tau(self):
        """At x = tau, f(tau, tau) = 100 * (1 - 1/e) ≈ 63.21%."""
        expected = 100.0 * (1.0 - math.exp(-1.0))
        self.assertAlmostEqual(_calc_saturation(4.0, 4.0), expected, places=3)
        self.assertAlmostEqual(_calc_saturation(8.0, 8.0), expected, places=3)
        self.assertAlmostEqual(_calc_saturation(12.0, 12.0), expected, places=3)

    def test_saturation_asymptote(self):
        """Diminishing returns as x becomes large."""
        val_1x = _calc_saturation(4.0, 4.0)   # 63.2%
        val_2x = _calc_saturation(8.0, 4.0)   # 86.5%
        val_4x = _calc_saturation(16.0, 4.0)  # 98.2%
        val_huge = _calc_saturation(100.0, 4.0) # ~100.0%
        self.assertGreater(val_2x, val_1x)
        self.assertGreater(val_4x, val_2x)
        self.assertAlmostEqual(val_huge, 100.0, places=2)


class TestEnergyStepwisePenalty(unittest.TestCase):
    """Tests for stepwise energy penalties: <250 -> 0, 250-400 -> 15, >400 -> 30."""

    def test_energy_steps(self):
        self.assertEqual(_calc_energy_penalty(0.0), 0.0)
        self.assertEqual(_calc_energy_penalty(249.0), 0.0)
        self.assertEqual(_calc_energy_penalty(249.9), 0.0)
        self.assertEqual(_calc_energy_penalty(250.0), 15.0)
        self.assertEqual(_calc_energy_penalty(300.0), 15.0)
        self.assertEqual(_calc_energy_penalty(400.0), 15.0)
        self.assertEqual(_calc_energy_penalty(400.1), 30.0)
        self.assertEqual(_calc_energy_penalty(600.0), 30.0)


class TestUnsaturatedFatCalculation(unittest.TestCase):
    """Tests for unsaturated fat calculation."""

    def test_derived_from_total_fat(self):
        norm = {
            "total_fat": {"amount_per_100g": 10.0, "is_missing": False},
            "saturated_fat": {"amount_per_100g": 3.0, "is_missing": False},
            "trans_fat": {"amount_per_100g": 0.5, "is_missing": False},
        }
        # mock object with attributes
        class MockItem:
            def __init__(self, val, missing=False):
                self.amount_per_100g = val
                self.is_missing = missing

        norm_obj = {
            "total_fat": MockItem(10.0),
            "saturated_fat": MockItem(3.0),
            "trans_fat": MockItem(0.5),
        }
        unsat, src = _calc_unsaturated_fat(norm_obj)
        self.assertAlmostEqual(unsat, 6.5, places=2)

    def test_non_negative_clamp(self):
        class MockItem:
            def __init__(self, val, missing=False):
                self.amount_per_100g = val
                self.is_missing = missing

        # Malformed reporting where sat fat > total fat
        norm_obj = {
            "total_fat": MockItem(5.0),
            "saturated_fat": MockItem(8.0),
        }
        unsat, src = _calc_unsaturated_fat(norm_obj)
        self.assertEqual(unsat, 0.0)


class TestSugarGuardrail(unittest.TestCase):
    """Tests for Sugar Guardrail (Added Sugar >= 2.5 * T_sugar => Spos=0, Smicro=0)."""

    def test_food_sugar_guardrail_trigger(self):
        # General food threshold = 10g, 2.5 * 10g = 25.0g
        # Below trigger (24.9g): guardrail not triggered
        data_below = {
            "energy": 200, "protein": 10, "carbohydrate": 30,
            "total_fat": 5, "sodium": 100, "added_sugars": 24.9, "dietary_fibre": 5
        }
        res_below = calculate_nutrition_score(data_below, category="food")
        self.assertFalse(res_below["guardrails"]["sugar_guardrail"])
        self.assertGreater(res_below["components"]["positive_nutrition"], 0.0)

        # Exactly at trigger (25.0g): guardrail triggered
        data_at = {
            "energy": 200, "protein": 10, "carbohydrate": 30,
            "total_fat": 5, "sodium": 100, "added_sugars": 25.0, "dietary_fibre": 5
        }
        res_at = calculate_nutrition_score(data_at, category="food")
        self.assertTrue(res_at["guardrails"]["sugar_guardrail"])
        self.assertEqual(res_at["components"]["positive_nutrition"], 0.0)
        self.assertEqual(res_at["components"]["micronutrient_contribution"], 0.0)

        # Above trigger (30.0g): guardrail triggered
        data_above = {
            "energy": 200, "protein": 10, "carbohydrate": 30,
            "total_fat": 5, "sodium": 100, "added_sugars": 30.0, "dietary_fibre": 5
        }
        res_above = calculate_nutrition_score(data_above, category="food")
        self.assertTrue(res_above["guardrails"]["sugar_guardrail"])
        self.assertEqual(res_above["components"]["positive_nutrition"], 0.0)

    def test_sugar_guardrail_food_boundary_equality(self):
        """
        Verify exact equality boundary for solid food sugar guardrail:
        T_sugar = 10.0g, 2.5 * T_sugar = 25.00g
        24.99g -> no sugar guardrail
        25.00g -> sugar guardrail triggers
        25.01g -> sugar guardrail triggers
        """
        base = {"energy": 200, "protein": 10, "carbohydrate": 30, "total_fat": 5, "sodium": 100, "dietary_fibre": 5}
        
        # 24.99g -> no guardrail
        res_2499 = calculate_nutrition_score({**base, "added_sugars": 24.99}, category="food")
        self.assertFalse(res_2499["guardrails"]["sugar_guardrail"])
        self.assertGreater(res_2499["components"]["positive_nutrition"], 0.0)

        # 25.00g -> triggers guardrail
        res_2500 = calculate_nutrition_score({**base, "added_sugars": 25.00}, category="food")
        self.assertTrue(res_2500["guardrails"]["sugar_guardrail"])
        self.assertEqual(res_2500["components"]["positive_nutrition"], 0.0)
        self.assertEqual(res_2500["components"]["micronutrient_contribution"], 0.0)

        # 25.01g -> triggers guardrail
        res_2501 = calculate_nutrition_score({**base, "added_sugars": 25.01}, category="food")
        self.assertTrue(res_2501["guardrails"]["sugar_guardrail"])
        self.assertEqual(res_2501["components"]["positive_nutrition"], 0.0)
        self.assertEqual(res_2501["components"]["micronutrient_contribution"], 0.0)

    def test_beverage_sugar_guardrail_trigger(self):
        # Beverage threshold = 5g, 2.5 * 5g = 12.5g
        data_bev_below = {
            "energy": 60, "protein": 1, "carbohydrate": 15,
            "total_fat": 0, "sodium": 20, "added_sugars": 12.4
        }
        res_below = calculate_nutrition_score(data_bev_below, category="food", product_text="Fruit Juice beverage (ml)")
        self.assertFalse(res_below["guardrails"]["sugar_guardrail"])

        data_bev_at = {
            "energy": 60, "protein": 1, "carbohydrate": 15,
            "total_fat": 0, "sodium": 20, "added_sugars": 12.5
        }
        res_at = calculate_nutrition_score(data_bev_at, category="food", product_text="Fruit Juice beverage (ml)")
        self.assertTrue(res_at["guardrails"]["sugar_guardrail"])
        self.assertEqual(res_at["components"]["positive_nutrition"], 0.0)

    def test_sugar_guardrail_beverage_boundary_equality(self):
        """
        Verify exact equality boundary for beverage sugar guardrail:
        T_sugar = 5.0g, 2.5 * T_sugar = 12.50g
        12.49g -> no sugar guardrail
        12.50g -> sugar guardrail triggers
        12.51g -> sugar guardrail triggers
        """
        bev_base = {"energy": 60, "protein": 2, "carbohydrate": 15, "total_fat": 0, "sodium": 20, "dietary_fibre": 2}
        bev_context = "Refreshing Fruit Juice drink (ml)"

        # 12.49g -> no guardrail
        res_1249 = calculate_nutrition_score({**bev_base, "added_sugars": 12.49}, category="food", product_text=bev_context)
        self.assertFalse(res_1249["guardrails"]["sugar_guardrail"])
        self.assertGreater(res_1249["components"]["positive_nutrition"], 0.0)

        # 12.50g -> triggers guardrail
        res_1250 = calculate_nutrition_score({**bev_base, "added_sugars": 12.50}, category="food", product_text=bev_context)
        self.assertTrue(res_1250["guardrails"]["sugar_guardrail"])
        self.assertEqual(res_1250["components"]["positive_nutrition"], 0.0)
        self.assertEqual(res_1250["components"]["micronutrient_contribution"], 0.0)

        # 12.51g -> triggers guardrail
        res_1251 = calculate_nutrition_score({**bev_base, "added_sugars": 12.51}, category="food", product_text=bev_context)
        self.assertTrue(res_1251["guardrails"]["sugar_guardrail"])
        self.assertEqual(res_1251["components"]["positive_nutrition"], 0.0)
        self.assertEqual(res_1251["components"]["micronutrient_contribution"], 0.0)


class TestCatastrophicRiskGuardrail(unittest.TestCase):
    """Tests for Catastrophic Risk Guardrail (any primary risk >= 2.5 * T_i => Final Score <= 35.0)."""

    def test_catastrophic_sodium(self):
        # Sodium threshold = 400mg, 2.5 * 400 = 1000mg
        data = {
            "energy": 200, "protein": 30, "dietary_fibre": 20,
            "total_fat": 5, "saturated_fat": 1.0, "sodium": 1000.0, "added_sugars": 0.0
        }
        res = calculate_nutrition_score(data, category="food")
        self.assertTrue(res["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res["nutrition_score"], CATASTROPHIC_CEILING)

    def test_catastrophic_saturated_fat(self):
        # Saturated fat threshold = 4g, 2.5 * 4 = 10.0g
        data = {
            "energy": 350, "protein": 25, "dietary_fibre": 10,
            "total_fat": 20, "saturated_fat": 10.0, "sodium": 150, "added_sugars": 0.0
        }
        res = calculate_nutrition_score(data, category="food")
        self.assertTrue(res["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res["nutrition_score"], CATASTROPHIC_CEILING)

    def test_catastrophic_trans_fat(self):
        # Trans fat threshold = 0.3g, 2.5 * 0.3 = 0.75g
        data = {
            "energy": 250, "protein": 10, "total_carbohydrate": 20,
            "total_fat": 10, "trans_fat": 0.75, "sodium": 150
        }
        res = calculate_nutrition_score(data, category="food")
        self.assertTrue(res["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res["nutrition_score"], CATASTROPHIC_CEILING)

    def test_catastrophic_boundary_saturated_fat_food(self):
        """
        Saturated fat general threshold = 4.0 g/100g, catastrophic = 10.0 g/100g
        9.99g -> no catastrophic trigger
        10.00g -> catastrophic trigger (score capped <= 35.0)
        10.01g -> catastrophic trigger (score capped <= 35.0)
        """
        base = {"energy": 200, "protein": 30, "dietary_fibre": 20, "total_fat": 15, "sodium": 100, "added_sugars": 0}
        
        # 9.99g -> no trigger
        res_999 = calculate_nutrition_score({**base, "saturated_fat": 9.99}, category="food")
        self.assertFalse(res_999["guardrails"]["catastrophic_risk"])
        self.assertGreater(res_999["nutrition_score"], CATASTROPHIC_CEILING)

        # 10.00g -> trigger
        res_1000 = calculate_nutrition_score({**base, "saturated_fat": 10.00}, category="food")
        self.assertTrue(res_1000["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_1000["nutrition_score"], CATASTROPHIC_CEILING)
        self.assertTrue(res_1000["guardrails"]["catastrophic_ceiling_applied"])

        # 10.01g -> trigger
        res_1001 = calculate_nutrition_score({**base, "saturated_fat": 10.01}, category="food")
        self.assertTrue(res_1001["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_1001["nutrition_score"], CATASTROPHIC_CEILING)

    def test_catastrophic_boundary_saturated_fat_culinary(self):
        """
        Saturated fat culinary fat threshold = 20.0 g/100g, catastrophic = 50.0 g/100g
        49.99g -> no catastrophic trigger
        50.00g -> catastrophic trigger
        50.01g -> catastrophic trigger
        """
        base = {"energy": 600, "protein": 2, "total_fat": 80, "sodium": 50, "added_sugars": 0}
        culinary_text = "Pure cooking oil and butter fat"

        # 49.99g -> no trigger
        res_4999 = calculate_nutrition_score({**base, "saturated_fat": 49.99}, category="food", product_text=culinary_text)
        self.assertFalse(res_4999["guardrails"]["catastrophic_risk"])

        # 50.00g -> trigger
        res_5000 = calculate_nutrition_score({**base, "saturated_fat": 50.00}, category="food", product_text=culinary_text)
        self.assertTrue(res_5000["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_5000["nutrition_score"], CATASTROPHIC_CEILING)

        # 50.01g -> trigger
        res_5001 = calculate_nutrition_score({**base, "saturated_fat": 50.01}, category="food", product_text=culinary_text)
        self.assertTrue(res_5001["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_5001["nutrition_score"], CATASTROPHIC_CEILING)

    def test_catastrophic_boundary_added_sugar_food(self):
        """
        Added sugar general threshold = 10.0 g/100g, catastrophic = 25.0 g/100g
        24.99g -> no catastrophic trigger
        25.00g -> catastrophic trigger
        25.01g -> catastrophic trigger
        """
        base = {"energy": 250, "protein": 15, "dietary_fibre": 10, "total_fat": 5, "sodium": 100}

        # 24.99g -> no catastrophic trigger (though high, below 25.00)
        res_2499 = calculate_nutrition_score({**base, "added_sugars": 24.99}, category="food")
        self.assertFalse(res_2499["guardrails"]["catastrophic_risk"])

        # 25.00g -> catastrophic trigger
        res_2500 = calculate_nutrition_score({**base, "added_sugars": 25.00}, category="food")
        self.assertTrue(res_2500["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_2500["nutrition_score"], CATASTROPHIC_CEILING)

        # 25.01g -> catastrophic trigger
        res_2501 = calculate_nutrition_score({**base, "added_sugars": 25.01}, category="food")
        self.assertTrue(res_2501["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_2501["nutrition_score"], CATASTROPHIC_CEILING)

    def test_catastrophic_boundary_added_sugar_beverage(self):
        """
        Added sugar beverage threshold = 5.0 g/100g, catastrophic = 12.5 g/100g
        12.49g -> no catastrophic trigger
        12.50g -> catastrophic trigger
        12.51g -> catastrophic trigger
        """
        bev_base = {"energy": 80, "protein": 2, "total_fat": 0, "sodium": 20}
        bev_text = "Orange juice drink (ml)"

        # 12.49g -> no catastrophic trigger
        res_1249 = calculate_nutrition_score({**bev_base, "added_sugars": 12.49}, category="food", product_text=bev_text)
        self.assertFalse(res_1249["guardrails"]["catastrophic_risk"])

        # 12.50g -> catastrophic trigger
        res_1250 = calculate_nutrition_score({**bev_base, "added_sugars": 12.50}, category="food", product_text=bev_text)
        self.assertTrue(res_1250["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_1250["nutrition_score"], CATASTROPHIC_CEILING)

        # 12.51g -> catastrophic trigger
        res_1251 = calculate_nutrition_score({**bev_base, "added_sugars": 12.51}, category="food", product_text=bev_text)
        self.assertTrue(res_1251["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_1251["nutrition_score"], CATASTROPHIC_CEILING)

    def test_catastrophic_boundary_total_sugar_fallback(self):
        """
        Total sugar fallback food threshold = 12.5 g/100g, catastrophic = 31.25 g/100g
        31.24g -> no catastrophic trigger
        31.25g -> catastrophic trigger
        31.26g -> catastrophic trigger
        """
        base = {"energy": 250, "protein": 10, "total_fat": 5, "sodium": 100}

        # 31.24g -> no catastrophic trigger
        res_3124 = calculate_nutrition_score({**base, "total_sugars": 31.24}, category="food")
        self.assertFalse(res_3124["guardrails"]["catastrophic_risk"])

        # 31.25g -> catastrophic trigger
        res_3125 = calculate_nutrition_score({**base, "total_sugars": 31.25}, category="food")
        self.assertTrue(res_3125["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_3125["nutrition_score"], CATASTROPHIC_CEILING)

        # 31.26g -> catastrophic trigger
        res_3126 = calculate_nutrition_score({**base, "total_sugars": 31.26}, category="food")
        self.assertTrue(res_3126["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_3126["nutrition_score"], CATASTROPHIC_CEILING)

    def test_catastrophic_boundary_sodium(self):
        """
        Sodium threshold = 400.0 mg/100g, catastrophic = 1000.0 mg/100g
        999.9mg -> no catastrophic trigger
        1000.0mg -> catastrophic trigger
        1000.1mg -> catastrophic trigger
        """
        base = {"energy": 200, "protein": 25, "dietary_fibre": 15, "total_fat": 5, "added_sugars": 0}

        # 999.9mg -> no trigger
        res_999 = calculate_nutrition_score({**base, "sodium": 999.9}, category="food")
        self.assertFalse(res_999["guardrails"]["catastrophic_risk"])
        self.assertGreater(res_999["nutrition_score"], CATASTROPHIC_CEILING)

        # 1000.0mg -> trigger
        res_1000 = calculate_nutrition_score({**base, "sodium": 1000.0}, category="food")
        self.assertTrue(res_1000["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_1000["nutrition_score"], CATASTROPHIC_CEILING)

        # 1000.1mg -> trigger
        res_1001 = calculate_nutrition_score({**base, "sodium": 1000.1}, category="food")
        self.assertTrue(res_1001["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_1001["nutrition_score"], CATASTROPHIC_CEILING)

    def test_catastrophic_boundary_trans_fat(self):
        """
        Trans fat threshold = 0.3 g/100g, catastrophic = 0.75 g/100g
        0.74g -> no catastrophic trigger
        0.75g -> catastrophic trigger
        0.76g -> catastrophic trigger
        """
        base = {"energy": 250, "protein": 20, "dietary_fibre": 10, "total_fat": 10, "sodium": 100}

        # 0.74g -> no trigger
        res_074 = calculate_nutrition_score({**base, "trans_fat": 0.74}, category="food")
        self.assertFalse(res_074["guardrails"]["catastrophic_risk"])

        # 0.75g -> trigger
        res_075 = calculate_nutrition_score({**base, "trans_fat": 0.75}, category="food")
        self.assertTrue(res_075["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_075["nutrition_score"], CATASTROPHIC_CEILING)

        # 0.76g -> trigger
        res_076 = calculate_nutrition_score({**base, "trans_fat": 0.76}, category="food")
        self.assertTrue(res_076["guardrails"]["catastrophic_risk"])
        self.assertLessEqual(res_076["nutrition_score"], CATASTROPHIC_CEILING)


class TestMissingVsExplicitZero(unittest.TestCase):
    """Tests for strict separation of Explicit Zero vs Missing values across all core and micronutrients."""

    def test_trans_fat_zero_vs_missing(self):
        # Explicit zero trans fat
        data_zero = {
            "energy": 200, "protein": 5, "total_carbohydrate": 20,
            "total_fat": 5, "trans_fat": 0.0, "sodium": 100
        }
        res_zero = calculate_nutrition_score(data_zero, category="food")
        p_trans_zero = res_zero["risk_details"]["individual_penalties"]["trans_fat"]
        self.assertAlmostEqual(p_trans_zero, 4.74, places=1)
        self.assertNotIn("trans_fat", res_zero["nutrients_missing"])

        # Missing trans fat without hydrogenated oil keywords -> baseline penalty = 10.0
        data_missing = {
            "energy": 200, "protein": 5, "total_carbohydrate": 20,
            "total_fat": 5, "sodium": 100
        }
        res_missing = calculate_nutrition_score(data_missing, category="food")
        p_trans_missing = res_missing["risk_details"]["individual_penalties"]["trans_fat"]
        self.assertEqual(p_trans_missing, 10.0)
        self.assertIn("trans_fat", res_missing["nutrients_missing"])

        # Missing trans fat WITH hydrogenated oil keywords -> severe penalty = 50.0
        res_hydro = calculate_nutrition_score(
            data_missing,
            category="food",
            ingredient_text="Wheat flour, partially hydrogenated vegetable oil, salt"
        )
        p_trans_hydro = res_hydro["risk_details"]["individual_penalties"]["trans_fat"]
        self.assertEqual(p_trans_hydro, 50.0)

    def test_added_sugars_zero_vs_missing_fallback(self):
        # Explicit 0g added sugar
        data_zero = {
            "energy": 150, "protein": 5, "total_carbohydrate": 15,
            "total_fat": 2, "added_sugars": 0.0, "sodium": 100
        }
        res_zero = calculate_nutrition_score(data_zero, category="food")
        p_sugar_zero = res_zero["risk_details"]["individual_penalties"]["added_sugars"]
        self.assertAlmostEqual(p_sugar_zero, 4.74, places=1)
        self.assertNotIn("added_sugars", res_zero["nutrients_missing"])

        # Missing added sugars falls back to total sugars
        data_fallback = {
            "energy": 150, "protein": 5, "total_carbohydrate": 15,
            "total_fat": 2, "total_sugars": 12.5, "sodium": 100
        }
        res_fallback = calculate_nutrition_score(data_fallback, category="food")
        # At fallback threshold (12.5g), penalty should be 50.0
        p_fallback = res_fallback["risk_details"]["individual_penalties"]["added_sugars"]
        self.assertAlmostEqual(p_fallback, 50.0, places=1)
        self.assertIn("added_sugars", res_fallback["nutrients_missing"])

    def test_fiber_zero_vs_missing(self):
        # Explicit 0g fiber: tracked as evaluated with is_explicit_zero=True, not missing
        data_zero = {
            "energy": 200, "protein": 10, "total_carbohydrate": 20,
            "total_fat": 5, "sodium": 100, "dietary_fibre": 0.0
        }
        res_zero = calculate_nutrition_score(data_zero, category="food")
        self.assertNotIn("dietary_fibre", res_zero["nutrients_missing"])
        fiber_eval = [n for n in res_zero["nutrients_evaluated"] if n["nutrient"] == "Dietary Fiber"]
        self.assertEqual(len(fiber_eval), 1)
        self.assertTrue(fiber_eval[0]["is_explicit_zero"])
        self.assertEqual(fiber_eval[0]["amount_per_100g"], 0.0)
        self.assertEqual(fiber_eval[0]["positive_score"], 0.0)

        # Missing fiber: tracked as missing, not in evaluated, contributes 0.0
        data_missing = {
            "energy": 200, "protein": 10, "total_carbohydrate": 20,
            "total_fat": 5, "sodium": 100
        }
        res_missing = calculate_nutrition_score(data_missing, category="food")
        self.assertIn("dietary_fibre", res_missing["nutrients_missing"])
        fiber_missing_eval = [n for n in res_missing["nutrients_evaluated"] if n["nutrient"] == "Dietary Fiber"]
        self.assertEqual(len(fiber_missing_eval), 0)

    def test_protein_zero_vs_missing(self):
        # Explicit 0g protein: tracked with is_explicit_zero=True, not in missing
        data_zero = {
            "energy": 100, "protein": 0.0, "total_carbohydrate": 20,
            "total_fat": 2, "sodium": 50
        }
        res_zero = calculate_nutrition_score(data_zero, category="food")
        self.assertNotIn("protein", res_zero["nutrients_missing"])
        protein_eval = [n for n in res_zero["nutrients_evaluated"] if n["nutrient"] == "Protein"]
        self.assertEqual(len(protein_eval), 1)
        self.assertTrue(protein_eval[0]["is_explicit_zero"])
        self.assertEqual(protein_eval[0]["positive_score"], 0.0)

        # Missing protein: in missing, contributes 0
        data_missing = {
            "energy": 100, "total_carbohydrate": 20,
            "total_fat": 2, "sodium": 50
        }
        res_missing = calculate_nutrition_score(data_missing, category="food")
        self.assertIn("protein", res_missing["nutrients_missing"])
        protein_missing_eval = [n for n in res_missing["nutrients_evaluated"] if n["nutrient"] == "Protein"]
        self.assertEqual(len(protein_missing_eval), 0)

    def test_total_fat_zero_vs_missing(self):
        # Explicit 0g total fat: evaluated with is_explicit_zero=True, not missing
        data_zero = {
            "energy": 100, "protein": 10, "total_carbohydrate": 15,
            "total_fat": 0.0, "sodium": 50
        }
        res_zero = calculate_nutrition_score(data_zero, category="food")
        self.assertNotIn("total_fat", res_zero["nutrients_missing"])
        fat_eval = [n for n in res_zero["nutrients_evaluated"] if n["nutrient"] == "Total Fat"]
        self.assertEqual(len(fat_eval), 1)
        self.assertTrue(fat_eval[0]["is_explicit_zero"])
        self.assertEqual(fat_eval[0]["amount_per_100g"], 0.0)

        # Missing total fat: in missing
        data_missing = {
            "energy": 100, "protein": 10, "total_carbohydrate": 15,
            "sodium": 50
        }
        res_missing = calculate_nutrition_score(data_missing, category="food")
        self.assertIn("total_fat", res_missing["nutrients_missing"])

    def test_saturated_fat_zero_vs_missing(self):
        # Explicit 0g sat fat: evaluated with p ≈ 4.74, is_explicit_zero=True, not missing
        data_zero = {
            "energy": 100, "protein": 5, "total_fat": 5,
            "saturated_fat": 0.0, "sodium": 50
        }
        res_zero = calculate_nutrition_score(data_zero, category="food")
        self.assertNotIn("saturated_fat", res_zero["nutrients_missing"])
        sat_eval = [n for n in res_zero["nutrients_evaluated"] if n["nutrient"] == "Saturated Fat"]
        self.assertEqual(len(sat_eval), 1)
        self.assertTrue(sat_eval[0]["is_explicit_zero"])
        self.assertAlmostEqual(sat_eval[0]["penalty"], 4.74, places=1)

        # Missing sat fat: in missing, not evaluated
        data_missing = {
            "energy": 100, "protein": 5, "total_fat": 5,
            "sodium": 50
        }
        res_missing = calculate_nutrition_score(data_missing, category="food")
        self.assertIn("saturated_fat", res_missing["nutrients_missing"])
        self.assertNotIn("saturated_fat", res_missing["risk_details"]["individual_penalties"])

    def test_sodium_zero_vs_missing(self):
        # Explicit 0mg sodium: evaluated with p ≈ 4.74, is_explicit_zero=True, not missing
        data_zero = {
            "energy": 100, "protein": 5, "total_fat": 2,
            "sodium": 0.0
        }
        res_zero = calculate_nutrition_score(data_zero, category="food")
        self.assertNotIn("sodium", res_zero["nutrients_missing"])
        sod_eval = [n for n in res_zero["nutrients_evaluated"] if n["nutrient"] == "Sodium"]
        self.assertEqual(len(sod_eval), 1)
        self.assertTrue(sod_eval[0]["is_explicit_zero"])
        self.assertAlmostEqual(sod_eval[0]["penalty"], 4.74, places=1)

        # Missing sodium: in missing, not evaluated
        data_missing = {
            "energy": 100, "protein": 5, "total_fat": 2
        }
        res_missing = calculate_nutrition_score(data_missing, category="food")
        self.assertIn("sodium", res_missing["nutrients_missing"])
        self.assertNotIn("sodium", res_missing["risk_details"]["individual_penalties"])

    def test_energy_zero_vs_missing(self):
        # Explicit 0 kcal energy: evaluated with penalty = 0.0, is_explicit_zero=True, not missing
        data_zero = {
            "energy": 0.0, "protein": 0, "total_carbohydrate": 0,
            "total_fat": 0, "sodium": 0
        }
        res_zero = calculate_nutrition_score(data_zero, category="food")
        self.assertNotIn("energy", res_zero["nutrients_missing"])
        energy_eval = [n for n in res_zero["nutrients_evaluated"] if "Energy" in n["nutrient"]]
        self.assertEqual(len(energy_eval), 1)
        self.assertTrue(energy_eval[0]["is_explicit_zero"])
        self.assertEqual(res_zero["energy_penalty"], 0.0)

        # Missing energy: in missing, warning emitted
        data_missing = {
            "protein": 5, "total_carbohydrate": 10,
            "total_fat": 2, "sodium": 50
        }
        res_missing = calculate_nutrition_score(data_missing, category="food")
        self.assertIn("energy", res_missing["nutrients_missing"])
        self.assertTrue(any("Energy" in w for w in res_missing["warnings"]))

    def test_total_sugars_zero_vs_missing(self):
        # Explicit 0g total sugars: recorded with is_explicit_zero=True
        data_zero = {
            "energy": 100, "protein": 5, "total_fat": 2,
            "added_sugars": 2.0, "total_sugars": 0.0, "sodium": 50
        }
        res_zero = calculate_nutrition_score(data_zero, category="food")
        self.assertNotIn("total_sugars", res_zero["nutrients_missing"])
        sugar_eval = [n for n in res_zero["nutrients_evaluated"] if n["nutrient"] == "Total Sugars"]
        self.assertEqual(len(sugar_eval), 1)
        self.assertTrue(sugar_eval[0]["is_explicit_zero"])

        # Missing total sugars: in missing
        data_missing = {
            "energy": 100, "protein": 5, "total_fat": 2,
            "added_sugars": 2.0, "sodium": 50
        }
        res_missing = calculate_nutrition_score(data_missing, category="food")
        self.assertIn("total_sugars", res_missing["nutrients_missing"])

    def test_micronutrients_zero_vs_missing(self):
        # Explicit 0mg calcium: in evaluated with is_explicit_zero=True, awards 0 adequacy
        data_zero = {
            "energy": 100, "protein": 5, "total_fat": 2,
            "sodium": 50, "calcium": 0.0
        }
        res_zero = calculate_nutrition_score(data_zero, category="food")
        calcium_eval = [n for n in res_zero["nutrients_evaluated"] if n["nutrient"] == "Calcium"]
        self.assertEqual(len(calcium_eval), 1)
        self.assertTrue(calcium_eval[0]["is_explicit_zero"])
        self.assertEqual(res_zero["components"]["micronutrient_contribution"], 0.0)

        # Missing calcium: not in evaluated, awards 0 adequacy
        data_missing = {
            "energy": 100, "protein": 5, "total_fat": 2,
            "sodium": 50
        }
        res_missing = calculate_nutrition_score(data_missing, category="food")
        calcium_missing = [n for n in res_missing["nutrients_evaluated"] if n["nutrient"] == "Calcium"]
        self.assertEqual(len(calcium_missing), 0)
        self.assertEqual(res_missing["components"]["micronutrient_contribution"], 0.0)


class TestUnitConversions(unittest.TestCase):
    """Tests for unit conversions (Salt -> Sodium, kJ -> kcal, mg <-> g)."""

    def test_salt_to_sodium_conversion(self):
        # 1g salt = 400mg sodium
        data = {
            "energy": 150, "protein": 5, "total_carbohydrate": 10,
            "total_fat": 2, "salt": 1.0  # 1g salt
        }
        res = calculate_nutrition_score(data, category="food")
        # Sodium penalty at 400mg is threshold -> p = 50.0
        p_sodium = res["risk_details"]["individual_penalties"]["sodium"]
        self.assertAlmostEqual(p_sodium, 50.0, places=1)

    def test_kj_to_kcal_conversion(self):
        # 1046 kJ = 1046 / 4.184 = 250 kcal -> penalty 15.0
        data = {
            "energy": {"value": 1046.0, "unit": "kj"},
            "protein": 5, "total_carbohydrate": 10, "total_fat": 2, "sodium": 100
        }
        res = calculate_nutrition_score(data, category="food")
        self.assertEqual(res["energy_penalty"], 15.0)

    def test_serving_size_scaling(self):
        # 30g serving with 3g protein -> 10g protein per 100g
        data = {
            "energy": {"per_serving": {"value": 60, "unit": "kcal"}},
            "protein": {"per_serving": {"value": 3, "unit": "g"}},
            "total_carbohydrate": {"per_serving": {"value": 6, "unit": "g"}},
            "total_fat": {"per_serving": {"value": 1, "unit": "g"}},
            "sodium": {"per_serving": {"value": 30, "unit": "mg"}},
        }
        res = calculate_nutrition_score(data, category="food", serving_size_grams=30.0)
        self.assertIsNotNone(res["nutrition_score"])
        self.assertEqual(res["status"], "scored")


class TestCompletenessAndCategoryGating(unittest.TestCase):
    """Tests for minimum data completeness and category isolation."""

    def test_personal_care_strictly_returns_none(self):
        data = {"energy": 100, "protein": 5, "sodium": 50, "total_fat": 2}
        res = calculate_nutrition_score(data, category="personal_care")
        self.assertIsNone(res)

    def test_insufficient_data_returns_none_score(self):
        # Only 2 core nutrients (energy, protein)
        data = {"energy": 200, "protein": 10}
        res = calculate_nutrition_score(data, category="food")
        self.assertIsNone(res["nutrition_score"])
        self.assertEqual(res["status"], "Insufficient Nutrition Data")
        self.assertLess(res["nutrition_completeness"], 0.6)

    def test_exactly_three_core_nutrients_succeeds(self):
        # Exactly 3 core nutrients: energy, protein, fat
        data = {"energy": 200, "protein": 10, "total_fat": 5}
        res = calculate_nutrition_score(data, category="food")
        self.assertIsNotNone(res["nutrition_score"])
        self.assertEqual(res["status"], "scored")
        self.assertEqual(res["nutrition_completeness"], 0.6)


class TestWaterOverride(unittest.TestCase):
    """Tests for pure drinking water override to score = 100.0."""

    def test_pure_water_override(self):
        data = {"energy": 0, "total_fat": 0, "sodium": 0, "total_sugars": 0}
        res = calculate_nutrition_score(data, category="food", product_text="Natural Spring Water")
        self.assertEqual(res["nutrition_score"], 100.0)
        self.assertEqual(res["status"], "water_override")
        self.assertTrue(res["guardrails"]["water_override"])


class TestDeterminismAndEdgeCases(unittest.TestCase):
    """Tests for determinism, extreme values, and malformed inputs."""

    def test_determinism_multiple_runs(self):
        data = {
            "energy": 350, "protein": 8.0, "dietary_fibre": 4.0,
            "total_fat": 12.0, "saturated_fat": 2.0, "added_sugars": 5.0,
            "sodium": 200.0, "calcium": 150.0
        }
        res1 = calculate_nutrition_score(data, category="food")
        res2 = calculate_nutrition_score(data, category="food")
        self.assertEqual(res1["nutrition_score"], res2["nutrition_score"])
        self.assertEqual(res1["components"], res2["components"])
        self.assertEqual(res1["risk_details"], res2["risk_details"])

    def test_negative_values_safely_ignored(self):
        # Malformed negative values should be treated as missing rather than crashing
        data = {
            "energy": -100, "protein": 10, "total_carbohydrate": 20,
            "total_fat": 5, "sodium": 100
        }
        res = calculate_nutrition_score(data, category="food")
        self.assertIsNotNone(res["nutrition_score"])
        self.assertEqual(res["energy_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
