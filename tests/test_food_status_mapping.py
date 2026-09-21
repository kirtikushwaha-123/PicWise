"""
tests/test_food_status_mapping.py

Comprehensive unit test suite for the PicWise Food Analysis presentation-status layer (Phase 9H).
Validates:
- Food Safety ML presentation mapping (Very Safe -> green, Safe -> yellow, Moderate Risk -> orange, High Risk -> red)
- Allergy Risk presentation mapping (No Risk -> green, Low -> yellow, Medium -> orange, High -> red)
- Nutrition presentation mapping (0-25 red, 26-50 orange, 51-75 yellow, 76-100 green)
- Strict boundary, mid-range, missing, and invalid value handling
- Strict independence across the three dimensions (NO overall score/color/verdict)
"""

import unittest

from backend.services.food_status_service import (
    STATUS_GREEN,
    STATUS_ORANGE,
    STATUS_RED,
    STATUS_UNAVAILABLE,
    STATUS_YELLOW,
    map_allergy_status,
    map_food_analysis_presentation,
    map_food_safety_status,
    map_nutrition_status,
)


class TestFoodStatusMapping(unittest.TestCase):
    # ----------------------------------------------------------------------
    # 1. Food Safety Presentation Mapping Tests
    # ----------------------------------------------------------------------
    def test_food_safety_very_safe_maps_to_green(self):
        """Very Safe maps strictly to green with label 'Very Safe'."""
        pres = map_food_safety_status("Very Safe")
        self.assertEqual(pres.status, STATUS_GREEN)
        self.assertEqual(pres.label, "Very Safe")
        self.assertEqual(pres.risk_class, "Very Safe")

    def test_food_safety_safe_maps_to_yellow_never_green(self):
        """Safe maps strictly to yellow, NEVER to green."""
        pres = map_food_safety_status("Safe")
        self.assertEqual(pres.status, STATUS_YELLOW)
        self.assertEqual(pres.label, "Safe")
        self.assertEqual(pres.risk_class, "Safe")
        self.assertNotEqual(pres.status, STATUS_GREEN)

    def test_food_safety_moderate_risk_maps_to_orange(self):
        """Moderate Risk maps strictly to orange with label 'Moderate Risk'."""
        pres = map_food_safety_status("Moderate Risk")
        self.assertEqual(pres.status, STATUS_ORANGE)
        self.assertEqual(pres.label, "Moderate Risk")
        self.assertEqual(pres.risk_class, "Moderate Risk")

    def test_food_safety_high_risk_maps_to_red(self):
        """High Risk maps strictly to red with label 'High Risk'."""
        pres = map_food_safety_status("High Risk")
        self.assertEqual(pres.status, STATUS_RED)
        self.assertEqual(pres.label, "High Risk")
        self.assertEqual(pres.risk_class, "High Risk")

    def test_food_safety_missing_and_invalid_values(self):
        """Missing, empty, or un-recognized risk classes produce status='unavailable'."""
        for invalid_val in [None, "", "   ", "Super Safe", "UnknownRisk", 123, False]:
            pres = map_food_safety_status(invalid_val)
            self.assertEqual(pres.status, STATUS_UNAVAILABLE)
            self.assertIsNone(pres.label)
            self.assertIsNone(pres.risk_class)

    # ----------------------------------------------------------------------
    # 2. Allergy Risk Presentation Mapping Tests
    # ----------------------------------------------------------------------
    def test_allergy_no_risk_maps_to_green(self):
        """No Risk maps strictly to green with presentation label 'Allergen-Free'."""
        pres = map_allergy_status("No Risk")
        self.assertEqual(pres.status, STATUS_GREEN)
        self.assertEqual(pres.label, "Allergen-Free")
        self.assertEqual(pres.risk_level, "No Risk")

    def test_allergy_low_maps_to_yellow(self):
        """Low maps strictly to yellow with presentation label 'Low Allergy Risk'."""
        pres = map_allergy_status("Low")
        self.assertEqual(pres.status, STATUS_YELLOW)
        self.assertEqual(pres.label, "Low Allergy Risk")
        self.assertEqual(pres.risk_level, "Low")

    def test_allergy_medium_maps_to_orange(self):
        """Medium maps strictly to orange with presentation label 'Moderate Allergy Risk'."""
        pres = map_allergy_status("Medium")
        self.assertEqual(pres.status, STATUS_ORANGE)
        self.assertEqual(pres.label, "Moderate Allergy Risk")
        self.assertEqual(pres.risk_level, "Medium")

    def test_allergy_high_maps_to_red(self):
        """High maps strictly to red with presentation label 'High Allergy Risk'."""
        pres = map_allergy_status("High")
        self.assertEqual(pres.status, STATUS_RED)
        self.assertEqual(pres.label, "High Allergy Risk")
        self.assertEqual(pres.risk_level, "High")

    def test_allergy_unavailable_and_insufficient_data(self):
        """unavailable, insufficient_data, and missing inputs produce status='unavailable', NEVER green."""
        for raw_st in ["unavailable", "insufficient_data", "skipped", "error"]:
            pres = map_allergy_status("No Risk", raw_status=raw_st)
            self.assertEqual(pres.status, STATUS_UNAVAILABLE)
            self.assertIsNone(pres.label)
            self.assertNotEqual(pres.status, STATUS_GREEN)

        for invalid_val in [None, "", "   ", "Extreme Risk", 42]:
            pres = map_allergy_status(invalid_val)
            self.assertEqual(pres.status, STATUS_UNAVAILABLE)
            self.assertIsNone(pres.label)
            self.assertNotEqual(pres.status, STATUS_GREEN)

    # ----------------------------------------------------------------------
    # 3. Nutrition Presentation Mapping Tests
    # ----------------------------------------------------------------------
    def test_nutrition_boundary_values(self):
        """Validates exact inclusive boundary values: 0, 25, 26, 50, 51, 75, 76, 100."""
        # 0 -> red (Low Nutrition)
        p0 = map_nutrition_status(0)
        self.assertEqual(p0.status, STATUS_RED)
        self.assertEqual(p0.label, "Low Nutrition")
        self.assertEqual(p0.score, 0.0)

        # 25 -> red (Low Nutrition)
        p25 = map_nutrition_status(25)
        self.assertEqual(p25.status, STATUS_RED)
        self.assertEqual(p25.label, "Low Nutrition")

        # 26 -> orange (Slightly Better Nutrition)
        p26 = map_nutrition_status(26)
        self.assertEqual(p26.status, STATUS_ORANGE)
        self.assertEqual(p26.label, "Slightly Better Nutrition")

        # 50 -> orange (Slightly Better Nutrition)
        p50 = map_nutrition_status(50)
        self.assertEqual(p50.status, STATUS_ORANGE)
        self.assertEqual(p50.label, "Slightly Better Nutrition")

        # 51 -> yellow (Better Nutrition)
        p51 = map_nutrition_status(51)
        self.assertEqual(p51.status, STATUS_YELLOW)
        self.assertEqual(p51.label, "Better Nutrition")

        # 75 -> yellow (Better Nutrition)
        p75 = map_nutrition_status(75)
        self.assertEqual(p75.status, STATUS_YELLOW)
        self.assertEqual(p75.label, "Better Nutrition")

        # 76 -> green (Good Nutrition)
        p76 = map_nutrition_status(76)
        self.assertEqual(p76.status, STATUS_GREEN)
        self.assertEqual(p76.label, "Good Nutrition")

        # 100 -> green (Good Nutrition)
        p100 = map_nutrition_status(100)
        self.assertEqual(p100.status, STATUS_GREEN)
        self.assertEqual(p100.label, "Good Nutrition")
        self.assertEqual(p100.score, 100.0)

    def test_nutrition_mid_range_values(self):
        """Validates mid-range fractional scores: 12.5, 38.0, 63.2, 85.7."""
        p_red = map_nutrition_status(12.5)
        self.assertEqual(p_red.status, STATUS_RED)
        self.assertEqual(p_red.label, "Low Nutrition")
        self.assertEqual(p_red.score, 12.5)

        p_orange = map_nutrition_status(38.0)
        self.assertEqual(p_orange.status, STATUS_ORANGE)
        self.assertEqual(p_orange.label, "Slightly Better Nutrition")

        p_yellow = map_nutrition_status(63.2)
        self.assertEqual(p_yellow.status, STATUS_YELLOW)
        self.assertEqual(p_yellow.label, "Better Nutrition")

        p_green = map_nutrition_status(85.7)
        self.assertEqual(p_green.status, STATUS_GREEN)
        self.assertEqual(p_green.label, "Good Nutrition")

    def test_nutrition_missing_data_is_unavailable_not_zero(self):
        """Missing nutrition score produces status='unavailable', NEVER 0 or red."""
        pres_none = map_nutrition_status(None)
        self.assertEqual(pres_none.status, STATUS_UNAVAILABLE)
        self.assertIsNone(pres_none.score)
        self.assertIsNone(pres_none.label)
        self.assertNotEqual(pres_none.status, STATUS_RED)

        pres_insufficient = map_nutrition_status(None, raw_status="Insufficient Nutrition Data")
        self.assertEqual(pres_insufficient.status, STATUS_UNAVAILABLE)

        pres_err = map_nutrition_status(50.0, raw_status="error")
        self.assertEqual(pres_err.status, STATUS_UNAVAILABLE)

    def test_nutrition_out_of_range_and_invalid(self):
        """Negative scores, scores > 100, booleans, and strings produce status='unavailable'."""
        for invalid_score in [-1.0, -0.01, 100.01, 150.0, True, False, "85", [50]]:
            pres = map_nutrition_status(invalid_score)
            self.assertEqual(pres.status, STATUS_UNAVAILABLE)
            self.assertIsNone(pres.score)
            self.assertIsNone(pres.label)

    # ----------------------------------------------------------------------
    # 4. Independence Test (Non-negotiable requirement)
    # ----------------------------------------------------------------------
    def test_dimension_independence_no_overall_verdict(self):
        """
        Confirms the three dimensions remain completely independent:
        Food Safety = Very Safe (green), Allergy = High (red), Nutrition = 80 (green).
        There is NO overall_status, NO overall_color, and NO overall_score.
        """
        pres = map_food_analysis_presentation(
            food_safety="Very Safe",
            allergy="High",
            nutrition=80.0,
        )

        self.assertEqual(pres.food_safety.status, STATUS_GREEN)
        self.assertEqual(pres.food_safety.label, "Very Safe")

        self.assertEqual(pres.allergy.status, STATUS_RED)
        self.assertEqual(pres.allergy.label, "High Allergy Risk")

        self.assertEqual(pres.nutrition.status, STATUS_GREEN)
        self.assertEqual(pres.nutrition.label, "Good Nutrition")
        self.assertEqual(pres.nutrition.score, 80.0)

        # Confirm serialized dictionary does NOT contain overall attributes
        pres_dict = pres.to_dict()
        self.assertNotIn("overall_status", pres_dict)
        self.assertNotIn("overall_color", pres_dict)
        self.assertNotIn("overall_score", pres_dict)
        self.assertNotIn("product_health_score", pres_dict)
        self.assertNotIn("health_score", pres_dict)

    def test_dimension_independence_mixed_combination(self):
        """
        Another combination:
        Food Safety = Moderate Risk (orange), Allergy = Low (yellow), Nutrition = 17 (red).
        """
        pres = map_food_analysis_presentation(
            food_safety={"risk_class": "Moderate Risk"},
            allergy={"product_risk_level": "Low", "status": "success"},
            nutrition={"nutrition_score": 17.0, "status": "scored"},
        )

        self.assertEqual(pres.food_safety.status, STATUS_ORANGE)
        self.assertEqual(pres.allergy.status, STATUS_YELLOW)
        self.assertEqual(pres.nutrition.status, STATUS_RED)
        self.assertEqual(pres.nutrition.score, 17.0)

        pres_dict = pres.to_dict()
        self.assertNotIn("overall_status", pres_dict)
        self.assertNotIn("overall_color", pres_dict)
        self.assertNotIn("overall_score", pres_dict)


if __name__ == "__main__":
    unittest.main()
