"""
tests/test_step_4a_summary_ui.py

Unit and integration tests for STEP 4A:
- Result summary has ONLY three product-level rows (Food Safety, Allergy Risk, Nutrition)
- Followed ONLY by Nutrition Score: NN / 100
- No Food Safety score, no Allergy score, no ingredient breakdowns, no confidence values
- Food Safety worst-case aggregation (Very Safe < Safe < Moderate Risk < High Risk)
- Unrated ingredients never treated as Safe
- Unrated warning surfaced separately
- If no ingredients rated, Food Safety = Unavailable
- Independent dimensions preserved, no overall combined score
"""

import unittest
from backend import create_app
from backend.services.food_status_service import (
    STATUS_GREEN,
    STATUS_YELLOW,
    STATUS_ORANGE,
    STATUS_RED,
    STATUS_UNAVAILABLE,
    map_food_safety_status,
    map_allergy_status,
    map_nutrition_status,
    map_food_analysis_presentation,
)
from backend.services.food_analysis_service.analyzer import analyze_food
from unittest.mock import patch


class TestStep4AResultSummaryUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_upload_html_contains_only_three_summary_rows_and_nutrition_score(self):
        """GET /upload serves result summary with only Food Safety, Allergy Risk, Nutrition, and Nutrition Score."""
        resp = self.client.get("/upload")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        # Summary container
        self.assertIn('id="foodSummaryCard"', html)

        # Three product-level rows
        self.assertIn('Food Safety', html)
        self.assertIn('Allergy Risk', html)
        self.assertIn('Nutrition', html)
        self.assertIn('id="foodSafetyStatusBadge"', html)
        self.assertIn('id="allergyStatusBadge"', html)
        self.assertIn('id="nutritionStatusBadge"', html)

        # Numerical score row
        self.assertIn('Nutrition Score:', html)
        self.assertIn('id="nutritionScoreValue"', html)
        self.assertIn('id="nutritionScoreDenominator"', html)

        # Disallowed scores must NOT be present in summary
        self.assertNotIn('Food Safety Score', html)
        self.assertNotIn('Allergy Score', html)
        self.assertNotIn('Overall Score', html)
        self.assertNotIn('overallScore', html)

    def test_food_safety_worst_case_aggregation_and_unrated_policy(self):
        """Verify worst-case Food Safety calculation and unrated handling."""
        # Case 1: Very Safe < Safe < Moderate Risk < High Risk
        items_safe = [
            {"ocr_text": "UnknownIng1", "matched_name": "UnknownIng1", "method": "exact", "raw_text": "UnknownIng1"},
            {"ocr_text": "UnknownIng2", "matched_name": "UnknownIng2", "method": "exact", "raw_text": "UnknownIng2"},
        ]
        ocr_mock = {
            "domain": "food",
            "ingredients": items_safe,
            "nutrition": None,
            "raw_text": {"all_text": "", "ingredients_text": "", "nutrition_text": ""},
        }

        # Mock predictor to return Safe for UnknownIng2 and Very Safe for UnknownIng1
        def mock_predict(name):
            if "ing2" in name.lower():
                return {"ingredient": name, "risk_class": "Safe", "confidence": 0.9}
            return {"ingredient": name, "risk_class": "Very Safe", "confidence": 0.95}

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", side_effect=mock_predict):
            res = analyze_food(b"dummy_bytes", category="food")
            # Worst of Very Safe and Safe is Safe
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_YELLOW)
            self.assertEqual(res.presentation["food_safety"]["label"], "Safe")

    def test_food_safety_unrated_ingredients_never_treated_as_safe(self):
        """Unrated ingredients do not pull risk down to Safe, and warning is shown separately."""
        items = [
            {"ocr_text": "UnknownChemical", "matched_name": "UnknownChemical", "method": "exact", "raw_text": "UnknownChemical"},
            {"ocr_text": "UnknownIng2", "matched_name": "UnknownIng2", "method": "exact", "raw_text": "UnknownIng2"},
        ]
        ocr_mock = {
            "domain": "food",
            "ingredients": items,
            "nutrition": None,
            "raw_text": {"all_text": "", "ingredients_text": "", "nutrition_text": ""},
        }

        def mock_predict(name):
            if "ing2" in name.lower():
                return {"ingredient": name, "risk_class": "Safe", "confidence": 0.9}
            return {"ingredient": name, "risk_class": None, "confidence": 0.0}

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", side_effect=mock_predict):
            res = analyze_food(b"dummy_bytes", category="food")
            # Overall result is based on rated ingredients ("Safe")
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_YELLOW)
            self.assertEqual(res.presentation["food_safety"]["label"], "Safe")
            # Warning is shown separately
            self.assertTrue(any("could not be evaluated" in w for w in res.warnings))

    def test_food_safety_all_unrated_is_unavailable(self):
        """If no ingredients are rated, Food Safety = Unavailable."""
        items = [
            {"ocr_text": "Mystery1", "matched_name": "Mystery1", "method": "exact", "raw_text": "Mystery1"},
        ]
        ocr_mock = {
            "domain": "food",
            "ingredients": items,
            "nutrition": None,
            "raw_text": {"all_text": "", "ingredients_text": "", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", return_value={"ingredient": "Mystery1", "risk_class": None, "confidence": 0.0}):
            res = analyze_food(b"dummy_bytes", category="food")
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_UNAVAILABLE)
            self.assertIsNone(res.presentation["food_safety"].get("label"))


if __name__ == "__main__":
    unittest.main()
