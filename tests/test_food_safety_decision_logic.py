"""
tests/test_food_safety_decision_logic.py

Unit and integration tests for STEP 4D: Food Safety Decision Logic.

Verifies:
1. KB ingredient -> knowledge_base + confidence 1.0
2. unknown + high-confidence model (>= 0.60) -> model
3. unknown + low-confidence model (< 0.60) -> unrated, risk_class=None, presentation.status="unavailable"
4. all Safe -> overall Safe
5. Safe + Moderate -> overall Moderate Risk
6. Safe + High -> overall High Risk
7. rated + unrated -> worst rated result + warning stating how many ingredients were unrated
8. all unrated -> Unavailable
9. Food Safety numerical score is NEVER added (Nutrition Score is the only numerical score)
10. Preservation of existing API response shape
"""

import unittest
from unittest.mock import patch

from backend.services.food_analysis_service.analyzer import analyze_food
from backend.services.knowledge_base import KnowledgeBase
from backend.services.food_status_service import (
    STATUS_GREEN,
    STATUS_YELLOW,
    STATUS_ORANGE,
    STATUS_RED,
    STATUS_UNAVAILABLE,
)


class TestFoodSafetyDecisionLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.from_env()

    def _make_ocr_mock(self, ingredients):
        return {
            "domain": "food",
            "ingredients": ingredients,
            "nutrition": None,
            "raw_text": {"all_text": "", "ingredients_text": "", "nutrition_text": ""},
        }

    # 1. KB ingredient -> source="knowledge_base", confidence=1.0
    def test_kb_ingredient_uses_kb_safety_level_and_confidence_one(self):
        """Ingredient found in Knowledge Base uses KB Safety Level, source='knowledge_base', and confidence=1.0."""
        # "Ajwain (Carom Seeds)" is in KB with Safety Level: "Very Safe"
        # "Citric Acid" is in KB with Safety Level: "Safe"
        ocr_mock = self._make_ocr_mock([
            {"raw_text": "Ajwain", "matched_name": "Ajwain (Carom Seeds)", "method": "fuzzy"},
            {"raw_text": "Citric Acid", "matched_name": "Citric Acid", "method": "exact"},
        ])

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            self.assertTrue(res.success)
            fs = res.food_safety
            self.assertEqual(fs["status"], "success")
            self.assertEqual(len(fs["ingredients"]), 2)

            ing_ajwain = fs["ingredients"][0]
            self.assertEqual(ing_ajwain["risk_class"], "Very Safe")
            self.assertEqual(ing_ajwain["source"], "knowledge_base")
            self.assertEqual(ing_ajwain["confidence"], 1.0)
            self.assertEqual(ing_ajwain["presentation_status"], STATUS_GREEN)

            ing_citric = fs["ingredients"][1]
            self.assertEqual(ing_citric["risk_class"], "Safe")
            self.assertEqual(ing_citric["source"], "knowledge_base")
            self.assertEqual(ing_citric["confidence"], 1.0)
            self.assertEqual(ing_citric["presentation_status"], STATUS_YELLOW)

    # 2. Unknown + high-confidence model (>= 0.60) -> source="model"
    def test_unknown_ingredient_high_confidence_uses_model(self):
        """Unknown ingredient not in KB with model confidence >= 0.60 is accepted with source='model'."""
        ocr_mock = self._make_ocr_mock([
            {"raw_text": "SyntheticStabilizer99", "matched_name": None, "method": "unmatched"}
        ])

        mock_pred = {
            "ingredient": "SyntheticStabilizer99",
            "risk_class": "Moderate Risk",
            "confidence": 0.85,
            "probabilities": {"Very Safe": 0.05, "Safe": 0.10, "Moderate Risk": 0.85, "High Risk": 0.0},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", return_value=mock_pred):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            ing = res.food_safety["ingredients"][0]
            self.assertEqual(ing["risk_class"], "Moderate Risk")
            self.assertEqual(ing["source"], "model")
            self.assertEqual(ing["confidence"], 0.85)
            self.assertEqual(ing["presentation_status"], STATUS_ORANGE)
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_ORANGE)
            self.assertEqual(res.presentation["food_safety"]["label"], "Moderate Risk")

    # 3. Unknown + low-confidence model (< 0.60) -> source="unrated", risk_class=None, presentation.status="unavailable"
    def test_unknown_ingredient_low_confidence_becomes_unrated(self):
        """Unknown ingredient with model confidence < 0.60 becomes unrated with risk_class=None and presentation='unavailable'."""
        ocr_mock = self._make_ocr_mock([
            {"raw_text": "AmbiguousPowder123", "matched_name": None, "method": "unmatched"}
        ])

        mock_pred = {
            "ingredient": "AmbiguousPowder123",
            "risk_class": "Safe",  # Model guessed Safe, but confidence is low!
            "confidence": 0.45,
            "probabilities": {"Very Safe": 0.20, "Safe": 0.45, "Moderate Risk": 0.25, "High Risk": 0.10},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", return_value=mock_pred):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            ing = res.food_safety["ingredients"][0]
            # Must NEVER become Safe or Very Safe when uncertain
            self.assertIsNone(ing["risk_class"])
            self.assertEqual(ing["source"], "unrated")
            self.assertEqual(ing["confidence"], 0.45)
            self.assertEqual(ing["presentation_status"], STATUS_UNAVAILABLE)
            self.assertIsNone(ing["presentation"].get("label"))

            # Overall product with only this unrated ingredient is Unavailable
            self.assertIsNone(res.food_safety.get("risk_class"))
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_UNAVAILABLE)
            self.assertIsNone(res.presentation["food_safety"].get("label"))

    # 4. All Safe -> overall Safe
    def test_overall_product_all_safe_yields_safe(self):
        """When all rated ingredients are Safe, overall product is Safe (yellow, strictly never green)."""
        ocr_mock = self._make_ocr_mock([
            {"raw_text": "Citric Acid", "matched_name": "Citric Acid", "method": "exact"},
            {"raw_text": "Refined Wheat Flour", "matched_name": "Refined Wheat Flour (Maida)", "method": "exact"},
        ])

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            self.assertEqual(res.food_safety["risk_class"], "Safe")
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_YELLOW)
            self.assertEqual(res.presentation["food_safety"]["label"], "Safe")

    # 5. Safe + Moderate -> overall Moderate Risk
    def test_overall_product_safe_plus_moderate_yields_moderate_risk(self):
        """Safe + Moderate Risk -> overall Moderate Risk (orange)."""
        # Citric Acid (Safe) + Sugar (Moderate Risk)
        ocr_mock = self._make_ocr_mock([
            {"raw_text": "Citric Acid", "matched_name": "Citric Acid", "method": "exact"},
            {"raw_text": "Sugar", "matched_name": "Sugar", "method": "exact"},
        ])

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            self.assertEqual(res.food_safety["risk_class"], "Moderate Risk")
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_ORANGE)
            self.assertEqual(res.presentation["food_safety"]["label"], "Moderate Risk")

    # 6. Safe + High -> overall High Risk
    def test_overall_product_safe_plus_high_yields_high_risk(self):
        """Safe + High Risk -> overall High Risk (red)."""
        # Citric Acid (Safe in KB) + UnknownHigh (High Risk with confidence 0.95)
        ocr_mock = self._make_ocr_mock([
            {"raw_text": "Citric Acid", "matched_name": "Citric Acid", "method": "exact"},
            {"raw_text": "UnknownHighAdditive", "matched_name": None, "method": "unmatched"},
        ])

        def mock_predict(name):
            return {
                "ingredient": name,
                "risk_class": "High Risk",
                "confidence": 0.95,
                "probabilities": {"Very Safe": 0.0, "Safe": 0.0, "Moderate Risk": 0.05, "High Risk": 0.95},
            }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", side_effect=mock_predict):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            self.assertEqual(res.food_safety["risk_class"], "High Risk")
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_RED)
            self.assertEqual(res.presentation["food_safety"]["label"], "High Risk")

    # 7. Rated + unrated -> worst rated result + warning
    def test_overall_product_rated_plus_unrated_preserves_worst_rated_and_adds_warning(self):
        """Rated + unrated ingredients keep the worst rated result and add an unrated warning."""
        # Citric Acid (Safe in KB) + LowConfidenceAdditive (< 0.60 confidence)
        ocr_mock = self._make_ocr_mock([
            {"raw_text": "Citric Acid", "matched_name": "Citric Acid", "method": "exact"},
            {"raw_text": "LowConfidenceAdditive", "matched_name": None, "method": "unmatched"},
        ])

        mock_pred = {
            "ingredient": "LowConfidenceAdditive",
            "risk_class": "Safe",
            "confidence": 0.35,
            "probabilities": {"Very Safe": 0.2, "Safe": 0.35, "Moderate Risk": 0.25, "High Risk": 0.2},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", return_value=mock_pred):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            # Worst rated is Safe (from Citric Acid)
            self.assertEqual(res.food_safety["risk_class"], "Safe")
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_YELLOW)
            self.assertEqual(res.presentation["food_safety"]["label"], "Safe")

            # Warning stating how many ingredients were unrated
            self.assertTrue(len(res.warnings) > 0)
            self.assertTrue(any("1 of 2 ingredients could not be evaluated for food safety" in w for w in res.warnings))
            self.assertTrue(any("1 of 2 ingredients could not be evaluated for food safety" in w for w in res.food_safety["warnings"]))

    # 8. All unrated -> Unavailable
    def test_overall_product_all_unrated_yields_unavailable(self):
        """When all ingredients are unrated (< 0.60 confidence), product Food Safety is Unavailable."""
        ocr_mock = self._make_ocr_mock([
            {"raw_text": "UncertainIng1", "matched_name": None, "method": "unmatched"},
            {"raw_text": "UncertainIng2", "matched_name": None, "method": "unmatched"},
        ])

        def mock_predict(name):
            return {
                "ingredient": name,
                "risk_class": "Safe",
                "confidence": 0.40,
                "probabilities": {"Very Safe": 0.25, "Safe": 0.40, "Moderate Risk": 0.20, "High Risk": 0.15},
            }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", side_effect=mock_predict):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            self.assertIsNone(res.food_safety.get("risk_class"))
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_UNAVAILABLE)
            self.assertIsNone(res.presentation["food_safety"].get("label"))
            self.assertTrue(any("None of the 2 ingredients could be evaluated" in w for w in res.warnings))

    # 9. No Food Safety numerical score
    def test_no_food_safety_score_in_output(self):
        """Food Safety must NOT contain a numerical score; only Nutrition Score is numerical."""
        ocr_mock = self._make_ocr_mock([
            {"raw_text": "Citric Acid", "matched_name": "Citric Acid", "method": "exact"}
        ])

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)
            fs = res.food_safety
            self.assertNotIn("score", fs)
            self.assertNotIn("food_safety_score", fs)
            self.assertNotIn("numerical_score", fs)


if __name__ == "__main__":
    unittest.main()
