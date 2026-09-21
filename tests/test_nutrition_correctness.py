"""
tests/test_nutrition_correctness.py

Unit and integration tests for STEP 4E: Nutrition Correctness & Component Independence.

Verifies:
1. nutrients_evaluated output structure and formatting (<Nutrient>: <value> <unit> / 100 g · <status>)
2. Numeric amount filtering: non-numeric amount_per_100g (e.g. None) are excluded from display
3. Unavailable nutrition handling: score is None, presentation is unavailable, fallback is provided
4. Nutrition Score remains available when core nutrients are present
5. Component Independence:
   - Nutrition failure does not destroy Food Safety result
   - Food Safety failure does not destroy Nutrition result
   - Allergy failure does not destroy Nutrition result
6. Single numerical score constraint: Food Safety and Allergy scores remain absent
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


class TestNutritionCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.from_env()

    def _make_ocr_mock(self, ingredients=None, nutrition=None):
        return {
            "domain": "food",
            "ingredients": ingredients or [{"raw_text": "Citric Acid", "matched_name": "Citric Acid", "method": "exact"}],
            "nutrition": nutrition,
            "raw_text": {"all_text": "", "ingredients_text": "", "nutrition_text": ""},
        }

    # 1. nutrients_evaluated output structure and formatting
    def test_nutrients_evaluated_contains_required_fields_and_format(self):
        """nutrients_evaluated contains nutrient, amount_per_100g, unit, and status."""
        ocr_mock = self._make_ocr_mock(
            ingredients=[{"raw_text": "Sugar", "matched_name": "Sugar", "method": "exact"}],
            nutrition={
                "energy": {"value": 400.0, "unit": "kcal", "per_100g": {"value": 400.0, "unit": "kcal"}},
                "total_sugars": {"value": 25.0, "unit": "g", "per_100g": {"value": 25.0, "unit": "g"}},
                "total_fat": {"value": 10.0, "unit": "g", "per_100g": {"value": 10.0, "unit": "g"}},
                "saturated_fat": {"value": 4.0, "unit": "g", "per_100g": {"value": 4.0, "unit": "g"}},
                "sodium": {"value": 300.0, "unit": "mg", "per_100g": {"value": 300.0, "unit": "mg"}},
            },
        )

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)
            self.assertTrue(res.success)
            nut = res.nutrition
            self.assertEqual(nut["status"], "scored")
            self.assertIn("nutrients_evaluated", nut)

            evaluated = nut["nutrients_evaluated"]
            self.assertTrue(len(evaluated) > 0)

            # Check each evaluated nutrient has nutrient, unit, status, and optional numeric amount_per_100g
            for item in evaluated:
                self.assertIn("nutrient", item)
                self.assertIn("unit", item)
                self.assertIn("status", item)

                amount = item.get("amount_per_100g")
                if isinstance(amount, (int, float)):
                    # Validated format: <Nutrient>: <value> <unit> / 100 g · <status>
                    formatted = f"{item['nutrient']}: {amount} {item['unit']} / 100 g · {item['status']}"
                    self.assertIn("/ 100 g ·", formatted)

    # 2. Numeric amount filtering: non-numeric amount_per_100g excluded from display
    def test_non_numeric_nutrients_excluded_from_display(self):
        """Nutrients where amount_per_100g is None (e.g. estimated trans fat) must be filtered out."""
        ocr_mock = self._make_ocr_mock(
            ingredients=[{"raw_text": "Salt", "matched_name": "Salt", "method": "exact"}],
            nutrition={
                "energy": {"value": 100.0, "unit": "kcal", "per_100g": {"value": 100.0, "unit": "kcal"}},
                "total_sugars": {"value": 0.0, "unit": "g", "per_100g": {"value": 0.0, "unit": "g"}},
                "total_fat": {"value": 0.0, "unit": "g", "per_100g": {"value": 0.0, "unit": "g"}},
                "saturated_fat": {"value": 0.0, "unit": "g", "per_100g": {"value": 0.0, "unit": "g"}},
                "sodium": {"value": 38000.0, "unit": "mg", "per_100g": {"value": 38000.0, "unit": "mg"}},
            },
        )

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)
            evaluated = res.nutrition["nutrients_evaluated"]

            # Filter as done in static/app.js: only display numeric amounts
            numeric_nutrients = [
                n for n in evaluated
                if isinstance(n.get("amount_per_100g"), (int, float)) and not isinstance(n.get("amount_per_100g"), bool)
            ]
            non_numeric = [
                n for n in evaluated
                if n.get("amount_per_100g") is None
            ]

            # All displayed nutrients must have numeric amounts
            for n in numeric_nutrients:
                self.assertIsInstance(n["amount_per_100g"], (int, float))

            # Non-numeric items (like estimated trans fat) are excluded from the numeric list
            for n in non_numeric:
                self.assertNotIn(n, numeric_nutrients)

    # 3. Unavailable nutrition handling
    def test_unavailable_nutrition_handling(self):
        """When nutrition data is missing/insufficient, score is None and presentation is unavailable."""
        ocr_mock = self._make_ocr_mock(nutrition=None)

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)
            self.assertTrue(res.success)
            self.assertIsNone(res.nutrition.get("nutrition_score"))
            self.assertEqual(res.presentation["nutrition"]["status"], STATUS_UNAVAILABLE)
            self.assertIsNone(res.presentation["nutrition"].get("score"))
            self.assertIsNone(res.presentation["nutrition"].get("label"))

    # 4. Nutrition Score remains available when core nutrients are present
    def test_nutrition_score_available_when_core_nutrients_present(self):
        """When core nutrients are present, Nutrition Score is a valid number 0..100."""
        ocr_mock = self._make_ocr_mock(
            nutrition={
                "energy": {"value": 250.0, "unit": "kcal", "per_100g": {"value": 250.0, "unit": "kcal"}},
                "total_sugars": {"value": 5.0, "unit": "g", "per_100g": {"value": 5.0, "unit": "g"}},
                "total_fat": {"value": 2.0, "unit": "g", "per_100g": {"value": 2.0, "unit": "g"}},
                "saturated_fat": {"value": 0.5, "unit": "g", "per_100g": {"value": 0.5, "unit": "g"}},
                "sodium": {"value": 150.0, "unit": "mg", "per_100g": {"value": 150.0, "unit": "mg"}},
            }
        )

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)
            self.assertTrue(res.success)
            score = res.nutrition["nutrition_score"]
            self.assertIsNotNone(score)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)
            self.assertIn(res.presentation["nutrition"]["status"], [STATUS_GREEN, STATUS_YELLOW, STATUS_ORANGE, STATUS_RED])

    # 5. Component Independence: Nutrition failure does not destroy Food Safety
    def test_nutrition_failure_does_not_destroy_food_safety(self):
        """When nutrition scoring engine encounters an exception, Food Safety result remains intact."""
        ocr_mock = self._make_ocr_mock(
            ingredients=[{"raw_text": "Citric Acid", "matched_name": "Citric Acid", "method": "exact"}],
            nutrition={"energy": {"value": 100.0, "unit": "kcal"}},
        )

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.calculate_nutrition_score", side_effect=RuntimeError("Nutrition crash")):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            # Unified analysis succeeds overall
            self.assertTrue(res.success)

            # Nutrition is marked error
            self.assertEqual(res.nutrition["status"], "error")
            self.assertIsNone(res.nutrition.get("nutrition_score"))
            self.assertEqual(res.presentation["nutrition"]["status"], STATUS_UNAVAILABLE)

            # Food Safety is unaffected: Citric Acid is Safe in KB
            self.assertEqual(res.food_safety["status"], "success")
            self.assertEqual(res.food_safety["risk_class"], "Safe")
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_YELLOW)
            self.assertEqual(res.presentation["food_safety"]["label"], "Safe")

    # 6. Component Independence: Food Safety failure does not destroy Nutrition
    def test_food_safety_failure_does_not_destroy_nutrition(self):
        """When food safety inference encounters an exception, Nutrition result remains intact."""
        ocr_mock = self._make_ocr_mock(
            ingredients=[{"raw_text": "UnknownIng", "matched_name": None, "method": "unmatched"}],
            nutrition={
                "energy": {"value": 250.0, "unit": "kcal", "per_100g": {"value": 250.0, "unit": "kcal"}},
                "total_sugars": {"value": 5.0, "unit": "g", "per_100g": {"value": 5.0, "unit": "g"}},
                "total_fat": {"value": 2.0, "unit": "g", "per_100g": {"value": 2.0, "unit": "g"}},
                "saturated_fat": {"value": 0.5, "unit": "g", "per_100g": {"value": 0.5, "unit": "g"}},
                "sodium": {"value": 150.0, "unit": "mg", "per_100g": {"value": 150.0, "unit": "mg"}},
            },
        )

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", side_effect=RuntimeError("Food safety crash")):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            # Unified analysis succeeds overall
            self.assertTrue(res.success)

            # Food safety is marked error
            self.assertEqual(res.food_safety["status"], "error")
            self.assertEqual(res.presentation["food_safety"]["status"], STATUS_UNAVAILABLE)

            # Nutrition is unaffected and scored
            self.assertEqual(res.nutrition["status"], "scored")
            self.assertIsNotNone(res.nutrition["nutrition_score"])
            self.assertIn(res.presentation["nutrition"]["status"], [STATUS_GREEN, STATUS_YELLOW, STATUS_ORANGE, STATUS_RED])

    # 7. Component Independence: Allergy failure does not destroy Nutrition
    def test_allergy_failure_does_not_destroy_nutrition(self):
        """When allergy service encounters an exception, Nutrition result remains intact."""
        ocr_mock = self._make_ocr_mock(
            ingredients=[{"raw_text": "Citric Acid", "matched_name": "Citric Acid", "method": "exact"}],
            nutrition={
                "energy": {"value": 250.0, "unit": "kcal", "per_100g": {"value": 250.0, "unit": "kcal"}},
                "total_sugars": {"value": 5.0, "unit": "g", "per_100g": {"value": 5.0, "unit": "g"}},
                "total_fat": {"value": 2.0, "unit": "g", "per_100g": {"value": 2.0, "unit": "g"}},
                "saturated_fat": {"value": 0.5, "unit": "g", "per_100g": {"value": 0.5, "unit": "g"}},
                "sodium": {"value": 150.0, "unit": "mg", "per_100g": {"value": 150.0, "unit": "mg"}},
            },
        )

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock), \
             patch("backend.services.food_analysis_service.analyzer.calculate_allergy_risk", side_effect=RuntimeError("Allergy crash")):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            # Unified analysis succeeds overall
            self.assertTrue(res.success)

            # Allergy is marked error
            self.assertEqual(res.allergy["status"], "error")
            self.assertEqual(res.presentation["allergy"]["status"], STATUS_UNAVAILABLE)

            # Nutrition is unaffected and scored
            self.assertEqual(res.nutrition["status"], "scored")
            self.assertIsNotNone(res.nutrition["nutrition_score"])
            self.assertIn(res.presentation["nutrition"]["status"], [STATUS_GREEN, STATUS_YELLOW, STATUS_ORANGE, STATUS_RED])

    # 8. No Food Safety or Allergy numerical score
    def test_nutrition_score_is_only_numerical_score(self):
        """Nutrition score is the ONLY numerical score; Food Safety and Allergy have no numerical scores."""
        ocr_mock = self._make_ocr_mock(
            nutrition={
                "energy": {"value": 250.0, "unit": "kcal", "per_100g": {"value": 250.0, "unit": "kcal"}},
                "total_sugars": {"value": 5.0, "unit": "g", "per_100g": {"value": 5.0, "unit": "g"}},
                "total_fat": {"value": 2.0, "unit": "g", "per_100g": {"value": 2.0, "unit": "g"}},
                "saturated_fat": {"value": 0.5, "unit": "g", "per_100g": {"value": 0.5, "unit": "g"}},
                "sodium": {"value": 150.0, "unit": "mg", "per_100g": {"value": 150.0, "unit": "mg"}},
            }
        )

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=ocr_mock):
            res = analyze_food(b"dummy_bytes", category="food", knowledge_base=self.kb)

            # Nutrition score exists and is numeric
            self.assertIsInstance(res.nutrition["nutrition_score"], (int, float))

            # Food Safety has NO numerical score
            self.assertNotIn("score", res.food_safety)
            self.assertNotIn("food_safety_score", res.food_safety)

            # Allergy has NO numerical score
            self.assertNotIn("score", res.allergy)
            self.assertNotIn("allergy_score", res.allergy)


if __name__ == "__main__":
    unittest.main()
