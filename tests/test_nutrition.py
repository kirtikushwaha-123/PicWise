import unittest
from backend.services.knowledge_base import KnowledgeBase
from backend.services.nutrition_service.lookup import (
    lookup_nutrient_term,
    find_relevant_nutrition,
    CONFIDENCE_EXACT_CANONICAL,
    CONFIDENCE_ALTERNATE_NAME,
)


class TestNutritionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.from_env()

    def test_canonical_nutrient_lookup(self):
        res = lookup_nutrient_term("Energy / Calories", self.kb)
        self.assertTrue(res["matched"])
        self.assertEqual(res["canonicalNutrient"], "Energy / Calories")
        self.assertEqual(res["confidence"], CONFIDENCE_EXACT_CANONICAL)
        self.assertIsNotNone(res["healthRole"])

    def test_alternate_nutrient_lookup(self):
        res = lookup_nutrient_term("kcal", self.kb)
        self.assertTrue(res["matched"])
        self.assertEqual(res["canonicalNutrient"], "Energy / Calories")
        self.assertEqual(res["originalInput"], "kcal")
        self.assertEqual(res["confidence"], CONFIDENCE_ALTERNATE_NAME)

    def test_nutrition_scanning_in_text(self):
        text = "Nutrition Information: Energy 200 kcal, Total Sugar 10g, Protein 5g"
        nutrition = find_relevant_nutrition(text, self.kb)

        nutrients = [n["canonicalNutrient"] for n in nutrition]
        self.assertIn("Energy / Calories", nutrients)
        self.assertIn("Total Sugars", nutrients)
        self.assertIn("Protein", nutrients)


if __name__ == "__main__":
    unittest.main()
