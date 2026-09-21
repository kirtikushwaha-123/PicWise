import unittest
from backend.services.knowledge_base import KnowledgeBase
from backend.services.ingredient_matching.matcher import (
    match_ingredients,
    CONFIDENCE_EXACT_CANONICAL,
    CONFIDENCE_ALTERNATE_NAME,
    CONFIDENCE_NORMALIZED,
    CONFIDENCE_UNMATCHED,
    MATCH_TYPE_EXACT_CANONICAL,
    MATCH_TYPE_ALTERNATE_NAME,
    MATCH_TYPE_NORMALIZED,
    MATCH_TYPE_NONE,
)


class TestIngredientMatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.from_env()

    def test_food_canonical_match(self):
        text = "Ingredients: Acesulfame Potassium, Whole Wheat Flour (Atta)"
        matches = match_ingredients(text, self.kb)

        matched_names = [m["canonicalName"] for m in matches if m["matched"]]
        self.assertIn("Acesulfame Potassium", matched_names)

        match_ace = next(m for m in matches if m["canonicalName"] == "Acesulfame Potassium")
        self.assertTrue(match_ace["matched"])
        self.assertEqual(match_ace["matchType"], MATCH_TYPE_EXACT_CANONICAL)
        self.assertEqual(match_ace["confidence"], CONFIDENCE_EXACT_CANONICAL)
        self.assertEqual(match_ace["domain"], "food")

    def test_food_alternate_name_match(self):
        text = "Ingredients: INS 950, E950, ace-k"
        matches = match_ingredients(text, self.kb)

        self.assertGreater(len(matches), 0)
        match_item = matches[0]
        self.assertTrue(match_item["matched"])
        self.assertEqual(match_item["canonicalName"], "Acesulfame Potassium")
        self.assertEqual(match_item["originalInput"], "INS 950")
        self.assertEqual(match_item["matchType"], MATCH_TYPE_ALTERNATE_NAME)
        self.assertEqual(match_item["confidence"], CONFIDENCE_ALTERNATE_NAME)

    def test_personal_care_match(self):
        text = "Ingredients: Sodium Lauryl Sulfate, Squalane, Aqua (Water)"
        matches = match_ingredients(text, self.kb)

        matched_names = [m["canonicalName"] for m in matches if m["matched"]]
        self.assertIn("Sodium Lauryl Sulfate", matched_names)

        sls_match = next(m for m in matches if m["canonicalName"] == "Sodium Lauryl Sulfate")
        self.assertTrue(sls_match["matched"])
        self.assertEqual(sls_match["domain"], "personal_care")
        self.assertIsNotNone(sls_match.get("primaryFunction"))

    def test_unknown_ingredient_remains_unmatched(self):
        text = "Ingredients: SomeUnknownIngredientXYZ123"
        matches = match_ingredients(text, self.kb)

        self.assertEqual(len(matches), 1)
        unmatched = matches[0]
        self.assertFalse(unmatched["matched"])
        self.assertIsNone(unmatched["canonicalName"])
        self.assertEqual(unmatched["originalInput"], "SomeUnknownIngredientXYZ123")
        self.assertEqual(unmatched["domain"], "unknown")
        self.assertEqual(unmatched["matchType"], MATCH_TYPE_NONE)
        self.assertEqual(unmatched["confidence"], CONFIDENCE_UNMATCHED)

    def test_false_positive_protection_no_fuzzy_matching(self):
        # Similar/partial terms that are NOT exact canonical or exact alternate names
        text = "Ingredients: acesulfame, tocopher, potassium, sulfate"
        matches = match_ingredients(text, self.kb)

        for match in matches:
            # Partial terms must NOT fuzzy match onto Acesulfame Potassium or Tocopherol
            self.assertFalse(match["matched"], f"Term '{match['originalInput']}' should NOT match via fuzzy guessing")
            self.assertIsNone(match["canonicalName"])
            self.assertEqual(match["matchType"], MATCH_TYPE_NONE)
            self.assertEqual(match["confidence"], CONFIDENCE_UNMATCHED)
            self.assertEqual(match["domain"], "unknown")

    def test_duplicate_inputs_deduplicated(self):
        text = "Ingredients: Acesulfame Potassium, Acesulfame Potassium, ins 950"
        matches = match_ingredients(text, self.kb)
        canonical_names = [m["canonicalName"] for m in matches]

        self.assertEqual(canonical_names.count("Acesulfame Potassium"), 1)

    def test_semantic_none_allergy_risk_preservation(self):
        text = "Ingredients: Acesulfame Potassium"
        matches = match_ingredients(text, self.kb)

        self.assertEqual(len(matches), 1)
        match_item = matches[0]
        self.assertIn(match_item["allergyRisk"], ("No Risk", "None"))
        self.assertIsNotNone(match_item["allergyRisk"])

    def test_complete_source_preservation(self):
        text = "Ingredients: Acesulfame Potassium"
        matches = match_ingredients(text, self.kb)

        self.assertEqual(len(matches), 1)
        match_item = matches[0]
        self.assertIn("category", match_item)
        self.assertIn("safetyLevel", match_item)
        self.assertIn("allergyRisk", match_item)
        self.assertIn("healthImpact", match_item)
        self.assertIn("processingLevel", match_item)
        self.assertIn("regulatoryStatus", match_item)


if __name__ == "__main__":
    unittest.main()
