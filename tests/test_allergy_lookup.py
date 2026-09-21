"""
tests/test_allergy_lookup.py

Comprehensive unit and integration test suite for PicWise Food Allergy Risk Service (Phase 9G).
Validates deterministic knowledge-base lookup, exact label preservation, UI presentation mapping,
unknown ingredient handling, highest-risk aggregation, category skipping, and error resilience.
"""

import unittest
from unittest.mock import MagicMock

from backend.services.allergy_service import (
    calculate_allergy_risk,
    AllergyResult,
    AllergyIngredientResult,
    RISK_NO_RISK,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_HIGH,
    UI_RISK_LABELS,
    INSUFFICIENT_DATA_LABEL,
    STATUS_SUCCESS,
    STATUS_UNAVAILABLE,
    STATUS_INSUFFICIENT_DATA,
    STATUS_SKIPPED,
    STATUS_NO_INGREDIENTS,
    STATUS_ERROR,
    REASON_NOT_FOUND_IN_KB,
)
from backend.services.knowledge_base import KnowledgeBase


class TestAllergyLookupService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.from_env()

    # ----------------------------------------------------------------------
    # 1. Canonical Name Exact Match Tests
    # ----------------------------------------------------------------------
    def test_canonical_high_risk_match(self):
        """Almonds resolves deterministically to High / High Allergy Risk."""
        res = calculate_allergy_risk(["Almonds"], category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_HIGH)
        self.assertEqual(res.product_ui_label, UI_RISK_LABELS[RISK_HIGH])
        self.assertIn("Almonds", res.allergens_detected)
        self.assertEqual(len(res.ingredients), 1)

        ing = res.ingredients[0]
        self.assertEqual(ing["ingredient"], "Almonds")
        self.assertEqual(ing["allergy_risk"], RISK_HIGH)
        self.assertEqual(ing["ui_label"], UI_RISK_LABELS[RISK_HIGH])
        self.assertEqual(ing["status"], STATUS_SUCCESS)

    def test_canonical_medium_risk_match(self):
        """Soybean resolves deterministically to Medium / Moderate Allergy Risk."""
        res = calculate_allergy_risk(["Soybean"], category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_MEDIUM)
        self.assertEqual(res.product_ui_label, UI_RISK_LABELS[RISK_MEDIUM])
        self.assertIn("Soybean", res.allergens_detected)

        ing = res.ingredients[0]
        self.assertEqual(ing["ingredient"], "Soybean")
        self.assertEqual(ing["allergy_risk"], RISK_MEDIUM)
        self.assertEqual(ing["ui_label"], UI_RISK_LABELS[RISK_MEDIUM])

    def test_canonical_low_risk_match(self):
        """Amylase resolves deterministically to Low / Low Allergy Risk."""
        res = calculate_allergy_risk(["Amylase"], category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_LOW)
        self.assertEqual(res.product_ui_label, UI_RISK_LABELS[RISK_LOW])
        self.assertIn("Amylase", res.allergens_detected)

        ing = res.ingredients[0]
        self.assertEqual(ing["ingredient"], "Amylase")
        self.assertEqual(ing["allergy_risk"], RISK_LOW)
        self.assertEqual(ing["ui_label"], UI_RISK_LABELS[RISK_LOW])

    def test_canonical_no_risk_match(self):
        """Citric Acid resolves deterministically to No Risk / Allergen-Free."""
        res = calculate_allergy_risk(["Citric Acid"], category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_NO_RISK)
        self.assertEqual(res.product_ui_label, UI_RISK_LABELS[RISK_NO_RISK])
        self.assertEqual(res.allergens_detected, [])

        ing = res.ingredients[0]
        self.assertEqual(ing["ingredient"], "Citric Acid")
        self.assertEqual(ing["allergy_risk"], RISK_NO_RISK)
        self.assertEqual(ing["ui_label"], UI_RISK_LABELS[RISK_NO_RISK])

    # ----------------------------------------------------------------------
    # 2. Alternate Name / Packaging Alias Tests
    # ----------------------------------------------------------------------
    def test_alternate_name_match_high(self):
        """'badam' maps to canonical 'Almonds' and resolves to High."""
        res = calculate_allergy_risk(["badam"], category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_HIGH)
        self.assertEqual(res.product_ui_label, "High Allergy Risk")
        self.assertEqual(res.ingredients[0]["ingredient"], "Almonds")
        self.assertEqual(res.ingredients[0]["allergy_risk"], RISK_HIGH)

    def test_alternate_name_match_medium(self):
        """'soya' maps to canonical 'Soybean' and resolves to Medium."""
        res = calculate_allergy_risk(["soya"], category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_MEDIUM)
        self.assertEqual(res.product_ui_label, "Moderate Allergy Risk")
        self.assertEqual(res.ingredients[0]["ingredient"], "Soybean")
        self.assertEqual(res.ingredients[0]["allergy_risk"], RISK_MEDIUM)

    def test_alternate_name_match_low(self):
        """'diastase' maps to canonical 'Amylase' and resolves to Low."""
        res = calculate_allergy_risk(["diastase"], category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_LOW)
        self.assertEqual(res.product_ui_label, "Low Allergy Risk")
        self.assertEqual(res.ingredients[0]["ingredient"], "Amylase")
        self.assertEqual(res.ingredients[0]["allergy_risk"], RISK_LOW)

    def test_alternate_name_match_no_risk(self):
        """'ins 330' maps to canonical 'Citric Acid' and resolves to No Risk."""
        res = calculate_allergy_risk(["ins 330"], category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_NO_RISK)
        self.assertEqual(res.product_ui_label, "Allergen-Free")
        self.assertEqual(res.ingredients[0]["ingredient"], "Citric Acid")
        self.assertEqual(res.ingredients[0]["allergy_risk"], RISK_NO_RISK)

    # ----------------------------------------------------------------------
    # 3. Unknown Ingredient Handling (Strict Unknown != No Risk)
    # ----------------------------------------------------------------------
    def test_unknown_ingredient_is_unavailable_not_no_risk(self):
        """Unknown ingredient must NOT default to No Risk or Allergen-Free."""
        res = calculate_allergy_risk(["UnknownChemicalCompoundXYZ999"], category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_INSUFFICIENT_DATA)
        self.assertIsNone(res.product_risk_level)
        self.assertEqual(res.product_ui_label, INSUFFICIENT_DATA_LABEL)
        self.assertEqual(res.known_ingredients, 0)
        self.assertEqual(res.unknown_ingredients, 1)

        ing = res.ingredients[0]
        self.assertEqual(ing["status"], STATUS_UNAVAILABLE)
        self.assertIsNone(ing.get("allergy_risk"))
        self.assertIsNone(ing.get("ui_label"))
        self.assertEqual(ing["reason"], REASON_NOT_FOUND_IN_KB)

    # ----------------------------------------------------------------------
    # 4. Product-Level Highest-Risk Aggregation Tests
    # ----------------------------------------------------------------------
    def test_product_highest_risk_aggregation_high_wins(self):
        """A product containing High, Medium, Low, and No Risk resolves to High."""
        ingredients = ["Almonds", "Soybean", "Amylase", "Citric Acid"]
        res = calculate_allergy_risk(ingredients, category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_HIGH)
        self.assertEqual(res.product_ui_label, UI_RISK_LABELS[RISK_HIGH])
        self.assertEqual(set(res.allergens_detected), {"Almonds", "Soybean", "Amylase"})

    def test_product_highest_risk_aggregation_medium_wins(self):
        """A product containing Medium, Low, and No Risk resolves to Medium."""
        ingredients = ["Soybean", "Amylase", "Citric Acid"]
        res = calculate_allergy_risk(ingredients, category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_MEDIUM)
        self.assertEqual(res.product_ui_label, UI_RISK_LABELS[RISK_MEDIUM])
        self.assertEqual(set(res.allergens_detected), {"Soybean", "Amylase"})

    def test_product_highest_risk_aggregation_low_wins(self):
        """A product containing Low and No Risk resolves to Low."""
        ingredients = ["Amylase", "Citric Acid"]
        res = calculate_allergy_risk(ingredients, category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_LOW)
        self.assertEqual(res.product_ui_label, UI_RISK_LABELS[RISK_LOW])
        self.assertEqual(res.allergens_detected, ["Amylase"])

    def test_product_highest_risk_aggregation_all_no_risk(self):
        """A product containing only No Risk ingredients resolves to Allergen-Free."""
        ingredients = ["Citric Acid", "Semolina (Suji/Rava)", "Carrageenan"]
        res = calculate_allergy_risk(ingredients, category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_NO_RISK)
        self.assertEqual(res.product_ui_label, UI_RISK_LABELS[RISK_NO_RISK])
        self.assertEqual(res.allergens_detected, [])

    # ----------------------------------------------------------------------
    # 5. Mixed Known & Unknown Ingredients
    # ----------------------------------------------------------------------
    def test_mixed_known_and_unknown_ingredients(self):
        """When some ingredients are unknown, product risk is based on highest known with a warning."""
        ingredients = ["Almonds", "UnknownAdditive123"]
        res = calculate_allergy_risk(ingredients, category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_HIGH)
        self.assertEqual(res.product_ui_label, UI_RISK_LABELS[RISK_HIGH])
        self.assertEqual(res.known_ingredients, 1)
        self.assertEqual(res.unknown_ingredients, 1)
        self.assertEqual(len(res.warnings), 1)
        self.assertIn("could not be matched", res.warnings[0])

    def test_all_unknown_ingredients_yields_insufficient_data(self):
        """When 100% of ingredients are unknown, status is insufficient_data."""
        ingredients = ["MysteryPowderA", "AlienElixirB"]
        res = calculate_allergy_risk(ingredients, category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_INSUFFICIENT_DATA)
        self.assertIsNone(res.product_risk_level)
        self.assertEqual(res.product_ui_label, INSUFFICIENT_DATA_LABEL)
        self.assertEqual(res.known_ingredients, 0)
        self.assertEqual(res.unknown_ingredients, 2)
        self.assertIn("Insufficient allergy data", res.warnings[0])

    # ----------------------------------------------------------------------
    # 6. Category Handling: Personal Care Skips Gracefully
    # ----------------------------------------------------------------------
    def test_personal_care_category_skips_without_raising_error(self):
        """Personal care category skips food allergy analysis gracefully without throwing exceptions."""
        res = calculate_allergy_risk(["Water", "Glycerin"], category="personal_care", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SKIPPED)
        self.assertIsNone(res.product_risk_level)
        self.assertIsNone(res.product_ui_label)
        self.assertEqual(res.allergens_detected, [])
        self.assertIn("skipped", res.warnings[0])

    def test_invalid_category_skips_without_error(self):
        """Any non-food category gracefully skips."""
        for non_food in [None, "", "cosmetics", "industrial", 42]:
            res = calculate_allergy_risk(["Sugar"], category=non_food, knowledge_base=self.kb)
            self.assertEqual(res.status, STATUS_SKIPPED)

    # ----------------------------------------------------------------------
    # 7. Empty Ingredients Handling
    # ----------------------------------------------------------------------
    def test_empty_ingredients_list(self):
        """Empty or None ingredient inputs return status='no_ingredients'."""
        res_empty = calculate_allergy_risk([], category="food", knowledge_base=self.kb)
        self.assertEqual(res_empty.status, STATUS_NO_INGREDIENTS)
        self.assertIsNone(res_empty.product_risk_level)

        res_none = calculate_allergy_risk(None, category="food", knowledge_base=self.kb)
        self.assertEqual(res_none.status, STATUS_NO_INGREDIENTS)

    # ----------------------------------------------------------------------
    # 8. Dict Input Format (OCR Pipeline Compatibility)
    # ----------------------------------------------------------------------
    def test_ocr_structured_dict_inputs(self):
        """Accepts structured OCR ingredient dicts with matched_name and raw_text."""
        ocr_ingredients = [
            {"raw_text": "badam pieces", "matched_name": "Almonds", "method": "fuzzy_kb"},
            {"raw_text": "pure citric acid", "matched_name": "Citric Acid", "method": "exact"},
        ]
        res = calculate_allergy_risk(ocr_ingredients, category="food", knowledge_base=self.kb)
        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_HIGH)
        self.assertEqual(res.product_ui_label, "High Allergy Risk")
        self.assertEqual(len(res.ingredients), 2)
        self.assertEqual(res.ingredients[0]["ingredient"], "Almonds")
        self.assertEqual(res.ingredients[0]["allergy_risk"], RISK_HIGH)
        self.assertEqual(res.ingredients[1]["ingredient"], "Citric Acid")
        self.assertEqual(res.ingredients[1]["allergy_risk"], RISK_NO_RISK)

    # ----------------------------------------------------------------------
    # 9. Component Isolation and Error Resilience
    # ----------------------------------------------------------------------
    def test_corrupted_kb_isolated_as_component_error(self):
        """Unexpected internal exceptions are caught and return status='error' without crashing."""
        broken_kb = MagicMock()
        broken_kb.get_food_ingredient.side_effect = RuntimeError("Database connection timed out")
        del broken_kb.food_index
        del broken_kb.food

        res = calculate_allergy_risk(["Sugar"], category="food", knowledge_base=broken_kb)
        self.assertEqual(res.status, STATUS_ERROR)
        self.assertIsNotNone(res.error)
        self.assertIn("Database connection timed out", res.error)


if __name__ == "__main__":
    unittest.main()
