"""
tests/test_allergy_unknown_guard.py

Tests for STEP 4C: Fix Allergy Unknown != Safe guard.

Verifies:
1. If ANY ingredient cannot be resolved by the allergy knowledge base:
   - it is NOT treated as safe.
   - product is NOT reported as "Allergen-Free" / "No Risk" when all known ingredients are safe.
2. If known ingredients = No Risk AND unknown_ingredients > 0:
   - status = "insufficient_data"
   - product_risk_level = None
   - product_ui_label = "Insufficient Allergy Data"
   - presentation.status = "unavailable" (never green)
   - warning states how many ingredients were unmatched
3. If a known ingredient has Low/Medium/High risk, that known risk still determines
   the product risk even when unknown ingredients exist (e.g. Almonds/Peanuts + unknown -> High).
4. If ALL ingredients are resolved and all are No Risk:
   - product_risk_level = "No Risk"
   - product_ui_label = "Allergen-Free"
   - presentation.status = "green"
5. Presentation status mapping for insufficient data is strictly "unavailable" (never green).
"""

import unittest
from backend.services.allergy_service import (
    calculate_allergy_risk,
    RISK_NO_RISK,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_HIGH,
    UI_RISK_LABELS,
    INSUFFICIENT_DATA_LABEL,
    STATUS_SUCCESS,
    STATUS_INSUFFICIENT_DATA,
    STATUS_UNAVAILABLE,
)
from backend.services.food_status_service import (
    STATUS_GREEN,
    STATUS_RED,
    STATUS_ORANGE,
    STATUS_YELLOW,
    map_allergy_status,
    map_food_analysis_presentation,
)
from backend.services.knowledge_base import KnowledgeBase


class TestAllergyUnknownGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.from_env()

    def test_known_no_risk_plus_unknown_yields_insufficient_data(self):
        """When known ingredients are No Risk but unknown ingredients exist,

        status must be insufficient_data, product_risk_level None, and label Insufficient Allergy Data.
        """
        # Citric Acid is known No Risk; UnknownChemicalXYZ is unmatched
        ingredients = ["Citric Acid", "UnknownChemicalXYZ"]
        res = calculate_allergy_risk(ingredients, category="food", knowledge_base=self.kb)

        self.assertEqual(res.status, STATUS_INSUFFICIENT_DATA)
        self.assertIsNone(res.product_risk_level)
        self.assertEqual(res.product_ui_label, "Insufficient Allergy Data")
        self.assertEqual(res.product_ui_label, INSUFFICIENT_DATA_LABEL)
        self.assertEqual(res.known_ingredients, 1)
        self.assertEqual(res.unknown_ingredients, 1)
        self.assertEqual(res.total_ingredients, 2)
        self.assertEqual(res.allergens_detected, [])

        # Warning must state how many ingredients were unmatched
        self.assertTrue(len(res.warnings) > 0)
        self.assertIn("1 of 2 ingredients could not be matched", res.warnings[0])

        # Presentation status must be unavailable (never green)
        self.assertEqual(res.presentation.status, "unavailable")
        self.assertNotEqual(res.presentation.status, STATUS_GREEN)

    def test_presentation_status_mapping_for_insufficient_data_is_unavailable(self):
        """presentation.status for insufficient_data is strictly 'unavailable' and NEVER 'green'."""
        # Direct mapper call with insufficient_data
        pres = map_allergy_status(None, raw_status=STATUS_INSUFFICIENT_DATA)
        self.assertEqual(pres.status, "unavailable")
        self.assertNotEqual(pres.status, STATUS_GREEN)
        self.assertIsNone(pres.label)
        self.assertIsNone(pres.risk_level)

        # Full presentation mapping call with insufficient_data allergy result
        allergy_dict = {
            "status": "insufficient_data",
            "product_risk_level": None,
            "product_ui_label": "Insufficient Allergy Data",
            "unknown_ingredients": 1,
            "known_ingredients": 1,
        }
        full_pres = map_food_analysis_presentation(
            food_safety="Safe",
            allergy=allergy_dict,
            nutrition=50.0,
        )
        self.assertEqual(full_pres.allergy.status, "unavailable")
        self.assertNotEqual(full_pres.allergy.status, STATUS_GREEN)

    def test_known_high_risk_plus_unknown_preserves_high_risk(self):
        """A known High Risk ingredient determines product risk even when unknown ingredients exist."""
        # Almonds is known High Risk; MysteryPowder99 is unmatched
        ingredients = ["Almonds", "MysteryPowder99"]
        res = calculate_allergy_risk(ingredients, category="food", knowledge_base=self.kb)

        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_HIGH)
        self.assertEqual(res.product_ui_label, UI_RISK_LABELS[RISK_HIGH])
        self.assertIn("Almonds", res.allergens_detected)
        self.assertEqual(res.known_ingredients, 1)
        self.assertEqual(res.unknown_ingredients, 1)

        # Presentation status must be red
        self.assertEqual(res.presentation.status, STATUS_RED)
        self.assertEqual(res.presentation.risk_level, RISK_HIGH)

        # Warning must still state the unmatched ingredient count
        self.assertTrue(any("1 of 2 ingredients could not be matched" in w for w in res.warnings))

    def test_known_medium_and_low_risk_plus_unknown_preserves_known_risk(self):
        """Known Medium and Low risk ingredients determine product risk even when unknown ingredients exist."""
        # Soybean is known Medium Risk
        res_med = calculate_allergy_risk(["Soybean", "MysteryA"], category="food", knowledge_base=self.kb)
        self.assertEqual(res_med.status, STATUS_SUCCESS)
        self.assertEqual(res_med.product_risk_level, RISK_MEDIUM)
        self.assertEqual(res_med.product_ui_label, UI_RISK_LABELS[RISK_MEDIUM])
        self.assertEqual(res_med.presentation.status, STATUS_ORANGE)

        # Amylase is known Low Risk
        res_low = calculate_allergy_risk(["Amylase", "MysteryB"], category="food", knowledge_base=self.kb)
        self.assertEqual(res_low.status, STATUS_SUCCESS)
        self.assertEqual(res_low.product_risk_level, RISK_LOW)
        self.assertEqual(res_low.product_ui_label, UI_RISK_LABELS[RISK_LOW])
        self.assertEqual(res_low.presentation.status, STATUS_YELLOW)

    def test_all_resolved_no_risk_reports_allergen_free(self):
        """When ALL ingredients are resolved and all are No Risk, report Allergen-Free."""
        ingredients = ["Citric Acid", "Semolina (Suji/Rava)", "Carrageenan"]
        res = calculate_allergy_risk(ingredients, category="food", knowledge_base=self.kb)

        self.assertEqual(res.status, STATUS_SUCCESS)
        self.assertEqual(res.product_risk_level, RISK_NO_RISK)
        self.assertEqual(res.product_ui_label, "Allergen-Free")
        self.assertEqual(res.allergens_detected, [])
        self.assertEqual(res.known_ingredients, 3)
        self.assertEqual(res.unknown_ingredients, 0)
        self.assertEqual(res.warnings, [])

        # Presentation status must be green
        self.assertEqual(res.presentation.status, STATUS_GREEN)
        self.assertEqual(res.presentation.label, "Allergen-Free")
        self.assertEqual(res.presentation.risk_level, RISK_NO_RISK)


if __name__ == "__main__":
    unittest.main()
