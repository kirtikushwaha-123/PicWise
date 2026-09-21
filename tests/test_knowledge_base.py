import unittest
from backend.services.knowledge_base import KnowledgeBase, normalize_value, _audit_records


class TestKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.from_env()

    def test_normalization_consistency(self):
        variations = ["Sugar", "sugar", " SUGAR", "Sugar,", "sugar;", "sugar   powder"]
        norm1 = normalize_value("Sugar")
        norm2 = normalize_value("sugar")
        norm3 = normalize_value(" SUGAR")
        norm4 = normalize_value("Sugar,")

        self.assertEqual(norm1, "sugar")
        self.assertEqual(norm2, "sugar")
        self.assertEqual(norm3, "sugar")
        self.assertEqual(norm4, "sugar")

    def test_datasets_loaded(self):
        self.assertGreater(len(self.kb.food), 0, "Food dataset should contain records")
        self.assertGreater(len(self.kb.personal_care), 0, "Personal Care dataset should contain records")
        self.assertGreater(len(self.kb.nutrition), 0, "Nutrition dataset should contain records")

    def test_semantic_none_preservation(self):
        # Find a record with explicit Allergy Risk ("No Risk" or legacy "None")
        food_explicit_rows = [r for r in self.kb.food if r.get("Allergy Risk") in ("No Risk", "None")]
        self.assertGreater(len(food_explicit_rows), 0, "Food dataset contains records with explicit Allergy Risk")
        sample_row = food_explicit_rows[0]
        self.assertIn(sample_row.get("Allergy Risk"), ("No Risk", "None"))

    def test_actual_missing_value_distinguishment(self):
        # Check that empty cells in Personal Care dataset produce Python None, not empty string or "None"
        missing_irritation_rows = [r for r in self.kb.personal_care if r.get("Irritation_Risk") is None]
        self.assertGreater(len(missing_irritation_rows), 0, "Personal Care dataset contains records with missing Irritation_Risk cells")
        sample_missing = missing_irritation_rows[0]
        self.assertIsNone(sample_missing.get("Irritation_Risk"))

    def test_quality_report_generation(self):
        report = self.kb.generate_quality_report()
        self.assertIn("food", report)
        self.assertIn("personal_care", report)
        self.assertIn("nutrition", report)

        food_audit = report["food"]
        self.assertTrue(food_audit["required_columns_present"])
        self.assertGreater(food_audit["row_count"], 0)
        has_no_risk_or_none = (
            "No Risk" in food_audit["allergy_risk_distribution"]
            or "None" in food_audit["allergy_risk_distribution"]
        )
        self.assertTrue(has_no_risk_or_none)

    def test_knowledge_base_allergy_lookup_methods(self):
        # Canonical lookup
        self.assertEqual(self.kb.lookup_food_allergy_risk("Almonds"), "High")
        # Alternate name lookup
        self.assertEqual(self.kb.lookup_food_allergy_risk("badam"), "High")
        # No Risk lookup
        self.assertEqual(self.kb.lookup_food_allergy_risk("Citric Acid"), "No Risk")
        # Unknown lookup
        self.assertIsNone(self.kb.lookup_food_allergy_risk("UnknownItem123"))


if __name__ == "__main__":
    unittest.main()
