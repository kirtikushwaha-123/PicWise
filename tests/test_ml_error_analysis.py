import os
import unittest
from backend.ml.evaluation.error_analysis import run_error_analysis


class TestMLErrorAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.rec_df = run_error_analysis(test_size=0.2, random_state=42)

    def test_error_records_correspond_to_test_data(self):
        # 1. Error records correspond to test data
        self.assertEqual(len(self.rec_df), self.report["dataset_summary"]["test_representations"])
        self.assertGreater(len(self.rec_df), 0)

    def test_actual_and_predicted_labels_valid(self):
        # 2. Actual and predicted labels are valid
        valid_safety = {"High Risk", "Moderate Risk", "Safe", "Very Safe"}
        valid_allergy = {"High", "Medium", "Low", "None"}

        for _, r in self.rec_df.iterrows():
            self.assertIn(r["actual_safety"], valid_safety)
            self.assertIn(r["predicted_safety"], valid_safety)
            self.assertIn(r["actual_allergy"], valid_allergy)
            self.assertIn(r["predicted_allergy"], valid_allergy)

    def test_correctness_fields_consistent(self):
        # 3. Correctness fields are consistent
        for _, r in self.rec_df.iterrows():
            self.assertEqual(r["safety_correct"], (r["actual_safety"] == r["predicted_safety"]))
            self.assertEqual(r["allergy_correct"], (r["actual_allergy"] == r["predicted_allergy"]))

    def test_confidence_values_valid(self):
        # 4. Confidence values are valid floats between 0.0 and 1.0
        for _, r in self.rec_df.iterrows():
            self.assertTrue(0.0 <= r["safety_confidence"] <= 1.0)
            self.assertTrue(0.0 <= r["allergy_confidence"] <= 1.0)

    def test_confusion_counts_reconcile(self):
        # 5. Confusion counts reconcile with reported incorrect count
        safety_incorrect = self.report["safety_analysis"]["total_errors"]
        safety_pair_sum = sum(pair["count"] for pair in self.report["confusion_pairs"]["safety"])
        self.assertEqual(safety_incorrect, safety_pair_sum)

        allergy_incorrect = self.report["allergy_analysis"]["total_errors"]
        allergy_pair_sum = sum(pair["count"] for pair in self.report["confusion_pairs"]["allergy"])
        self.assertEqual(allergy_incorrect, allergy_pair_sum)

    def test_domain_analysis_reconciles(self):
        # 6. Domain analysis counts reconcile with total test representations
        dom = self.report["domain_analysis"]
        total_reps = self.report["dataset_summary"]["test_representations"]
        sum_domain_counts = sum(d["test_count"] for d in dom.values())
        self.assertEqual(total_reps, sum_domain_counts)

    def test_high_confidence_errors_are_actually_incorrect(self):
        # 7. High-confidence errors are actually incorrect
        for err in self.report["high_confidence_errors"]["safety"]:
            self.assertNotEqual(err["actual_label"], err["predicted_label"])

        for err in self.report["high_confidence_errors"]["allergy"]:
            self.assertNotEqual(err["actual_label"], err["predicted_label"])

    def test_json_report_structure(self):
        # 8. JSON report structure contains all required top-level sections
        required_keys = [
            "dataset_summary",
            "safety_analysis",
            "allergy_analysis",
            "confusion_pairs",
            "domain_analysis",
            "representation_analysis",
            "canonical_ingredient_errors",
            "high_confidence_errors",
            "confidence_bins",
            "name_pattern_analysis",
            "cross_model_error_analysis",
            "findings",
            "limitations",
        ]
        for key in required_keys:
            self.assertIn(key, self.report)

    def test_deterministic_error_analysis(self):
        # 9. Deterministic error analysis — repeated runs with random_state=42 produce identical error records and aggregate results
        rep1, df1 = run_error_analysis(test_size=0.2, random_state=42)
        rep2, df2 = run_error_analysis(test_size=0.2, random_state=42)

        self.assertEqual(rep1, rep2)
        self.assertEqual(df1["ingredient_name"].tolist(), df2["ingredient_name"].tolist())
        self.assertEqual(df1["safety_correct"].tolist(), df2["safety_correct"].tolist())
        self.assertEqual(df1["allergy_correct"].tolist(), df2["allergy_correct"].tolist())


if __name__ == "__main__":
    unittest.main()
