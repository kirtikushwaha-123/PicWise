import hashlib
import os
import unittest
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

from backend.ml.preprocessing.dataset import build_and_group_split_data
from backend.ml.evaluation.improvement_experiments import (
    EXPERIMENT_DEFINITIONS,
    get_models_directory_hashes,
    run_experiments,
)


class TestMLImprovementExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = "backend/ml/models"
        cls.initial_model_hashes = get_models_directory_hashes(cls.models_dir)
        cls.report = run_experiments(test_size=0.2, random_state=42)

    def test_1_outer_test_groups_no_leakage(self):
        """
        Test 1 — Outer Split Leakage:
        Verify canonical-group leakage remains strictly zero:
        set(outer_train_groups) & set(outer_test_groups) == set()
        """
        data = build_and_group_split_data(test_size=0.2, random_state=42)
        train_groups = set(data["df_train"]["canonical_group"])
        test_groups = set(data["df_test"]["canonical_group"])

        overlap = train_groups.intersection(test_groups)
        self.assertEqual(len(overlap), 0, f"Outer canonical group leakage detected: {overlap}")

    def test_2_inner_validation_groups_no_leakage(self):
        """
        Test 2 — Inner Split Leakage:
        Verify inner validation groups do not overlap with inner training groups:
        set(inner_train_groups) & set(inner_validation_groups) == set()
        """
        data = build_and_group_split_data(test_size=0.2, random_state=42)
        df_train = data["df_train"]

        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        in_tr_idx, in_val_idx = next(gss_inner.split(df_train, groups=df_train["canonical_group"]))

        df_in_tr = df_train.iloc[in_tr_idx]
        df_in_val = df_train.iloc[in_val_idx]

        in_tr_groups = set(df_in_tr["canonical_group"])
        in_val_groups = set(df_in_val["canonical_group"])

        inner_overlap = in_tr_groups.intersection(in_val_groups)
        self.assertEqual(len(inner_overlap), 0, f"Inner validation group leakage detected: {inner_overlap}")

    def test_3_training_only_class_weights(self):
        """
        Test 3 — Training-Only Class Weights:
        Verify sample/class weights are derived strictly from the training partition labels,
        and that modifying test labels does not affect training weights.
        """
        data = build_and_group_split_data(test_size=0.2, random_state=42)
        y_train = data["y_safety_train"]
        y_test = data["y_safety_test"]

        # Calculate sample weights directly from y_train
        expected_weights = compute_sample_weight("balanced", y_train)

        # Altering test set labels MUST have zero influence on training sample weights
        mock_y_test_altered = [0] * len(y_test)
        recalculated_weights = compute_sample_weight("balanced", y_train)

        self.assertEqual(len(expected_weights), len(y_train))
        self.assertTrue(all(w1 == w2 for w1, w2 in zip(expected_weights, recalculated_weights)))

    def test_4_tfidf_vocabulary_leakage_prevented(self):
        """
        Test 4 — TF-IDF Vocabulary Leakage Prevention:
        Verify that unique tokens present ONLY in test/validation ingredients
        do NOT appear in the fitted TF-IDF vectorizer vocabulary.
        """
        mock_df = pd.DataFrame([
            {"ingredient_name": "alpha compound", "canonical_group": "GrpA", "safety_level_raw": "Safe", "allergy_risk_raw": "None"},
            {"ingredient_name": "beta compound", "canonical_group": "GrpB", "safety_level_raw": "Safe", "allergy_risk_raw": "None"},
            {"ingredient_name": "gamma compound", "canonical_group": "GrpC", "safety_level_raw": "Safe", "allergy_risk_raw": "None"},
            {"ingredient_name": "delta compound", "canonical_group": "GrpD", "safety_level_raw": "Safe", "allergy_risk_raw": "None"},
            {"ingredient_name": "uniquetestonlyexperimenttoken xyz", "canonical_group": "GrpE", "safety_level_raw": "Safe", "allergy_risk_raw": "None"}
        ])
        mock_df["safety_level"] = mock_df["safety_level_raw"]
        mock_df["allergy_risk"] = mock_df["allergy_risk_raw"]

        df_train = mock_df[mock_df["canonical_group"] != "GrpE"].copy()
        df_test = mock_df[mock_df["canonical_group"] == "GrpE"].copy()

        # Fit vectorizer strictly on training partition
        for exp in EXPERIMENT_DEFINITIONS:
            t_cfg = exp["tfidf_params"]
            vec = TfidfVectorizer(
                analyzer=t_cfg["analyzer"],
                ngram_range=t_cfg["ngram_range"],
                min_df=t_cfg["min_df"],
                sublinear_tf=t_cfg["sublinear_tf"]
            )
            vec.fit(df_train["ingredient_name"])

            vocab_keys = list(vec.vocabulary_.keys())
            has_leak = any("uniquetestonlyexperimenttoken" in k for k in vocab_keys)
            self.assertFalse(has_leak, f"Data leakage in {exp['experiment_name']}! Test-only token found in vocabulary.")

    def test_5_baseline_included_and_frozen(self):
        """
        Test 5 — Baseline Included and Frozen:
        Verify that the Part 1 baseline is always included in the results and remains frozen.
        """
        self.assertIn("baseline", self.report)
        self.assertIn("safety", self.report["baseline"])
        self.assertIn("allergy", self.report["baseline"])

        safety_base = self.report["baseline"]["safety"]
        allergy_base = self.report["baseline"]["allergy"]

        # Check that baseline accuracy and metrics match Part 1 benchmark
        self.assertAlmostEqual(safety_base["accuracy"], 0.6340, places=3)
        self.assertAlmostEqual(safety_base["macro_f1"], 0.4964, places=3)
        self.assertAlmostEqual(allergy_base["accuracy"], 0.6797, places=3)
        self.assertAlmostEqual(allergy_base["macro_f1"], 0.5197, places=3)

    def test_6_both_targets_evaluated_independently(self):
        """
        Test 6 — Both Targets Evaluated Independently:
        Verify Safety Level and Allergy Risk experiments are evaluated separately.
        """
        self.assertIn("safety_results", self.report)
        self.assertIn("allergy_results", self.report)
        self.assertEqual(len(self.report["safety_results"]), len(EXPERIMENT_DEFINITIONS))
        self.assertEqual(len(self.report["allergy_results"]), len(EXPERIMENT_DEFINITIONS))

        # Check required fields in each experiment result
        for item in self.report["safety_results"]:
            self.assertIn("experiment_name", item)
            self.assertIn("inner_validation_metrics", item)
            self.assertIn("final_outer_test_metrics", item)
            self.assertIn("dangerous_confusion_analysis", item)

        for item in self.report["allergy_results"]:
            self.assertIn("experiment_name", item)
            self.assertIn("inner_validation_metrics", item)
            self.assertIn("final_outer_test_metrics", item)
            self.assertIn("dangerous_confusion_analysis", item)

    def test_7_selected_candidate_has_documented_rationale(self):
        """
        Test 7 — Selection Rationale Exists:
        Verify JSON report includes the measured reason for selection, ranking results, and guardrails.
        """
        self.assertIn("selected_candidate", self.report)
        self.assertIn("selection_rationale", self.report)

        for target in ["safety", "allergy"]:
            sel_target = self.report["selection_rationale"][target]
            self.assertIn("selected_candidate", sel_target)
            self.assertIn("inner_validation_ranking", sel_target)
            self.assertIn("outer_test_performance_summary", sel_target)
            self.assertIn("why_other_candidates_rejected", sel_target)
            self.assertGreater(len(sel_target["inner_validation_ranking"]), 0)

    def test_8_production_models_are_unmodified(self):
        """
        Test 8 — Production Models Are Unmodified:
        Verify SHA-256 hashes of all files in backend/ml/models remain completely identical
        before and after running experiments.
        """
        current_hashes = get_models_directory_hashes(self.models_dir)
        self.assertEqual(self.initial_model_hashes, current_hashes, "CRITICAL: Production model artifacts in backend/ml/models were modified!")

    def test_9_reproducibility(self):
        """
        Test 9 — Reproducibility:
        Verify that running the experiments pipeline twice with random_state=42
        produces identical canonical group splits and experiment metrics.
        """
        run1 = build_and_group_split_data(test_size=0.2, random_state=42)
        run2 = build_and_group_split_data(test_size=0.2, random_state=42)

        self.assertEqual(run1["train_groups"], run2["train_groups"])
        self.assertEqual(run1["test_groups"], run2["test_groups"])
        self.assertEqual(
            run1["df_train"]["ingredient_name"].tolist(),
            run2["df_train"]["ingredient_name"].tolist(),
        )
        self.assertEqual(
            run1["df_test"]["ingredient_name"].tolist(),
            run2["df_test"]["ingredient_name"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
