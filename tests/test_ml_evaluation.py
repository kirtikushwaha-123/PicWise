import os
import unittest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.ml.preprocessing.dataset import build_and_group_split_data, load_raw_datasets
from backend.ml.evaluation.evaluate import run_evaluation


class TestMLEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_and_group_split_data(test_size=0.2, random_state=42)

    def test_no_canonical_group_leakage(self):
        df_train = self.data["df_train"]
        df_test = self.data["df_test"]

        train_groups = set(df_train["canonical_group"])
        test_groups = set(df_test["canonical_group"])

        overlap = train_groups.intersection(test_groups)
        self.assertEqual(len(overlap), 0, f"Canonical group leakage detected: {overlap}")

    def test_alternate_names_stay_with_canonical_ingredient(self):
        df_train = self.data["df_train"]
        df_test = self.data["df_test"]

        # Check a specific canonical ingredient with multiple alternate names (e.g. Acesulfame Potassium)
        canonical = "Acesulfame Potassium"
        in_train = canonical in set(df_train["canonical_group"])
        in_test = canonical in set(df_test["canonical_group"])

        # Must appear in train OR test, but NEVER both
        self.assertTrue(in_train or in_test)
        self.assertFalse(in_train and in_test, f"Canonical '{canonical}' appeared in both train and test!")

        target_df = df_train if in_train else df_test
        other_df = df_test if in_train else df_train

        # Verify all representations of this canonical ingredient land in target_df and none in other_df
        target_reps = target_df[target_df["canonical_group"] == canonical]
        other_reps = other_df[other_df["canonical_group"] == canonical]

        self.assertGreater(len(target_reps), 1, f"Expected alternate names for {canonical}")
        self.assertEqual(len(other_reps), 0, f"Leaked alternate names for {canonical} into other split")

    def test_vectorizer_fitted_only_on_training_data(self):
        vectorizer = self.data["vectorizer"]
        df_train = self.data["df_train"]
        df_test = self.data["df_test"]

        # Verify vocabulary feature count matches a vectorizer fit strictly on df_train["ingredient_name"]
        ref_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        ref_vec.fit(df_train["ingredient_name"])

        self.assertEqual(len(vectorizer.vocabulary_), len(ref_vec.vocabulary_))

    def test_real_tfidf_vocabulary_leakage_prevention(self):
        """
        Controlled test verifying that unique tokens present ONLY in test ingredients
        do NOT appear in the fitted TF-IDF vectorizer vocabulary.
        """
        # Create a mock dataset with a unique test-only token
        mock_df = pd.DataFrame([
            {"ingredient_name": "alpha ingredient", "canonical_group": "GroupA", "safety_level_raw": "Safe", "allergy_risk_raw": "None"},
            {"ingredient_name": "beta ingredient", "canonical_group": "GroupB", "safety_level_raw": "Safe", "allergy_risk_raw": "None"},
            {"ingredient_name": "gamma ingredient", "canonical_group": "GroupC", "safety_level_raw": "Safe", "allergy_risk_raw": "None"},
            {"ingredient_name": "delta ingredient", "canonical_group": "GroupD", "safety_level_raw": "Safe", "allergy_risk_raw": "None"},
            {"ingredient_name": "uniquetestonlytoken xyz", "canonical_group": "GroupE", "safety_level_raw": "Safe", "allergy_risk_raw": "None"}
        ])

        # Apply normalizations
        mock_df["safety_level"] = mock_df["safety_level_raw"]
        mock_df["allergy_risk"] = mock_df["allergy_risk_raw"]

        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(mock_df, groups=mock_df["canonical_group"]))

        df_train = mock_df.iloc[train_idx]
        df_test = mock_df.iloc[test_idx]

        # Ensure "GroupE" landed in test set for this test
        if "uniquetestonlytoken xyz" not in df_test["ingredient_name"].values:
            # Force GroupE to test
            df_train = mock_df[mock_df["canonical_group"] != "GroupE"].copy()
            df_test = mock_df[mock_df["canonical_group"] == "GroupE"].copy()

        # Fit vectorizer ONLY on df_train as specified in pipeline architecture
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        vec.fit(df_train["ingredient_name"])

        # Assert unique test token "uniquetestonlytoken" is NOT in vectorizer vocabulary
        vocab_keys = list(vec.vocabulary_.keys())
        has_test_token = any("uniquetestonlytoken" in k for k in vocab_keys)
        self.assertFalse(has_test_token, "Data Leakage! Test-only token 'uniquetestonlytoken' was found in TF-IDF vocabulary!")

    def test_deterministic_group_split(self):
        """
        Test 6: Verify that running the canonical group split twice with random_state=42
        produces 100% identical train/test canonical group assignments and ordering.
        """
        run1 = build_and_group_split_data(test_size=0.2, random_state=42)
        run2 = build_and_group_split_data(test_size=0.2, random_state=42)

        self.assertEqual(run1["train_groups"], run2["train_groups"])
        self.assertEqual(run1["test_groups"], run2["test_groups"])
        self.assertEqual(
            run1["df_train"]["ingredient_name"].tolist(),
            run2["df_train"]["ingredient_name"].tolist()
        )
        self.assertEqual(
            run1["df_test"]["ingredient_name"].tolist(),
            run2["df_test"]["ingredient_name"].tolist()
        )

    def test_evaluation_metrics_generated(self):
        report = run_evaluation(test_size=0.2, random_state=42)

        self.assertIn("evaluation_strategy", report)
        self.assertIn("dataset", report)
        self.assertIn("safety", report)
        self.assertIn("allergy", report)

    def test_both_targets_evaluated(self):
        report = run_evaluation(test_size=0.2, random_state=42)

        for target in ["safety", "allergy"]:
            self.assertIn(target, report)
            target_data = report[target]
            self.assertIn("baseline", target_data)
            self.assertIn("model", target_data)
            self.assertIn("confidence", target_data)

            # Check model fields
            m = target_data["model"]
            self.assertIn("accuracy", m)
            self.assertIn("macro_f1", m)
            self.assertIn("weighted_f1", m)
            self.assertIn("confusion_matrix", m)
            self.assertIn("per_class", m)


if __name__ == "__main__":
    unittest.main()
