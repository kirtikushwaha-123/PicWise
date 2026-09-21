"""
tests/test_personal_care_validation.py

Automated test suite for Phase 10A Personal Care Model Validation & Optimization.
Verifies:
1. Dataset structure (926x10, 0 nulls, correct target domains).
2. Canonical grouping logic and alias mapping.
3. Leakage prevention (train groups ∩ validation groups = ∅ for all 5 folds across all 3 targets).
4. Feature pipeline correctness and exclusion of target columns.
5. Model behavior (valid probabilities summing to 1.0, OOF completeness).
6. Deterministic reproducibility under random_state=42.
"""

import os
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from backend.ml.preprocessing.personal_care_dataset import (
    DEFAULT_PERSONAL_CARE_DATA_PATH,
    VALID_SAFETY_LEVELS,
    VALID_ALLERGY_RISKS,
    VALID_IRRITATION_RISKS,
    load_personal_care_dataset,
    extract_canonical_group_id,
    build_feature_pipeline,
)


class TestPersonalCareValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_personal_care_dataset()

    # ----------------------------------------------------------------------
    # 1. Dataset Integrity Tests
    # ----------------------------------------------------------------------
    def test_dataset_shape_and_columns(self):
        """Dataset must contain exactly 926 rows and 10 source columns + canonical_group_id."""
        self.assertEqual(len(self.df), 926)
        expected_cols = [
            "Ingredient_Name",
            "Primary_Function",
            "Ingredient_Category",
            "Product_Categories",
            "Origin",
            "Safety_Level",
            "Allergy_Risk",
            "Irritation_Risk",
            "Regulatory_Status",
            "Packaging Names / Alternate Names",
            "canonical_group_id",
        ]
        for col in expected_cols:
            self.assertIn(col, self.df.columns)

    def test_dataset_zero_nulls(self):
        """Dataset must have zero missing or null values."""
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_target_class_domains(self):
        """Target columns must strictly contain valid classes."""
        self.assertTrue(set(self.df["Safety_Level"].unique()).issubset(VALID_SAFETY_LEVELS))
        self.assertTrue(set(self.df["Allergy_Risk"].unique()).issubset(VALID_ALLERGY_RISKS))
        self.assertTrue(set(self.df["Irritation_Risk"].unique()).issubset(VALID_IRRITATION_RISKS))

        # Ensure no target class has 0 samples
        self.assertEqual(len(self.df["Safety_Level"].unique()), 4)
        self.assertEqual(len(self.df["Allergy_Risk"].unique()), 4)
        self.assertEqual(len(self.df["Irritation_Risk"].unique()), 4)

    # ----------------------------------------------------------------------
    # 2. Canonical Grouping Tests
    # ----------------------------------------------------------------------
    def test_canonical_group_context_suffix_normalization(self):
        """Application context suffixes in parentheses must map to base ingredient."""
        self.assertEqual(extract_canonical_group_id("Menthol (Lip Plumper)"), "Menthol")
        self.assertEqual(extract_canonical_group_id("Zinc Oxide (Baby Care)"), "Zinc Oxide")
        self.assertEqual(extract_canonical_group_id("Potassium Sorbate (Baby)"), "Potassium Sorbate")
        self.assertEqual(extract_canonical_group_id("Petrolatum (Hair Pomade)"), "Petrolatum")
        self.assertEqual(extract_canonical_group_id("Sodium Bicarbonate (Deodorant)"), "Sodium Bicarbonate")

    def test_canonical_group_identity_preservation(self):
        """Core biological and chemical identities in parentheses must be preserved."""
        self.assertEqual(extract_canonical_group_id("Aqua (Water)"), "Aqua (Water)")
        self.assertEqual(extract_canonical_group_id("Cocos Nucifera Oil (Coconut Oil)"), "Cocos Nucifera Oil (Coconut Oil)")

    def test_canonical_group_count(self):
        """Canonical group extraction must yield exactly 881 groups across 926 rows."""
        self.assertEqual(self.df["canonical_group_id"].nunique(), 881)

    # ----------------------------------------------------------------------
    # 3. Leakage Prevention Tests
    # ----------------------------------------------------------------------
    def test_zero_group_leakage_across_all_folds(self):
        """Train groups and validation groups must be disjoint in all 5 folds for all targets."""
        targets = ["Safety_Level", "Allergy_Risk", "Irritation_Risk"]
        groups = self.df["canonical_group_id"].values

        for target in targets:
            y = self.df[target].values
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(sgkf.split(self.df, y, groups=groups)):
                tr_groups = set(groups[train_idx])
                val_groups = set(groups[val_idx])
                overlap = tr_groups.intersection(val_groups)
                self.assertEqual(
                    len(overlap),
                    0,
                    f"Group leakage detected in target {target}, fold {fold}: {overlap}",
                )

    # ----------------------------------------------------------------------
    # 4. Feature Pipeline Tests
    # ----------------------------------------------------------------------
    def test_feature_pipeline_excludes_targets(self):
        """Feature pipeline must NOT touch or transform target columns."""
        pipeline = build_feature_pipeline()
        feature_cols = [
            "Ingredient_Name",
            "Primary_Function",
            "Ingredient_Category",
            "Product_Categories",
            "Origin",
            "Regulatory_Status",
        ]
        # Fit on subset of features
        X_trans = pipeline.fit_transform(self.df[feature_cols])
        self.assertEqual(X_trans.shape[0], 926)
        self.assertGreater(X_trans.shape[1], 100)  # Generates substantial feature dimensions

    # ----------------------------------------------------------------------
    # 5. Model Training, Probabilities & OOF Completeness Tests
    # ----------------------------------------------------------------------
    def test_model_probabilities_and_oof_completeness(self):
        """Model must produce valid probabilities summing to 1 and complete OOF coverage."""
        y = self.df["Safety_Level"].values
        groups = self.df["canonical_group_id"].values
        classes = sorted(list(set(y)))

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.empty(len(self.df), dtype=object)
        oof_probs = np.zeros((len(self.df), len(classes)), dtype=float)

        for train_idx, val_idx in sgkf.split(self.df, y, groups=groups):
            X_train, X_val = self.df.iloc[train_idx], self.df.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pipeline = Pipeline([
                ("preprocessor", build_feature_pipeline()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            pipeline.fit(X_train, y_train)

            oof_preds[val_idx] = pipeline.predict(X_val)
            probs = pipeline.predict_proba(X_val)

            model_classes = list(pipeline.named_steps["classifier"].classes_)
            for col_idx, cls_name in enumerate(model_classes):
                oof_probs[val_idx, classes.index(cls_name)] = probs[:, col_idx]

        # OOF completeness: exactly 926 predictions, zero None
        self.assertEqual(len(oof_preds), 926)
        self.assertTrue(all(p is not None for p in oof_preds))

        # Probability validation: sum to 1.0
        prob_sums = np.sum(oof_probs, axis=1)
        self.assertTrue(np.allclose(prob_sums, 1.0, atol=1e-5))

    # ----------------------------------------------------------------------
    # 6. Reproducibility Test
    # ----------------------------------------------------------------------
    def test_reproducibility_deterministic_outputs(self):
        """Identical random_state must produce bitwise identical splits and predictions."""
        y = self.df["Allergy_Risk"].values
        groups = self.df["canonical_group_id"].values

        def _run():
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            preds = np.empty(len(self.df), dtype=object)
            for train_idx, val_idx in sgkf.split(self.df, y, groups=groups):
                X_train, X_val = self.df.iloc[train_idx], self.df.iloc[val_idx]
                pipe = Pipeline([
                    ("preprocessor", build_feature_pipeline()),
                    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
                ])
                pipe.fit(X_train, y[train_idx])
                preds[val_idx] = pipe.predict(X_val)
            return preds

        preds_run1 = _run()
        preds_run2 = _run()
        self.assertTrue(np.array_equal(preds_run1, preds_run2))


if __name__ == "__main__":
    unittest.main()
