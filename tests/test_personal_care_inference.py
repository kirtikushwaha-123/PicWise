"""
tests/test_personal_care_inference.py

Comprehensive test suite for the Personal Care ML inference service,
validating model loading, metadata integrity, feature parity with Phase 10A,
strict target isolation, and multi-target prediction correctness.
"""

import json
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from backend.ml.inference.personal_care_service import (
    PersonalCarePredictor,
    TargetPredictor,
    predict_personal_care,
    get_personal_care_predictor,
)


class TestPersonalCareInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = Path("backend/ml/models/personal_care")
        cls.predictor = get_personal_care_predictor()

    def test_model_files_and_metadata_exist(self):
        """Verify all 3 serialized pipelines and metadata JSON files exist."""
        for target in ["safety", "allergy", "irritation"]:
            pipeline_path = self.models_dir / target / "pipeline.joblib"
            metadata_path = self.models_dir / target / "model_metadata.json"

            self.assertTrue(pipeline_path.exists(), f"Missing pipeline for {target}")
            self.assertTrue(metadata_path.exists(), f"Missing metadata for {target}")

            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.assertEqual(meta["target_column"], {
                "safety": "Safety_Level",
                "allergy": "Allergy_Risk",
                "irritation": "Irritation_Risk",
            }[target])

            # Phase 10A frozen configuration audit
            hp = meta["hyperparameters"]
            self.assertEqual(hp["C"], 10.0)
            self.assertEqual(hp["class_weight"], "balanced")
            self.assertEqual(hp["solver"], "lbfgs")
            self.assertEqual(hp["random_state"], 42)

            # Feature dimensions parity check
            self.assertEqual(meta["feature_dimensions"]["total"], 15229)
            self.assertEqual(meta["dataset_rows"], 926)

    def test_target_safety_isolation_in_pipeline(self):
        """Verify the trained feature pipeline does not include or depend on any target columns."""
        forbidden_targets = {"Safety_Level", "Allergy_Risk", "Irritation_Risk"}

        for target in ["safety", "allergy", "irritation"]:
            tp = self.predictor.targets[target]
            preprocessor = tp.pipeline.named_steps["preprocessor"]
            transformers = preprocessor.transformers_

            for trans_name, trans_obj, cols in transformers:
                if isinstance(cols, list):
                    for c in cols:
                        self.assertNotIn(c, forbidden_targets, f"Target column leaked into {trans_name} of {target}")
                elif isinstance(cols, str):
                    self.assertNotIn(cols, forbidden_targets, f"Target column leaked into {trans_name} of {target}")

    def test_prediction_known_ingredient(self):
        """Test inference on a standard recognized ingredient: Aqua (Water)."""
        from backend.services.personal_care_analysis_service.enrichment import get_personal_care_knowledge_base
        enrichment = get_personal_care_knowledge_base().lookup("Aqua (Water)")
        self.assertIsNotNone(enrichment)

        res = self.predictor.predict(**enrichment.to_model_input_dict())

        self.assertIn("safety", res)
        self.assertIn("allergy", res)
        self.assertIn("irritation", res)

        for target in ["safety", "allergy", "irritation"]:
            sub = res[target]
            self.assertEqual(sub["status"], "success")
            self.assertIsInstance(sub["risk_class"], str)
            self.assertGreater(len(sub["risk_class"]), 0)
            self.assertIsInstance(sub["confidence"], float)
            self.assertGreaterEqual(sub["confidence"], 0.0)
            self.assertLessEqual(sub["confidence"], 1.0)
            self.assertIsInstance(sub["probabilities"], dict)

            # Probabilities should sum to approximately 1.0
            prob_sum = sum(sub["probabilities"].values())
            self.assertAlmostEqual(prob_sum, 1.0, places=3)

        # Expected classes for Aqua: Safe/Very Safe, No Risk/Low
        self.assertIn(res["safety"]["risk_class"], ["Very Safe", "Safe"])
        self.assertIn(res["allergy"]["risk_class"], ["No Risk", "Low"])
        self.assertIn(res["irritation"]["risk_class"], ["No Risk", "Low"])

    def test_predict_personal_care_module_function(self):
        """Test the top-level predict_personal_care utility function."""
        res = predict_personal_care(
            ingredient_name="Glycerin",
            primary_function="Humectant",
            ingredient_category="Humectant",
            product_categories="Skincare",
            origin="Synthetic / Plant",
            regulatory_status="Approved",
        )
        self.assertEqual(res["safety"]["status"], "success")
        self.assertEqual(res["allergy"]["status"], "success")
        self.assertEqual(res["irritation"]["status"], "success")

    def test_component_failure_isolation(self):
        """Verify that a failure in one model does not disrupt other targets."""
        with patch.object(self.predictor.targets["allergy"], "predict", side_effect=RuntimeError("Simulated allergy model crash")):
            res = self.predictor.predict(
                ingredient_name="Aqua (Water)",
                primary_function="Solvent",
                ingredient_category="Solvent",
                product_categories="Skincare",
                origin="Natural",
                regulatory_status="Approved",
            )

            # Safety and Irritation succeed
            self.assertEqual(res["safety"]["status"], "success")
            self.assertEqual(res["irritation"]["status"], "success")

            # Allergy reports failure with diagnostics, without raising exception
            self.assertEqual(res["allergy"]["status"], "model_prediction_failure")
            self.assertIsNone(res["allergy"]["risk_class"])
            self.assertEqual(res["allergy"]["confidence"], 0.0)
            self.assertIn("Simulated allergy model crash", res["allergy"]["error"])


if __name__ == "__main__":
    unittest.main()
