import json
import os
import unittest
from pathlib import Path

from backend import create_app
from backend.ml.inference.food_safety_service import (
    CANONICAL_CLASSES,
    DEFAULT_FOOD_SAFETY_MODEL_DIR,
    FoodSafetyPredictor,
    predict_food_safety,
)
from backend.services.analysis_service.analyzer import analyze_product_image


class TestFoodSafetyProduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = Path(DEFAULT_FOOD_SAFETY_MODEL_DIR)
        cls.predictor = FoodSafetyPredictor.get_instance(model_dir=cls.model_dir)
        cls.app = create_app()
        cls.client = cls.app.test_client()

    # ----------------------------------------------------------------------
    # 1. Model Loading Tests
    # ----------------------------------------------------------------------
    def test_production_artifacts_exist_and_load(self):
        """Verify all production artifacts exist in backend/ml/models/food_safety/ and load."""
        vectorizer_path = self.model_dir / "vectorizer.joblib"
        classifier_path = self.model_dir / "classifier.joblib"
        metadata_path = self.model_dir / "model_metadata.json"

        self.assertTrue(vectorizer_path.exists(), f"Missing {vectorizer_path}")
        self.assertTrue(classifier_path.exists(), f"Missing {classifier_path}")
        self.assertTrue(metadata_path.exists(), f"Missing {metadata_path}")

        self.assertIsNotNone(self.predictor.vectorizer)
        self.assertIsNotNone(self.predictor.classifier)
        self.assertIsNotNone(self.predictor.embedder)
        self.assertIsNotNone(self.predictor.metadata)
        self.assertTrue(self.predictor._loaded)

    # ----------------------------------------------------------------------
    # 2. Feature Pipeline Specification Tests
    # ----------------------------------------------------------------------
    def test_feature_pipeline_configuration(self):
        """Verify TF-IDF, MiniLM, and LogisticRegression configurations match frozen specification."""
        vec = self.predictor.vectorizer
        self.assertEqual(vec.analyzer, "char")
        self.assertEqual(vec.ngram_range, (3, 5))
        self.assertTrue(vec.sublinear_tf)
        self.assertEqual(vec.min_df, 1)

        clf = self.predictor.classifier
        self.assertEqual(clf.class_weight, "balanced")
        self.assertEqual(clf.solver, "lbfgs")
        self.assertGreaterEqual(clf.max_iter, 1000)

        meta = self.predictor.metadata
        self.assertEqual(meta["semantic_embeddings"]["dimension"], 384)
        self.assertEqual(meta["semantic_embeddings"]["model_name"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertTrue(meta["semantic_embeddings"]["normalize_embeddings"])
        self.assertEqual(meta["canonical_classes"], CANONICAL_CLASSES)

    # ----------------------------------------------------------------------
    # 3. Class Order and Probability Mapping Tests
    # ----------------------------------------------------------------------
    def test_class_mapping_and_probability_validation(self):
        """Verify model.classes_, canonical class order, probability dictionary, and sum ≈ 1."""
        res = predict_food_safety("Sodium Benzoate")

        self.assertIn("ingredient", res)
        self.assertIn("risk_class", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)

        self.assertIn(res["risk_class"], CANONICAL_CLASSES)
        self.assertEqual(set(res["probabilities"].keys()), set(CANONICAL_CLASSES))

        # Check probability bounds and sum
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

        for c, p in res["probabilities"].items():
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

        # Confirm confidence corresponds to predicted class probability
        self.assertEqual(res["confidence"], res["probabilities"][res["risk_class"]])

        # Confirm predicted class is indeed argmax
        max_prob_class = max(res["probabilities"], key=res["probabilities"].get)
        self.assertEqual(res["risk_class"], max_prob_class)

    # ----------------------------------------------------------------------
    # 4. Deterministic Inference Tests
    # ----------------------------------------------------------------------
    def test_deterministic_inference(self):
        """Verify repeated calls on the same ingredient return identical predictions."""
        test_ingredients = ["Sodium Benzoate", "Citric Acid", "Potassium Bromate", "Apple Pulp"]

        for ing in test_ingredients:
            first_res = predict_food_safety(ing)
            for _ in range(5):
                rep_res = predict_food_safety(ing)
                self.assertEqual(first_res["risk_class"], rep_res["risk_class"])
                self.assertEqual(first_res["confidence"], rep_res["confidence"])
                self.assertEqual(first_res["probabilities"], rep_res["probabilities"])

    # ----------------------------------------------------------------------
    # 5. Empty and Edge Case Inputs
    # ----------------------------------------------------------------------
    def test_empty_and_edge_case_inputs(self):
        """Verify graceful handling for empty, whitespace, or None inputs."""
        for val in ["", "   ", None]:
            res = predict_food_safety(val)
            self.assertIsNone(res["risk_class"])
            self.assertEqual(res["confidence"], 0.0)
            for c in CANONICAL_CLASSES:
                self.assertEqual(res["probabilities"][c], 0.0)

    # ----------------------------------------------------------------------
    # 6. Representative Known Ingredients from Approved Dataset
    # ----------------------------------------------------------------------
    def test_representative_ingredients(self):
        """Verify predictions on representative ingredients from approved dataset."""
        # Sodium Nitrite is a known High Risk preservative
        res_high = predict_food_safety("Sodium Nitrite")
        self.assertEqual(res_high["risk_class"], "High Risk")

        # Titanium Dioxide is a known High Risk food colorant
        res_high_td = predict_food_safety("Titanium Dioxide")
        self.assertEqual(res_high_td["risk_class"], "High Risk")

        # Sodium Benzoate is a known Moderate Risk preservative
        res_mod = predict_food_safety("Sodium Benzoate")
        self.assertEqual(res_mod["risk_class"], "Moderate Risk")

        # Citric Acid is a known Safe acidity regulator
        res_safe = predict_food_safety("Citric Acid")
        self.assertEqual(res_safe["risk_class"], "Safe")

        # Ajwain (Carom Seeds) is a known Very Safe natural spice
        res_very_safe = predict_food_safety("Ajwain (Carom Seeds)")
        self.assertEqual(res_very_safe["risk_class"], "Very Safe")

    # ----------------------------------------------------------------------
    # 7. Downstream OCR Integration (Category: Food)
    # ----------------------------------------------------------------------
    def test_ocr_downstream_food_safety_integration(self):
        """Verify food image OCR analysis attaches foodSafety prediction to all ingredients."""
        fixture_path = os.path.join("tests", "fixtures", "product_food.jpeg")
        self.assertTrue(os.path.exists(fixture_path), f"Fixture {fixture_path} not found")

        with open(fixture_path, "rb") as f:
            image_bytes = f.read()

        result = analyze_product_image(image_bytes, category="food")
        ingredients = result.get("ingredients", [])
        self.assertGreater(len(ingredients), 0)

        for item in ingredients:
            self.assertIn("foodSafety", item, "Expected foodSafety object on food ingredient")
            fs = item["foodSafety"]
            self.assertIsNotNone(fs)
            self.assertIn(fs["risk_class"], CANONICAL_CLASSES)
            self.assertGreater(fs["confidence"], 0.0)
            self.assertEqual(set(fs["probabilities"].keys()), set(CANONICAL_CLASSES))

            # Backward-compatible safetyLevel should match risk_class
            self.assertEqual(item["safetyLevel"], fs["risk_class"])

            # OCR information preserved
            self.assertIn("raw_text", item)
            self.assertIn("canonicalName", item)
            self.assertIn("confidence", item)

    # ----------------------------------------------------------------------
    # 8. Personal Care Strict Isolation Tests
    # ----------------------------------------------------------------------
    def test_personal_care_strict_isolation(self):
        """Verify personal_care analysis NEVER invokes Food Safety model."""
        fixture_path = os.path.join("tests", "fixtures", "product_personal_care.png")
        self.assertTrue(os.path.exists(fixture_path), f"Fixture {fixture_path} not found")

        with open(fixture_path, "rb") as f:
            image_bytes = f.read()

        result = analyze_product_image(image_bytes, category="personal_care")
        ingredients = result.get("ingredients", [])
        self.assertGreater(len(ingredients), 0)

        for item in ingredients:
            self.assertIsNone(
                item.get("foodSafety"),
                f"Personal care ingredient '{item.get('name')}' must NOT contain foodSafety predictions",
            )

    # ----------------------------------------------------------------------
    # 9. API Analyze Endpoint Verification
    # ----------------------------------------------------------------------
    def test_api_analyze_endpoint_food_and_personal_care(self):
        """Verify /api/analyze returns foodSafety for food, and omits it for personal care."""
        food_fixture = os.path.join("tests", "fixtures", "product_food.jpeg")
        with open(food_fixture, "rb") as f:
            resp_food = self.client.post(
                "/api/analyze",
                data={"image": (f, "product_food.jpeg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
        self.assertEqual(resp_food.status_code, 200)
        food_data = resp_food.get_json()
        self.assertEqual(food_data["product"]["domain"], "food")
        self.assertGreater(len(food_data["ingredients"]), 0)
        self.assertIsNotNone(food_data["ingredients"][0].get("foodSafety"))

        pc_fixture = os.path.join("tests", "fixtures", "product_personal_care.png")
        with open(pc_fixture, "rb") as f:
            resp_pc = self.client.post(
                "/api/analyze",
                data={"image": (f, "product_personal_care.png", "image/png"), "category": "personal_care"},
                content_type="multipart/form-data",
            )
        self.assertEqual(resp_pc.status_code, 200)
        pc_data = resp_pc.get_json()
        self.assertEqual(pc_data["product"]["domain"], "personal_care")
        self.assertGreater(len(pc_data["ingredients"]), 0)
        self.assertIsNone(pc_data["ingredients"][0].get("foodSafety"))


if __name__ == "__main__":
    unittest.main()
