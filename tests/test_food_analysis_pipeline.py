"""
tests/test_food_analysis_pipeline.py

Comprehensive integration test suite for the PicWise Unified Food Analysis Pipeline (Phase 9E).
Validates orchestration, component contracts, domain isolation, error handling,
partial-failure resilience, determinism, API endpoints, and real OCR fixture integration.
"""

import io
import os
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image

from backend import create_app
from backend.services.food_analysis_service import (
    analyze_food,
    FoodAnalysisResult,
    FoodSafetyResult,
    AllergyResult,
    FoodAnalysisError,
    InvalidCategoryError,
    ImageProcessingError,
)


def _create_dummy_image_bytes(format="JPEG", size=(100, 100), color="white") -> bytes:
    """Helper to generate valid in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestFoodAnalysisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.real_food_fixture = os.path.join("tests", "fixtures", "product_food.jpeg")

    # ----------------------------------------------------------------------
    # 1. Category Validation Tests
    # ----------------------------------------------------------------------
    def test_category_validation_success(self):
        """category='food' and whitespace-padded variants are accepted."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.food_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {"domain": "food", "ingredients": [], "nutrition": None}
            result = analyze_food(dummy_bytes, category="food")
            self.assertTrue(result.success)
            self.assertEqual(result.category, "food")

            result_padded = analyze_food(dummy_bytes, category="  FOOD  ")
            self.assertTrue(result_padded.success)
            self.assertEqual(result_padded.category, "food")

    def test_category_validation_personal_care_rejected(self):
        """Personal care category is strictly rejected by the Food Analysis service."""
        dummy_bytes = _create_dummy_image_bytes()
        with self.assertRaises(InvalidCategoryError) as ctx:
            analyze_food(dummy_bytes, category="personal_care")
        self.assertIn("strictly handles 'food'", str(ctx.exception))

    def test_category_validation_invalid_and_empty_rejected(self):
        """Empty, None, or invalid categories are rejected."""
        dummy_bytes = _create_dummy_image_bytes()
        for invalid_cat in [None, "", "   ", "automotive", "cosmetics", 123]:
            with self.assertRaises(InvalidCategoryError):
                analyze_food(dummy_bytes, category=invalid_cat)

    # ----------------------------------------------------------------------
    # 2. Image Validation & OCR Failure Handling
    # ----------------------------------------------------------------------
    def test_empty_image_bytes_rejected(self):
        """Empty or zero-length image bytes raise ImageProcessingError."""
        with self.assertRaises(ImageProcessingError):
            analyze_food(b"", category="food")
        with self.assertRaises(ImageProcessingError):
            analyze_food(None, category="food")

    def test_ocr_failure_does_not_fabricate_downstream_data(self):
        """When OCR raises an exception, downstream models are NOT called with fabricated data."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.food_analysis_service.analyzer.run_ocr") as mock_ocr, \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety") as mock_fs, \
             patch("backend.services.food_analysis_service.analyzer.calculate_nutrition_score") as mock_ns:
            
            mock_ocr.side_effect = RuntimeError("PaddleOCR engine initialization failure")

            result = analyze_food(dummy_bytes, category="food")

            self.assertFalse(result.success)
            self.assertEqual(result.category, "food")
            self.assertIsNone(result.ocr)
            self.assertIsNone(result.food_safety)
            self.assertIsNone(result.nutrition)
            self.assertIsNone(result.allergy)
            self.assertTrue(len(result.errors) > 0)
            self.assertIn("OCR processing failed", result.errors[0])

            # Downstream models must NOT be invoked
            mock_fs.assert_not_called()
            mock_ns.assert_not_called()

    # ----------------------------------------------------------------------
    # 3. End-to-End Successful Analysis & Contract Structure
    # ----------------------------------------------------------------------
    def test_successful_food_analysis_contract(self):
        """Valid OCR output produces a complete unified result adhering to the contract."""
        dummy_bytes = _create_dummy_image_bytes()
        mocked_ocr_output = {
            "domain": "food",
            "ingredients": [
                {
                    "ocr_text": "Wheat Flour",
                    "matched_name": "Refined Wheat Flour (Maida)",
                    "similarity": 95.0,
                    "method": "fuzzy",
                    "raw_text": "Wheat Flour",
                },
                {
                    "ocr_text": "Sugar",
                    "matched_name": "Sugar",
                    "similarity": 100.0,
                    "method": "exact",
                    "raw_text": "Sugar",
                },
            ],
            "nutrition": {
                "energy": {"value": 450.0, "unit": "kcal", "per_100g": {"value": 450.0, "unit": "kcal"}},
                "total_sugars": {"value": 24.0, "unit": "g", "per_100g": {"value": 24.0, "unit": "g"}},
                "total_fat": {"value": 15.0, "unit": "g", "per_100g": {"value": 15.0, "unit": "g"}},
                "saturated_fat": {"value": 6.0, "unit": "g", "per_100g": {"value": 6.0, "unit": "g"}},
                "sodium": {"value": 400.0, "unit": "mg", "per_100g": {"value": 400.0, "unit": "mg"}},
            },
            "raw_text": {
                "all_text": "Wheat flour sugar edible oil salt",
                "ingredients_text": "Wheat Flour, Sugar",
                "nutrition_text": "Energy 450kcal, Sugars 24g, Fat 15g, Saturated Fat 6g, Sodium 400mg",
            },
            "packet_detection": {"bbox": [10, 10, 200, 200], "confidence": 0.95},
            "image_quality": {"passes_thresholds": True},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mocked_ocr_output):
            result = analyze_food(dummy_bytes, category="food")

            self.assertTrue(result.success)
            self.assertEqual(result.category, "food")
            self.assertEqual(result.errors, [])

            # Verify serialized dictionary
            res_dict = result.to_dict()
            self.assertIn("category", res_dict)
            self.assertIn("success", res_dict)
            self.assertIn("ocr", res_dict)
            self.assertIn("food_safety", res_dict)
            self.assertIn("nutrition", res_dict)
            self.assertIn("allergy", res_dict)
            self.assertIn("errors", res_dict)
            self.assertIn("warnings", res_dict)

            # Check Food Safety component
            fs = res_dict["food_safety"]
            self.assertEqual(fs["status"], "success")
            self.assertEqual(fs["total_ingredients"], 2)
            self.assertEqual(len(fs["ingredients"]), 2)
            for ing in fs["ingredients"]:
                self.assertIn("ingredient", ing)
                self.assertIn("risk_class", ing)
                self.assertIn("confidence", ing)
                self.assertIn("probabilities", ing)
                self.assertIn("Very Safe", ing["probabilities"])
                self.assertIn("High Risk", ing["probabilities"])

            # Check Nutrition component
            nut = res_dict["nutrition"]
            self.assertEqual(nut["status"], "scored")
            self.assertIsNotNone(nut["nutrition_score"])
            self.assertGreaterEqual(nut["nutrition_score"], 0.0)
            self.assertLessEqual(nut["nutrition_score"], 100.0)
            self.assertIn("components", nut)
            self.assertIn("risk_details", nut)
            self.assertIn("guardrails", nut)
            self.assertIn("nutrients_evaluated", nut)

            # Check Allergy component (deterministic KB lookup)
            al = res_dict["allergy"]
            self.assertEqual(al["status"], "success")
            self.assertEqual(al["product_risk_level"], "Medium")
            self.assertEqual(al["product_ui_label"], "Moderate Allergy Risk")
            self.assertEqual(al["allergens_detected"], ["Refined Wheat Flour (Maida)"])

            # Phase 9H: Check presentation field in unified response
            self.assertIn("presentation", res_dict)
            pres = res_dict["presentation"]
            self.assertIsNotNone(pres)
            self.assertIn("food_safety", pres)
            self.assertIn("allergy", pres)
            self.assertIn("nutrition", pres)
            # CRITICAL CONSTRAINT: No overall product health score, no overall color, no overall verdict
            self.assertNotIn("overall_status", pres)
            self.assertNotIn("overall_color", pres)
            self.assertNotIn("overall_score", pres)
            self.assertNotIn("product_color", pres)

            # Check ingredient-level presentation status
            for ing in fs["ingredients"]:
                self.assertIn("presentation_status", ing)
                self.assertIn(ing["presentation_status"], ["green", "yellow", "orange", "red", "unavailable"])
                self.assertIn("presentation", ing)

            # Check Nutrition presentation status
            self.assertIn("presentation_status", nut)
            self.assertIn(nut["presentation_status"], ["green", "yellow", "orange", "red", "unavailable"])
            self.assertEqual(pres["nutrition"]["score"], nut["nutrition_score"])

            # Check Allergy presentation status (Medium -> orange, Moderate Allergy Risk)
            self.assertEqual(al["presentation_status"], "orange")
            self.assertEqual(pres["allergy"]["status"], "orange")
            self.assertEqual(pres["allergy"]["label"], "Moderate Allergy Risk")

    # ----------------------------------------------------------------------
    # 4. Nutrition Insufficient Data Handling
    # ----------------------------------------------------------------------
    def test_nutrition_insufficient_data(self):
        """When fewer than 3 core nutrients are detected, score is None and status is Insufficient."""
        dummy_bytes = _create_dummy_image_bytes()
        mocked_ocr_output = {
            "domain": "food",
            "ingredients": [{"matched_name": "Salt", "raw_text": "Salt"}],
            "nutrition": {
                # Only 1 core nutrient (sodium)
                "sodium": {"value": 500.0, "unit": "mg", "per_100g": {"value": 500.0, "unit": "mg"}}
            },
            "raw_text": {"all_text": "Salt", "ingredients_text": "Salt", "nutrition_text": "Sodium 500mg"},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mocked_ocr_output):
            result = analyze_food(dummy_bytes, category="food")
            self.assertTrue(result.success)

            nut = result.nutrition
            self.assertIsNone(nut["nutrition_score"])
            self.assertEqual(nut["status"], "Insufficient Nutrition Data")
            self.assertIn("energy", nut["nutrients_missing"])

            # Phase 9H: Nutrition presentation for insufficient data
            self.assertEqual(nut["presentation_status"], "unavailable")
            self.assertEqual(nut["presentation"]["status"], "unavailable")
            self.assertIsNone(nut["presentation"].get("label"))

    # ----------------------------------------------------------------------
    # 5. Food Safety No-Ingredients Handling
    # ----------------------------------------------------------------------
    def test_food_safety_no_ingredients_detected(self):
        """When OCR extracts zero ingredients, food safety sets status='no_ingredients' without fabricating."""
        dummy_bytes = _create_dummy_image_bytes()
        mocked_ocr_output = {
            "domain": "food",
            "ingredients": [],
            "nutrition": None,
            "raw_text": {"all_text": "", "ingredients_text": "", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mocked_ocr_output):
            result = analyze_food(dummy_bytes, category="food")
            self.assertTrue(result.success)

            fs = result.food_safety
            self.assertEqual(fs["status"], "no_ingredients")
            self.assertEqual(fs["ingredients"], [])
            self.assertEqual(fs["total_ingredients"], 0)
            self.assertTrue(len(fs["warnings"]) > 0)
            self.assertIn("No ingredients detected", fs["warnings"][0])

    # ----------------------------------------------------------------------
    # 6. Allergy Component Deterministic Lookup
    # ----------------------------------------------------------------------
    def test_allergy_component_deterministic_lookup(self):
        """Allergy engine deterministically looks up ingredients in knowledge base."""
        dummy_bytes = _create_dummy_image_bytes()
        mocked_ocr_output = {
            "domain": "food",
            "ingredients": [{"matched_name": "Peanuts", "raw_text": "Peanuts"}],
            "nutrition": None,
            "raw_text": {"all_text": "Peanuts", "ingredients_text": "Peanuts", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mocked_ocr_output):
            result = analyze_food(dummy_bytes, category="food")
            al = result.allergy
            self.assertEqual(al["status"], "success")
            self.assertEqual(al["product_risk_level"], "High")
            self.assertEqual(al["product_ui_label"], "High Allergy Risk")
            self.assertEqual(al["allergens_detected"], ["Peanuts"])

            # Phase 9H: Allergy presentation status (High -> red, High Allergy Risk)
            self.assertEqual(al["presentation_status"], "red")
            self.assertEqual(al["presentation"]["status"], "red")
            self.assertEqual(al["presentation"]["label"], "High Allergy Risk")

    # ----------------------------------------------------------------------
    # 7. Partial Failure Resilience
    # ----------------------------------------------------------------------
    def test_partial_failure_food_safety_error_preserves_nutrition(self):
        """An error in Food Safety inference does not crash or invalidate the Nutrition Scoring result."""
        dummy_bytes = _create_dummy_image_bytes()
        mocked_ocr_output = {
            "domain": "food",
            "ingredients": [{"matched_name": "Sugar", "raw_text": "Sugar"}],
            "nutrition": {
                "energy": {"value": 400.0, "unit": "kcal", "per_100g": {"value": 400.0, "unit": "kcal"}},
                "total_sugars": {"value": 10.0, "unit": "g", "per_100g": {"value": 10.0, "unit": "g"}},
                "total_fat": {"value": 5.0, "unit": "g", "per_100g": {"value": 5.0, "unit": "g"}},
            },
            "raw_text": {"all_text": "", "ingredients_text": "", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mocked_ocr_output), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", side_effect=RuntimeError("Model vectorizer failure")):
            
            result = analyze_food(dummy_bytes, category="food")
            self.assertTrue(result.success)

            # Food safety is marked as error
            fs = result.food_safety
            self.assertEqual(fs["status"], "error")
            self.assertEqual(fs["total_ingredients"], 0)
            self.assertIn("Model vectorizer failure", fs["error"])

            # Nutrition is still evaluated and valid
            nut = result.nutrition
            self.assertEqual(nut["status"], "scored")
            self.assertIsNotNone(nut["nutrition_score"])

    def test_partial_failure_nutrition_error_preserves_food_safety(self):
        """An error in Nutrition scoring does not crash or invalidate Food Safety inference."""
        dummy_bytes = _create_dummy_image_bytes()
        mocked_ocr_output = {
            "domain": "food",
            "ingredients": [{"matched_name": "Sugar", "raw_text": "Sugar"}],
            "nutrition": {"energy": {"value": 100.0, "unit": "kcal"}},
            "raw_text": {"all_text": "", "ingredients_text": "", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mocked_ocr_output), \
             patch("backend.services.food_analysis_service.analyzer.calculate_nutrition_score", side_effect=ValueError("Unexpected calculation error")):
            
            result = analyze_food(dummy_bytes, category="food")
            self.assertTrue(result.success)

            # Food safety succeeded
            fs = result.food_safety
            self.assertEqual(fs["status"], "success")
            self.assertEqual(len(fs["ingredients"]), 1)

            # Nutrition is marked as error
            nut = result.nutrition
            self.assertEqual(nut["status"], "error")
            self.assertIsNone(nut["nutrition_score"])

    # ----------------------------------------------------------------------
    # 8. Component Independence
    # ----------------------------------------------------------------------
    def test_component_independence_between_safety_and_nutrition(self):
        """Food safety risk classification and nutrition score do not cross-contaminate."""
        dummy_bytes = _create_dummy_image_bytes()
        mocked_ocr_output = {
            "domain": "food",
            "ingredients": [{"matched_name": "Sodium Benzoate", "raw_text": "Sodium Benzoate"}],
            "nutrition": {
                "energy": {"value": 100.0, "unit": "kcal", "per_100g": {"value": 100.0, "unit": "kcal"}},
                "total_sugars": {"value": 1.0, "unit": "g", "per_100g": {"value": 1.0, "unit": "g"}},
                "total_fat": {"value": 0.5, "unit": "g", "per_100g": {"value": 0.5, "unit": "g"}},
                "saturated_fat": {"value": 0.1, "unit": "g", "per_100g": {"value": 0.1, "unit": "g"}},
                "sodium": {"value": 50.0, "unit": "mg", "per_100g": {"value": 50.0, "unit": "mg"}},
            },
            "raw_text": {"all_text": "", "ingredients_text": "", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mocked_ocr_output):
            result = analyze_food(dummy_bytes, category="food")

            # Sodium Benzoate is classified as Moderate Risk by Food Safety ML
            fs_class = result.food_safety["ingredients"][0]["risk_class"]
            self.assertEqual(fs_class, "Moderate Risk")

            # Nutrition score is computed independently strictly from the nutritional profile
            nut_score = result.nutrition["nutrition_score"]
            self.assertIsNotNone(nut_score)
            self.assertEqual(nut_score, 55.9)

            # Asserting that Moderate Risk does not alter nutrition score,
            # and nutrition score does not alter the ML classification
            self.assertEqual(result.food_safety["ingredients"][0]["risk_class"], "Moderate Risk")
            self.assertEqual(result.nutrition["nutrition_score"], 55.9)

    # ----------------------------------------------------------------------
    # 9. Determinism Tests
    # ----------------------------------------------------------------------
    def test_determinism_across_multiple_runs(self):
        """Identical inputs produce identical unified food analysis results."""
        dummy_bytes = _create_dummy_image_bytes()
        mocked_ocr_output = {
            "domain": "food",
            "ingredients": [{"matched_name": "Sugar", "raw_text": "Sugar"}],
            "nutrition": {
                "energy": {"value": 400.0, "unit": "kcal", "per_100g": {"value": 400.0, "unit": "kcal"}},
                "total_sugars": {"value": 25.0, "unit": "g", "per_100g": {"value": 25.0, "unit": "g"}},
                "total_fat": {"value": 10.0, "unit": "g", "per_100g": {"value": 10.0, "unit": "g"}},
            },
            "raw_text": {"all_text": "", "ingredients_text": "", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mocked_ocr_output):
            result1 = analyze_food(dummy_bytes, category="food").to_dict()
            result2 = analyze_food(dummy_bytes, category="food").to_dict()

            self.assertEqual(result1, result2)

    # ----------------------------------------------------------------------
    # 10. Real OCR Fixture Integration Test
    # ----------------------------------------------------------------------
    def test_real_ocr_fixture_integration(self):
        """End-to-end integration test executing real OCR on tests/fixtures/product_food.jpeg."""
        self.assertTrue(
            os.path.exists(self.real_food_fixture),
            f"Required fixture {self.real_food_fixture} not found."
        )

        with open(self.real_food_fixture, "rb") as f:
            image_bytes = f.read()

        result = analyze_food(image_bytes, category="food")

        self.assertTrue(result.success)
        self.assertEqual(result.category, "food")
        self.assertIsNotNone(result.ocr)
        self.assertEqual(result.ocr.get("domain"), "food")

        # Real OCR extracts ingredients from the food fixture
        self.assertIsNotNone(result.food_safety)
        self.assertEqual(result.food_safety["status"], "success")
        self.assertGreater(result.food_safety["total_ingredients"], 0)

        first_ing = result.food_safety["ingredients"][0]
        self.assertIn("ingredient", first_ing)
        self.assertIn("risk_class", first_ing)
        self.assertIn(first_ing["risk_class"], ["Very Safe", "Safe", "Moderate Risk", "High Risk"])
        self.assertGreater(first_ing["confidence"], 0.0)

        # Real OCR extracts nutrition from the food fixture
        self.assertIsNotNone(result.nutrition)
        self.assertIn(result.nutrition["status"], ["scored", "Insufficient Nutrition Data"])

        # Allergy component resolves active lookup
        self.assertIsNotNone(result.allergy)
        self.assertEqual(result.allergy["status"], "success")
        self.assertEqual(result.allergy["product_risk_level"], "Medium")
        self.assertEqual(result.allergy["product_ui_label"], "Moderate Allergy Risk")
        self.assertEqual(result.allergy["presentation_status"], "orange")

        # Phase 9H: Presentation structure verification
        self.assertIsNotNone(result.presentation)
        self.assertIn("food_safety", result.presentation)
        self.assertIn("allergy", result.presentation)
        self.assertIn("nutrition", result.presentation)
        self.assertNotIn("overall_status", result.presentation)
        self.assertNotIn("overall_color", result.presentation)
        self.assertNotIn("overall_score", result.presentation)

    # ----------------------------------------------------------------------
    # 11. Dedicated API Endpoint Tests (/api/food/analyze)
    # ----------------------------------------------------------------------
    def test_api_food_analyze_success(self):
        """POST /api/food/analyze returns 200 and unified FoodAnalysisResult."""
        with open(self.real_food_fixture, "rb") as f:
            resp = self.client.post(
                "/api/food/analyze",
                data={"image": (f, "product_food.jpeg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertEqual(data["category"], "food")
        self.assertTrue(data["success"])
        self.assertIn("ocr", data)
        self.assertIn("food_safety", data)
        self.assertIn("nutrition", data)
        self.assertIn("allergy", data)
        self.assertEqual(data["allergy"]["status"], "success")
        self.assertEqual(data["allergy"]["product_risk_level"], "Medium")
        self.assertEqual(data["allergy"]["product_ui_label"], "Moderate Allergy Risk")
        self.assertEqual(data["allergy"]["presentation_status"], "orange")

        # Phase 9H: API response presentation verification
        self.assertIn("presentation", data)
        self.assertIn("food_safety", data["presentation"])
        self.assertIn("allergy", data["presentation"])
        self.assertIn("nutrition", data["presentation"])
        self.assertEqual(data["presentation"]["allergy"]["status"], "orange")
        self.assertNotIn("overall_status", data["presentation"])
        self.assertNotIn("overall_color", data["presentation"])
        self.assertNotIn("overall_score", data["presentation"])

    def test_api_food_analyze_category_enforcement(self):
        """POST /api/food/analyze strictly requires category='food'."""
        dummy_bytes = _create_dummy_image_bytes()

        # Missing category
        resp_missing = self.client.post(
            "/api/food/analyze",
            data={"image": (io.BytesIO(dummy_bytes), "test.jpg", "image/jpeg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_missing.status_code, 400)
        self.assertIn("category is required", resp_missing.get_json()["error"])

        # Personal care category
        resp_pc = self.client.post(
            "/api/food/analyze",
            data={
                "image": (io.BytesIO(dummy_bytes), "test.jpg", "image/jpeg"),
                "category": "personal_care",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_pc.status_code, 400)
        self.assertIn("strictly handles 'food'", resp_pc.get_json()["error"])

        # Invalid category
        resp_inv = self.client.post(
            "/api/food/analyze",
            data={
                "image": (io.BytesIO(dummy_bytes), "test.jpg", "image/jpeg"),
                "category": "beverage",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_inv.status_code, 400)
        self.assertIn("strictly handles 'food'", resp_inv.get_json()["error"])

    def test_api_food_analyze_image_validation(self):
        """POST /api/food/analyze validates image presence, format, and content."""
        # Missing image
        resp_no_img = self.client.post(
            "/api/food/analyze",
            data={"category": "food"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_no_img.status_code, 400)
        self.assertIn("Image file is required", resp_no_img.get_json()["error"])

        # Unsupported format (e.g. .txt)
        resp_bad_ext = self.client.post(
            "/api/food/analyze",
            data={
                "image": (io.BytesIO(b"not an image"), "test.txt", "text/plain"),
                "category": "food",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_bad_ext.status_code, 400)
        self.assertIn("Only JPG, JPEG, PNG, and WEBP", resp_bad_ext.get_json()["error"])

        # Empty image bytes
        resp_empty_img = self.client.post(
            "/api/food/analyze",
            data={
                "image": (io.BytesIO(b""), "test.jpg", "image/jpeg"),
                "category": "food",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_empty_img.status_code, 400)
        self.assertIn("empty", resp_empty_img.get_json()["error"])

    # ----------------------------------------------------------------------
    # 12. Backward Compatibility for Existing /api/analyze Endpoint
    # ----------------------------------------------------------------------
    def test_legacy_analyze_endpoint_remains_intact(self):
        """Existing /api/analyze route continues to function for both food and personal care."""
        with open(self.real_food_fixture, "rb") as f:
            resp = self.client.post(
                "/api/analyze",
                data={"image": (f, "product_food.jpeg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["product"]["domain"], "food")
        self.assertIn("ingredients", data)
        self.assertIn("nutrition", data)
        self.assertEqual(data["personalCare"], [])

    # ----------------------------------------------------------------------
    # 13. Phase 9H Presentation Status Mapping Integration & Dimension Independence
    # ----------------------------------------------------------------------
    def test_phase9h_presentation_status_mapping_integration(self):
        """Phase 9H: Validate that presentation status mapping is integrated across all 3 independent dimensions.
        Strictly verifies:
        - food_safety, allergy, and nutrition independent presentation objects
        - Safe -> yellow (never green) in ingredient presentations
        - No overall product health score, no overall color, no overall verdict
        - Missing nutrition != 0 (unavailable)
        """
        dummy_bytes = _create_dummy_image_bytes()
        mocked_ocr_output = {
            "domain": "food",
            "ingredients": [
                {"matched_name": "Sugar", "raw_text": "Sugar"},
                {"matched_name": "Almonds", "raw_text": "Almonds"},
            ],
            "nutrition": {
                "energy": {"value": 500.0, "unit": "kcal", "per_100g": {"value": 500.0, "unit": "kcal"}},
                "total_sugars": {"value": 30.0, "unit": "g", "per_100g": {"value": 30.0, "unit": "g"}},
                "total_fat": {"value": 20.0, "unit": "g", "per_100g": {"value": 20.0, "unit": "g"}},
                "saturated_fat": {"value": 5.0, "unit": "g", "per_100g": {"value": 5.0, "unit": "g"}},
                "sodium": {"value": 100.0, "unit": "mg", "per_100g": {"value": 100.0, "unit": "mg"}},
            },
            "raw_text": {"all_text": "Sugar Almonds", "ingredients_text": "Sugar, Almonds", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mocked_ocr_output):
            result = analyze_food(dummy_bytes, category="food")
            self.assertTrue(result.success)

            # Top-level presentation dictionary exists
            self.assertIsNotNone(result.presentation)
            pres = result.presentation

            # 1. Food Safety Dimension (Ingredient Level)
            self.assertIn("food_safety", pres)
            self.assertIn("ingredients", result.food_safety)
            for ing in result.food_safety["ingredients"]:
                self.assertIn("presentation_status", ing)
                self.assertIn(ing["presentation_status"], ["green", "yellow", "orange", "red", "unavailable"])
                # Explicit constraint check: If risk_class is 'Safe', presentation_status MUST be 'yellow', NEVER 'green'
                if ing.get("risk_class") == "Safe":
                    self.assertEqual(ing["presentation_status"], "yellow")
                    self.assertEqual(ing["presentation"]["status"], "yellow")

            # 2. Allergy Dimension (Deterministic KB lookup: Almonds -> High -> red)
            self.assertIn("allergy", pres)
            self.assertEqual(pres["allergy"]["risk_level"], "High")
            self.assertEqual(pres["allergy"]["status"], "red")
            self.assertEqual(pres["allergy"]["label"], "High Allergy Risk")

            # 3. Nutrition Dimension (Deterministic 0-100 score -> independent bracket)
            self.assertIn("nutrition", pres)
            nut_score = result.nutrition["nutrition_score"]
            self.assertIsNotNone(nut_score)
            self.assertEqual(pres["nutrition"]["score"], nut_score)
            if nut_score <= 25.0:
                self.assertEqual(pres["nutrition"]["status"], "red")
            elif nut_score <= 50.0:
                self.assertEqual(pres["nutrition"]["status"], "orange")
            elif nut_score <= 75.0:
                self.assertEqual(pres["nutrition"]["status"], "yellow")
            else:
                self.assertEqual(pres["nutrition"]["status"], "green")

            # 4. Strict Absence of Any Overall Product Color or Score
            self.assertNotIn("overall_status", pres)
            self.assertNotIn("overall_color", pres)
            self.assertNotIn("overall_score", pres)
            self.assertNotIn("product_color", pres)
            self.assertNotIn("verdict", pres)
            self.assertNotIn("health_score", pres)


if __name__ == "__main__":
    unittest.main()
