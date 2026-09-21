"""
tests/test_personal_care_analysis_pipeline.py

Comprehensive test suite for the Unified Personal Care Analysis Pipeline (Phase 10B).
Validates:
- Strict explicit category validation
- Integration with shared OCR pipeline
- Semantic enrichment and unknown ingredient policy
- Component failure isolation
- Product-level conservative aggregation
- API route POST /api/personal-care/analyze
- Zero target leakage in data contracts
- Real-image fixture end-to-end integration
"""

import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image

from backend import create_app
from backend.services.personal_care_analysis_service import (
    analyze_personal_care,
    PersonalCareAnalysisResult,
    CategoryValidationError,
    ImageValidationError,
    PersonalCareAnalysisError,
)


def _create_dummy_image_bytes(format="JPEG", size=(100, 100), color="white") -> bytes:
    """Helper to generate valid in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestPersonalCareAnalysisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.real_pc_fixture = Path("tests/fixtures/product_personal_care.png")

    # ----------------------------------------------------------------------
    # 1. Category Validation Tests
    # ----------------------------------------------------------------------
    def test_category_validation_success(self):
        """category='personal_care' and whitespace-padded variants are accepted."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {"domain": "personal_care", "ingredients": []}
            for valid_cat in ["personal_care", "PERSONAL_CARE", " Personal_Care "]:
                res = analyze_personal_care(dummy_bytes, category=valid_cat)
                self.assertIsInstance(res, PersonalCareAnalysisResult)
                self.assertEqual(res.category, "personal_care")

    def test_category_validation_rejection(self):
        """Invalid categories (food, cosmetics, empty) must raise CategoryValidationError."""
        dummy_bytes = _create_dummy_image_bytes()
        for invalid_cat in ["food", "cosmetics", "skincare", "", None, 123]:
            with self.assertRaises(CategoryValidationError):
                analyze_personal_care(dummy_bytes, category=invalid_cat)

    # ----------------------------------------------------------------------
    # 2. Image Validation Tests
    # ----------------------------------------------------------------------
    def test_invalid_image_bytes(self):
        """Empty image bytes must raise ImageValidationError."""
        with self.assertRaises(ImageValidationError):
            analyze_personal_care(b"", category="personal_care")

        with self.assertRaises(ImageValidationError):
            analyze_personal_care(None, category="personal_care")

    def test_ocr_failure_handling(self):
        """When OCR fails, pipeline returns success=False and unavailable presentation without crash."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr", side_effect=RuntimeError("OCR engine failure")):
            res = analyze_personal_care(dummy_bytes, category="personal_care")
            self.assertFalse(res.success)
            self.assertEqual(res.presentation["personal_care_safety"]["status"], "unavailable")
            self.assertEqual(res.presentation["allergy"]["status"], "unavailable")
            self.assertEqual(res.presentation["irritation"]["status"], "unavailable")
            self.assertTrue(any("OCR processing failed" in err for err in res.errors))

    # ----------------------------------------------------------------------
    # 3. OCR and Ingredient Recognition Orchestration
    # ----------------------------------------------------------------------
    def test_no_ingredients_detected(self):
        """When OCR finds no text, pipeline completes with unavailable presentation and warnings."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {"domain": "personal_care", "ingredients": []}
            res = analyze_personal_care(dummy_bytes, category="personal_care")

            self.assertTrue(res.success)
            self.assertEqual(res.personal_care["total_ingredients"], 0)
            self.assertEqual(res.personal_care["recognized_ingredients"], 0)
            self.assertEqual(res.presentation["personal_care_safety"]["status"], "unavailable")
            self.assertEqual(res.presentation["allergy"]["status"], "unavailable")
            self.assertEqual(res.presentation["irritation"]["status"], "unavailable")
            self.assertTrue(any("No ingredients detected" in w for w in res.warnings))

    def test_known_ingredients_end_to_end(self):
        """Pipeline successfully enriches and predicts recognized ingredients."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [
                    {"raw_text": "Aqua (Water)", "matched_name": "Aqua (Water)"},
                    {"raw_text": "Glycerin", "matched_name": "Glycerin"},
                ],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")

            self.assertTrue(res.success)
            self.assertEqual(res.personal_care["total_ingredients"], 2)
            self.assertEqual(res.personal_care["recognized_ingredients"], 2)

            # Check ingredient records
            for ing in res.personal_care["ingredients"]:
                self.assertEqual(ing["status"], "success")
                self.assertIsNotNone(ing["features"])
                self.assertEqual(ing["safety"]["status"], "success")
                self.assertEqual(ing["allergy"]["status"], "success")
                self.assertEqual(ing["irritation"]["status"], "success")

            # Check presentation status
            self.assertIn(res.presentation["personal_care_safety"]["status"], ["green", "yellow", "red"])
            self.assertIn(res.presentation["allergy"]["status"], ["green", "yellow", "orange", "red"])
            self.assertIn(res.presentation["irritation"]["status"], ["green", "yellow", "orange", "red"])

    def test_unknown_ingredient_policy(self):
        """
        Unknown ingredients must NOT be passed to ML models with fake features,
        must NOT default to Safe, and must be flagged as warnings.
        """
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [
                    {"raw_text": "UnknownFakeChemicalXYZ123"},
                ],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")

            self.assertTrue(res.success)
            self.assertEqual(res.personal_care["total_ingredients"], 1)
            self.assertEqual(res.personal_care["recognized_ingredients"], 0)

            ing = res.personal_care["ingredients"][0]
            self.assertEqual(ing["status"], "ingredient_not_recognized")
            self.assertIsNone(ing["features"])
            self.assertEqual(ing["safety"]["status"], "unavailable")
            self.assertEqual(ing["allergy"]["status"], "unavailable")
            self.assertEqual(ing["irritation"]["status"], "unavailable")

            # Product presentation must be unavailable because no recognized ingredients exist
            self.assertEqual(res.presentation["personal_care_safety"]["status"], "unavailable")
            self.assertEqual(res.presentation["allergy"]["status"], "unavailable")
            self.assertEqual(res.presentation["irritation"]["status"], "unavailable")

            # Warning must be emitted
            self.assertTrue(any("UnknownFakeChemicalXYZ123" in w for w in res.warnings))

    def test_partial_product_recognition(self):
        """
        When some ingredients are recognized and some are unknown:
        - Recognized ingredients drive the conservative product rating.
        - Unknown ingredients are preserved and flagged in warnings.
        """
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [
                    {"raw_text": "Aqua (Water)", "matched_name": "Aqua (Water)"},
                    {"raw_text": "UnobtaniumExtract"},
                ],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")

            self.assertTrue(res.success)
            self.assertEqual(res.personal_care["total_ingredients"], 2)
            self.assertEqual(res.personal_care["recognized_ingredients"], 1)

            # Product status is successfully computed from Aqua
            self.assertNotEqual(res.presentation["personal_care_safety"]["status"], "unavailable")
            # Warning mentions the unrecognized ingredient
            self.assertTrue(any("UnobtaniumExtract" in w for w in res.warnings))

    def test_component_failure_resilience(self):
        """If model prediction throws on an ingredient, orchestrator catches it without pipeline crash."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr, \
             patch("backend.services.personal_care_analysis_service.analyzer.predict_personal_care", side_effect=RuntimeError("Simulated inference failure")):
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [{"raw_text": "Aqua (Water)"}],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")

            self.assertTrue(res.success)
            ing = res.personal_care["ingredients"][0]
            self.assertEqual(ing["status"], "model_prediction_failure")
            self.assertTrue(any("Model prediction failure" in w for w in res.warnings))
            self.assertEqual(res.presentation["personal_care_safety"]["status"], "unavailable")

    # ----------------------------------------------------------------------
    # 4. Strict Target Isolation Check
    # ----------------------------------------------------------------------
    def test_target_isolation_in_serialized_result(self):
        """Verify target columns (Safety_Level, Allergy_Risk, Irritation_Risk) are never present in features."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [{"raw_text": "Aqua (Water)"}],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")
            res_dict = res.to_dict()

            forbidden_targets = {"Safety_Level", "Allergy_Risk", "Irritation_Risk"}

            for ing in res_dict["personal_care"]["ingredients"]:
                if ing["features"]:
                    for k in ing["features"].keys():
                        self.assertNotIn(k, forbidden_targets)

    # ----------------------------------------------------------------------
    # 5. API Endpoint Tests (POST /api/personal-care/analyze)
    # ----------------------------------------------------------------------
    def test_api_missing_image(self):
        """POST without image field returns 400."""
        resp = self.client.post(
            "/api/personal-care/analyze",
            data={"category": "personal_care"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("image", data["error"].lower())

    def test_api_invalid_image(self):
        """POST with corrupt image returns 400."""
        resp = self.client.post(
            "/api/personal-care/analyze",
            data={
                "image": (io.BytesIO(b"not-an-image"), "corrupt.png"),
                "category": "personal_care",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])

    def test_api_invalid_category(self):
        """POST with category='food' to personal-care endpoint returns 400."""
        dummy_bytes = _create_dummy_image_bytes()
        resp = self.client.post(
            "/api/personal-care/analyze",
            data={
                "image": (io.BytesIO(dummy_bytes), "test.jpg"),
                "category": "food",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("category", data["error"].lower())

    def test_api_success_with_mocked_ocr(self):
        """Valid POST returns 200 with complete presentation schema."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [
                    {"raw_text": "Aqua (Water)", "matched_name": "Aqua (Water)"},
                    {"raw_text": "Glycerin", "matched_name": "Glycerin"},
                ],
            }
            resp = self.client.post(
                "/api/personal-care/analyze",
                data={
                    "image": (io.BytesIO(dummy_bytes), "test.jpg"),
                    "category": "personal_care",
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()

            self.assertTrue(data["success"])
            self.assertEqual(data["category"], "personal_care")
            self.assertIn("presentation", data)
            self.assertIn("personal_care", data)

            pres = data["presentation"]
            self.assertIn("personal_care_safety", pres)
            self.assertIn("allergy", pres)
            self.assertIn("irritation", pres)

            for dim in ["personal_care_safety", "allergy", "irritation"]:
                self.assertIn(pres[dim]["status"], ["green", "yellow", "orange", "red", "unavailable"])
                self.assertIsInstance(pres[dim]["label"], str)
                self.assertIsInstance(pres[dim]["color"], str)

    # ----------------------------------------------------------------------
    # 6. Real Image Fixture Test
    # ----------------------------------------------------------------------
    def test_real_fixture_e2e(self):
        """End-to-end integration test with real image fixture tests/fixtures/product_personal_care.png."""
        if not self.real_pc_fixture.exists():
            self.skipTest(f"Fixture {self.real_pc_fixture} not found.")

        with open(self.real_pc_fixture, "rb") as f:
            img_bytes = f.read()

        # Run through API client
        resp = self.client.post(
            "/api/personal-care/analyze",
            data={
                "image": (io.BytesIO(img_bytes), "product_personal_care.png"),
                "category": "personal_care",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertTrue(data["success"])
        self.assertEqual(data["category"], "personal_care")
        self.assertIn("presentation", data)
        self.assertIn("personal_care", data)
        self.assertIn("ingredients", data["personal_care"])
        self.assertIsInstance(data["warnings"], list)


if __name__ == "__main__":
    unittest.main()
