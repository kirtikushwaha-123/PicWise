"""
tests/test_food_backend_hardening.py

Comprehensive final hardening and integration test suite for PicWise Food Backend (Phase 9I).
Validates request validation, category routing, pipeline failure isolation,
missing/unknown data semantics, strict JSON serialization with value preservation,
error response consistency, and resource protections.
"""

import io
import json
from enum import Enum
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

from backend import create_app
from backend.routes.api import MAX_IMAGE_SIZE_BYTES
from backend.services.food_analysis_service import (
    analyze_food,
    FoodAnalysisResult,
    FoodSafetyResult,
    FoodSafetyIngredientResult,
    AllergyResult,
    InvalidCategoryError,
    ImageProcessingError,
)
from backend.services.food_analysis_service.models import _sanitize_for_serialization
from backend.services.food_status_service import (
    STATUS_GREEN,
    STATUS_YELLOW,
    STATUS_ORANGE,
    STATUS_RED,
    STATUS_UNAVAILABLE,
)


def _create_dummy_image_bytes(format="JPEG", size=(100, 100), color="white") -> bytes:
    """Helper to generate valid in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestFoodBackendHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    # ----------------------------------------------------------------------
    # 1. Request Validation Tests
    # ----------------------------------------------------------------------
    def test_missing_image_returns_400(self):
        """Request without image file is rejected with 400 and structured JSON."""
        resp = self.client.post(
            "/api/food/analyze",
            data={"category": "food"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertIn("Image file is required", data["error"])
        self.assertFalse(data.get("success", True))

    def test_empty_image_bytes_returns_400(self):
        """Zero-byte image upload is rejected with 400 and no model execution."""
        resp = self.client.post(
            "/api/food/analyze",
            data={
                "image": (io.BytesIO(b""), "empty.jpg", "image/jpeg"),
                "category": "food",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertIn("empty", data["error"])
        self.assertFalse(data.get("success", True))

    def test_corrupt_image_bytes_returns_400(self):
        """Malformed or non-image bytes are rejected with 400 without downstream execution."""
        corrupt_bytes = b"NOT_A_VALID_IMAGE_CONTENT_HEADER_XYZ"
        resp = self.client.post(
            "/api/food/analyze",
            data={
                "image": (io.BytesIO(corrupt_bytes), "corrupt.jpg", "image/jpeg"),
                "category": "food",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertIn("Invalid or corrupt image", data["error"])
        self.assertFalse(data.get("success", True))

    def test_unsupported_image_format_rejected(self):
        """Unsupported extensions (.txt, .pdf, .bin) are rejected with 400."""
        for filename, mime in [
            ("doc.txt", "text/plain"),
            ("scan.pdf", "application/pdf"),
            ("raw.bin", "application/octet-stream"),
        ]:
            resp = self.client.post(
                "/api/food/analyze",
                data={
                    "image": (io.BytesIO(b"data"), filename, mime),
                    "category": "food",
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            self.assertIn("error", data)
            self.assertIn("Only JPG, JPEG, PNG, and WEBP", data["error"])
            self.assertFalse(data.get("success", True))

    def test_oversized_image_rejected(self):
        """Image exceeding MAX_IMAGE_SIZE_BYTES (16MB) is rejected with 413 JSON."""
        oversized_data = b"0" * (MAX_IMAGE_SIZE_BYTES + 1024)
        resp = self.client.post(
            "/api/food/analyze",
            data={
                "image": (io.BytesIO(oversized_data), "large.jpg", "image/jpeg"),
                "category": "food",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 413)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertIn("exceeds the maximum allowed size", data["error"])
        self.assertFalse(data.get("success", True))

    # ----------------------------------------------------------------------
    # 2. Category Routing & Domain Isolation Tests
    # ----------------------------------------------------------------------
    def test_category_routing_food_accepted(self):
        """Category 'food' is accepted."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.food_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {"domain": "food", "ingredients": [], "nutrition": None}
            result = analyze_food(dummy_bytes, category="food")
            self.assertTrue(result.success)
            self.assertEqual(result.category, "food")

    def test_personal_care_strictly_rejected_from_food_endpoint(self):
        """Personal care category is rejected with 400 and never processed by food models."""
        dummy_bytes = _create_dummy_image_bytes()
        resp = self.client.post(
            "/api/food/analyze",
            data={
                "image": (io.BytesIO(dummy_bytes), "test.jpg", "image/jpeg"),
                "category": "personal_care",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertIn("strictly handles 'food'", data["error"])
        self.assertFalse(data.get("success", True))

    def test_invalid_and_missing_categories_rejected(self):
        """Missing, whitespace, and non-food categories are rejected with 400."""
        dummy_bytes = _create_dummy_image_bytes()
        for cat in ["", "   ", "automotive", "cosmetics", "electronics"]:
            resp = self.client.post(
                "/api/food/analyze",
                data={
                    "image": (io.BytesIO(dummy_bytes), "test.jpg", "image/jpeg"),
                    "category": cat,
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            self.assertIn("error", data)
            self.assertFalse(data.get("success", True))

    # ----------------------------------------------------------------------
    # 3. Pipeline Failure Isolation Tests
    # ----------------------------------------------------------------------
    def test_ocr_complete_failure_halts_downstream_and_sets_presentation_unavailable(self):
        """When OCR fails, downstream models are NOT called, and presentation is unavailable."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.food_analysis_service.analyzer.run_ocr") as mock_ocr, \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety") as mock_fs, \
             patch("backend.services.food_analysis_service.analyzer.calculate_nutrition_score") as mock_nut, \
             patch("backend.services.food_analysis_service.analyzer.calculate_allergy_risk") as mock_al:

            mock_ocr.side_effect = RuntimeError("PaddleOCR CUDA initialization failure")

            result = analyze_food(dummy_bytes, category="food")

            # Contract verification
            self.assertFalse(result.success)
            self.assertEqual(result.category, "food")
            self.assertIsNone(result.ocr)
            self.assertIsNone(result.food_safety)
            self.assertIsNone(result.nutrition)
            self.assertIsNone(result.allergy)
            self.assertTrue(len(result.errors) > 0)
            self.assertIn("OCR processing failed", result.errors[0])

            # Downstream models must NOT be executed
            mock_fs.assert_not_called()
            mock_nut.assert_not_called()
            mock_al.assert_not_called()

            # Status mapping must be explicitly unavailable across all three dimensions
            self.assertIsNotNone(result.presentation)
            self.assertEqual(result.presentation["food_safety"]["status"], STATUS_UNAVAILABLE)
            self.assertEqual(result.presentation["allergy"]["status"], STATUS_UNAVAILABLE)
            self.assertEqual(result.presentation["nutrition"]["status"], STATUS_UNAVAILABLE)

    def test_food_safety_failure_isolates_and_preserves_nutrition_and_allergy(self):
        """Food Safety ML inference crash does not corrupt Nutrition or Allergy results."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_ocr_data = {
            "domain": "food",
            "ingredients": [{"matched_name": "Almonds", "raw_text": "Almonds"}],
            "nutrition": {
                "energy": {"value": 500.0, "unit": "kcal", "per_100g": {"value": 500.0, "unit": "kcal"}},
                "total_sugars": {"value": 5.0, "unit": "g", "per_100g": {"value": 5.0, "unit": "g"}},
                "total_fat": {"value": 45.0, "unit": "g", "per_100g": {"value": 45.0, "unit": "g"}},
            },
            "raw_text": {"all_text": "Almonds", "ingredients_text": "Almonds", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mock_ocr_data), \
             patch("backend.services.food_analysis_service.analyzer.predict_food_safety", side_effect=RuntimeError("Model inference error")):

            result = analyze_food(dummy_bytes, category="food")
            self.assertTrue(result.success)

            # Food safety is cleanly marked error
            self.assertEqual(result.food_safety["status"], "error")
            self.assertEqual(result.food_safety["total_ingredients"], 0)

            # Allergy remains valid and accurate (Almonds -> High -> red)
            self.assertEqual(result.allergy["status"], "success")
            self.assertEqual(result.allergy["product_risk_level"], "High")
            self.assertEqual(result.presentation["allergy"]["status"], STATUS_RED)

            # Nutrition remains valid
            self.assertEqual(result.nutrition["status"], "scored")
            self.assertIsNotNone(result.nutrition["nutrition_score"])

    def test_nutrition_failure_isolates_and_preserves_safety_and_allergy(self):
        """Nutrition calculation failure does not corrupt Food Safety or Allergy results."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_ocr_data = {
            "domain": "food",
            "ingredients": [{"matched_name": "Sugar", "raw_text": "Sugar"}],
            "nutrition": {"energy": {"value": 100.0, "unit": "kcal"}},
            "raw_text": {"all_text": "Sugar", "ingredients_text": "Sugar", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mock_ocr_data), \
             patch("backend.services.food_analysis_service.analyzer.calculate_nutrition_score", side_effect=ValueError("Invalid nutrient math")):

            result = analyze_food(dummy_bytes, category="food")
            self.assertTrue(result.success)

            # Nutrition is marked error with unavailable presentation
            self.assertEqual(result.nutrition["status"], "error")
            self.assertIsNone(result.nutrition["nutrition_score"])
            self.assertEqual(result.presentation["nutrition"]["status"], STATUS_UNAVAILABLE)

            # Safety and Allergy remain unaffected
            self.assertEqual(result.food_safety["status"], "success")
            self.assertEqual(result.allergy["status"], "success")

    def test_status_mapping_failure_does_not_destroy_raw_outputs(self):
        """Unexpected exception in presentation mapping preserves raw component dictionaries."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_ocr_data = {
            "domain": "food",
            "ingredients": [{"matched_name": "Sugar", "raw_text": "Sugar"}],
            "nutrition": None,
            "raw_text": {"all_text": "Sugar", "ingredients_text": "Sugar", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mock_ocr_data), \
             patch("backend.services.food_analysis_service.analyzer.map_food_analysis_presentation", side_effect=TypeError("Unexpected presentation mapper error")):

            result = analyze_food(dummy_bytes, category="food")
            self.assertTrue(result.success)

            # Raw component outputs are intact
            self.assertIsNotNone(result.food_safety)
            self.assertEqual(result.food_safety["status"], "success")
            self.assertIsNotNone(result.allergy)
            self.assertEqual(result.allergy["status"], "success")

            # Fallback presentation is unavailable
            self.assertIsNotNone(result.presentation)
            self.assertEqual(result.presentation["food_safety"]["status"], STATUS_UNAVAILABLE)
            self.assertEqual(result.presentation["allergy"]["status"], STATUS_UNAVAILABLE)
            self.assertEqual(result.presentation["nutrition"]["status"], STATUS_UNAVAILABLE)

            # Warning captured
            self.assertTrue(any("Presentation status mapping failed" in w for w in result.warnings))

    # ----------------------------------------------------------------------
    # 4. Missing / Unknown Data Semantics Tests
    # ----------------------------------------------------------------------
    def test_unknown_ingredient_never_converts_to_safe_or_allergen_free(self):
        """Unknown ingredient must never be coerced to 'No Risk', 'Allergen-Free', or 'Very Safe'."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_ocr_data = {
            "domain": "food",
            "ingredients": [{"matched_name": "Unobtainium Extracted Compound", "raw_text": "Unobtainium"}],
            "nutrition": None,
            "raw_text": {"all_text": "Unobtainium", "ingredients_text": "Unobtainium", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mock_ocr_data):
            result = analyze_food(dummy_bytes, category="food")

            # Allergy lookup for unknown ingredient:
            al = result.allergy
            self.assertNotEqual(al.get("product_risk_level"), "No Risk")
            self.assertNotEqual(al.get("product_ui_label"), "Allergen-Free")
            self.assertEqual(result.presentation["allergy"]["status"], STATUS_UNAVAILABLE)
            self.assertNotEqual(result.presentation["allergy"]["status"], STATUS_GREEN)

    def test_missing_nutrition_score_remains_none_and_unavailable_never_zero(self):
        """Missing nutrition must strictly remain score=None, status=unavailable, and never coerced to 0.0 or red."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_ocr_data = {
            "domain": "food",
            "ingredients": [{"matched_name": "Sugar", "raw_text": "Sugar"}],
            "nutrition": None,  # Missing nutrition
            "raw_text": {"all_text": "Sugar", "ingredients_text": "Sugar", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mock_ocr_data):
            result = analyze_food(dummy_bytes, category="food")

            nut = result.nutrition
            self.assertIsNone(nut["nutrition_score"])
            self.assertNotEqual(nut["nutrition_score"], 0)
            self.assertNotEqual(nut["nutrition_score"], 0.0)

            # Presentation must be unavailable, NEVER red
            self.assertEqual(result.presentation["nutrition"]["status"], STATUS_UNAVAILABLE)
            self.assertNotEqual(result.presentation["nutrition"]["status"], STATUS_RED)

    # ----------------------------------------------------------------------
    # 5. Serialization Hardening & Strict Semantic Preservation Tests
    # ----------------------------------------------------------------------
    def test_sanitizer_strict_preservation_of_none_zero_and_booleans(self):
        """
        Critical requirement: _sanitize_for_serialization must strictly preserve:
          - None is None (never converted to 0, false, or string)
          - 0 is int 0 (identity and type preserved)
          - 0.0 is float 0.0 (identity and type preserved)
          - False is bool False (never coerced to int 0)
          - True is bool True
        """
        # None
        res_none = _sanitize_for_serialization(None)
        self.assertIsNone(res_none)

        # 0 (integer)
        res_zero = _sanitize_for_serialization(0)
        self.assertEqual(res_zero, 0)
        self.assertIs(type(res_zero), int)
        self.assertIsNot(type(res_zero), bool)

        # 0.0 (float)
        res_zero_f = _sanitize_for_serialization(0.0)
        self.assertEqual(res_zero_f, 0.0)
        self.assertIs(type(res_zero_f), float)

        # Booleans
        res_false = _sanitize_for_serialization(False)
        self.assertIs(res_false, False)
        self.assertIs(type(res_false), bool)

        res_true = _sanitize_for_serialization(True)
        self.assertIs(res_true, True)
        self.assertIs(type(res_true), bool)

        # In nested dicts
        nested = {
            "score_none": None,
            "score_zero": 0.0,
            "count_zero": 0,
            "flag_false": False,
            "flag_true": True,
        }
        res_nested = _sanitize_for_serialization(nested)
        self.assertIsNone(res_nested["score_none"])
        self.assertEqual(res_nested["score_zero"], 0.0)
        self.assertIs(type(res_nested["score_zero"]), float)
        self.assertEqual(res_nested["count_zero"], 0)
        self.assertIs(type(res_nested["count_zero"]), int)
        self.assertIs(res_nested["flag_false"], False)
        self.assertIs(res_nested["flag_true"], True)

    def test_sanitizer_handles_numpy_and_custom_types_cleanly(self):
        """NumPy scalars, NumPy arrays, sets, enums, and dataclasses convert cleanly to JSON-safe primitives."""
        class TestEnum(Enum):
            ALPHA = "alpha_val"

        data_with_numpy = {
            "int_val": np.int64(42),
            "float_val": np.float32(3.14),
            "bool_val": np.bool_(True),
            "arr_val": np.array([10, 20, 30], dtype=np.int32),
            "set_val": {"banana", "apple"},
            "enum_val": TestEnum.ALPHA,
            "none_val": None,
            "zero_val": 0,
        }

        sanitized = _sanitize_for_serialization(data_with_numpy)

        # Verify exact primitives
        self.assertIs(type(sanitized["int_val"]), int)
        self.assertEqual(sanitized["int_val"], 42)
        self.assertIs(type(sanitized["float_val"]), float)
        self.assertAlmostEqual(sanitized["float_val"], 3.14, places=2)
        self.assertIs(type(sanitized["bool_val"]), bool)
        self.assertTrue(sanitized["bool_val"])
        self.assertEqual(sanitized["arr_val"], [10, 20, 30])
        self.assertEqual(sanitized["set_val"], ["apple", "banana"])
        self.assertEqual(sanitized["enum_val"], "alpha_val")
        self.assertIsNone(sanitized["none_val"])
        self.assertEqual(sanitized["zero_val"], 0)

        # Must be valid in standard json.dumps
        dumped = json.dumps(sanitized)
        self.assertIsInstance(dumped, str)

    def test_food_analysis_result_to_dict_full_json_serializability(self):
        """End-to-end FoodAnalysisResult.to_dict() is directly serializable via json.dumps."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_ocr_data = {
            "domain": "food",
            "ingredients": [
                {"matched_name": "Sugar", "raw_text": "Sugar"},
                {"matched_name": "Almonds", "raw_text": "Almonds"},
            ],
            "nutrition": {
                "energy": {"value": 500.0, "unit": "kcal", "per_100g": {"value": 500.0, "unit": "kcal"}},
                "total_sugars": {"value": 20.0, "unit": "g", "per_100g": {"value": 20.0, "unit": "g"}},
                "total_fat": {"value": 30.0, "unit": "g", "per_100g": {"value": 30.0, "unit": "g"}},
                "saturated_fat": {"value": 4.0, "unit": "g", "per_100g": {"value": 4.0, "unit": "g"}},
                "sodium": {"value": 50.0, "unit": "mg", "per_100g": {"value": 50.0, "unit": "mg"}},
            },
            "raw_text": {"all_text": "Sugar Almonds", "ingredients_text": "Sugar, Almonds", "nutrition_text": ""},
        }

        with patch("backend.services.food_analysis_service.analyzer.run_ocr", return_value=mock_ocr_data):
            result = analyze_food(dummy_bytes, category="food")
            res_dict = result.to_dict()

            # Must serialize with Python standard json.dumps without error
            json_str = json.dumps(res_dict)
            self.assertIsInstance(json_str, str)
            reloaded = json.loads(json_str)
            self.assertEqual(reloaded["category"], "food")
            self.assertTrue(reloaded["success"])
            self.assertIn("presentation", reloaded)
            self.assertIn("food_safety", reloaded["presentation"])
            self.assertIn("allergy", reloaded["presentation"])
            self.assertIn("nutrition", reloaded["presentation"])

    # ----------------------------------------------------------------------
    # 6. Error Response Consistency Tests
    # ----------------------------------------------------------------------
    def test_all_api_validation_errors_have_consistent_structure(self):
        """All client-facing validation errors from /api/food/analyze return 'error' and 'success': False."""
        endpoints_to_test = [
            # Missing category
            ({}, 400),
            # Invalid category
            ({"category": "automotive"}, 400),
            # Personal care category
            ({"category": "personal_care"}, 400),
            # Missing image
            ({"category": "food"}, 400),
        ]

        for payload, expected_status in endpoints_to_test:
            resp = self.client.post(
                "/api/food/analyze",
                data=payload,
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, expected_status)
            data = resp.get_json()
            self.assertIsInstance(data, dict)
            self.assertIn("error", data)
            self.assertIsInstance(data["error"], str)
            self.assertFalse(data.get("success", True))


if __name__ == "__main__":
    unittest.main()
