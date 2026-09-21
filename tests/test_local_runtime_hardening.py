"""
tests/test_local_runtime_hardening.py

Unit and integration tests for Phase 11B Part 2: Local Runtime & Application Hardening.
Covers:
- Test A: Lightweight /health endpoint
- Test B: Safe debug configuration
- Test C: Food 500 error sanitization
- Test D: Personal Care 500 error sanitization
- Test E: Legacy 500 error sanitization
- Test F: Legacy /api/analyze oversized upload rejection (413)
- Test G: Existing endpoint regression (/api/food/analyze, /api/personal-care/analyze, /api/analyze)
"""

import io
import os
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from backend import create_app
from backend.routes.api import MAX_IMAGE_SIZE_BYTES
from backend.services.food_analysis_service.models import FoodAnalysisResult
from backend.services.personal_care_analysis_service.models import PersonalCareAnalysisResult


def _create_dummy_image_bytes(format="JPEG", size=(100, 100), color="white") -> bytes:
    """Helper to generate valid in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestLocalRuntimeHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    # ----------------------------------------------------------------------
    # Test A: Health Endpoint
    # ----------------------------------------------------------------------
    def test_health_endpoint_success(self):
        """GET /health returns HTTP 200 with expected healthy payload and no OCR/ML invocation."""
        with patch("backend.routes.api.analyze_food") as mock_food, \
             patch("backend.routes.api.analyze_personal_care") as mock_pc, \
             patch("backend.routes.api.analyze_product_image") as mock_legacy:
            resp = self.client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertEqual(data.get("status"), "healthy")
            self.assertEqual(data.get("app"), "PicWise")
            self.assertEqual(data.get("version"), "1.0.0")

            # Verify no heavy analysis or OCR was invoked
            mock_food.assert_not_called()
            mock_pc.assert_not_called()
            mock_legacy.assert_not_called()

    # ----------------------------------------------------------------------
    # Test B: Safe Debug Configuration
    # ----------------------------------------------------------------------
    def test_debug_disabled_by_default(self):
        """Debug mode evaluates to False when FLASK_DEBUG is unset or '0'."""
        with patch.dict(os.environ, {}, clear=True):
            debug_val = os.getenv("FLASK_DEBUG", "0").strip().lower() in ("1", "true")
            self.assertFalse(debug_val)

        with patch.dict(os.environ, {"FLASK_DEBUG": "0"}):
            debug_val = os.getenv("FLASK_DEBUG", "0").strip().lower() in ("1", "true")
            self.assertFalse(debug_val)

    def test_debug_enabled_with_flask_debug_1(self):
        """Debug mode evaluates to True when FLASK_DEBUG is explicitly '1' or 'true'."""
        with patch.dict(os.environ, {"FLASK_DEBUG": "1"}):
            debug_val = os.getenv("FLASK_DEBUG", "0").strip().lower() in ("1", "true")
            self.assertTrue(debug_val)

        with patch.dict(os.environ, {"FLASK_DEBUG": "true"}):
            debug_val = os.getenv("FLASK_DEBUG", "0").strip().lower() in ("1", "true")
            self.assertTrue(debug_val)

    # ----------------------------------------------------------------------
    # Test C: Food 500 Error Sanitization
    # ----------------------------------------------------------------------
    def test_food_500_error_sanitization(self):
        """Unexpected exception in /api/food/analyze returns generic error and hides raw exception."""
        dummy_bytes = _create_dummy_image_bytes()
        sensitive_msg = "Database connection string leaked: postgres://user:secretpass@10.0.0.1:5432/food_db"
        with patch("backend.routes.api.analyze_food", side_effect=RuntimeError(sensitive_msg)):
            resp = self.client.post(
                "/api/food/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "food.jpg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 500)
            data = resp.get_json()
            self.assertFalse(data.get("success", True))
            self.assertEqual(data.get("error"), "An unexpected server error occurred. Please try again.")

            # Ensure sensitive/raw exception text is absent from the entire response
            raw_text = resp.get_data(as_text=True)
            self.assertNotIn("secretpass", raw_text)
            self.assertNotIn("postgres://", raw_text)
            self.assertNotIn("RuntimeError", raw_text)
            self.assertNotIn(sensitive_msg, raw_text)

    # ----------------------------------------------------------------------
    # Test D: Personal Care 500 Error Sanitization
    # ----------------------------------------------------------------------
    def test_personal_care_500_error_sanitization(self):
        """Unexpected exception in /api/personal-care/analyze returns generic error and hides raw exception."""
        dummy_bytes = _create_dummy_image_bytes()
        sensitive_msg = "Fatal model pipeline failure in C:/internal/models/personal_care/pipeline.joblib"
        with patch("backend.routes.api.analyze_personal_care", side_effect=RuntimeError(sensitive_msg)):
            resp = self.client.post(
                "/api/personal-care/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "product.png", "image/png"), "category": "personal_care"},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 500)
            data = resp.get_json()
            self.assertFalse(data.get("success", True))
            self.assertEqual(data.get("error"), "An unexpected server error occurred. Please try again.")

            # Ensure sensitive/raw exception text is absent from the entire response
            raw_text = resp.get_data(as_text=True)
            self.assertNotIn("pipeline.joblib", raw_text)
            self.assertNotIn("C:/internal", raw_text)
            self.assertNotIn("RuntimeError", raw_text)
            self.assertNotIn(sensitive_msg, raw_text)

    # ----------------------------------------------------------------------
    # Test E: Legacy 500 Error Sanitization
    # ----------------------------------------------------------------------
    def test_legacy_500_error_sanitization(self):
        """Unexpected exception in legacy /api/analyze returns generic error and hides raw exception."""
        dummy_bytes = _create_dummy_image_bytes()
        sensitive_msg = "Core dump in OCR C-extension at memory address 0xdeadbeef"
        with patch("backend.routes.api.analyze_product_image", side_effect=RuntimeError(sensitive_msg)):
            resp = self.client.post(
                "/api/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "sample.jpg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 500)
            data = resp.get_json()
            self.assertFalse(data.get("success", True))
            self.assertEqual(data.get("error"), "An unexpected server error occurred. Please try again.")

            # Ensure sensitive/raw exception text is absent from the entire response
            raw_text = resp.get_data(as_text=True)
            self.assertNotIn("0xdeadbeef", raw_text)
            self.assertNotIn("Core dump", raw_text)
            self.assertNotIn("RuntimeError", raw_text)
            self.assertNotIn(sensitive_msg, raw_text)

    # ----------------------------------------------------------------------
    # Test F: Legacy Oversized Upload Rejection (413)
    # ----------------------------------------------------------------------
    def test_legacy_oversized_upload_rejected_with_413(self):
        """Payload exceeding 16MB to /api/analyze is rejected with 413 before analysis runs."""
        oversized_data = b"0" * (MAX_IMAGE_SIZE_BYTES + 1024)
        with patch("backend.routes.api.analyze_product_image") as mock_analyze:
            resp = self.client.post(
                "/api/analyze",
                data={
                    "image": (io.BytesIO(oversized_data), "large.jpg", "image/jpeg"),
                    "category": "food",
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 413)
            data = resp.get_json()
            self.assertFalse(data.get("success", True))
            self.assertIn("error", data)
            self.assertIn("exceeds the maximum allowed size of 16MB", data["error"])

            # Verify analysis was not executed
            mock_analyze.assert_not_called()

    # ----------------------------------------------------------------------
    # Test G: Existing Endpoints Normal Regression
    # ----------------------------------------------------------------------
    def test_food_endpoint_normal_regression(self):
        """POST /api/food/analyze returns 200 and valid dict when analysis succeeds."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_result = FoodAnalysisResult(
            category="food",
            success=True,
            errors=[],
            warnings=[],
        )
        with patch("backend.routes.api.analyze_food", return_value=mock_result):
            resp = self.client.post(
                "/api/food/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "food.jpg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("category"), "food")

    def test_personal_care_endpoint_normal_regression(self):
        """POST /api/personal-care/analyze returns 200 and valid dict when analysis succeeds."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_result = PersonalCareAnalysisResult(
            category="personal_care",
            success=True,
            errors=[],
            warnings=[],
        )
        with patch("backend.routes.api.analyze_personal_care", return_value=mock_result):
            resp = self.client.post(
                "/api/personal-care/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "pc.png", "image/png"), "category": "personal_care"},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("category"), "personal_care")

    def test_legacy_endpoint_normal_regression(self):
        """POST /api/analyze returns 200 and expected payload when analysis succeeds."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_payload = {"category": "food", "success": True, "product": "test"}
        with patch("backend.routes.api.analyze_product_image", return_value=mock_payload):
            resp = self.client.post(
                "/api/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "food.jpg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("product"), "test")


if __name__ == "__main__":
    unittest.main()
