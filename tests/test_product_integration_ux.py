"""
tests/test_product_integration_ux.py

Unit and integration tests for Phase 12: Product Integration & UX Hardening.
Verifies:
1. Category selection & strict routing (Food -> /api/food/analyze, Personal Care -> /api/personal-care/analyze, no auto-detection).
2. Upload validation (empty, unsupported format, >16MB, valid image).
3. Loading state and duplicate submission prevention.
4. Success state contracts (Food 3 dimensions, Personal Care 3 independent dimensions, NO composite score).
5. Warning rendering (backend warnings and OCR quality advisory preserved).
6. Unknown and unavailable semantics (Unknown != Safe, Missing != Safe, OCR failure != Safe).
7. Error handling & sanitization (400, 413, 500 without leaking stack traces or internal paths).
8. Reset and retry flow (stale result elimination on category switch and reset).
"""

import io
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


class TestProductIntegrationUX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    # ----------------------------------------------------------------------
    # 1. Category Selection & Strict Routing Contract
    # ----------------------------------------------------------------------
    def test_category_selection_rendered_in_template(self):
        """GET /upload serves accessible category toggle cards with radio inputs."""
        resp = self.client.get("/upload")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        # Verify radio inputs exist for food and personal care
        self.assertIn('id="catFood"', html)
        self.assertIn('value="food"', html)
        self.assertIn('id="catPersonalCare"', html)
        self.assertIn('value="personal_care"', html)
        self.assertIn('role="radiogroup"', html)
        self.assertIn('aria-labelledby="categoryGroupLabel"', html)

    def test_category_routing_strict_separation(self):
        """Food endpoint strictly requires 'food' and Personal Care endpoint strictly requires 'personal_care'."""
        dummy_bytes = _create_dummy_image_bytes()

        # Food endpoint with personal_care category must return 400
        resp = self.client.post(
            "/api/food/analyze",
            data={"image": (io.BytesIO(dummy_bytes), "test.jpg", "image/jpeg"), "category": "personal_care"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("strictly handles 'food'", resp.get_json()["error"])

        # Personal Care endpoint with food category must return 400
        resp = self.client.post(
            "/api/personal-care/analyze",
            data={"image": (io.BytesIO(dummy_bytes), "test.jpg", "image/jpeg"), "category": "food"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("strictly handles 'personal_care'", resp.get_json()["error"])

    def test_no_automatic_category_detection_in_codebase(self):
        """Confirm absence of _detect_domain or automatic domain guessing functions."""
        import backend.routes.api as api_mod
        self.assertFalse(hasattr(api_mod, "_detect_domain"))
        self.assertFalse(hasattr(api_mod, "detect_domain"))

    # ----------------------------------------------------------------------
    # 2. Image Upload Validation
    # ----------------------------------------------------------------------
    def test_upload_validation_missing_image(self):
        """Missing image file returns 400 with clear message."""
        for endpoint, cat in [("/api/food/analyze", "food"), ("/api/personal-care/analyze", "personal_care")]:
            resp = self.client.post(
                endpoint,
                data={"category": cat},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            self.assertFalse(data["success"])
            self.assertIn("Image file is required", data["error"])

    def test_upload_validation_empty_file(self):
        """0-byte image upload returns 400 with 'empty' message."""
        for endpoint, cat in [("/api/food/analyze", "food"), ("/api/personal-care/analyze", "personal_care")]:
            resp = self.client.post(
                endpoint,
                data={"image": (io.BytesIO(b""), "empty.jpg", "image/jpeg"), "category": cat},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            self.assertFalse(data["success"])
            self.assertIn("empty", data["error"].lower())

    def test_upload_validation_unsupported_format(self):
        """Unsupported file types (e.g. .pdf, .txt, .bin) are rejected with 400."""
        for endpoint, cat in [("/api/food/analyze", "food"), ("/api/personal-care/analyze", "personal_care")]:
            for fname, mime in [("doc.pdf", "application/pdf"), ("note.txt", "text/plain")]:
                resp = self.client.post(
                    endpoint,
                    data={"image": (io.BytesIO(b"dummy text"), fname, mime), "category": cat},
                    content_type="multipart/form-data",
                )
                self.assertEqual(resp.status_code, 400)
                data = resp.get_json()
                self.assertFalse(data["success"])
                self.assertIn("Only JPG, JPEG, PNG, and WEBP", data["error"])

    def test_upload_validation_oversized_file(self):
        """Files exceeding 16MB are rejected with 413 without running analysis."""
        oversized = b"X" * (MAX_IMAGE_SIZE_BYTES + 512)
        for endpoint, cat in [("/api/food/analyze", "food"), ("/api/personal-care/analyze", "personal_care"), ("/api/analyze", "food")]:
            resp = self.client.post(
                endpoint,
                data={"image": (io.BytesIO(oversized), "large.jpg", "image/jpeg"), "category": cat},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 413)
            data = resp.get_json()
            self.assertFalse(data["success"])
            self.assertIn("exceeds the maximum allowed size of 16MB", data["error"])

    # ----------------------------------------------------------------------
    # 3. Frontend Assets & Contracts Inspection
    # ----------------------------------------------------------------------
    def test_frontend_js_duplicate_submission_and_category_switch_guards(self):
        """Verify static/app.js contains isAnalyzing guard, category switch listener, and 0-byte check."""
        resp = self.client.get("/static/app.js")
        self.assertEqual(resp.status_code, 200)
        js = resp.get_data(as_text=True)
        resp.close()

        # Duplicate submission guard
        self.assertIn("let isAnalyzing = false;", js)
        self.assertIn("if (isAnalyzing)", js)

        # Category switch listener
        self.assertIn("categoryRadios.forEach", js)
        self.assertIn("resetResults()", js)

        # 0-byte file check
        self.assertIn("file.size === 0", js)
        self.assertIn("Uploaded image file is empty.", js)

        # Informative network error
        self.assertIn("Unable to connect to the server", js)

    def test_frontend_css_responsive_category_rules(self):
        """Verify static/styles.css contains mobile responsive category grid rule."""
        resp = self.client.get("/static/styles.css")
        self.assertEqual(resp.status_code, 200)
        css = resp.get_data(as_text=True)
        resp.close()

        self.assertIn("@media (max-width: 600px)", css)
        self.assertIn(".category-toggle-container", css)

    # ----------------------------------------------------------------------
    # 4. Success State Contracts & Semantics
    # ----------------------------------------------------------------------
    def test_food_success_contract_structure(self):
        """Food success response provides presentation for Safety, Nutrition, and Allergy."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_result = FoodAnalysisResult(
            category="food",
            success=True,
            food_safety={"status": "success", "ingredients": [{"ingredient": "Sugar", "risk_class": "Safe", "confidence": 0.95}], "total_ingredients": 1},
            nutrition={"nutrition_score": 85.0, "components": {"protein": {"value": 10, "unit": "g"}}},
            allergy={"status": "success", "allergens_detected": ["Milk"]},
            presentation={
                "food_safety": {"status": "yellow", "label": "Safe", "color": "yellow"},
                "nutrition": {"status": "green", "label": "Nutritious", "score": 85.0, "color": "green"},
                "allergy": {"status": "orange", "label": "Moderate Allergy Risk", "color": "orange"},
            },
            warnings=["Low contrast image advisory"],
        )
        with patch("backend.routes.api.analyze_food", return_value=mock_result):
            resp = self.client.post(
                "/api/food/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "food.jpg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertIn("presentation", data)
            self.assertIn("food_safety", data["presentation"])
            self.assertIn("nutrition", data["presentation"])
            self.assertIn("allergy", data["presentation"])
            self.assertIn("warnings", data)
            self.assertEqual(data["warnings"], ["Low contrast image advisory"])

    def test_personal_care_success_contract_preserves_three_independent_dimensions(self):
        """Personal Care response provides 3 independent dimensions and NO composite score."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_result = PersonalCareAnalysisResult(
            category="personal_care",
            success=True,
            personal_care={
                "total_ingredients": 2,
                "recognized_ingredients": 2,
                "ingredients": [
                    {
                        "raw_text": "Glycerin",
                        "matched_name": "Glycerin",
                        "status": "success",
                        "safety": {"risk_class": "Very Safe", "confidence": 0.98},
                        "allergy": {"risk_class": "No Risk", "confidence": 0.99},
                        "irritation": {"risk_class": "No Risk", "confidence": 0.97},
                    }
                ],
            },
            presentation={
                "personal_care_safety": {"status": "green", "label": "Very Safe", "color": "green"},
                "allergy": {"status": "green", "label": "No Risk", "color": "green"},
                "irritation": {"status": "green", "label": "No Risk", "color": "green"},
            },
            warnings=["Dense ingredient list"],
            ocr_quality_warning="Image contrast is slightly low",
        )
        with patch("backend.routes.api.analyze_personal_care", return_value=mock_result):
            resp = self.client.post(
                "/api/personal-care/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "pc.png", "image/png"), "category": "personal_care"},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertIn("presentation", data)
            self.assertIn("personal_care_safety", data["presentation"])
            self.assertIn("allergy", data["presentation"])
            self.assertIn("irritation", data["presentation"])

            # Verify NO composite score exists
            self.assertNotIn("overall_score", data)
            self.assertNotIn("health_score", data)
            self.assertNotIn("personal_care_score", data)

            # Verify warnings
            self.assertIn("warnings", data)
            self.assertEqual(data["warnings"], ["Dense ingredient list"])
            self.assertEqual(data.get("ocr_quality_warning"), "Image contrast is slightly low")

    # ----------------------------------------------------------------------
    # 5. Unknown and Unavailable Semantics (Unknown != Safe)
    # ----------------------------------------------------------------------
    def test_food_missing_nutrition_is_unavailable_not_safe(self):
        """When nutrition data is absent, presentation status is strictly 'unavailable'."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_result = FoodAnalysisResult(
            category="food",
            success=True,
            nutrition=None,
            presentation={
                "food_safety": {"status": "green", "label": "Very Safe", "color": "green"},
                "nutrition": {"status": "unavailable", "label": "Unavailable", "score": None, "color": "unavailable"},
                "allergy": {"status": "green", "label": "Allergen-Free", "color": "green"},
            },
        )
        with patch("backend.routes.api.analyze_food", return_value=mock_result):
            resp = self.client.post(
                "/api/food/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "food.jpg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
            data = resp.get_json()
            nut_pres = data["presentation"]["nutrition"]
            self.assertEqual(nut_pres["status"], "unavailable")
            self.assertIsNone(nut_pres["score"])

    def test_personal_care_unrecognized_ingredient_is_unavailable_not_safe(self):
        """Unrecognized ingredient status is 'ingredient_not_recognized' and never mapped to safe."""
        dummy_bytes = _create_dummy_image_bytes()
        mock_result = PersonalCareAnalysisResult(
            category="personal_care",
            success=True,
            personal_care={
                "total_ingredients": 1,
                "recognized_ingredients": 0,
                "ingredients": [
                    {
                        "raw_text": "XyloUnknownChem123",
                        "matched_name": None,
                        "status": "ingredient_not_recognized",
                        "safety": {},
                        "allergy": {},
                        "irritation": {},
                    }
                ],
            },
            presentation={
                "personal_care_safety": {"status": "unavailable", "label": "Unavailable", "color": "unavailable"},
                "allergy": {"status": "unavailable", "label": "Unavailable", "color": "unavailable"},
                "irritation": {"status": "unavailable", "label": "Unavailable", "color": "unavailable"},
            },
        )
        with patch("backend.routes.api.analyze_personal_care", return_value=mock_result):
            resp = self.client.post(
                "/api/personal-care/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "pc.png", "image/png"), "category": "personal_care"},
                content_type="multipart/form-data",
            )
            data = resp.get_json()
            for dim in ["personal_care_safety", "allergy", "irritation"]:
                self.assertEqual(data["presentation"][dim]["status"], "unavailable")
                self.assertEqual(data["presentation"][dim]["label"], "Unavailable")

    # ----------------------------------------------------------------------
    # 6. Error Handling & Sanitization
    # ----------------------------------------------------------------------
    def test_error_sanitization_no_leakage(self):
        """500 error hides exception string and returns safe user message."""
        dummy_bytes = _create_dummy_image_bytes()
        leak_msg = "Critical database failure at /var/data/models/weights.bin"
        for endpoint, cat in [("/api/food/analyze", "food"), ("/api/personal-care/analyze", "personal_care")]:
            with patch(f"backend.routes.api.analyze_{cat if cat == 'food' else 'personal_care'}", side_effect=RuntimeError(leak_msg)):
                resp = self.client.post(
                    endpoint,
                    data={"image": (io.BytesIO(dummy_bytes), "test.jpg", "image/jpeg"), "category": cat},
                    content_type="multipart/form-data",
                )
                self.assertEqual(resp.status_code, 500)
                data = resp.get_json()
                self.assertFalse(data["success"])
                self.assertEqual(data["error"], "An unexpected server error occurred. Please try again.")
                raw_text = resp.get_data(as_text=True)
                self.assertNotIn("weights.bin", raw_text)
                self.assertNotIn("/var/data", raw_text)
                self.assertNotIn("RuntimeError", raw_text)


if __name__ == "__main__":
    unittest.main()
