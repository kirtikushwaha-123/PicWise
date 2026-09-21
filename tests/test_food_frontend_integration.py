"""
tests/test_food_frontend_integration.py

Verification tests for Phase 9J Food Frontend and End-to-End Integration.
Validates:
1. Template rendering: upload.html serves the three independent food cards and warnings container.
2. Static assets: styles.css contains dimension-card and status-pill definitions.
3. JavaScript logic: app.js consumes presentation objects without calculating scores or colors.
4. End-to-end API contract: /api/food/analyze delivers the exact presentation structure consumed by the frontend.
"""

import io
import unittest
from PIL import Image

from backend import create_app


def _create_dummy_image_bytes(format="JPEG", size=(100, 100), color="white") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestFoodFrontendIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_upload_page_renders_food_cards_structure(self):
        """GET /upload renders the 3 independent dimension cards and no overall score container."""
        resp = cls_resp = self.client.get("/upload")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        # Confirm 3 independent dimension cards exist
        self.assertIn('id="foodResultsContainer"', html)
        self.assertIn('id="foodSafetyCard"', html)
        self.assertIn('id="nutritionCard"', html)
        self.assertIn('id="allergyCard"', html)
        self.assertIn('id="foodWarningsBanner"', html)

        # Confirm specific sub-elements
        self.assertIn('id="foodSafetyStatusBadge"', html)
        self.assertIn('id="foodSafetyIngredientsList"', html)
        self.assertIn('id="nutritionStatusBadge"', html)
        self.assertIn('id="nutritionScoreValue"', html)
        self.assertIn('id="nutritionNutrientsList"', html)
        self.assertIn('id="allergyStatusBadge"', html)
        self.assertIn('id="allergyDetectedList"', html)

        # Confirm backwards compatibility container exists
        self.assertIn('id="personalCareResultsContainer"', html)

        # Confirm NO overall product score or traffic-light verdict container exists
        self.assertNotIn('overallScore', html)
        self.assertNotIn('overallHealthScore', html)
        self.assertNotIn('productHealthScore', html)

    def test_static_styles_css_contains_phase9j_rules(self):
        """GET /static/styles.css serves the required status pills and grid layouts."""
        resp = self.client.get("/static/styles.css")
        self.assertEqual(resp.status_code, 200)
        css = resp.get_data(as_text=True)

        self.assertIn(".status-pill.green", css)
        self.assertIn(".status-pill.yellow", css)
        self.assertIn(".status-pill.orange", css)
        self.assertIn(".status-pill.red", css)
        self.assertIn(".status-pill.unavailable", css)
        self.assertIn(".food-cards-grid", css)
        self.assertIn(".dimension-card", css)
        self.assertIn(".warning-banner", css)

    def test_static_app_js_routes_food_and_consumes_presentation(self):
        """GET /static/app.js routes category='food' to /api/food/analyze and uses presentation objects."""
        resp = self.client.get("/static/app.js")
        self.assertEqual(resp.status_code, 200)
        js = resp.get_data(as_text=True)

        # Confirm routing
        self.assertIn("/api/food/analyze", js)
        self.assertIn("renderFoodAnalysis", js)

        # Confirm consumption of presentation object
        self.assertIn("data.presentation", js)
        self.assertIn("pres.food_safety", js)
        self.assertIn("pres.nutrition", js)
        self.assertIn("pres.allergy", js)

        # Confirm allergen handling adheres strictly to allergens_detected
        self.assertIn("data.allergy?.allergens_detected", js)

        # Confirm error handling
        self.assertIn("maxFileSizeBytes", js)

    def test_api_food_analyze_response_contract_matches_frontend_needs(self):
        """POST /api/food/analyze delivers structure consumed by renderFoodAnalysis()."""
        img_bytes = _create_dummy_image_bytes()
        data = {
            "category": "food",
            "image": (io.BytesIO(img_bytes), "test_food.jpg", "image/jpeg"),
        }
        resp = self.client.post("/api/food/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        res = resp.get_json()

        self.assertTrue(res.get("success"))
        self.assertIn("presentation", res)
        pres = res["presentation"]
        self.assertIn("food_safety", pres)
        self.assertIn("nutrition", pres)
        self.assertIn("allergy", pres)

        # Statuses must be valid PicWise color statuses
        allowed_statuses = {"green", "yellow", "orange", "red", "unavailable"}
        self.assertIn(pres["food_safety"]["status"], allowed_statuses)
        self.assertIn(pres["nutrition"]["status"], allowed_statuses)
        self.assertIn(pres["allergy"]["status"], allowed_statuses)

        # Confirm allergy detected list structure
        self.assertIn("allergy", res)
        if res["allergy"]:
            self.assertIn("allergens_detected", res["allergy"])
            self.assertIsInstance(res["allergy"]["allergens_detected"], list)


if __name__ == "__main__":
    unittest.main()
