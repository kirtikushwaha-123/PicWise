"""
tests/test_ocr_review_and_quality_gate.py

Targeted tests for:
- Quality gate assessment (sharp vs blurry, blank, low-res)
- POST /api/food/ocr endpoint
- POST /api/food/analyze-text endpoint
- Authoritative edited OCR text flow to Food Safety, Allergy, and Nutrition
- Backward compatibility of POST /api/food/analyze
"""

import io
import os
import unittest
import numpy as np
import cv2
from unittest.mock import patch

from backend import create_app
from backend.services.ocr_service.preprocessing.image_utils import (
    assess_image_quality_gate,
    check_image_quality,
)
from backend.services.food_analysis_service import analyze_food_from_text


def _create_synthetic_image(width=500, height=500, blur_sigma=0, blank=False):
    """Generates synthetic image with controllable sharpness/blur/blank properties."""
    if blank:
        return np.full((height, width, 3), 245, dtype=np.uint8)

    # Image with high-contrast text-like pattern
    img = np.full((height, width, 3), 240, dtype=np.uint8)
    for y in range(50, height - 50, 40):
        cv2.putText(
            img,
            f"SAMPLE PACKAGING TEXT ROW AT Y={y}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
        )

    if blur_sigma > 0:
        ksize = int(blur_sigma * 6) | 1
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

    return img


def _encode_image(img_arr, format=".jpg"):
    success, buf = cv2.imencode(format, img_arr)
    return buf.tobytes()


class TestOCRReviewAndQualityGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    # ----------------------------------------------------------------------
    # Quality Gate Unit Tests
    # ----------------------------------------------------------------------
    def test_sharp_image_accepted_by_quality_gate(self):
        """Sharp image with readable text patterns must be accepted."""
        sharp_img = _create_synthetic_image(width=600, height=800, blur_sigma=0)
        is_usable, status, msg, details = assess_image_quality_gate(sharp_img)
        self.assertTrue(is_usable)
        self.assertEqual(status, "usable")

    def test_severely_blurry_image_rejected(self):
        """Severely blurred image must be rejected with 'unusable_image'."""
        blurry_img = _create_synthetic_image(width=600, height=800, blur_sigma=12)
        is_usable, status, msg, details = assess_image_quality_gate(blurry_img)
        self.assertFalse(is_usable)
        self.assertEqual(status, "unusable_image")
        self.assertIn("Image quality is too low", msg)
        self.assertEqual(details.get("reason"), "severe_blur")

    def test_blank_image_rejected(self):
        """Completely blank/uniform image must be rejected."""
        blank_img = _create_synthetic_image(width=500, height=500, blank=True)
        is_usable, status, msg, details = assess_image_quality_gate(blank_img)
        self.assertFalse(is_usable)
        self.assertEqual(status, "unusable_image")
        self.assertEqual(details.get("reason"), "blank_or_uniform_image")

    def test_extremely_low_res_image_rejected(self):
        """Extremely low resolution image (e.g. 50x50) must be rejected."""
        low_res = np.full((50, 50, 3), 128, dtype=np.uint8)
        is_usable, status, msg, details = assess_image_quality_gate(low_res)
        self.assertFalse(is_usable)
        self.assertEqual(status, "unusable_image")
        self.assertEqual(details.get("reason"), "extremely_low_resolution")

    def test_sharp_image_with_no_ocr_items_not_rejected_as_blurry(self):
        """A sharp image must not be rejected merely because OCR finds 0 items."""
        sharp_img = _create_synthetic_image(width=600, height=800, blur_sigma=0)
        is_usable, status, msg, details = assess_image_quality_gate(
            sharp_img, ocr_items=[]
        )
        self.assertTrue(is_usable)
        self.assertEqual(status, "usable")

    # ----------------------------------------------------------------------
    # POST /api/food/ocr Tests
    # ----------------------------------------------------------------------
    def test_api_food_ocr_rejects_blurry_image(self):
        """POST /api/food/ocr returns HTTP 400 with unusable_image for severe blur."""
        blurry_bytes = _encode_image(_create_synthetic_image(blur_sigma=14))
        resp = self.client.post(
            "/api/food/ocr",
            data={"image": (io.BytesIO(blurry_bytes), "blurry.jpg", "image/jpeg"), "category": "food"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data.get("status"), "unusable_image")
        self.assertIn("Image quality is too low", data.get("error"))

    def test_api_food_ocr_success_returns_structured_raw_text(self):
        """POST /api/food/ocr returns 200 with raw_text sections for a good image."""
        sharp_bytes = _encode_image(_create_synthetic_image(blur_sigma=0))
        mock_ocr_return = {
            "domain": "food",
            "ingredients": [{"name": "Potato", "raw_text": "Potato"}],
            "nutrition": {"energy": {"value": 160.0, "unit": "kcal"}},
            "raw_text": {
                "ingredients_text": "Ingredients: Potato, Salt",
                "nutrition_text": "Calories 160\nTotal Fat 10g",
                "other_text": "Best Before 2026",
                "all_text": "Ingredients: Potato, Salt\nCalories 160\nTotal Fat 10g\nBest Before 2026",
            },
        }
        with patch("backend.routes.api.run_ocr", return_value=mock_ocr_return):
            resp = self.client.post(
                "/api/food/ocr",
                data={"image": (io.BytesIO(sharp_bytes), "sharp.jpg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("status"), "ready_for_review")
        self.assertIn("raw_text", data)
        self.assertEqual(data["raw_text"]["ingredients_text"], "Ingredients: Potato, Salt")
        self.assertEqual(data["raw_text"]["nutrition_text"], "Calories 160\nTotal Fat 10g")
        self.assertEqual(data["raw_text"]["other_text"], "Best Before 2026")

    # ----------------------------------------------------------------------
    # POST /api/food/analyze-text & Authoritative Edited Text Flow
    # ----------------------------------------------------------------------
    def test_api_food_analyze_text_authoritative_edits(self):
        """POST /api/food/analyze-text evaluates the exact user-edited text."""
        payload = {
            "category": "food",
            "ingredients_text": "Potatoes, Salt, Milk Solids Non-Fat",
            "nutrition_text": "Energy: 537 kcal\nProtein: 6.7 g\nTotal Fat: 33.1 g\nTotal Carbohydrate: 53.0 g\nTotal Sugars: 3.4 g",
            "all_text": "Product Label",
        }
        resp = self.client.post(
            "/api/food/analyze-text",
            json=payload,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("category"), "food")

        # 1. Edited ingredients reach Food Safety
        self.assertIn("food_safety", data)
        fs_ings = [it.get("ingredient") for it in data["food_safety"].get("ingredients", [])]
        self.assertTrue(any("potato" in (i or "").lower() for i in fs_ings))

        # 2. Edited ingredients reach Allergy (Milk Solids Non-Fat should trigger allergy risk)
        self.assertIn("allergy", data)
        self.assertEqual(data["allergy"]["status"], "success")
        self.assertIn(data["allergy"]["product_risk_level"], ["Medium", "High"])

        # 3. Edited nutrition reaches Nutrition scoring
        self.assertIn("nutrition", data)
        self.assertEqual(data["nutrition"]["status"], "scored")
        self.assertIsNotNone(data["nutrition"]["nutrition_score"])

        # 4. Presentation matches
        self.assertIn("presentation", data)
        self.assertIn("food_safety", data["presentation"])
        self.assertIn("allergy", data["presentation"])
        self.assertIn("nutrition", data["presentation"])

    def test_edited_text_modifies_nutrition_score(self):
        """Modifying nutrition numbers directly alters the evaluated nutrients and score."""
        # Baseline low-calorie
        res_low = analyze_food_from_text(
            ingredients_text="Wheat Flour",
            nutrition_text="Calories: 100 kcal\nProtein: 5 g\nTotal Fat: 1 g\nTotal Carbohydrate: 20 g\nTotal Sugars: 1 g",
            category="food",
        )
        # Modified high-calorie / high-sugar
        res_high = analyze_food_from_text(
            ingredients_text="Wheat Flour",
            nutrition_text="Calories: 550 kcal\nProtein: 2 g\nTotal Fat: 30 g\nTotal Carbohydrate: 65 g\nTotal Sugars: 45 g",
            category="food",
        )
        self.assertEqual(res_low.nutrition["status"], "scored")
        self.assertEqual(res_high.nutrition["status"], "scored")
        # Healthier profile should have higher nutrition score
        self.assertGreater(res_low.nutrition["nutrition_score"], res_high.nutrition["nutrition_score"])

    def test_edited_protein_alters_nutrients_evaluated_and_score(self):
        """Editing Protein from 5g to 25g alters evaluated nutrients and score via /api/food/analyze-text."""
        payload_baseline = {
            "category": "food",
            "ingredients_text": "Wheat Flour, Water",
            "nutrition_text": "Energy 200 kcal\nProtein 5.0 g\nTotal Fat 5.0 g\nTotal Sugars 5.0 g",
        }
        resp_base = self.client.post("/api/food/analyze-text", json=payload_baseline)
        self.assertEqual(resp_base.status_code, 200)
        data_base = resp_base.get_json()

        payload_edited = {
            "category": "food",
            "ingredients_text": "Wheat Flour, Water",
            "nutrition_text": "Energy 200 kcal\nProtein 25.0 g\nTotal Fat 5.0 g\nTotal Sugars 5.0 g",
        }
        resp_edit = self.client.post("/api/food/analyze-text", json=payload_edited)
        self.assertEqual(resp_edit.status_code, 200)
        data_edit = resp_edit.get_json()

        # Both scored
        self.assertEqual(data_base["nutrition"]["status"], "scored")
        self.assertEqual(data_edit["nutrition"]["status"], "scored")

        # Verify evaluated nutrient for protein reflects exact edited value
        nutrients_base = {n["nutrient"].lower(): n.get("amount_per_100g") for n in data_base["nutrition"]["nutrients_evaluated"]}
        nutrients_edit = {n["nutrient"].lower(): n.get("amount_per_100g") for n in data_edit["nutrition"]["nutrients_evaluated"]}

        self.assertIn("protein", nutrients_base)
        self.assertIn("protein", nutrients_edit)
        self.assertEqual(nutrients_base["protein"], 5.0)
        self.assertEqual(nutrients_edit["protein"], 25.0)

        # Higher protein results in higher nutrition score
        self.assertGreater(data_edit["nutrition"]["nutrition_score"], data_base["nutrition"]["nutrition_score"])

    def test_edited_ingredients_removes_allergen_and_drops_risk_level(self):
        """Editing ingredients to remove Milk Solids Non-Fat removes Milk allergen and drops risk level."""
        payload_with_milk = {
            "category": "food",
            "ingredients_text": "Potato Flakes, Rock Salt, Milk Solids Non-Fat",
            "nutrition_text": "Energy 200 kcal\nProtein 5.0 g\nTotal Fat 5.0 g\nTotal Sugars 5.0 g",
        }
        resp_milk = self.client.post("/api/food/analyze-text", json=payload_with_milk)
        self.assertEqual(resp_milk.status_code, 200)
        data_milk = resp_milk.get_json()

        payload_without_milk = {
            "category": "food",
            "ingredients_text": "Potato Flakes, Rock Salt",
            "nutrition_text": "Energy 200 kcal\nProtein 5.0 g\nTotal Fat 5.0 g\nTotal Sugars 5.0 g",
        }
        resp_no_milk = self.client.post("/api/food/analyze-text", json=payload_without_milk)
        self.assertEqual(resp_no_milk.status_code, 200)
        data_no_milk = resp_no_milk.get_json()

        # Check allergen results
        allergens_milk = [a.lower() for a in data_milk["allergy"].get("allergens_detected", [])]
        allergens_no_milk = [a.lower() for a in data_no_milk["allergy"].get("allergens_detected", [])]

        self.assertTrue(any("milk" in a for a in allergens_milk))
        self.assertFalse(any("milk" in a for a in allergens_no_milk))

        # Risk level should drop from Medium/High to Low/None
        self.assertIn(data_milk["allergy"]["product_risk_level"], ["Medium", "High"])
        self.assertIn(data_no_milk["allergy"]["product_risk_level"], ["Low", "None", "No Risk"])

    def test_original_ocr_never_silently_reused(self):
        """POST /api/food/analyze-text never queries original OCR or uses unedited cached data."""
        # Custom unique ingredient that cannot be in any image
        payload = {
            "category": "food",
            "ingredients_text": "Organic Quinoa, Chia Seeds, Himalayan Pink Salt",
            "nutrition_text": "Energy: 350 kcal\nProtein: 14 g\nTotal Fat: 6 g\nTotal Carbohydrate: 60 g\nTotal Sugars: 2 g",
            "all_text": "Custom Brand Organic Superfood",
        }
        resp = self.client.post("/api/food/analyze-text", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Food safety evaluated our custom ingredients
        raw_ings = data.get("ocr", {}).get("raw_text", {}).get("ingredients_text", "")
        self.assertEqual(raw_ings, payload["ingredients_text"])
        fs_ings = [it.get("ingredient") for it in data["food_safety"].get("ingredients", [])]
        self.assertTrue(any("quinoa" in (i or "").lower() for i in fs_ings))
        self.assertTrue(any("chia" in (i or "").lower() for i in fs_ings))

    # ----------------------------------------------------------------------
    # Backward Compatibility Tests
    # ----------------------------------------------------------------------
    def test_api_food_analyze_remains_backward_compatible(self):
        """POST /api/food/analyze still works directly with uploaded image."""
        fixture_path = os.path.join("tests", "fixtures", "product_food.jpeg")
        if not os.path.exists(fixture_path):
            self.skipTest(f"Fixture {fixture_path} not available.")

        with open(fixture_path, "rb") as f:
            resp = self.client.post(
                "/api/food/analyze",
                data={"image": (f, "product_food.jpeg", "image/jpeg"), "category": "food"},
                content_type="multipart/form-data",
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("category"), "food")
        self.assertIn("presentation", data)

    # ----------------------------------------------------------------------
    # HTML Structure & Sibling Sections Regression Test
    # ----------------------------------------------------------------------
    def test_upload_template_html_structure_and_sibling_sections(self):
        """Verify upload.html contains independent sibling sections and proper loading copy."""
        resp = self.client.get("/upload")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        # 1. Independent sibling sections: loadingState must close before qualityErrorPanel
        import re
        self.assertRegex(
            html,
            r'<section id="loadingState"[^>]*>[\s\S]*?</section>\s*(?:<!--[\s\S]*?-->\s*)?<section id="qualityErrorPanel"',
        )
        # qualityErrorPanel must close before ocrReviewPanel
        self.assertRegex(
            html,
            r'<section id="qualityErrorPanel"[^>]*>[\s\S]*?</section>\s*(?:<!--[\s\S]*?-->\s*)?<section id="ocrReviewPanel"',
        )
        # ocrReviewPanel must close before resultsPanel
        self.assertRegex(
            html,
            r'<section id="ocrReviewPanel"[^>]*>[\s\S]*?</section>\s*(?:<!--[\s\S]*?-->\s*)?<section id="resultsPanel"',
        )

        # 2. Loading messages must be accurate for OCR
        self.assertIn("Reading your product label...", html)
        self.assertIn("PicWise is extracting text from the uploaded label.", html)

        # 3. Cache-busting on app.js
        self.assertRegex(html, r'src="[^"]*app\.js\?v=[^"]*"')


if __name__ == "__main__":
    unittest.main()
