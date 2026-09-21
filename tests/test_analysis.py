import os
import unittest
from unittest.mock import patch
from backend import create_app
from backend.services.knowledge_base import KnowledgeBase
from backend.services.analysis_service.analyzer import analyze_product_image


class TestAnalysisService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.from_env()
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.mock_ocr_output = {
            "domain": "food",
            "ingredients": [
                {
                    "raw_text": "Wheat Flour",
                    "matched_name": "Refined Wheat Flour (Maida)",
                    "similarity": 95.0,
                    "method": "fuzzy",
                    "metadata": {"Safety Level": "Safe", "Category": "Grain"},
                }
            ],
            "nutrition": {"energy": {"value": 450.0, "unit": "kcal"}},
            "personalCare": [],
            "raw_text": {"all_text": "Wheat Flour", "ingredients_text": "Wheat Flour", "nutrition_text": ""},
        }

    def test_analysis_pipeline_integration(self):
        fixture_path = os.path.join("tests", "fixtures", "test_product.jpg")
        with open(fixture_path, "rb") as f:
            image_bytes = f.read()

        with patch("backend.services.analysis_service.analyzer.run_ocr", return_value=self.mock_ocr_output):
            response = analyze_product_image(image_bytes, self.kb)

        self.assertIn("product", response)
        self.assertIn("domain", response["product"])
        self.assertIn("ingredients", response)
        self.assertIn("nutrition", response)
        self.assertIn("personalCare", response)
        self.assertIn("warnings", response)

        # Verify ingredient structure
        self.assertGreater(len(response["ingredients"]), 0)
        for item in response["ingredients"]:
            self.assertIn("name", item)
            self.assertIn("matched", item)
            self.assertIn("confidence", item)
            self.assertIn("domain", item)

    def test_api_analyze_endpoint_with_image_fixture(self):
        fixture_path = os.path.join("tests", "fixtures", "test_product.jpg")
        with open(fixture_path, "rb") as f:
            with patch("backend.services.analysis_service.analyzer.run_ocr", return_value=self.mock_ocr_output):
                response = self.client.post(
                    "/api/analyze",
                    data={"image": (f, "test_product.jpg"), "category": "food"},
                    content_type="multipart/form-data"
                )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("product", data)
        self.assertIn("ingredients", data)
        self.assertIn("nutrition", data)
        self.assertIn("personalCare", data)


if __name__ == "__main__":
    unittest.main()

