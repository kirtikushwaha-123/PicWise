import io
import os
import unittest

from backend import create_app
from backend.services.knowledge_base import KnowledgeBase


class TestOCRIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.from_env()
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_food_analysis_success(self):
        """Test 1: Valid food image + category='food' returns structured ingredients and nutrition dict."""
        fixture_path = os.path.join("tests", "fixtures", "product_food.jpeg")
        self.assertTrue(os.path.exists(fixture_path), f"Fixture {fixture_path} not found")

        with open(fixture_path, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                data={
                    "image": (f, "product_food.jpeg", "image/jpeg"),
                    "category": "food",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        # Check domain
        self.assertEqual(data.get("product", {}).get("domain"), "food")

        # Check ingredients
        ingredients = data.get("ingredients", [])
        self.assertIsInstance(ingredients, list)
        self.assertGreater(len(ingredients), 0, "Expected at least one extracted ingredient for food fixture")
        for item in ingredients:
            self.assertIn("name", item)
            self.assertIn("matched", item)
            self.assertIn("confidence", item)
            self.assertEqual(item.get("domain"), "food")

        # Check nutrition dictionary
        nutrition = data.get("nutrition")
        self.assertIsInstance(nutrition, dict, "Expected nutrition to be a dictionary for food")
        self.assertGreater(len(nutrition), 0, "Expected extracted nutritional facts for food fixture")

        # Personal care should be empty for food
        personal_care = data.get("personalCare")
        self.assertEqual(personal_care, [])

    def test_personal_care_analysis_success(self):
        """Test 2: Valid personal care image + category='personal_care' returns ingredients and skips nutrition."""
        fixture_path = os.path.join("tests", "fixtures", "product_personal_care.png")
        self.assertTrue(os.path.exists(fixture_path), f"Fixture {fixture_path} not found")

        with open(fixture_path, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                data={
                    "image": (f, "product_personal_care.png", "image/png"),
                    "category": "personal_care",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        # Check domain
        self.assertEqual(data.get("product", {}).get("domain"), "personal_care")

        # Check ingredients
        ingredients = data.get("ingredients", [])
        self.assertIsInstance(ingredients, list)
        self.assertGreater(len(ingredients), 0, "Expected at least one extracted ingredient for personal care")
        for item in ingredients:
            self.assertIn("name", item)
            self.assertEqual(item.get("domain"), "personal_care")

        # Check that nutrition is strictly None (skipped for personal care)
        self.assertIsNone(data.get("nutrition"), "Expected nutrition to be None for personal care")

        # Check that personalCare data is present
        personal_care = data.get("personalCare")
        self.assertIsInstance(personal_care, list)
        self.assertGreater(len(personal_care), 0, "Expected personalCare list to be populated")

    def test_missing_category_returns_400(self):
        """Test 3: Missing category in request returns 400 Bad Request."""
        fixture_path = os.path.join("tests", "fixtures", "test_product.jpg")
        with open(fixture_path, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                data={
                    "image": (f, "test_product.jpg", "image/jpeg"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertIn("category", data["error"].lower())

    def test_invalid_category_returns_400(self):
        """Test 4: Invalid category (e.g. 'automotive') returns 400 Bad Request."""
        fixture_path = os.path.join("tests", "fixtures", "test_product.jpg")
        with open(fixture_path, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                data={
                    "image": (f, "test_product.jpg", "image/jpeg"),
                    "category": "automotive",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertIn("invalid category", data["error"].lower())

    def test_corrupt_image_returns_400(self):
        """Test 5: Corrupt or unreadable image bytes returns 400 Bad Request."""
        corrupt_data = io.BytesIO(b"NotAValidImageHeaderOrContent123456789")
        response = self.client.post(
            "/api/analyze",
            data={
                "image": (corrupt_data, "bad_image.png", "image/png"),
                "category": "food",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertIn("corrupt", data["error"].lower())

    def test_empty_image_returns_400(self):
        """Test 6: Empty (0 bytes) image file returns 400 Bad Request."""
        empty_data = io.BytesIO(b"")
        response = self.client.post(
            "/api/analyze",
            data={
                "image": (empty_data, "empty.png", "image/png"),
                "category": "food",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
