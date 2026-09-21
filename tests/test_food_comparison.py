"""
tests/test_food_comparison.py

Unit and integration tests for STEP 5: Food-Only Product Comparison.

Requirements:
- Food products only (non-food rejected).
- Accepts 2 to 4 products.
- Rejects < 2 or > 4 products with 400.
- Compares Food Safety, Allergy Risk, Nutrition, and Nutrition Score independently.
- No overall winner, no ranking, no combined/total score.
- Unavailable values remain unavailable.
- Input analysis objects are not mutated.
- GET /compare renders 200 OK and valid page structure without overall winner/score.
"""

import copy
import json
import unittest
from backend import create_app
from backend.services.comparison_service.comparator import (
    validate_comparison_input,
    compare_food_products,
)


class TestFoodComparisonLogic(unittest.TestCase):
    def setUp(self):
        self.sample_food_1 = {
            "category": "food",
            "success": True,
            "food_safety": {
                "risk_class": "Safe",
                "status": "success",
                "presentation_status": "yellow",
            },
            "allergy": {
                "product_risk_level": "No Risk",
                "product_ui_label": "Allergen-Free",
                "status": "success",
                "presentation_status": "green",
            },
            "nutrition": {
                "nutrition_score": 75.0,
                "status": "scored",
                "presentation_status": "yellow",
            },
            "presentation": {
                "food_safety": {"status": "yellow", "label": "Safe", "risk_class": "Safe"},
                "allergy": {"status": "green", "label": "Allergen-Free", "risk_level": "No Risk"},
                "nutrition": {"status": "yellow", "label": "Better Nutrition", "score": 75.0},
            },
        }

        self.sample_food_2 = {
            "category": "food",
            "success": True,
            "food_safety": {
                "risk_class": "Moderate Risk",
                "status": "success",
                "presentation_status": "orange",
            },
            "allergy": {
                "product_risk_level": "High",
                "product_ui_label": "High Allergy Risk",
                "status": "success",
                "presentation_status": "red",
            },
            "nutrition": {
                "nutrition_score": 30.0,
                "status": "scored",
                "presentation_status": "red",
            },
            "presentation": {
                "food_safety": {"status": "orange", "label": "Moderate Risk", "risk_class": "Moderate Risk"},
                "allergy": {"status": "red", "label": "High Allergy Risk", "risk_level": "High"},
                "nutrition": {"status": "red", "label": "Low Nutrition", "score": 30.0},
            },
        }

        self.sample_food_unavailable = {
            "category": "food",
            "success": True,
            "food_safety": {
                "risk_class": None,
                "status": "insufficient_data",
                "presentation_status": "unavailable",
            },
            "allergy": {
                "product_risk_level": None,
                "product_ui_label": "Insufficient Allergy Data",
                "status": "insufficient_data",
                "presentation_status": "unavailable",
            },
            "nutrition": {
                "nutrition_score": None,
                "status": "unavailable",
                "presentation_status": "unavailable",
            },
            "presentation": {
                "food_safety": {"status": "unavailable", "label": None, "risk_class": None},
                "allergy": {"status": "unavailable", "label": "Insufficient Allergy Data", "risk_level": None},
                "nutrition": {"status": "unavailable", "label": "Nutritional Data Unavailable", "score": None},
            },
        }

    # -------------------------------------------------------------------------
    # Validation tests
    # -------------------------------------------------------------------------
    def test_validate_accepts_2_products(self):
        payload = {
            "products": [
                {"name": "Bread", "analysis": self.sample_food_1},
                {"name": "Peanut Butter", "analysis": self.sample_food_2},
            ]
        }
        is_valid, err, validated = validate_comparison_input(payload)
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertEqual(len(validated), 2)

    def test_validate_accepts_4_products(self):
        payload = {
            "products": [
                {"name": "P1", "analysis": self.sample_food_1},
                {"name": "P2", "analysis": self.sample_food_2},
                {"name": "P3", "analysis": self.sample_food_1},
                {"name": "P4", "analysis": self.sample_food_2},
            ]
        }
        is_valid, err, validated = validate_comparison_input(payload)
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertEqual(len(validated), 4)

    def test_validate_rejects_fewer_than_2_products(self):
        payload = {
            "products": [
                {"name": "Solo Product", "analysis": self.sample_food_1},
            ]
        }
        is_valid, err, _ = validate_comparison_input(payload)
        self.assertFalse(is_valid)
        self.assertIn("At least 2 products", err)

    def test_validate_rejects_more_than_4_products(self):
        payload = {
            "products": [
                {"name": f"P{i}", "analysis": self.sample_food_1} for i in range(5)
            ]
        }
        is_valid, err, _ = validate_comparison_input(payload)
        self.assertFalse(is_valid)
        self.assertIn("At most 4 products", err)

    def test_validate_rejects_non_dict_payload(self):
        is_valid, err, _ = validate_comparison_input("not a dict")
        self.assertFalse(is_valid)
        self.assertIn("JSON object", err)

    def test_validate_rejects_missing_or_non_list_products(self):
        is_valid, err, _ = validate_comparison_input({"products": "not-a-list"})
        self.assertFalse(is_valid)
        self.assertIn("list", err)

    def test_validate_rejects_missing_product_name(self):
        payload = {
            "products": [
                {"name": "", "analysis": self.sample_food_1},
                {"name": "Valid Name", "analysis": self.sample_food_2},
            ]
        }
        is_valid, err, _ = validate_comparison_input(payload)
        self.assertFalse(is_valid)
        self.assertIn("non-empty 'name'", err)

    def test_validate_rejects_missing_analysis(self):
        payload = {
            "products": [
                {"name": "P1"},
                {"name": "P2", "analysis": self.sample_food_2},
            ]
        }
        is_valid, err, _ = validate_comparison_input(payload)
        self.assertFalse(is_valid)
        self.assertIn("'analysis' object", err)

    def test_validate_rejects_non_food_category(self):
        cosmetic_analysis = copy.deepcopy(self.sample_food_1)
        cosmetic_analysis["category"] = "cosmetics"

        payload = {
            "products": [
                {"name": "P1", "analysis": self.sample_food_1},
                {"name": "Cosmetic Cream", "analysis": cosmetic_analysis},
            ]
        }
        is_valid, err, _ = validate_comparison_input(payload)
        self.assertFalse(is_valid)
        self.assertIn("only supports food products", err)

    # -------------------------------------------------------------------------
    # Comparison logic tests
    # -------------------------------------------------------------------------
    def test_independent_dimensions_extraction(self):
        """Food Safety, Allergy, and Nutrition are compared independently."""
        products = [
            {"name": "Bread", "analysis": self.sample_food_1},
            {"name": "Snack", "analysis": self.sample_food_2},
        ]
        res = compare_food_products(products)
        self.assertEqual(len(res["products"]), 2)

        p1 = res["products"][0]
        self.assertEqual(p1["name"], "Bread")
        self.assertEqual(p1["food_safety"]["status"], "yellow")
        self.assertEqual(p1["food_safety"]["label"], "Safe")
        self.assertEqual(p1["allergy"]["status"], "green")
        self.assertEqual(p1["allergy"]["label"], "Allergen-Free")
        self.assertEqual(p1["nutrition"]["status"], "yellow")
        self.assertEqual(p1["nutrition_score"], 75.0)

        p2 = res["products"][1]
        self.assertEqual(p2["name"], "Snack")
        self.assertEqual(p2["food_safety"]["status"], "orange")
        self.assertEqual(p2["food_safety"]["label"], "Moderate Risk")
        self.assertEqual(p2["allergy"]["status"], "red")
        self.assertEqual(p2["allergy"]["label"], "High Allergy Risk")
        self.assertEqual(p2["nutrition"]["status"], "red")
        self.assertEqual(p2["nutrition_score"], 30.0)

    def test_no_winner_no_ranking_no_total_score(self):
        """Must NOT contain overall winner, ranking, or total comparison score."""
        products = [
            {"name": "P1", "analysis": self.sample_food_1},
            {"name": "P2", "analysis": self.sample_food_2},
        ]
        res = compare_food_products(products)

        self.assertNotIn("winner", res)
        self.assertNotIn("overall_winner", res)
        self.assertNotIn("rank", res)
        self.assertNotIn("ranking", res)
        self.assertNotIn("ranks", res)
        self.assertNotIn("total_score", res)
        self.assertNotIn("overall_score", res)
        self.assertNotIn("combined_score", res)

        for p in res["products"]:
            self.assertNotIn("food_safety_score", p)
            self.assertNotIn("allergy_score", p)
            self.assertNotIn("rank", p)
            self.assertNotIn("winner", p)

    def test_unavailable_values_preserved(self):
        """Unavailable status and None scores are preserved and never promoted to Safe/Good."""
        products = [
            {"name": "Known Product", "analysis": self.sample_food_1},
            {"name": "Unknown Product", "analysis": self.sample_food_unavailable},
        ]
        res = compare_food_products(products)
        p_unavail = res["products"][1]

        self.assertEqual(p_unavail["food_safety"]["status"], "unavailable")
        self.assertEqual(p_unavail["allergy"]["status"], "unavailable")
        self.assertEqual(p_unavail["nutrition"]["status"], "unavailable")
        self.assertIsNone(p_unavail["nutrition_score"])

    def test_input_analysis_objects_not_mutated(self):
        """Original analysis objects passed to compare_food_products must NOT be mutated."""
        orig_1 = copy.deepcopy(self.sample_food_1)
        orig_2 = copy.deepcopy(self.sample_food_2)

        products = [
            {"name": "P1", "analysis": self.sample_food_1},
            {"name": "P2", "analysis": self.sample_food_2},
        ]

        _ = compare_food_products(products)

        self.assertEqual(self.sample_food_1, orig_1)
        self.assertEqual(self.sample_food_2, orig_2)


class TestFoodComparisonAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def setUp(self):
        self.sample_p1 = {
            "category": "food",
            "success": True,
            "presentation": {
                "food_safety": {"status": "yellow", "label": "Safe", "risk_class": "Safe"},
                "allergy": {"status": "green", "label": "Allergen-Free", "risk_level": "No Risk"},
                "nutrition": {"status": "yellow", "label": "Better Nutrition", "score": 80.0},
            },
        }
        self.sample_p2 = {
            "category": "food",
            "success": True,
            "presentation": {
                "food_safety": {"status": "red", "label": "High Risk", "risk_class": "High Risk"},
                "allergy": {"status": "red", "label": "High Allergy Risk", "risk_level": "High"},
                "nutrition": {"status": "red", "label": "Low Nutrition", "score": 25.0},
            },
        }

    def test_post_compare_200_ok_two_products(self):
        resp = self.client.post(
            "/api/compare",
            json={
                "products": [
                    {"name": "Product A", "analysis": self.sample_p1},
                    {"name": "Product B", "analysis": self.sample_p2},
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(len(data.get("products", [])), 2)
        self.assertEqual(len(data.get("rows", [])), 4)

        # Confirm dimensions
        row_keys = [r["key"] for r in data["rows"]]
        self.assertEqual(row_keys, ["food_safety", "allergy", "nutrition", "nutrition_score"])

        # Confirm no winner / ranking
        self.assertNotIn("winner", data)
        self.assertNotIn("ranking", data)
        self.assertNotIn("total_score", data)

    def test_post_compare_400_one_product(self):
        resp = self.client.post(
            "/api/compare",
            json={
                "products": [
                    {"name": "Product A", "analysis": self.sample_p1},
                ]
            },
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data.get("success"))
        self.assertIn("error", data)

    def test_post_compare_400_five_products(self):
        resp = self.client.post(
            "/api/compare",
            json={
                "products": [
                    {"name": f"P{i}", "analysis": self.sample_p1} for i in range(5)
                ]
            },
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data.get("success"))
        self.assertIn("error", data)

    def test_post_compare_400_non_food(self):
        cosmetic = copy.deepcopy(self.sample_p1)
        cosmetic["category"] = "cosmetics"
        resp = self.client.post(
            "/api/compare",
            json={
                "products": [
                    {"name": "Product A", "analysis": self.sample_p1},
                    {"name": "Product B", "analysis": cosmetic},
                ]
            },
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data.get("success"))
        self.assertIn("only supports food products", data.get("error", ""))

    def test_get_compare_page_200_ok(self):
        """GET /compare returns 200 OK and renders the comparison interface without winner/score."""
        resp = self.client.get("/compare")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("Compare Food Products", html)
        self.assertIn("comparisonTable", html)
        self.assertIn("compareBtn", html)
        self.assertIn("productSlotsContainer", html)

        # Must NOT contain winner or total score
        self.assertNotIn("Winner", html)
        self.assertNotIn("Overall Score", html)
        self.assertNotIn("totalScore", html)
        self.assertNotIn("Rank", html)


if __name__ == "__main__":
    unittest.main()
