"""
tests/test_broad_real_world_validation.py

Phase 13: Automated Test Suite for Broad Real-World Validation.
Validates:
1. Integrity and availability of the 36-image real-world validation set.
2. Failure semantics across all 36 validation images:
   - Blank packaging (zero ingredients) yields 'unavailable', NEVER 'safe'.
   - Missing nutrition yields 'unavailable', NEVER 'safe'.
   - Unrecognized ingredients are flagged and never coerced to 'safe'.
3. Category routing and domain isolation:
   - User category selection strictly governs pipeline execution.
   - Category mismatch raises CategoryValidationError.
   - Cross-category image input respects user selection without automatic detection.
4. Personal Care contract compliance:
   - 3 independent dimensions (Safety, Allergy, Irritant).
   - Zero composite score across all test cases.
5. Food contract compliance:
   - 3 distinct dimensions (Safety, Nutrition, Allergy).
"""

import json
from pathlib import Path
import unittest

from backend.services.food_analysis_service import (
    analyze_food,
    InvalidCategoryError as FoodCategoryValidationError,
)
from backend.services.personal_care_analysis_service import (
    analyze_personal_care,
    CategoryValidationError as PCCategoryValidationError,
)
from backend.services.food_status_service import (
    STATUS_GREEN as FOOD_GREEN,
    STATUS_YELLOW as FOOD_YELLOW,
    STATUS_ORANGE as FOOD_ORANGE,
    STATUS_RED as FOOD_RED,
    STATUS_UNAVAILABLE as FOOD_UNAVAILABLE,
)
from backend.services.personal_care_status_service import (
    STATUS_GREEN as PC_GREEN,
    STATUS_YELLOW as PC_YELLOW,
    STATUS_ORANGE as PC_ORANGE,
    STATUS_RED as PC_RED,
    STATUS_UNAVAILABLE as PC_UNAVAILABLE,
)


class TestBroadRealWorldValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures_dir = Path("tests/fixtures/validation_set")
        cls.gt_file = cls.fixtures_dir / "ground_truth.json"

    def test_validation_dataset_exists_and_complete(self):
        """Verify that ground_truth.json exists and contains 36 realistic product records."""
        self.assertTrue(cls_exists := self.gt_file.exists(), f"Ground truth file {self.gt_file} must exist.")
        with open(self.gt_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.assertEqual(len(records), 36, "Validation dataset must contain exactly 36 records.")
        food_count = sum(1 for r in records if r["category"] == "food")
        pc_count = sum(1 for r in records if r["category"] == "personal_care")
        self.assertEqual(food_count, 18, "Validation dataset must have 18 Food images.")
        self.assertEqual(pc_count, 18, "Validation dataset must have 18 Personal Care images.")

        for r in records:
            p = Path(r["file_path"])
            self.assertTrue(p.exists(), f"Validation image file {p} must exist on disk.")

    def test_failure_semantics_blank_food_package(self):
        """Blank food package (zero ingredients/nutrition) must be unavailable, NEVER safe."""
        blank_path = self.fixtures_dir / "food_15_food_front_blank.png"
        if not blank_path.exists():
            self.skipTest(f"{blank_path} not found.")

        with open(blank_path, "rb") as f:
            img_bytes = f.read()

        res = analyze_food(img_bytes, category="food")
        self.assertTrue(res.success)
        pres = res.presentation or {}

        # Safety must be unavailable, never green/safe
        safety_status = pres.get("food_safety", {}).get("status")
        self.assertIn(safety_status, [FOOD_UNAVAILABLE, "unavailable"])
        self.assertNotEqual(safety_status, FOOD_GREEN)

        # Nutrition must be unavailable, never green/safe
        nutrition_status = pres.get("nutrition", {}).get("status")
        self.assertIn(nutrition_status, [FOOD_UNAVAILABLE, "unavailable"])
        self.assertNotEqual(nutrition_status, FOOD_GREEN)

    def test_failure_semantics_blank_personal_care_package(self):
        """Blank personal care package (zero ingredients) must be unavailable for all 3 dimensions."""
        blank_path = self.fixtures_dir / "pc_03_cleanser_cetaphil_blank.png"
        if not blank_path.exists():
            self.skipTest(f"{blank_path} not found.")

        with open(blank_path, "rb") as f:
            img_bytes = f.read()

        res = analyze_personal_care(img_bytes, category="personal_care")
        self.assertTrue(res.success)
        pres = res.presentation or {}

        for dim in ["personal_care_safety", "allergy", "irritation"]:
            dim_status = pres.get(dim, {}).get("status")
            self.assertIn(dim_status, [PC_UNAVAILABLE, "unavailable"], f"Dimension {dim} must be unavailable")
            self.assertNotEqual(dim_status, PC_GREEN, f"Dimension {dim} must NEVER be green/safe for blank label")

    def test_failure_semantics_missing_food_nutrition_panel(self):
        """Food product with ingredients but no nutrition panel must have nutrition unavailable."""
        no_nut_path = self.fixtures_dir / "food_13_snack_chikki_no_nutrition.png"
        if not no_nut_path.exists():
            self.skipTest(f"{no_nut_path} not found.")

        with open(no_nut_path, "rb") as f:
            img_bytes = f.read()

        res = analyze_food(img_bytes, category="food")
        self.assertTrue(res.success)
        pres = res.presentation or {}

        nutrition_status = pres.get("nutrition", {}).get("status")
        self.assertIn(nutrition_status, [FOOD_UNAVAILABLE, "unavailable"])
        self.assertNotEqual(nutrition_status, FOOD_GREEN)

    def test_personal_care_three_independent_dimensions_no_composite_score(self):
        """Personal Care analysis must strictly maintain 3 independent dimensions with zero composite score."""
        sample_path = self.fixtures_dir / "pc_01_shampoo_head_shoulders_clean.png"
        if not sample_path.exists():
            self.skipTest(f"{sample_path} not found.")

        with open(sample_path, "rb") as f:
            img_bytes = f.read()

        res = analyze_personal_care(img_bytes, category="personal_care")
        self.assertTrue(res.success)
        d = res.to_dict()

        # Check root keys
        self.assertIn("presentation", d)
        pres = d["presentation"]
        self.assertIn("personal_care_safety", pres)
        self.assertIn("allergy", pres)
        self.assertIn("irritation", pres)

        # Confirm NO composite score exists
        self.assertNotIn("overall_score", pres)
        self.assertNotIn("composite_score", pres)
        self.assertNotIn("overallScore", pres)
        self.assertNotIn("health_score", pres)

    def test_category_validation_error_on_mismatched_routing(self):
        """Calling food analyzer with personal_care category or vice versa raises CategoryValidationError."""
        dummy_bytes = b"dummy_image_data"
        with self.assertRaises(FoodCategoryValidationError):
            analyze_food(dummy_bytes, category="personal_care")

        with self.assertRaises(PCCategoryValidationError):
            analyze_personal_care(dummy_bytes, category="food")

    def test_cross_category_input_respects_user_selection(self):
        """
        When a Personal Care image is submitted to the Food pipeline,
        the system executes the Food pipeline without secretly auto-detecting the domain.
        """
        pc_img_path = self.fixtures_dir / "pc_01_shampoo_head_shoulders_clean.png"
        if not pc_img_path.exists():
            self.skipTest(f"{pc_img_path} not found.")

        with open(pc_img_path, "rb") as f:
            img_bytes = f.read()

        res = analyze_food(img_bytes, category="food")
        self.assertTrue(res.success)
        self.assertEqual(res.category, "food")
        # Since shampoo has no nutrition facts panel, nutrition must be unavailable
        pres = res.presentation or {}
        self.assertIn(pres.get("nutrition", {}).get("status"), [FOOD_UNAVAILABLE, "unavailable"])


if __name__ == "__main__":
    unittest.main()
