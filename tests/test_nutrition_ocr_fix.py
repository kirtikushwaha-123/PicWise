"""
tests/test_nutrition_ocr_fix.py

Targeted tests for Nutrition OCR fixes:
1. Multi-column nutrition table detection & absorption
2. Pattern precedence: Saturated Fat and Trans Fat before generic Total Fat
3. Pattern precedence: Added Sugars before Total Sugars
4. Dimensionless calories (e.g. 'Calories 160' -> 160 kcal)
5. Comparison values (e.g. '<0.4 g', '< 0.1 g', '<=0.5 g')
6. Table layouts where nutrient names and values are separated into adjacent lines/cells
7. Nutrients evaluated receives numeric values and calculates score
"""

import unittest
from backend.services.ocr_service.parsing.nutrition_parser import (
    parse_nutrition,
    _parse_nutrition_string_fallback,
    _find_key,
    _extract_value_and_unit,
)
from backend.services.ocr_service.detection.nutrition_region import (
    _absorb_table_columns,
    expand_nutrition_region,
)
from backend.services.nutrition_service.scorer import calculate_nutrition_score


class TestNutritionOCRFix(unittest.TestCase):
    def test_pattern_precedence_saturated_trans_fat(self):
        """Saturated Fat and Trans Fat must be matched before generic Total Fat."""
        self.assertEqual(_find_key("saturated fat 1.5g"), "saturated_fat")
        self.assertEqual(_find_key("saturates 3.54g"), "saturated_fat")
        self.assertEqual(_find_key("saturated 3.54 g"), "saturated_fat")
        self.assertEqual(_find_key("trans fat 0g"), "trans_fat")
        self.assertEqual(_find_key("trans fatty acids 0.1g"), "trans_fat")
        self.assertEqual(_find_key("total fat 10g"), "total_fat")
        self.assertEqual(_find_key("fat 10g"), "total_fat")

    def test_pattern_precedence_added_sugars(self):
        """Added Sugars must be matched before generic Total Sugars."""
        self.assertEqual(_find_key("added sugars 2.0g"), "added_sugars")
        self.assertEqual(_find_key("added sugar 2.0g"), "added_sugars")
        self.assertEqual(_find_key("total sugars 3.4g"), "total_sugars")
        self.assertEqual(_find_key("sugar 14g"), "total_sugars")

    def test_dimensionless_calories_extraction(self):
        """Dimensionless calories (e.g. 'Calories 160') must extract 160 kcal."""
        val_unit = _extract_value_and_unit("Calories 160", nutrient_key="energy")
        self.assertIsNotNone(val_unit)
        val, unit = val_unit
        self.assertEqual(val, 160.0)
        self.assertEqual(unit, "kcal")

        # Also via string fallback
        parsed = _parse_nutrition_string_fallback("Calories 160\nTotal Fat 10g")
        self.assertIn("energy", parsed)
        self.assertEqual(parsed["energy"]["value"], 160.0)
        self.assertEqual(parsed["energy"]["unit"], "kcal")

    def test_comparison_values_extraction(self):
        """Comparison values like '<0.4 g' or '<=0.5 g' must parse correctly."""
        val_unit_1 = _extract_value_and_unit("<0.4 g", nutrient_key="total_sugars")
        self.assertIsNotNone(val_unit_1)
        self.assertEqual(val_unit_1[0], 0.4)
        self.assertEqual(val_unit_1[1], "g")

        val_unit_2 = _extract_value_and_unit("< 0.1 g", nutrient_key="trans_fat")
        self.assertIsNotNone(val_unit_2)
        self.assertEqual(val_unit_2[0], 0.1)
        self.assertEqual(val_unit_2[1], "g")

        val_unit_3 = _extract_value_and_unit("<=0.5 g", nutrient_key="saturated_fat")
        self.assertIsNotNone(val_unit_3)
        self.assertEqual(val_unit_3[0], 0.5)
        self.assertEqual(val_unit_3[1], "g")

    def test_separated_name_and_value_lines(self):
        """Nutrient names and values on separate lines must pair correctly."""
        table_text = """
Calories
160
Total Fat
10g
Saturated Fat
1.5g
Trans Fat
0g
Total Carbohydrate
15g
Total Sugars
1g
Protein
2g
Sodium
170mg
"""
        parsed = _parse_nutrition_string_fallback(table_text)
        self.assertIn("energy", parsed)
        self.assertEqual(parsed["energy"]["value"], 160.0)
        self.assertEqual(parsed["energy"]["unit"], "kcal")

        self.assertIn("total_fat", parsed)
        self.assertEqual(parsed["total_fat"]["value"], 10.0)

        self.assertIn("saturated_fat", parsed)
        self.assertEqual(parsed["saturated_fat"]["value"], 1.5)

        self.assertIn("trans_fat", parsed)
        self.assertEqual(parsed["trans_fat"]["value"], 0.0)

        self.assertIn("total_carbohydrate", parsed)
        self.assertEqual(parsed["total_carbohydrate"]["value"], 15.0)

        self.assertIn("total_sugars", parsed)
        self.assertEqual(parsed["total_sugars"]["value"], 1.0)

        self.assertIn("protein", parsed)
        self.assertEqual(parsed["protein"]["value"], 2.0)

        self.assertIn("sodium", parsed)
        self.assertEqual(parsed["sodium"]["value"], 170.0)

    def test_multi_column_table_column_absorption(self):
        """_absorb_table_columns must absorb value cells horizontally aligned with rows."""
        # Simulated lines: col 0 labels, col 1 values
        col0_lines = [
            {"rect": [50.0, 100.0, 200.0, 130.0], "text": "Nutrition Facts", "column_id": 0},
            {"rect": [50.0, 140.0, 150.0, 165.0], "text": "Calories", "column_id": 0},
            {"rect": [50.0, 170.0, 150.0, 195.0], "text": "Total Fat", "column_id": 0},
            {"rect": [50.0, 200.0, 160.0, 225.0], "text": "Saturated Fat", "column_id": 0},
        ]
        all_lines = list(col0_lines) + [
            {"rect": [400.0, 138.0, 450.0, 163.0], "text": "160", "column_id": 1},
            {"rect": [400.0, 168.0, 450.0, 193.0], "text": "10g", "column_id": 1},
            {"rect": [400.0, 198.0, 450.0, 223.0], "text": "1.5g", "column_id": 1},
            # Competitor line far away or ingredient line
            {"rect": [50.0, 300.0, 400.0, 320.0], "text": "Ingredients: Potatoes, Oil", "ingredient_score": 0.9, "column_id": 0},
        ]

        absorbed = _absorb_table_columns(col0_lines, all_lines, line_h=25.0)
        # Should contain 4 labels + 3 values = 7 lines (not the ingredients line)
        self.assertEqual(len(absorbed), 7)
        absorbed_texts = [ln["text"] for ln in absorbed]
        self.assertIn("160", absorbed_texts)
        self.assertIn("10g", absorbed_texts)
        self.assertIn("1.5g", absorbed_texts)
        self.assertNotIn("Ingredients: Potatoes, Oil", absorbed_texts)

    def test_parsed_nutrition_feeds_scoring_engine(self):
        """Parsed multi-nutrient dict must successfully score in calculate_nutrition_score."""
        raw_nut = {
            "energy": {"value": 160.0, "unit": "kcal"},
            "total_fat": {"value": 10.0, "unit": "g"},
            "saturated_fat": {"value": 1.5, "unit": "g"},
            "total_carbohydrate": {"value": 15.0, "unit": "g"},
            "total_sugars": {"value": 1.0, "unit": "g"},
            "protein": {"value": 2.0, "unit": "g"},
            "sodium": {"value": 170.0, "unit": "mg"},
        }
        res = calculate_nutrition_score(raw_nut, category="food")
        self.assertIsNotNone(res)
        self.assertEqual(res.get("status"), "scored")
        self.assertIsNotNone(res.get("nutrition_score"))
        self.assertGreater(len(res.get("nutrients_evaluated", [])), 3)


if __name__ == "__main__":
    unittest.main()
