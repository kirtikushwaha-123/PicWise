"""
tests/test_personal_care_status_mapping.py

Unit test suite for Personal Care status mapping and conservative aggregation.
Validates:
- 4-class Safety mapping to green/yellow/red/unavailable
- 4-class Allergy mapping to green/yellow/orange/red/unavailable
- 4-class Irritation mapping to green/yellow/orange/red/unavailable
- Conservative worst-case product-level aggregation per dimension
- Unknown ingredient non-fatal handling in aggregation
- Complete presentation dictionary serialization
"""

import unittest

from backend.services.personal_care_status_service import (
    map_personal_care_safety_status,
    map_personal_care_allergy_status,
    map_personal_care_irritation_status,
    aggregate_product_dimension,
    map_personal_care_presentation,
    STATUS_GREEN,
    STATUS_YELLOW,
    STATUS_ORANGE,
    STATUS_RED,
    STATUS_UNAVAILABLE,
)


class TestPersonalCareStatusMapping(unittest.TestCase):
    # ----------------------------------------------------------------------
    # 1. Safety Status Mapping
    # ----------------------------------------------------------------------
    def test_safety_status_mapping_classes(self):
        """Test exact color and label for each safety risk class."""
        cases = [
            ("Very Safe", STATUS_GREEN, "Very Safe"),
            ("Safe", STATUS_YELLOW, "Safe"),
            ("Moderate Risk", STATUS_ORANGE, "Moderate Risk"),
            ("High Risk", STATUS_RED, "High Risk"),
        ]
        for rc, expected_color, expected_label in cases:
            pres = map_personal_care_safety_status(rc, "success")
            self.assertEqual(pres.status, expected_color)
            self.assertEqual(pres.color, expected_color)
            self.assertEqual(pres.label, expected_label)
            self.assertEqual(pres.risk_class, rc)

    def test_safety_status_mapping_unavailable(self):
        """Test fallback to unavailable when risk_class is missing or status != success."""
        for rc in [None, "", "UnknownRisk", "Invalid"]:
            pres = map_personal_care_safety_status(rc, "success")
            self.assertEqual(pres.status, STATUS_UNAVAILABLE)
            self.assertEqual(pres.color, STATUS_UNAVAILABLE)
            self.assertEqual(pres.label, "Unavailable")

        # When raw_status is not success, must be unavailable even if class is provided
        pres = map_personal_care_safety_status("Very Safe", "ingredient_not_recognized")
        self.assertEqual(pres.status, STATUS_UNAVAILABLE)
        self.assertEqual(pres.label, "Unavailable")

    # ----------------------------------------------------------------------
    # 2. Allergy Status Mapping
    # ----------------------------------------------------------------------
    def test_allergy_status_mapping_classes(self):
        """Test exact color and label for each allergy risk class."""
        cases = [
            ("No Risk", STATUS_GREEN, "No Allergy Risk"),
            ("Low", STATUS_YELLOW, "Low Allergy Risk"),
            ("Medium", STATUS_ORANGE, "Moderate Allergy Risk"),
            ("High", STATUS_RED, "High Allergy Risk"),
        ]
        for rc, expected_color, expected_label in cases:
            pres = map_personal_care_allergy_status(rc, "success")
            self.assertEqual(pres.status, expected_color)
            self.assertEqual(pres.color, expected_color)
            self.assertEqual(pres.label, expected_label)
            self.assertEqual(pres.risk_class, rc)

    def test_allergy_status_mapping_unavailable(self):
        """Test fallback to unavailable for allergy mapping."""
        for rc in [None, "", "Extreme"]:
            pres = map_personal_care_allergy_status(rc, "success")
            self.assertEqual(pres.status, STATUS_UNAVAILABLE)
            self.assertEqual(pres.label, "Unavailable")

        pres = map_personal_care_allergy_status("Low", "model_prediction_failure")
        self.assertEqual(pres.status, STATUS_UNAVAILABLE)

    # ----------------------------------------------------------------------
    # 3. Irritation Status Mapping
    # ----------------------------------------------------------------------
    def test_irritation_status_mapping_classes(self):
        """Test exact color and label for each irritation risk class."""
        cases = [
            ("No Risk", STATUS_GREEN, "No Irritation Risk"),
            ("Low", STATUS_YELLOW, "Low Irritation Risk"),
            ("Medium", STATUS_ORANGE, "Moderate Irritation Risk"),
            ("High", STATUS_RED, "High Irritation Risk"),
        ]
        for rc, expected_color, expected_label in cases:
            pres = map_personal_care_irritation_status(rc, "success")
            self.assertEqual(pres.status, expected_color)
            self.assertEqual(pres.color, expected_color)
            self.assertEqual(pres.label, expected_label)
            self.assertEqual(pres.risk_class, rc)

    def test_irritation_status_mapping_unavailable(self):
        """Test fallback to unavailable for irritation mapping."""
        for rc in [None, "", "Severe"]:
            pres = map_personal_care_irritation_status(rc, "success")
            self.assertEqual(pres.status, STATUS_UNAVAILABLE)
            self.assertEqual(pres.label, "Unavailable")

        pres = map_personal_care_irritation_status("No Risk", "unavailable")
        self.assertEqual(pres.status, STATUS_UNAVAILABLE)

    # ----------------------------------------------------------------------
    # 4. Conservative Product-Level Aggregation
    # ----------------------------------------------------------------------
    def test_aggregation_safety_conservative_ordering(self):
        """Very Safe < Safe < Moderate Risk < High Risk"""
        # Very Safe + Safe -> Safe
        ings = [
            {"status": "success", "safety": {"status": "success", "risk_class": "Very Safe"}},
            {"status": "success", "safety": {"status": "success", "risk_class": "Safe"}},
        ]
        worst, stat = aggregate_product_dimension(ings, "safety")
        self.assertEqual(worst, "Safe")
        self.assertEqual(stat, "success")

        # Safe + Moderate Risk -> Moderate Risk
        ings.append({"status": "success", "safety": {"status": "success", "risk_class": "Moderate Risk"}})
        worst, stat = aggregate_product_dimension(ings, "safety")
        self.assertEqual(worst, "Moderate Risk")

        # Moderate Risk + High Risk -> High Risk
        ings.append({"status": "success", "safety": {"status": "success", "risk_class": "High Risk"}})
        worst, stat = aggregate_product_dimension(ings, "safety")
        self.assertEqual(worst, "High Risk")

    def test_aggregation_allergy_conservative_ordering(self):
        """No Risk < Low < Medium < High"""
        ings = [
            {"status": "success", "allergy": {"status": "success", "risk_class": "No Risk"}},
            {"status": "success", "allergy": {"status": "success", "risk_class": "Low"}},
        ]
        worst, stat = aggregate_product_dimension(ings, "allergy")
        self.assertEqual(worst, "Low")

        ings.append({"status": "success", "allergy": {"status": "success", "risk_class": "Medium"}})
        worst, stat = aggregate_product_dimension(ings, "allergy")
        self.assertEqual(worst, "Medium")

        ings.append({"status": "success", "allergy": {"status": "success", "risk_class": "High"}})
        worst, stat = aggregate_product_dimension(ings, "allergy")
        self.assertEqual(worst, "High")

    def test_aggregation_irritation_conservative_ordering(self):
        """No Risk < Low < Medium < High"""
        ings = [
            {"status": "success", "irritation": {"status": "success", "risk_class": "No Risk"}},
            {"status": "success", "irritation": {"status": "success", "risk_class": "Medium"}},
        ]
        worst, stat = aggregate_product_dimension(ings, "irritation")
        self.assertEqual(worst, "Medium")

    def test_aggregation_unknown_ingredients_behavior(self):
        """
        Unrecognized ingredients must NOT be counted in valid predictions,
        and must NOT invalidate recognized ingredients.
        """
        # 1 recognized (Safe) + 1 unrecognized
        ings = [
            {"status": "success", "safety": {"status": "success", "risk_class": "Safe"}},
            {"status": "ingredient_not_recognized", "safety": {"status": "unavailable", "risk_class": None}},
        ]
        worst, stat = aggregate_product_dimension(ings, "safety")
        self.assertEqual(worst, "Safe")
        self.assertEqual(stat, "success")

        # Only unrecognized ingredients -> unavailable
        only_unknown = [
            {"status": "ingredient_not_recognized", "safety": {"status": "unavailable", "risk_class": None}},
            {"status": "ingredient_not_recognized", "allergy": {"status": "unavailable", "risk_class": None}},
        ]
        worst, stat = aggregate_product_dimension(only_unknown, "safety")
        self.assertIsNone(worst)
        self.assertEqual(stat, STATUS_UNAVAILABLE)

        # Empty ingredient list -> unavailable
        worst, stat = aggregate_product_dimension([], "safety")
        self.assertIsNone(worst)
        self.assertEqual(stat, STATUS_UNAVAILABLE)

    # ----------------------------------------------------------------------
    # 5. Top-Level Presentation Object
    # ----------------------------------------------------------------------
    def test_map_personal_care_presentation_complete(self):
        """Validate top-level presentation object structure and serialization."""
        pres = map_personal_care_presentation(
            safety_risk="Safe",
            safety_status="success",
            allergy_risk="Low",
            allergy_status="success",
            irritation_risk="No Risk",
            irritation_status="success",
        )
        d = pres.to_dict()

        self.assertIn("personal_care_safety", d)
        self.assertIn("allergy", d)
        self.assertIn("irritation", d)

        self.assertEqual(d["personal_care_safety"]["status"], STATUS_YELLOW)
        self.assertEqual(d["personal_care_safety"]["label"], "Safe")

        self.assertEqual(d["allergy"]["status"], STATUS_YELLOW)
        self.assertEqual(d["allergy"]["label"], "Low Allergy Risk")

        self.assertEqual(d["irritation"]["status"], STATUS_GREEN)
        self.assertEqual(d["irritation"]["label"], "No Irritation Risk")


if __name__ == "__main__":
    unittest.main()
