"""
tests/test_personal_care_real_world_validation.py

Phase 10C: Personal Care Real-World Validation & Robustness Audit Test Suite.
Validates:
1. End-to-end real image fixtures across varied label conditions.
2. Ingredient recognition robustness (canonical, packaging aliases, OCR distortions, unknown chemicals).
3. Semantic enrichment target isolation and attribute integrity.
4. Model inference robustness and failure isolation across 3 independent predictors.
5. Product-level conservative worst-case aggregation for Safety, Allergy, and Irritation.
6. Unknown + known ingredient handling and warning preservation.
7. Complete unavailable scenario handling (unknown != safe).
8. Category routing and API contract compliance.
9. Frontend contract compliance (independent dimensions, no composite score).
"""

import io
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
from PIL import Image

from backend import create_app
from backend.ml.inference.personal_care_service import (
    PersonalCarePredictor,
    TargetPredictor,
    predict_personal_care,
)
from backend.services.personal_care_analysis_service import (
    analyze_personal_care,
    CategoryValidationError,
    ImageValidationError,
    PersonalCareAnalysisResult,
)
from backend.services.personal_care_analysis_service.enrichment import (
    get_personal_care_knowledge_base,
    normalize_lookup_key,
    PersonalCareSemanticFeatures,
)
from backend.services.personal_care_status_service import (
    aggregate_product_dimension,
    map_personal_care_allergy_status,
    map_personal_care_irritation_status,
    map_personal_care_presentation,
    map_personal_care_safety_status,
    STATUS_GREEN,
    STATUS_ORANGE,
    STATUS_RED,
    STATUS_UNAVAILABLE,
    STATUS_YELLOW,
)


def _create_dummy_image_bytes(format="PNG", size=(100, 100), color="white") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestPersonalCareRealImageEndToEnd(unittest.TestCase):
    """Real-image end-to-end testing with actual PaddleOCR execution."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures_dir = Path("tests/fixtures")
        cls.clear_fixture = cls.fixtures_dir / "product_personal_care.png"
        cls.dense_fixture = cls.fixtures_dir / "product_pc_dense.png"
        cls.blank_fixture = cls.fixtures_dir / "product_pc_blank.png"

    def test_real_image_clear_label(self):
        """
        Baseline clear high-resolution real label (Head & Shoulders).
        Verifies OCR extraction, canonical & alias matching, unrecognized ingredient warnings,
        and conservative product-level aggregation across all 3 dimensions.
        """
        if not self.clear_fixture.exists():
            self.skipTest(f"Fixture {self.clear_fixture} not found.")

        with open(self.clear_fixture, "rb") as f:
            img_bytes = f.read()

        res = analyze_personal_care(img_bytes, category="personal_care")
        self.assertTrue(res.success)
        self.assertEqual(res.category, "personal_care")

        # Total ingredients extracted
        total_ing = res.personal_care["total_ingredients"]
        rec_ing = res.personal_care["recognized_ingredients"]
        self.assertGreaterEqual(total_ing, 15)
        self.assertGreaterEqual(rec_ing, 14)

        # Unrecognized ingredients produce warnings
        self.assertTrue(len(res.warnings) >= 2)
        self.assertTrue(any("sodum lauryd sultite" in w for w in res.warnings))
        self.assertTrue(any("sodum aylenesulfonate" in w for w in res.warnings))

        # Presentation status
        pres = res.presentation
        self.assertEqual(pres["personal_care_safety"]["status"], STATUS_ORANGE)
        self.assertEqual(pres["personal_care_safety"]["risk_class"], "Moderate Risk")
        self.assertEqual(pres["allergy"]["status"], STATUS_RED)
        self.assertEqual(pres["allergy"]["risk_class"], "High")
        self.assertEqual(pres["irritation"]["status"], STATUS_ORANGE)
        self.assertEqual(pres["irritation"]["risk_class"], "Medium")

    def test_real_image_dense_multiline_with_aliases(self):
        """
        Dense multi-line label with packaging aliases (Aqua, Tocopherol, Shea Butter).
        Verifies OCR parses multi-line layout and aliases resolve properly.
        """
        if not self.dense_fixture.exists():
            self.skipTest(f"Fixture {self.dense_fixture} not found.")

        with open(self.dense_fixture, "rb") as f:
            img_bytes = f.read()

        res = analyze_personal_care(img_bytes, category="personal_care")
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.personal_care["total_ingredients"], 10)
        self.assertGreaterEqual(res.personal_care["recognized_ingredients"], 10)

        # Confirm dimensions are validly computed
        pres = res.presentation
        self.assertIn(pres["personal_care_safety"]["status"], [STATUS_YELLOW, STATUS_ORANGE, STATUS_RED])
        self.assertIn(pres["allergy"]["status"], [STATUS_YELLOW, STATUS_ORANGE, STATUS_RED])
        self.assertIn(pres["irritation"]["status"], [STATUS_YELLOW, STATUS_ORANGE, STATUS_RED])

    def test_real_image_non_ingredient_blank(self):
        """
        Product label without ingredients (brand and volume only).
        OCR detects no valid ingredients -> pipeline MUST return 'unavailable' for all 3 dimensions.
        Must NEVER default to Safe or No Risk.
        """
        if not self.blank_fixture.exists():
            self.skipTest(f"Fixture {self.blank_fixture} not found.")

        with open(self.blank_fixture, "rb") as f:
            img_bytes = f.read()

        res = analyze_personal_care(img_bytes, category="personal_care")
        self.assertTrue(res.success)
        self.assertEqual(res.personal_care["recognized_ingredients"], 0)

        pres = res.presentation
        self.assertEqual(pres["personal_care_safety"]["status"], STATUS_UNAVAILABLE)
        self.assertEqual(pres["personal_care_safety"]["label"], "Unavailable")
        self.assertEqual(pres["allergy"]["status"], STATUS_UNAVAILABLE)
        self.assertEqual(pres["allergy"]["label"], "Unavailable")
        self.assertEqual(pres["irritation"]["status"], STATUS_UNAVAILABLE)
        self.assertEqual(pres["irritation"]["label"], "Unavailable")


class TestIngredientRecognitionRobustness(unittest.TestCase):
    """Evaluates recognition across canonical names, aliases, OCR distortions, and unknown items."""

    @classmethod
    def setUpClass(cls):
        cls.kb = get_personal_care_knowledge_base()

    def test_canonical_ingredient_recognition(self):
        """Canonical names in dataset must resolve to their canonical record."""
        canonical_samples = [
            "Glycerin",
            "Dimethicone",
            "Citric Acid",
            "Sodium Citrate",
            "Menthol",
            "Sodium Benzoate",
            "Propylene Glycol",
            "Benzyl Alcohol",
        ]
        for name in canonical_samples:
            feat = self.kb.lookup(name)
            self.assertIsNotNone(feat, f"Canonical ingredient '{name}' failed to resolve")
            self.assertEqual(feat.ingredient_name.strip().lower(), name.strip().lower())

    def test_packaging_alias_resolution(self):
        """Packaging aliases in knowledge base must resolve to their canonical record with valid semantic attributes."""
        alias_test_cases = [
            ("Water", "Aqua (Water)"),
            ("Vitamin E", "Tocopherol (Vitamin E)"),
            ("Pro-Vitamin B5", "Panthenol"),
            ("Shea Butter", "Butyrospermum Parkii Butter (Shea Butter)"),
            ("Mineral Oil", "Paraffinum Liquidum (Mineral Oil)"),
            ("Jojoba Oil", "Simmondsia Chinensis Seed Oil (Jojoba Oil)"),
        ]
        for alias, expected_canonical in alias_test_cases:
            feat = self.kb.lookup(alias)
            self.assertIsNotNone(feat, f"Packaging alias '{alias}' failed to resolve")
            self.assertEqual(
                feat.ingredient_name.strip().lower(),
                expected_canonical.strip().lower(),
                f"Alias '{alias}' resolved to '{feat.ingredient_name}', expected '{expected_canonical}'",
            )
            # Verify features are populated
            self.assertTrue(len(feat.primary_function) > 0)
            self.assertTrue(len(feat.ingredient_category) > 0)

    def test_knowledge_base_unrepresented_alias_limitation(self):
        """
        Documents Phase 10D REMEDIATION of previously unrepresented alias limitation:
        Parenthetical INCI entries like 'Aqua (Water)' now index both prefix 'Aqua' and inner 'Water';
        the INCI prefix 'Aqua' alone successfully resolves in the knowledge base.
        """
        feat = self.kb.lookup("Aqua")
        self.assertIsNotNone(feat, "Aqua alone is now successfully resolved via Phase 10D parenthetical alias coverage")
        self.assertEqual(feat.ingredient_name, "Aqua (Water)")

    def test_ocr_spelling_distortion_handling(self):
        """Genuinely distorted OCR text must NOT match erroneously and must NOT be classified as Safe."""
        distorted_tokens = [
            "sodum lauryd sultite",
            "sodum aylenesulfonate",
            "sodum laut suud",
            "tea-dodecybenzenesuonae",
        ]
        for token in distorted_tokens:
            feat = self.kb.lookup(token)
            self.assertIsNone(feat, f"Severely distorted token '{token}' should not match KB directly")

    def test_unknown_ingredients_never_safe(self):
        """Unknown chemical names must produce ingredient_not_recognized and unavailable status."""
        unknowns = [
            "PhonyChemicalX",
            "Unobtainium Silicate",
            "CryptoniteExtract99",
            "FakeSurfactant123",
        ]
        for token in unknowns:
            feat = self.kb.lookup(token)
            self.assertIsNone(feat, f"Unknown ingredient '{token}' must not exist in KB")

        # Test through pipeline with mocked OCR returning unknown
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [{"raw_text": "PhonyChemicalX"}],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")
            self.assertTrue(res.success)
            self.assertEqual(res.personal_care["recognized_ingredients"], 0)
            ing = res.personal_care["ingredients"][0]
            self.assertEqual(ing["status"], "ingredient_not_recognized")
            self.assertEqual(ing["safety"]["status"], "unavailable")
            self.assertEqual(ing["allergy"]["status"], "unavailable")
            self.assertEqual(ing["irritation"]["status"], "unavailable")
            self.assertEqual(res.presentation["personal_care_safety"]["status"], STATUS_UNAVAILABLE)


class TestSemanticEnrichmentValidation(unittest.TestCase):
    """Validates semantic enrichment target isolation and field contract."""

    @classmethod
    def setUpClass(cls):
        cls.kb = get_personal_care_knowledge_base()

    def test_strictly_permitted_input_fields(self):
        """Enrichment container exposes only the 6 permitted input features."""
        feat = self.kb.lookup("Glycerin")
        self.assertIsNotNone(feat)
        model_input = feat.to_model_input_dict()

        expected_fields = {
            "Ingredient_Name",
            "Primary_Function",
            "Ingredient_Category",
            "Product_Categories",
            "Origin",
            "Regulatory_Status",
        }
        self.assertEqual(set(model_input.keys()), expected_fields)

    def test_target_leakage_prohibition(self):
        """Target columns (Safety_Level, Allergy_Risk, Irritation_Risk) must NEVER exist in input features."""
        feat = self.kb.lookup("Aqua (Water)")
        self.assertIsNotNone(feat)

        forbidden_targets = {
            "Safety_Level", "safety_level",
            "Allergy_Risk", "allergy_risk",
            "Irritation_Risk", "irritation_risk",
        }
        for field in forbidden_targets:
            self.assertFalse(hasattr(feat, field))
            self.assertNotIn(field, feat.to_model_input_dict())
            self.assertNotIn(field, feat.to_dict())

    def test_enrichment_unrecognized_returns_none(self):
        """Unrecognized ingredient lookup returns None."""
        self.assertIsNone(self.kb.lookup("NonExistentSubstance999"))
        self.assertIsNone(self.kb.lookup(""))
        self.assertIsNone(self.kb.lookup(None))


class TestModelInferenceRobustnessAndIsolation(unittest.TestCase):
    """Validates inference robustness and failure isolation across 3 independent predictors."""

    def test_independent_target_predictions(self):
        """Predictor produces independent predictions with probability distributions for all 3 targets."""
        kb = get_personal_care_knowledge_base()
        feat = kb.lookup("Parfum (Fragrance)")
        self.assertIsNotNone(feat)
        input_dict = feat.to_model_input_dict()

        pred = predict_personal_care(
            ingredient_name=input_dict["Ingredient_Name"],
            primary_function=input_dict["Primary_Function"],
            ingredient_category=input_dict["Ingredient_Category"],
            product_categories=input_dict["Product_Categories"],
            origin=input_dict["Origin"],
            regulatory_status=input_dict["Regulatory_Status"],
        )

        for target in ["safety", "allergy", "irritation"]:
            self.assertIn(target, pred)
            self.assertEqual(pred[target]["status"], "success")
            self.assertIsNotNone(pred[target]["risk_class"])
            self.assertGreater(pred[target]["confidence"], 0.0)
            self.assertIsInstance(pred[target]["probabilities"], dict)

    def test_isolation_when_safety_predictor_crashes(self):
        """If safety predictor fails, allergy and irritation must still succeed."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr, \
             patch.object(PersonalCarePredictor.get_instance().safety_predictor, "predict", side_effect=RuntimeError("Safety crash")):
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [{"raw_text": "Glycerin", "matched_name": "Glycerin"}],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")
            self.assertTrue(res.success)
            ing = res.personal_care["ingredients"][0]

            self.assertEqual(ing["safety"]["status"], "model_prediction_failure")
            self.assertEqual(ing["allergy"]["status"], "success")
            self.assertEqual(ing["irritation"]["status"], "success")

            # Product presentation reflects isolated safety failure while preserving allergy & irritation
            pres = res.presentation
            self.assertEqual(pres["personal_care_safety"]["status"], STATUS_UNAVAILABLE)
            self.assertNotEqual(pres["allergy"]["status"], STATUS_UNAVAILABLE)
            self.assertNotEqual(pres["irritation"]["status"], STATUS_UNAVAILABLE)

    def test_isolation_when_allergy_predictor_crashes(self):
        """If allergy predictor fails, safety and irritation must still succeed."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr, \
             patch.object(PersonalCarePredictor.get_instance().allergy_predictor, "predict", side_effect=RuntimeError("Allergy crash")):
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [{"raw_text": "Glycerin", "matched_name": "Glycerin"}],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")
            self.assertTrue(res.success)
            ing = res.personal_care["ingredients"][0]

            self.assertEqual(ing["safety"]["status"], "success")
            self.assertEqual(ing["allergy"]["status"], "model_prediction_failure")
            self.assertEqual(ing["irritation"]["status"], "success")

            pres = res.presentation
            self.assertNotEqual(pres["personal_care_safety"]["status"], STATUS_UNAVAILABLE)
            self.assertEqual(pres["allergy"]["status"], STATUS_UNAVAILABLE)
            self.assertNotEqual(pres["irritation"]["status"], STATUS_UNAVAILABLE)

    def test_isolation_when_irritation_predictor_crashes(self):
        """If irritation predictor fails, safety and allergy must still succeed."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr, \
             patch.object(PersonalCarePredictor.get_instance().irritation_predictor, "predict", side_effect=RuntimeError("Irritation crash")):
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [{"raw_text": "Glycerin", "matched_name": "Glycerin"}],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")
            self.assertTrue(res.success)
            ing = res.personal_care["ingredients"][0]

            self.assertEqual(ing["safety"]["status"], "success")
            self.assertEqual(ing["allergy"]["status"], "success")
            self.assertEqual(ing["irritation"]["status"], "model_prediction_failure")

            pres = res.presentation
            self.assertNotEqual(pres["personal_care_safety"]["status"], STATUS_UNAVAILABLE)
            self.assertNotEqual(pres["allergy"]["status"], STATUS_UNAVAILABLE)
            self.assertEqual(pres["irritation"]["status"], STATUS_UNAVAILABLE)


class TestProductLevelAggregation(unittest.TestCase):
    """Validates deterministic worst-case product aggregation across independent dimensions."""

    def test_safety_worst_case_aggregation(self):
        """Product safety risk equals the highest valid predicted risk class."""
        # Safe + Moderate Risk + High Risk -> High Risk
        ing_list = [
            {"status": "success", "safety": {"status": "success", "risk_class": "Safe"}},
            {"status": "success", "safety": {"status": "success", "risk_class": "Moderate Risk"}},
            {"status": "success", "safety": {"status": "success", "risk_class": "High Risk"}},
        ]
        worst, status = aggregate_product_dimension(ing_list, "safety")
        self.assertEqual(worst, "High Risk")
        self.assertEqual(status, "success")

        # Very Safe + Safe -> Safe
        ing_list2 = [
            {"status": "success", "safety": {"status": "success", "risk_class": "Very Safe"}},
            {"status": "success", "safety": {"status": "success", "risk_class": "Safe"}},
        ]
        worst2, status2 = aggregate_product_dimension(ing_list2, "safety")
        self.assertEqual(worst2, "Safe")
        self.assertEqual(status2, "success")

    def test_allergy_worst_case_aggregation(self):
        """Product allergy risk equals highest valid risk (No Risk < Low < Medium < High)."""
        ing_list = [
            {"status": "success", "allergy": {"status": "success", "risk_class": "Low"}},
            {"status": "success", "allergy": {"status": "success", "risk_class": "Medium"}},
            {"status": "success", "allergy": {"status": "success", "risk_class": "High"}},
        ]
        worst, status = aggregate_product_dimension(ing_list, "allergy")
        self.assertEqual(worst, "High")
        self.assertEqual(status, "success")

        # No Risk + Low -> Low
        ing_list2 = [
            {"status": "success", "allergy": {"status": "success", "risk_class": "No Risk"}},
            {"status": "success", "allergy": {"status": "success", "risk_class": "Low"}},
        ]
        worst2, status2 = aggregate_product_dimension(ing_list2, "allergy")
        self.assertEqual(worst2, "Low")
        self.assertEqual(status2, "success")

    def test_irritation_worst_case_aggregation(self):
        """Product irritation risk equals highest valid risk (No Risk < Low < Medium < High)."""
        ing_list = [
            {"status": "success", "irritation": {"status": "success", "risk_class": "No Risk"}},
            {"status": "success", "irritation": {"status": "success", "risk_class": "Medium"}},
        ]
        worst, status = aggregate_product_dimension(ing_list, "irritation")
        self.assertEqual(worst, "Medium")
        self.assertEqual(status, "success")

    def test_mixed_known_and_unknown_ingredients(self):
        """Known ingredients compute dimension status; unknown ingredients emit warnings without clearing results."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [
                    {"raw_text": "Glycerin", "matched_name": "Glycerin"},
                    {"raw_text": "UnknownSubstance123"},
                ],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")
            self.assertTrue(res.success)
            self.assertEqual(res.personal_care["total_ingredients"], 2)
            self.assertEqual(res.personal_care["recognized_ingredients"], 1)

            # Valid recognized prediction drives the dimension
            self.assertNotEqual(res.presentation["personal_care_safety"]["status"], STATUS_UNAVAILABLE)
            # Warning is emitted for the unknown item
            self.assertTrue(any("UnknownSubstance123" in w for w in res.warnings))

    def test_all_unknown_ingredients_yields_unavailable(self):
        """If 100% of ingredients are unknown, dimension status is strictly 'unavailable'."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [
                    {"raw_text": "Unknown1"},
                    {"raw_text": "Unknown2"},
                ],
            }
            res = analyze_personal_care(dummy_bytes, category="personal_care")
            self.assertTrue(res.success)
            self.assertEqual(res.personal_care["recognized_ingredients"], 0)
            self.assertEqual(res.presentation["personal_care_safety"]["status"], STATUS_UNAVAILABLE)
            self.assertEqual(res.presentation["allergy"]["status"], STATUS_UNAVAILABLE)
            self.assertEqual(res.presentation["irritation"]["status"], STATUS_UNAVAILABLE)


class TestStatusMappingDeterminism(unittest.TestCase):
    """Validates deterministic color and label mapping according to Phase 10B constants."""

    def test_safety_status_mapping(self):
        """Very Safe -> green, Safe -> yellow (strictly yellow), Moderate Risk -> orange, High Risk -> red."""
        self.assertEqual(map_personal_care_safety_status("Very Safe").color, STATUS_GREEN)
        self.assertEqual(map_personal_care_safety_status("Safe").color, STATUS_YELLOW)
        self.assertEqual(map_personal_care_safety_status("Moderate Risk").color, STATUS_ORANGE)
        self.assertEqual(map_personal_care_safety_status("High Risk").color, STATUS_RED)
        self.assertEqual(map_personal_care_safety_status(None).color, STATUS_UNAVAILABLE)

    def test_allergy_status_mapping(self):
        """No Risk -> green, Low -> yellow, Medium -> orange, High -> red."""
        self.assertEqual(map_personal_care_allergy_status("No Risk").color, STATUS_GREEN)
        self.assertEqual(map_personal_care_allergy_status("Low").color, STATUS_YELLOW)
        self.assertEqual(map_personal_care_allergy_status("Medium").color, STATUS_ORANGE)
        self.assertEqual(map_personal_care_allergy_status("High").color, STATUS_RED)
        self.assertEqual(map_personal_care_allergy_status(None).color, STATUS_UNAVAILABLE)

    def test_irritation_status_mapping(self):
        """No Risk -> green, Low -> yellow, Medium -> orange, High -> red."""
        self.assertEqual(map_personal_care_irritation_status("No Risk").color, STATUS_GREEN)
        self.assertEqual(map_personal_care_irritation_status("Low").color, STATUS_YELLOW)
        self.assertEqual(map_personal_care_irritation_status("Medium").color, STATUS_ORANGE)
        self.assertEqual(map_personal_care_irritation_status("High").color, STATUS_RED)
        self.assertEqual(map_personal_care_irritation_status(None).color, STATUS_UNAVAILABLE)


class TestCategoryRoutingAndAPIContract(unittest.TestCase):
    """Validates strict explicit routing and API contract."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_personal_care_endpoint_rejects_food(self):
        """POST /api/personal-care/analyze with category='food' returns HTTP 400."""
        dummy_bytes = _create_dummy_image_bytes()
        resp = self.client.post(
            "/api/personal-care/analyze",
            data={"image": (io.BytesIO(dummy_bytes), "test.png"), "category": "food"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("category", data["error"].lower())

    def test_food_endpoint_rejects_personal_care(self):
        """POST /api/food/analyze with category='personal_care' returns HTTP 400."""
        dummy_bytes = _create_dummy_image_bytes()
        resp = self.client.post(
            "/api/food/analyze",
            data={"image": (io.BytesIO(dummy_bytes), "test.png"), "category": "personal_care"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("category", data["error"].lower())

    def test_personal_care_api_response_contract(self):
        """POST /api/personal-care/analyze returns complete contract with 3 independent dimensions."""
        dummy_bytes = _create_dummy_image_bytes()
        with patch("backend.services.personal_care_analysis_service.analyzer.run_ocr") as mock_ocr:
            mock_ocr.return_value = {
                "domain": "personal_care",
                "ingredients": [{"raw_text": "Glycerin", "matched_name": "Glycerin"}],
            }
            resp = self.client.post(
                "/api/personal-care/analyze",
                data={"image": (io.BytesIO(dummy_bytes), "test.png"), "category": "personal_care"},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(data["category"], "personal_care")
            self.assertIn("personal_care", data)
            self.assertIn("presentation", data)
            self.assertIn("warnings", data)
            self.assertIn("errors", data)

            pres = data["presentation"]
            for dim in ["personal_care_safety", "allergy", "irritation"]:
                self.assertIn(dim, pres)
                self.assertIn("status", pres[dim])
                self.assertIn("label", pres[dim])
                self.assertIn("color", pres[dim])


class TestFrontendContractCompliance(unittest.TestCase):
    """Verifies that static/app.js complies strictly with independent 3-dimension contract."""

    def test_frontend_does_not_calculate_composite_score(self):
        """Ensures app.js does not calculate a composite score or overall color for personal care."""
        app_js_path = Path("static/app.js")
        self.assertTrue(app_js_path.exists())

        with open(app_js_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that personal care rendering references all 3 independent cards
        self.assertIn("renderPersonalCareDimensionCard", content)
        self.assertIn("personal_care_safety", content)
        self.assertIn("allergy", content)
        self.assertIn("irritation", content)

        # Check that no composite health score is computed for personal care
        self.assertNotIn("calculatePersonalCareHealthScore", content)
        self.assertNotIn("personalCareCompositeScore", content)


if __name__ == "__main__":
    unittest.main()
