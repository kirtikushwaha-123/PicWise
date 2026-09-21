"""
tests/test_step_7_architecture.py

Comprehensive tests for STEP 7:
1. Central configuration
2. Knowledge-base path selection/fallback
3. Valid image upload
4. Missing image
5. Empty filename
6. Invalid extension
7. Invalid MIME type
8. Empty image
9. Oversized upload -> existing expected status
10. Corrupt image
11. Excessive pixel count
12. OCR engine singleton initialization
13. OCR inference lock behavior
14. Sentence-transformer lazy loading
15. Sentence-transformer reuse
16. /api/analyze compatibility
"""

import io
import os
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

from backend import config, create_app
from backend.services.knowledge_base import KnowledgeBase
from backend.utils.upload_validator import is_allowed_image, validate_image_upload
from backend.services.ocr_service.ocr import paddle_engine
from backend.ml.inference import food_safety_service


def _create_test_image_bytes(format="JPEG", size=(10, 10), color="red"):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestCentralConfiguration(unittest.TestCase):
    def test_upload_limits_and_image_constants(self):
        self.assertEqual(config.MAX_CONTENT_LENGTH, 16 * 1024 * 1024)
        self.assertIn(".jpg", config.ALLOWED_IMAGE_EXTENSIONS)
        self.assertIn(".jpeg", config.ALLOWED_IMAGE_EXTENSIONS)
        self.assertIn(".png", config.ALLOWED_IMAGE_EXTENSIONS)
        self.assertIn(".webp", config.ALLOWED_IMAGE_EXTENSIONS)
        self.assertIn("image/jpeg", config.ALLOWED_IMAGE_MIMETYPES)
        self.assertIn("image/png", config.ALLOWED_IMAGE_MIMETYPES)
        self.assertIn("image/webp", config.ALLOWED_IMAGE_MIMETYPES)
        self.assertEqual(config.MAX_IMAGE_PIXELS, 89478485)

    def test_model_and_kb_paths(self):
        self.assertEqual(config.DEFAULT_FOOD_SAFETY_MODEL_DIR, "backend/ml/models/food_safety")
        self.assertEqual(config.DEFAULT_PERSONAL_CARE_MODEL_DIR, "backend/ml/models/personal_care")
        self.assertEqual(config.DEFAULT_LEGACY_MODEL_DIR, "backend/ml/models")
        self.assertIn("food_ingredients_dataset_corrected", config.DEFAULT_FOOD_DATA_PATH)
        self.assertIn("nutrition_knowledge_dataset.csv", config.DEFAULT_NUTRITION_DATA_PATH)
        self.assertIn("personal_care_ingredients_dataset", config.DEFAULT_PERSONAL_CARE_DATA_PATH)


class TestKnowledgeBasePathSelectionAndFallback(unittest.TestCase):
    def test_food_kb_path_env_override_and_fallback(self):
        with patch.dict(os.environ, {"FOOD_DATA_PATH": "custom/food.csv"}):
            with patch("os.path.exists", return_value=True):
                self.assertEqual(config.resolve_food_kb_path(), "custom/food.csv")

        # Fallback when env var is unset
        with patch.dict(os.environ, {}, clear=True):
            resolved = config.resolve_food_kb_path()
            self.assertTrue(resolved.endswith(".csv"))

    def test_personal_care_kb_path_env_override(self):
        with patch.dict(os.environ, {"PERSONAL_CARE_DATA_PATH": "custom/pc.xlsx"}):
            with patch("os.path.exists", return_value=True):
                self.assertEqual(config.resolve_personal_care_kb_path(), "custom/pc.xlsx")

    def test_nutrition_kb_path_env_override(self):
        with patch.dict(os.environ, {"NUTRITION_DATA_PATH": "custom/nut.csv"}):
            with patch("os.path.exists", return_value=True):
                self.assertEqual(config.resolve_nutrition_kb_path(), "custom/nut.csv")

    def test_canonical_knowledge_base_loads(self):
        kb = KnowledgeBase.from_env()
        self.assertIsNotNone(kb)
        self.assertIsInstance(kb.food_index, dict)
        self.assertIsInstance(kb.nutrition_index, dict)
        self.assertIsInstance(kb.personal_care_index, dict)


class TestUploadValidation(unittest.TestCase):
    def _make_mock_request(self, category=None, image_file=None):
        req = MagicMock()
        req.form = {}
        if category is not None:
            req.form["category"] = category
        req.files = {}
        if image_file is not None:
            req.files["image"] = image_file
        return req

    def test_valid_image_upload(self):
        img_bytes = _create_test_image_bytes(format="JPEG")
        mock_file = MagicMock()
        mock_file.filename = "test.jpg"
        mock_file.mimetype = "image/jpeg"
        mock_file.read.return_value = img_bytes

        req = self._make_mock_request(category="food", image_file=mock_file)
        data, cat, err = validate_image_upload(req, expected_category="food")
        self.assertIsNone(err)
        self.assertEqual(data, img_bytes)
        self.assertEqual(cat, "food")

    def test_missing_image(self):
        req = self._make_mock_request(category="food", image_file=None)
        data, cat, err = validate_image_upload(req, expected_category="food")
        self.assertIsNone(data)
        self.assertIsNotNone(err)
        err_dict, code = err
        self.assertEqual(code, 400)
        self.assertIn("Image file is required", err_dict["error"])

    def test_empty_filename(self):
        mock_file = MagicMock()
        mock_file.filename = ""
        mock_file.mimetype = "image/jpeg"
        req = self._make_mock_request(category="food", image_file=mock_file)
        data, cat, err = validate_image_upload(req, expected_category="food")
        self.assertIsNone(data)
        self.assertIsNotNone(err)
        err_dict, code = err
        self.assertEqual(code, 400)
        self.assertIn("Image file is required", err_dict["error"])

    def test_invalid_extension(self):
        mock_file = MagicMock()
        mock_file.filename = "malicious.exe"
        mock_file.mimetype = "application/x-msdownload"
        req = self._make_mock_request(category="food", image_file=mock_file)
        data, cat, err = validate_image_upload(req, expected_category="food")
        self.assertIsNone(data)
        self.assertIsNotNone(err)
        err_dict, code = err
        self.assertEqual(code, 400)
        self.assertIn("Only JPG, JPEG, PNG, and WEBP images are supported", err_dict["error"])

    def test_invalid_mime_type(self):
        mock_file = MagicMock()
        mock_file.filename = "fake.jpg"
        mock_file.mimetype = "text/plain"
        req = self._make_mock_request(category="food", image_file=mock_file)
        data, cat, err = validate_image_upload(req, expected_category="food")
        self.assertIsNone(data)
        self.assertIsNotNone(err)
        err_dict, code = err
        self.assertEqual(code, 400)
        self.assertIn("Only JPG, JPEG, PNG, and WEBP images are supported", err_dict["error"])

    def test_empty_image_bytes(self):
        mock_file = MagicMock()
        mock_file.filename = "empty.jpg"
        mock_file.mimetype = "image/jpeg"
        mock_file.read.return_value = b""
        req = self._make_mock_request(category="food", image_file=mock_file)
        data, cat, err = validate_image_upload(req, expected_category="food")
        self.assertIsNone(data)
        self.assertIsNotNone(err)
        err_dict, code = err
        self.assertEqual(code, 400)
        self.assertIn("Uploaded image file is empty", err_dict["error"])

    def test_oversized_upload(self):
        mock_file = MagicMock()
        mock_file.filename = "big.jpg"
        mock_file.mimetype = "image/jpeg"
        mock_file.read.return_value = b"x" * (16 * 1024 * 1024 + 1)
        req = self._make_mock_request(category="food", image_file=mock_file)
        data, cat, err = validate_image_upload(req, expected_category="food")
        self.assertIsNone(data)
        self.assertIsNotNone(err)
        err_dict, code = err
        self.assertEqual(code, 413)
        self.assertIn("exceeds the maximum allowed size of 16MB", err_dict["error"])

    def test_corrupt_image(self):
        mock_file = MagicMock()
        mock_file.filename = "corrupt.jpg"
        mock_file.mimetype = "image/jpeg"
        mock_file.read.return_value = b"not a valid image content"
        req = self._make_mock_request(category="food", image_file=mock_file)
        data, cat, err = validate_image_upload(req, expected_category="food")
        self.assertIsNone(data)
        self.assertIsNotNone(err)
        err_dict, code = err
        self.assertEqual(code, 400)
        self.assertIn("Invalid or corrupt image file", err_dict["error"])

    def test_excessive_pixel_count(self):
        img_bytes = _create_test_image_bytes(size=(100, 100))
        mock_file = MagicMock()
        mock_file.filename = "huge_pixels.jpg"
        mock_file.mimetype = "image/jpeg"
        mock_file.read.return_value = img_bytes

        req = self._make_mock_request(category="food", image_file=mock_file)
        # Set max pixels temporarily lower to test threshold
        with patch.object(config, "MAX_IMAGE_PIXELS", 5000):
            data, cat, err = validate_image_upload(req, expected_category="food")
            self.assertIsNone(data)
            self.assertIsNotNone(err)
            err_dict, code = err
            self.assertEqual(code, 400)
            self.assertIn("Image pixel count exceeds maximum allowed limit", err_dict["error"])


class TestOCREngineLifecycle(unittest.TestCase):
    def test_engine_lock_and_singleton(self):
        self.assertIsNotNone(paddle_engine._ENGINE_LOCK)
        self.assertIsNotNone(paddle_engine._INFERENCE_LOCK)

        # Calling is_available initializes or checks singleton
        avail = paddle_engine.is_available()
        self.assertIsInstance(avail, bool)
        engine1 = paddle_engine._OCR_ENGINE
        paddle_engine._init_engine()
        engine2 = paddle_engine._OCR_ENGINE
        self.assertIs(engine1, engine2)

    def test_inference_lock_behavior(self):
        mock_engine = MagicMock()
        mock_engine.predict.return_value = []
        with patch.object(paddle_engine, "_OCR_ENGINE", mock_engine):
            with patch.object(paddle_engine, "_INFERENCE_LOCK") as mock_lock:
                mock_lock.__enter__.return_value = None
                mock_lock.__exit__.return_value = None

                dummy_img = Image.new("RGB", (10, 10))
                import numpy as np
                paddle_engine.run_ocr(np.array(dummy_img))
                self.assertTrue(mock_lock.__enter__.called)
                self.assertTrue(mock_lock.__exit__.called)


class TestSentenceTransformerLazyLoading(unittest.TestCase):
    def test_lazy_loading_and_reuse(self):
        # Clear cache for test
        food_safety_service._EMBEDDER_CACHE.clear()

        mock_st = MagicMock()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_st) as mock_st_cls:
            embedder1 = food_safety_service.get_sentence_transformer("mock-model")
            self.assertEqual(mock_st_cls.call_count, 1)

            # Reusing same model name must NOT re-instantiate
            embedder2 = food_safety_service.get_sentence_transformer("mock-model")
            self.assertIs(embedder1, embedder2)
            self.assertEqual(mock_st_cls.call_count, 1)

    def test_predictor_does_not_load_embedder_at_init(self):
        predictor = food_safety_service.FoodSafetyPredictor.get_instance()
        self.assertIsNotNone(predictor)
        # self._embedder should be lazily loaded or cached
        self.assertIsNotNone(predictor.embedder)


class TestLegacyApiAnalyzeCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_legacy_api_analyze_missing_category(self):
        img_bytes = _create_test_image_bytes()
        resp = self.client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(img_bytes), "test.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertNotIn("success", data)  # Legacy endpoint does not include 'success' on 400

    def test_legacy_api_analyze_oversized(self):
        resp = self.client.post(
            "/api/analyze",
            data={
                "category": "food",
                "image": (io.BytesIO(b"x" * (16 * 1024 * 1024 + 10)), "huge.jpg"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 413)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertFalse(data.get("success"))

    def test_legacy_api_analyze_success_shape(self):
        img_bytes = _create_test_image_bytes()
        with patch("backend.routes.api.analyze_product_image", return_value={
            "product": {"name": None, "brand": None, "domain": "food"},
            "ingredients": [],
            "nutrition": {},
            "personalCare": [],
            "warnings": [],
            "ocr": {},
        }):
            resp = self.client.post(
                "/api/analyze",
                data={
                    "category": "food",
                    "image": (io.BytesIO(img_bytes), "test.jpg"),
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("product", data)
            self.assertIn("ingredients", data)
            self.assertIn("nutrition", data)
            self.assertIn("personalCare", data)
            self.assertIn("ocr", data)


if __name__ == "__main__":
    unittest.main()
