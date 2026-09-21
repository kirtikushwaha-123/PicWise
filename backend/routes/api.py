import io
from flask import Blueprint, current_app, jsonify, request
from PIL import Image

from backend.services.analysis_service import analyze_product_image
from backend.services.food_analysis_service import analyze_food
from backend.services.personal_care_analysis_service import analyze_personal_care

api_bp = Blueprint("api", __name__, url_prefix="/api")


MAX_IMAGE_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB ceiling


@api_bp.post("/food/analyze")
def analyze_food_endpoint():
    category = request.form.get("category")
    if not category or not category.strip():
        return jsonify({"error": "Product category is required in form field 'category'.", "success": False}), 400

    category = category.strip().lower()
    if category != "food":
        return jsonify({
            "error": f"Invalid category '{category}'. This endpoint strictly handles 'food' analysis.",
            "success": False,
        }), 400

    image = request.files.get("image")
    if image is None or image.filename == "":
        return jsonify({"error": "Image file is required in form field 'image'.", "success": False}), 400

    if not _is_allowed_image(image.mimetype, image.filename):
        return jsonify({"error": "Only JPG, JPEG, PNG, and WEBP images are supported.", "success": False}), 400

    image_bytes = image.read()
    if not image_bytes or len(image_bytes) == 0:
        return jsonify({"error": "Uploaded image file is empty.", "success": False}), 400

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        return jsonify({
            "error": "Uploaded image file exceeds the maximum allowed size of 16MB.",
            "success": False,
        }), 413

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
    except Exception:
        return jsonify({"error": "Invalid or corrupt image file.", "success": False}), 400

    try:
        knowledge_base = current_app.config.get("KNOWLEDGE_BASE")
        result = analyze_food(image_bytes, category=category, knowledge_base=knowledge_base)
        return jsonify(result.to_dict()), 200
    except Exception as exc:
        current_app.logger.error(f"Unexpected error in /api/food/analyze: {exc}", exc_info=True)
        return jsonify({
            "error": "An unexpected server error occurred. Please try again.",
            "success": False,
        }), 500


@api_bp.post("/personal-care/analyze")
def analyze_personal_care_endpoint():
    category = request.form.get("category")
    if not category or not category.strip():
        return jsonify({"error": "Product category is required in form field 'category'.", "success": False}), 400

    category = category.strip().lower()
    if category != "personal_care":
        return jsonify({
            "error": f"Invalid category '{category}'. This endpoint strictly handles 'personal_care' analysis.",
            "success": False,
        }), 400

    image = request.files.get("image")
    if image is None or image.filename == "":
        return jsonify({"error": "Image file is required in form field 'image'.", "success": False}), 400

    if not _is_allowed_image(image.mimetype, image.filename):
        return jsonify({"error": "Only JPG, JPEG, PNG, and WEBP images are supported.", "success": False}), 400

    image_bytes = image.read()
    if not image_bytes or len(image_bytes) == 0:
        return jsonify({"error": "Uploaded image file is empty.", "success": False}), 400

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        return jsonify({
            "error": "Uploaded image file exceeds the maximum allowed size of 16MB.",
            "success": False,
        }), 413

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
    except Exception:
        return jsonify({"error": "Invalid or corrupt image file.", "success": False}), 400

    try:
        knowledge_base = current_app.config.get("KNOWLEDGE_BASE")
        result = analyze_personal_care(image_bytes, category=category, knowledge_base=knowledge_base)
        return jsonify(result.to_dict()), 200
    except Exception as exc:
        current_app.logger.error(f"Unexpected error in /api/personal-care/analyze: {exc}", exc_info=True)
        return jsonify({
            "error": "An unexpected server error occurred. Please try again.",
            "success": False,
        }), 500


@api_bp.post("/analyze")
def analyze():
    category = request.form.get("category")
    if not category or not category.strip():
        return jsonify({"error": "Product category is required in form field 'category'."}), 400

    category = category.strip().lower()
    if category not in ("food", "personal_care"):
        return jsonify({
            "error": f"Invalid category '{category}'. Allowed values are 'food' or 'personal_care'."
        }), 400

    image = request.files.get("image")
    if image is None or image.filename == "":
        return jsonify({"error": "Image file is required in form field 'image'."}), 400

    if not _is_allowed_image(image.mimetype, image.filename):
        return jsonify({"error": "Only JPG, JPEG, PNG, and WEBP images are supported."}), 400

    image_bytes = image.read()
    if not image_bytes or len(image_bytes) == 0:
        return jsonify({"error": "Uploaded image file is empty."}), 400

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        return jsonify({
            "error": "Uploaded image file exceeds the maximum allowed size of 16MB.",
            "success": False,
        }), 413

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
    except Exception:
        return jsonify({"error": "Invalid or corrupt image file."}), 400

    try:
        knowledge_base = current_app.config.get("KNOWLEDGE_BASE")
        result = analyze_product_image(image_bytes, knowledge_base, category=category)
        return jsonify(result), 200
    except Exception as exc:
        current_app.logger.error(f"Unexpected error in /api/analyze: {exc}", exc_info=True)
        return jsonify({
            "error": "An unexpected server error occurred. Please try again.",
            "success": False,
        }), 500


def _is_allowed_image(mimetype, filename):
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    allowed_mimetypes = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
    lower_name = filename.lower()
    return (
        any(lower_name.endswith(ext) for ext in allowed_extensions)
        and mimetype in allowed_mimetypes
    )

