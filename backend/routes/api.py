from flask import Blueprint, current_app, jsonify, request

from backend.services.analysis_service import analyze_product_image
from backend.services.food_analysis_service import analyze_food
from backend.services.personal_care_analysis_service import analyze_personal_care
from backend.services.comparison_service import (
    compare_food_products,
    validate_comparison_input,
)
from backend.services.ocr_service import run_ocr
from backend import config
from backend.utils.upload_validator import validate_image_upload

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Backward compatibility alias
MAX_IMAGE_SIZE_BYTES = config.MAX_CONTENT_LENGTH


@api_bp.post("/food/ocr")
def food_ocr_endpoint():
    """
    POST /api/food/ocr
    Uploads image, runs conservative image quality check, then runs OCR and returns
    structured raw text sections for user review.
    """
    image_bytes, category, err = validate_image_upload(
        request, expected_category="food", include_success_flag=True
    )
    if err is not None:
        err_dict, status_code = err
        return jsonify(err_dict), status_code

    try:
        from backend.services.ocr_service.preprocessing.image_utils import (
            load_image_from_bytes,
            assess_image_quality_gate,
        )
        img = load_image_from_bytes(image_bytes)
        is_usable, status, q_msg, q_details = assess_image_quality_gate(img)
        if not is_usable:
            return jsonify({
                "success": False,
                "status": "unusable_image",
                "error": q_msg,
                "quality": q_details,
            }), 400

        knowledge_base = current_app.config.get("KNOWLEDGE_BASE")
        ocr_kb = knowledge_base if (knowledge_base and hasattr(knowledge_base, "get_ingredient_names")) else None
        ocr_output = run_ocr(image_bytes, category="food", kb=ocr_kb)

        raw_text = ocr_output.get("raw_text", {})
        return jsonify({
            "success": True,
            "status": "ready_for_review",
            "raw_text": {
                "ingredients_text": raw_text.get("ingredients_text", ""),
                "nutrition_text": raw_text.get("nutrition_text", ""),
                "other_text": raw_text.get("other_text", ""),
                "all_text": raw_text.get("all_text", ""),
            },
            "ocr": ocr_output,
        }), 200
    except Exception as exc:
        current_app.logger.error(f"Unexpected error in /api/food/ocr: {exc}", exc_info=True)
        return jsonify({
            "error": "An unexpected server error occurred during OCR extraction. Please try again.",
            "success": False,
        }), 500


@api_bp.post("/food/analyze-text")
def analyze_food_text_endpoint():
    """
    POST /api/food/analyze-text
    Accepts user-reviewed / edited OCR text and runs Food Safety, Allergy, and Nutrition analysis.
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON.", "success": False}), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object.", "success": False}), 400

    category = data.get("category", "food")
    if not category or not isinstance(category, str) or category.strip().lower() != "food":
        return jsonify({"error": "Category is required and must be 'food'.", "success": False}), 400

    ingredients_text = str(data.get("ingredients_text") or "")
    nutrition_text = str(data.get("nutrition_text") or "")
    all_text = str(data.get("all_text") or "")

    try:
        from backend.services.food_analysis_service import analyze_food_from_text
        knowledge_base = current_app.config.get("KNOWLEDGE_BASE")
        result = analyze_food_from_text(
            ingredients_text=ingredients_text,
            nutrition_text=nutrition_text,
            all_text=all_text,
            category="food",
            knowledge_base=knowledge_base,
        )
        return jsonify(result.to_dict()), 200
    except Exception as exc:
        current_app.logger.error(f"Unexpected error in /api/food/analyze-text: {exc}", exc_info=True)
        return jsonify({
            "error": "An unexpected server error occurred during text analysis. Please try again.",
            "success": False,
        }), 500


@api_bp.post("/food/analyze")
def analyze_food_endpoint():
    image_bytes, category, err = validate_image_upload(
        request, expected_category="food", include_success_flag=True
    )
    if err is not None:
        err_dict, status_code = err
        return jsonify(err_dict), status_code

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
    image_bytes, category, err = validate_image_upload(
        request, expected_category="personal_care", include_success_flag=True
    )
    if err is not None:
        err_dict, status_code = err
        return jsonify(err_dict), status_code

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
    image_bytes, category, err = validate_image_upload(
        request, allowed_categories=("food", "personal_care"), include_success_flag=False
    )
    if err is not None:
        err_dict, status_code = err
        return jsonify(err_dict), status_code

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


@api_bp.post("/compare")
def compare_endpoint():
    """
    POST /api/compare
    Compares 2–4 already analyzed food products across Food Safety, Allergy Risk, and Nutrition.
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON.", "success": False}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON.", "success": False}), 400

    is_valid, err_msg, products_list = validate_comparison_input(data)
    if not is_valid:
        return jsonify({"error": err_msg, "success": False}), 400

    try:
        result = compare_food_products(products_list)
        return jsonify(result), 200
    except Exception as exc:
        current_app.logger.error(f"Unexpected error in /api/compare: {exc}", exc_info=True)
        return jsonify({
            "error": "An unexpected server error occurred during product comparison.",
            "success": False,
        }), 500

