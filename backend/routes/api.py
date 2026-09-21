from flask import Blueprint, current_app, jsonify, request

from backend.services.analysis_service import analyze_product_image
from backend.services.food_analysis_service import analyze_food
from backend.services.personal_care_analysis_service import analyze_personal_care
from backend.services.comparison_service import (
    compare_food_products,
    validate_comparison_input,
)
from backend import config
from backend.utils.upload_validator import validate_image_upload

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Backward compatibility alias
MAX_IMAGE_SIZE_BYTES = config.MAX_CONTENT_LENGTH


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

