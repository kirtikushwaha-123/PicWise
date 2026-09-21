"""
backend/services/personal_care_analysis_service/analyzer.py

PicWise Unified Personal Care Analysis Orchestration Service.
Coordinates OCR, target-isolated runtime semantic enrichment, frozen Logistic Regression
inference (Safety, Allergy, Irritation), conservative product-level aggregation,
and deterministic presentation status mapping.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.ml.inference.personal_care_service import predict_personal_care
from backend.services.ocr_service import run_ocr
from backend.services.personal_care_analysis_service.enrichment import (
    PersonalCareSemanticFeatures,
    get_personal_care_knowledge_base,
)
from backend.services.personal_care_analysis_service.errors import (
    ImageProcessingError,
    InvalidCategoryError,
    OCRError,
)
from backend.services.personal_care_analysis_service.models import (
    PersonalCareAnalysisResult,
    PersonalCareIngredientAnalysis,
)
from backend.services.personal_care_status_service import (
    aggregate_product_dimension,
    map_personal_care_allergy_status,
    map_personal_care_irritation_status,
    map_personal_care_presentation,
    map_personal_care_safety_status,
)

logger = logging.getLogger(__name__)


def analyze_personal_care(
    image_bytes: bytes,
    category: str = "personal_care",
    knowledge_base: Optional[Any] = None,
) -> PersonalCareAnalysisResult:
    """
    Orchestrates end-to-end Personal Care product analysis.

    Parameters:
        image_bytes: Raw image file bytes.
        category: Must explicitly be 'personal_care'.
        knowledge_base: Optional KnowledgeBase for OCR fuzzy matching.

    Returns:
        PersonalCareAnalysisResult: Unified structured result.
    """
    # 1. Category Validation (strictly 'personal_care', no heuristic guessing)
    if not category or not isinstance(category, str):
        raise InvalidCategoryError("Category is required and must be 'personal_care'.")

    category_norm = category.strip().lower()
    if category_norm != "personal_care":
        raise InvalidCategoryError(
            f"Invalid category '{category}'. The Personal Care service strictly handles 'personal_care'."
        )

    # 2. Image Bytes Validation
    if not image_bytes or len(image_bytes) == 0:
        raise ImageProcessingError("Image bytes are empty or missing.")

    # 3. Shared OCR Pipeline Execution
    try:
        ocr_kb = knowledge_base if (knowledge_base and hasattr(knowledge_base, "get_ingredient_names")) else None
        ocr_output = run_ocr(image_bytes, category="personal_care", kb=ocr_kb)
    except Exception as exc:
        err_msg = f"OCR processing failed: {str(exc)}"
        logger.error(err_msg, exc_info=True)
        # On OCR failure, stop downstream processing; presentation is completely unavailable
        unav_pres = map_personal_care_presentation(
            safety_status="unavailable",
            allergy_status="unavailable",
            irritation_status="unavailable",
        ).to_dict()
        return PersonalCareAnalysisResult(
            category="personal_care",
            success=False,
            ocr=None,
            personal_care={},
            errors=[err_msg],
            warnings=[],
            presentation=unav_pres,
        )

    all_warnings: List[str] = []
    raw_ingredients = ocr_output.get("ingredients", [])

    # Phase 10D / Phase 14: Conservative Image Quality Advisory
    img_quality = ocr_output.get("image_quality") if ocr_output else None
    raw_text_dict = ocr_output.get("raw_text", {}) if ocr_output else {}
    all_text = raw_text_dict.get("all_text", "").strip() if raw_text_dict else ""
    ing_region = (ocr_output.get("ingredients_region") or {}) if ocr_output else {}
    line_count = ing_region.get("line_count", 0) if ing_region else 0
    anchor = ing_region.get("anchor") if ing_region else None

    lap_var = img_quality.get("laplacian_variance") if img_quality else None
    if lap_var is None and img_quality:
        lap_var = img_quality.get("laplacian_var")

    is_degraded = (
        (img_quality and (img_quality.get("is_blurry") or img_quality.get("is_too_dark")))
        or (len(all_text) > 0 and lap_var is not None and lap_var < 150.0)
        or (line_count <= 1 and anchor is None and len(all_text) > 0 and not raw_ingredients)
        or (img_quality and img_quality.get("contrast_std", 100.0) < 25.0 and not raw_ingredients)
    )

    ocr_quality_advisory = None
    if is_degraded:
        ocr_quality_advisory = "OCR quality may be unreliable: image appears degraded or blurred. Please capture a clearer, well-lit photo of the ingredient list."
        all_warnings.append(ocr_quality_advisory)

    if not raw_ingredients:
        no_ing_warning = "No ingredients detected in OCR output."
        all_warnings.append(no_ing_warning)

        unav_pres = map_personal_care_presentation(
            safety_status="unavailable",
            allergy_status="unavailable",
            irritation_status="unavailable",
        ).to_dict()
        return PersonalCareAnalysisResult(
            category="personal_care",
            success=True,
            ocr=ocr_output,
            personal_care={
                "safety": {"status": "unavailable", "product_risk_class": None, "presentation": unav_pres["personal_care_safety"]},
                "allergy": {"status": "unavailable", "product_risk_class": None, "presentation": unav_pres["allergy"]},
                "irritation": {"status": "unavailable", "product_risk_class": None, "presentation": unav_pres["irritation"]},
                "ingredients": [],
                "total_ingredients": 0,
                "recognized_ingredients": 0,
            },
            errors=[],
            warnings=all_warnings,
            presentation=unav_pres,
            ocr_quality_warning=ocr_quality_advisory,
        )

    # 4. Target-Isolated Semantic Enrichment and ML Predictions
    pc_kb = get_personal_care_knowledge_base()
    ingredient_analyses: List[Dict[str, Any]] = []

    for item in raw_ingredients:
        target_name = (
            item.get("matched_name")
            or item.get("raw_text")
            or item.get("ocr_text")
            or item.get("name")
            or ""
        ).strip()

        if not target_name:
            continue

        raw_display_text = item.get("raw_text") or item.get("ocr_text") or target_name
        matched_name = item.get("matched_name")

        # Semantic enrichment lookup
        enrichment = pc_kb.lookup(target_name)
        if enrichment is None and matched_name:
            enrichment = pc_kb.lookup(matched_name)

        if enrichment is None:
            # Unknown Ingredient Policy: Do NOT fabricate features or map to Safe
            reason = f"Ingredient '{target_name}' not recognized in personal care knowledge base."
            all_warnings.append(f"Unrecognized ingredient: {target_name}")

            ing_record = PersonalCareIngredientAnalysis(
                raw_text=raw_display_text,
                matched_name=matched_name,
                status="ingredient_not_recognized",
                reason=reason,
                features=None,
                safety={"status": "unavailable", "risk_class": None, "confidence": 0.0, "reason": "Ingredient not recognized"},
                allergy={"status": "unavailable", "risk_class": None, "confidence": 0.0, "reason": "Ingredient not recognized"},
                irritation={"status": "unavailable", "risk_class": None, "confidence": 0.0, "reason": "Ingredient not recognized"},
                presentation={
                    "safety": map_personal_care_safety_status(None, "unavailable").to_dict(),
                    "allergy": map_personal_care_allergy_status(None, "unavailable").to_dict(),
                    "irritation": map_personal_care_irritation_status(None, "unavailable").to_dict(),
                },
            )
            ingredient_analyses.append(ing_record.to_dict())
            continue

        # Known ingredient: Execute ML predictions with component isolation
        features_dict = enrichment.to_model_input_dict()
        try:
            pred = predict_personal_care(
                ingredient_name=features_dict["Ingredient_Name"],
                primary_function=features_dict["Primary_Function"],
                ingredient_category=features_dict["Ingredient_Category"],
                product_categories=features_dict["Product_Categories"],
                origin=features_dict["Origin"],
                regulatory_status=features_dict["Regulatory_Status"],
            )

            safety_pred = pred.get("safety", {})
            allergy_pred = pred.get("allergy", {})
            irritation_pred = pred.get("irritation", {})

            safety_pres = map_personal_care_safety_status(safety_pred.get("risk_class"), safety_pred.get("status"))
            allergy_pres = map_personal_care_allergy_status(allergy_pred.get("risk_class"), allergy_pred.get("status"))
            irritation_pres = map_personal_care_irritation_status(irritation_pred.get("risk_class"), irritation_pred.get("status"))

            ing_record = PersonalCareIngredientAnalysis(
                raw_text=raw_display_text,
                matched_name=enrichment.ingredient_name,
                status="success",
                reason=None,
                features=enrichment.to_dict(),
                safety=safety_pred,
                allergy=allergy_pred,
                irritation=irritation_pred,
                presentation={
                    "safety": safety_pres.to_dict(),
                    "allergy": allergy_pres.to_dict(),
                    "irritation": irritation_pres.to_dict(),
                },
            )
            ingredient_analyses.append(ing_record.to_dict())

        except Exception as exc:
            err_desc = f"Model prediction failure for '{target_name}': {str(exc)}"
            logger.error(err_desc, exc_info=True)
            all_warnings.append(err_desc)

            ing_record = PersonalCareIngredientAnalysis(
                raw_text=raw_display_text,
                matched_name=enrichment.ingredient_name,
                status="model_prediction_failure",
                reason=str(exc),
                features=enrichment.to_dict(),
                safety={"status": "model_prediction_failure", "risk_class": None, "confidence": 0.0, "error": str(exc)},
                allergy={"status": "model_prediction_failure", "risk_class": None, "confidence": 0.0, "error": str(exc)},
                irritation={"status": "model_prediction_failure", "risk_class": None, "confidence": 0.0, "error": str(exc)},
                presentation={
                    "safety": map_personal_care_safety_status(None, "unavailable").to_dict(),
                    "allergy": map_personal_care_allergy_status(None, "unavailable").to_dict(),
                    "irritation": map_personal_care_irritation_status(None, "unavailable").to_dict(),
                },
            )
            ingredient_analyses.append(ing_record.to_dict())

    recognized_count = sum(1 for ing in ingredient_analyses if ing.get("status") == "success")

    # 5. Conservative Product-Level Aggregation (Worst-Case per Dimension)
    if is_degraded or recognized_count == 0:
        worst_safety, safety_status = None, "unavailable"
        worst_allergy, allergy_status = None, "unavailable"
        worst_irritation, irritation_status = None, "unavailable"
    else:
        worst_safety, safety_status = aggregate_product_dimension(ingredient_analyses, "safety")
        worst_allergy, allergy_status = aggregate_product_dimension(ingredient_analyses, "allergy")
        worst_irritation, irritation_status = aggregate_product_dimension(ingredient_analyses, "irritation")

    # 6. Presentation Status Mapping across independent dimensions
    presentation_obj = map_personal_care_presentation(
        safety_risk=worst_safety,
        safety_status=safety_status,
        allergy_risk=worst_allergy,
        allergy_status=allergy_status,
        irritation_risk=worst_irritation,
        irritation_status=irritation_status,
    )
    presentation_dict = presentation_obj.to_dict()

    personal_care_summary = {
        "safety": {
            "status": safety_status,
            "product_risk_class": worst_safety,
            "presentation": presentation_dict["personal_care_safety"],
        },
        "allergy": {
            "status": allergy_status,
            "product_risk_class": worst_allergy,
            "presentation": presentation_dict["allergy"],
        },
        "irritation": {
            "status": irritation_status,
            "product_risk_class": worst_irritation,
            "presentation": presentation_dict["irritation"],
        },
        "ingredients": ingredient_analyses,
        "total_ingredients": len(ingredient_analyses),
        "recognized_ingredients": recognized_count,
    }

    return PersonalCareAnalysisResult(
        category="personal_care",
        success=True,
        ocr=ocr_output,
        personal_care=personal_care_summary,
        errors=[],
        warnings=all_warnings,
        presentation=presentation_dict,
        ocr_quality_warning=ocr_quality_advisory,
    )
