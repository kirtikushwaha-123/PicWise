"""
backend/services/food_analysis_service/analyzer.py

PicWise Unified Food Analysis Orchestration Service.
Coordinates OCR, Food Safety ML inference, Nutrition Scoring Engine,
and Allergy analysis into a deterministic, unified pipeline.
"""

import logging
from typing import Optional, Any, Dict, List

from backend.ml.inference.food_safety_service import predict_food_safety
from backend.services.food_analysis_service.errors import (
    InvalidCategoryError,
    ImageProcessingError,
    OCRError,
)
from backend.services.food_analysis_service.models import (
    FoodAnalysisResult,
    FoodSafetyResult,
    FoodSafetyIngredientResult,
    AllergyResult,
)
from backend.services.nutrition_service import calculate_nutrition_score
from backend.services.ocr_service import run_ocr
from backend.services.allergy_service import calculate_allergy_risk
from backend.services.food_status_service import (
    map_food_safety_status,
    map_allergy_status,
    map_nutrition_status,
    map_food_analysis_presentation,
)

from backend.services.knowledge_base import normalize_value

logger = logging.getLogger(__name__)

CANONICAL_SAFETY_CLASSES = ["Very Safe", "Safe", "Moderate Risk", "High Risk"]
CANONICAL_SAFETY_MAP = {
    "very safe": "Very Safe",
    "safe": "Safe",
    "moderate risk": "Moderate Risk",
    "high risk": "High Risk",
}


def _lookup_in_kb(query: str, kb: Any) -> Optional[Dict[str, Any]]:
    """Helper to look up a food ingredient row in the KnowledgeBase."""
    if not query or not isinstance(query, str) or kb is None:
        return None

    # 1. Preferred method: get_food_ingredient
    if hasattr(kb, "get_food_ingredient"):
        res = kb.get_food_ingredient(query)
        if res:
            return res

    # 2. Check food_index dictionary directly
    if hasattr(kb, "food_index") and isinstance(kb.food_index, dict):
        norm_key = normalize_value(query)
        if norm_key in kb.food_index:
            return kb.food_index[norm_key]

    # 3. Fallback: linear scan over kb.food if available
    if hasattr(kb, "food") and isinstance(kb.food, list):
        norm_query = normalize_value(query)
        for row in kb.food:
            name = row.get("Ingredient Name")
            if name and normalize_value(name) == norm_query:
                return row
            alts = row.get("Packaging Names / Alternate Names")
            if alts:
                for alt in str(alts).split(";"):
                    if normalize_value(alt) == norm_query:
                        return row

    return None


def analyze_food(
    image_bytes: bytes,
    category: str = "food",
    knowledge_base: Optional[Any] = None,
) -> FoodAnalysisResult:
    """
    Orchestrates end-to-end food product analysis.

    Parameters:
        image_bytes: Raw bytes of the uploaded food product image.
        category: Must explicitly be 'food'. Any other category or missing value raises InvalidCategoryError.
        knowledge_base: Optional KnowledgeBase instance for OCR fuzzy matching.

    Returns:
        FoodAnalysisResult: Unified typed result combining OCR, Food Safety, Nutrition,
                            and Allergy components.
    """
    # 1. Category Validation (strictly 'food', no automated category guessing)
    if not category or not isinstance(category, str):
        raise InvalidCategoryError("Category is required and must be 'food'.")

    category_norm = category.strip().lower()
    if category_norm != "food":
        raise InvalidCategoryError(
            f"Invalid category '{category}'. The Food Analysis service strictly handles 'food'."
        )

    # 2. Image Byte Validation
    if not image_bytes or len(image_bytes) == 0:
        raise ImageProcessingError("Image bytes are empty or missing.")

    # 3. OCR Pipeline Execution
    try:
        # Only pass knowledge_base to OCR if it supports the OCR KB interface (get_ingredient_names)
        ocr_kb = knowledge_base if (knowledge_base and hasattr(knowledge_base, "get_ingredient_names")) else None
        ocr_output = run_ocr(image_bytes, category="food", kb=ocr_kb)
    except Exception as exc:
        err_msg = f"OCR processing failed: {str(exc)}"
        logger.error(err_msg, exc_info=True)
        # On OCR failure, downstream models are NOT called with fabricated data
        # Presentation mapping is explicitly unavailable across all dimensions
        return FoodAnalysisResult(
            category="food",
            success=False,
            ocr=None,
            food_safety=None,
            nutrition=None,
            allergy=None,
            errors=[err_msg],
            warnings=[],
            presentation=map_food_analysis_presentation(None, None, None).to_dict(),
        )

    if not ocr_output:
        ocr_output = {}

    raw_ingredients = ocr_output.get("ingredients", [])
    raw_nutrition = ocr_output.get("nutrition")
    raw_text_dict = ocr_output.get("raw_text") or {}
    product_text = raw_text_dict.get("all_text", "")
    ingredient_text = raw_text_dict.get("ingredients_text", "")

    return _evaluate_food_components(
        raw_ingredients=raw_ingredients,
        raw_nutrition=raw_nutrition,
        product_text=product_text,
        ingredient_text=ingredient_text,
        knowledge_base=knowledge_base,
        ocr_output=ocr_output,
    )


def analyze_food_from_text(
    ingredients_text: str,
    nutrition_text: str,
    all_text: str = "",
    category: str = "food",
    knowledge_base: Optional[Any] = None,
) -> FoodAnalysisResult:
    """
    Analyzes food product from user-reviewed / edited OCR text.
    Reuses the existing:
    - ingredient parser / corrector
    - nutrition parser
    - Food Safety analyzer
    - Allergy analyzer
    - Nutrition analyzer
    """
    if not category or not isinstance(category, str):
        raise InvalidCategoryError("Category is required and must be 'food'.")

    category_norm = category.strip().lower()
    if category_norm != "food":
        raise InvalidCategoryError(
            f"Invalid category '{category}'. The Food Analysis service strictly handles 'food'."
        )

    ocr_kb = knowledge_base if (knowledge_base and hasattr(knowledge_base, "get_ingredient_names")) else None

    # 1. Parse Ingredients from corrected text
    ingredients_output = []
    if ingredients_text and ingredients_text.strip():
        from backend.services.ocr_service.nlp.ingredient_corrector import IngredientCorrector
        from backend.services.ocr_service.parsing.ingredient_parser import parse_ingredients
        corrector = IngredientCorrector(kb=ocr_kb)
        ingredients_output = corrector.correct_and_match(ingredients_text.strip(), domain="food")
        if not ingredients_output:
            parsed_tokens = parse_ingredients(ingredients_text)
            if parsed_tokens and ocr_kb:
                matched_kb = ocr_kb.match_ingredient_list(parsed_tokens, domain="food")
                for tok, m in zip(parsed_tokens, matched_kb):
                    sim = m["similarity"] / 100.0 if m["matched_name"] else None
                    ingredients_output.append({
                        "ocr_text": tok,
                        "normalized_text": tok,
                        "corrected_ingredient": m["matched_name"],
                        "match_confidence": sim,
                        "matched_name": m["matched_name"],
                        "confidence": sim,
                    })
            else:
                for tok in parsed_tokens:
                    ingredients_output.append({
                        "ocr_text": tok,
                        "normalized_text": tok,
                        "corrected_ingredient": None,
                        "match_confidence": None,
                        "matched_name": None,
                        "confidence": None,
                    })

    # 2. Parse Nutrition from corrected text
    nutrition_output = None
    if nutrition_text and nutrition_text.strip():
        from backend.services.ocr_service.parsing.nutrition_parser import parse_nutrition
        nutrition_output = parse_nutrition(nutrition_text.strip())

    # 3. Construct synthetic ocr dictionary for downstream consistency
    ocr_dict = {
        "domain": "food",
        "ingredients": ingredients_output,
        "nutrition": nutrition_output,
        "raw_text": {
            "ingredients_text": ingredients_text or "",
            "nutrition_text": nutrition_text or "",
            "other_text": all_text or "",
            "all_text": f"{ingredients_text or ''}\n{nutrition_text or ''}\n{all_text or ''}".strip(),
        },
    }

    # 4. Evaluate all food components using the exact same evaluation logic
    return _evaluate_food_components(
        raw_ingredients=ingredients_output,
        raw_nutrition=nutrition_output,
        product_text=ocr_dict["raw_text"]["all_text"],
        ingredient_text=ingredients_text or "",
        knowledge_base=knowledge_base,
        ocr_output=ocr_dict,
    )


def _evaluate_food_components(
    raw_ingredients: List[Any],
    raw_nutrition: Optional[Dict[str, Any]],
    product_text: str = "",
    ingredient_text: str = "",
    knowledge_base: Optional[Any] = None,
    ocr_output: Optional[Dict[str, Any]] = None,
) -> FoodAnalysisResult:
    """
    Unified evaluation engine for food components (Food Safety, Nutrition, Allergy).
    Used by both image-based analyze_food and text-based analyze_food_from_text.
    """
    all_warnings: List[str] = []

    # 4. Food Safety ML Inference
    food_safety_dict: Dict[str, Any]

    if not raw_ingredients:
        no_ing_warning = "No ingredients detected in OCR output."
        all_warnings.append(no_ing_warning)
        food_safety_dict = FoodSafetyResult(
            status="no_ingredients",
            ingredients=[],
            total_ingredients=0,
            warnings=[no_ing_warning],
        ).to_dict()
    else:
        try:
            safety_items: List[Dict[str, Any]] = []
            for item in raw_ingredients:
                if isinstance(item, str):
                    raw_text = item.strip()
                    matched_name = None
                    match_type = "unmatched"
                    candidates = [raw_text] if raw_text else []
                elif isinstance(item, dict):
                    raw_text = str(
                        item.get("raw_text")
                        or item.get("ocr_text")
                        or item.get("name")
                        or ""
                    ).strip()
                    matched_name = item.get("matched_name")
                    match_type = item.get("method") or item.get("match_type") or "unmatched"

                    candidates = []
                    if matched_name and str(matched_name).strip():
                        candidates.append(str(matched_name).strip())
                    if raw_text and raw_text not in candidates:
                        candidates.append(raw_text)
                    name_val = item.get("name")
                    if name_val and str(name_val).strip() not in candidates:
                        candidates.append(str(name_val).strip())
                    ocr_val = item.get("ocr_text")
                    if ocr_val and str(ocr_val).strip() not in candidates:
                        candidates.append(str(ocr_val).strip())
                else:
                    raw_text = str(item).strip()
                    matched_name = None
                    match_type = "unmatched"
                    candidates = [raw_text] if raw_text else []

                target_name = (
                    matched_name
                    or (candidates[0] if candidates else raw_text)
                ).strip()

                if not target_name:
                    continue

                # 1. For an ingredient FOUND in the Knowledge Base:
                #    use KB Safety Level, source = "knowledge_base", confidence = 1.0
                row = None
                if knowledge_base is not None:
                    for cand in candidates:
                        row = _lookup_in_kb(cand, knowledge_base)
                        if row:
                            break

                kb_safety = row.get("Safety Level") if row else None

                if row and kb_safety and str(kb_safety).strip().lower() in CANONICAL_SAFETY_MAP:
                    risk_class = CANONICAL_SAFETY_MAP[str(kb_safety).strip().lower()]
                    source = "knowledge_base"
                    confidence = 1.0
                    resolved_name = row.get("Ingredient Name") or matched_name or target_name
                    probabilities = {c: (1.0 if c == risk_class else 0.0) for c in CANONICAL_SAFETY_CLASSES}
                else:
                    # 2. For an ingredient NOT found in the Knowledge Base:
                    #    run ML predictor, accept only when top probability >= 0.60
                    pred = predict_food_safety(target_name)
                    pred_conf = float(pred.get("confidence", 0.0))
                    pred_risk = pred.get("risk_class")
                    probabilities = pred.get("probabilities", {})
                    resolved_name = pred.get("ingredient") or target_name

                    if pred_conf >= 0.60 and pred_risk and str(pred_risk).strip().lower() in CANONICAL_SAFETY_MAP:
                        risk_class = CANONICAL_SAFETY_MAP[str(pred_risk).strip().lower()]
                        source = "model"
                        confidence = pred_conf
                    else:
                        # If ML confidence < 0.60: risk_class = None, source = "unrated"
                        risk_class = None
                        source = "unrated"
                        confidence = pred_conf

                safety_entry = FoodSafetyIngredientResult(
                    ingredient=resolved_name,
                    risk_class=risk_class,
                    confidence=confidence,
                    probabilities=probabilities,
                    raw_text=raw_text,
                    matched_name=matched_name,
                    match_type=match_type,
                    source=source,
                )
                entry_dict = safety_entry.to_dict()
                ing_pres = map_food_safety_status(risk_class)
                entry_dict["presentation_status"] = ing_pres.status
                entry_dict["presentation"] = ing_pres.to_dict()
                safety_items.append(entry_dict)

            # Compute product-level risk_class using worst-case aggregation:
            # Very Safe < Safe < Moderate Risk < High Risk
            product_risk_class = None
            rated_count = 0
            unrated_count = 0
            food_safety_warnings = []
            if safety_items:
                risk_order = {
                    "high risk": (4, "High Risk"),
                    "moderate risk": (3, "Moderate Risk"),
                    "safe": (2, "Safe"),
                    "very safe": (1, "Very Safe"),
                }
                max_rank = 0
                for item in safety_items:
                    rc = item.get("risk_class")
                    if rc and isinstance(rc, str) and rc.strip().lower() in risk_order:
                        rank, canonical_rc = risk_order[rc.strip().lower()]
                        rated_count += 1
                        if rank > max_rank:
                            max_rank = rank
                            product_risk_class = canonical_rc
                    else:
                        unrated_count += 1

                # If no ingredients are rated, Food Safety = Unavailable
                if rated_count == 0:
                    product_risk_class = None

                # If some ingredients are unrated, keep the overall result based on rated ingredients
                # and show the warning.
                if unrated_count > 0:
                    if rated_count > 0:
                        unrated_warn = f"{unrated_count} of {len(safety_items)} ingredients could not be evaluated for food safety."
                    else:
                        unrated_warn = f"None of the {len(safety_items)} ingredients could be evaluated for food safety."
                    food_safety_warnings.append(unrated_warn)
                    if unrated_warn not in all_warnings:
                        all_warnings.append(unrated_warn)

            food_safety_dict = FoodSafetyResult(
                status="success",
                ingredients=safety_items,
                total_ingredients=len(safety_items),
                warnings=food_safety_warnings,
                risk_class=product_risk_class,
            ).to_dict()
        except Exception as exc:
            # Component isolation: food safety failure does not crash the entire food analysis
            err_msg = f"Food safety inference encountered an error: {str(exc)}"
            logger.error(err_msg, exc_info=True)
            all_warnings.append(err_msg)
            food_safety_dict = FoodSafetyResult(
                status="error",
                ingredients=[],
                total_ingredients=0,
                warnings=[err_msg],
                error=str(exc),
                risk_class=None,
            ).to_dict()

    # Food Safety presentation mapping
    fs_pres = map_food_safety_status(food_safety_dict.get("risk_class"))
    food_safety_dict["presentation_status"] = fs_pres.status
    food_safety_dict["presentation"] = fs_pres.to_dict()

    # 5. Nutrition Scoring Engine Execution
    nutrition_dict: Dict[str, Any]
    try:
        nutrition_res = calculate_nutrition_score(
            raw_nutrition=raw_nutrition,
            category="food",
            product_text=product_text,
            ingredient_text=ingredient_text,
            knowledge_base=knowledge_base,
        )
        nutrition_dict = nutrition_res if isinstance(nutrition_res, dict) else {}
        if nutrition_dict.get("warnings"):
            all_warnings.extend(nutrition_dict["warnings"])
    except Exception as exc:
        # Component isolation: nutrition error does not crash other components
        nut_err_msg = f"Nutrition scoring engine encountered an error: {str(exc)}"
        logger.error(nut_err_msg, exc_info=True)
        all_warnings.append(nut_err_msg)
        nutrition_dict = {
            "nutrition_score": None,
            "status": "error",
            "warnings": [nut_err_msg],
            "error": str(exc),
        }

    # Nutrition presentation mapping
    nut_score = nutrition_dict.get("nutrition_score")
    nut_pres = map_nutrition_status(
        score=nut_score,
        raw_status=nutrition_dict.get("status"),
    )
    nutrition_dict["presentation_status"] = nut_pres.status
    nutrition_dict["presentation"] = nut_pres.to_dict()
    if "score" not in nutrition_dict:
        nutrition_dict["score"] = nut_score
    if "label" not in nutrition_dict:
        nutrition_dict["label"] = nut_pres.label

    # 6. Allergy Pipeline Execution (Deterministic Knowledge Base Lookup)
    allergy_dict: Dict[str, Any]
    try:
        allergy_res = calculate_allergy_risk(
            ingredients=raw_ingredients,
            category="food",
            knowledge_base=knowledge_base,
        )
        allergy_dict = allergy_res.to_dict() if hasattr(allergy_res, "to_dict") else allergy_res
        if allergy_dict.get("warnings"):
            all_warnings.extend(allergy_dict["warnings"])
    except Exception as exc:
        # Component isolation: allergy lookup error does not crash the entire food analysis
        all_err_msg = f"Allergy risk lookup encountered an error: {str(exc)}"
        logger.error(all_err_msg, exc_info=True)
        all_warnings.append(all_err_msg)
        allergy_dict = AllergyResult(
            status="error",
            warnings=[all_err_msg],
            error=str(exc),
        ).to_dict()

    # Allergy presentation mapping
    al_risk = allergy_dict.get("product_risk_level") or allergy_dict.get("risk_level")
    al_pres = map_allergy_status(
        risk_level=al_risk,
        raw_status=allergy_dict.get("status"),
    )
    allergy_dict["presentation_status"] = al_pres.status
    allergy_dict["presentation"] = al_pres.to_dict()
    if "risk_level" not in allergy_dict and al_risk:
        allergy_dict["risk_level"] = al_risk
    if "ui_label" not in allergy_dict and allergy_dict.get("product_ui_label"):
        allergy_dict["ui_label"] = allergy_dict.get("product_ui_label")

    # 7. Presentation Status Mapping across independent dimensions (Phase 9H)
    presentation_dict: Optional[Dict[str, Any]] = None
    try:
        presentation_obj = map_food_analysis_presentation(
            food_safety=food_safety_dict,
            allergy=allergy_dict,
            nutrition=nutrition_dict,
        )
        presentation_dict = presentation_obj.to_dict()
    except Exception as exc:
        pres_err_msg = f"Presentation status mapping failed: {str(exc)}"
        logger.error(pres_err_msg, exc_info=True)
        all_warnings.append(pres_err_msg)
        presentation_dict = {
            "food_safety": {"status": "unavailable"},
            "allergy": {"status": "unavailable"},
            "nutrition": {"status": "unavailable"},
        }

    # 8. Assemble Unified Result
    logger.info(
        "Food analysis completed. Ingredients: %d, Allergy status: %s, Nutrition score: %s",
        len(raw_ingredients),
        allergy_dict.get("presentation_status"),
        str(nutrition_dict.get("nutrition_score")),
    )
    return FoodAnalysisResult(
        category="food",
        success=True,
        ocr=ocr_output,
        food_safety=food_safety_dict,
        nutrition=nutrition_dict,
        allergy=allergy_dict,
        errors=[],
        warnings=all_warnings,
        presentation=presentation_dict,
    )
