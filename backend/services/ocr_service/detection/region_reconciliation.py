"""
detection/region_reconciliation.py

Line Ownership Engine & Section Reconciliation.
Assigns every reconstructed line strictly to:
- INGREDIENT
- NUTRITION
- OTHER
- REJECTED

Resolves line ownership conflicts, eliminates contamination from neighboring
columns, instructions, allergen, manufacturer, storage, mrp, legal, and contact info,
and computes final tight polygon ROIs for Ingredients and Nutrition.
"""

from backend.services.ocr_service import config
from backend.services.ocr_service.detection.geometry import bbox_iou, validate_region
from backend.services.ocr_service.detection.tight_roi import refine_ingredient_roi, refine_nutrition_roi

def classify_line_ownership(line):
    """
    Classifies a line's strict semantic ownership:
    'INGREDIENT', 'NUTRITION', 'OTHER', or 'REJECTED'.
    """
    ing_score = float(line.get("ingredient_score", 0.0))
    nut_score = float(line.get("nutrition_score", 0.0))
    other_score = float(line.get("other_score", 0.0))

    # Check competitor signals
    comp_scores = [
        float(line.get("manufacturer_score", 0.0)),
        float(line.get("contact_score", 0.0)),
        float(line.get("storage_score", 0.0)),
        float(line.get("mrp_score", 0.0)),
        float(line.get("legal_score", 0.0)),
        float(line.get("instruction_score", 0.0)),
        float(line.get("marketing_score", 0.0)),
    ]
    if not getattr(config, "INCLUDE_ALLERGEN_IN_INGREDIENTS", False):
        comp_scores.append(float(line.get("allergen_score", 0.0)))

    comp_max = max(comp_scores) if comp_scores else 0.0
    stop_thresh = float(getattr(config, "STOP_SECTION_SCORE_THRESHOLD", 0.55))
    min_own = float(getattr(config, "MIN_OWNERSHIP_SCORE", 0.25))
    min_margin = float(getattr(config, "MIN_OWNERSHIP_MARGIN", 0.15))

    if comp_max >= stop_thresh:
        return "OTHER"

    # Check Ingredients
    is_ing = (
        ing_score >= min_own
        and (ing_score - nut_score) >= min_margin
        and ing_score >= other_score
    )

    # Check Nutrition
    is_nut = (
        nut_score >= min_own
        and (nut_score - ing_score) >= min_margin
        and nut_score >= other_score
    )

    if is_ing and not is_nut:
        return "INGREDIENT"
    if is_nut and not is_ing:
        return "NUTRITION"

    if other_score > max(ing_score, nut_score) or other_score >= 0.60:
        return "OTHER"

    return "REJECTED"


def assign_all_lines_ownership(lines):
    """
    Annotates every line in the list with an 'owner' field.
    """
    for ln in lines:
        ln["owner"] = classify_line_ownership(ln)
    return lines


def reconcile_regions(ingredient_result, nutrition_result, image_shape, all_lines=None, ingredient_vocab=None):
    """
    Enforces explicit line ownership across detected candidate regions.
    Ensures that every line belongs strictly to at most one region,
    cleans competitor/unrelated lines, and calculates tight polygon ROIs.
    """
    if all_lines:
        assign_all_lines_ownership(all_lines)

    new_ingredient_result = dict(ingredient_result)
    new_nutrition_result = dict(nutrition_result)

    ing_lines = ingredient_result.get("lines") or ingredient_result.get("matched_items", [])
    nut_lines = nutrition_result.get("lines") or nutrition_result.get("matched_items", [])

    # Annotate candidate lines
    for ln in ing_lines:
        if "owner" not in ln:
            ln["owner"] = classify_line_ownership(ln)
    for ln in nut_lines:
        if "owner" not in ln:
            ln["owner"] = classify_line_ownership(ln)

    # Filter candidate lines:
    final_ing_lines = []
    for ln in ing_lines:
        owner = ln.get("owner")
        comp_scores = [
            float(ln.get("manufacturer_score", 0.0)),
            float(ln.get("contact_score", 0.0)),
            float(ln.get("storage_score", 0.0)),
            float(ln.get("mrp_score", 0.0)),
            float(ln.get("legal_score", 0.0)),
            float(ln.get("instruction_score", 0.0)),
            float(ln.get("marketing_score", 0.0)),
        ]
        if not getattr(config, "INCLUDE_ALLERGEN_IN_INGREDIENTS", False):
            comp_scores.append(float(ln.get("allergen_score", 0.0)))
        comp_max = max(comp_scores) if comp_scores else 0.0

        if owner == "INGREDIENT":
            final_ing_lines.append(ln)
        elif owner != "NUTRITION" and comp_max < getattr(config, "STOP_SECTION_SCORE_THRESHOLD", 0.55) and float(ln.get("nutrition_score", 0.0)) < 0.35:
            ln["owner"] = "INGREDIENT"
            final_ing_lines.append(ln)

    final_nut_lines = []
    for ln in nut_lines:
        owner = ln.get("owner")
        comp_scores = [
            float(ln.get("manufacturer_score", 0.0)),
            float(ln.get("contact_score", 0.0)),
            float(ln.get("storage_score", 0.0)),
            float(ln.get("mrp_score", 0.0)),
            float(ln.get("legal_score", 0.0)),
            float(ln.get("instruction_score", 0.0)),
            float(ln.get("marketing_score", 0.0)),
        ]
        comp_max = max(comp_scores) if comp_scores else 0.0

        if owner == "NUTRITION":
            final_nut_lines.append(ln)
        elif owner != "INGREDIENT" and comp_max < getattr(config, "STOP_SECTION_SCORE_THRESHOLD", 0.55) and float(ln.get("ingredient_score", 0.0)) < 0.35:
            ln["owner"] = "NUTRITION"
            final_nut_lines.append(ln)

    # Recompute Ingredients ROI
    if final_ing_lines:
        refine_res = refine_ingredient_roi(final_ing_lines, image_shape)
        new_ingredient_result["roi_polygon"] = refine_res["roi_polygon"]
        new_ingredient_result["bbox"] = refine_res["refined_bbox"]
        new_ingredient_result["line_count"] = refine_res["line_count"]
        new_ingredient_result["owner_lines"] = refine_res["owner_lines"]
        new_ingredient_result["matched_items"] = final_ing_lines
        new_ingredient_result["lines"] = final_ing_lines
        new_ingredient_result["refinement"] = refine_res
    else:
        new_ingredient_result["roi_polygon"] = []
        new_ingredient_result["bbox"] = None
        new_ingredient_result["line_count"] = 0
        new_ingredient_result["owner_lines"] = []
        new_ingredient_result["matched_items"] = []
        new_ingredient_result["lines"] = []
        new_ingredient_result["confidence"] = 0.0

    # Recompute Nutrition ROI
    if final_nut_lines:
        refine_res = refine_nutrition_roi(final_nut_lines, image_shape)
        new_nutrition_result["roi_polygon"] = refine_res["roi_polygon"]
        new_nutrition_result["bbox"] = refine_res["refined_bbox"]
        new_nutrition_result["line_count"] = refine_res["line_count"]
        new_nutrition_result["owner_lines"] = refine_res["owner_lines"]
        new_nutrition_result["matched_items"] = final_nut_lines
        new_nutrition_result["lines"] = final_nut_lines
        new_nutrition_result["refinement"] = refine_res
    else:
        new_nutrition_result["roi_polygon"] = []
        new_nutrition_result["bbox"] = None
        new_nutrition_result["line_count"] = 0
        new_nutrition_result["owner_lines"] = []
        new_nutrition_result["matched_items"] = []
        new_nutrition_result["lines"] = []
        new_nutrition_result["confidence"] = 0.0

    return new_ingredient_result, new_nutrition_result

