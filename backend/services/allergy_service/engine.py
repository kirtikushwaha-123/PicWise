"""
backend/services/allergy_service/engine.py

Deterministic food allergy risk engine using ingredient matching and knowledge-base lookup.
Adheres strictly to the frozen authoritative food dataset:
- No ML model or training.
- 100% deterministic lookup.
- Unknown ingredients produce status='unavailable' with reason='Ingredient not found in knowledge base'.
- Highest observed known risk determines product-level risk.
- If no ingredients resolve to a known risk, product status is 'insufficient_data'.
- Personal care category skips food allergy analysis gracefully without raising an error.
"""

from typing import Any, Dict, List, Optional

from backend.services.allergy_service.constants import (
    INSUFFICIENT_DATA_LABEL,
    REASON_NOT_FOUND_IN_KB,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_NO_RISK,
    RISK_RANKS,
    STATUS_ERROR,
    STATUS_INSUFFICIENT_DATA,
    STATUS_NO_INGREDIENTS,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_UNAVAILABLE,
    UI_RISK_LABELS,
    VALID_ALLERGY_RISKS,
)
from backend.services.allergy_service.models import (
    AllergyIngredientResult,
    AllergyResult,
)
from backend.services.knowledge_base import normalize_value

_DEFAULT_KB = None


def _get_default_kb():
    global _DEFAULT_KB
    if _DEFAULT_KB is None:
        from backend.services.knowledge_base import KnowledgeBase
        _DEFAULT_KB = KnowledgeBase.from_env()
    return _DEFAULT_KB


def _lookup_in_kb(query: str, kb: Any) -> Optional[Dict[str, Any]]:
    """Helper to look up a food ingredient row across different KB implementations."""
    if not query or not isinstance(query, str):
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


def calculate_allergy_risk(
    ingredients: Any,
    category: str = "food",
    knowledge_base: Optional[Any] = None,
) -> AllergyResult:
    """
    Deterministically computes allergy risk for food ingredients via knowledge base lookup.

    Parameters:
        ingredients: List of ingredient strings or OCR ingredient dicts.
        category: Must be 'food'. If another category is supplied (e.g. 'personal_care'),
                  the food allergy lookup is gracefully skipped.
        knowledge_base: Optional KnowledgeBase instance. If None, loads the default KB.

    Returns:
        AllergyResult: Comprehensive allergy assessment for the product and ingredients.
    """
    try:
        # 1. Domain / Category Check
        if not category or not isinstance(category, str) or category.strip().lower() != "food":
            return AllergyResult(
                status=STATUS_SKIPPED,
                product_risk_level=None,
                product_ui_label=None,
                allergens_detected=[],
                ingredients=[],
                total_ingredients=0,
                known_ingredients=0,
                unknown_ingredients=0,
                warnings=["Food allergy analysis skipped for non-food category."],
            )

        # 2. Ingredients Existence Check
        if not ingredients:
            return AllergyResult(
                status=STATUS_NO_INGREDIENTS,
                product_risk_level=None,
                product_ui_label=None,
                allergens_detected=[],
                ingredients=[],
                total_ingredients=0,
                known_ingredients=0,
                unknown_ingredients=0,
                warnings=["No ingredients detected for allergy risk analysis."],
            )

        # 3. Acquire KnowledgeBase
        kb = knowledge_base if knowledge_base is not None else _get_default_kb()

        all_results: List[AllergyIngredientResult] = []
        known_results: List[AllergyIngredientResult] = []
        unknown_results: List[AllergyIngredientResult] = []

        # 4. Process Each Ingredient
        for item in ingredients:
            if isinstance(item, str):
                raw_text = item.strip()
                matched_name = None
                match_type = "unmatched"
                candidates = [raw_text]
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
            else:
                raw_text = str(item).strip()
                matched_name = None
                match_type = "unmatched"
                candidates = [raw_text]

            row = None
            for cand in candidates:
                row = _lookup_in_kb(cand, kb)
                if row:
                    break

            if row:
                canonical_name = row.get("Ingredient Name") or matched_name or raw_text
                raw_risk = row.get("Allergy Risk")
                if raw_risk is not None and str(raw_risk).strip():
                    clean_risk = str(raw_risk).strip()
                    if clean_risk.lower() == "none":
                        clean_risk = RISK_NO_RISK

                    if clean_risk in VALID_ALLERGY_RISKS:
                        resolved_type = match_type if match_type != "unmatched" else "exact_canonical"
                        ing_res = AllergyIngredientResult(
                            ingredient=canonical_name,
                            allergy_risk=clean_risk,
                            ui_label=UI_RISK_LABELS[clean_risk],
                            status=STATUS_SUCCESS,
                            reason=None,
                            matched_name=canonical_name,
                            raw_text=raw_text,
                            match_type=resolved_type,
                        )
                        known_results.append(ing_res)
                        all_results.append(ing_res)
                        continue

            # If not found or allergy risk column empty
            fallback_name = (candidates[0] if candidates else raw_text) or "unknown"
            ing_res = AllergyIngredientResult(
                ingredient=fallback_name,
                allergy_risk=None,
                ui_label=None,
                status=STATUS_UNAVAILABLE,
                reason=REASON_NOT_FOUND_IN_KB,
                matched_name=None,
                raw_text=raw_text,
                match_type="unmatched",
            )
            unknown_results.append(ing_res)
            all_results.append(ing_res)

        # 5. Product-Level Aggregation
        total_count = len(all_results)
        known_count = len(known_results)
        unknown_count = len(unknown_results)

        if known_count == 0:
            return AllergyResult(
                status=STATUS_INSUFFICIENT_DATA,
                product_risk_level=None,
                product_ui_label=INSUFFICIENT_DATA_LABEL,
                allergens_detected=[],
                ingredients=[r.to_dict() for r in all_results],
                total_ingredients=total_count,
                known_ingredients=0,
                unknown_ingredients=unknown_count,
                warnings=["Insufficient allergy data: no ingredients could be matched to the food knowledge base."],
            )

        # Highest observed known risk determines product risk
        highest_risk = max(
            (r.allergy_risk for r in known_results if r.allergy_risk in RISK_RANKS),
            key=lambda risk: RISK_RANKS[risk],
        )
        product_ui_label = UI_RISK_LABELS[highest_risk]

        # Detected allergens: ingredients with Low, Medium, or High risk
        allergens_detected = [
            r.ingredient for r in known_results
            if r.allergy_risk in (RISK_LOW, RISK_MEDIUM, RISK_HIGH)
        ]

        warnings = []
        if unknown_count > 0:
            warnings.append(
                f"{unknown_count} of {total_count} ingredients could not be matched to the food knowledge base."
            )

        # If all known ingredients are No Risk but there are unknown ingredients,
        # we cannot treat unknown as safe or report Allergen-Free / No Risk.
        if highest_risk == RISK_NO_RISK and unknown_count > 0:
            return AllergyResult(
                status=STATUS_INSUFFICIENT_DATA,
                product_risk_level=None,
                product_ui_label=INSUFFICIENT_DATA_LABEL,
                allergens_detected=[],
                ingredients=[r.to_dict() for r in all_results],
                total_ingredients=total_count,
                known_ingredients=known_count,
                unknown_ingredients=unknown_count,
                warnings=warnings,
            )

        return AllergyResult(
            status=STATUS_SUCCESS,
            product_risk_level=highest_risk,
            product_ui_label=product_ui_label,
            allergens_detected=allergens_detected,
            ingredients=[r.to_dict() for r in all_results],
            total_ingredients=total_count,
            known_ingredients=known_count,
            unknown_ingredients=unknown_count,
            warnings=warnings,
        )

    except Exception as exc:
        err_msg = f"Allergy risk analysis encountered an error: {str(exc)}"
        return AllergyResult(
            status=STATUS_ERROR,
            product_risk_level=None,
            product_ui_label=None,
            allergens_detected=[],
            ingredients=[],
            total_ingredients=0,
            known_ingredients=0,
            unknown_ingredients=0,
            warnings=[err_msg],
            error=str(exc),
        )
