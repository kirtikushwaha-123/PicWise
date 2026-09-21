import re
from backend.services.knowledge_base import normalize_value

CONFIDENCE_EXACT_CANONICAL = 1.0
CONFIDENCE_ALTERNATE_NAME = 0.95
CONFIDENCE_NORMALIZED = 0.90
CONFIDENCE_UNMATCHED = 0.0

MATCH_TYPE_EXACT_CANONICAL = "exact_canonical"
MATCH_TYPE_ALTERNATE_NAME = "alternate_name"
MATCH_TYPE_NORMALIZED = "normalized"
MATCH_TYPE_NONE = "none"

STOP_WORDS = {
    "ingredient",
    "ingredients",
    "nutrition",
    "contains",
    "and",
    "with",
    "for",
}


def match_ingredients(extracted_text, knowledge_base):
    """
    Extracts candidate ingredient tokens and performs multi-stage deterministic matching
    against Food and Personal Care knowledge bases.
    """
    candidates = _extract_candidates(extracted_text)
    matches = []

    for candidate in candidates:
        normalized = normalize_value(candidate)
        if not normalized or normalized in STOP_WORDS:
            continue

        match_result = _resolve_candidate(candidate, normalized, knowledge_base)
        matches.append(match_result)

    return _dedupe(matches)


def _resolve_candidate(candidate, normalized, kb):
    # Stage 1: Exact Canonical Match
    food_canonical = kb.food_canonical_index.get(normalized)
    if food_canonical:
        return _build_food_match(candidate, food_canonical, MATCH_TYPE_EXACT_CANONICAL, CONFIDENCE_EXACT_CANONICAL)

    pc_canonical = kb.personal_care_canonical_index.get(normalized)
    if pc_canonical:
        return _build_personal_care_match(candidate, pc_canonical, MATCH_TYPE_EXACT_CANONICAL, CONFIDENCE_EXACT_CANONICAL)

    # Stage 2: Exact Alternate Name Match
    food_alt = kb.food_alternate_index.get(normalized)
    if food_alt:
        return _build_food_match(candidate, food_alt, MATCH_TYPE_ALTERNATE_NAME, CONFIDENCE_ALTERNATE_NAME)

    pc_alt = kb.personal_care_alternate_index.get(normalized)
    if pc_alt:
        return _build_personal_care_match(candidate, pc_alt, MATCH_TYPE_ALTERNATE_NAME, CONFIDENCE_ALTERNATE_NAME)

    # Stage 3: Safe Normalized Fallback Match
    food_fallback = kb.food_index.get(normalized)
    if food_fallback:
        return _build_food_match(candidate, food_fallback, MATCH_TYPE_NORMALIZED, CONFIDENCE_NORMALIZED)

    pc_fallback = kb.personal_care_index.get(normalized)
    if pc_fallback:
        return _build_personal_care_match(candidate, pc_fallback, MATCH_TYPE_NORMALIZED, CONFIDENCE_NORMALIZED)

    # Stage 4: Unknown
    return _build_unmatched(candidate)


def _extract_candidates(text):
    if not text:
        return []
    ingredient_text = re.split(r"\bnutrition\s*:", text, flags=re.IGNORECASE)[0]
    cleaned = re.sub(r"\bingredients?\s*:", "", ingredient_text, flags=re.IGNORECASE)
    parts = re.split(r"[,.;\n]", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _build_food_match(candidate, row, match_type, confidence):
    canonical_name = row.get("Ingredient Name") or candidate
    return {
        "name": canonical_name,
        "canonicalName": canonical_name,
        "originalInput": candidate,
        "matched": True,
        "matchType": match_type,
        "confidence": confidence,
        "domain": "food",
        "category": row.get("Category"),
        "safetyLevel": row.get("Safety Level"),
        "allergyRisk": row.get("Allergy Risk"),
        "healthImpact": row.get("Health Impact"),
        "processingLevel": row.get("Processing Level"),
        "regulatoryStatus": row.get("Regulatory Status"),
        "function": None,
        "irritationRisk": None,
    }


def _build_personal_care_match(candidate, row, match_type, confidence):
    canonical_name = row.get("Ingredient_Name") or candidate
    return {
        "name": canonical_name,
        "canonicalName": canonical_name,
        "originalInput": candidate,
        "matched": True,
        "matchType": match_type,
        "confidence": confidence,
        "domain": "personal_care",
        "primaryFunction": row.get("Primary_Function"),
        "ingredientCategory": row.get("Ingredient_Category"),
        "productCategories": row.get("Product_Categories"),
        "origin": row.get("Origin"),
        "safetyLevel": row.get("Safety_Level"),
        "allergyRisk": row.get("Allergy_Risk"),
        "irritationRisk": row.get("Irritation_Risk"),
        "regulatoryStatus": row.get("Regulatory_Status"),
        "function": row.get("Primary_Function"),
        "healthImpact": None,
        "processingLevel": None,
    }


def _build_unmatched(candidate):
    return {
        "name": candidate,
        "canonicalName": None,
        "originalInput": candidate,
        "matched": False,
        "matchType": MATCH_TYPE_NONE,
        "confidence": CONFIDENCE_UNMATCHED,
        "domain": "unknown",
        "category": None,
        "safetyLevel": None,
        "allergyRisk": None,
        "healthImpact": None,
        "processingLevel": None,
        "regulatoryStatus": None,
        "function": None,
        "irritationRisk": None,
    }


def _dedupe(matches):
    seen = set()
    unique = []
    for match in matches:
        identifier = match["canonicalName"] or match["originalInput"]
        key = normalize_value(identifier)
        if key not in seen:
            seen.add(key)
            unique.append(match)
    return unique
