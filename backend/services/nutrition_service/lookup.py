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


def lookup_nutrient_term(term, knowledge_base):
    """
    Look up a single nutrient term independently against the Nutrition Knowledge Base.
    """
    if not term:
        return _build_unmatched_nutrient(term)

    normalized = normalize_value(term)
    if not normalized:
        return _build_unmatched_nutrient(term)

    # Stage 1: Exact Canonical Match
    row = knowledge_base.nutrition_canonical_index.get(normalized)
    if row:
        return _build_nutrient_match(term, row, MATCH_TYPE_EXACT_CANONICAL, CONFIDENCE_EXACT_CANONICAL)

    # Stage 2: Exact Alternate Name Match
    row = knowledge_base.nutrition_alternate_index.get(normalized)
    if row:
        return _build_nutrient_match(term, row, MATCH_TYPE_ALTERNATE_NAME, CONFIDENCE_ALTERNATE_NAME)

    # Stage 3: Safe Normalized Fallback Match
    row = knowledge_base.nutrition_index.get(normalized)
    if row:
        return _build_nutrient_match(term, row, MATCH_TYPE_NORMALIZED, CONFIDENCE_NORMALIZED)

    # Stage 4: Unknown
    return _build_unmatched_nutrient(term)


def find_relevant_nutrition(extracted_text, knowledge_base):
    """
    Scans extracted text for nutrition knowledge matches.
    """
    if not extracted_text:
        return []

    normalized_text = normalize_value(extracted_text)
    words = set(normalized_text.split())
    nutrition = []

    # Check canonical nutrients first
    for key, row in knowledge_base.nutrition_canonical_index.items():
        if _mentions_key(key, normalized_text, words):
            nutrition.append(_build_nutrient_match(key, row, MATCH_TYPE_EXACT_CANONICAL, CONFIDENCE_EXACT_CANONICAL))

    # Check alternate nutrient terms
    for key, row in knowledge_base.nutrition_alternate_index.items():
        if _mentions_key(key, normalized_text, words):
            nutrition.append(_build_nutrient_match(key, row, MATCH_TYPE_ALTERNATE_NAME, CONFIDENCE_ALTERNATE_NAME))

    return _dedupe_nutrition(nutrition)


def _mentions_key(key, normalized_text, words):
    if len(key) < 3:
        return False
    if " " in key:
        return f" {key} " in f" {normalized_text} "
    return key in words


def _build_nutrient_match(input_term, row, match_type, confidence):
    canonical_name = row.get("Nutrient") or input_term
    return {
        "nutrient": canonical_name,
        "canonicalNutrient": canonical_name,
        "originalInput": input_term,
        "matched": True,
        "matchType": match_type,
        "confidence": confidence,
        "healthRole": row.get("Health Role"),
        "role": row.get("Health Role"),
        "healthImpact": row.get("Health Impact"),
        "decisionPriority": row.get("Decision Priority"),
        "betterDirection": row.get("Better Direction"),
    }


def _build_unmatched_nutrient(input_term):
    return {
        "nutrient": input_term,
        "canonicalNutrient": None,
        "originalInput": input_term,
        "matched": False,
        "matchType": MATCH_TYPE_NONE,
        "confidence": CONFIDENCE_UNMATCHED,
        "healthRole": None,
        "role": None,
        "healthImpact": None,
        "decisionPriority": None,
        "betterDirection": None,
    }


def _dedupe_nutrition(items):
    seen = set()
    unique = []
    for item in items:
        key = normalize_value(item["canonicalNutrient"] or item["originalInput"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
