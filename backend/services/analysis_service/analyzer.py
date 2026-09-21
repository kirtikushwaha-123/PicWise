from backend.ml.inference.food_safety_service import predict_food_safety
from backend.services.ocr_service import run_ocr


def analyze_product_image(image_bytes, knowledge_base=None, category="food"):
    """
    Analyzes a product image using the integrated OCR service and user-selected category.
    Downstream of OCR, if category == 'food', executes production Food Safety inference
    (Character TF-IDF + MiniLM-L6-v2 + Balanced Logistic Regression) for each ingredient.
    If category == 'personal_care', Food Safety inference is strictly skipped.

    Parameters:
        image_bytes (bytes): Raw bytes of the uploaded image.
        knowledge_base: Optional legacy KnowledgeBase instance (for backward compatibility).
        category (str): Product category, strictly 'food' or 'personal_care'.
    """
    category = (category or "food").strip().lower()
    if category not in ("food", "personal_care"):
        raise ValueError(f"Invalid category '{category}'. Must be 'food' or 'personal_care'.")

    ocr_output = run_ocr(image_bytes, category=category)

    # Normalize extracted ingredients
    raw_ingredients = ocr_output.get("ingredients", [])
    normalized_ingredients = []

    for item in raw_ingredients:
        name = item.get("matched_name") or item.get("raw_text") or "Unknown"
        matched = bool(item.get("matched_name"))
        confidence = float(item.get("similarity", 0.0) or item.get("score", 0.0) or 0.0)
        meta = item.get("metadata") or {}

        entry = {
            "name": name,
            "raw_text": item.get("raw_text", ""),
            "canonicalName": item.get("matched_name") or name,
            "matched": matched,
            "confidence": round(confidence, 4),
            "domain": category,
            "matchType": item.get("method", "unmatched"),
            "metadata": meta,
        }

        if category == "food":
            # Execute production Food Safety ML inference downstream of OCR
            score_target = item.get("matched_name") or item.get("raw_text") or name
            food_safety_result = predict_food_safety(score_target)

            entry["foodSafety"] = food_safety_result
            entry["safetyLevel"] = food_safety_result.get("risk_class") or meta.get("Safety Level")
            entry["allergyRisk"] = meta.get("Allergy Risk")
            entry["healthImpact"] = meta.get("Health Impact")
            entry["processingLevel"] = meta.get("Processing Level")
            entry["regulatoryStatus"] = meta.get("Regulatory Status")
            entry["category"] = meta.get("Category")
        else:
            # Personal care: strictly NO Food Safety inference
            entry["foodSafety"] = None
            entry["safetyLevel"] = meta.get("Safety_Level")
            entry["allergyRisk"] = meta.get("Allergy_Risk")
            entry["irritationRisk"] = meta.get("Irritation_Risk")
            entry["regulatoryStatus"] = meta.get("Regulatory_Status")
            entry["primaryFunction"] = meta.get("Primary_Function")
            entry["function"] = meta.get("Primary_Function")
            entry["ingredientCategory"] = meta.get("Ingredient_Category")
            entry["productCategories"] = meta.get("Product_Categories")
            entry["origin"] = meta.get("Origin")

        normalized_ingredients.append(entry)

    # Category-specific nutrition and personal care structures
    if category == "food":
        nutrition_data = ocr_output.get("nutrition") or {}
        personal_care_data = []
    else:
        nutrition_data = None
        personal_care_data = _build_personal_care_results(normalized_ingredients)

    warnings = []
    if knowledge_base and hasattr(knowledge_base, "warnings"):
        warnings.extend(knowledge_base.warnings)

    if not normalized_ingredients:
        warnings.append("No ingredients could be reliably detected or matched from the image.")

    return {
        "product": {
            "name": None,
            "brand": None,
            "domain": category,
        },
        "ingredients": normalized_ingredients,
        "nutrition": nutrition_data,
        "personalCare": personal_care_data,
        "warnings": warnings,
        "ocr": ocr_output,
    }


def _build_personal_care_results(ingredients):
    return [
        {
            "ingredient": item["name"],
            "canonicalName": item.get("canonicalName") or item["name"],
            "originalInput": item.get("raw_text") or item["name"],
            "matchType": item.get("matchType"),
            "confidence": item.get("confidence"),
            "function": item.get("function"),
            "primaryFunction": item.get("primaryFunction"),
            "ingredientCategory": item.get("ingredientCategory"),
            "productCategories": item.get("productCategories"),
            "origin": item.get("origin"),
            "safetyLevel": item.get("safetyLevel"),
            "allergyRisk": item.get("allergyRisk"),
            "irritationRisk": item.get("irritationRisk"),
            "regulatoryStatus": item.get("regulatoryStatus"),
            "metadata": item.get("metadata", {}),
        }
        for item in ingredients
    ]
