"""
backend/services/ocr_service/pipeline.py

PicWise Production OCR Service Boundary.
Adapts the accepted final-ocr pipeline for in-memory byte buffers and category-driven routing.
"""

import os
import time

from backend.services.ocr_service import config
from backend.services.ocr_service.preprocessing.image_utils import (
    load_image,
    normalize_image,
    check_image_quality,
    safe_crop,
    save_image,
)
from backend.services.ocr_service.preprocessing.enhancement import preprocess_roi
from backend.services.ocr_service.preprocessing.deskew import deskew
from backend.services.ocr_service.preprocessing.perspective import correct_perspective

from backend.services.ocr_service.detection.packet_region import detect_packet_region
from backend.services.ocr_service.detection.ocr_detector import run_full_image_ocr
from backend.services.ocr_service.detection.ingredient_region import detect_ingredient_region
from backend.services.ocr_service.detection.nutrition_region import detect_nutrition_region
from backend.services.ocr_service.detection.region_reconciliation import reconcile_regions
from backend.services.ocr_service.detection.document_layout import analyze_document

from backend.services.ocr_service.ocr.ensemble import run_variant_ocr

from backend.services.ocr_service.parsing.ingredient_parser import parse_ingredients
from backend.services.ocr_service.parsing.nutrition_parser import parse_nutrition

from backend.services.ocr_service.matching.knowledge_base import KnowledgeBase
from backend.services.ocr_service.nlp.ingredient_corrector import IngredientCorrector

# Module-level cached KnowledgeBase singleton for OCR fuzzy matching
_CACHED_KB = None


def get_ocr_knowledge_base():
    global _CACHED_KB
    if _CACHED_KB is None:
        _CACHED_KB = KnowledgeBase()
    return _CACHED_KB


def process_region(image, region_result, mode, save_prefix, output_dir=None, test_mode=False):
    """
    Shared post-detection pipeline for a detected region (ingredients or nutrition):
    pad+crop -> deskew -> perspective correction -> multi-variant preprocessing -> OCR ensemble.
    """
    bbox = region_result.get("bbox")
    if bbox is None:
        return None, None, None

    crop = safe_crop(image, bbox)
    if crop is None:
        return None, None, None

    # Small/dense text enhancement: if crop height is small (< 120px), upscale crop
    # with INTER_CUBIC so PaddleOCR can reliably detect small text lines (DEF-03)
    if crop.shape[0] < 120 and crop.shape[1] > 100:
        import cv2
        scale = max(1.5, 150.0 / crop.shape[0])
        new_w = int(round(crop.shape[1] * scale))
        new_h = int(round(crop.shape[0] * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    deskewed, angle = deskew(crop)
    corrected, applied_perspective = correct_perspective(deskewed)
    variants = preprocess_roi(corrected)

    if test_mode and output_dir:
        stage_dir = os.path.join(output_dir, "stages", save_prefix)
        os.makedirs(stage_dir, exist_ok=True)
        save_image(crop, os.path.join(stage_dir, "01_raw_crop.jpg"))
        save_image(deskewed, os.path.join(stage_dir, "02_deskewed.jpg"))
        save_image(corrected, os.path.join(stage_dir, "03_perspective_corrected.jpg"))
        for name, img in variants.items():
            save_image(img, os.path.join(stage_dir, f"variant_{name}.jpg"))

    ensemble_result = run_variant_ocr(variants, mode=mode)

    best_variant_name = ensemble_result.get("best_variant")
    best_processed_image = variants.get(best_variant_name) if best_variant_name else corrected

    return crop, best_processed_image, ensemble_result


def _reconstruct_nutrition_table_text(matched_items):
    """
    Pairs nutrient labels with nearby value cells based on geometry and y-alignment.
    Preserves row reading order and formats clean lines:
    'Label Value' (e.g. 'Calories 160', 'Total Fat 10g')
    """
    if not matched_items:
        return ""

    import numpy as np
    sorted_items = sorted(
        matched_items,
        key=lambda it: (it["rect"][1] + it["rect"][3]) / 2.0
    )

    rows = []
    for it in sorted_items:
        it_yc = (it["rect"][1] + it["rect"][3]) / 2.0
        it_h = max(1.0, it["rect"][3] - it["rect"][1])
        placed = False
        for row in rows:
            row_yc = row["yc"]
            if abs(it_yc - row_yc) <= it_h * 0.7:
                row["items"].append(it)
                row["yc"] = float(np.mean([(m["rect"][1] + m["rect"][3]) / 2.0 for m in row["items"]]))
                placed = True
                break
        if not placed:
            rows.append({"yc": it_yc, "items": [it]})

    formatted_lines = []
    for row in rows:
        row_items = sorted(row["items"], key=lambda it: it["rect"][0])
        row_text = " ".join(m.get("text", "").strip() for m in row_items if m.get("text", "").strip())
        if row_text:
            formatted_lines.append(row_text)

    return "\n".join(formatted_lines)


def run_ocr(image_bytes, category="food", output_dir=None, test_mode=False, kb=None):
    """
    Executes category-aware OCR analysis on an input image.

    Args:
        image_bytes (bytes | bytearray | np.ndarray | str): Image content as in-memory bytes,
            decoded array, or filepath.
        category (str): Mandatory explicit domain - "food" or "personal_care".
        output_dir (str, optional): Directory to save debug visual artifacts.
        test_mode (bool, optional): Whether to record and save intermediate stages.
        kb (KnowledgeBase, optional): KnowledgeBase instance for fuzzy matching.

    Returns:
        dict: Standardized structured OCR output containing ingredients, nutrition,
              regions, confidence, and metadata.
    """
    t_start = time.time()

    # 1. Validate Category
    if not category or not isinstance(category, str):
        raise ValueError("Category is required and must be 'food' or 'personal_care'.")

    domain = category.strip().lower()
    if domain not in ("food", "personal_care"):
        raise ValueError(f"Invalid category '{category}'. Must be 'food' or 'personal_care'.")

    # 2. Decode / Load Image
    original = load_image(image_bytes)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_image(original, os.path.join(output_dir, "original.jpg"))

    # 3. Quality Assessment
    quality = check_image_quality(original)

    # 4. Normalize Image
    normalized = normalize_image(original)

    # 5. Packet Isolation
    packet_crop, packet_bbox, packet_confidence = detect_packet_region(normalized)
    working_image = (
        packet_crop
        if packet_confidence >= config.PACKET_MIN_CONFIDENCE_TO_CROP
        else normalized
    )

    if output_dir:
        save_image(packet_crop, os.path.join(output_dir, "packet_crop.jpg"))

    # 6. Full Image OCR (PaddleOCR)
    all_items = run_full_image_ocr(working_image, quality=quality)
    all_text_lower = " ".join(it.get("norm_text", "") for it in all_items)

    # 7. Knowledge Base Vocabulary
    if kb is None:
        kb = get_ocr_knowledge_base()

    ingredient_vocab = kb.get_ingredient_names(domain=domain)

    # 8. Unified Document Layout Analysis
    layout_analysis = analyze_document(all_items, working_image.shape, ingredient_vocab=ingredient_vocab)

    # 9. Region Detection (Domain-Specific)
    ingredient_result = detect_ingredient_region(
        layout_analysis, working_image.shape, ingredient_vocab=ingredient_vocab, debug=test_mode
    )

    nutrition_result = {
        "bbox": None,
        "confidence": 0.0,
        "anchor": None,
        "matched_items": [],
        "lines": [],
        "matched_terms": [],
        "method": "skipped_personal_care",
    }

    if domain == "food":
        nutrition_result = detect_nutrition_region(
            layout_analysis, working_image.shape, ingredient_vocab=ingredient_vocab, debug=test_mode
        )

    # 10. Region Reconciliation
    ingredient_result, nutrition_result = reconcile_regions(
        ingredient_result,
        nutrition_result,
        working_image.shape,
        all_lines=layout_analysis.get("lines", []),
        ingredient_vocab=ingredient_vocab,
    )

    # 11. Region Re-OCR & Multi-variant Ensemble
    ing_raw_crop, ing_best_img, ing_ocr = process_region(
        working_image, ingredient_result, "ingredient", "ingredients", output_dir, test_mode
    )

    nut_raw_crop, nut_best_img, nut_ocr = (None, None, None)
    if domain == "food":
        nut_raw_crop, nut_best_img, nut_ocr = process_region(
            working_image, nutrition_result, "nutrition", "nutrition", output_dir, test_mode
        )

    # 12. Parse Ingredients
    ingredients_output = []
    best_ingredient_variant = None
    ing_text = ing_ocr.get("best_text", "") if ing_ocr else ""

    # DEF-03: If re-OCR on the crop missed text that was already detected during full-image OCR
    # (e.g. crop re-OCR only saw the heading "INGREDIENTS:" while matched_items has the ingredient lines),
    # use the matched lines text as a robust fallback/source.
    matched_lines = ingredient_result.get("matched_items", []) or ingredient_result.get("lines", [])
    if matched_lines:
        import re
        matched_text = "\n".join(ln.get("text", "") for ln in matched_lines if ln.get("text"))
        cleaned_ing = re.sub(r"^(?:ingredients?|composition|contents)\s*[:\-\.]?", "", ing_text, flags=re.IGNORECASE).strip()
        cleaned_matched = re.sub(r"^(?:ingredients?|composition|contents)\s*[:\-\.]?", "", matched_text, flags=re.IGNORECASE).strip()
        if len(cleaned_ing) < 15 or len(cleaned_matched) > 2 * len(cleaned_ing):
            ing_text = matched_text

    if ing_text:
        best_ingredient_variant = ing_ocr.get("best_variant") if ing_ocr else "full_image_layout"
        corrector = IngredientCorrector(kb=kb)
        ingredients_output = corrector.correct_and_match(ing_text, domain=domain)
    elif all_text_lower:
        # DEF-01: Fallback if no specific ingredients region crop succeeded,
        # BUT only if all_text_lower actually contains an ingredient heading/anchor.
        # This prevents front-of-pack/marketing text from being parsed as ingredients.
        has_anchor = any(anchor in all_text_lower for anchor in config.ALL_INGREDIENT_ANCHORS)
        if has_anchor:
            parsed_tokens = parse_ingredients(all_text_lower)
            if parsed_tokens and kb:
                matched_kb = kb.match_ingredient_list(parsed_tokens, domain=domain)
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

    # 13. Parse Nutrition (Food Only)
    nutrition_output = None
    best_nutrition_variant = None
    nut_text = nut_ocr.get("best_text", "") if nut_ocr else ""

    matched_nut_lines = nutrition_result.get("matched_items", []) or nutrition_result.get("lines", [])
    reconstructed_nut_text = ""
    if matched_nut_lines:
        reconstructed_nut_text = _reconstruct_nutrition_table_text(matched_nut_lines)

    if domain == "food":
        import re
        has_numbers_nut = bool(re.search(r"\d", nut_text))
        has_numbers_recon = bool(re.search(r"\d", reconstructed_nut_text))
        if (not nut_text) or (not has_numbers_nut and has_numbers_recon) or (len(reconstructed_nut_text) > 1.5 * len(nut_text)):
            nut_text = reconstructed_nut_text

        if nut_ocr and nut_ocr.get("best_items"):
            best_nutrition_variant = nut_ocr.get("best_variant")
            nutrition_output = parse_nutrition(nut_text, nut_ocr)
        elif nut_text:
            best_nutrition_variant = "full_image_layout"
            nutrition_output = parse_nutrition(nut_text)

    # Build other_text from lines not in ingredients or nutrition
    other_lines = []
    ing_lower = ing_text.lower()
    nut_lower = nut_text.lower()
    for it in all_items:
        t = it.get("text", "").strip()
        t_low = t.lower()
        if not t:
            continue
        if t_low in ing_lower or t_low in nut_lower:
            continue
        other_lines.append(t)
    other_text = "\n".join(other_lines)

    elapsed = round(time.time() - t_start, 3)

    return {
        "domain": domain,
        "ingredients": ingredients_output,
        "nutrition": nutrition_output if domain == "food" else None,
        "packet_detection": {
            "bbox": packet_bbox,
            "confidence": round(float(packet_confidence), 3),
        },
        "ingredients_region": {
            "roi_polygon": ingredient_result.get("roi_polygon", []),
            "bbox": ingredient_result.get("bbox"),
            "confidence": round(float(ingredient_result.get("confidence", 0.0)), 3),
            "anchor": ingredient_result.get("anchor"),
            "method": ingredient_result.get("method"),
            "line_count": ingredient_result.get("line_count", 0),
        },
        "nutrition_region": (
            {
                "roi_polygon": nutrition_result.get("roi_polygon", []),
                "bbox": nutrition_result.get("bbox"),
                "confidence": round(float(nutrition_result.get("confidence", 0.0)), 3),
                "anchor": nutrition_result.get("anchor"),
                "method": nutrition_result.get("method"),
                "line_count": nutrition_result.get("line_count", 0),
            }
            if domain == "food"
            else None
        ),
        "raw_text": {
            "all_text": all_text_lower,
            "ingredients_text": ing_text,
            "nutrition_text": nut_text,
            "other_text": other_text,
        },
        "processing": {
            "best_ingredient_variant": best_ingredient_variant,
            "best_nutrition_variant": best_nutrition_variant,
            "elapsed_seconds": elapsed,
        },
        "image_quality": quality,
    }
