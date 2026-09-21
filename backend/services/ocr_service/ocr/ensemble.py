"""
ocr/ensemble.py

Runs OCR across multiple preprocessing variants of an ROI and selects the
best result using a combined score (not just raw OCR confidence), per
STEP 14 of the spec.
"""

import re

from backend.services.ocr_service.ocr.paddle_engine import run_ocr
from backend.services.ocr_service import config


GARBAGE_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9%.,:;()\-\s]")


def _garbage_penalty(text):
    """
    Rough heuristic: fraction of characters that are neither alphanumeric
    nor common punctuation, plus a penalty for very short "words" that
    are pure symbols.
    """
    if not text:
        return 1.0
    junk = len(GARBAGE_CHAR_PATTERN.findall(text))
    return junk / max(len(text), 1)


def _word_count(text):
    return len(re.findall(r"[A-Za-z]{2,}", text))


def _keyword_hits(text_lower, keywords):
    return sum(1 for kw in keywords if kw in text_lower)


def _unit_hits(text_lower):
    return sum(1 for u in config.NUTRIENT_UNITS if u in text_lower)


def _number_hits(text):
    return len(re.findall(r"\d+(\.\d+)?", text))


def join_ocr_items_with_line_boundaries(items):
    """
    Joins OCR items preserving line breaks (\n) between distinct visual lines,
    while joining items on the same visual line with spaces (" ").
    """
    if not items:
        return ""

    if len(items) == 1:
        return items[0].get("text", "").strip()

    parsed = []
    for it in items:
        text = it.get("text", "").strip()
        if not text:
            continue
        bbox = it.get("bbox")
        if bbox and len(bbox) >= 4:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            rect = [min(xs), min(ys), max(xs), max(ys)]
        else:
            rect = it.get("rect", [0, 0, 0, 0])
        h = max(rect[3] - rect[1], 1.0)
        cy = (rect[1] + rect[3]) / 2.0
        parsed.append({
            "text": text,
            "rect": rect,
            "height": h,
            "center_y": cy,
            "min_x": rect[0],
            "min_y": rect[1],
            "max_y": rect[3],
        })

    if not parsed:
        return ""

    import numpy as np
    median_h = float(np.median([p["height"] for p in parsed])) if parsed else 20.0
    y_band = max(median_h * 0.5, 5.0)

    # Sort items by Y band first, then by X
    parsed.sort(key=lambda p: (round(p["min_y"] / y_band), p["min_x"]))

    # Group into lines
    lines = []
    current_line = [parsed[0]["text"]]
    current_min_y = parsed[0]["min_y"]
    current_max_y = parsed[0]["max_y"]
    current_cy = parsed[0]["center_y"]
    current_h = parsed[0]["height"]

    for p in parsed[1:]:
        overlap_y = max(0.0, min(current_max_y, p["max_y"]) - max(current_min_y, p["min_y"]))
        min_h = min(current_h, p["height"])
        v_overlap_ratio = overlap_y / min_h if min_h > 0 else 0.0
        cy_dist = abs(current_cy - p["center_y"])

        if v_overlap_ratio >= 0.40 or cy_dist <= 0.45 * median_h:
            # Same visual line: join with space
            current_line.append(p["text"])
            current_min_y = min(current_min_y, p["min_y"])
            current_max_y = max(current_max_y, p["max_y"])
            current_cy = (current_min_y + current_max_y) / 2.0
            current_h = current_max_y - current_min_y
        else:
            # New visual line: start new line
            lines.append(" ".join(current_line).strip())
            current_line = [p["text"]]
            current_min_y = p["min_y"]
            current_max_y = p["max_y"]
            current_cy = p["center_y"]
            current_h = p["height"]

    if current_line:
        lines.append(" ".join(current_line).strip())

    return "\n".join(lines)


def is_result_sufficiently_complete(items, text, conf, mode, min_conf=0.88):
    """
    Conservative check to determine if an initial OCR variant is strong and complete
    enough to safely short-circuit expensive secondary variant passes.
    Requires:
      1. Confidence >= min_conf
      2. Meaningful token count (len(items) >= 2 and word_count >= 3)
      3. Non-trivial text length
      4. Low garbage character ratio (<= 0.25)
      5. Mode-specific structural signals:
         - For 'ingredient': presence of delimiters (comma, newline, semicolon) or anchor hits
         - For 'nutrition': presence of numbers or nutrient keywords
    """
    if not items or conf < min_conf:
        return False

    text_clean = text.strip()
    if len(text_clean) < 15:
        return False

    words = _word_count(text_clean)
    if words < 3 or len(items) < 2:
        return False

    garbage = _garbage_penalty(text_clean)
    if garbage > 0.12:
        return False

    if mode == "ingredient":
        has_delimiters = ("," in text_clean) or ("\n" in text_clean) or (";" in text_clean)
        has_anchor = _keyword_hits(text_clean.lower(), config.INGREDIENT_ANCHORS) > 0
        if not (has_delimiters or has_anchor or words >= 6):
            return False
    elif mode == "nutrition":
        has_numbers = _number_hits(text_clean) >= 2
        has_keywords = _keyword_hits(text_clean.lower(), config.NUTRIENT_KEYWORDS) > 0
        if not (has_numbers or has_keywords):
            return False

    return True


def score_ocr_items(items, mode):
    """
    mode: "ingredient" | "nutrition" | "generic"

    Returns (score: float, joined_text: str, avg_confidence: float)
    """
    if not items:
        return 0.0, "", 0.0

    joined_text = join_ocr_items_with_line_boundaries(items)
    text_lower = joined_text.lower()
    avg_conf = sum(i["confidence"] for i in items) / len(items)

    words = _word_count(joined_text)
    garbage = _garbage_penalty(joined_text)

    if mode == "ingredient":
        ingredient_kw = _keyword_hits(text_lower, config.INGREDIENT_ANCHORS)
        # reasonable line structure: presence of commas (ingredient lists
        # are typically comma separated)
        comma_bonus = min(joined_text.count(","), 10) * 0.05
        score = (
            avg_conf * 2.0
            + words * 0.05
            + ingredient_kw * 0.3
            + comma_bonus
            - garbage * 1.5
        )
    elif mode == "nutrition":
        nutrient_kw = _keyword_hits(text_lower, config.NUTRIENT_KEYWORDS)
        units = _unit_hits(text_lower)
        numbers = _number_hits(joined_text)
        # crude "table structure" signal: multiple lines each containing a number
        lines_with_numbers = sum(
            1 for i in items if re.search(r"\d", i["text"])
        )
        score = (
            avg_conf * 2.0
            + nutrient_kw * 0.35
            + units * 0.15
            + min(numbers, 20) * 0.05
            + lines_with_numbers * 0.1
            - garbage * 1.5
        )
    else:
        score = avg_conf * 2.0 + words * 0.05 - garbage * 1.5

    return float(score), joined_text, float(avg_conf)


def run_variant_ocr(variants, mode="generic"):
    """
    variants: dict of {variant_name: image_array} (see
              preprocessing.enhancement.preprocess_roi)
    mode: passed through to score_ocr_items

    Returns:
        {
          "best_variant": str or None,
          "best_score": float,
          "best_text": str,
          "best_confidence": float,
          "best_items": [...],
          "all_variants": {
              variant_name: {
                  "score": float, "text": str, "confidence": float,
                  "items": [...]
              }
          }
        }
    """
    from backend.services.ocr_service.preprocessing.image_utils import check_image_quality

    all_results = {}
    best_variant = None
    best_score = float("-inf")

    short_circuit_conf = getattr(config, "ENSEMBLE_SHORT_CIRCUIT_CONFIDENCE", 0.88)

    # 1. Run the "original" variant first
    original_img = variants.get("original")
    if original_img is not None:
        items = run_ocr(original_img)
        score, text, conf = score_ocr_items(items, mode)
        all_results["original"] = {
            "score": score,
            "text": text,
            "confidence": conf,
            "items": items,
        }
        best_score = score
        best_variant = "original"
        
        # Phase 10D: Conservative short-circuit to skip redundant secondary variants
        if is_result_sufficiently_complete(items, text, conf, mode, min_conf=short_circuit_conf):
            return {
                "best_variant": "original",
                "best_score": score,
                "best_text": text,
                "best_confidence": conf,
                "best_items": items,
                "all_variants": all_results,
            }

    # 2. Decide other variants to run based on quality checks of "original"
    if original_img is not None:
        try:
            quality = check_image_quality(original_img)
        except Exception:
            quality = None
    else:
        quality = None

    to_run = []
    if quality:
        if quality.get("is_blurry", False):
            # focus on sharpening variants
            to_run = ["sharpened", "sharpened_threshold"]
        elif quality.get("is_too_dark", False) or quality.get("contrast_std", 100.0) < 45.0:
            # focus on contrast enhancement and thresholding variants
            to_run = ["clahe", "adaptive_threshold", "denoised_clahe"]
        else:
            # balanced subset
            to_run = ["clahe", "sharpened", "adaptive_threshold"]
    else:
        # fallback to all variants
        to_run = [k for k in variants.keys() if k != "original"]

    for name in to_run:
        if name in all_results or name not in variants:
            continue
        img = variants[name]
        items = run_ocr(img)
        score, text, conf = score_ocr_items(items, mode)
        all_results[name] = {
            "score": score,
            "text": text,
            "confidence": conf,
            "items": items,
        }
        if score > best_score:
            best_score = score
            best_variant = name

    if best_variant is None:
        return {
            "best_variant": None,
            "best_score": 0.0,
            "best_text": "",
            "best_confidence": 0.0,
            "best_items": [],
            "all_variants": all_results,
        }

    best = all_results[best_variant]
    return {
        "best_variant": best_variant,
        "best_score": best["score"],
        "best_text": best["text"],
        "best_confidence": best["confidence"],
        "best_items": best["items"],
        "all_variants": all_results,
    }
