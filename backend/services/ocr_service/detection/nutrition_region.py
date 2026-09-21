"""
detection/nutrition_region.py

Locates the FULL Nutrition Information / Facts section, treating it as a table.
Uses OCR line builder, line classifier, block detector, and table heuristics to
determine precise bounds. Operates completely independently of Ingredients.
"""

import numpy as np
from backend.services.ocr_service import config
from backend.services.ocr_service.detection.geometry import (
    union_rect,
    vertical_distance,
    horizontal_overlap_ratio,
    median_line_height,
    sort_reading_order,
    cluster_by_proximity,
    validate_region,
    rect_height,
    rect_width,
)
from backend.services.ocr_service.detection.line_builder import reconstruct_lines
from backend.services.ocr_service.detection.line_classifier import classify_lines
from backend.services.ocr_service.detection.block_detector import detect_logical_blocks
from backend.services.ocr_service.detection.ocr_detector import normalize_ocr_text, find_anchor_candidates, best_anchor_match
from backend.services.ocr_service.detection.section_signals import classify_nutrition_row, classify_ingredient_line, detect_section_boundaries

def find_nutrition_anchor_candidates(lines, top_n=None):
    """
    Finds top-N candidates for Nutrition headings.
    """
    top_n = top_n if top_n is not None else config.NUM_ANCHOR_CANDIDATES_TO_TRY
    candidates = find_anchor_candidates(
        lines, config.NUTRITION_ANCHORS, threshold=config.FUZZY_ANCHOR_THRESHOLD
    )
    return candidates[:top_n]

def _expand_directional(seed_rect, band, ordered_lines, line_h, max_gap, band_tolerance, seed_col=0, seed_angle=0.0, ingredient_vocab=None):
    """
    Greedily walks lines in one direction, absorbing nutrient table rows.
    Validates column consistency, orientation, table boundaries, and stops at competitor sections.
    """
    collected = []
    rejected = []
    current_rect = list(seed_rect)
    band_left, band_right = band
    stop_reason = "exhausted_candidate_lines"
    max_block_height = line_h * config.REGION_MAX_BLOCK_HEIGHT_LINE_FACTOR
    block_start_y = seed_rect[1]
    angle_tol = getattr(config, "ANGLE_COMPATIBILITY_THRESHOLD", 12.0)

    for ln in ordered_lines:
        if len(collected) >= config.REGION_EXPANSION_MAX_LINES:
            stop_reason = "max_lines_reached"
            break

        rect = ln["rect"]
        ln_col = ln.get("column_id", 0)

        # Rule 1: Column compatibility
        if seed_col is not None and ln_col != seed_col:
            # Allow only if there is significant horizontal overlap
            overlap_ratio = horizontal_overlap_ratio((band_left, 0, band_right, 1), (rect[0], 0, rect[2], 1))
            if overlap_ratio < 0.30:
                rejected.append({"text": ln["text"], "reason": "different_column"})
                continue

        # Rule 2: Vertical gap
        gap = vertical_distance(current_rect, rect)
        if gap > max_gap:
            stop_reason = "vertical_gap_exceeded"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Rule 3: Orientation compatibility
        ln_angle = ln.get("angle", 0.0)
        diff_ang = abs(seed_angle - ln_angle)
        if diff_ang > 90.0:
            diff_ang = abs(diff_ang - 180.0)
        if diff_ang > angle_tol * 1.5:
            stop_reason = "incompatible_orientation"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Rule 4: Total block height sanity
        prospective_height = max(rect[3], current_rect[3]) - min(block_start_y, rect[1])
        if prospective_height > max_block_height:
            stop_reason = "max_block_height_exceeded"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Rule 5: Column alignment / band overlap
        overlap_ratio = horizontal_overlap_ratio((band_left, 0, band_right, 1), (rect[0], 0, rect[2], 1))
        is_strictly_left = rect[2] <= band_left + 10
        is_strictly_right = rect[0] >= band_right - 10

        if (is_strictly_left or is_strictly_right) and overlap_ratio < 0.15:
            is_ing_or_other = (
                ln.get("ingredient_score", 0.0) > ln.get("nutrition_score", 0.0)
                or ln.get("predicted_class") in ["ingredients", "other", "manufacturer", "instructions"]
            )
            is_nut_row = (
                not is_ing_or_other
                and (
                    ln.get("predicted_class") == "nutrition"
                    or classify_nutrition_row(ln["text"])["keyword_hits"]
                    or (ln.get("nutrition_score", 0.0) >= 0.25 and classify_nutrition_row(ln["text"])["has_number_unit"])
                )
            )
            if not is_nut_row:
                rejected.append({"text": ln["text"], "reason": "different_horizontal_column"})
                continue

        left_close = abs(rect[0] - band_left) <= band_tolerance
        overlaps_band = overlap_ratio >= config.REGION_BAND_OVERLAP_MIN_RATIO

        if not (left_close or overlaps_band):
            if ln.get("nutrition_score", 0.0) < 0.25 and ln.get("predicted_class") != "nutrition":
                rejected.append({"text": ln["text"], "reason": "outside_table_band"})
                continue

        # Rule 6: STOP CONDITIONS on competitor sections
        norm_text = normalize_ocr_text(ln["text"])

        # Never absorb ingredients into nutrition
        if ln.get("ingredient_score", 0.0) > 0.50 and ln.get("ingredient_score", 0.0) > ln.get("nutrition_score", 0.0):
            stop_reason = "ingredient_line_detected"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Manufacturer stop
        if ln.get("manufacturer_score", 0.0) > 0.50 or any(msig in norm_text for msig in getattr(config, "MANUFACTURER_SIGNALS", [])):
            stop_reason = "manufacturer_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Instructions stop
        if ln.get("instruction_score", 0.0) > 0.50 or any(isig in norm_text for isig in getattr(config, "INSTRUCTION_SIGNALS", [])):
            stop_reason = "instruction_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Contact stop
        if ln.get("contact_score", 0.0) > 0.50 or any(csig in norm_text for csig in getattr(config, "CONTACT_SIGNALS", [])):
            stop_reason = "contact_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Storage stop
        if ln.get("storage_score", 0.0) > 0.50 or any(ssig in norm_text for ssig in getattr(config, "STORAGE_SIGNALS", [])):
            stop_reason = "storage_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # MRP / Date stop
        if ln.get("mrp_score", 0.0) > 0.50 or any(msig in norm_text for msig in getattr(config, "MRP_SIGNALS", [])):
            if not (ln.get("nutrition_score", 0.0) >= 0.20 or classify_nutrition_row(ln["text"])["keyword_hits"] or any(sk in norm_text for sk in ["average", "per 100", "per serve", "serving", "approx"])):
                stop_reason = "mrp_boundary"
                rejected.append({"text": ln["text"], "reason": stop_reason})
                break

        # Legal stop
        if ln.get("legal_score", 0.0) > 0.50 or "fssai" in norm_text or "lic. no" in norm_text:
            stop_reason = "legal_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Stop words: filter out nutrition anchors, nutrient keywords, and serving terms
        nut_stop_words = [
            sw for sw in config.SECTION_STOP_WORDS
            if not any(na in sw for na in config.NUTRITION_ANCHORS)
            and sw not in config.NUTRIENT_KEYWORDS
            and sw not in ["contains", "nutrition", "nutritional", "serving", "servings", "values", "approx"]
        ]
        is_stop, stop_anchor, stop_score, reason = detect_section_boundaries(
            ln["text"], nut_stop_words
        )
        if is_stop:
            if not (ln.get("nutrition_score", 0.0) >= 0.35 or classify_nutrition_row(ln["text"])["keyword_hits"]):
                stop_reason = f"stop_word:{stop_anchor}({reason})"
                rejected.append({"text": ln["text"], "reason": stop_reason})
                break

        if ln.get("other_score", 0.0) > 0.65:
            if not (ln.get("nutrition_score", 0.0) >= 0.25 or classify_nutrition_row(ln["text"])["keyword_hits"] or classify_nutrition_row(ln["text"])["has_number_unit"]):
                stop_reason = "other_section_dominance"
                rejected.append({"text": ln["text"], "reason": stop_reason})
                break

        collected.append(ln)
        current_rect = union_rect([current_rect, rect])
        band_left = min(band_left, rect[0])
        band_right = max(band_right, rect[2])

    return collected, rejected, stop_reason, (band_left, band_right)

def cluster_semantic_lines_nutrition(lines, max_gap_y, band_tolerance):
    """
    Groups lines with high nutrition probability into spatial clusters.
    """
    candidates = [ln for ln in lines if ln.get("scores", {}).get("nutrition", 0.0) >= 0.20]
    if not candidates:
        return []
        
    n = len(candidates)
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
        
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            
    for i in range(n):
        rect_i = candidates[i]["rect"]
        for j in range(i + 1, n):
            rect_j = candidates[j]["rect"]
            
            # Check vertical distance
            v_gap = vertical_distance(rect_i, rect_j)
            if v_gap > max_gap_y:
                continue
                
            # Check horizontal alignment
            left_close = abs(rect_i[0] - rect_j[0]) <= band_tolerance
            overlap = horizontal_overlap_ratio(rect_i, rect_j) >= config.REGION_BAND_OVERLAP_MIN_RATIO
            
            if left_close or overlap:
                union(i, j)
                
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(candidates[i])
        
    clusters = []
    for g in groups.values():
        clusters.append(sorted(g, key=lambda ln: (ln["rect"][1], ln["rect"][0])))
    return clusters


def _spatial_semantic_fallback_candidate_nutrition(lines, max_gap_y, band_tolerance, image_shape):
    """
    Fallback using spatial-semantic clustering when no anchor is found.
    """
    clusters = cluster_semantic_lines_nutrition(lines, max_gap_y, band_tolerance)
    if not clusters:
        return None
        
    scored = []
    for c_lines in clusters:
        if len(c_lines) < 2:
            continue
            
        matched_terms = set()
        for ln in c_lines:
            matched_terms.update(classify_nutrition_row(ln["text"])["keyword_hits"])
            
        # Require at least 2 distinct nutrient keyword hits to be a valid fallback table
        if len(matched_terms) < 2:
            continue
            
        avg_nut = float(np.mean([ln.get("scores", {}).get("nutrition", 0.0) for ln in c_lines]))
        if avg_nut < 0.35:
            continue
            
        confidence = min(0.75, 0.3 + 0.45 * avg_nut)
        if confidence < 0.55:
            continue
            
        bbox = union_rect([ln["rect"] for ln in c_lines])
        
        scored.append({
            "bbox": bbox,
            "confidence": round(float(confidence), 3),
            "anchor": None,
            "matched_items": c_lines,
            "matched_terms": sorted(matched_terms),
            "method": "spatial_semantic_fallback",
            "debug": {"num_lines_collected": len(c_lines), "stop_reason": "spatial_clustering"}
        })
        
    scored.sort(key=lambda x: x["confidence"], reverse=True)
    return scored[0] if scored else None


def _absorb_table_columns(collected, all_lines, line_h):
    """
    In multi-column nutrition tables, value cells (e.g. '160', '10g', '<0.4g')
    may be placed in a separate OCR column to the right of nutrient labels.
    Absorbs lines from all_lines that:
    - Vertically overlap with the collected nutrient rows (or table vertical span)
    - Are horizontally aligned/to the right of the table within reasonable table width
    - Do not belong to competitor sections (ingredients, manufacturer, instructions, legal, etc.)
    """
    if not collected:
        return collected

    y_min = min(ln["rect"][1] for ln in collected)
    y_max = max(ln["rect"][3] for ln in collected)
    x_min = min(ln["rect"][0] for ln in collected)
    x_max = max(ln["rect"][2] for ln in collected)

    absorbed = []
    collected_ids = set(id(ln) for ln in collected)

    for ln in all_lines:
        if id(ln) in collected_ids:
            continue
        r = ln["rect"]
        # Must fall within the vertical extent of the table (with a small margin)
        if r[1] >= y_min - line_h * 0.75 and r[3] <= y_max + line_h * 0.75:
            # Must be within horizontal proximity to the table
            if r[0] >= x_min - line_h and r[0] <= x_max + line_h * 35:
                # Exclude competitor sections
                if ln.get("ingredient_score", 0.0) >= 0.50:
                    continue
                if ln.get("manufacturer_score", 0.0) >= 0.50:
                    continue
                if ln.get("instruction_score", 0.0) >= 0.50:
                    continue
                if ln.get("contact_score", 0.0) >= 0.50:
                    continue
                if ln.get("storage_score", 0.0) >= 0.50:
                    continue
                norm_text = normalize_ocr_text(ln["text"])
                if any(sw in norm_text for sw in ["ingredients:", "ingredients", "mfg", "fssai", "batch"]):
                    continue
                absorbed.append(ln)

    return collected + absorbed


def expand_nutrition_region(anchor_line, all_lines, image_shape, ingredient_vocab=None):
    """
    Expands the nutrition region from the anchor heading.
    """
    anchor_rect = anchor_line["rect"]
    others = [ln for ln in all_lines if ln is not anchor_line]
    below = [ln for ln in others if ln["rect"][1] >= anchor_rect[1] - 1e-6]
    ordered = sort_reading_order(below)

    line_h = median_line_height([ln["rect"] for ln in all_lines]) or 20.0
    max_gap = line_h * config.REGION_EXPANSION_MAX_LINE_GAP_FACTOR
    band_tolerance = line_h * config.REGION_BAND_LEFT_TOLERANCE_FACTOR

    seed_col = anchor_line.get("column_id", 0)
    seed_angle = anchor_line.get("angle", 0.0)

    collected_rest, rejected, stop_reason, band = _expand_directional(
        anchor_rect, (anchor_rect[0], anchor_rect[2]), ordered, line_h, max_gap, band_tolerance,
        seed_col=seed_col, seed_angle=seed_angle,
        ingredient_vocab=ingredient_vocab,
    )

    collected = [anchor_line] + collected_rest
    collected = _absorb_table_columns(collected, all_lines, line_h)
    bbox = union_rect([ln["rect"] for ln in collected])

    debug_info = {
        "stop_reason": stop_reason,
        "rejected_lines": rejected,
        "num_lines_collected": len(collected),
        "column_band": list(band),
    }
    return bbox, collected, debug_info

def _table_geometry_score(cluster_lines):
    """
    Evaluates how much a cluster of lines resembles a nutrition table.
    """
    if len(cluster_lines) < 2:
        return 0.0

    rects = [ln["rect"] for ln in cluster_lines]
    heights = [rect_height(r) for r in rects]
    widths = [rect_width(r) for r in rects]

    mean_h = float(np.mean(heights)) if heights else 1.0
    std_h = float(np.std(heights)) if heights else 0.0
    height_consistency = max(0.0, 1.0 - min(std_h / max(mean_h, 1e-6), 1.0))

    mean_w = float(np.mean(widths)) if widths else 1.0
    shortness = 1.0 if mean_w < 600 else max(0.0, 1.0 - (mean_w - 600) / 1000.0)

    bbox = union_rect(rects)
    bbox_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    items_area = sum(rect_width(r) * rect_height(r) for r in rects)
    compactness = min(1.0, items_area / bbox_area * 3.0)

    return float(0.4 * height_consistency + 0.3 * shortness + 0.3 * compactness)

def detect_nutrition_rows(lines, image_shape, ingredient_vocab=None):
    """
    Fallback table-geometry region builder if no nutrition facts header is found.
    """
    hits = []
    for ln in lines:
        classification = classify_nutrition_row(ln["text"])
        if classification["keyword_hits"]:
            hits.append(ln)

    if len(hits) < config.NUTRITION_TABLE_MIN_KEYWORDS:
        return None

    clusters = cluster_by_proximity(
        hits, max_gap_factor=config.NUTRITION_TABLE_CLUSTER_DIST_FACTOR
    )

    best_cluster = None
    best_score = 0.0
    best_terms = set()

    for cluster in clusters:
        distinct_terms = set()
        for ln in cluster:
            distinct_terms.update(classify_nutrition_row(ln["text"])["keyword_hits"])

        if len(distinct_terms) < config.NUTRITION_TABLE_MIN_KEYWORDS:
            continue

        table_score = _table_geometry_score(cluster)
        term_score = min(1.0, len(distinct_terms) / 8.0)
        combined = 0.55 * term_score + 0.45 * table_score

        if combined > best_score:
            best_score = combined
            best_cluster = cluster
            best_terms = distinct_terms

    if best_cluster is None:
        return None

    seed_line = sort_reading_order(best_cluster)[0]
    line_h = median_line_height([ln["rect"] for ln in lines]) or 20.0
    max_gap = line_h * config.REGION_EXPANSION_MAX_LINE_GAP_FACTOR
    band_tolerance = line_h * config.REGION_BAND_LEFT_TOLERANCE_FACTOR
    header_gap = line_h * config.NUTRITION_HEADER_FOOTER_GAP_FACTOR

    seed_rect = seed_line["rect"]
    seed_col = seed_line.get("column_id", 0)
    seed_angle = seed_line.get("angle", 0.0)
    others = [ln for ln in lines if ln is not seed_line]

    below = sort_reading_order([ln for ln in others if ln["rect"][1] >= seed_rect[1] - 1e-6])
    above = sort_reading_order([ln for ln in others if ln["rect"][1] < seed_rect[1] - 1e-6])[::-1]

    down_collected, down_rejected, down_stop, band_after_down = _expand_directional(
        seed_rect, (seed_rect[0], seed_rect[2]), below, line_h, max_gap, band_tolerance,
        seed_col=seed_col, seed_angle=seed_angle,
        ingredient_vocab=ingredient_vocab,
    )
    up_collected, up_rejected, up_stop, band_after_up = _expand_directional(
        seed_rect, (seed_rect[0], seed_rect[2]), above, line_h, header_gap, band_tolerance,
        seed_col=seed_col, seed_angle=seed_angle,
        ingredient_vocab=ingredient_vocab,
    )

    collected = list(reversed(up_collected)) + [seed_line] + down_collected
    collected = _absorb_table_columns(collected, lines, line_h)
    bbox = union_rect([ln["rect"] for ln in collected])

    confidence = round(float(min(0.9, best_score)), 3)

    debug_info = {
        "stop_reason_down": down_stop,
        "stop_reason_up": up_stop,
        "rejected_lines": up_rejected + down_rejected,
        "num_lines_collected": len(collected),
        "column_band": [
            min(band_after_down[0], band_after_up[0]),
            max(band_after_down[1], band_after_up[1]),
        ],
    }

    return {
        "bbox": bbox,
        "confidence": confidence,
        "anchor": None,
        "matched_items": collected,
        "lines": collected,
        "matched_terms": sorted(best_terms),
        "method": "table_geometry_fallback",
        "debug": debug_info,
    }

def score_nutrition_region(collected_lines, anchor_score, stop_reason, ingredient_vocab=None):
    """
    Computes confidence score for the nutrition facts table region.
    """
    if not collected_lines:
        return 0.0, []

    matched_terms = set()
    for ln in collected_lines:
        matched_terms.update(classify_nutrition_row(ln["text"])["keyword_hits"])

    avg_nut_score = float(np.mean([ln.get("nutrition_score", 0.0) for ln in collected_lines]))

    line_bonus = min(0.3, 0.04 * max(0, len(collected_lines) - 1))
    term_bonus = min(0.2, 0.03 * len(matched_terms))

    boundary_bonus = 0.1 if stop_reason and (stop_reason.startswith("stop_word") or stop_reason == "ingredient_line_detected") else 0.03

    contamination_penalty = 0.0
    for ln in collected_lines:
        if ln.get("ingredient_score", 0.0) > 0.6:
            contamination_penalty += 0.15
        if ln.get("other_score", 0.0) > 0.75:
            contamination_penalty += 0.1

    mean_ocr_confidence = float(np.mean([ln.get("confidence", 0.0) for ln in collected_lines]))
    ocr_confidence_bonus = mean_ocr_confidence * 0.10

    confidence = (
        (anchor_score / 100.0) * 0.4
        + avg_nut_score * 0.25
        + line_bonus
        + term_bonus
        + boundary_bonus
        + ocr_confidence_bonus
        - contamination_penalty
    )
    return round(float(max(0.0, min(0.98, confidence))), 3), sorted(matched_terms)

def _vocabulary_fallback_candidate_v2(blocks):
    """
    Fallback candidate from block detector: returns best NUTRITION_TABLE_BLOCK.
    """
    nut_blocks = [b for b in blocks if b["type"] == "NUTRITION_TABLE_BLOCK"]
    if not nut_blocks:
        return None

    scored = []
    for b in nut_blocks:
        lines = b["lines"]
        avg_nut = float(np.mean([ln.get("nutrition_score", 0.0) for ln in lines]))
        confidence = min(0.7, 0.3 + 0.4 * avg_nut)

        matched_terms = set()
        for ln in lines:
            matched_terms.update(classify_nutrition_row(ln["text"])["keyword_hits"])

        scored.append({
            "bbox": b["rect"],
            "confidence": round(float(confidence), 3),
            "anchor": None,
            "matched_items": lines,
            "matched_terms": sorted(matched_terms),
            "method": "table_geometry_fallback",
            "debug": {"num_lines_collected": len(lines), "stop_reason": "block_cluster"}
        })

    scored.sort(key=lambda x: x["confidence"], reverse=True)
    return scored[0] if scored else None

def detect_nutrition_region(layout_analysis, image_shape, ingredient_vocab=None, debug=False):
    """
    Main entry point for Nutrition Facts Table region detection. Supports consuming unified document layout analysis.
    """
    # 1. Backward-compatibility / Direct call support
    if not isinstance(layout_analysis, dict) or "lines" not in layout_analysis:
        from backend.services.ocr_service.detection.document_layout import analyze_document
        layout_analysis = analyze_document(layout_analysis, image_shape, ingredient_vocab)

    include_debug = debug or config.DEBUG_REGION_DETECTION
    lines = layout_analysis["lines"]

    if not lines:
        return {
            "bbox": None, "confidence": 0.0, "anchor": None,
            "matched_items": [], "lines": [], "matched_terms": [], "method": "none",
        }

    line_h = median_line_height([ln["rect"] for ln in lines]) or 20.0
    max_gap_y = line_h * config.REGION_EXPANSION_MAX_LINE_GAP_FACTOR
    band_tolerance = line_h * config.REGION_BAND_LEFT_TOLERANCE_FACTOR

    scored_candidates = []

    # Step 1: Heading anchor candidates (PRIMARY PATH)
    anchor_candidates = find_nutrition_anchor_candidates(lines)
    for cand in anchor_candidates:
        anchor_line = cand["line"]
        anchor_text = cand["matched_anchor"]
        anchor_score = cand["score"]

        bbox, collected, debug_info = expand_nutrition_region(
            anchor_line, lines, image_shape, ingredient_vocab=ingredient_vocab
        )
        if not collected:
            continue
        confidence, matched_terms = score_nutrition_region(
            collected, anchor_score, debug_info["stop_reason"], ingredient_vocab
        )
        debug_info["anchor_text"] = anchor_line["text"]
        debug_info["anchor_score"] = anchor_score

        scored_candidates.append({
            "bbox": bbox,
            "confidence": confidence,
            "anchor": anchor_text,
            "matched_items": collected,
            "lines": collected,
            "matched_terms": matched_terms,
            "method": "anchor_expansion",
            "debug": debug_info,
        })

    # Fallbacks are only added if no strong anchor candidate was found
    has_strong_anchor = any(c.get("method") == "anchor_expansion" and c["confidence"] >= 0.65 for c in scored_candidates)
    if not has_strong_anchor:
        # Step 2: Table geometry fallback (keyword hits like protein, fat, carb, energy)
        fallback = detect_nutrition_rows(lines, image_shape, ingredient_vocab=ingredient_vocab)
        if fallback is not None:
            scored_candidates.append(fallback)

        # Step 3: Block fallback
        blocks = layout_analysis.get("blocks", [])
        block_fallback = _vocabulary_fallback_candidate_v2(blocks)
        if block_fallback is not None:
            scored_candidates.append(block_fallback)

        # Step 4: Spatial semantic fallback
        spatial_candidate = _spatial_semantic_fallback_candidate_nutrition(lines, max_gap_y, band_tolerance, image_shape)
        if spatial_candidate is not None:
            scored_candidates.append(spatial_candidate)

    # Step 5: Layout analysis candidates
    candidates = [c for c in layout_analysis.get("section_candidates", []) if c["type"] == "nutrition"]
    for c in candidates:
        matched_terms = set()
        for ln in c["lines"]:
            matched_terms.update(classify_nutrition_row(ln["text"])["keyword_hits"])
        scored_candidates.append({
            "bbox": c["bbox"],
            "confidence": c["final_score"],
            "anchor": None,
            "matched_items": c["lines"],
            "lines": c["lines"],
            "matched_terms": sorted(matched_terms),
            "method": c["method"],
            "debug": {"num_lines_collected": len(c["lines"])},
        })

    if not scored_candidates:
        return {
            "bbox": None, "confidence": 0.0, "anchor": None,
            "matched_items": [], "lines": [], "matched_terms": [], "method": "none",
        }

    scored_candidates.sort(key=lambda c: c["confidence"], reverse=True)
    best = scored_candidates[0]
    best_bbox = validate_region(best["bbox"], image_shape)
    best["bbox"] = best_bbox

    if include_debug:
        best["debug"] = best.get("debug", {})
        best["debug"]["all_candidate_scores"] = [
            {"method": c["method"], "anchor": c.get("anchor"), "confidence": c["confidence"]}
            for c in scored_candidates
        ]
    else:
        best.pop("debug", None)

    return best
