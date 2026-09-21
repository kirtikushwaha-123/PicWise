"""
detection/ingredient_region.py

Locates the FULL Ingredients / Composition section of a package image.
Uses OCR line reconstruction, block detection, and semantic line scoring to
ensure a tight bounding box that excludes unrelated surrounding text.
"""

import numpy as np
from backend.services.ocr_service import config
from backend.services.ocr_service.detection.geometry import (
    union_rect,
    vertical_distance,
    horizontal_overlap_ratio,
    median_line_height,
    sort_reading_order,
    validate_region,
)
from backend.services.ocr_service.detection.line_builder import reconstruct_lines
from backend.services.ocr_service.detection.line_classifier import classify_lines
from backend.services.ocr_service.detection.block_detector import detect_logical_blocks
from backend.services.ocr_service.detection.ocr_detector import normalize_ocr_text, find_anchor_candidates, best_anchor_match
from backend.services.ocr_service.detection.section_signals import classify_nutrition_row, detect_section_boundaries

def split_merged_lines(lines, image_width):
    """
    Splits any line that contains a horizontal gap wider than the threshold.
    """
    split_lines = []
    line_h = median_line_height([ln["rect"] for ln in lines]) or 20.0
    gap_threshold = max(20.0, min(image_width * 0.025, line_h * 0.75))
    
    for ln in lines:
        items = ln.get("items", [])
        if len(items) <= 1:
            split_lines.append(ln)
            continue
            
        sorted_items = sorted(items, key=lambda it: it["rect"][0])
        
        current_group = [sorted_items[0]]
        groups = [current_group]
        
        for it in sorted_items[1:]:
            prev_it = current_group[-1]
            gap = it["rect"][0] - prev_it["rect"][2]
            if gap > gap_threshold:
                current_group = [it]
                groups.append(current_group)
            else:
                current_group.append(it)
                
        if len(groups) == 1:
            split_lines.append(ln)
        else:
            for gp in groups:
                merged_text = " ".join(m["text"] for m in gp)
                merged_rect = union_rect([m["rect"] for m in gp])
                avg_conf = float(np.mean([m.get("confidence", 0.0) for m in gp]))
                
                new_ln = dict(ln)
                new_ln["text"] = merged_text
                new_ln["rect"] = merged_rect
                new_ln["items"] = gp
                new_ln["child_items"] = gp
                new_ln["confidence"] = avg_conf
                
                x1, y1, x2, y2 = merged_rect
                new_ln["center"] = [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
                new_ln["height"] = float(y2 - y1)
                new_ln["width"] = float(x2 - x1)
                
                # Recompute polygon and child polygons strictly for this group
                child_polys = [m["polygon"] for m in gp if m.get("polygon") is not None]
                new_ln["polygons"] = child_polys
                if len(gp) == 1 and gp[0].get("polygon") is not None:
                    new_ln["polygon"] = gp[0]["polygon"]
                else:
                    upper = []
                    lower = []
                    for m in gp:
                        p = m.get("polygon")
                        if p is not None and len(p) >= 4:
                            upper.extend([p[0], p[1]])
                            lower.extend([p[2], p[3]])
                        else:
                            rx1, ry1, rx2, ry2 = m["rect"]
                            upper.extend([[rx1, ry1], [rx2, ry1]])
                            lower.extend([[rx2, ry2], [rx1, ry2]])
                    new_ln["polygon"] = upper + list(reversed(lower)) if upper else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                # Copy semantic scores if present
                for key in ["ingredient_score", "nutrition_score", "other_score"]:
                    if key in ln:
                        new_ln[key] = ln[key]
                        
                split_lines.append(new_ln)
                
    return split_lines

def find_ingredient_anchor_candidates(lines, top_n=None):
    """
    Finds lines that look like Ingredients headings.
    """
    top_n = top_n if top_n is not None else config.NUM_ANCHOR_CANDIDATES_TO_TRY
    candidates = find_anchor_candidates(
        lines, config.ALL_INGREDIENT_ANCHORS, threshold=config.FUZZY_ANCHOR_THRESHOLD
    )
    if not candidates:
        return []

    # Filter out lines that are clearly instructions or MRP/Batch
    valid_cands = []
    for c in candidates:
        ln = c["line"]
        if ln.get("instruction_score", 0.0) >= 0.50 or ln.get("mrp_score", 0.0) >= 0.50:
            continue
        valid_cands.append(c)

    # Sort: strong anchors first, weak ("contains") later
    strong = [c for c in valid_cands if c["matched_anchor"] != "contains"]
    weak = [c for c in valid_cands if c["matched_anchor"] == "contains"]
    return (strong + weak)[:top_n]

def cluster_semantic_lines(lines, target_class, max_gap_y, band_tolerance):
    """
    Groups lines with high target_class probability into spatial clusters.
    """
    candidates = [ln for ln in lines if ln.get("scores", {}).get(target_class, 0.0) >= 0.20]
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


def _spatial_semantic_fallback_candidate(lines, max_gap_y, band_tolerance, image_shape):
    """
    Fallback using spatial-semantic clustering when no anchor is found.
    """
    clusters = cluster_semantic_lines(lines, "ingredients", max_gap_y, band_tolerance)
    if not clusters:
        return None
        
    scored = []
    for c_lines in clusters:
        if len(c_lines) < 2:
            continue
        avg_ing = float(np.mean([ln.get("scores", {}).get("ingredients", 0.0) for ln in c_lines]))
        if avg_ing < 0.35:
            continue
            
        confidence = min(0.75, 0.3 + 0.45 * avg_ing)
        if confidence < 0.55:
            continue
            
        bbox = union_rect([ln["rect"] for ln in c_lines])
        
        scored.append({
            "bbox": bbox,
            "confidence": round(float(confidence), 3),
            "anchor": None,
            "matched_items": c_lines,
            "method": "spatial_semantic_fallback",
            "debug": {"num_lines_collected": len(c_lines), "stop_reason": "spatial_clustering"}
        })
        
    scored.sort(key=lambda x: x["confidence"], reverse=True)
    return scored[0] if scored else None


def expand_ingredient_region(anchor_line, all_lines, image_shape, anchor_column_lines=None):
    """
    Expands the ingredient region from the anchor heading, walking line-by-line
    and validating column, orientation, line spacing, and semantic evidence.
    Immediately stops at foreign headings (allergen, instructions, manufacturer, etc.).
    """
    anchor_rect = anchor_line["rect"]
    anchor_col = anchor_line.get("column_id", 0)
    anchor_angle = anchor_line.get("angle", 0.0)

    if anchor_column_lines is not None:
        col_set = {id(ln) for ln in anchor_column_lines}
        others = [ln for ln in all_lines if ln is not anchor_line and id(ln) in col_set]
    else:
        # Filter lines to same column or nearby compatible column
        others = [
            ln for ln in all_lines
            if ln is not anchor_line and ln.get("column_id", 0) == anchor_col
        ]
        if not others:
            others = [ln for ln in all_lines if ln is not anchor_line]

    ordered = sort_reading_order(others)

    line_h = median_line_height([ln["rect"] for ln in all_lines]) or 20.0
    max_gap = line_h * config.REGION_EXPANSION_MAX_LINE_GAP_FACTOR
    band_tolerance = line_h * config.REGION_BAND_LEFT_TOLERANCE_FACTOR
    max_block_height = line_h * config.REGION_MAX_BLOCK_HEIGHT_LINE_FACTOR
    angle_tol = getattr(config, "ANGLE_COMPATIBILITY_THRESHOLD", 12.0)

    collected = [anchor_line]
    current_rect = list(anchor_rect)
    band_left, band_right = anchor_rect[0], anchor_rect[2]

    rejected = []
    stop_reason = "exhausted_candidate_lines"

    for ln in ordered:
        if len(collected) >= config.REGION_EXPANSION_MAX_LINES:
            stop_reason = "max_lines_reached"
            break

        rect = ln["rect"]

        # 1. Skip lines positioned above the anchor
        if rect[1] < anchor_rect[1] - line_h * 0.3:
            rejected.append({"text": ln["text"], "reason": "above_anchor"})
            continue

        # 2. Check vertical gap to currently collected region
        gap = vertical_distance(current_rect, rect)
        if gap > max_gap:
            stop_reason = "vertical_gap_exceeded"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # 3. Orientation compatibility
        ln_angle = ln.get("angle", 0.0)
        diff_ang = abs(anchor_angle - ln_angle)
        if diff_ang > 90.0:
            diff_ang = abs(diff_ang - 180.0)
        if diff_ang > angle_tol * 1.5:
            stop_reason = "incompatible_orientation"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # 4. Total height sanity
        prospective_height = max(rect[3], current_rect[3]) - min(anchor_rect[1], rect[1])
        if prospective_height > max_block_height:
            stop_reason = "max_block_height_exceeded"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # 5. Check column alignment or band overlap
        overlap_ratio = horizontal_overlap_ratio((band_left, 0, band_right, 1), (rect[0], 0, rect[2], 1))
        is_strictly_left = rect[2] <= band_left + 10
        is_strictly_right = rect[0] >= band_right - 10

        if (is_strictly_left or is_strictly_right) and overlap_ratio < 0.15:
            rejected.append({"text": ln["text"], "reason": "different_horizontal_column"})
            continue

        left_close = abs(rect[0] - band_left) <= band_tolerance
        overlaps_band = overlap_ratio >= config.REGION_BAND_OVERLAP_MIN_RATIO

        if not (left_close or overlaps_band):
            if ln.get("ingredient_score", 0.0) < 0.35:
                rejected.append({"text": ln["text"], "reason": "outside_paragraph_band"})
                continue

        # 6. STOP CONDITIONS: Enforce strict semantic boundaries
        norm_text = normalize_ocr_text(ln["text"])

        # Nutrition row / heading stop
        if ln.get("nutrition_score", 0.0) > 0.55 or classify_nutrition_row(ln["text"])["is_strong_nutrition_row"]:
            stop_reason = "nutrition_detected"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Instructions stop (e.g. Pump Activation Procedure, Directions)
        if ln.get("instruction_score", 0.0) > 0.50 or any(isig in norm_text for isig in getattr(config, "INSTRUCTION_SIGNALS", [])):
            stop_reason = "instruction_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Allergen advice stop (unless config allows)
        if not getattr(config, "INCLUDE_ALLERGEN_IN_INGREDIENTS", False):
            is_statutory = any(term in norm_text for term in ["colour", "color", "flavour", "flavor", "caffeine"])
            if not is_statutory:
                if ln.get("allergen_score", 0.0) > 0.50 or any(asig in norm_text for asig in getattr(config, "ALLERGEN_SIGNALS", [])):
                    stop_reason = "allergen_boundary"
                    rejected.append({"text": ln["text"], "reason": stop_reason})
                    break

        # Manufacturer stop
        if ln.get("manufacturer_score", 0.0) > 0.50 or any(msig in norm_text for msig in getattr(config, "MANUFACTURER_SIGNALS", [])):
            stop_reason = "manufacturer_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Storage stop
        if ln.get("storage_score", 0.0) > 0.50 or any(ssig in norm_text for ssig in getattr(config, "STORAGE_SIGNALS", [])):
            stop_reason = "storage_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Contact / Helpline stop
        if ln.get("contact_score", 0.0) > 0.50 or any(csig in norm_text for csig in getattr(config, "CONTACT_SIGNALS", [])):
            stop_reason = "contact_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # MRP / Batch / Date stop
        if ln.get("mrp_score", 0.0) > 0.50 or any(msig in norm_text for msig in getattr(config, "MRP_SIGNALS", [])):
            stop_reason = "mrp_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Legal / Regulatory stop
        if ln.get("legal_score", 0.0) > 0.50 or "fssai" in norm_text or "lic. no" in norm_text or "lic.no" in norm_text or "mfg.lic" in norm_text:
            stop_reason = "legal_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Marketing stop
        if ln.get("marketing_score", 0.0) > 0.50 or any(msig in norm_text for msig in getattr(config, "MARKETING_SIGNALS", [])):
            stop_reason = "marketing_boundary"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Check generic stop word headings
        is_stop, stop_anchor, stop_score, reason = detect_section_boundaries(
            ln["text"], config.SECTION_STOP_WORDS
        )
        if is_stop:
            stop_reason = f"stop_word:{stop_anchor}({reason})"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # Check other section score dominance
        if ln.get("other_score", 0.0) > 0.65:
            stop_reason = "other_section_dominance"
            rejected.append({"text": ln["text"], "reason": stop_reason})
            break

        # If everything passes, line belongs to the ingredient continuation block
        collected.append(ln)
        current_rect = union_rect([current_rect, rect])
        band_left = min(band_left, rect[0])
        band_right = max(band_right, rect[2])

    bbox = union_rect([ln["rect"] for ln in collected])
    debug_info = {
        "stop_reason": stop_reason,
        "rejected_lines": rejected,
        "num_lines_collected": len(collected),
        "column_band": [band_left, band_right],
    }
    return bbox, collected, debug_info

def score_ingredient_region(collected_lines, anchor_score, stop_reason, ingredient_vocab=None):
    """
    Computes a region confidence score based on line count, semantic scores,
    vocabulary density, and clean boundaries.
    """
    if not collected_lines:
        return 0.0

    avg_ing_score = float(np.mean([ln.get("ingredient_score", 0.0) for ln in collected_lines]))
    
    line_count_bonus = min(0.30, 0.04 * max(0, len(collected_lines) - 1))
    
    vocab_bonus = 0.0
    if ingredient_vocab:
        vocab_terms = [v.lower() for v in ingredient_vocab if len(v) >= 4]
        if vocab_terms:
            hits = 0
            for ln in collected_lines:
                norm = normalize_ocr_text(ln["text"])
                if any(term in norm for term in vocab_terms):
                    hits += 1
            vocab_bonus = min(0.15, (hits / len(collected_lines)) * 0.15)

    boundary_bonus = 0.12 if stop_reason and (stop_reason.startswith("stop_word") or stop_reason == "nutrition_row_detected") else 0.04

    contamination_penalty = 0.0
    for ln in collected_lines:
        if ln.get("nutrition_score", 0.0) > 0.6:
            contamination_penalty += 0.15
        if ln.get("other_score", 0.0) > 0.7:
            contamination_penalty += 0.1

    mean_ocr_confidence = float(np.mean([ln.get("confidence", 0.0) for ln in collected_lines]))
    ocr_confidence_bonus = mean_ocr_confidence * 0.10

    confidence = (
        (anchor_score / 100.0) * 0.4
        + avg_ing_score * 0.25
        + line_count_bonus
        + vocab_bonus
        + boundary_bonus
        + ocr_confidence_bonus
        - contamination_penalty
    )
    return round(float(max(0.0, min(0.98, confidence))), 3)

def _vocabulary_fallback_candidate(blocks, ingredient_vocab):
    """
    Fallback when no heading anchor is found: returns the best scoring INGREDIENT_BLOCK.
    """
    ingredient_blocks = [b for b in blocks if b["type"] == "INGREDIENT_BLOCK"]
    if not ingredient_blocks:
        return None

    # Score blocks by density of ingredients and comma count
    scored = []
    for b in ingredient_blocks:
        lines = b["lines"]
        avg_ing = float(np.mean([ln.get("ingredient_score", 0.0) for ln in lines]))
        confidence = min(0.7, 0.3 + 0.4 * avg_ing)

        scored.append({
            "bbox": b["rect"],
            "confidence": round(float(confidence), 3),
            "anchor": None,
            "matched_items": lines,
            "method": "vocabulary_fallback",
            "debug": {"num_lines_collected": len(lines), "stop_reason": "block_cluster"}
        })

    scored.sort(key=lambda x: x["confidence"], reverse=True)
    return scored[0] if scored else None

def detect_ingredient_region(layout_analysis, image_shape, ingredient_vocab=None, debug=False):
    """
    Main entry point for Ingredients Region Detection.
    Begins from a strong seed (heading anchor), performs line-by-line continuation walk,
    and applies fallbacks only when no heading exists.
    """
    if not isinstance(layout_analysis, dict) or "lines" not in layout_analysis:
        from backend.services.ocr_service.detection.document_layout import analyze_document
        layout_analysis = analyze_document(layout_analysis, image_shape, ingredient_vocab)

    include_debug = debug or config.DEBUG_REGION_DETECTION
    lines = layout_analysis["lines"]

    if not lines:
        return {
            "bbox": None, "confidence": 0.0, "anchor": None,
            "matched_items": [], "method": "none",
        }

    h, w = image_shape[:2]
    line_h = median_line_height([ln["rect"] for ln in lines]) or 20.0
    max_gap_y = line_h * config.REGION_EXPANSION_MAX_LINE_GAP_FACTOR
    band_tolerance = line_h * config.REGION_BAND_LEFT_TOLERANCE_FACTOR

    # Step 1: Search for strong heading anchor candidates (PRIMARY PATH)
    anchor_candidates = find_ingredient_anchor_candidates(lines)
    scored_candidates = []

    for cand in anchor_candidates:
        anchor_line = cand["line"]
        anchor_text = cand["matched_anchor"]
        anchor_score = cand["score"]

        # Run strict continuation walk
        bbox, collected, debug_info = expand_ingredient_region(
            anchor_line, lines, image_shape
        )
        if not collected:
            continue

        confidence = score_ingredient_region(
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
            "method": "anchor_expansion",
            "debug": debug_info,
        })

    # Step 2: If no heading anchor found, check fallback candidates (vocabulary / logical block)
    if not scored_candidates:
        blocks = layout_analysis.get("blocks", [])
        block_fallback = _vocabulary_fallback_candidate(blocks, ingredient_vocab)
        if block_fallback is not None and block_fallback["confidence"] >= 0.50:
            scored_candidates.append(block_fallback)

        spatial_fallback = _spatial_semantic_fallback_candidate(lines, max_gap_y, band_tolerance, image_shape)
        if spatial_fallback is not None and spatial_fallback["confidence"] >= 0.55:
            scored_candidates.append(spatial_fallback)

    if not scored_candidates:
        return {
            "bbox": None, "confidence": 0.0, "anchor": None,
            "matched_items": [], "lines": [], "method": "none",
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
