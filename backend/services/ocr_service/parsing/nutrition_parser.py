"""
parsing/nutrition_parser.py

Parses raw OCR'd nutrition-table text into a structured dict of
{nutrient_key: {"value": float, "unit": str}} per STEP 16.
"""

import re
import numpy as np
from backend.services.ocr_service import config
from backend.services.ocr_service.detection.geometry import union_rect
from backend.services.ocr_service.detection.ocr_detector import normalize_ocr_text


# Canonical nutrient key -> list of text patterns (already lowercase) that
# should map to it. Longer/more specific phrases are listed first so they
# match before their shorter substrings (e.g. "saturated fat" before "fat").
NUTRIENT_KEY_PATTERNS = [
    ("energy", [r"energy", r"calories?", r"caloric\s*value"]),
    ("protein", [r"proteins?", r"total\s*protein"]),
    ("saturated_fat", [r"saturated\s*fat", r"saturates?", r"saturated"]),
    ("trans_fat", [r"trans\s*fat", r"trans\s*fatty\s*acids?"]),
    ("total_fat", [r"total\s*fat", r"fats?"]),
    ("added_sugars", [r"added\s*sugars?"]),
    ("total_sugars", [r"total\s*sugars?", r"sugars?"]),
    ("total_carbohydrate", [r"total\s*carbohydrates?", r"carbohydrates?", r"carbs?"]),
    ("dietary_fibre", [r"dietary\s*fib(?:re|er)", r"fib(?:re|er)"]),
    ("sodium", [r"sodium"]),
    ("salt", [r"salt"]),
    ("cholesterol", [r"cholesterol"]),
    ("calcium", [r"calcium"]),
    ("iron", [r"iron"]),
    ("vitamin", [r"vitamin\s*[a-z0-9]*"]),
]

# Units we recognize, ordered longest-first so e.g. "kcal" matches before "cal"
UNIT_PATTERN = r"(kcal|kj|mcg|mg|iu|g|%)"
COMPARISON_PATTERN = r"(?:<=|>=|<|>|~)?\s*"

VALUE_UNIT_PATTERN = re.compile(
    COMPARISON_PATTERN + r"(\d+(?:\.\d+)?)\s*" + UNIT_PATTERN, re.IGNORECASE
)


def _find_key(line_lower):
    for key, patterns in NUTRIENT_KEY_PATTERNS:
        for pat in patterns:
            if re.search(r"\b" + pat + r"\b", line_lower):
                return key
    return None


def _extract_value_and_unit(text, nutrient_key=None):
    if not text:
        return None

    # 1. Check for explicit kcal or kJ
    kcal_m = re.search(COMPARISON_PATTERN + r"(\d+(?:\.\d+)?)\s*kcal\b", text, re.IGNORECASE)
    if kcal_m:
        try:
            return float(kcal_m.group(1)), "kcal"
        except ValueError:
            pass

    kj_m = re.search(COMPARISON_PATTERN + r"(\d+(?:\.\d+)?)\s*kj\b", text, re.IGNORECASE)
    if kj_m:
        try:
            return float(kj_m.group(1)), "kj"
        except ValueError:
            pass

    # 2. Check for general value + unit (with optional comparison sign <, <=, etc.)
    val_unit_m = VALUE_UNIT_PATTERN.search(text)
    if val_unit_m:
        try:
            val = float(val_unit_m.group(1))
            unit = val_unit_m.group(2).lower()
            return val, unit
        except ValueError:
            pass

    # 3. Dimensionless Calories/Energy check
    if nutrient_key == "energy":
        dim_m = re.search(
            r"(?:calories?|energy)?\s*[:\-\s]?\s*" + COMPARISON_PATTERN + r"(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if dim_m:
            try:
                return float(dim_m.group(1)), "kcal"
            except ValueError:
                pass

    # 4. If in a cell that contains only a number (e.g. '192', '2.97', '<0.4', '14.9')
    # and nutrient_key is known:
    if nutrient_key:
        dim_m = re.search(r"^" + COMPARISON_PATTERN + r"(\d+(?:\.\d+)?)\s*$", text.strip())
        if dim_m:
            try:
                val = float(dim_m.group(1))
                if nutrient_key == "energy":
                    return val, "kcal"
                elif nutrient_key in (
                    "protein",
                    "total_carbohydrate",
                    "total_sugars",
                    "added_sugars",
                    "total_fat",
                    "saturated_fat",
                    "trans_fat",
                    "dietary_fibre",
                ):
                    return val, "g"
                elif nutrient_key == "sodium":
                    unit = "mg" if val >= 10.0 else "g"
                    return val, unit
            except ValueError:
                pass

    return None


def parse_nutrition(raw_text, ocr_dict=None):
    """
    Parses raw nutrition OCR text and/or OCR dictionary into structured dict.
    Supports both single-value and multi-column (per_100g, per_serving) tables.
    """
    # 1. Fallback to basic string parsing if no ocr_dict is provided
    if not ocr_dict or not ocr_dict.get("best_items"):
        return _parse_nutrition_string_fallback(raw_text)

    items = ocr_dict["best_items"]
    from backend.services.ocr_service.detection.ocr_detector import enrich_items
    if items and "height" not in items[0]:
        items = enrich_items(items)
    # Reconstruct lines
    from backend.services.ocr_service.detection.line_builder import reconstruct_lines
    # Fake image shape since we don't need real clipping here, just sorted lines
    lines = reconstruct_lines(items, (1000, 1000))
    if not lines:
        return _parse_nutrition_string_fallback(raw_text)

    # Calculate median line height to define cell gaps
    line_hs = [ln["height"] for ln in lines if ln["height"] > 0]
    median_h = float(np.median(line_hs)) if line_hs else 20.0
    cell_gap_threshold = max(25.0, median_h * 1.5)

    # 2. Divide each line into cells based on horizontal gaps
    processed_lines = []
    for ln in lines:
        sorted_members = sorted(ln.get("items", []), key=lambda it: it["rect"][0])
        if not sorted_members:
            continue
            
        cells = []
        curr_cell = [sorted_members[0]]
        for it in sorted_members[1:]:
            prev_it = curr_cell[-1]
            gap = it["rect"][0] - prev_it["rect"][2]
            if gap > cell_gap_threshold:
                cells.append(curr_cell)
                curr_cell = [it]
            else:
                curr_cell.append(it)
        cells.append(curr_cell)

        # Represent cells as dicts with text and horizontal range
        cell_dicts = []
        for c in cells:
            c_text = " ".join(m["text"] for m in c)
            c_rect = union_rect([m["rect"] for m in c])
            cell_dicts.append({
                "text": c_text,
                "norm_text": normalize_ocr_text(c_text),
                "rect": c_rect,
                "center_x": (c_rect[0] + c_rect[2]) / 2.0
            })
        processed_lines.append(cell_dicts)

    # 3. Identify header cells and columns
    col_headers = {}
    header_lines = []
    nutrient_rows = []

    for ln in processed_lines:
        is_header = False
        text_full = " ".join(c["text"] for c in ln).lower()
        if any(h in text_full for h in ["per 100", "per serving", "per serve", "approx", "% rda", "rda", "% gd"]):
            is_header = True
            
        if is_header:
            header_lines.append(ln)
        else:
            first_cell = ln[0]["text"].lower()
            if _find_key(first_cell) is not None:
                nutrient_rows.append(ln)

    # If we found header lines, parse their cell positions
    per_100g_centers = []
    per_serving_centers = []
    for h_ln in header_lines:
        for c in h_ln:
            t_low = c["text"].lower()
            if "100" in t_low:
                per_100g_centers.append(c["center_x"])
            elif "serving" in t_low or "serve" in t_low or "pkg" in t_low or "pack" in t_low:
                per_serving_centers.append(c["center_x"])

    col_100g_x = np.mean(per_100g_centers) if per_100g_centers else None
    col_serving_x = np.mean(per_serving_centers) if per_serving_centers else None

    # If no headers detected, look at horizontal positions of value cells in nutrient rows
    if col_100g_x is None or col_serving_x is None:
        value_centers = []
        for r in nutrient_rows:
            for c in r[1:]:
                value_centers.append(c["center_x"])
        if len(value_centers) >= 2:
            value_centers.sort()
            gaps = [value_centers[i+1] - value_centers[i] for i in range(len(value_centers)-1)]
            if gaps:
                max_gap_idx = int(np.argmax(gaps))
                if gaps[max_gap_idx] > 30.0:
                    left_group = value_centers[:max_gap_idx+1]
                    right_group = value_centers[max_gap_idx+1:]
                    if col_100g_x is None:
                        col_100g_x = float(np.mean(left_group))
                    if col_serving_x is None:
                        col_serving_x = float(np.mean(right_group))

    # 4. Parse nutrient rows and assign values to columns
    result = {}
    for r in nutrient_rows:
        if len(r) < 2:
            continue
            
        nutrient_cell = r[0]
        nutrient_key = _find_key(nutrient_cell["text"].lower())
        if not nutrient_key:
            continue
            
        row_data = {}
        for val_cell in r[1:]:
            val_text = val_cell["text"]
            extracted = _extract_value_and_unit(val_text, nutrient_key=nutrient_key)
            if not extracted:
                continue
            val, unit = extracted

            cx = val_cell["center_x"]
            col_key = "per_100g"
            if col_100g_x is not None and col_serving_x is not None:
                dist_100g = abs(cx - col_100g_x)
                dist_serving = abs(cx - col_serving_x)
                if dist_serving < dist_100g:
                    col_key = "per_serving"
                else:
                    col_key = "per_100g"
            elif len(r) > 2:
                val_cell_idx = r.index(val_cell)
                if val_cell_idx == 1:
                    col_key = "per_100g"
                elif val_cell_idx == 2:
                    col_key = "per_serving"

            row_data[col_key] = {"value": val, "unit": unit}

        if row_data:
            # Maintain backward compatibility
            first_col = "per_100g" if "per_100g" in row_data else (list(row_data.keys())[0] if row_data else None)
            if first_col and first_col in row_data:
                row_data["value"] = row_data[first_col]["value"]
                row_data["unit"] = row_data[first_col]["unit"]
            result[nutrient_key] = row_data

    if not result:
        return _parse_nutrition_string_fallback(raw_text)
    return result


def _split_line_into_nutrient_segments(line: str) -> list:
    if not line or not line.strip():
        return []

    line_lower = line.lower()
    matches = []
    for key, patterns in NUTRIENT_KEY_PATTERNS:
        for pat in patterns:
            for m in re.finditer(r"\b" + pat + r"\b", line_lower):
                matches.append((m.start(), m.end(), key))

    if not matches:
        return [line.strip()]

    # Filter out matches that are strictly contained within longer matches
    # e.g., "fat" inside "saturated fat" or "sugars" inside "total sugars"
    valid_matches = []
    for start, end, key in matches:
        contained = False
        for o_start, o_end, o_key in matches:
            if o_start <= start and end <= o_end and (o_end - o_start) > (end - start):
                contained = True
                break
        if not contained:
            valid_matches.append((start, end, key))

    # Sort matches by start position; if same start, take longest
    valid_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    unique_matches = []
    last_end = -1
    for start, end, key in valid_matches:
        if start >= last_end:
            unique_matches.append((start, end, key))
            last_end = end

    if not unique_matches:
        return [line.strip()]

    segments = []
    if unique_matches[0][0] > 0:
        prefix = line[:unique_matches[0][0]].strip()
        if prefix:
            segments.append(prefix)

    for idx, (start, end, key) in enumerate(unique_matches):
        next_start = unique_matches[idx + 1][0] if idx + 1 < len(unique_matches) else len(line)
        seg = line[start:next_start].strip()
        if seg:
            segments.append(seg)

    return segments


def _parse_nutrition_string_fallback(raw_text):
    if not raw_text or not raw_text.strip():
        return {}

    lines = re.split(r"[\n]+", raw_text)
    expanded_lines = []
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        segs = _split_line_into_nutrient_segments(line_s)
        expanded_lines.extend(segs)

    result = {}
    i = 0
    while i < len(expanded_lines):
        line = expanded_lines[i]
        line_lower = line.lower()
        key = _find_key(line_lower)
        if key is None or key in result:
            i += 1
            continue

        extracted = _extract_value_and_unit(line, nutrient_key=key)
        # If value was not on the same line, check if it is on the next line (requirement 8)
        if not extracted and i + 1 < len(expanded_lines):
            next_line = expanded_lines[i + 1]
            if _find_key(next_line.lower()) is None:
                next_extracted = _extract_value_and_unit(next_line, nutrient_key=key)
                if next_extracted:
                    extracted = next_extracted
                    i += 1

        if not extracted:
            i += 1
            continue

        value, unit = extracted
        result[key] = {
            "value": value,
            "unit": unit,
            "per_100g": {"value": value, "unit": unit}
        }
        i += 1

    return result
