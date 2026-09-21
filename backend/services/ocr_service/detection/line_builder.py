"""
detection/line_builder.py

Column-aware and polygon-aware OCR line reconstruction.
Detects column structure on raw OCR items before line reconstruction.
Groups word/fragment polygon items into logical text lines using:
- column ID compatibility
- baseline and vertical center alignment
- polygon orientation angle compatibility
- text height similarity
- horizontal distance constraints
- reading direction
"""

import numpy as np
import cv2
import config
from detection.geometry import (
    poly_to_rect,
    union_rect,
    vertical_overlap_ratio,
    horizontal_overlap_ratio,
    horizontal_distance,
    polygon_angle,
    baseline_angle,
    perpendicular_distance_to_baseline,
    convex_hull,
)


def detect_columns(ocr_items, image_shape):
    """
    Detects column structure from raw OCR items BEFORE line reconstruction.

    Analyzes horizontal spans, overlaps, gaps, and local density.
    Assigns 'column_id' to each OCR item in-place and returns column metadata.

    Args:
        ocr_items: list of enriched OCR item dicts
        image_shape: tuple of (height, width) or (height, width, channels)

    Returns:
        list of column dicts:
            [{"column_id": int, "bbox": [x1, y1, x2, y2], "items": [...]}, ...]
    """
    if not ocr_items:
        return []

    h, w = image_shape[:2]
    for it in ocr_items:
        it["column_id"] = 0
        if "rect" not in it and "bbox" in it:
            from detection.geometry import poly_to_rect
            it["rect"] = poly_to_rect(it["bbox"])
        r = it.get("rect", [0, 0, 0, 0])
        if "width" not in it:
            it["width"] = float(r[2] - r[0])
        if "height" not in it:
            it["height"] = float(r[3] - r[1])
        if "center" not in it:
            it["center"] = [float((r[0] + r[2]) / 2.0), float((r[1] + r[3]) / 2.0)]

    heights = [it["height"] for it in ocr_items if it.get("height", 0) > 0]
    median_h = float(np.median(heights)) if heights else 20.0

    if len(ocr_items) < 6:
        # Too few items to reliably detect multiple columns
        rects = [it["rect"] for it in ocr_items]
        return [{"column_id": 0, "bbox": union_rect(rects), "items": list(ocr_items)}]

    # Filter out full-width headers or tiny noise for gutter finding
    valid_items = [
        it for it in ocr_items
        if it["width"] < w * 0.70 and it["height"] >= median_h * 0.35 and it["width"] < median_h * 15
    ]
    if len(valid_items) < 6:
        valid_items = ocr_items

    # Build 1D horizontal coverage histogram to detect true vertical gutters
    bin_size = max(4, int(round(median_h * 0.4)))
    num_bins = int(np.ceil(w / bin_size)) + 1
    coverage = np.zeros(num_bins, dtype=np.int32)

    for it in valid_items:
        r = it["rect"]
        b_start = max(0, int(np.floor(r[0] / bin_size)))
        b_end = min(num_bins, int(np.ceil(r[2] / bin_size)))
        coverage[b_start:b_end] += 1

    # Look for continuous gutter bins where coverage == 0 or is very sparse (<= 1 item)
    min_gutter_px = max(12.0, min(median_h * 1.0, w * config.COLUMN_GAP_MIN_WIDTH_RATIO))
    min_gutter_bins = max(2, int(round(min_gutter_px / bin_size)))

    gutters = []
    in_gutter = False
    g_start = 0

    # Avoid image borders
    margin_bins = int(round(w * 0.03 / bin_size))
    scan_start = margin_bins
    scan_end = num_bins - margin_bins

    for b in range(scan_start, scan_end):
        if coverage[b] <= 1:
            if not in_gutter:
                in_gutter = True
                g_start = b
        else:
            if in_gutter:
                in_gutter = False
                g_len = b - g_start
                if g_len >= min_gutter_bins:
                    gutters.append((g_start * bin_size, b * bin_size))

    if not gutters:
        # Single column layout
        rects = [it["rect"] for it in ocr_items]
        return [{"column_id": 0, "bbox": union_rect(rects), "items": list(ocr_items)}]

    # Cut column boundaries at midpoints of gutters
    cut_x = [float((g[0] + g[1]) / 2.0) for g in gutters]
    cut_x.sort()

    # Assign column IDs based on center X of each item
    columns_map = {}
    for it in ocr_items:
        cx = it["center"][0]
        col_id = 0
        for split_x in cut_x:
            if cx > split_x:
                col_id += 1
            else:
                break
        it["column_id"] = col_id
        columns_map.setdefault(col_id, []).append(it)

    column_results = []
    for col_id in sorted(columns_map.keys()):
        c_items = columns_map[col_id]
        rects = [it["rect"] for it in c_items]
        column_results.append({
            "column_id": col_id,
            "bbox": union_rect(rects),
            "items": c_items
        })

    return column_results


def reconstruct_lines(ocr_items, image_shape):
    """
    Groups individual OCR items (words/fragments) into logical text lines.
    Preserves polygon geometry, orientation, and column separation.

    Args:
        ocr_items: list of canonical OCR item dicts
        image_shape: tuple of (height, width, channels) or (height, width)

    Returns:
        list of line dicts sorted in reading order
    """
    if not ocr_items:
        return []

    h_img, w_img = image_shape[:2]

    # 1. Run column detection first to partition items into columns
    detect_columns(ocr_items, image_shape)

    # 2. Scale-aware parameters
    heights = [it["height"] for it in ocr_items if it["height"] > 0]
    median_h = float(np.median(heights)) if heights else 18.0

    v_tolerance = median_h * config.LINE_BUILDER_V_TOLERANCE_FACTOR
    max_h_gap = median_h * config.LINE_BUILDER_H_GAP_FACTOR
    width_cap = w_img * getattr(config, "LINE_GROUP_MAX_H_GAP_WIDTH_RATIO", 0.05)
    max_h_gap = min(max_h_gap, width_cap)

    h_tol = config.LINE_BUILDER_HEIGHT_TOLERANCE_FACTOR
    angle_tol = getattr(config, "ANGLE_COMPATIBILITY_THRESHOLD", 12.0)

    n = len(ocr_items)
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

    # Compare pairs of items to see if they belong to the same visual line
    for i in range(n):
        it_i = ocr_items[i]
        rect_i = it_i["rect"]
        h_i = it_i["height"]
        cy_i = it_i["center"][1]
        col_i = it_i.get("column_id", 0)
        ang_i = it_i.get("angle", 0.0)

        for j in range(i + 1, n):
            it_j = ocr_items[j]
            rect_j = it_j["rect"]
            h_j = it_j["height"]
            cy_j = it_j["center"][1]
            col_j = it_j.get("column_id", 0)
            ang_j = it_j.get("angle", 0.0)

            # Rule 1: Must share same column ID
            if col_i != col_j:
                continue

            # Rule 2: Orientation / Angle compatibility
            angle_diff = abs(ang_i - ang_j)
            if angle_diff > 90.0:
                angle_diff = abs(angle_diff - 180.0)
            if angle_diff > angle_tol:
                continue

            # Rule 3: Vertical alignment / Baseline similarity AND vertical overlap
            v_dist = abs(cy_i - cy_j)
            p_dist = perpendicular_distance_to_baseline(it_i.get("polygon"), it_j.get("polygon"))
            min_v = min(v_dist, p_dist)

            if min_v > v_tolerance:
                continue

            v_overlap = vertical_overlap_ratio(rect_i, rect_j)
            if v_overlap < config.LINE_GROUP_Y_OVERLAP_THRESHOLD:
                continue

            # Rule 4: Height similarity
            height_diff = abs(h_i - h_j) / max(h_i, h_j, 1e-6)
            if height_diff > h_tol:
                continue

            # Rule 5: Cannot be vertically stacked (cannot have significant horizontal overlap)
            h_overlap = horizontal_overlap_ratio(rect_i, rect_j)
            if h_overlap >= 0.20:
                continue

            # Rule 6: Horizontal distance
            h_dist = horizontal_distance(rect_i, rect_j)
            if h_dist > max_h_gap:
                continue

            # All checks pass -> merge into same line
            union(i, j)

    # Group items by parent
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(ocr_items[i])

    reconstructed = []
    for members in groups.values():
        # Sort items left-to-right within the line
        sorted_members = sorted(members, key=lambda it: (it["rect"][0], it["center"][1]))

        merged_text = " ".join(m["text"] for m in sorted_members).strip()
        if not merged_text:
            continue

        merged_rect = union_rect([m["rect"] for m in sorted_members])
        x1, y1, x2, y2 = merged_rect
        center = [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
        width = float(x2 - x1)
        height = float(y2 - y1)
        mean_conf = float(np.mean([m.get("confidence", 0.0) for m in sorted_members]))
        median_angle = float(np.median([m.get("angle", 0.0) for m in sorted_members]))
        col_id = sorted_members[0].get("column_id", 0)

        # Collect child polygons
        child_polygons = [m["polygon"] for m in sorted_members if m.get("polygon") is not None]

        # Reconstruct line polygon boundary from child polygons
        if len(sorted_members) == 1:
            line_poly = sorted_members[0]["polygon"]
        else:
            # Build polygon from ordered top boundary + reversed bottom boundary
            upper_boundary = []
            lower_boundary = []
            for m in sorted_members:
                poly = m.get("polygon")
                if poly is not None and len(poly) >= 4:
                    upper_boundary.extend([poly[0], poly[1]])
                    lower_boundary.extend([poly[2], poly[3]])
                else:
                    rx1, ry1, rx2, ry2 = m["rect"]
                    upper_boundary.extend([[rx1, ry1], [rx2, ry1]])
                    lower_boundary.extend([[rx2, ry2], [rx1, ry2]])

            line_poly = upper_boundary + list(reversed(lower_boundary))

        if not line_poly:
            line_poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        reconstructed.append({
            "text": merged_text,
            "child_items": sorted_members,
            "items": sorted_members,  # backward compatibility alias
            "polygons": child_polygons,
            "polygon": line_poly,
            "rect": merged_rect,
            "center": center,
            "height": height,
            "width": width,
            "confidence": mean_conf,
            "mean_confidence": mean_conf,
            "angle": median_angle,
            "column_id": col_id,
        })

    # Sort lines in reading order: column-aware, then top-to-bottom, left-to-right
    line_h = float(np.median([ln["height"] for ln in reconstructed])) if reconstructed else 20.0
    y_band = max(line_h * 0.5, 5.0)

    reconstructed.sort(
        key=lambda ln: (
            ln.get("column_id", 0),
            round(ln["rect"][1] / y_band),
            ln["rect"][0]
        )
    )

    for idx, ln in enumerate(reconstructed):
        ln["line_index"] = idx

    return reconstructed

