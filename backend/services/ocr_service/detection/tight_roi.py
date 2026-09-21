"""
detection/tight_roi.py

Refines raw region candidates by applying semantic line filtering, boundary
detection, and calculating the tight convex hull polygon envelope of accepted lines.
Applies font-scale and image-scale aware padding and clips to image boundaries
to produce the final roi_polygon (primary) and bounding box (secondary).
"""

import numpy as np
from backend.services.ocr_service import config
from backend.services.ocr_service.detection.geometry import (
    convex_hull,
    polygon_to_bbox,
    median_line_height,
    union_rect,
    validate_region,
)

def compute_polygon_roi(lines, image_shape):
    """
    Computes a tight polygon ROI from accepted lines using their polygons.
    Computes convex hull with font-scale and image-scale aware padding.

    Returns:
        dict with:
            roi_polygon: list of [x, y] coordinates
            refined_bbox: [x1, y1, x2, y2]
            line_count: int
            owner_lines: list of line texts
    """
    if not lines:
        return {
            "roi_polygon": [],
            "refined_bbox": None,
            "line_count": 0,
            "owner_lines": [],
        }

    # Gather all point coordinates from line polygons (with rect consistency check)
    all_pts = []
    for ln in lines:
        rx1, ry1, rx2, ry2 = ln.get("rect", [0, 0, 0, 0])
        poly = ln.get("polygon")
        valid_poly = True
        if poly is not None and len(poly) >= 3:
            for pt in poly:
                if not (rx1 - 25 <= pt[0] <= rx2 + 25 and ry1 - 25 <= pt[1] <= ry2 + 25):
                    valid_poly = False
                    break
            if valid_poly:
                all_pts.extend(poly)
            else:
                all_pts.extend([[rx1, ry1], [rx2, ry1], [rx2, ry2], [rx1, ry2]])
        elif ln.get("rect"):
            all_pts.extend([[rx1, ry1], [rx2, ry1], [rx2, ry2], [rx1, ry2]])

    if not all_pts:
        return {
            "roi_polygon": [],
            "refined_bbox": None,
            "line_count": 0,
            "owner_lines": [],
        }

    # Font-scale and image-scale aware padding
    rects = [ln["rect"] for ln in lines if ln.get("rect")]
    line_h = float(median_line_height(rects)) if rects else 20.0
    h, w = image_shape[:2]
    pad_y = float(line_h * getattr(config, "POLYGON_ROI_MARGIN_LINE_FACTOR", 0.25))
    pad_x = float(w * getattr(config, "POLYGON_ROI_MARGIN_IMAGE_RATIO", 0.005))

    roi_poly = convex_hull([all_pts], pad_x=pad_x, pad_y=pad_y, image_shape=image_shape)
    bbox = polygon_to_bbox(roi_poly)
    if bbox is not None:
        bbox = validate_region(bbox, image_shape)

    owner_lines = [ln.get("text", "") for ln in lines]

    return {
        "roi_polygon": roi_poly,
        "refined_bbox": bbox,
        "line_count": len(lines),
        "owner_lines": owner_lines,
    }


def refine_ingredient_roi(candidate, image_shape):
    """
    Computes tight polygon and bbox for Ingredients ROI.
    """
    if isinstance(candidate, list):
        lines = candidate
    elif isinstance(candidate, dict):
        lines = candidate.get("lines") or candidate.get("matched_items", [])
    else:
        lines = []

    res = compute_polygon_roi(lines, image_shape)
    res["refinement_method"] = "convex_hull_polygon_roi"
    res["original_candidate_bbox"] = candidate.get("bbox") if isinstance(candidate, dict) else None
    res["changed"] = (res["refined_bbox"] != res["original_candidate_bbox"])
    return res


def refine_nutrition_roi(candidate, image_shape):
    """
    Computes tight polygon and bbox for Nutrition ROI.
    """
    if isinstance(candidate, list):
        lines = candidate
    elif isinstance(candidate, dict):
        lines = candidate.get("lines") or candidate.get("matched_items", [])
    else:
        lines = []

    res = compute_polygon_roi(lines, image_shape)
    res["refinement_method"] = "convex_hull_polygon_roi"
    res["original_candidate_bbox"] = candidate.get("bbox") if isinstance(candidate, dict) else None
    res["changed"] = (res["refined_bbox"] != res["original_candidate_bbox"])
    return res
