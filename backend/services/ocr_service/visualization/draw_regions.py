"""
visualization/draw_regions.py

Draws the final debugging visualization:
  GREEN  = Ingredients / Composition polygon ROI + secondary bbox
  BLUE   = Nutrition Information polygon ROI + secondary bbox
  GRAY   = other detected OCR text boxes (not part of either region)
"""

import cv2
import numpy as np
from backend.services.ocr_service import config

def _draw_polygon_region(image, poly, bbox, color, label, thickness=3):
    if not poly and not bbox:
        return

    if poly and len(poly) >= 3:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.18, image, 0.82, 0, image)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    if bbox:
        bx1, by1, bx2, by2 = [int(round(v)) for v in bbox]
        # Thin secondary bounding box
        cv2.rectangle(image, (bx1, by1), (bx2, by2), color, 1, lineType=cv2.LINE_AA)

        # Label at top of bbox
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(1.0, (bx2 - bx1) / 400.0))
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 2)
            label_y1 = max(0, by1 - text_h - baseline - 6)
            cv2.rectangle(image, (bx1, label_y1), (bx1 + text_w + 8, by1), color, -1)
            cv2.putText(
                image,
                label,
                (bx1 + 4, by1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


def _is_inside(rect, poly, bbox):
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    if poly and len(poly) >= 3:
        pts = np.array(poly, dtype=np.float32).reshape((-1, 1, 2))
        dist = cv2.pointPolygonTest(pts, (float(cx), float(cy)), False)
        if dist >= 0:
            return True
    if bbox:
        if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
            return True
    return False


def draw_regions(
    image,
    ingredient_result=None,
    nutrition_result=None,
    all_ocr_items=None,
    draw_other_text=True,
    draw_debug_rejected=False,
):
    """
    Returns a new BGR image (copy of `image`) with polygon ROIs and bounding boxes drawn.
    """
    vis = image.copy()

    ing_poly = ingredient_result.get("roi_polygon") if ingredient_result else None
    ing_box = ingredient_result.get("bbox") if ingredient_result else None

    nut_poly = nutrition_result.get("roi_polygon") if nutrition_result else None
    nut_box = nutrition_result.get("bbox") if nutrition_result else None

    # Draw other text in gray
    if draw_other_text and all_ocr_items:
        gray = (160, 160, 160)
        for it in all_ocr_items:
            rect = it["rect"]
            if _is_inside(rect, ing_poly, ing_box) or _is_inside(rect, nut_poly, nut_box):
                continue
            if "polygon" in it and it["polygon"] and len(it["polygon"]) >= 3:
                pts = np.array(it["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], isClosed=True, color=gray, thickness=1, lineType=cv2.LINE_AA)
            else:
                rx1, ry1, rx2, ry2 = [int(round(v)) for v in it["rect"]]
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), gray, 1, lineType=cv2.LINE_AA)

    # Draw Ingredients Polygon ROI + Box
    if ing_poly or ing_box:
        conf = ingredient_result.get("confidence", 0.0)
        _draw_polygon_region(
            vis, ing_poly, ing_box, config.COLOR_INGREDIENTS, f"INGREDIENTS ({conf:.2f})", thickness=3
        )

    # Draw Nutrition Polygon ROI + Box
    if nut_poly or nut_box:
        conf = nutrition_result.get("confidence", 0.0)
        _draw_polygon_region(
            vis, nut_poly, nut_box, config.COLOR_NUTRITION, f"NUTRITION ({conf:.2f})", thickness=3
        )

    return vis

