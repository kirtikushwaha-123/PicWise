"""
detection/geometry.py

Shared geometric helpers used by ingredient/nutrition region detection:
bbox conversions, distances, overlap, line grouping, column clustering,
region validation and margin helpers.

Two additions over the original version fix the "ROI too small" problem:

  - group_ocr_boxes_into_lines(): merges fragmented OCR boxes that belong
    to the same visual line (e.g. a nutrient name box + its separate
    value/unit box) into a single "line" record, so expansion logic
    reasons about whole lines instead of arbitrary OCR fragments.

  - cluster_into_columns(): splits a page's lines into left-to-right
    column groups when a wide horizontal gutter is present, so that
    side-by-side sections (e.g. Nutrition on the left, Ingredients on
    the right) don't bleed into each other during expansion.
"""

import cv2
import numpy as np

from backend.services.ocr_service import config


# --------------------------------------------------------------------------
# Polygon geometry helpers
# --------------------------------------------------------------------------

def polygon_angle(poly):
    """
    Computes dominant orientation angle (in degrees, roughly -45 to 45) from
    the polygon's top edge or baseline.
    """
    if poly is None or len(poly) < 2:
        return 0.0
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) >= 4:
        # PaddleOCR convention: [top-left, top-right, bottom-right, bottom-left]
        dx = float(pts[1, 0] - pts[0, 0])
        dy = float(pts[1, 1] - pts[0, 0] if False else pts[1, 1] - pts[0, 1])
    else:
        dx = float(pts[-1, 0] - pts[0, 0])
        dy = float(pts[-1, 1] - pts[0, 1])
    angle = np.degrees(np.arctan2(dy, dx))
    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    return float(angle)


def baseline_angle(poly):
    """
    Computes orientation angle of the bottom edge (baseline) of an OCR polygon.
    """
    if poly is None or len(poly) < 4:
        return polygon_angle(poly)
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    dx = float(pts[2, 0] - pts[3, 0])
    dy = float(pts[2, 1] - pts[3, 1])
    angle = np.degrees(np.arctan2(dy, dx))
    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    return float(angle)


def polygon_area(poly):
    """Calculates exact polygon area using Green's theorem / cv2.contourArea."""
    if poly is None or len(poly) < 3:
        return 0.0
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    return float(cv2.contourArea(pts))


def polygon_center(poly):
    """Calculates center of mass / mean coordinates [cx, cy] of a polygon."""
    if poly is None or len(poly) == 0:
        return [0.0, 0.0]
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    return [float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))]


def polygon_width(poly):
    """Calculates width of polygon bounding box or edge span."""
    if poly is None or len(poly) == 0:
        return 0.0
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    return float(np.max(pts[:, 0]) - np.min(pts[:, 0]))


def polygon_height(poly):
    """Calculates height of polygon bounding box or edge span."""
    if poly is None or len(poly) == 0:
        return 0.0
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    return float(np.max(pts[:, 1]) - np.min(pts[:, 1]))


def polygon_distance(poly_a, poly_b):
    """Euclidean distance between polygon centroids."""
    ca = polygon_center(poly_a)
    cb = polygon_center(poly_b)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def perpendicular_distance_to_baseline(poly_a, poly_b):
    """
    Computes perpendicular distance from center of poly_b to the baseline of poly_a.
    """
    if poly_a is None or len(poly_a) < 4 or poly_b is None:
        ca = polygon_center(poly_a)
        cb = polygon_center(poly_b)
        return abs(ca[1] - cb[1])
    pts_a = np.array(poly_a, dtype=np.float32).reshape(-1, 2)
    cb = np.array(polygon_center(poly_b), dtype=np.float32)
    p3, p2 = pts_a[3], pts_a[2]
    line_vec = p2 - p3
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return float(np.linalg.norm(cb - p3))
    # cross product gives area of parallelogram; divide by base length for perpendicular distance
    cross_2d = float(line_vec[0] * (cb[1] - p3[1]) - line_vec[1] * (cb[0] - p3[0]))
    return float(abs(cross_2d) / line_len)


def polygon_overlap(poly_a, poly_b):
    """
    Fraction of the smaller polygon's area that overlaps with the other polygon.
    """
    if poly_a is None or poly_b is None:
        return 0.0
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    pts_a = np.array(poly_a, dtype=np.float32).reshape(-1, 2)
    pts_b = np.array(poly_b, dtype=np.float32).reshape(-1, 2)
    try:
        ret, inter_pts = cv2.intersectConvexConvex(pts_a, pts_b)
        if ret > 0 and inter_pts is not None and len(inter_pts) >= 3:
            inter_area = cv2.contourArea(inter_pts)
            return float(inter_area / min(area_a, area_b))
    except Exception:
        pass
    # fallback to rect overlap
    ra = poly_to_rect(poly_a)
    rb = poly_to_rect(poly_b)
    inter = bbox_intersection_area(ra, rb)
    return float(inter / min(bbox_area(ra), bbox_area(rb), 1e-6))


def polygon_iou(poly_a, poly_b):
    """
    Computes Intersection over Union of two polygons.
    """
    if poly_a is None or poly_b is None:
        return 0.0
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    pts_a = np.array(poly_a, dtype=np.float32).reshape(-1, 2)
    pts_b = np.array(poly_b, dtype=np.float32).reshape(-1, 2)
    try:
        ret, inter_pts = cv2.intersectConvexConvex(pts_a, pts_b)
        if ret > 0 and inter_pts is not None and len(inter_pts) >= 3:
            inter_area = cv2.contourArea(inter_pts)
            union_area = area_a + area_b - inter_area
            return float(inter_area / max(union_area, 1e-6))
    except Exception:
        pass
    ra = poly_to_rect(poly_a)
    rb = poly_to_rect(poly_b)
    return bbox_iou(ra, rb)


def convex_hull(polygons, pad_x=0.0, pad_y=0.0, image_shape=None):
    """
    Collects all vertices from a list of polygons, computes the convex hull,
    optionally applies scale-aware padding (expanding outward from the centroid),
    and clips coordinates to image_shape [height, width].
    """
    if not polygons:
        return []
    all_pts = []
    for p in polygons:
        if p is None:
            continue
        pts = np.array(p, dtype=np.float32).reshape(-1, 2)
        all_pts.append(pts)
    if not all_pts:
        return []
    merged_pts = np.vstack(all_pts)
    hull = cv2.convexHull(merged_pts)
    hull_pts = hull.reshape(-1, 2).astype(np.float32)

    if pad_x > 0 or pad_y > 0:
        cx = float(np.mean(hull_pts[:, 0]))
        cy = float(np.mean(hull_pts[:, 1]))
        padded = []
        for pt in hull_pts:
            dx = float(pt[0] - cx)
            dy = float(pt[1] - cy)
            dist = float(np.hypot(dx, dy))
            if dist > 1e-6:
                nx = float(pt[0] + (dx / dist) * pad_x)
                ny = float(pt[1] + (dy / dist) * pad_y)
            else:
                nx = float(pt[0] + pad_x)
                ny = float(pt[1] + pad_y)
            padded.append([nx, ny])
        hull_pts = np.array(padded, dtype=np.float32)

    if image_shape is not None:
        h, w = image_shape[:2]
        hull_pts[:, 0] = np.clip(hull_pts[:, 0], 0, w)
        hull_pts[:, 1] = np.clip(hull_pts[:, 1], 0, h)

    return hull_pts.tolist()


def minimum_area_rectangle(pts_or_polygons):
    """
    Computes the minimum-area bounding rectangle from points or polygons.
    Returns 4 corner points [[x, y], ...] as float list.
    """
    if not pts_or_polygons:
        return []
    all_pts = []
    for item in pts_or_polygons:
        if item is None:
            continue
        pts = np.array(item, dtype=np.float32).reshape(-1, 2)
        all_pts.append(pts)
    if not all_pts:
        return []
    merged_pts = np.vstack(all_pts)
    rect = cv2.minAreaRect(merged_pts)
    box = cv2.boxPoints(rect)
    return box.astype(float).tolist()


def polygon_to_bbox(poly):
    """
    Converts a polygon [[x, y], ...] into an axis-aligned [x1, y1, x2, y2] bbox.
    """
    if not poly:
        return None
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    return [
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    ]


# --------------------------------------------------------------------------
# Basic rect helpers
# --------------------------------------------------------------------------

def poly_to_rect(bbox_poly):
    """
    Convert a 4-point polygon [[x,y]*4] to an axis-aligned rect
    [x1, y1, x2, y2].
    """
    pts = np.array(bbox_poly)
    x1 = float(pts[:, 0].min())
    y1 = float(pts[:, 1].min())
    x2 = float(pts[:, 0].max())
    y2 = float(pts[:, 1].max())
    return [x1, y1, x2, y2]


def rect_center(rect):
    x1, y1, x2, y2 = rect
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def rect_height(rect):
    return rect[3] - rect[1]


def rect_width(rect):
    return rect[2] - rect[0]


def union_rect(rects):
    if not rects:
        return None
    xs1 = [r[0] for r in rects]
    ys1 = [r[1] for r in rects]
    xs2 = [r[2] for r in rects]
    ys2 = [r[3] for r in rects]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def vertical_distance(rect_a, rect_b):
    """Vertical gap between two rects (0 if they vertically overlap)."""
    a_top, a_bottom = rect_a[1], rect_a[3]
    b_top, b_bottom = rect_b[1], rect_b[3]
    if a_bottom < b_top:
        return b_top - a_bottom
    if b_bottom < a_top:
        return a_top - b_bottom
    return 0.0


def vertical_overlap_ratio(rect_a, rect_b):
    """
    Fraction of the SHORTER rect's height that overlaps vertically with
    the other rect. Used to decide if two boxes sit on the same text row.
    """
    a_top, a_bottom = rect_a[1], rect_a[3]
    b_top, b_bottom = rect_b[1], rect_b[3]
    overlap = max(0.0, min(a_bottom, b_bottom) - max(a_top, b_top))
    shorter = min(a_bottom - a_top, b_bottom - b_top)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def horizontal_overlap_ratio(rect_a, rect_b):
    """
    Fraction of the narrower rect's width that overlaps horizontally with
    the other rect. Useful to decide if two text lines belong to the same
    column/block.
    """
    a_left, a_right = rect_a[0], rect_a[2]
    b_left, b_right = rect_b[0], rect_b[2]
    overlap = max(0.0, min(a_right, b_right) - max(a_left, b_left))
    narrower_width = min(a_right - a_left, b_right - b_left)
    if narrower_width <= 0:
        return 0.0
    return overlap / narrower_width


def horizontal_distance(rect_a, rect_b):
    a_left, a_right = rect_a[0], rect_a[2]
    b_left, b_right = rect_b[0], rect_b[2]
    if a_right < b_left:
        return b_left - a_right
    if b_right < a_left:
        return a_left - b_right
    return 0.0


def euclidean_center_distance(rect_a, rect_b):
    ca = rect_center(rect_a)
    cb = rect_center(rect_b)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def median_line_height(rects):
    if not rects:
        return 20.0
    heights = [rect_height(r) for r in rects if rect_height(r) > 0]
    if not heights:
        return 20.0
    return float(np.median(heights))


# --------------------------------------------------------------------------
# Line grouping: merge fragmented OCR boxes into whole visual lines
# --------------------------------------------------------------------------

def group_ocr_boxes_into_lines(items, image_width=None, y_overlap_threshold=None, max_h_gap_factor=None):
    """
    Groups enriched OCR items (dicts with 'rect', 'text', 'norm_text',
    'confidence') that sit on the same visual line into a single "line"
    record. Two items are merged if they vertically overlap enough AND
    the horizontal gap between them is small relative to line height
    (this keeps same-row fragments like "Energy" + "123 kcal" together)
    -- BUT the horizontal gap must also be within a fraction of the
    image width. This second cap is essential: without it, two
    side-by-side sections (e.g. a Nutrition table on the left and an
    Ingredients paragraph on the right) can have their same-row
    fragments incorrectly merged into one full-width "line" whenever the
    page gutter between them is narrower than
    line_height * max_h_gap_factor - which is a common real-photo case
    and was the root cause of Ingredients/Nutrition boxes collapsing
    into one shared region. Passing `image_width` (recommended: the
    working image's width) activates this cap; if omitted, only the
    line-height-based cap applies (kept for backward compatibility).

    Returns a list of line dicts:
        {
            "text": "merged text left-to-right",
            "rect": [x1,y1,x2,y2]  (union of member rects),
            "confidence": average confidence of members,
            "items": [original OCR items, in left-to-right order],
        }
    """
    if not items:
        return []

    y_overlap_threshold = (
        y_overlap_threshold
        if y_overlap_threshold is not None
        else config.LINE_GROUP_Y_OVERLAP_THRESHOLD
    )
    max_h_gap_factor = (
        max_h_gap_factor if max_h_gap_factor is not None else config.LINE_GROUP_MAX_H_GAP_FACTOR
    )

    rects = [it["rect"] for it in items]
    line_h = median_line_height(rects)
    max_h_gap = line_h * max_h_gap_factor

    if image_width:
        width_cap = image_width * config.LINE_GROUP_MAX_H_GAP_WIDTH_RATIO
        max_h_gap = min(max_h_gap, width_cap)

    n = len(items)
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
        for j in range(i + 1, n):
            ri, rj = rects[i], rects[j]
            if vertical_overlap_ratio(ri, rj) >= y_overlap_threshold:
                if horizontal_distance(ri, rj) <= max_h_gap:
                    union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(items[i])

    lines = []
    for members in groups.values():
        members_sorted = sorted(members, key=lambda it: it["rect"][0])
        merged_text = " ".join(m["text"] for m in members_sorted)
        merged_rect = union_rect([m["rect"] for m in members_sorted])
        avg_conf = float(np.mean([m.get("confidence", 0.0) for m in members_sorted]))

        lines.append(
            {
                "text": merged_text,
                "rect": merged_rect,
                "confidence": avg_conf,
                "items": members_sorted,
            }
        )

    # sort lines in reading order: top-to-bottom, then left-to-right
    lines.sort(key=lambda ln: (ln["rect"][1], ln["rect"][0]))
    return lines


def sort_reading_order(lines):
    """Sort line/item dicts (with 'rect') into top-to-bottom, left-to-right order."""
    if not lines:
        return lines
    line_h = median_line_height([ln["rect"] for ln in lines])
    band = max(line_h * 0.6, 5.0)

    def key(ln):
        y_band = round(ln["rect"][1] / band)
        return (y_band, ln["rect"][0])

    return sorted(lines, key=key)


# --------------------------------------------------------------------------
# Column clustering: separate side-by-side sections
# --------------------------------------------------------------------------

def cluster_into_columns(lines, image_width, min_gap_ratio=None):
    """
    Splits `lines` into left-to-right column groups. We look at the
    horizontal span [rect[0], rect[2]] of every line, sort by left edge,
    and cut a new column whenever there's a gap between the running
    rightmost edge and the next line's left edge that's wider than
    min_gap_ratio * image_width - i.e. a genuine page gutter rather than
    normal word/paragraph spacing.

    Returns a list of column groups, each a list of lines, ordered
    left-to-right. If no wide gutter is found, returns a single column
    containing all lines.
    """
    if not lines:
        return []

    min_gap_ratio = min_gap_ratio if min_gap_ratio is not None else config.COLUMN_GAP_MIN_WIDTH_RATIO
    min_gap = image_width * min_gap_ratio

    by_left = sorted(lines, key=lambda ln: ln["rect"][0])

    columns = []
    current = [by_left[0]]
    current_right = by_left[0]["rect"][2]

    for ln in by_left[1:]:
        gap = ln["rect"][0] - current_right
        if gap > min_gap:
            columns.append(current)
            current = [ln]
            current_right = ln["rect"][2]
        else:
            current.append(ln)
            current_right = max(current_right, ln["rect"][2])

    columns.append(current)

    columns = [sort_reading_order(col) for col in columns]
    return columns


def find_column_for_line(target_line, columns):
    """Returns the column (list of lines) that contains target_line, by identity."""
    for col in columns:
        for ln in col:
            if ln is target_line:
                return col
    return None


# --------------------------------------------------------------------------
# Clustering by proximity (used by vocabulary / table fallbacks)
# --------------------------------------------------------------------------

def cluster_by_proximity(items, max_gap_factor=3.0):
    """
    Simple single-linkage clustering of items (dicts with 'rect') based on
    center-to-center distance relative to median line height. Returns a
    list of clusters, each a list of items.
    """
    if not items:
        return []

    line_h = median_line_height([it["rect"] for it in items])
    max_dist = max(line_h * max_gap_factor, 15.0)

    n = len(items)
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
        for j in range(i + 1, n):
            d = euclidean_center_distance(items[i]["rect"], items[j]["rect"])
            if d <= max_dist:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(items[i])

    return list(clusters.values())


# --------------------------------------------------------------------------
# Region finalization helpers
# --------------------------------------------------------------------------

def clip_rect(bbox, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [x1, y1, x2, y2]


def add_safe_margin(bbox, image_shape, margin_ratio=None, margin_ratio_x=None, margin_ratio_y=None):
    """
    Expands bbox = [x1,y1,x2,y2] by a small margin (relative to image
    size), then clips to image bounds. `margin_ratio` sets both axes;
    `margin_ratio_x`/`margin_ratio_y` override it per-axis if given.

    NOTE: region detectors (detect_ingredient_region/detect_nutrition_region)
    intentionally do NOT call this anymore - they return a tight, unpadded
    bbox. The single padding step lives in preprocessing/image_utils.pad_bbox,
    applied once at final-crop time in main.py. Calling this AND pad_bbox on
    the same bbox would double the margin.
    """
    margin_ratio = margin_ratio if margin_ratio is not None else config.REGION_PADDING_RATIO
    margin_ratio_x = margin_ratio_x if margin_ratio_x is not None else margin_ratio
    margin_ratio_y = margin_ratio_y if margin_ratio_y is not None else margin_ratio
    h, w = image_shape[:2]
    pad_x = int(round(w * margin_ratio_x))
    pad_y = int(round(h * margin_ratio_y))

    x1, y1, x2, y2 = bbox
    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    return clip_rect([x1, y1, x2, y2], image_shape)


def bbox_area(rect):
    if rect is None:
        return 0.0
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def bbox_intersection_area(rect_a, rect_b):
    if rect_a is None or rect_b is None:
        return 0.0
    x1 = max(rect_a[0], rect_b[0])
    y1 = max(rect_a[1], rect_b[1])
    x2 = min(rect_a[2], rect_b[2])
    y2 = min(rect_a[3], rect_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def bbox_iou(rect_a, rect_b):
    if rect_a is None or rect_b is None:
        return 0.0
    inter = bbox_intersection_area(rect_a, rect_b)
    if inter <= 0:
        return 0.0
    union = bbox_area(rect_a) + bbox_area(rect_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def validate_region(bbox, image_shape, min_width_px=8, min_height_px=8):
    """
    Sanity-checks a candidate bbox: not None, not degenerate, clipped to
    image bounds. Returns a valid clipped bbox, or None if the region is
    unusable (too small / inverted).
    """
    if bbox is None:
        return None
    clipped = clip_rect(bbox, image_shape)
    x1, y1, x2, y2 = clipped
    if (x2 - x1) < min_width_px or (y2 - y1) < min_height_px:
        return None
    return clipped
