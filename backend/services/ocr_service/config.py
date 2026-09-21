"""
config.py

Central configuration for PicWise.

Everything that is "tunable" lives here: vocabulary lists, thresholds,
resize targets, fuzzy-matching cutoffs, file paths, etc. Nothing in this
file talks to OpenCV/PaddleOCR directly - it is pure data + simple helpers.
"""

import os
from backend import config as backend_config

# Constrain native OpenMP and MKL thread pools for local runtime stability
backend_config.setup_runtime_environment()

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PICWISE_ROOT = str(backend_config.PROJECT_ROOT)

INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, "knowledge_base")

_DEFAULT_ING_CSV = os.path.join(PICWISE_ROOT, backend_config.DEFAULT_FOOD_DATA_PATH)
_FALLBACK_ING_CSV = os.path.join(PICWISE_ROOT, backend_config.FALLBACK_FOOD_ALT_DATA_PATH)
_DEFAULT_NUT_CSV = os.path.join(PICWISE_ROOT, backend_config.DEFAULT_NUTRITION_DATA_PATH)
_DEFAULT_PC_XLSX = os.path.join(PICWISE_ROOT, backend_config.FALLBACK_PERSONAL_CARE_DATA_PATH)

_resolved_ing = backend_config.resolve_food_kb_path()
INGREDIENT_KB_CSV = _resolved_ing if os.path.isabs(_resolved_ing) else os.path.join(PICWISE_ROOT, _resolved_ing)

_resolved_nut = backend_config.resolve_nutrition_kb_path()
NUTRITION_KB_CSV = _resolved_nut if os.path.isabs(_resolved_nut) else os.path.join(PICWISE_ROOT, _resolved_nut)

_resolved_pc = backend_config.resolve_personal_care_kb_path()
PERSONAL_CARE_KB_XLSX = _resolved_pc if os.path.isabs(_resolved_pc) else os.path.join(PICWISE_ROOT, _resolved_pc)

# --------------------------------------------------------------------------
# IMAGE NORMALIZATION
# --------------------------------------------------------------------------

MIN_USEFUL_OCR_HEIGHT = 1000   # px - upscale if smaller than this
MAX_IMAGE_DIMENSION = 2800     # px - downscale if larger than this
UPSCALE_INTERPOLATION = "cubic"

# --------------------------------------------------------------------------
# PACKET DETECTION (OpenCV fallback, no YOLO)
# --------------------------------------------------------------------------

PACKET_MIN_AREA_RATIO = 0.15      # candidate contour must cover >= 15% of image
PACKET_MIN_CONFIDENCE_TO_CROP = 0.55
PACKET_CANNY_LOW = 50
PACKET_CANNY_HIGH = 150
PACKET_MORPH_KERNEL = (9, 9)

# --------------------------------------------------------------------------
# OCR ENGINE
# --------------------------------------------------------------------------

OCR_LANG = "en"
OCR_USE_ANGLE_CLS = True

# --------------------------------------------------------------------------
# VOCABULARY: INGREDIENT ANCHORS
# --------------------------------------------------------------------------

INGREDIENT_ANCHORS = [
    "ingredients",
    "ingredient",
    "ingredients:",
    "ingredient:",
    "composition",
    "made from",
    "made with",
    "contains",
]

INGREDIENT_ANCHORS_HI = [
    "सामग्री",
    "सामग्रियां",
    "घटक",
]

PERSONAL_CARE_ANCHORS = [
    "ingredients",
    "composition",
    "key ingredients",
    "active ingredients",
    "other ingredients",
    "contains",
]

ALL_INGREDIENT_ANCHORS = list(
    dict.fromkeys(
        INGREDIENT_ANCHORS + INGREDIENT_ANCHORS_HI + PERSONAL_CARE_ANCHORS
    )
)

# --------------------------------------------------------------------------
# VOCABULARY: NUTRITION ANCHORS
# --------------------------------------------------------------------------

NUTRITION_ANCHORS = [
    "nutrition information",
    "nutritional information",
    "nutrition facts",
    "nutritional facts",
    "nutrition",
    "nutrient information",
    "nutrition facts panel",
]

NUTRIENT_KEYWORDS = [
    "energy",
    "calories",
    "protein",
    "total protein",
    "carbohydrate",
    "total carbohydrate",
    "total carbohydrates",
    "sugar",
    "total sugar",
    "total sugars",
    "added sugar",
    "added sugars",
    "fat",
    "total fat",
    "saturated fat",
    "trans fat",
    "dietary fibre",
    "fiber",
    "fibre",
    "sodium",
    "salt",
    "potassium",
    "calcium",
    "iron",
    "vitamin",
    "vitamin a",
    "vitamin c",
    "vitamin d",
]

# Units commonly seen next to nutrient values - used for table-geometry scoring
NUTRIENT_UNITS = ["kcal", "kj", "g", "mg", "mcg", "iu", "%"]

# --------------------------------------------------------------------------
# SECTION STOP WORDS (used to bound the ingredient section)
# --------------------------------------------------------------------------

SECTION_STOP_WORDS = [
    "nutrition",
    "nutrition facts",
    "nutrition information",
    "nutritional information",
    "allergen",
    "allergen information",
    "allergen advice",
    "allergy advice",
    "storage",
    "storage instructions",
    "directions",
    "cooking instructions",
    "preparation",
    "net weight",
    "net wt",
    "net quantity",
    "mrp",
    "batch",
    "batch no",
    "manufactured by",
    "marketed by",
    "customer care",
    "fssai",
    "barcode",
    "expiry",
    "best before",
    "use by",
]

# --------------------------------------------------------------------------
# DOMAIN CLASSIFICATION SIGNALS
# --------------------------------------------------------------------------

FOOD_SIGNALS = [
    "nutrition",
    "ingredients",
    "energy",
    "protein",
    "carbohydrate",
    "fat",
    "sugar",
    "serving",
    "calories",
]

PERSONAL_CARE_SIGNALS = [
    "shampoo",
    "conditioner",
    "moisturizer",
    "cream",
    "serum",
    "soap",
    "face wash",
    "body wash",
    "toothpaste",
    "cosmetic",
]

# --------------------------------------------------------------------------
# FUZZY MATCHING
# --------------------------------------------------------------------------

FUZZY_ANCHOR_THRESHOLD = 78        # threshold for matching OCR text -> anchor words
FUZZY_KB_MATCH_THRESHOLD = 85      # threshold for matching OCR token -> knowledge base entry
FUZZY_KB_MATCH_MIN_ACCEPT = 80     # never accept anything below this even if "best available"

# --------------------------------------------------------------------------
# REGION EXPANSION
# --------------------------------------------------------------------------

# How far (in units of median line-height) we look for continuation lines
# below/right/left of an anchor before deciding the section has ended.
REGION_EXPANSION_MAX_LINE_GAP_FACTOR = 2.6
REGION_EXPANSION_MAX_LINES = 75

# Safety cap on total collected block height (in units of median OCR line
# height), independent of the per-line vertical-gap check above.
REGION_MAX_BLOCK_HEIGHT_LINE_FACTOR = 45

# Final crop padding: kept SMALL and applied ONCE, at final-crop time
# (see preprocessing/image_utils.pad_bbox, called from main.py). Region
# detectors (detect_ingredient_region / detect_nutrition_region /
# reconcile_regions) return a TIGHT, unpadded bbox - they must NOT also
# call add_safe_margin, or the margin gets applied twice (once inside the
# detector, once again in main.py's pad_bbox before cropping), silently
# doubling the padding and being a real contributor to "crop is too
# large" bug reports. Kept small (~1.5%) per spec ("approximately 1-2% of
# image width/height"). Separate X/Y knobs so horizontal and vertical
# margins can be tuned independently.
REGION_PADDING_RATIO = 0.015          # legacy single-value fallback
REGION_PADDING_RATIO_X = 0.015        # ~1.5% of image width
REGION_PADDING_RATIO_Y = 0.015        # ~1.5% of image height

# --------------------------------------------------------------------------
# LINE GROUPING (merging fragmented OCR boxes that belong to one visual line)
# --------------------------------------------------------------------------

# Two OCR boxes are considered the same visual line if their vertical
# (y-range) overlap ratio is >= this, AND their horizontal gap is within
# BOTH LINE_GROUP_MAX_H_GAP_FACTOR * median_line_height AND
# LINE_GROUP_MAX_H_GAP_WIDTH_RATIO * image_width (the smaller of the two
# wins). The width-ratio cap is what stops two side-by-side sections
# (e.g. a Nutrition table and an Ingredients paragraph) from having their
# same-row fragments incorrectly merged into one full-width "line" when
# the page gutter between them happens to be narrower than
# line_height * LINE_GROUP_MAX_H_GAP_FACTOR would otherwise allow.
LINE_GROUP_Y_OVERLAP_THRESHOLD = 0.35
LINE_GROUP_MAX_H_GAP_FACTOR = 6.0
LINE_GROUP_MAX_H_GAP_WIDTH_RATIO = 0.05

# --------------------------------------------------------------------------
# COLUMN CLUSTERING (SOFT signal only - see detection/geometry.py
# cluster_into_columns). Per-section detectors do NOT use this as a hard
# gate; independent, self-limiting expansion (cross-section line
# disqualification) is the real mechanism that keeps sections apart. This
# threshold remains only for optional debug annotation.
# --------------------------------------------------------------------------
COLUMN_GAP_MIN_WIDTH_RATIO = 0.025

# --------------------------------------------------------------------------
# REGION EXPANSION: LEFT-MARGIN / BAND TOLERANCE
# --------------------------------------------------------------------------

# A candidate line is accepted into a growing ingredient/nutrition block if
# its left edge is within this many median-line-heights of the *current*
# column band's left edge (paragraphs/tables are usually left-aligned),
# OR if it horizontally overlaps the band at all.
REGION_BAND_LEFT_TOLERANCE_FACTOR = 3.0

# When a candidate line's left edge is NOT close to the growing band's
# left edge (see REGION_BAND_LEFT_TOLERANCE_FACTOR above), it used to be
# accepted anyway if it overlapped the band by ANY amount (> 0 px). That
# is too permissive: as the band widens from earlier accepted lines, even
# a 1px sliver of horizontal overlap with a completely unrelated,
# far-off-column line (e.g. a manufacturer address that happens to start
# where the ingredients paragraph happens to end) was enough to pull it
# in, and every subsequent line only made the band wider - a runaway
# feedback loop that is a real contributor to oversized crops. Now a
# non-left-aligned line must overlap the band by at least this fraction
# of its own (narrower) width to be accepted.
REGION_BAND_OVERLAP_MIN_RATIO = 0.25

# --------------------------------------------------------------------------
# STOP-WORD CONTEXTUAL VALIDATION
# --------------------------------------------------------------------------

# Stop-word matches at/above this fuzzy score are treated as authoritative
# (always stop) regardless of context.
STOP_WORD_AUTO_SCORE = 92

# Stop-word matches at/above this (lower) score require contextual
# confirmation before triggering a stop (see detect_section_boundaries).
STOP_WORD_CONTEXTUAL_SCORE = 78

# "contains" is ambiguous: it's both a weak ingredient anchor phrase and
# commonly the lead-in to an allergen-advice line ("Contains: Wheat, Milk").
# We only trust "contains" as a STOP if the line also mentions a common
# allergen term - otherwise it's left alone (neither forces a stop nor is
# it required to be an ingredient anchor by itself).
ALLERGEN_CONTEXT_WORDS = [
    "wheat", "milk", "soy", "soya", "nut", "nuts", "peanut", "egg", "eggs",
    "gluten", "sesame", "shellfish", "fish", "mustard", "celery", "lupin",
    "sulphite", "sulfite", "tree nut", "dairy",
]

# --------------------------------------------------------------------------
# NUTRITION ROW / TABLE DETECTION (fallback + contiguous-inclusion pass)
# --------------------------------------------------------------------------

# After locating the vertical span of nutrient-keyword rows, we do a
# second pass that also absorbs adjacent non-keyword lines (headers like
# "per 100g", "Serving Size", footnotes like "% Daily Value*") as long as
# the gap to the nearest keyword row is within this many line-heights.
NUTRITION_HEADER_FOOTER_GAP_FACTOR = 2.0

# When aligning nutrient-name column vs value column within a row cluster,
# lines whose left-x sits within this many line-heights of the table's
# established left margin are treated as belonging to the same table
# column (used to pull in additional un-keyworded rows, e.g. "Vitamin B12").
NUTRITION_ROW_X_TOLERANCE_FACTOR = 4.0

# --------------------------------------------------------------------------
# CROSS-SECTION LINE CLASSIFICATION
# --------------------------------------------------------------------------

# A line is treated as a strong "nutrition row" (and therefore
# disqualified from ever joining an Ingredients region, regardless of
# spatial proximity) if it contains a nutrient keyword AND a number+unit
# pattern, e.g. "Protein 8 g" or "Energy 123 kcal".
NUTRITION_ROW_UNIT_PATTERN = r"\d+(\.\d+)?\s*(kcal|kj|mcg|µg|mg|ml|g\b|%)"

# A line is treated as a strong "ingredient paragraph" line (and
# therefore disqualified from ever joining a Nutrition region) if it has
# at least this many comma-separated items AND does not itself look like
# a nutrition row.
INGREDIENT_LINE_MIN_COMMAS = 2

# --------------------------------------------------------------------------
# INDEPENDENT MULTI-CANDIDATE DETECTION
# --------------------------------------------------------------------------

# Each detector tries this many top-scoring anchor matches (not just the
# single best) as independent starting points, expands each into its own
# candidate region, and keeps the highest-scoring valid candidate.
NUM_ANCHOR_CANDIDATES_TO_TRY = 3

# --------------------------------------------------------------------------
# FINAL OVERLAP RECONCILIATION
# --------------------------------------------------------------------------

# After Ingredients and Nutrition are detected completely independently,
# if their IoU exceeds this, we don't merge them - we reassign contested
# lines to whichever section they match better and shrink both boxes to
# their (now mutually exclusive) line sets.
REGION_OVERLAP_IOU_RECONCILE_THRESHOLD = 0.15

# --------------------------------------------------------------------------
# DEBUG
# --------------------------------------------------------------------------

# When True (wired to --test-mode in main.py), region detectors return an
# extra "debug" dict: matched anchor, accepted/rejected lines with
# reasons, stop reason, and candidate scores.
DEBUG_REGION_DETECTION = False

# --------------------------------------------------------------------------
# NUTRITION TABLE DETECTION (fallback, no heading found)
# --------------------------------------------------------------------------

NUTRITION_TABLE_MIN_KEYWORDS = 3          # need >=3 distinct nutrient terms nearby
NUTRITION_TABLE_CLUSTER_DIST_FACTOR = 3.0  # x line-height, for clustering boxes

# --------------------------------------------------------------------------
# VISUALIZATION COLORS (BGR, since OpenCV uses BGR)
# --------------------------------------------------------------------------

COLOR_INGREDIENTS = (0, 200, 0)     # GREEN
COLOR_NUTRITION = (200, 100, 0)     # BLUE-ish
COLOR_OTHER_TEXT = (128, 128, 128)  # GRAY
COLOR_PACKET = (0, 200, 200)        # YELLOW-ish, for packet bbox

# --------------------------------------------------------------------------
# MISC
# --------------------------------------------------------------------------

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

DEFAULT_DOMAIN = "auto"  # auto | food | personal_care

# --------------------------------------------------------------------------
# V2.2 ARCHITECTURE CONFIGURATION (POLYGON-AWARE, COLUMN-FIRST, LINE OWNERSHIP)
# --------------------------------------------------------------------------
# Line grouping vertical center similarity, horizontal gap, and angle constraints
LINE_BUILDER_V_TOLERANCE_FACTOR = 0.45
LINE_BUILDER_H_GAP_FACTOR = 3.5
LINE_BUILDER_HEIGHT_TOLERANCE_FACTOR = 0.35
ANGLE_COMPATIBILITY_THRESHOLD = 12.0  # degrees

# Column detection
COLUMN_GAP_MIN_WIDTH_RATIO = 0.08

# Block clustering distance tolerance (in median line-heights)
BLOCK_DETECTOR_GAP_FACTOR = 2.2
BLOCK_DETECTOR_MIN_OVERLAP_RATIO = 0.2

# Line ownership & boundary validation
MIN_OWNERSHIP_SCORE = 0.25
MIN_OWNERSHIP_MARGIN = 0.15
MIN_INGREDIENT_CONTINUATION_SCORE = 0.20
MIN_NUTRITION_ROW_SCORE = 0.25
STOP_SECTION_SCORE_THRESHOLD = 0.55
INCLUDE_ALLERGEN_IN_INGREDIENTS = False

# Tight polygon ROI margin scaling
POLYGON_ROI_MARGIN_LINE_FACTOR = 0.25   # factor * median line height
POLYGON_ROI_MARGIN_IMAGE_RATIO = 0.005  # fraction of image dimension

# Semantic class signal keywords
ALLERGEN_SIGNALS = [
    "allergen advice", "allergy advice", "allergen information",
    "allergen info", "contains wheat", "contains milk", "contains soy",
    "contains peanut", "contains tree nut", "contains nuts", "contains egg",
    "may contain"
]

INSTRUCTION_SIGNALS = [
    "directions for use", "directions", "instructions", "how to use",
    "how to prepare", "cooking instructions", "preparation", "prep method",
    "pump activation procedure", "suggested use", "dosage", "precautions",
    "warning", "caution", "for best results", "step 1", "step 2"
]

MANUFACTURER_SIGNALS = [
    "manufactured by", "mfg by", "marketed by", "mkt by", "packed by",
    "pkg by", "imported by", "distributor", "manufactured for",
    "factory", "regd office", "registered office", "corporate office",
    "plant address", "mfg lic no", "manufacturing unit"
]

CONTACT_SIGNALS = [
    "customer care", "consumer care", "contact us", "reach us", "helpline",
    "toll free", "feedback", "queries", "email:", "call:", "write to",
    "consumer cell", "customer support"
]

STORAGE_SIGNALS = [
    "storage instructions", "storage:", "store in", "keep in", "store away",
    "cool and dry place", "cool, dry", "refrigerate", "do not freeze",
    "keep container tightly", "keep out of reach"
]

MRP_SIGNALS = [
    "mrp", "m.r.p", "maximum retail price", "incl. of all taxes",
    "batch no", "lot no", "batch:", "lot:", "net weight", "net wt",
    "net quantity", "net qty", "pkg date", "pkd", "mfd date", "mfg date",
    "best before", "use before", "use by", "expiry date", "exp date", "exp."
]

LEGAL_SIGNALS = [
    "fssai", "lic. no.", "lic no", "lic.no", "mfg.lic", "mfg lic", "m.l.no", "ml no", "standard", "statutory", "hallmark",
    "registered tm", "trademark", "patent", "isi mark", "cert.", "under license"
]

MARKETING_SIGNALS = [
    "taste the goodness", "100% natural", "no artificial", "guaranteed quality",
    "heritage of quality", "crispy and crunchy", "crunchy",
    "irresistible", "crafted with love", "premium quality", "our promise",
    "provides strong protection", "protection from germs", "keeps your skin",
    "moisturizing", "with soothing", "clinically proven", "dermatologist recommended",
    "trusted protection"
]

# Unified candidate scoring weights
WEIGHT_ING_SEMANTIC = 0.35
WEIGHT_ING_VOCAB = 0.25
WEIGHT_ING_HEADING = 0.15
WEIGHT_ING_SPATIAL = 0.15
WEIGHT_ING_OCR = 0.10

WEIGHT_NUT_VOCAB = 0.35
WEIGHT_NUT_NUM_UNIT = 0.20
WEIGHT_NUT_TABLE = 0.20
WEIGHT_NUT_HEADING = 0.15
WEIGHT_NUT_SPATIAL = 0.10

