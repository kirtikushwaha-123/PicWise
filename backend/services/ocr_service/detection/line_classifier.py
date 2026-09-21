"""
detection/line_classifier.py

Classifies logical text lines semantically by computing three distinct scores:
- ingredient_score
- nutrition_score
- other_section_score

Uses a rich set of keyword cues, structural regex patterns (commas, units, parentheses,
INS numbers), and vocabulary matching.
"""

import re
from backend.services.ocr_service import config
from backend.services.ocr_service.detection.ocr_detector import normalize_ocr_text, best_anchor_match

# Regex patterns
_INS_NUMBER_RE = re.compile(r'\b(?:ins|e)\s*\d+[a-z]?(\([i|v|x]+\))?\b', re.IGNORECASE)
_PERCENTAGE_RE = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|percent)\b', re.IGNORECASE)
_PARENTHESIS_RE = re.compile(r'\(.*?\)')
_NUTRITION_UNIT_RE = re.compile(config.NUTRITION_ROW_UNIT_PATTERN, re.IGNORECASE)
_NUMERIC_RE = re.compile(r'\b\d+(?:\.\d+)?\b')

_ING_CUES = [
    "chill", "cumin", "cumn", "pepper", "sesame", "flour", "sugar", "salt", "water",
    "oil", "fat", "acid", "aciit", "spice", "masala", "starch", "extract", "flavor",
    "flavour", "colour", "color", "preservative", "regulator", "regl", "emulsifier",
    "wheat", "rice", "corn", "syrup", "butter", "milk", "cocoa", "paste", "powder",
    "peanut", "almond", "cashew", "garlic", "ginger", "onion", "tomato", "potato",
    "yeast", "dextrose", "maltodextrin", "lecithin", "gluten"
]

_INS_ADDITIVE_RE = re.compile(
    r'(?:(?:ins|e|reg|regl|regulator|stabilizer|emulsifier|thickener|acid|aciit|preservative|antioxidant)[a-z0-9\s]*\d{3,4}[a-z]?|\b[1-9]\d{2}[a-z]?\b)',
    re.IGNORECASE
)

def classify_lines(lines, ingredient_vocab=None):
    """
    Computes semantic classification scores for each logical line.
    
    Args:
        lines: list of logical line dicts (from line_builder)
        ingredient_vocab: list/iterable of known ingredient name strings

    Returns:
        list of logical line dicts, each augmented with:
            {
                "ingredient_score": float (0.0 to 1.0),
                "nutrition_score": float (0.0 to 1.0),
                "other_section_score": float (0.0 to 1.0)
            }
    """
    if not lines:
        return []

    vocab_lower = [v.lower() for v in ingredient_vocab if len(v) >= 4] if ingredient_vocab else []

    classified_lines = []
    for ln in lines:
        text = ln["text"]
        norm = normalize_ocr_text(text)

        # -------------------------------------------------------------
        # 1. INGREDIENT SCORE COMPUTATION
        # -------------------------------------------------------------
        ing_score = 0.0

        # Anchor heading match
        matched_ing_anchor, ing_anchor_score = best_anchor_match(
            text, config.ALL_INGREDIENT_ANCHORS, threshold=config.FUZZY_ANCHOR_THRESHOLD
        )
        if matched_ing_anchor:
            ing_score += 0.85 * (ing_anchor_score / 100.0)

        # Comma separated items
        comma_count = text.count(",")
        if comma_count >= 1:
            ing_score += min(0.4, 0.15 * comma_count)
        
        # Parentheses
        if _PARENTHESIS_RE.search(text):
            ing_score += 0.15

        # Percentages
        if _PERCENTAGE_RE.search(norm):
            ing_score += 0.2

        # INS / E-numbers / Additive codes
        if _INS_NUMBER_RE.search(norm) or _INS_ADDITIVE_RE.search(norm):
            ing_score += 0.35

        # Ingredient stems / food keywords
        cues_hits = sum(1 for cue in _ING_CUES if cue in norm)
        if cues_hits > 0:
            ing_score += min(0.40, 0.12 * cues_hits)

        # Ingredient vocabulary density (check full list)
        if vocab_lower:
            vocab_hits = sum(1 for v in vocab_lower if v in norm)
            if vocab_hits > 0:
                ing_score += min(0.45, 0.15 * vocab_hits)

        ing_score = min(1.0, max(0.0, ing_score))

        # -------------------------------------------------------------
        # 2. NUTRITION SCORE COMPUTATION
        # -------------------------------------------------------------
        nut_score = 0.0

        # Anchor heading match
        matched_nut_anchor, nut_anchor_score = best_anchor_match(
            text, config.NUTRITION_ANCHORS, threshold=config.FUZZY_ANCHOR_THRESHOLD
        )
        if matched_nut_anchor:
            nut_score += 0.9 * (nut_anchor_score / 100.0)

        # Nutrient keywords
        nut_hits = sum(1 for kw in config.NUTRIENT_KEYWORDS if kw in norm)
        if nut_hits > 0:
            nut_score += min(0.45, 0.15 * nut_hits)

        # Number + Unit pattern
        if _NUTRITION_UNIT_RE.search(norm):
            nut_score += 0.35

        # Serving information / headers
        serving_kws = ["serving", "servings", "serves", "pack size", "per 100g", "per 100 g", "per serving", "approx."]
        if any(sk in norm for sk in serving_kws):
            nut_score += 0.25

        # Check raw numeric alignment/presence
        numeric_count = len(_NUMERIC_RE.findall(norm))
        if numeric_count >= 2:
            nut_score += min(0.2, 0.08 * numeric_count)

        nut_score = min(1.0, max(0.0, nut_score))

_PHONE_RE = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}')
_EMAIL_RE = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
_URL_RE = re.compile(r'(?:https?://|www\.)[^\s]+|[a-zA-Z0-9-]+\.(?:com|in|org|net|co)\b')
_MRP_RE = re.compile(r'(?:rs\.?|₹|inr)\s*\d+(?:\.\d{1,2})?|\bmrp\b', re.IGNORECASE)
_BATCH_RE = re.compile(r'\b(?:batch|lot|b\.no|bno)\b[:.\s]*[a-zA-Z0-9-]+', re.IGNORECASE)
_DATE_RE = re.compile(r'\b(?:mfd|mfg|exp|use\s*by|best\s*before)\b[:.\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', re.IGNORECASE)


def classify_lines(lines, ingredient_vocab=None):
    """
    Computes 11 independent semantic classification scores for each logical line:
    - ingredient_score
    - nutrition_score
    - allergen_score
    - instruction_score
    - manufacturer_score
    - contact_score
    - storage_score
    - mrp_score
    - legal_score
    - marketing_score
    - other_score
    """
    if not lines:
        return []

    vocab_lower = [v.lower() for v in ingredient_vocab if len(v) >= 4] if ingredient_vocab else []

    classified_lines = []
    for ln in lines:
        text = ln["text"]
        norm = normalize_ocr_text(text)

        # -------------------------------------------------------------
        # 1. INGREDIENT SCORE
        # -------------------------------------------------------------
        ing_score = 0.0
        matched_ing_anchor, ing_anchor_score = best_anchor_match(
            text, config.ALL_INGREDIENT_ANCHORS, threshold=config.FUZZY_ANCHOR_THRESHOLD
        )
        if matched_ing_anchor and matched_ing_anchor != "contains":
            ing_score += 0.85 * (ing_anchor_score / 100.0)

        comma_count = text.count(",")
        if comma_count >= 2:
            ing_score += min(0.40, 0.15 * comma_count)
        elif comma_count == 1:
            ing_score += 0.10

        if _PARENTHESIS_RE.search(text):
            ing_score += 0.15

        if _PERCENTAGE_RE.search(norm):
            ing_score += 0.15

        if _INS_NUMBER_RE.search(norm):
            ing_score += 0.35

        # Common personal care / food chemical & natural ingredients
        common_ingredients = [
            "water", "aqua", "sugar", "salt", "oil", "palm", "wheat", "flour", "starch",
            "glycerin", "glycol", "acid", "sodium", "extract", "parfum", "fragrance",
            "emulsifier", "stabilizer", "preservative", "colour", "flavour", "vitamins",
            "protein", "cocoa", "milk", "syrup", "maltodextrin", "spices", "alcohol",
            "sulfate", "sulphate", "sorbate", "benzoate", "citric"
        ]
        ing_hits = sum(1 for ci in common_ingredients if ci in norm)
        if ing_hits > 0:
            ing_score += min(0.45, 0.12 * ing_hits)

        if vocab_lower:
            vocab_hits = sum(1 for v in vocab_lower[:400] if v in norm)
            if vocab_hits > 0:
                ing_score += min(0.35, 0.10 * vocab_hits)

        ing_score = min(1.0, max(0.0, ing_score))

        # -------------------------------------------------------------
        # 2. NUTRITION SCORE
        # -------------------------------------------------------------
        nut_score = 0.0
        matched_nut_anchor, nut_anchor_score = best_anchor_match(
            text, config.NUTRITION_ANCHORS, threshold=config.FUZZY_ANCHOR_THRESHOLD
        )
        if matched_nut_anchor:
            nut_score += 0.90 * (nut_anchor_score / 100.0)

        nut_hits = sum(1 for kw in config.NUTRIENT_KEYWORDS if kw in norm)
        if nut_hits > 0:
            nut_score += min(0.50, 0.18 * nut_hits)

        if _NUTRITION_UNIT_RE.search(norm):
            nut_score += 0.35

        serving_kws = ["serving", "servings", "serves", "pack size", "per 100g", "per 100ml", "per serving", "approx.", "rda", "gda", "daily value"]
        if any(sk in norm for sk in serving_kws):
            nut_score += 0.30

        numeric_count = len(_NUMERIC_RE.findall(norm))
        if numeric_count >= 2 and (nut_hits > 0 or _NUTRITION_UNIT_RE.search(norm)):
            nut_score += min(0.25, 0.08 * numeric_count)

        nut_score = min(1.0, max(0.0, nut_score))

        # -------------------------------------------------------------
        # 3. ALLERGEN SCORE
        # -------------------------------------------------------------
        allergen_score = 0.0
        allergen_signals = getattr(config, "ALLERGEN_SIGNALS", [
            "allergen advice", "allergy advice", "contains wheat", "contains milk", "contains soy", "may contain"
        ])
        matched_allergen, allergen_anchor_score = best_anchor_match(
            text, allergen_signals, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_allergen:
            allergen_score += 0.85 * (allergen_anchor_score / 100.0)
        if any(asig in norm for asig in allergen_signals):
            allergen_score += 0.40
        if "contains" in norm and any(cw in norm for cw in config.ALLERGEN_CONTEXT_WORDS):
            allergen_score = max(allergen_score, 0.70)
        # Safeguards: if it contains statutory ingredients (colour, flavour, caffeine) or serving/nutrition info
        if any(term in norm for term in ["colour", "color", "flavour", "flavor", "caffeine", "serve", "serving", "approx."]):
            allergen_score = 0.0
        if nut_score >= 0.35 and not any(term in norm for term in ["allergen", "allergy"]):
            allergen_score = 0.0
        allergen_score = min(1.0, max(0.0, allergen_score))

        # -------------------------------------------------------------
        # 4. INSTRUCTION SCORE
        # -------------------------------------------------------------
        instruction_score = 0.0
        instruction_signals = getattr(config, "INSTRUCTION_SIGNALS", [
            "directions for use", "directions", "instructions", "how to use", "how to prepare", "pump activation procedure"
        ])
        matched_inst, inst_anchor_score = best_anchor_match(
            text, instruction_signals, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_inst:
            instruction_score += 0.85 * (inst_anchor_score / 100.0)
        if any(isig in norm for isig in instruction_signals):
            instruction_score += 0.40
        instruction_score = min(1.0, max(0.0, instruction_score))

        # -------------------------------------------------------------
        # 5. MANUFACTURER SCORE
        # -------------------------------------------------------------
        manufacturer_score = 0.0
        mfg_signals = getattr(config, "MANUFACTURER_SIGNALS", [
            "manufactured by", "mfg by", "marketed by", "packed by", "imported by", "factory", "regd office"
        ])
        matched_mfg, mfg_anchor_score = best_anchor_match(
            text, mfg_signals, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_mfg:
            manufacturer_score += 0.85 * (mfg_anchor_score / 100.0)
        if any(ms in norm for ms in mfg_signals):
            manufacturer_score += 0.40
        manufacturer_score = min(1.0, max(0.0, manufacturer_score))

        # -------------------------------------------------------------
        # 6. CONTACT SCORE
        # -------------------------------------------------------------
        contact_score = 0.0
        contact_signals = getattr(config, "CONTACT_SIGNALS", [
            "customer care", "consumer care", "contact us", "helpline", "toll free", "feedback", "queries"
        ])
        matched_contact, contact_anchor_score = best_anchor_match(
            text, contact_signals, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_contact:
            contact_score += 0.85 * (contact_anchor_score / 100.0)
        if any(cs in norm for cs in contact_signals):
            contact_score += 0.40
        if _PHONE_RE.search(text) or _EMAIL_RE.search(text):
            contact_score += 0.35
        contact_score = min(1.0, max(0.0, contact_score))

        # -------------------------------------------------------------
        # 7. STORAGE SCORE
        # -------------------------------------------------------------
        storage_score = 0.0
        storage_signals = getattr(config, "STORAGE_SIGNALS", [
            "storage instructions", "storage:", "store in", "keep in", "cool and dry", "refrigerate"
        ])
        matched_storage, storage_anchor_score = best_anchor_match(
            text, storage_signals, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_storage:
            storage_score += 0.85 * (storage_anchor_score / 100.0)
        if any(ss in norm for ss in storage_signals):
            storage_score += 0.40
        storage_score = min(1.0, max(0.0, storage_score))

        # -------------------------------------------------------------
        # 8. MRP / BATCH SCORE
        # -------------------------------------------------------------
        mrp_score = 0.0
        mrp_signals = getattr(config, "MRP_SIGNALS", [
            "mrp", "incl. of all taxes", "batch no", "lot no", "net weight", "net wt", "best before", "expiry date"
        ])
        matched_mrp, mrp_anchor_score = best_anchor_match(
            text, mrp_signals, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_mrp:
            mrp_score += 0.85 * (mrp_anchor_score / 100.0)
        if any(ms in norm for ms in mrp_signals):
            mrp_score += 0.35
        if _MRP_RE.search(norm) or _BATCH_RE.search(norm) or _DATE_RE.search(norm):
            mrp_score += 0.35
        mrp_score = min(1.0, max(0.0, mrp_score))

        # -------------------------------------------------------------
        # 9. LEGAL / REGULATORY SCORE
        # -------------------------------------------------------------
        legal_score = 0.0
        legal_signals = getattr(config, "LEGAL_SIGNALS", [
            "fssai", "lic. no.", "lic no", "lic.no", "mfg.lic", "mfg lic", "m.l.no", "ml no", "trademark", "patent", "standard", "statutory", "isi mark"
        ])
        matched_legal, legal_anchor_score = best_anchor_match(
            text, legal_signals, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_legal:
            legal_score += 0.85 * (legal_anchor_score / 100.0)
        if any(ls in norm for ls in legal_signals):
            legal_score += 0.40
        if "acidity regulator" in norm or "regulator" in norm or "preservative" in norm:
            legal_score = 0.0
        legal_score = min(1.0, max(0.0, legal_score))

        # -------------------------------------------------------------
        # 10. MARKETING SCORE
        # -------------------------------------------------------------
        marketing_score = 0.0
        marketing_signals = getattr(config, "MARKETING_SIGNALS", [
            "taste the goodness", "100% natural", "no artificial", "guaranteed quality", "premium quality",
            "provides strong protection", "protection from germs", "keeps your skin", "moisturizing", "with soothing",
            "clinically proven", "dermatologist recommended"
        ])
        matched_mkt, mkt_anchor_score = best_anchor_match(
            text, marketing_signals, threshold=config.STOP_WORD_CONTEXTUAL_SCORE
        )
        if matched_mkt:
            marketing_score += 0.80 * (mkt_anchor_score / 100.0)
        if any(ms in norm for ms in marketing_signals):
            marketing_score += 0.35
        marketing_score = min(1.0, max(0.0, marketing_score))

        # -------------------------------------------------------------
        # 11. OTHER SCORE
        # -------------------------------------------------------------
        other_base_score = 0.0
        if _URL_RE.search(text):
            other_base_score += 0.40
        if "barcode" in norm or "recycle" in norm:
            other_base_score += 0.40
        other_base_score = min(1.0, max(0.0, other_base_score))

        # Competing non-target score (for checking if any other section dominates)
        competing_other = max(
            allergen_score,
            instruction_score,
            manufacturer_score,
            contact_score,
            storage_score,
            mrp_score,
            legal_score,
            marketing_score,
            other_base_score
        )

        scores_dict = {
            "ingredients": round(float(ing_score), 3),
            "nutrition": round(float(nut_score), 3),
            "allergen": round(float(allergen_score), 3),
            "instructions": round(float(instruction_score), 3),
            "manufacturer": round(float(manufacturer_score), 3),
            "contact": round(float(contact_score), 3),
            "storage": round(float(storage_score), 3),
            "mrp": round(float(mrp_score), 3),
            "legal": round(float(legal_score), 3),
            "marketing": round(float(marketing_score), 3),
            "other": round(float(other_base_score), 3),
        }

        winning_class = max(scores_dict, key=scores_dict.get)
        if scores_dict[winning_class] == 0.0:
            winning_class = "other"

        ln["ingredient_score"] = scores_dict["ingredients"]
        ln["nutrition_score"] = scores_dict["nutrition"]
        ln["allergen_score"] = scores_dict["allergen"]
        ln["instruction_score"] = scores_dict["instructions"]
        ln["manufacturer_score"] = scores_dict["manufacturer"]
        ln["contact_score"] = scores_dict["contact"]
        ln["storage_score"] = scores_dict["storage"]
        ln["mrp_score"] = scores_dict["mrp"]
        ln["legal_score"] = scores_dict["legal"]
        ln["marketing_score"] = scores_dict["marketing"]
        ln["other_score"] = round(float(competing_other), 3)
        ln["scores"] = scores_dict
        ln["predicted_class"] = winning_class

        classified_lines.append(ln)

    return classified_lines

