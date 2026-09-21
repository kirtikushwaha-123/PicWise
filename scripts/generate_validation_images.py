"""
scripts/generate_validation_images.py

Generates the Phase 13 validation image dataset in tests/fixtures/validation_set/
covering 36 realistic Food and Personal Care product labels across 9 difficulty conditions:
A. Clean / easy
B. Angled (perspective distortion)
C. Dense text (long ingredient list, dense nutrition table)
D. Small text (small font size relative to image)
E. Low contrast (faint printing, gray on light background)
F. Mild blur (camera shake / focus blur)
G. Glare / reflections (specular packaging reflection)
H. Curved packaging (cylindrical container warp)
I. Complex layouts (multi-panel, claims, nutrition, ingredients)
"""

import json
import math
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = Path("tests/fixtures/validation_set")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
FONT_BOLD_PATH = r"C:\Windows\Fonts\arialbd.ttf"


def get_font(size: int, bold: bool = False):
    try:
        path = FONT_BOLD_PATH if bold and os.path.exists(FONT_BOLD_PATH) else FONT_PATH
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


# ----------------------------------------------------------------------
# Physical Distortion Transforms
# ----------------------------------------------------------------------

def apply_perspective_tilt(cv_img: np.ndarray, angle_deg: float = 15.0) -> np.ndarray:
    """Applies a realistic perspective tilt (e.g. looking at package from an angle)."""
    h, w = cv_img.shape[:2]
    rad = math.radians(angle_deg)
    dx = int(w * 0.12 * math.sin(rad))
    dy = int(h * 0.08 * math.sin(rad))

    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[dx, dy], [w - dx, 0], [dx // 2, h - dy], [w - dx // 2, h]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(cv_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(240, 240, 240))


def apply_cylindrical_warp(cv_img: np.ndarray, curvature: float = 0.35) -> np.ndarray:
    """Applies cylindrical packaging curvature (e.g. bottle or can)."""
    h, w = cv_img.shape[:2]
    cx = w / 2.0
    result = np.full_like(cv_img, (235, 235, 235))

    for y in range(h):
        for x in range(w):
            nx = (x - cx) / cx
            if abs(nx) < 1.0:
                theta = math.asin(nx * curvature)
                orig_x = int(cx + (theta / curvature) * cx * 0.95)
                if 0 <= orig_x < w:
                    result[y, x] = cv_img[y, orig_x]
    return result


def apply_mild_blur(cv_img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Applies realistic mild motion/lens blur."""
    return cv2.GaussianBlur(cv_img, (ksize, ksize), 1.5)


def apply_glare(cv_img: np.ndarray, center_ratio=(0.5, 0.4), intensity: float = 0.45) -> np.ndarray:
    """Applies realistic packaging reflection / specular glare."""
    h, w = cv_img.shape[:2]
    cx, cy = int(w * center_ratio[0]), int(h * center_ratio[1])
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + ((Y - cy) * 1.5) ** 2)
    radius = min(h, w) * 0.4
    glare_mask = np.clip(1.0 - (dist / radius), 0, 1) ** 2
    glare_mask = (glare_mask * intensity * 255).astype(np.uint8)

    glare_3ch = cv2.merge([glare_mask, glare_mask, glare_mask])
    return cv2.add(cv_img, glare_3ch)


def apply_low_contrast(cv_img: np.ndarray, contrast: float = 0.45, brightness: int = 70) -> np.ndarray:
    """Applies low-contrast faint printing effect."""
    return cv2.convertScaleAbs(cv_img, alpha=contrast, beta=brightness)


# ----------------------------------------------------------------------
# Label Rendering Functions
# ----------------------------------------------------------------------

def render_food_label(
    title: str,
    ingredients_text: str,
    nutrition_rows: list[tuple[str, str]] | None,
    serving_size: str | None = None,
    allergen_text: str | None = None,
    sub_claims: list[str] | None = None,
    small_font: bool = False,
    bg_color=(252, 252, 250),
    width=650,
    height=900,
) -> Image.Image:
    """Renders an authentic, realistic food packaging label."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    margin = 30
    curr_y = 35

    # Title
    font_title = get_font(22 if not small_font else 16, bold=True)
    draw.text((margin, curr_y), title.upper(), fill=(20, 20, 20), font=font_title)
    curr_y += (32 if not small_font else 24)

    # Sub claims
    if sub_claims:
        font_claim = get_font(13 if not small_font else 10, bold=False)
        for claim in sub_claims:
            draw.text((margin, curr_y), claim, fill=(80, 80, 80), font=font_claim)
            curr_y += 18
        curr_y += 8

    # Nutrition Facts Panel (if present)
    if nutrition_rows is not None:
        panel_w = width - (margin * 2)
        panel_top = curr_y

        font_nf_title = get_font(24 if not small_font else 18, bold=True)
        draw.text((margin + 10, curr_y + 8), "Nutrition Facts", fill=(0, 0, 0), font=font_nf_title)
        curr_y += (36 if not small_font else 28)

        if serving_size:
            font_serv = get_font(12 if not small_font else 9)
            draw.text((margin + 10, curr_y), f"Serving Size {serving_size}", fill=(0, 0, 0), font=font_serv)
            curr_y += 20

        # Thick line
        draw.line([(margin + 10, curr_y), (margin + panel_w - 10, curr_y)], fill=(0, 0, 0), width=5)
        curr_y += 10

        font_row = get_font(13 if not small_font else 10)
        font_row_bold = get_font(13 if not small_font else 10, bold=True)

        for name, val in nutrition_rows:
            is_bold = name in ["Calories", "Total Fat", "Total Carbohydrate", "Protein", "Sodium"]
            f = font_row_bold if is_bold else font_row
            draw.text((margin + 12, curr_y), name, fill=(0, 0, 0), font=f)
            bbox = draw.textbbox((0, 0), val, font=f)
            rw = bbox[2] - bbox[0]
            draw.text((margin + panel_w - 15 - rw, curr_y), val, fill=(0, 0, 0), font=f)
            curr_y += (22 if not small_font else 16)
            draw.line([(margin + 10, curr_y - 2), (margin + panel_w - 10, curr_y - 2)], fill=(180, 180, 180), width=1)

        panel_bottom = curr_y + 6
        # Draw outer box
        draw.rectangle([margin, panel_top, margin + panel_w, panel_bottom], outline=(0, 0, 0), width=2)
        curr_y = panel_bottom + 25

    # Ingredients Section
    font_ing_head = get_font(14 if not small_font else 11, bold=True)
    draw.text((margin, curr_y), "INGREDIENTS:", fill=(0, 0, 0), font=font_ing_head)
    curr_y += (22 if not small_font else 16)

    font_ing_body = get_font(12 if not small_font else 9)
    ing_lines = wrap_text(ingredients_text, font_ing_body, width - (margin * 2), draw)
    for line in ing_lines:
        draw.text((margin, curr_y), line, fill=(30, 30, 30), font=font_ing_body)
        curr_y += (18 if not small_font else 13)

    curr_y += 12

    # Allergen Statement
    if allergen_text:
        font_all_head = get_font(12 if not small_font else 9, bold=True)
        font_all_body = get_font(12 if not small_font else 9)
        draw.text((margin, curr_y), "CONTAINS: ", fill=(0, 0, 0), font=font_all_head)
        bbox = draw.textbbox((0, 0), "CONTAINS: ", font=font_all_head)
        offset_x = margin + (bbox[2] - bbox[0])
        draw.text((offset_x, curr_y), allergen_text.upper(), fill=(0, 0, 0), font=font_all_body)

    return img


def render_personal_care_label(
    brand: str,
    product_name: str,
    ingredients_text: str,
    claims: list[str] | None = None,
    active_ingredients: list[tuple[str, str]] | None = None,
    small_font: bool = False,
    bg_color=(250, 250, 252),
    width=650,
    height=850,
) -> Image.Image:
    """Renders an authentic, realistic personal care cosmetic / skincare packaging label."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    margin = 35
    curr_y = 40

    # Brand
    font_brand = get_font(16 if not small_font else 12, bold=True)
    draw.text((margin, curr_y), brand.upper(), fill=(70, 70, 70), font=font_brand)
    curr_y += (26 if not small_font else 20)

    # Product Name
    font_prod = get_font(22 if not small_font else 15, bold=True)
    draw.text((margin, curr_y), product_name, fill=(20, 20, 20), font=font_prod)
    curr_y += (34 if not small_font else 24)

    # Claims
    if claims:
        font_claim = get_font(12 if not small_font else 9)
        for claim in claims:
            draw.text((margin, curr_y), claim, fill=(90, 90, 90), font=font_claim)
            curr_y += 18
        curr_y += 12

    # Active Ingredients (if Drug Facts style e.g. Sunscreen)
    if active_ingredients:
        font_df = get_font(14 if not small_font else 10, bold=True)
        draw.text((margin, curr_y), "Drug Facts - Active Ingredients", fill=(0, 0, 0), font=font_df)
        curr_y += 22
        font_act = get_font(12 if not small_font else 9)
        for act_name, act_pct in active_ingredients:
            draw.text((margin + 10, curr_y), f"{act_name} ({act_pct})", fill=(30, 30, 30), font=font_act)
            curr_y += 18
        curr_y += 14

    # Ingredients Section
    font_ing_head = get_font(14 if not small_font else 10, bold=True)
    header_text = "INACTIVE INGREDIENTS:" if active_ingredients else "INGREDIENTS:"
    draw.text((margin, curr_y), header_text, fill=(0, 0, 0), font=font_ing_head)
    curr_y += (22 if not small_font else 16)

    font_ing_body = get_font(12 if not small_font else 9)
    lines = wrap_text(ingredients_text, font_ing_body, width - (margin * 2), draw)
    for line in lines:
        draw.text((margin, curr_y), line, fill=(40, 40, 40), font=font_ing_body)
        curr_y += (18 if not small_font else 13)

    return img


def render_blank_label(brand: str, title: str, volume: str, bg_color=(250, 250, 250), width=500, height=700) -> Image.Image:
    """Renders a front-of-package label with zero ingredients or nutrition."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    f_brand = get_font(24, bold=True)
    f_title = get_font(28, bold=True)
    f_vol = get_font(18)

    # Center text
    bbox_b = draw.textbbox((0, 0), brand, font=f_brand)
    draw.text(((width - (bbox_b[2] - bbox_b[0])) // 2, 220), brand, fill=(30, 30, 30), font=f_brand)

    bbox_t = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((width - (bbox_t[2] - bbox_t[0])) // 2, 270), title, fill=(10, 10, 10), font=f_title)

    bbox_v = draw.textbbox((0, 0), volume, font=f_vol)
    draw.text(((width - (bbox_v[2] - bbox_v[0])) // 2, 450), volume, fill=(100, 100, 100), font=f_vol)

    return img


# ----------------------------------------------------------------------
# Ground Truth Dataset Definition
# ----------------------------------------------------------------------

DATASET_SPECS = [
    # ==================================================================
    # FOOD DATASET (18 items)
    # ==================================================================
    {
        "id": "food_01_biscuit_oreo_clean",
        "category": "food",
        "product_type": "Biscuits / Cookies",
        "product_name": "Oreo Chocolate Sandwich Cookies",
        "condition": "Clean / Easy",
        "source": "Existing fixture tests/fixtures/product_food.jpeg",
        "use_existing_fixture": "tests/fixtures/product_food.jpeg",
        "visible_ingredients": [
            "Unbleached Enriched Flour", "Sugar", "Palm Oil", "Soybean Oil", "Cocoa",
            "High Fructose Corn Syrup", "Baking Soda", "Salt", "Soy Lecithin", "Chocolate", "Vanillin"
        ],
        "nutrition_values": {
            "Calories": "160 kcal", "Total Fat": "7g", "Saturated Fat": "2g",
            "Total Carbohydrate": "25g", "Sugars": "14g", "Protein": "1g", "Sodium": "135mg"
        },
        "allergens": ["Wheat", "Soy"],
    },
    {
        "id": "food_02_chips_lays_angled",
        "category": "food",
        "product_type": "Chips / Snacks",
        "product_name": "Lay's Classic Potato Chips",
        "condition": "Angled (perspective distortion)",
        "transform": "perspective_tilt",
        "visible_ingredients": ["Potatoes", "Vegetable Oil", "Canola Oil", "Corn Oil", "Soybean Oil", "Sunflower Oil", "Salt"],
        "nutrition_values": {
            "Calories": "160 kcal", "Total Fat": "10g", "Saturated Fat": "1.5g",
            "Total Carbohydrate": "15g", "Sugars": "1g", "Protein": "2g", "Sodium": "170mg"
        },
        "allergens": [],
        "render_spec": {
            "title": "Lay's Classic Potato Chips",
            "sub_claims": ["Guaranteed Fresh Until Printed Date", "No Artificial Flavors"],
            "ingredients": "Potatoes, Vegetable Oil (Canola, Corn, Soybean, and/or Sunflower Oil), and Salt.",
            "serving_size": "28g (About 15 chips)",
            "nutrition": [
                ("Calories", "160"), ("Total Fat", "10g"), ("Saturated Fat", "1.5g"),
                ("Trans Fat", "0g"), ("Cholesterol", "0mg"), ("Sodium", "170mg"),
                ("Total Carbohydrate", "15g"), ("Dietary Fiber", "1g"), ("Sugars", "1g"),
                ("Protein", "2g")
            ],
            "allergen": None,
        }
    },
    {
        "id": "food_03_noodles_maggi_dense",
        "category": "food",
        "product_type": "Instant Noodles",
        "product_name": "Maggi 2-Minute Masala Noodles",
        "condition": "Dense text",
        "transform": None,
        "visible_ingredients": [
            "Wheat Flour", "Palm Oil", "Salt", "Wheat Gluten", "Calcium Carbonate", "Guar Gum",
            "Hydrolyzed Peanut Protein", "Onion Powder", "Coriander", "Turmeric", "Cumin", "Aniseed",
            "Black Pepper", "Fenugreek", "Ginger", "Clove", "Nutmeg", "Cardamom", "Sugar", "Edible Starch"
        ],
        "nutrition_values": {
            "Calories": "310 kcal", "Total Fat": "11.5g", "Saturated Fat": "5.3g",
            "Total Carbohydrate": "44.5g", "Sugars": "1.8g", "Protein": "7.1g", "Sodium": "850mg"
        },
        "allergens": ["Wheat", "Peanut"],
        "render_spec": {
            "title": "Maggi 2-Minute Masala Noodles",
            "sub_claims": ["Good Food, Good Life", "With 10 Spices"],
            "ingredients": "Wheat Flour, Palm Oil, Salt, Wheat Gluten, Mineral (Calcium Carbonate), Thickeners (508, 412), Acidity Regulators (501(i), 500(i)), Humectant (451(i)), Hydrolyzed Peanut Protein, Mixed Spices (Onion Powder, Coriander, Turmeric, Cumin, Aniseed, Black Pepper, Fenugreek, Ginger, Clove, Nutmeg, Cardamom), Sugar, Edible Starch, Noodle Powder, Flavor Enhancer (635).",
            "serving_size": "70g (1 pack)",
            "nutrition": [
                ("Calories", "310 kcal"), ("Total Fat", "11.5g"), ("Saturated Fat", "5.3g"),
                ("Sodium", "850mg"), ("Total Carbohydrate", "44.5g"), ("Sugars", "1.8g"),
                ("Protein", "7.1g")
            ],
            "allergen": "Wheat, Peanut. May contain Soy, Milk and Mustard.",
        }
    },
    {
        "id": "food_04_cereal_kelloggs_small_text",
        "category": "food",
        "product_type": "Breakfast Cereal",
        "product_name": "Kellogg's Corn Flakes",
        "condition": "Small text",
        "transform": None,
        "small_font": True,
        "visible_ingredients": [
            "Milled Corn", "Sugar", "Malt Flavor", "Salt", "Iron", "Niacinamide",
            "Vitamin B6", "Vitamin B2", "Vitamin B1", "Folic Acid", "Vitamin D3", "Vitamin B12"
        ],
        "nutrition_values": {
            "Calories": "115 kcal", "Total Fat": "0.3g", "Saturated Fat": "0.1g",
            "Total Carbohydrate": "26g", "Sugars": "2.4g", "Protein": "2.1g", "Sodium": "210mg"
        },
        "allergens": ["Barley"],
        "render_spec": {
            "title": "Kellogg's The Original Corn Flakes",
            "sub_claims": ["Crispy golden flakes of corn", "Essential vitamins & iron"],
            "ingredients": "Milled Corn, Sugar, Malt Flavor, Salt, Iron, Niacinamide, Vitamin B6, Vitamin B2 (Riboflavin), Vitamin B1 (Thiamin Hydrochloride), Folic Acid, Vitamin D3, Vitamin B12.",
            "serving_size": "30g (1 cup)",
            "nutrition": [
                ("Calories", "115 kcal"), ("Total Fat", "0.3g"), ("Saturated Fat", "0.1g"),
                ("Sodium", "210mg"), ("Total Carbohydrate", "26g"), ("Sugars", "2.4g"),
                ("Dietary Fiber", "0.9g"), ("Protein", "2.1g")
            ],
            "allergen": "Contains Barley (Malt).",
        }
    },
    {
        "id": "food_05_beverage_cola_curved",
        "category": "food",
        "product_type": "Beverages",
        "product_name": "Classic Cola Can",
        "condition": "Curved packaging (cylindrical warp)",
        "transform": "cylindrical_warp",
        "visible_ingredients": ["Carbonated Water", "High Fructose Corn Syrup", "Caramel Color", "Phosphoric Acid", "Natural Flavors", "Caffeine"],
        "nutrition_values": {
            "Calories": "140 kcal", "Total Fat": "0g", "Saturated Fat": "0g",
            "Total Carbohydrate": "39g", "Sugars": "39g", "Protein": "0g", "Sodium": "45mg"
        },
        "allergens": [],
        "render_spec": {
            "title": "Classic Cola Sparkling Beverage",
            "sub_claims": ["100% Natural Flavors", "Refreshing Taste"],
            "ingredients": "Carbonated Water, High Fructose Corn Syrup, Caramel Color, Phosphoric Acid, Natural Flavors, Caffeine (34mg / 355mL).",
            "serving_size": "355 mL (1 Can)",
            "nutrition": [
                ("Calories", "140"), ("Total Fat", "0g"), ("Saturated Fat", "0g"),
                ("Sodium", "45mg"), ("Total Carbohydrate", "39g"), ("Sugars", "39g"),
                ("Protein", "0g")
            ],
            "allergen": None,
        }
    },
    {
        "id": "food_06_sauce_heinz_low_contrast",
        "category": "food",
        "product_type": "Sauces / Condiments",
        "product_name": "Heinz Tomato Ketchup",
        "condition": "Low contrast (faint print)",
        "transform": "low_contrast",
        "visible_ingredients": ["Tomato Concentrate", "Distilled Vinegar", "High Fructose Corn Syrup", "Corn Syrup", "Salt", "Spice", "Onion Powder", "Natural Flavoring"],
        "nutrition_values": {
            "Calories": "20 kcal", "Total Fat": "0g", "Saturated Fat": "0g",
            "Total Carbohydrate": "5g", "Sugars": "4g", "Protein": "0g", "Sodium": "160mg"
        },
        "allergens": [],
        "render_spec": {
            "title": "Heinz Tomato Ketchup 57 Varieties",
            "sub_claims": ["Thick & Rich", "Grown Not Made"],
            "ingredients": "Tomato Concentrate from Red Ripe Tomatoes, Distilled Vinegar, High Fructose Corn Syrup, Corn Syrup, Salt, Spice, Onion Powder, Natural Flavoring.",
            "serving_size": "17g (1 Tbsp)",
            "nutrition": [
                ("Calories", "20"), ("Total Fat", "0g"), ("Saturated Fat", "0g"),
                ("Sodium", "160mg"), ("Total Carbohydrate", "5g"), ("Sugars", "4g"),
                ("Protein", "0g")
            ],
            "allergen": None,
        }
    },
    {
        "id": "food_07_snack_pretzels_blur",
        "category": "food",
        "product_type": "Packaged Snacks",
        "product_name": "Cheddar Cheese Pretzel Pieces",
        "condition": "Mild blur",
        "transform": "mild_blur",
        "visible_ingredients": [
            "Enriched Flour", "Palm Oil", "Whey", "Cheddar Cheese", "Salt", "Maltodextrin",
            "Buttermilk Powder", "Yeast", "Onion Powder", "Disodium Phosphate", "Sodium Caseinate",
            "Tomato Powder", "Citric Acid", "Nonfat Dry Milk"
        ],
        "nutrition_values": {
            "Calories": "140 kcal", "Total Fat": "7g", "Saturated Fat": "3.5g",
            "Total Carbohydrate": "18g", "Sugars": "1g", "Protein": "2g", "Sodium": "260mg"
        },
        "allergens": ["Wheat", "Milk"],
        "render_spec": {
            "title": "Cheddar Cheese Pretzel Pieces",
            "sub_claims": ["Bursting with flavor", "Slow baked"],
            "ingredients": "Enriched Flour (Wheat Flour, Niacin, Reduced Iron, Thiamine Mononitrate, Riboflavin, Folic Acid), Palm Oil, Whey, Cheddar Cheese (Milk, Cultures, Salt, Enzymes), Salt, Maltodextrin, Buttermilk Powder, Yeast, Onion Powder, Disodium Phosphate, Sodium Caseinate, Tomato Powder, Citric Acid, Spice, Nonfat Dry Milk.",
            "serving_size": "28g (About 1/3 cup)",
            "nutrition": [
                ("Calories", "140"), ("Total Fat", "7g"), ("Saturated Fat", "3.5g"),
                ("Sodium", "260mg"), ("Total Carbohydrate", "18g"), ("Sugars", "1g"),
                ("Protein", "2g")
            ],
            "allergen": "Wheat, Milk.",
        }
    },
    {
        "id": "food_08_protein_bar_glare",
        "category": "food",
        "product_type": "Protein / Health Food",
        "product_name": "Quest Nutrition Chocolate Almond Crunch Protein Bar",
        "condition": "Glare / reflections",
        "transform": "glare",
        "visible_ingredients": [
            "Milk Protein Isolate", "Whey Protein Isolate", "Soluble Corn Fiber", "Almonds",
            "Water", "Cocoa Butter", "Erythritol", "Sea Salt", "Stevia", "Sucralose"
        ],
        "nutrition_values": {
            "Calories": "200 kcal", "Total Fat": "7g", "Saturated Fat": "2.5g",
            "Total Carbohydrate": "22g", "Sugars": "1g", "Protein": "21g", "Sodium": "220mg"
        },
        "allergens": ["Milk", "Almonds"],
        "render_spec": {
            "title": "Quest Protein Bar - Chocolate Almond Crunch",
            "sub_claims": ["21g Protein", "4g Net Carbs", "1g Sugar"],
            "ingredients": "Protein Blend (Milk Protein Isolate, Whey Protein Isolate), Soluble Corn Fiber, Almonds, Water, Cocoa Butter, Natural Flavors, Erythritol, Sea Salt, Stevia Sweetener, Sucralose.",
            "serving_size": "60g (1 Bar)",
            "nutrition": [
                ("Calories", "200"), ("Total Fat", "7g"), ("Saturated Fat", "2.5g"),
                ("Sodium", "220mg"), ("Total Carbohydrate", "22g"), ("Dietary Fiber", "14g"),
                ("Sugars", "1g"), ("Protein", "21g")
            ],
            "allergen": "Milk, Almonds. Processed in a facility that also processes eggs, peanuts, soy, wheat.",
        }
    },
    {
        "id": "food_09_staple_pasta_complex",
        "category": "food",
        "product_type": "Packaged Staples",
        "product_name": "Barilla Penne Rigate Pasta",
        "condition": "Complex layout",
        "transform": None,
        "visible_ingredients": ["Semolina", "Durum Wheat Flour", "Niacin", "Iron", "Thiamine Mononitrate", "Riboflavin", "Folic Acid"],
        "nutrition_values": {
            "Calories": "200 kcal", "Total Fat": "1g", "Saturated Fat": "0g",
            "Total Carbohydrate": "42g", "Sugars": "2g", "Protein": "7g", "Sodium": "0mg"
        },
        "allergens": ["Wheat"],
        "render_spec": {
            "title": "Barilla Penne Rigate - Non GMO Project Verified",
            "sub_claims": ["Italy's #1 Brand of Pasta", "Cooks in 11 Minutes", "Al Dente Perfection"],
            "ingredients": "Semolina (Wheat), Durum Wheat Flour. Vitamins/Minerals: Niacin, Iron (Ferrous Sulfate), Thiamine Mononitrate, Riboflavin, Folic Acid.",
            "serving_size": "56g (2 oz / about 3/4 cup)",
            "nutrition": [
                ("Calories", "200"), ("Total Fat", "1g"), ("Saturated Fat", "0g"),
                ("Sodium", "0mg"), ("Total Carbohydrate", "42g"), ("Dietary Fiber", "3g"),
                ("Sugars", "2g"), ("Protein", "7g")
            ],
            "allergen": "Contains Wheat. May contain traces of Egg.",
        }
    },
    {
        "id": "food_10_ready_to_eat_soup_curved",
        "category": "food",
        "product_type": "Ready-to-eat Foods",
        "product_name": "Chunky Chicken Noodle Soup",
        "condition": "Curved packaging (cylindrical warp)",
        "transform": "cylindrical_warp",
        "visible_ingredients": [
            "Chicken Stock", "Enriched Egg Noodles", "Chicken Meat", "Carrots", "Celery",
            "Salt", "Chicken Fat", "Monosodium Glutamate", "Modified Food Starch", "Dehydrated Onions", "Yeast Extract"
        ],
        "nutrition_values": {
            "Calories": "130 kcal", "Total Fat": "3g", "Saturated Fat": "1g",
            "Total Carbohydrate": "15g", "Sugars": "1g", "Protein": "10g", "Sodium": "790mg"
        },
        "allergens": ["Wheat", "Egg"],
        "render_spec": {
            "title": "Chunky Classic Chicken Noodle Soup Can",
            "sub_claims": ["Soup That Eats Like A Meal", "Big Chunks of Real Chicken"],
            "ingredients": "Chicken Stock, Enriched Egg Noodles (Wheat Flour, Eggs, Niacin, Ferrous Sulfate), Chicken Meat, Carrots, Celery, Salt, Chicken Fat, Monosodium Glutamate, Modified Food Starch, Dehydrated Onions, Yeast Extract, Spice Extract.",
            "serving_size": "240 mL (1 cup)",
            "nutrition": [
                ("Calories", "130"), ("Total Fat", "3g"), ("Saturated Fat", "1g"),
                ("Sodium", "790mg"), ("Total Carbohydrate", "15g"), ("Sugars", "1g"),
                ("Protein", "10g")
            ],
            "allergen": "Contains Wheat, Egg.",
        }
    },
    {
        "id": "food_11_namkeen_haldiram_dense",
        "category": "food",
        "product_type": "Chips / Namkeen",
        "product_name": "Haldiram's Bhujia Sev",
        "condition": "Dense text",
        "transform": None,
        "visible_ingredients": [
            "Tepary Bean Flour", "Bengal Gram Flour", "Edible Vegetable Oil", "Cotton Seed Oil",
            "Corn Oil", "Palmolein Oil", "Iodized Salt", "Red Chilli Powder", "Black Pepper",
            "Ginger Powder", "Clove", "Cardamom", "Nutmeg", "Mace", "Citric Acid"
        ],
        "nutrition_values": {
            "Calories": "580 kcal", "Total Fat": "42g", "Saturated Fat": "16g",
            "Total Carbohydrate": "40g", "Sugars": "1.5g", "Protein": "11g", "Sodium": "720mg"
        },
        "allergens": [],
        "render_spec": {
            "title": "Haldiram's Nagpur Bhujia Sev",
            "sub_claims": ["Crispy spicy gram & tepary bean snack", "Traditional Indian Namkeen"],
            "ingredients": "Tepary Bean Flour (Moth Dal), Bengal Gram Flour (Besan), Edible Vegetable Oil (Cotton Seed, Corn and Palmolein Oil), Iodized Salt, Red Chilli Powder, Black Pepper, Ginger Powder, Clove, Cardamom, Nutmeg, Mace, Citric Acid (E330).",
            "serving_size": "100g",
            "nutrition": [
                ("Calories", "580 kcal"), ("Total Fat", "42g"), ("Saturated Fat", "16g"),
                ("Sodium", "720mg"), ("Total Carbohydrate", "40g"), ("Sugars", "1.5g"),
                ("Protein", "11g")
            ],
            "allergen": "May contain Peanuts, Tree Nuts, Gluten, Soy and Sesame.",
        }
    },
    {
        "id": "food_12_energy_drink_redbull_units",
        "category": "food",
        "product_type": "Beverages",
        "product_name": "Red Bull Energy Drink",
        "condition": "Complex layout (units: mg, kJ, kcal)",
        "transform": None,
        "visible_ingredients": [
            "Carbonated Water", "Sucrose", "Glucose", "Citric Acid", "Taurine",
            "Sodium Bicarbonate", "Magnesium Carbonate", "Caffeine", "Niacinamide",
            "Calcium Pantothenate", "Pyridoxine HCl", "Vitamin B12"
        ],
        "nutrition_values": {
            "Calories": "110 kcal", "Total Fat": "0g", "Saturated Fat": "0g",
            "Total Carbohydrate": "27g", "Sugars": "27g", "Protein": "1g", "Sodium": "105mg"
        },
        "allergens": [],
        "render_spec": {
            "title": "Red Bull Energy Drink - Vitalizes Body and Mind",
            "sub_claims": ["With Taurine 1000mg", "Caffeine 80mg / 250mL"],
            "ingredients": "Carbonated Water, Sucrose, Glucose, Citric Acid, Taurine (0.4%), Sodium Bicarbonate, Magnesium Carbonate, Caffeine (0.03%), Niacinamide, Calcium Pantothenate, Pyridoxine HCl, Vitamin B12, Natural and Artificial Flavors, Colors.",
            "serving_size": "250 mL (1 Can)",
            "nutrition": [
                ("Calories", "110 kcal / 460 kJ"), ("Total Fat", "0g"), ("Saturated Fat", "0g"),
                ("Sodium", "105mg"), ("Total Carbohydrate", "27g"), ("Sugars", "27g"),
                ("Protein", "1g")
            ],
            "allergen": None,
        }
    },
    {
        "id": "food_13_snack_chikki_no_nutrition",
        "category": "food",
        "product_type": "Packaged Snacks",
        "product_name": "Traditional Peanut Chikki Jaggery Bar",
        "condition": "Missing nutrition panel (Failure Semantics: Nutrition Unavailable != Safe)",
        "transform": None,
        "visible_ingredients": ["Roasted Peanuts", "Jaggery", "Sugar", "Liquid Glucose"],
        "nutrition_values": None,
        "allergens": ["Peanuts"],
        "render_spec": {
            "title": "Traditional Peanut Chikki Jaggery Bar",
            "sub_claims": ["Crunchy Peanut Brittle", "No Added Artificial Preservatives"],
            "ingredients": "Roasted Peanuts (60%), Jaggery (Gur 25%), Sugar, Liquid Glucose.",
            "serving_size": None,
            "nutrition": None,  # No nutrition panel present on label!
            "allergen": "Contains Peanuts. Manufactured in a facility that handles tree nuts and sesame.",
        }
    },
    {
        "id": "food_14_exotic_botanical_food",
        "category": "food",
        "product_type": "Protein / Health Food",
        "product_name": "Ayurvedic Ashwagandha Herbal Elixir",
        "condition": "Unknown ingredients (Semantics: Unknown != Safe)",
        "transform": None,
        "visible_ingredients": [
            "Withania Somnifera", "Convolvulus Pluricaulis", "Bacopa Monnieri",
            "Asparagus Racemosus", "Purified Water", "Sodium Benzoate", "Potassium Sorbate"
        ],
        "nutrition_values": None,
        "allergens": [],
        "render_spec": {
            "title": "Ayurvedic Ashwagandha Herbal Elixir",
            "sub_claims": ["Classical Ayurvedic Formulation", "Natural Adaptogen Support"],
            "ingredients": "Each 100mL contains: Withania Somnifera (Ashwagandha Root Extract 500mg), Convolvulus Pluricaulis (Shankhpushpi 300mg), Bacopa Monnieri (Brahmi 250mg), Asparagus Racemosus (Shatavari 200mg), Purified Water q.s., Preservatives (Sodium Benzoate, Potassium Sorbate).",
            "serving_size": None,
            "nutrition": None,
            "allergen": None,
        }
    },
    {
        "id": "food_15_food_front_blank",
        "category": "food",
        "product_type": "Biscuits / Cookies",
        "product_name": "Royal Butter Cookies Gift Tin (Front Panel)",
        "condition": "Front branding only (Semantics: 0 text / OCR failure != Safe)",
        "transform": None,
        "visible_ingredients": [],
        "nutrition_values": None,
        "allergens": [],
        "is_blank": True,
        "blank_spec": {
            "brand": "ROYAL DANISH",
            "title": "Butter Cookies Tin",
            "volume": "Net Wt 454g (16 oz)",
        }
    },
    {
        "id": "food_16_chocolate_dark_low_contrast",
        "category": "food",
        "product_type": "Packaged Snacks",
        "product_name": "Lindt Excellence 85% Cocoa Dark Chocolate",
        "condition": "Low contrast (dark packaging)",
        "transform": "low_contrast",
        "visible_ingredients": ["Chocolate", "Cocoa Powder", "Cocoa Butter", "Demerara Sugar", "Bourbon Vanilla Beans"],
        "nutrition_values": {
            "Calories": "230 kcal", "Total Fat": "20g", "Saturated Fat": "12g",
            "Total Carbohydrate": "15g", "Sugars": "5g", "Protein": "4g", "Sodium": "10mg"
        },
        "allergens": [],
        "render_spec": {
            "title": "Lindt Excellence 85% Cocoa Dark Chocolate",
            "sub_claims": ["Smooth and intensely rich", "Master Chocolatier since 1845"],
            "ingredients": "Chocolate, Cocoa Powder processed with alkali, Cocoa Butter, Demerara Sugar, Bourbon Vanilla Beans.",
            "serving_size": "40g (4 squares)",
            "nutrition": [
                ("Calories", "230"), ("Total Fat", "20g"), ("Saturated Fat", "12g"),
                ("Sodium", "10mg"), ("Total Carbohydrate", "15g"), ("Dietary Fiber", "6g"),
                ("Sugars", "5g"), ("Protein", "4g")
            ],
            "allergen": "May contain Milk, Soy, Tree Nuts and Sesame.",
        }
    },
    {
        "id": "food_17_peanut_butter_angled",
        "category": "food",
        "product_type": "Packaged Staples",
        "product_name": "Skippy Creamy Peanut Butter",
        "condition": "Angled (perspective distortion)",
        "transform": "perspective_tilt",
        "visible_ingredients": ["Roasted Peanuts", "Sugar", "Hydrogenated Vegetable Oil", "Cottonseed Oil", "Soybean Oil", "Rapeseed Oil", "Salt"],
        "nutrition_values": {
            "Calories": "190 kcal", "Total Fat": "16g", "Saturated Fat": "3g",
            "Total Carbohydrate": "7g", "Sugars": "3g", "Protein": "7g", "Sodium": "150mg"
        },
        "allergens": ["Peanuts"],
        "render_spec": {
            "title": "Skippy Creamy Peanut Butter Spread",
            "sub_claims": ["7g Protein per serving", "No Need to Stir"],
            "ingredients": "Roasted Peanuts, Sugar, Hydrogenated Vegetable Oil (Cottonseed, Soybean and Rapeseed Oil) to Prevent Separation, Salt.",
            "serving_size": "32g (2 Tbsp)",
            "nutrition": [
                ("Calories", "190"), ("Total Fat", "16g"), ("Saturated Fat", "3g"),
                ("Sodium", "150mg"), ("Total Carbohydrate", "7g"), ("Dietary Fiber", "2g"),
                ("Sugars", "3g"), ("Protein", "7g")
            ],
            "allergen": "Contains Peanuts.",
        }
    },
    {
        "id": "food_18_instant_oats_clean",
        "category": "food",
        "product_type": "Breakfast Cereal",
        "product_name": "Quaker Instant Oatmeal Original",
        "condition": "Clean / Easy",
        "transform": None,
        "visible_ingredients": ["Whole Grain Rolled Oats", "Calcium Carbonate", "Salt", "Guar Gum", "Caramel Color", "Reduced Iron", "Vitamin A Palmitate"],
        "nutrition_values": {
            "Calories": "150 kcal", "Total Fat": "3g", "Saturated Fat": "0.5g",
            "Total Carbohydrate": "27g", "Sugars": "1g", "Protein": "5g", "Sodium": "75mg"
        },
        "allergens": ["Oats"],
        "render_spec": {
            "title": "Quaker Instant Oatmeal Original",
            "sub_claims": ["100% Whole Grains", "Heart Healthy", "Good Source of Fiber"],
            "ingredients": "Whole Grain Rolled Oats, Calcium Carbonate, Salt, Guar Gum, Caramel Color, Reduced Iron, Vitamin A Palmitate.",
            "serving_size": "40g (1 packet)",
            "nutrition": [
                ("Calories", "150"), ("Total Fat", "3g"), ("Saturated Fat", "0.5g"),
                ("Sodium", "75mg"), ("Total Carbohydrate", "27g"), ("Dietary Fiber", "4g"),
                ("Sugars", "1g"), ("Protein", "5g")
            ],
            "allergen": "Contains Oats (Gluten).",
        }
    },

    # ==================================================================
    # PERSONAL CARE DATASET (18 items)
    # ==================================================================
    {
        "id": "pc_01_shampoo_head_shoulders_clean",
        "category": "personal_care",
        "product_type": "Shampoo / Hair Care",
        "product_name": "Head & Shoulders Classic Clean",
        "condition": "Clean / Easy",
        "source": "Existing fixture tests/fixtures/product_personal_care.png",
        "use_existing_fixture": "tests/fixtures/product_personal_care.png",
        "visible_ingredients": [
            "Water", "Sodium Laureth Sulfate", "Sodium Xylenesulfonate", "Zinc Carbonate",
            "Glycol Distearate", "Cocamide MEA", "Cocamidopropyl Betaine", "Dimethicone",
            "Fragrance", "Sodium Benzoate", "Polyquaternium-10", "Stearyl Alcohol",
            "Magnesium Carbonate Hydroxide", "Cetyl Alcohol", "Methylchloroisothiazolinone", "Methylisothiazolinone"
        ],
    },
    {
        "id": "pc_02_moisturizer_cerave_dense",
        "category": "personal_care",
        "product_type": "Moisturizer / Skin Care",
        "product_name": "CeraVe Moisturizing Cream",
        "condition": "Dense text",
        "source": "Existing fixture tests/fixtures/product_pc_dense.png",
        "use_existing_fixture": "tests/fixtures/product_pc_dense.png",
        "visible_ingredients": [
            "Aqua (Water)", "Glycerin", "Cetearyl Alcohol", "Caprylic/Capric Triglyceride", "Cetyl Alcohol",
            "Ceteareth-20", "Petrolatum", "Potassium Phosphate", "Ceramide NP", "Ceramide AP", "Ceramide EOP",
            "Carbomer", "Dimethicone", "Sodium Lauroyl Lactylate", "Sodium Hyaluronate", "Cholesterol",
            "Phenoxyethanol", "Disodium EDTA", "Dipotassium Phosphate", "Tocopherol", "Ethylhexylglycerin"
        ],
    },
    {
        "id": "pc_03_cleanser_cetaphil_blank",
        "category": "personal_care",
        "product_type": "Cleanser / Skin Care",
        "product_name": "Cetaphil Gentle Skin Cleanser (Front Panel)",
        "condition": "Brand/volume only (Semantics: 0 ingredients -> Unavailable != Safe)",
        "source": "Existing fixture tests/fixtures/product_pc_blank.png",
        "use_existing_fixture": "tests/fixtures/product_pc_blank.png",
        "visible_ingredients": [],
    },
    {
        "id": "pc_04_sunscreen_neutrogena_angled",
        "category": "personal_care",
        "product_type": "Sunscreen",
        "product_name": "Neutrogena Ultra Sheer Sunscreen",
        "condition": "Angled (perspective distortion)",
        "source": "Existing fixture tests/fixtures/product_pc_angled.png",
        "use_existing_fixture": "tests/fixtures/product_pc_angled.png",
        "visible_ingredients": [
            "Avobenzone", "Homosalate", "Octisalate", "Octocrylene", "Water", "Silica",
            "Styrene/Acrylates Copolymer", "Butyloctyl Salicylate", "Glycerin", "Glyceryl Stearate",
            "PEG-100 Stearate", "Dimethicone", "Caprylyl Glycol", "Ethylhexylglycerin", "Xanthan Gum", "Fragrance"
        ],
    },
    {
        "id": "pc_05_hair_oil_blurred",
        "category": "personal_care",
        "product_type": "Hair Care / Oil",
        "product_name": "Moroccanoil Treatment Original",
        "condition": "Mild blur",
        "source": "Existing fixture tests/fixtures/product_pc_blurred.png",
        "use_existing_fixture": "tests/fixtures/product_pc_blurred.png",
        "visible_ingredients": [
            "Cyclomethicone", "Dimethicone", "Argania Spinosa Kernel Oil", "Fragrance", "Linum Usitatissimum Seed Extract"
        ],
    },
    {
        "id": "pc_06_lotion_dove_lighting",
        "category": "personal_care",
        "product_type": "Body Lotion",
        "product_name": "Dove Nourishing Body Lotion",
        "condition": "Glare / reflections / uneven lighting",
        "source": "Existing fixture tests/fixtures/product_pc_lighting.png",
        "use_existing_fixture": "tests/fixtures/product_pc_lighting.png",
        "visible_ingredients": [
            "Aqua", "Glycerin", "Stearic Acid", "Caprylic/Capric Triglyceride", "Dimethicone",
            "Glycol Stearate", "PEG-100 Stearate", "Petrolatum", "Cyclopentasiloxane", "Glyceryl Stearate",
            "Phenoxyethanol", "Cetyl Alcohol", "Triethanolamine", "Methylparaben", "Propylparaben", "Disodium EDTA", "Parfum"
        ],
    },
    {
        "id": "pc_07_face_wash_simple_lowres",
        "category": "personal_care",
        "product_type": "Face Wash",
        "product_name": "Simple Kind to Skin Refreshing Facial Wash",
        "condition": "Small text / low resolution",
        "source": "Existing fixture tests/fixtures/product_pc_lowres.png",
        "use_existing_fixture": "tests/fixtures/product_pc_lowres.png",
        "visible_ingredients": [
            "Aqua", "Cocamidopropyl Betaine", "Propylene Glycol", "Hydroxypropyl Methylcellulose",
            "Panthenol", "Tocopheryl Acetate", "Pantolactone", "Sodium Hydroxide", "Disodium EDTA", "Sodium Hydroxymethylglycinate"
        ],
    },
    {
        "id": "pc_08_conditioner_tresemme_curved",
        "category": "personal_care",
        "product_type": "Conditioner / Hair Care",
        "product_name": "Tresemme Keratin Smooth Conditioner",
        "condition": "Curved packaging (cylindrical warp)",
        "transform": "cylindrical_warp",
        "visible_ingredients": [
            "Aqua", "Cetearyl Alcohol", "Dimethicone", "Stearamidopropyl Dimethylamine", "Behentrimonium Chloride",
            "Parfum", "Dipropylene Glycol", "Lactic Acid", "Amodimethicone", "Sodium Chloride", "Disodium EDTA",
            "PEG-7 Propylheptyl Ether", "Cetrimonium Chloride", "Phenoxyethanol", "Keratin", "Argania Spinosa Kernel Oil",
            "Methylchloroisothiazolinone", "Methylisothiazolinone"
        ],
        "render_spec": {
            "brand": "Tresemme Professional",
            "product_name": "Keratin Smooth Weightless Conditioner",
            "claims": ["With Marula Oil", "Up to 72 Hours Frizz Control", "System Pro Collection"],
            "ingredients": "Aqua, Cetearyl Alcohol, Dimethicone, Stearamidopropyl Dimethylamine, Behentrimonium Chloride, Parfum, Dipropylene Glycol, Lactic Acid, Amodimethicone, Sodium Chloride, Disodium EDTA, PEG-7 Propylheptyl Ether, Cetrimonium Chloride, Phenoxyethanol, Hydrolyzed Keratin, Argania Spinosa Kernel Oil, Methylchloroisothiazolinone, Methylisothiazolinone.",
        }
    },
    {
        "id": "pc_09_serum_ordinary_small_text",
        "category": "personal_care",
        "product_type": "Skin Care / Serum",
        "product_name": "The Ordinary Niacinamide 10% + Zinc 1%",
        "condition": "Small text",
        "transform": None,
        "small_font": True,
        "visible_ingredients": [
            "Aqua (Water)", "Niacinamide", "Pentylene Glycol", "Zinc PCA", "Dimethyl Isosorbide",
            "Tamarindus Indica Seed Gum", "Xanthan Gum", "Isoceteth-20", "Ethoxydiglycol", "Phenoxyethanol", "Chlorphenesin"
        ],
        "render_spec": {
            "brand": "The Ordinary.",
            "product_name": "Niacinamide 10% + Zinc 1%",
            "claims": ["High-Strength Vitamin and Mineral Blemish Formula", "Clinical Formulations with Integrity"],
            "ingredients": "Aqua (Water), Niacinamide, Pentylene Glycol, Zinc PCA, Dimethyl Isosorbide, Tamarindus Indica Seed Gum, Xanthan Gum, Isoceteth-20, Ethoxydiglycol, Phenoxyethanol, Chlorphenesin.",
        }
    },
    {
        "id": "pc_10_lip_balm_burts_low_contrast",
        "category": "personal_care",
        "product_type": "Cosmetic Products / Lip Care",
        "product_name": "Burt's Bees Beeswax Lip Balm",
        "condition": "Low contrast (faint print on pale tube)",
        "transform": "low_contrast",
        "visible_ingredients": [
            "Cera Alba (Beeswax)", "Cocos Nucifera (Coconut) Oil", "Helianthus Annuus (Sunflower) Seed Oil",
            "Mentha Piperita (Peppermint) Oil", "Lanolin", "Tocopherol", "Rosmarinus Officinalis (Rosemary) Leaf Extract",
            "Glycine Soja (Soybean) Oil", "Canola Oil", "Limonene", "Linalool", "Eugenol"
        ],
        "render_spec": {
            "brand": "Burt's Bees",
            "product_name": "Beeswax Lip Balm with Vitamin E & Peppermint",
            "claims": ["100% Natural Origin", "Responsible Sourcing", "Cruelty Free"],
            "ingredients": "Cera Alba (Beeswax), Cocos Nucifera (Coconut) Oil, Helianthus Annuus (Sunflower) Seed Oil, Mentha Piperita (Peppermint) Oil, Lanolin, Tocopherol (Vitamin E), Rosmarinus Officinalis (Rosemary) Leaf Extract, Glycine Soja (Soybean) Oil, Canola Oil, Limonene, Linalool, Eugenol.",
        }
    },
    {
        "id": "pc_11_foundation_maybelline_complex",
        "category": "personal_care",
        "product_type": "Cosmetic Products",
        "product_name": "Maybelline Fit Me Matte + Poreless Foundation",
        "condition": "Complex layout",
        "transform": None,
        "visible_ingredients": [
            "Aqua", "Cyclohexasiloxane", "Nylon-12", "Isododecane", "Alcohol Denat.", "Cyclopentasiloxane",
            "PEG-10 Dimethicone", "Cetyl PEG/PPG-10/1 Dimethicone", "PEG-20", "Polyglyceryl-4 Isostearate",
            "Disteardimonium Hectorite", "Phenoxyethanol", "Magnesium Sulfate", "Disodium Stearoyl Glutamate",
            "Titanium Dioxide", "Methylparaben", "Acrylates Copolymer", "Tocopherol", "Butylparaben",
            "Aluminum Hydroxide", "Alumina", "Silica", "Glycerin"
        ],
        "render_spec": {
            "brand": "Maybelline New York",
            "product_name": "Fit Me Matte + Poreless Liquid Foundation",
            "claims": ["Normal to Oily Skin", "Matches Natural Tone", "Blurs Pores", "Oil-Free"],
            "ingredients": "Aqua / Water / Eau, Cyclohexasiloxane, Nylon-12, Isododecane, Alcohol Denat., Cyclopentasiloxane, PEG-10 Dimethicone, Cetyl PEG/PPG-10/1 Dimethicone, PEG-20, Polyglyceryl-4 Isostearate, Disteardimonium Hectorite, Phenoxyethanol, Magnesium Sulfate, Disodium Stearoyl Glutamate, HDI/Trimethylol Hexyllactone Crosspolymer, Titanium Dioxide, Methylparaben, Acrylates Copolymer, Tocopherol, Butylparaben, Aluminum Hydroxide, Alumina, Silica, Glycerin.",
        }
    },
    {
        "id": "pc_12_sunscreen_laroche_active_inactive",
        "category": "personal_care",
        "product_type": "Sunscreen",
        "product_name": "La Roche-Posay Anthelios 50 Mineral Sunscreen",
        "condition": "Complex layout (Drug facts format)",
        "transform": None,
        "visible_ingredients": [
            "Titanium Dioxide", "Zinc Oxide", "Water", "Isododecane", "C12-15 Alkyl Benzoate", "Dimethicone",
            "Undecane", "Triethylhexanoin", "Isohexadecane", "Nylon-12", "Caprylyl Methicone", "Butyloctyl Salicylate",
            "Phenoxyethanol", "Caprylyl Glycol", "Polyhydroxystearic Acid", "Silica", "Tocopherol"
        ],
        "render_spec": {
            "brand": "La Roche-Posay Laboratoire Dermatologique",
            "product_name": "Anthelios 50 Mineral Ultra Light Sunscreen Fluid",
            "claims": ["Broad Spectrum SPF 50", "100% Mineral UV Filters", "Cellox-B3 Shield"],
            "active_ingredients": [
                ("Titanium Dioxide", "11%"),
                ("Zinc Oxide", "5%"),
            ],
            "ingredients": "Water, Isododecane, C12-15 Alkyl Benzoate, Dimethicone, Undecane, Triethylhexanoin, Isohexadecane, Nylon-12, Caprylyl Methicone, Butyloctyl Salicylate, Phenoxyethanol, Caprylyl Glycol, Polyhydroxystearic Acid, Silica, Tocopherol (Vitamin E).",
        }
    },
    {
        "id": "pc_13_botanical_body_wash_aliases",
        "category": "personal_care",
        "product_type": "Body Lotion / Wash",
        "product_name": "Aveeno Daily Moisturizing Body Wash",
        "condition": "Dense text / INCI packaging aliases",
        "transform": None,
        "visible_ingredients": [
            "Water", "Glycerin", "Cocamidopropyl Betaine", "Sodium Laureth Sulfate",
            "Avena Sativa (Oat) Kernel Flour", "Avena Sativa (Oat) Kernel Extract", "Avena Sativa (Oat) Kernel Oil",
            "Glycol Distearate", "Sodium Hydroxide", "Citric Acid", "Mineral Oil",
            "Glycine Soja (Soybean) Oil", "Potassium Sorbate", "Sodium Benzoate"
        ],
        "render_spec": {
            "brand": "Aveeno Active Naturals",
            "product_name": "Daily Moisturizing Body Wash with Soothing Oat",
            "claims": ["Dermatologist Recommended for over 65 years", "Soap-Free", "Dye-Free"],
            "ingredients": "Water (Aqua), Glycerin, Cocamidopropyl Betaine, Sodium Laureth Sulfate, Avena Sativa (Oat) Kernel Flour, Avena Sativa (Oat) Kernel Extract, Avena Sativa (Oat) Kernel Oil, Glycol Distearate, Acrylates/C10-30 Alkyl Acrylate Crosspolymer, Sodium Hydroxide, Citric Acid, Paraffinum Liquidum (Mineral Oil), Glycine Soja (Soybean) Oil, Potassium Sorbate, Sodium Benzoate.",
        }
    },
    {
        "id": "pc_14_anti_aging_cream_high_risk",
        "category": "personal_care",
        "product_type": "Skin Care",
        "product_name": "Revitalizing Intensive Retinol Night Cream",
        "condition": "High risk detection (sensitizers & allergens)",
        "transform": None,
        "visible_ingredients": [
            "Aqua", "Caprylic/Capric Triglyceride", "Glycerin", "Cetearyl Alcohol", "Dimethicone",
            "Polysorbate 60", "Retinol", "BHT", "Disodium EDTA", "Triethanolamine",
            "Methylisothiazolinone", "Fragrance", "Benzyl Salicylate", "Hexyl Cinnamal", "Linalool", "Hydroxycitronellal"
        ],
        "render_spec": {
            "brand": "Dermacare Advanced",
            "product_name": "Intensive Retinol Night Renewal Cream",
            "claims": ["Pure Active Retinol", "Visible Wrinkle Reduction", "Night Treatment"],
            "ingredients": "Aqua, Caprylic/Capric Triglyceride, Glycerin, Cetearyl Alcohol, Dimethicone, Polysorbate 60, Retinol, BHT, Disodium EDTA, Triethanolamine, Methylisothiazolinone, Fragrance (Parfum), Benzyl Salicylate, Hexyl Cinnamal, Linalool, Hydroxycitronellal.",
        }
    },
    {
        "id": "pc_15_perfume_box_blank",
        "category": "personal_care",
        "product_type": "Cosmetic Products",
        "product_name": "Eau de Parfum Luxury Fragrance (Front Panel)",
        "condition": "Brand name only (Semantics: 0 ingredients -> Unavailable != Safe)",
        "transform": None,
        "visible_ingredients": [],
        "is_blank": True,
        "blank_spec": {
            "brand": "MAISON DE PARFUM",
            "title": "Rose & Oud Eau de Parfum",
            "volume": "100 mL - 3.4 FL. OZ.",
        }
    },
    {
        "id": "pc_16_toner_witch_hazel_parentheticals",
        "category": "personal_care",
        "product_type": "Skin Care / Toner",
        "product_name": "Thayers Witch Hazel Facial Toner",
        "condition": "Dense text / parentheticals",
        "transform": None,
        "visible_ingredients": [
            "Purified Water", "Glycerin", "Hamamelis Virginiana Extract", "Aloe Barbadensis Leaf Juice",
            "Phenoxyethanol", "Rosa Centifolia Flower Water", "Fragrance", "Citric Acid", "Citrus Grandis Seed Extract"
        ],
        "render_spec": {
            "brand": "Thayers Since 1847",
            "product_name": "Witch Hazel Facial Toner - Rose Petal",
            "claims": ["Alcohol-Free", "Gentle Hydration", "Dermatologist Tested"],
            "ingredients": "Purified Water, Glycerin, Certified Organic Witch Hazel Extract Blend (Hamamelis Virginiana Extract (Witch Hazel), Aloe Barbadensis Leaf Juice (Aloe Vera)), Phenoxyethanol, Rosa Centifolia (Rose) Flower Water, Fragrance (Natural Rose), Citric Acid, Citrus Grandis (Grapefruit) Seed Extract.",
        }
    },
    {
        "id": "pc_17_micellar_water_bioderma_clean",
        "category": "personal_care",
        "product_type": "Cleanser / Skin Care",
        "product_name": "Bioderma Sensibio H2O Micellar Water",
        "condition": "Clean / Easy",
        "transform": None,
        "visible_ingredients": [
            "Aqua", "PEG-6 Caprylic/Capric Glycerides", "Fructooligosaccharides", "Mannitol",
            "Xylitol", "Rhamnose", "Cucumis Sativus (Cucumber) Fruit Extract", "Propylene Glycol",
            "Cetrimonium Bromide", "Disodium EDTA"
        ],
        "render_spec": {
            "brand": "Bioderma Laboratoire Dermatologique",
            "product_name": "Sensibio H2O Micelle Solution",
            "claims": ["Cleanses and Removes Makeup", "Sensitive Skin", "Non-Rinse"],
            "ingredients": "Aqua/Water/Eau, PEG-6 Caprylic/Capric Glycerides, Fructooligosaccharides, Mannitol, Xylitol, Rhamnose, Cucumis Sativus (Cucumber) Fruit Extract, Propylene Glycol, Cetrimonium Bromide, Disodium EDTA.",
        }
    },
    {
        "id": "pc_18_hair_mask_shea_moisture_angled",
        "category": "personal_care",
        "product_type": "Hair Care / Conditioner",
        "product_name": "SheaMoisture Manuka Honey & Mafura Oil Hair Mask",
        "condition": "Angled (perspective distortion)",
        "transform": "perspective_tilt",
        "visible_ingredients": [
            "Water", "Cetearyl Alcohol", "Cocos Nucifera (Coconut) Oil", "Butyrospermum Parkii (Shea) Butter",
            "Glycerin", "Stearyl Alcohol", "Behentrimonium Chloride", "Panthenol", "Trichilia Emetica (Mafura) Seed Oil",
            "Honey", "Adansonia Digitata (Baobab) Seed Oil", "Ficus Carica (Fig) Fruit Extract", "Tocopherol",
            "Aloe Barbadensis Leaf Juice", "Caprylyl Glycol", "Phenoxyethanol", "Fragrance"
        ],
        "render_spec": {
            "brand": "SheaMoisture Established 1912",
            "product_name": "Manuka Honey & Mafura Oil Intensive Hydration Hair Masque",
            "claims": ["With Fig Extract & Baobab Oil", "Dry, Damaged Hair", "No Parabens, No Phthalates"],
            "ingredients": "Water, Cetearyl Alcohol, Cocos Nucifera (Coconut) Oil, Butyrospermum Parkii (Shea) Butter, Glycerin, Stearyl Alcohol, Behentrimonium Chloride, Panthenol, Trichilia Emetica (Mafura) Seed Oil, Honey, Adansonia Digitata (Baobab) Seed Oil, Ficus Carica (Fig) Fruit Extract, Tocopherol (Vitamin E), Aloe Barbadensis Leaf Juice, Caprylyl Glycol, Phenoxyethanol, Fragrance (Essential Oil Blend).",
        }
    },
]


def generate_images():
    print(f"Generating Phase 13 validation dataset in {OUTPUT_DIR}...")
    ground_truth_records = []

    for spec in DATASET_SPECS:
        img_id = spec["id"]
        out_path = OUTPUT_DIR / f"{img_id}.png"

        # Check if existing fixture is to be used
        if spec.get("use_existing_fixture"):
            src_fixture = Path(spec["use_existing_fixture"])
            if src_fixture.exists():
                shutil.copyfile(src_fixture, out_path)
                print(f"  [COPIED] {img_id} from {src_fixture}")
            else:
                print(f"  [ERROR] Source fixture {src_fixture} not found for {img_id}")
        elif spec.get("is_blank"):
            bs = spec["blank_spec"]
            pil_img = render_blank_label(bs["brand"], bs["title"], bs["volume"])
            pil_img.save(out_path, format="PNG")
            print(f"  [RENDERED BLANK] {img_id}")
        elif spec["category"] == "food":
            rs = spec["render_spec"]
            small_font = spec.get("small_font", False)
            pil_img = render_food_label(
                title=rs["title"],
                ingredients_text=rs["ingredients"],
                nutrition_rows=rs.get("nutrition"),
                serving_size=rs.get("serving_size"),
                allergen_text=rs.get("allergen"),
                sub_claims=rs.get("sub_claims"),
                small_font=small_font,
            )
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Apply transform
            trans = spec.get("transform")
            if trans == "perspective_tilt":
                cv_img = apply_perspective_tilt(cv_img, angle_deg=18.0)
            elif trans == "cylindrical_warp":
                cv_img = apply_cylindrical_warp(cv_img, curvature=0.35)
            elif trans == "mild_blur":
                cv_img = apply_mild_blur(cv_img, ksize=5)
            elif trans == "glare":
                cv_img = apply_glare(cv_img, intensity=0.45)
            elif trans == "low_contrast":
                cv_img = apply_low_contrast(cv_img, contrast=0.42, brightness=75)

            cv2.imwrite(str(out_path), cv_img)
            print(f"  [RENDERED FOOD] {img_id} (transform: {trans})")
        elif spec["category"] == "personal_care":
            rs = spec["render_spec"]
            small_font = spec.get("small_font", False)
            pil_img = render_personal_care_label(
                brand=rs["brand"],
                product_name=rs["product_name"],
                ingredients_text=rs["ingredients"],
                claims=rs.get("claims"),
                active_ingredients=rs.get("active_ingredients"),
                small_font=small_font,
            )
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Apply transform
            trans = spec.get("transform")
            if trans == "perspective_tilt":
                cv_img = apply_perspective_tilt(cv_img, angle_deg=18.0)
            elif trans == "cylindrical_warp":
                cv_img = apply_cylindrical_warp(cv_img, curvature=0.35)
            elif trans == "mild_blur":
                cv_img = apply_mild_blur(cv_img, ksize=5)
            elif trans == "glare":
                cv_img = apply_glare(cv_img, intensity=0.45)
            elif trans == "low_contrast":
                cv_img = apply_low_contrast(cv_img, contrast=0.42, brightness=75)

            cv2.imwrite(str(out_path), cv_img)
            print(f"  [RENDERED PC] {img_id} (transform: {trans})")

        # Ground truth record
        gt_record = {
            "id": img_id,
            "category": spec["category"],
            "product_type": spec["product_type"],
            "product_name": spec["product_name"],
            "condition": spec["condition"],
            "file_path": str(out_path),
            "visible_ingredients": spec["visible_ingredients"],
            "nutrition_values": spec.get("nutrition_values"),
            "allergens": spec.get("allergens", []),
        }
        ground_truth_records.append(gt_record)

    gt_file = OUTPUT_DIR / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth_records, f, indent=2)
    print(f"Saved ground truth metadata to {gt_file}")
    print(f"Total images generated: {len(ground_truth_records)}")


if __name__ == "__main__":
    generate_images()
