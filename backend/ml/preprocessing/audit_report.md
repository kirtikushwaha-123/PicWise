# PicWise Dataset Audit & Data Cleaning Report

**Audit Timestamp**: 2026-08-21 00:01:24

## 1. Duplicate & Near-Duplicate Check

- **Food Dataset Exact Duplicates**: 0 names found.
- **Personal Care Dataset Exact Duplicates**: 0 names found.

### Candidate Near-Duplicate Merges (High Similarity / Shared Alt Names):

- [Food] Candidate merge: `Green Peas` <-> `Green Beans` (similarity: 0.857)
- [Food] Candidate merge: `Soy Protein Isolate` <-> `Whey Protein Isolate` (similarity: 0.872)
- [Food] Candidate merge: `Soy Protein Isolate` <-> `Pea Protein Isolate` (similarity: 0.842)
- [Food] Candidate merge: `Soy Protein Concentrate` <-> `Whey Protein Concentrate` (similarity: 0.894)
- [Food] Candidate merge: `Soy Protein Concentrate` <-> `Milk Protein Concentrate` (similarity: 0.851)
- [Food] Candidate merge: `Whey Protein Isolate` <-> `Pea Protein Isolate` (similarity: 0.872)
- [Food] Candidate merge: `Whey Protein Concentrate` <-> `Milk Protein Concentrate` (similarity: 0.833)
- [Food] Candidate merge: `Coconut Oil` <-> `Coconut Milk` (similarity: 0.87)
- [Food] Candidate merge: `Lactose` <-> `Lactase` (similarity: 0.857)
- [Food] Candidate merge: `Sodium Caseinate` <-> `Sodium Alginate` (similarity: 0.839)
- [Food] Candidate merge: `Calcium Caseinate` <-> `Calcium Carbonate` (similarity: 0.824)
- [Food] Candidate merge: `Orange Juice Concentrate` <-> `Grape Juice Concentrate` (similarity: 0.894)
- [Food] Candidate merge: `Orange Juice Concentrate` <-> `Lemon Juice Concentrate` (similarity: 0.851)
- [Food] Candidate merge: `Grape Juice Concentrate` <-> `Lemon Juice Concentrate` (similarity: 0.826)
- [Food] Candidate merge: `Peach Puree` <-> `Pear Puree` (similarity: 0.857)
- [Food] Candidate merge: `Potassium Sorbate` <-> `Potassium Nitrate` (similarity: 0.824)
- [Food] Candidate merge: `Potassium Sorbate` <-> `Potassium Citrate` (similarity: 0.824)
- [Food] Candidate merge: `Sodium Nitrite` <-> `Sodium Nitrate` (similarity: 0.929)
- [Food] Candidate merge: `Sodium Nitrite` <-> `Sodium Citrate` (similarity: 0.857)
- [Food] Candidate merge: `Sodium Metabisulphite` <-> `Potassium Metabisulphite` (similarity: 0.844)
- [Food] Candidate merge: `Sodium Metabisulphite` <-> `Sodium Sulphite` (similarity: 0.833)
- [Food] Candidate merge: `Sodium Metabisulphite` <-> `Sodium Bisulphite` (similarity: 0.895)
- [Food] Candidate merge: `Sodium Nitrate` <-> `Sodium Citrate` (similarity: 0.929)
- [Food] Candidate merge: `Potassium Nitrate` <-> `Potassium Citrate` (similarity: 0.941)
- [Food] Candidate merge: `Sodium Diacetate` <-> `Sodium Lactate` (similarity: 0.867)
- [Food] Candidate merge: `Formic Acid` <-> `Folic Acid` (similarity: 0.857)
- [Food] Candidate merge: `Sodium Sulphite` <-> `Sodium Bisulphite` (similarity: 0.938)
- [Food] Candidate merge: `Polysorbate 80` <-> `Polysorbate 60` (similarity: 0.929)
- [Food] Candidate merge: `Sodium Stearoyl Lactylate` <-> `Calcium Stearoyl Lactylate` (similarity: 0.863)
- [Food] Candidate merge: `Ammonium Phosphatide` <-> `Diammonium Phosphate` (similarity: 0.9)

## 2. Alternate-Name Collision Check

- **Bucket (a) Harmless Collisions (Identical Labels)**: 14 alternate names.
  - Alt Name: `wheat gluten` -> Canonical: ['Wheat Gluten Protein', 'Vital Wheat Gluten'] (Safety: Safe, Allergy: High)
  - Alt Name: `casein` -> Canonical: ['Casein Protein', 'Casein'] (Safety: Safe, Allergy: Medium)
  - Alt Name: `milk casein` -> Canonical: ['Casein Protein', 'Casein'] (Safety: Safe, Allergy: Medium)
  - Alt Name: `hydrolysed whey protein` -> Canonical: ['Hydrolyzed Whey Protein', 'Whey Protein Hydrolysate'] (Safety: Safe, Allergy: Medium)
  - Alt Name: `sonth` -> Canonical: ['Dry Ginger Powder (Sonth)', 'Dry Ginger'] (Safety: Very Safe, Allergy: None)
  - Alt Name: `sorbitol` -> Canonical: ['Sorbitol (Humectant)', 'Sorbitol (Sweetener)'] (Safety: Safe, Allergy: None)
  - Alt Name: `INS 420` -> Canonical: ['Sorbitol (Humectant)', 'Sorbitol (Sweetener)'] (Safety: Safe, Allergy: None)
  - Alt Name: `E420` -> Canonical: ['Sorbitol (Humectant)', 'Sorbitol (Sweetener)'] (Safety: Safe, Allergy: None)
  - Alt Name: `INS 407` -> Canonical: ['Carrageenan', 'Kappa Carrageenan'] (Safety: Safe, Allergy: None)
  - Alt Name: `INS 418` -> Canonical: ['Gellan Gum', 'Clarified Gellan Gum'] (Safety: Safe, Allergy: None)

- **Bucket (b) TRUE CONFLICTS (Differing Safety or Allergy Labels)**: **0** alternate names remaining.

## 3. Missing / Blank Value Check

### Food Dataset Missing Value Summary:

- `Ingredient Name`: 0 missing / blank (0.0%)
- `Category`: 0 missing / blank (0.0%)
- `Safety Level`: 0 missing / blank (0.0%)
- `Allergy Risk`: 360 missing / blank (72.3%)
- `Health Impact`: 0 missing / blank (0.0%)
- `Processing Level`: 0 missing / blank (0.0%)
- `Regulatory Status`: 0 missing / blank (0.0%)
- `Packaging Names / Alternate Names`: 0 missing / blank (0.0%)

### Personal Care Dataset Missing Value Summary:

- `Ingredient_Name`: 0 missing / blank (0.0%)
- `Primary_Function`: 0 missing / blank (0.0%)
- `Ingredient_Category`: 0 missing / blank (0.0%)
- `Product_Categories`: 0 missing / blank (0.0%)
- `Origin`: 0 missing / blank (0.0%)
- `Safety_Level`: 0 missing / blank (0.0%)
- `Allergy_Risk`: 11 missing / blank (1.4%)
- `Irritation_Risk`: 446 missing / blank (56.7%)
- `Regulatory_Status`: 0 missing / blank (0.0%)

## 4. Label Consistency by Category & Outlier Detection

### Group 1: Outlier is HIGHER Risk than Category Peers (40 items)
*Expected domain variations (e.g. allergens/additives in generally safe categories).*
- [Food | `Anti-Caking Agents`] `Sodium Ferrocyanide` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Food | `Anti-Caking Agents`] `Potassium Ferrocyanide` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Food | `Cereals & Grains`] `Refined Wheat Flour (Maida)` (Safety Level): **Safe** vs Category Dominant: **Very Safe**
- [Food | `Cereals & Grains`] `Barley` (Allergy Risk): **Low** vs Category Dominant: **None**
- [Food | `Dairy Ingredients`] `Condensed Milk` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Food | `Emulsifiers`] `Soy Lecithin` (Allergy Risk): **Medium** vs Category Dominant: **None**
- [Food | `Emulsifiers`] `Lecithin (source unspecified)` (Allergy Risk): **Medium** vs Category Dominant: **None**
- [Food | `Emulsifiers`] `Egg Yolk Lecithin` (Allergy Risk): **High** vs Category Dominant: **None**
- [Food | `Flour & Starch`] `Modified Starch` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Food | `Flour & Starch`] `Maltodextrin` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Food | `Flour & Starch`] `Wheat Starch` (Allergy Risk): **Medium** vs Category Dominant: **None**
- [Food | `Flour & Starch`] `Vital Wheat Gluten` (Allergy Risk): **High** vs Category Dominant: **None**
- [Food | `Herbs & Spices`] `Asafoetida (Hing)` (Allergy Risk): **Low** vs Category Dominant: **None**
- [Food | `Humectants`] `Propylene Glycol` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Food | `Humectants`] `Hexylene Glycol` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Food | `Protein Sources`] `Egg White Powder` (Allergy Risk): **High** vs Category Dominant: **Medium**
- [Food | `Protein Sources`] `Wheat Gluten Protein` (Allergy Risk): **High** vs Category Dominant: **Medium**
- [Food | `Pulses & Legumes`] `Soy Flour` (Allergy Risk): **Medium** vs Category Dominant: **Low**
- [Food | `Pulses & Legumes`] `Soybean` (Allergy Risk): **Medium** vs Category Dominant: **Low**
- [Food | `Raising Agents`] `Sodium Aluminium Phosphate` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Food | `Vegetable Oils`] `Groundnut Oil` (Allergy Risk): **High** vs Category Dominant: **None**
- [Food | `Vegetable Oils`] `Sesame Oil` (Allergy Risk): **High** vs Category Dominant: **None**
- [Personal Care | `Acid`] `Glycolic Acid` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Personal Care | `Acid`] `Kojic Acid` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Personal Care | `Acid`] `Glycolic Acid` (Allergy Risk): **Medium** vs Category Dominant: **Low**
- [Personal Care | `Acid`] `Kojic Acid` (Allergy Risk): **Medium** vs Category Dominant: **Low**
- [Personal Care | `Botanical Extract`] `Capsicum Extract (Lip Plumper)` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Personal Care | `Butter`] `Theobroma Cacao Seed Butter (Cocoa Butter)` (Allergy Risk): **Medium** vs Category Dominant: **Low**
- [Personal Care | `Colorant`] `p-Phenylenediamine (PPD)` (Safety Level): **High Risk** vs Category Dominant: **Moderate Risk**
- [Personal Care | `Humectant`] `Propylene Glycol` (Allergy Risk): **Medium** vs Category Dominant: **Low**
- [Personal Care | `Mineral`] `Sulfur` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Personal Care | `Mineral`] `Talc` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Personal Care | `Mineral`] `Sulfur` (Allergy Risk): **Medium** vs Category Dominant: **Low**
- [Personal Care | `Pigment`] `Carmine (CI 75470)` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Personal Care | `Pigment`] `Carmine (CI 75470)` (Allergy Risk): **Medium** vs Category Dominant: **Low**
- [Personal Care | `Silicone`] `Cyclotetrasiloxane` (Safety Level): **Moderate Risk** vs Category Dominant: **Safe**
- [Personal Care | `Vitamin`] `Retinol` (Allergy Risk): **Medium** vs Category Dominant: **Low**
- [Personal Care | `Vitamin`] `Retinaldehyde` (Allergy Risk): **Medium** vs Category Dominant: **Low**
- [Personal Care | `Wax`] `Lanolin` (Allergy Risk): **High** vs Category Dominant: **Low**
- [Personal Care | `Wax`] `Lanolin (Hair Pomade)` (Allergy Risk): **High** vs Category Dominant: **Low**

### Group 2: Outlier is LOWER Risk than Category Peers (23 items)
*Candidate rows for manual review (ingredient labeled safer than its category norm).*
- [Food | `Dairy Ingredients`] `Ghee` (Allergy Risk): **Low** vs Category Dominant: **Medium**
- [Food | `Flavour Enhancers`] `Yeast Extract` (Safety Level): **Safe** vs Category Dominant: **Moderate Risk**
- [Food | `Flavour Enhancers`] `Autolyzed Yeast Extract` (Safety Level): **Safe** vs Category Dominant: **Moderate Risk**
- [Food | `Nuts & Seeds`] `Flax Seeds` (Allergy Risk): **Low** vs Category Dominant: **High**
- [Food | `Nuts & Seeds`] `Chia Seeds` (Allergy Risk): **Low** vs Category Dominant: **High**
- [Food | `Nuts & Seeds`] `Poppy Seeds (Khus Khus)` (Allergy Risk): **Medium** vs Category Dominant: **High**
- [Food | `Processing Aids`] `Activated Carbon` (Safety Level): **Safe** vs Category Dominant: **Moderate Risk**
- [Personal Care | `Acid`] `Citric Acid` (Safety Level): **Very Safe** vs Category Dominant: **Safe**
- [Personal Care | `Emulsifier`] `Glyceryl Stearate` (Safety Level): **Very Safe** vs Category Dominant: **Safe**
- [Personal Care | `Essential Oil`] `Chamomile Oil (Baby Care)` (Safety Level): **Safe** vs Category Dominant: **Moderate Risk**
- [Personal Care | `Fatty Acid`] `Stearic Acid` (Safety Level): **Very Safe** vs Category Dominant: **Safe**
- [Personal Care | `Humectant`] `Glycerin` (Safety Level): **Very Safe** vs Category Dominant: **Safe**
- [Personal Care | `Mineral`] `Kaolin` (Safety Level): **Very Safe** vs Category Dominant: **Safe**
- [Personal Care | `Mineral`] `Zinc Oxide (Baby Care)` (Safety Level): **Very Safe** vs Category Dominant: **Safe**
- [Personal Care | `Salt`] `Sodium Chloride` (Allergy Risk): **None** vs Category Dominant: **Low**
- [Personal Care | `Silicone`] `Dimethicone` (Safety Level): **Very Safe** vs Category Dominant: **Safe**
- [Personal Care | `Solvent`] `Aqua (Water)` (Allergy Risk): **None** vs Category Dominant: **Low**
- [Personal Care | `Sugar`] `Sorbitol` (Allergy Risk): **None** vs Category Dominant: **Low**
- [Personal Care | `Sugar`] `Trehalose` (Allergy Risk): **None** vs Category Dominant: **Low**
- [Personal Care | `Surfactant`] `Decyl Glucoside` (Safety Level): **Very Safe** vs Category Dominant: **Safe**
- [Personal Care | `Surfactant`] `Coco-Glucoside` (Safety Level): **Very Safe** vs Category Dominant: **Safe**
- [Personal Care | `Wax`] `Petrolatum` (Safety Level): **Very Safe** vs Category Dominant: **Safe**
- [Personal Care | `Wax`] `Petrolatum (Baby Care)` (Safety Level): **Very Safe** vs Category Dominant: **Safe**

## 5. Spelling & Formatting Normalization

Applied resolution rules and non-semantic text formatting to `_cleaned` files.
- Food dataset: `data/food/ingredient_knowledge_base_500_cleaned.csv`
- Personal care dataset: `data/personal_care/personal_care_ingredients_dataset_cleaned.xlsx`

## 6. Class Distribution Summary

### Normalized Safety Level Distribution:

**Food Dataset Safety Level:**
- Safe: 233
- Very Safe: 171
- Moderate Risk: 71
- High Risk: 23

**Personal Care Dataset Safety Level:**
- Safe: 633
- Moderate Risk: 96
- Very Safe: 56
- High Risk: 1

**Combined Safety Level:**
- Safe: 866
- Very Safe: 227
- Moderate Risk: 167
- High Risk: 24

### Normalized Allergy Risk Distribution:

**Food Dataset Allergy Risk:**
- None: 360
- Low: 60
- Medium: 58
- High: 20

**Personal Care Dataset Allergy Risk:**
- Low: 647
- Medium: 121
- None: 11
- High: 7

**Combined Allergy Risk:**
- Low: 707
- None: 371
- Medium: 179
- High: 27