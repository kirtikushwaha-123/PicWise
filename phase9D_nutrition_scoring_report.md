# Phase 9D — Nutrition Scoring Methodology Reconstruction & Audit Report

**Project:** PicWise  
**Phase:** 9D — Nutrition Scoring Engine  
**Date:** 2026-09-16  
**Status:** Audit Complete — Missing Methodology Reported (Execution Halted per Section 4 Stop Condition)

---

## 1. Executive Summary & Audit Verdict

Per **Phase 9D Requirements Section 4 & 7**, an exhaustive audit and reconstruction of the PicWise repository was conducted to identify any existing mathematical scoring formulas, numerical nutrient weights, thresholds, priority mappings, normalization rules, or score ranges.

### Audit Result:
* **No quantitative scoring formula, numerical weights, nutrient thresholds, or point systems were ever implemented or documented in the PicWise repository.**
* This directly verifies the earlier Phase 9A architectural audit findings ([`phase9A_complete_codebase_audit.md`](phase9A_complete_codebase_audit.md#L37-L40), Lines 37–40 & 880):
  > *"The repository contains a 47-entry lookup table (`data/nutrition/nutrition_knowledge_dataset.csv`) providing qualitative nutritional properties (`Health Role`, `Health Impact`, `Decision Priority`, `Better Direction`). There is **no quantitative scoring formula or calculation engine** (e.g., Nutri-Score, NOVA score, or nutrient threshold algorithm) implemented in Python."*
* Per instructions, **no arbitrary thresholds, weights, penalties, or ranges have been invented or assumed**.
* In accordance with Section 7 of the user prompt, **implementation of the scoring engine is halted**. This report documents everything successfully established, details every missing methodology component, enumerates decisions requiring manual approval, and provides a parameter-agnostic schema for the upcoming implementation.

---

## 2. Successfully Established Components (Empirical Evidence)

The repository provides solid structural foundations for qualitative lookups and structured OCR extraction, which the scoring engine will consume once quantitative rules are specified.

### 2.1 Authoritative Nutrition Knowledge Base
**Source:** [`data/nutrition/nutrition_knowledge_dataset.csv`](data/nutrition/nutrition_knowledge_dataset.csv)  
**Total Records:** 47 rows, 6 columns (`Nutrient`, `Health Role`, `Health Impact`, `Decision Priority`, `Better Direction`, `Alternative / Packaging Names`). Zero null entries.

#### Categorical Distributions Across the 47 Nutrients:

1. **`Better Direction` (Directionality):**
   * **`Adequate` (27 nutrients, 57.4%):** Essential building blocks and micronutrients where sufficiency without gross deficiency is targeted.
     * *Nutrients:* Protein, Potassium, Calcium, Iron, Magnesium, Phosphorus, Zinc, Copper, Manganese, Selenium, Iodine, Chromium, Fluoride, Vitamins A, D, E, K, C, B1, B2, B3, B5, B6, B7, B9, B12, Choline.
   * **`Lower is Better` (7 nutrients, 14.9%):** Chronic disease risk factors where lower intake is nutritionally favorable.
     * *Nutrients:* Saturated Fat, Trans Fat, Sodium, Added Sugars, Total Sugars, Cholesterol, Caffeine.
   * **`Higher is Better` (6 nutrients, 12.8%):** Beneficial macro/micronutrients where higher intake is protective.
     * *Nutrients:* Dietary Fiber, Soluble Fiber, Insoluble Fiber, Monounsaturated Fat, Polyunsaturated Fat, Omega-3 Fatty Acids.
   * **`Appropriate` (1 nutrient, 2.1%):** Caloric balance where intake must match metabolic expenditure.
     * *Nutrients:* Energy / Calories.
   * **`Context Dependent` (6 nutrients, 12.8%):** Macronutrients whose health impact depends on source quality and dietary context.
     * *Nutrients:* Total Fat, Total Carbohydrate, Omega-6 Fatty Acids, Naturally Occurring Sugars, Starch, Sugar Alcohols / Polyols.

2. **`Decision Priority` (Importance):**
   * **`Very High` (5 nutrients):** Energy / Calories, Saturated Fat, Trans Fat, Added Sugars, Sodium.
   * **`High` (3 nutrients):** Total Fat, Total Sugars, Dietary Fiber.
   * **`Medium-High` (2 nutrients):** Protein, Total Carbohydrate.
   * **`Medium` (11 nutrients):** Cholesterol, Potassium, Calcium, Iron, Iodine, Vitamins A, D, C, B9, B12, Caffeine.
   * **`Low` (26 nutrients):** Unsaturated fats, individual fiber fractions, B-complex vitamins, minor trace minerals.

3. **`Health Impact`:**
   * `Positive` (29): Protein, beneficial fats, fiber fractions, all vitamins/minerals except context-dependent ones.
   * `Context Dependent` (11): Energy, Total Fat, Total Carbohydrate, Omega-6, natural sugars, starch, sugar alcohols, phosphorus, chromium, fluoride, caffeine.
   * `Negative in Excess` (4): Saturated Fat, Cholesterol, Total Sugars, Sodium.
   * `Negative` (1): Added Sugars.
   * `Very Negative` (1): Trans Fat.
   * `Very Positive` (1): Dietary Fiber.

4. **`Health Role`:**
   * `Essential` (26): Protein, essential vitamins and minerals.
   * `Context Dependent` (8): Energy, Total Fat, Total Carbohydrate, Omega-6, natural sugars, starch, sugar alcohols, caffeine.
   * `Beneficial` (6): Dietary Fiber, soluble/insoluble fiber, monounsaturated fat, polyunsaturated fat, omega-3.
   * `Risk Factor` (5): Saturated Fat, Trans Fat, Added Sugars, Total Sugars, Cholesterol.
   * `Essential/Risk` (2): Sodium, Fluoride.

---

### 2.2 Phase 9B OCR Structured Output
**Source:** [`backend/services/ocr_service/parsing/nutrition_parser.py`](backend/services/ocr_service/parsing/nutrition_parser.py)

The OCR parser extracts and outputs structured dictionaries mapping canonical keys to amounts, units, and spatial columns:
```python
{
    "energy": {
        "value": 450.0,
        "unit": "kcal",
        "per_100g": {"value": 450.0, "unit": "kcal"},
        "per_serving": {"value": 135.0, "unit": "kcal"}
    },
    "protein": {
        "value": 7.5,
        "unit": "g",
        "per_100g": {"value": 7.5, "unit": "g"}
    },
    "saturated_fat": {
        "value": 11.2,
        "unit": "g",
        "per_100g": {"value": 11.2, "unit": "g"}
    },
    "sodium": {
        "value": 420.0,
        "unit": "mg",
        "per_100g": {"value": 420.0, "unit": "mg"}
    }
}
```

* **Canonical Keys Handled by Parser:**
  `energy`, `protein`, `total_carbohydrate`, `total_sugars`, `total_fat`, `saturated_fat`, `trans_fat`, `dietary_fibre`, `sodium`, `salt`, `cholesterol`, `calcium`, `iron`, `vitamin`.
* **Units Detected:** `g`, `mg`, `mcg`, `kcal`, `kj`, `%`.
* **Column Separation:** Reconstructs table geometry to reliably isolate `per_100g` from `per_serving`.

---

### 2.3 Category Isolation & Routing
**Source:** [`backend/services/analysis_service/analyzer.py`](backend/services/analysis_service/analyzer.py)
* Product category is strictly user-selected (`food` vs `personal_care`).
* If `category == "food"`: Downstream of OCR, Food Safety ML runs on ingredients, and Nutrition Scoring will run on parsed nutrition table.
* If `category == "personal_care"`: Nutrition scoring is strictly skipped (`nutrition = None`).

---

## 3. Missing Quantitative Methodology Components

To build a deterministic, rule-based scoring engine without inventing arbitrary numbers, the following 12 exact methodology components must be resolved:

| # | Methodology Component | Current Project State | What Is Missing |
|---|---|---|---|
| **1** | **Positive vs. Negative Attribution** | `Better Direction` groups nutrients into `Higher is Better`, `Lower is Better`, `Adequate`, `Appropriate`, `Context Dependent`. | Exact mathematical rule defining whether `Adequate` (e.g., Protein, Micronutrients) adds positive points, or only prevents penalties. Whether `Appropriate` (Energy) is penalized if above a ceiling or penalized bidirectionally. |
| **2** | **Directionality Handling** | Qualitative strings only (`Higher is Better`, `Lower is Better`). | Mathematical transformation function (e.g., linear penalty, step-wise penalty bands, sigmoid, or threshold clipping) for each direction. |
| **3** | **Decision Priority Weighting** | 5 discrete qualitative ranks: `Very High`, `High`, `Medium-High`, `Medium`, `Low`. | Quantitative weights (e.g., is `Very High` = 4x, 3x, or exponential? Are `Low` priority nutrients included in the composite score or excluded?). |
| **4** | **Nutrient Thresholds / Bands** | No numerical thresholds exist anywhere in PicWise. | Baseline reference values per 100g (e.g., maximum recommended daily values, cutoffs for "high sugar", "high sodium", "high saturated fat", "sufficient fiber"). |
| **5** | **Normalization** | Units parsed as strings (`g`, `mg`, `kcal`, `kJ`). | Equation to convert heterogeneous nutrient units (g, mg, kcal) into a normalized dimensionless index before aggregation. |
| **6** | **Score Aggregation Formula** | None. | Mathematical operator: Weighted sum, deduction from 100 ($\text{Base} - \sum \text{Penalties} + \sum \text{Bonuses}$), ratio, or standard point-based system (FSA/Rayner/Nutri-Score $N - P$). |
| **7** | **Score Range & Representation** | Mockups display generic pills; no numerical range defined. | Output scale: `0 to 100` (integer), `0.0 to 10.0` (float), or points scale. |
| **8** | **Missing Nutrient Handling** | Labels rarely disclose all 47 nutrients (often only 4 to 8 mandatory fields appear). | Rule for undisclosed nutrients: Are they treated as 0, assumed neutral, imputed from category averages, or used to discount score confidence/coverage? |
| **9** | **Explicit-Zero Handling** | Parser extracts `{"value": 0.0, "unit": "g"}`. | Distinction between explicitly zero (e.g., "Trans Fat: 0g" -> maximum positive bonus/zero penalty) vs omitted nutrient (no data). |
| **10** | **Unit Conversion Rules** | Detected as raw units. | Mathematical conversion factors: $\text{salt (g)} \rightarrow \text{sodium (mg)}$ ($\times 400$ or $\times 393.4$), $\text{energy (kJ)} \rightarrow \text{kcal}$ ($\div 4.184$), $\text{mcg} \rightarrow \text{mg} \rightarrow \text{g}$. |
| **11** | **Serving-to-Per-100g Conversion** | Parser detects `per_serving` and `per_100g` columns. | Rule when label provides *only* `per_serving` amounts: If serving size (in grams) is known, scale $\times (100 / \text{serving\_size})$; if serving size is missing, how to handle? |
| **12** | **Minimum Viable Coverage Threshold** | None. | Minimum set of nutrients required to produce a valid score (e.g., minimum of Energy + at least 3 out of [Protein, Sugars, Fat, Sodium]) vs returning `None` / "Insufficient Nutrition Data". |

---

## 4. Specific Decisions & Parameters Requiring Approval

Before code implementation can begin, the user must select or provide the exact parameters for the following items:

### Decision 1: Scoring Paradigm
* **Approach A (FSA/Ofcom / Nutri-Score Rayner Profiling — Recommended):**  
  Internationally validated, deterministic, per-100g point deduction system.
  * Negative points ($N \in [0, 40]$): Energy (kJ), Saturated Fat (g), Total Sugars (g), Sodium (mg).
  * Positive points ($P \in [0, 15]$): Dietary Fiber (g), Protein (g).
  * Raw Points = $N - P$.
  * Normalized Health Score (0–100): $\text{Score} = \text{clamp}(100 - 2.5 \times (\text{Points} + 15), 0, 100)$.
* **Approach B (Priority-Weighted Daily Reference Value Penalty Index):**  
  Uses standard adult Daily Reference Values (DRV / Codex / FSSAI) per 100g.
  * Baseline = 100.
  * Subtracts penalties weighted by `Decision Priority` for each nutrient exceeding reference thresholds.
  * Adds bonuses for nutrients meeting `Higher is Better` / `Adequate` criteria.
* **Approach C (User Proprietary Formula):**  
  The user provides their exact custom mathematical formulation.

### Decision 2: Quantitative Priority Weights (if Approach B is selected)
If weights are derived from `Decision Priority`, the exact multiplier constants must be approved:
* `Very High` = ? (e.g., 4.0)
* `High` = ? (e.g., 3.0)
* `Medium-High` = ? (e.g., 2.0)
* `Medium` = ? (e.g., 1.0)
* `Low` = ? (e.g., 0.5 or ignored in composite score)

### Decision 3: Per-100g Reference Baselines (if Approach B is selected)
The exact baseline reference values for adult intake per 100g must be approved:
* Energy baseline (e.g., 2000 kcal / 8400 kJ daily, or per-100g ceiling)
* Saturated Fat limit (e.g., 5g / 100g)
* Trans Fat limit (e.g., 0.1g / 100g)
* Total Sugars limit (e.g., 10g / 100g)
* Sodium limit (e.g., 400mg / 100g or Salt 1.0g / 100g)
* Dietary Fiber target (e.g., 3g / 100g for "source of fiber", 6g / 100g for "high fiber")
* Protein target (e.g., 5g / 100g)

### Decision 4: Missing Nutrient Policy
Select policy when label provides only a partial nutrition table:
* *Policy 1 (Coverage Discount):* Compute score solely from available nutrients, but scale confidence / coverage indicator down proportionally.
* *Policy 2 (Neutral Default):* Missing nutrients neither penalize nor reward the score.
* *Policy 3 (Strict Abort):* If fewer than $K$ core nutrients are detected (e.g., $K < 3$), return `score = None` with status `"Insufficient Nutrition Data"`.

---

## 5. Proposed Parameter-Agnostic Engine Schema

Below is the proposed architectural interface for [`backend/services/nutrition_service/scoring_engine.py`](backend/services/nutrition_service/scoring_engine.py). It enforces strict separation of concerns, deterministic execution, and unit normalization **without hardcoding arbitrary constants**.

```python
"""
backend/services/nutrition_service/scoring_engine.py

Deterministic Nutrition Scoring Engine for PicWise Food Products.
Consumes structured OCR output and KnowledgeBase metadata.
"""

from typing import Dict, Any, Optional, Tuple


class NutritionScoringConfig:
    """
    Holds approved quantitative parameters, thresholds, and weights.
    Configurable to allow approved methodology injection.
    """
    def __init__(
        self,
        methodology_name: str,
        priority_weights: Dict[str, float],
        nutrient_thresholds: Dict[str, Dict[str, float]],
        score_range: Tuple[float, float] = (0.0, 100.0),
        min_required_nutrients: int = 3,
    ):
        self.methodology_name = methodology_name
        self.priority_weights = priority_weights
        self.nutrient_thresholds = nutrient_thresholds
        self.score_range = score_range
        self.min_required_nutrients = min_required_nutrients


def normalize_nutrient_values(
    raw_nutrition: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """
    Normalizes units (kJ -> kcal, salt -> sodium, mg -> g)
    and prefers 'per_100g' values where available.
    
    Returns:
        Dict mapping canonical nutrient name -> {"value_per_100g": float, "unit": str, "is_explicit_zero": bool}
    """
    pass


def calculate_nutrition_score(
    nutrition_data: Dict[str, Any],
    knowledge_base: Any,
    config: Optional[NutritionScoringConfig] = None,
) -> Dict[str, Any]:
    """
    Main entry point for nutrition scoring.
    
    Returns structured result:
    {
        "score": Optional[float],          # e.g., 72.5 (or None if insufficient data)
        "score_range": [0.0, 100.0],
        "grade": Optional[str],            # e.g., "A", "B", "C", "D", "E" if configured
        "confidence": float,               # 0.0 to 1.0 based on data coverage
        "available_nutrients_count": int,
        "evaluated_nutrients": {
            "saturated_fat": {
                "amount_per_100g": 3.2,
                "unit": "g",
                "direction": "Lower is Better",
                "priority": "Very High",
                "points_or_penalty": -2.0,
                "status": "Moderate"
            },
            ...
        },
        "missing_nutrients": ["dietary_fibre", "trans_fat"],
        "warnings": []
    }
    """
    pass
```

### Integration Point in Analyzer:
In [`backend/services/analysis_service/analyzer.py`](backend/services/analysis_service/analyzer.py):
```python
if category == "food":
    raw_nutrition = ocr_output.get("nutrition") or {}
    nutrition_scoring_result = calculate_nutrition_score(raw_nutrition, knowledge_base)
    nutrition_data = {
        "raw_ocr": raw_nutrition,
        "score_analysis": nutrition_scoring_result,
    }
else:
    nutrition_data = None  # Strictly skipped for Personal Care
```

---

## 6. Next Steps & Action Required

1. **User Review & Decision:**
   * Review Section 4 decisions (Scoring Paradigm, Weights, Thresholds, Missing-Value Policy).
   * Specify or approve the exact quantitative formula and parameters to be codified into `NutritionScoringConfig`.
2. **Implementation (Upon Approval):**
   * Implement `backend/services/nutrition_service/scoring_engine.py`.
   * Implement unit tests in `tests/test_nutrition_scoring.py` verifying deterministic scoring, zero handling, unit conversion, and missing nutrient handling.
   * Integrate downstream of OCR in `backend/services/analysis_service/analyzer.py` for `food` category only.
   * Verify zero regression across existing 58 tests.
