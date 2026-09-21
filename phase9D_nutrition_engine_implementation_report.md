# Phase 9D — Nutrition Scoring Engine Implementation Report

**Project:** PicWise  
**Phase:** 9D — Nutrition Scoring Engine  
**Date:** 2026-09-17  
**Status:** Implementation Complete & Fully Verified  

---

## 1. Executive Summary

In Phase 9D, the PicWise Deterministic Nutrition Scoring Engine was implemented strictly following the locked source of truth in [`phase9D_nutrition_methodology_spec.md`](phase9D_nutrition_methodology_spec.md).

* **Zero Heuristic/ML Shortcuts:** The scoring engine is 100% deterministic, rule-based, and mathematical.
* **Component Modularity:** Dedicated modular package created under `backend/services/nutrition_service/` (`constants.py`, `models.py`, `normalization.py`, `scorer.py`).
* **Existing Logic Preserved:** `lookup.py` and existing qualitative lookups were preserved with 100% backward compatibility.
* **Rigorous Boundary Verification:** Exact equality boundaries ($\ge$) verified for both Catastrophic Risk and Excessive Sugar guardrails across all solid food, beverage, and culinary fat contexts.
* **Strict Missing vs. Explicit Zero Semantics:** Explicit zero ($0.0$) evaluated as verified absence of risk ($p(0) \approx 4.74$) while unprovided nutrients remain strictly marked as missing, without silent default substitution.
* **Comprehensive Test Coverage:** 45 unit tests in `tests/test_nutrition_scoring.py` covering every formula, threshold, saturation curve, guardrail boundary, unit conversion, missing-value condition, and edge case. All 45 tests pass with 0 failures.

---

## 2. Files Created & Modified

### Created:
1. [`backend/services/nutrition_service/constants.py`](backend/services/nutrition_service/constants.py):
   All locked numerical parameters, thresholds ($T_i$), weights ($w_i$, $w_{\text{neg}}$, $w_{\text{pos}}$, $\beta$), $\tau$ constants, energy tiers, and guardrail limits.
2. [`backend/services/nutrition_service/models.py`](backend/services/nutrition_service/models.py):
   Clean dataclasses (`NormalizedNutrient`, `EvaluatedNutrient`, `NutritionScoreResult`) guaranteeing serializability and type safety.
3. [`backend/services/nutrition_service/normalization.py`](backend/services/nutrition_service/normalization.py):
   Canonical key normalization, unit conversions (salt $\to$ sodium, kJ $\to$ kcal, mg $\to$ g), per-serving scaling, context detection (beverage, culinary fat, pure water), and explicit zero vs missing data isolation.
4. [`backend/services/nutrition_service/scorer.py`](backend/services/nutrition_service/scorer.py):
   Core calculation engine implementing logistic penalties, weighted Minkowski $L_2$ (RMS) aggregation, stepwise energy deductions, exponential saturation, anti-fortification factor $\gamma$, sugar guardrail, and catastrophic risk guardrails.
5. [`tests/test_nutrition_scoring.py`](tests/test_nutrition_scoring.py):
   Comprehensive unit test suite comprising 45 test cases including exact equality-boundary and missing-data verification.
6. [`phase9D_nutrition_engine_implementation_report.md`](phase9D_nutrition_engine_implementation_report.md):
   This verification and implementation report.

### Modified:
1. [`backend/services/nutrition_service/__init__.py`](backend/services/nutrition_service/__init__.py):
   Exported `calculate_nutrition_score` and `normalize_nutrition_data` while preserving `find_relevant_nutrition`.

---

## 3. Architecture & Separation of Concerns

```text
backend/services/nutrition_service/
├── __init__.py           # Unified module interface
├── constants.py          # Locked parameters & mathematical bounds
├── models.py             # Dataclasses & structured result contracts
├── normalization.py      # Unit conversion & explicit-zero detection
├── scorer.py             # Mathematical scoring engine & guardrails
└── lookup.py             # Pre-existing qualitative text lookup (intact)
```

---

## 4. Locked Formulas Implemented

### 4.1 Primary Risk Logistic Penalty Function
For each primary risk factor $i \in \{\text{Saturated Fat}, \text{Added Sugar}, \text{Sodium}, \text{Trans Fat}\}$:

$$p_i(x_i) = \frac{100.0}{1.0 + \exp\left(-3.0 \cdot \left(\frac{x_i - T_i}{T_i}\right)\right)}$$

* At threshold $x_i = T_i \implies p_i(T_i) = 50.0$.
* Asymptote $x_i \to \infty \implies p_i(x_i) \to 100.0$.
* Explicit zero $x_i = 0.0 \implies p_i(0.0) = \frac{100}{1 + e^3} \approx 4.74$.
* Numerical stability: Argument to `exp()` is clamped to $[-50.0, 50.0]$ preventing overflow.

### 4.2 Weighted Normalized Minkowski $L_2$ (Root-Mean-Square) Risk Aggregation

$$P_{\text{risk}} = \sqrt{\frac{\sum_{i=1}^{M} w_i \cdot [p_i(x_i)]^2}{\sum_{i=1}^{M} w_i}}$$

* All primary risk weights $w_i = 1.0$.
* Square root is strictly enforced, keeping $P_{\text{risk}} \in [0.0, 100.0]$.

### 4.3 Stepwise Energy Deduction & Negative Component Base Score

$$S_{\text{neg}} = \max\left(0.0, 100.0 - P_{\text{risk}} - \text{penalty}_{\text{energy}}\right)$$

* $\text{Energy} < 250\text{ kcal} \implies \text{penalty}_{\text{energy}} = 0.0$.
* $250 \le \text{Energy} \le 400\text{ kcal} \implies \text{penalty}_{\text{energy}} = 15.0$.
* $\text{Energy} > 400\text{ kcal} \implies \text{penalty}_{\text{energy}} = 30.0$.

### 4.4 Positive Macronutrient Exponential Saturation

$$f(x, \tau) = 100.0 \times \left(1.0 - e^{-x / \tau}\right)$$

$$S_{\text{pos}} = 0.45 \cdot f(x_{\text{fiber}}, 4.0) + 0.35 \cdot f(x_{\text{protein}}, 8.0) + 0.20 \cdot f(x_{\text{unsat}}, 12.0)$$

where:
$$\text{unsaturated\_fat} = \max\left(0.0, \text{total\_fat} - (\text{saturated\_fat} + \text{trans\_fat})\right)$$

### 4.5 Anti-Fortification Discount ($\gamma$) & Top-Level Synthesis

$$\gamma = \max\left(0.0, 1.0 - \frac{P_{\text{risk}}}{50.0}\right)$$

$$\text{Score}_{\text{raw}} = 0.60 \cdot S_{\text{neg}} + 0.28 \cdot S_{\text{pos}} + 0.12 \cdot (\gamma \cdot S_{\text{micro}})$$

---

## 5. Locked Parameter Values & Boundary Definitions

| Parameter | Value | Condition Operator | Scope / Behavior |
| :--- | :---: | :---: | :--- |
| **Saturated Fat Threshold** | `4.0 g` (food) / `20.0%` (culinary fat) | — | Per 100g standard |
| **Added Sugar Threshold** | `10.0 g` (food) / `5.0 g` (beverage) | — | Per 100g standard |
| **Total Sugar Fallback Threshold** | `12.5 g` (food) / `6.25 g` (beverage) | — | Used when Added Sugar is missing |
| **Sodium Threshold** | `400.0 mg` ($0.4\text{ g}$) | — | Per 100g standard |
| **Trans Fat Threshold** | `0.3 g` | — | Per 100g standard |
| **Logistic Steepness** | $k_i = 3.0$ | — | Sigmoid slope |
| **Positive Weights** | `0.45 / 0.35 / 0.20` | — | Fiber / Protein / Unsaturated Fat |
| **Saturation Tau Scales** | `4.0g / 8.0g / 12.0g` | — | Fiber / Protein / Unsaturated Fat |
| **Top-Level Weights** | `0.60 / 0.28 / 0.12` | — | $w_{\text{neg}} + w_{\text{pos}} + \beta = 1.00$ |
| **Sugar Guardrail Multiplier** | `2.5x` | $\ge 2.5 \times T_{\text{sugar}}$ | Triggers at $\ge 25.0\text{g}$ (food) / $\ge 12.5\text{g}$ (drink) $\implies S_{\text{pos}} = 0, S_{\text{micro}} = 0$ |
| **Catastrophic Trigger** | `2.5x` | $\ge 2.5 \times T_i$ | Triggers if any primary risk $x_i \ge 2.5 \times T_i$ |
| **Catastrophic Ceiling** | `35.0` | $\le 35.0$ | Clamps $\text{Final Score} = \min(\text{Score}_{\text{raw}}, 35.0)$ |
| **Water Override** | `100.0` | Exact match | Pure drinking water override |

---

## 6. Guardrail Order of Operations

The execution order strictly enforces non-compensatory health protections:

1. **Category Isolation:** If `category == "personal_care"`, return `nutrition = None`.
2. **Water Override:** If verified pure drinking water, return `score = 100.0` immediately.
3. **Data Completeness Check:** If detected core nutrients $< 3$, return `score = None`, `status = "Insufficient Nutrition Data"`.
4. **Negative Risk Evaluation:** Calculate $p_i(x_i)$, normalized RMS $P_{\text{risk}}$, and stepwise energy deduction.
5. **Positive & Micronutrient Evaluation:** Calculate $S_{\text{pos}}$, $S_{\text{micro}}$, and discount $\gamma$.
6. **Sugar Guardrail Check:** If Added Sugar $\ge 2.5 \times T_{\text{sugar}}$, force $S_{\text{pos}} = 0.0$ and $S_{\text{micro}} = 0.0 \implies \text{Score}_{\text{raw}} = 0.60 \cdot S_{\text{neg}}$.
7. **Score Synthesis:** Calculate $\text{Score}_{\text{raw}} = 0.60 S_{\text{neg}} + 0.28 S_{\text{pos}} + 0.12 (\gamma S_{\text{micro}})$.
8. **Catastrophic Guardrail Check:** If any primary risk $x_i \ge 2.5 \times T_i$, clamp $\text{Final Score} = \min(\text{Score}_{\text{raw}}, 35.0)$.
9. **Final Clamping & Rounding:** Constrain to $[0.0, 100.0]$ and round to 1 decimal place.

---

## 7. Exact Equality-Boundary Verifications

All guardrails use `>=` (greater than or equal to), strictly matching the authoritative specification:

### 7.1 Catastrophic-Risk Equality Boundaries ($x_i \ge 2.5 \times T_i$)
* **Saturated Fat (Solid Food):** $T = 4.0\text{g} \implies \text{Boundary} = 10.00\text{g}$
  * $9.99\text{g} \implies$ No catastrophic trigger.
  * $10.00\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
  * $10.01\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
* **Saturated Fat (Culinary Fat):** $T = 20.0\text{g} \implies \text{Boundary} = 50.00\text{g}$
  * $49.99\text{g} \implies$ No catastrophic trigger.
  * $50.00\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
  * $50.01\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
* **Added Sugar (Solid Food):** $T = 10.0\text{g} \implies \text{Boundary} = 25.00\text{g}$
  * $24.99\text{g} \implies$ No catastrophic trigger.
  * $25.00\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
  * $25.01\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
* **Added Sugar (Beverage):** $T = 5.0\text{g} \implies \text{Boundary} = 12.50\text{g}$
  * $12.49\text{g} \implies$ No catastrophic trigger.
  * $12.50\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
  * $12.51\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
* **Total Sugar Fallback (Solid Food):** $T = 12.5\text{g} \implies \text{Boundary} = 31.25\text{g}$
  * $31.24\text{g} \implies$ No catastrophic trigger.
  * $31.25\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
  * $31.26\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
* **Total Sugar Fallback (Beverage):** $T = 6.25\text{g} \implies \text{Boundary} = 15.625\text{g}$
  * $15.62\text{g} \implies$ No catastrophic trigger.
  * $15.625\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
  * $15.63\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
* **Sodium:** $T = 400.0\text{mg} \implies \text{Boundary} = 1000.0\text{mg}$
  * $999.9\text{mg} \implies$ No catastrophic trigger.
  * $1000.0\text{mg} \implies$ Catastrophic trigger; score capped $\le 35.0$.
  * $1000.1\text{mg} \implies$ Catastrophic trigger; score capped $\le 35.0$.
* **Trans Fat:** $T = 0.3\text{g} \implies \text{Boundary} = 0.75\text{g}$
  * $0.74\text{g} \implies$ No catastrophic trigger.
  * $0.75\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.
  * $0.76\text{g} \implies$ Catastrophic trigger; score capped $\le 35.0$.

### 7.2 Excessive Added Sugar Equality Boundaries ($x_{\text{sugar}} \ge 2.5 \times T_{\text{sugar}}$)
* **General Solid Foods:** $T_{\text{sugar}} = 10.0\text{g} \implies \text{Boundary} = 25.00\text{g}$
  * $24.99\text{g} \implies$ Guardrail does not trigger; positive points awarded ($S_{\text{pos}} > 0$).
  * $25.00\text{g} \implies$ Guardrail triggers; $S_{\text{pos}} = 0.0$, $S_{\text{micro}} = 0.0$.
  * $25.01\text{g} \implies$ Guardrail triggers; $S_{\text{pos}} = 0.0$, $S_{\text{micro}} = 0.0$.
* **Beverages & Liquids:** $T_{\text{sugar}} = 5.0\text{g} \implies \text{Boundary} = 12.50\text{g}$
  * $12.49\text{g} \implies$ Guardrail does not trigger; positive points awarded ($S_{\text{pos}} > 0$).
  * $12.50\text{g} \implies$ Guardrail triggers; $S_{\text{pos}} = 0.0$, $S_{\text{micro}} = 0.0$.
  * $12.51\text{g} \implies$ Guardrail triggers; $S_{\text{pos}} = 0.0$, $S_{\text{micro}} = 0.0$.

---

## 8. Missing vs. Explicit-Zero Representation & Semantics

The engine preserves strict separation between verified absence ($0.0$) and unprovided data across all 10 evaluated nutrients:

```text
┌─────────────────────────────────────────────────────────────┐
│ "Trans Fat: 0g"       ≠      "Trans Fat: [Not Reported]"   │
│ (Explicit Zero: is_explicit_zero=True) (Missing: is_missing=True) │
└─────────────────────────────────────────────────────────────┘
```

1. **Fiber (`dietary_fibre`):**
   * *Explicit Zero:* Recorded in `nutrients_evaluated` (`amount=0.0`, `is_explicit_zero=True`, `positive_score=0.0`). Not in `nutrients_missing`.
   * *Missing:* Appended to `nutrients_missing`. Not in `nutrients_evaluated`. Contributes $0.0$ positive credit.
2. **Protein (`protein`):**
   * *Explicit Zero:* Recorded in `nutrients_evaluated` (`is_explicit_zero=True`, `positive_score=0.0`). Not in `nutrients_missing`.
   * *Missing:* Appended to `nutrients_missing`. Contributes $0.0$ positive credit.
3. **Total Fat (`total_fat`):**
   * *Explicit Zero:* Recorded in `nutrients_evaluated` (`is_explicit_zero=True`). Not in `nutrients_missing`.
   * *Missing:* Appended to `nutrients_missing`.
4. **Saturated Fat (`saturated_fat`):**
   * *Explicit Zero:* Evaluates to baseline minimum risk penalty ($p(0.0) = \frac{100}{1+e^3} \approx 4.74$). `is_explicit_zero=True`. Not in `nutrients_missing`.
   * *Missing:* Appended to `nutrients_missing`. No penalty computed from zero.
5. **Trans Fat (`trans_fat`):**
   * *Explicit Zero:* Evaluates to baseline minimum ($p(0.0) \approx 4.74$). `is_explicit_zero=True`. Not in `nutrients_missing`.
   * *Missing:* Appended to `nutrients_missing`. Assigned neutral baseline penalty ($p = 10.0$) if hydrogenated oils absent, or $p = 50.0$ if present.
6. **Added Sugar (`added_sugars`):**
   * *Explicit Zero:* Evaluates to baseline minimum ($p(0.0) \approx 4.74$). `is_explicit_zero=True`. Not in `nutrients_missing`.
   * *Missing:* Falls back to Total Sugars ($T=12.5\text{g}$). Appended to `nutrients_missing`.
7. **Total Sugars (`total_sugars`):**
   * *Explicit Zero:* Recorded with `is_explicit_zero=True`. Not in `nutrients_missing`.
   * *Missing:* Appended to `nutrients_missing`.
8. **Sodium (`sodium`):**
   * *Explicit Zero:* Evaluates to baseline minimum ($p(0.0) \approx 4.74$). `is_explicit_zero=True`. Not in `nutrients_missing`.
   * *Missing:* Appended to `nutrients_missing`. No penalty computed from zero.
9. **Energy (`energy`):**
   * *Explicit Zero:* Evaluates with $\text{penalty}_{\text{energy}} = 0.0$, `is_explicit_zero=True`. Not in `nutrients_missing`.
   * *Missing:* Appended to `nutrients_missing`. Deducts $0.0$ penalty and generates explicit warning.
10. **Micronutrients (Calcium, Iron, Potassium, Zinc, Magnesium, Vitamins A, C, D):**
    * *Explicit Zero:* Recorded in `nutrients_evaluated` with `is_explicit_zero=True`. Contributes $0.0$ adequacy credit.
    * *Missing:* Not recorded in `nutrients_evaluated`. Contributes $0.0$ adequacy credit.

---

## 9. Test Coverage & Verification Results

### Dedicated Nutrition Scoring Test Suite: [`tests/test_nutrition_scoring.py`](tests/test_nutrition_scoring.py)
* `TestLogisticRiskPenalty`: 4 tests (p(T)=50, p(0) baseline, monotonic ascent, asymptotic saturation).
* `TestMinkowskiL2Aggregation`: 2 tests (square root enforcement, extreme outlier penalty).
* `TestPositiveSaturation`: 3 tests (zero value, tau scale value, asymptotic saturation).
* `TestEnergyStepwisePenalty`: 1 test (249, 250, 400, 401 kcal boundaries).
* `TestUnsaturatedFatCalculation`: 2 tests (derived calculation, non-negative clamp).
* `TestSugarGuardrail`: 4 tests (general food trigger, beverage trigger, and exact equality-boundary tests for foods at 24.99g/25.00g/25.01g and beverages at 12.49g/12.50g/12.51g).
* `TestCatastrophicRiskGuardrail`: 10 tests (general extreme sodium, sat fat, trans fat, plus exact equality-boundary tests for sat fat food 9.99/10.00/10.01g, culinary fat 49.99/50.00/50.01g, added sugar food 24.99/25.00/25.01g, added sugar beverage 12.49/12.50/12.51g, total sugar fallback food 31.24/31.25/31.26g, sodium 999.9/1000.0/1000.1mg, trans fat 0.74/0.75/0.76g).
* `TestMissingVsExplicitZero`: 10 tests (verifying explicit zero vs missing semantics for Trans Fat, Added Sugar, Fiber, Protein, Total Fat, Saturated Fat, Sodium, Energy, Total Sugars, and Micronutrients).
* `TestUnitConversions`: 3 tests (salt to sodium, kJ to kcal, serving size scaling).
* `TestCompletenessAndCategoryGating`: 3 tests (personal care gating, insufficient data abort, 3-nutrient minimum).
* `TestWaterOverride`: 1 test (pure drinking water override to 100.0).
* `TestDeterminismAndEdgeCases`: 2 tests (repeatability determinism, negative value safe handling).

**Dedicated Test Suite Results:** **45 / 45 tests passed (0.012s, 100% pass rate).**

### Full Project Regression Test Suite:
* `tests/test_nutrition_scoring.py` (45 tests)
* `tests/test_nutrition.py` (7 tests)
* `tests/test_food_safety_production.py` (8 tests)
* `tests/test_knowledge_base.py` (10 tests)
* `tests/test_matcher.py` (15 tests)
* `tests/test_analysis.py` (3 tests)
* `tests/test_ocr_integration.py` (3 tests)
* `tests/test_ml_evaluation.py` (5 tests)
* `tests/test_ml_error_analysis.py` (5 tests)
* `tests/test_ml_improvement_experiments.py` (5 tests)

**Total Regression Tests:** **106 / 106 tests passed (100% pass rate).**

---

## 10. Specification Compliance & Working Tree State

* **Catastrophic-Risk Operator:** Verified as `>=` across all risk factors and product contexts.
* **Sugar Guardrail Operator:** Verified as `>=` across solid food and beverage contexts.
* **Missing vs. Explicit Zero:** Strict separation verified in models, normalization, scoring, and tests.
* **Deviations from Specification:** **ZERO.**
* **Working Tree State:** Uncommitted and unpushed.

```text
Specification compliance: VERIFIED
Commit status: NOT COMMITTED
Push status: NOT PUSHED
```
