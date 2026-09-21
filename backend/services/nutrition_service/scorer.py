"""
backend/services/nutrition_service/scorer.py

PicWise Deterministic Nutrition Scoring Engine.
Consumes normalized nutritional data per 100g and calculates a deterministic
health score (0.0 - 100.0) in accordance with phase9D_nutrition_methodology_spec.md.
"""

import math
from typing import Dict, Any, Optional, List, Tuple
from backend.services.nutrition_service.constants import (
    SCORE_MIN,
    SCORE_MAX,
    THRESHOLD_SATURATED_FAT_FOOD,
    THRESHOLD_SATURATED_FAT_CULINARY,
    THRESHOLD_ADDED_SUGAR_FOOD,
    THRESHOLD_ADDED_SUGAR_BEVERAGE,
    THRESHOLD_TOTAL_SUGAR_FALLBACK_FOOD,
    THRESHOLD_TOTAL_SUGAR_FALLBACK_BEVERAGE,
    THRESHOLD_SODIUM,
    THRESHOLD_TRANS_FAT,
    LOGISTIC_K,
    EXP_CLAMP_MIN,
    EXP_CLAMP_MAX,
    PRIMARY_RISK_WEIGHTS,
    MISSING_TRANS_FAT_PENALTY,
    HYDROGENATED_OIL_KEYWORDS,
    ENERGY_LOW_THRESHOLD,
    ENERGY_HIGH_THRESHOLD,
    ENERGY_PENALTY_LOW,
    ENERGY_PENALTY_MEDIUM,
    ENERGY_PENALTY_HIGH,
    WEIGHT_FIBER,
    TAU_FIBER,
    WEIGHT_PROTEIN,
    TAU_PROTEIN,
    WEIGHT_UNSATURATED_FAT,
    TAU_UNSATURATED_FAT,
    WEIGHT_SNEG,
    WEIGHT_SPOS,
    WEIGHT_SMICRO,
    GAMMA_PRISK_DIVISOR,
    WATER_SCORE,
    SUGAR_GUARDRAIL_MULTIPLIER,
    CATASTROPHIC_MULTIPLIER,
    CATASTROPHIC_CEILING,
    MIN_CORE_NUTRIENTS,
)
from backend.services.nutrition_service.models import (
    NormalizedNutrient,
    EvaluatedNutrient,
    NutritionScoreResult,
)
from backend.services.nutrition_service.normalization import (
    normalize_nutrition_data,
)


def calculate_nutrition_score(
    raw_nutrition: Optional[Dict[str, Any]],
    category: str = "food",
    product_text: str = "",
    ingredient_text: str = "",
    serving_size_grams: Optional[float] = None,
    knowledge_base: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Main entrypoint for the PicWise Deterministic Nutrition Scoring Engine.
    
    Parameters:
        raw_nutrition: Dict of extracted nutrition facts (e.g. from OCR or direct input).
        category: "food" or "personal_care". If not "food", returns None.
        product_text: Contextual packaging text for beverage/culinary fat/water detection.
        ingredient_text: Ingredient list text used for hydrogenated oil checks.
        serving_size_grams: Optional serving size in grams if converting per-serving data.
        knowledge_base: Optional KnowledgeBase instance.
        
    Returns:
        Structured dictionary matching NutritionScoreResult schema, or None if category is personal_care.
    """
    category_clean = (category or "food").strip().lower()
    if category_clean == "personal_care":
        return None

    if raw_nutrition is None:
        raw_nutrition = {}

    # 1. Normalization & Unit Standardization
    normalized, metadata = normalize_nutrition_data(
        raw_nutrition,
        serving_size_grams=serving_size_grams,
        product_text=product_text,
    )

    is_beverage = metadata.get("is_beverage", False)
    is_culinary_fat = metadata.get("is_culinary_fat", False)
    is_pure_water = metadata.get("is_pure_water", False)
    warnings = list(metadata.get("warnings", []))

    # 2. Water Override Guardrail
    # Pure water has 100.0 health score
    if is_pure_water and _verify_water_profile(normalized):
        result = NutritionScoreResult(
            nutrition_score=WATER_SCORE,
            status="water_override",
            nutrition_completeness=1.0,
            components={
                "negative_risk": 0.0,
                "positive_nutrition": 100.0,
                "micronutrient_contribution": 0.0,
            },
            risk_details={"P_risk": 0.0},
            energy_penalty=0.0,
            guardrails={
                "sugar_guardrail": False,
                "catastrophic_risk": False,
                "catastrophic_ceiling_applied": False,
                "water_override": True,
            },
            nutrients_evaluated=[],
            nutrients_missing=[],
            warnings=["Pure drinking water detected; automatic score 100.0 applied."],
        )
        return result.to_dict()

    # 3. Data Completeness & Minimum Viability Check
    completeness_ratio, core_detected, core_missing = _evaluate_completeness(normalized)
    
    if len(core_detected) < MIN_CORE_NUTRIENTS:
        result = NutritionScoreResult(
            nutrition_score=None,
            status="Insufficient Nutrition Data",
            nutrition_completeness=completeness_ratio,
            components={
                "negative_risk": 0.0,
                "positive_nutrition": 0.0,
                "micronutrient_contribution": 0.0,
            },
            risk_details={},
            energy_penalty=0.0,
            guardrails={
                "sugar_guardrail": False,
                "catastrophic_risk": False,
                "catastrophic_ceiling_applied": False,
                "water_override": False,
            },
            nutrients_evaluated=[],
            nutrients_missing=core_missing,
            warnings=["Fewer than 3 core nutrients detected. Minimum required for scoring is 3."],
        )
        return result.to_dict()

    # 4. Evaluate Primary Risk Nutrients (Logistic Penalty Function)
    evaluated_nutrients: List[Dict[str, Any]] = []
    missing_nutrients: List[str] = []
    
    risk_penalties: Dict[str, float] = {}
    catastrophic_flags: Dict[str, bool] = {}

    # 4.1 Saturated Fat
    t_sat_fat = THRESHOLD_SATURATED_FAT_CULINARY if is_culinary_fat else THRESHOLD_SATURATED_FAT_FOOD
    sat_item = normalized.get("saturated_fat")
    if sat_item and not sat_item.is_missing:
        val = sat_item.amount_per_100g
        pen = _calc_logistic_penalty(val, t_sat_fat)
        risk_penalties["saturated_fat"] = pen
        catastrophic_flags["saturated_fat"] = (val >= CATASTROPHIC_MULTIPLIER * t_sat_fat)
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Saturated Fat",
            amount_per_100g=val,
            unit="g",
            threshold=t_sat_fat,
            penalty=round(pen, 2),
            direction="Lower is Better",
            status="Safe" if pen < 50.0 else ("Catastrophic" if catastrophic_flags["saturated_fat"] else "Excessive"),
            is_explicit_zero=sat_item.is_explicit_zero,
        ).to_dict())
    else:
        missing_nutrients.append("saturated_fat")

    # 4.2 Added Sugar (with fallback to Total Sugar)
    t_sugar = THRESHOLD_ADDED_SUGAR_BEVERAGE if is_beverage else THRESHOLD_ADDED_SUGAR_FOOD
    t_sugar_fallback = THRESHOLD_TOTAL_SUGAR_FALLBACK_BEVERAGE if is_beverage else THRESHOLD_TOTAL_SUGAR_FALLBACK_FOOD
    
    added_sugar_item = normalized.get("added_sugars")
    sugar_eval_val = None
    sugar_eval_thresh = t_sugar
    is_sugar_fallback = False
    
    if added_sugar_item and not added_sugar_item.is_missing:
        sugar_eval_val = added_sugar_item.amount_per_100g
        is_zero = added_sugar_item.is_explicit_zero
    else:
        # Fallback to Total Sugars
        total_sugar_item = normalized.get("total_sugars")
        if total_sugar_item and not total_sugar_item.is_missing:
            sugar_eval_val = total_sugar_item.amount_per_100g
            sugar_eval_thresh = t_sugar_fallback
            is_sugar_fallback = True
            is_zero = total_sugar_item.is_explicit_zero
            warnings.append("Added Sugars not reported; fell back to Total Sugars with adjusted benchmark.")
        else:
            is_zero = False

    if sugar_eval_val is not None:
        pen = _calc_logistic_penalty(sugar_eval_val, sugar_eval_thresh)
        risk_penalties["added_sugars"] = pen
        catastrophic_flags["added_sugars"] = (sugar_eval_val >= CATASTROPHIC_MULTIPLIER * sugar_eval_thresh)
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Added Sugars" if not is_sugar_fallback else "Total Sugars (Fallback)",
            amount_per_100g=sugar_eval_val,
            unit="g",
            threshold=sugar_eval_thresh,
            penalty=round(pen, 2),
            direction="Lower is Better",
            status="Safe" if pen < 50.0 else ("Catastrophic" if catastrophic_flags["added_sugars"] else "Excessive"),
            is_explicit_zero=is_zero,
        ).to_dict())
        if is_sugar_fallback:
            missing_nutrients.append("added_sugars")
    else:
        missing_nutrients.append("added_sugars")
        missing_nutrients.append("total_sugars")

    # Record Total Sugars if reported and not already used as fallback
    total_sugar_item = normalized.get("total_sugars")
    if total_sugar_item and not total_sugar_item.is_missing and not is_sugar_fallback:
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Total Sugars",
            amount_per_100g=total_sugar_item.amount_per_100g,
            unit="g",
            direction="Contextual",
            status="Reported",
            is_explicit_zero=total_sugar_item.is_explicit_zero,
        ).to_dict())
    elif not is_sugar_fallback and (not total_sugar_item or total_sugar_item.is_missing):
        missing_nutrients.append("total_sugars")

    # 4.3 Sodium
    sod_item = normalized.get("sodium")
    if sod_item and not sod_item.is_missing:
        val = sod_item.amount_per_100g
        pen = _calc_logistic_penalty(val, THRESHOLD_SODIUM)
        risk_penalties["sodium"] = pen
        catastrophic_flags["sodium"] = (val >= CATASTROPHIC_MULTIPLIER * THRESHOLD_SODIUM)
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Sodium",
            amount_per_100g=val,
            unit="mg",
            threshold=THRESHOLD_SODIUM,
            penalty=round(pen, 2),
            direction="Lower is Better",
            status="Safe" if pen < 50.0 else ("Catastrophic" if catastrophic_flags["sodium"] else "Excessive"),
            is_explicit_zero=sod_item.is_explicit_zero,
        ).to_dict())
    else:
        missing_nutrients.append("sodium")

    # 4.4 Trans Fat
    trans_item = normalized.get("trans_fat")
    if trans_item and not trans_item.is_missing:
        val = trans_item.amount_per_100g
        pen = _calc_logistic_penalty(val, THRESHOLD_TRANS_FAT)
        risk_penalties["trans_fat"] = pen
        catastrophic_flags["trans_fat"] = (val >= CATASTROPHIC_MULTIPLIER * THRESHOLD_TRANS_FAT)
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Trans Fat",
            amount_per_100g=val,
            unit="g",
            threshold=THRESHOLD_TRANS_FAT,
            penalty=round(pen, 2),
            direction="Lower is Better",
            status="Safe" if pen < 50.0 else ("Catastrophic" if catastrophic_flags["trans_fat"] else "Excessive"),
            is_explicit_zero=trans_item.is_explicit_zero,
        ).to_dict())
    else:
        # Missing Trans Fat policy:
        # Check ingredient text for partially hydrogenated oil evidence
        has_hydrogenated = _check_hydrogenated_oils(ingredient_text)
        if has_hydrogenated:
            pen = 50.0  # Significant trans fat risk from hydrogenated oils
            warnings.append("Trans fat undisclosed but hydrogenated oils found in ingredients.")
        else:
            pen = MISSING_TRANS_FAT_PENALTY  # Neutral baseline penalty = 10.0
        risk_penalties["trans_fat"] = pen
        catastrophic_flags["trans_fat"] = False
        missing_nutrients.append("trans_fat")
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Trans Fat (Estimated)",
            amount_per_100g=None,
            unit="g",
            threshold=THRESHOLD_TRANS_FAT,
            penalty=round(pen, 2),
            direction="Lower is Better",
            status="Estimated (Hydrogenated)" if has_hydrogenated else "Estimated (Neutral Baseline)",
            is_missing=True,
        ).to_dict())

    # 5. Aggregate Risk via Weighted Normalized Minkowski L2 (RMS)
    p_risk = _calc_minkowski_l2_risk(risk_penalties)

    # 6. Stepwise Energy Penalty
    energy_item = normalized.get("energy")
    if energy_item and not energy_item.is_missing:
        energy_val = energy_item.amount_per_100g
        penalty_energy = _calc_energy_penalty(energy_val)
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Energy / Calories",
            amount_per_100g=energy_val,
            unit="kcal",
            threshold=ENERGY_HIGH_THRESHOLD,
            penalty=penalty_energy,
            direction="Appropriate",
            status="Low Calorie" if penalty_energy == 0 else ("Moderate" if penalty_energy == 15 else "High Calorie"),
            is_explicit_zero=energy_item.is_explicit_zero,
        ).to_dict())
    else:
        penalty_energy = 0.0
        missing_nutrients.append("energy")
        warnings.append("Energy/Calories not reported; energy penalty treated as 0.0.")

    # Negative base score: Sneg = max(0, 100 - Prisk - penalty_energy)
    s_neg = max(0.0, 100.0 - p_risk - penalty_energy)

    # 7. Positive Macronutrient Component (Exponential Saturation)
    # Fiber
    fiber_item = normalized.get("dietary_fibre")
    fiber_val = fiber_item.amount_per_100g if (fiber_item and not fiber_item.is_missing) else 0.0
    f_fiber = _calc_saturation(fiber_val, TAU_FIBER)
    if fiber_item and not fiber_item.is_missing:
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Dietary Fiber",
            amount_per_100g=fiber_val,
            unit="g",
            threshold=TAU_FIBER,
            positive_score=round(f_fiber, 2),
            direction="Higher is Better",
            status="Beneficial",
            is_explicit_zero=fiber_item.is_explicit_zero,
        ).to_dict())
    else:
        missing_nutrients.append("dietary_fibre")

    # Protein
    protein_item = normalized.get("protein")
    protein_val = protein_item.amount_per_100g if (protein_item and not protein_item.is_missing) else 0.0
    f_protein = _calc_saturation(protein_val, TAU_PROTEIN)
    if protein_item and not protein_item.is_missing:
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Protein",
            amount_per_100g=protein_val,
            unit="g",
            threshold=TAU_PROTEIN,
            positive_score=round(f_protein, 2),
            direction="Adequate",
            status="Beneficial",
            is_explicit_zero=protein_item.is_explicit_zero,
        ).to_dict())
    else:
        missing_nutrients.append("protein")

    # Total Fat & Unsaturated Fat: max(0, total_fat - (sat_fat + trans_fat)) or mufa + pufa
    tf_item = normalized.get("total_fat")
    if tf_item and not tf_item.is_missing:
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Total Fat",
            amount_per_100g=tf_item.amount_per_100g,
            unit="g",
            direction="Contextual",
            status="Reported",
            is_explicit_zero=tf_item.is_explicit_zero,
        ).to_dict())
    else:
        missing_nutrients.append("total_fat")

    unsat_val, unsat_source = _calc_unsaturated_fat(normalized)
    f_unsat = _calc_saturation(unsat_val, TAU_UNSATURATED_FAT)
    if unsat_val > 0.0 or (tf_item and not tf_item.is_missing):
        evaluated_nutrients.append(EvaluatedNutrient(
            nutrient="Unsaturated Fat",
            amount_per_100g=round(unsat_val, 2),
            unit="g",
            threshold=TAU_UNSATURATED_FAT,
            positive_score=round(f_unsat, 2),
            direction="Higher is Better",
            status="Beneficial" if unsat_val > 0 else "None",
            is_explicit_zero=(unsat_val == 0.0 and bool(tf_item and tf_item.is_explicit_zero)),
        ).to_dict())
    else:
        missing_nutrients.append("unsaturated_fat")

    s_pos = (
        WEIGHT_FIBER * f_fiber
        + WEIGHT_PROTEIN * f_protein
        + WEIGHT_UNSATURATED_FAT * f_unsat
    )
    s_pos = max(0.0, min(100.0, s_pos))

    # 8. Micronutrient Component & Anti-Fortification Discount
    s_micro = _calc_micronutrient_adequacy(normalized)
    gamma = max(0.0, 1.0 - (p_risk / GAMMA_PRISK_DIVISOR))

    # Record any reported micronutrients in evaluated_nutrients
    qualifying_micros = [
        ("calcium", "Calcium", "mg"),
        ("iron", "Iron", "mg"),
        ("potassium", "Potassium", "mg"),
        ("zinc", "Zinc", "mg"),
        ("magnesium", "Magnesium", "mg"),
        ("vitamin_a", "Vitamin A", "mcg"),
        ("vitamin_c", "Vitamin C", "mg"),
        ("vitamin_d", "Vitamin D", "mcg"),
    ]
    for m_key, m_name, m_unit in qualifying_micros:
        m_item = normalized.get(m_key)
        if m_item and not m_item.is_missing:
            evaluated_nutrients.append(EvaluatedNutrient(
                nutrient=m_name,
                amount_per_100g=m_item.amount_per_100g,
                unit=m_item.unit or m_unit,
                direction="Higher is Better",
                status="Adequate" if (m_item.amount_per_100g or 0.0) > 0 else "None",
                is_explicit_zero=m_item.is_explicit_zero,
            ).to_dict())

    # 9. Guardrails
    
    # 9.1 Sugar Guardrail: Added Sugar >= 2.5 * T_sugar => Spos = 0, Smicro = 0
    sugar_guardrail_triggered = False
    if sugar_eval_val is not None:
        trigger_threshold = SUGAR_GUARDRAIL_MULTIPLIER * sugar_eval_thresh
        if sugar_eval_val >= trigger_threshold:
            sugar_guardrail_triggered = True
            s_pos = 0.0
            s_micro = 0.0
            warnings.append(
                f"Sugar Guardrail Triggered: Sugar ({sugar_eval_val}g) >= {trigger_threshold}g. "
                "Positive nutrition and micronutrient contributions nullified."
            )

    # 10. Top-Level Raw Score Synthesis
    score_raw = (
        WEIGHT_SNEG * s_neg
        + WEIGHT_SPOS * s_pos
        + WEIGHT_SMICRO * (gamma * s_micro)
    )

    # 11. Catastrophic Risk Guardrail: Any primary risk >= 2.5 * T_i => Final Score <= 35.0
    catastrophic_triggered = any(catastrophic_flags.values())
    catastrophic_ceiling_applied = False
    
    if catastrophic_triggered:
        final_score = min(score_raw, CATASTROPHIC_CEILING)
        catastrophic_ceiling_applied = (score_raw > CATASTROPHIC_CEILING)
        warnings.append(
            f"Catastrophic Risk Guardrail Triggered: Primary risk nutrient exceeded {CATASTROPHIC_MULTIPLIER}x threshold. "
            f"Score clamped to ceiling {CATASTROPHIC_CEILING}."
        )
    else:
        final_score = score_raw

    # Clamp and round final score
    final_score = round(max(SCORE_MIN, min(SCORE_MAX, final_score)), 1)

    result = NutritionScoreResult(
        nutrition_score=final_score,
        status="scored",
        nutrition_completeness=completeness_ratio,
        components={
            "negative_risk": round(s_neg, 2),
            "positive_nutrition": round(s_pos, 2),
            "micronutrient_contribution": round(gamma * s_micro, 2),
        },
        risk_details={
            "P_risk": round(p_risk, 2),
            "individual_penalties": {k: round(v, 2) for k, v in risk_penalties.items()},
            "anti_fortification_gamma": round(gamma, 4),
        },
        energy_penalty=round(penalty_energy, 1),
        guardrails={
            "sugar_guardrail": sugar_guardrail_triggered,
            "catastrophic_risk": catastrophic_triggered,
            "catastrophic_ceiling_applied": catastrophic_ceiling_applied,
            "water_override": False,
        },
        nutrients_evaluated=evaluated_nutrients,
        nutrients_missing=missing_nutrients,
        warnings=warnings,
    )
    return result.to_dict()


def _calc_logistic_penalty(x: float, threshold: float) -> float:
    """
    Computes exact locked logistic penalty:
    p(x) = 100 / (1 + exp(-3.0 * ((x - T) / T)))
    """
    if threshold <= 0:
        return 100.0
    rel_dev = (x - threshold) / threshold
    arg = -LOGISTIC_K * rel_dev
    # Clamp argument to avoid math.exp overflow/underflow
    arg_clamped = max(EXP_CLAMP_MIN, min(EXP_CLAMP_MAX, arg))
    return 100.0 / (1.0 + math.exp(arg_clamped))


def _calc_minkowski_l2_risk(penalties: Dict[str, float]) -> float:
    """
    Weighted normalized Minkowski L2 / RMS aggregation:
    P_risk = sqrt( sum(w_i * p_i^2) / sum(w_i) )
    """
    if not penalties:
        return 0.0
    
    sum_w_p2 = 0.0
    sum_w = 0.0
    
    for k, pen in penalties.items():
        w = PRIMARY_RISK_WEIGHTS.get(k, 1.0)
        sum_w_p2 += w * (pen ** 2)
        sum_w += w
        
    if sum_w <= 0:
        return 0.0
        
    return math.sqrt(sum_w_p2 / sum_w)


def _calc_energy_penalty(energy_kcal: float) -> float:
    """
    Stepwise energy penalty:
    < 250 kcal -> 0
    250 - 400 kcal -> 15
    > 400 kcal -> 30
    """
    if energy_kcal < ENERGY_LOW_THRESHOLD:
        return ENERGY_PENALTY_LOW
    elif energy_kcal <= ENERGY_HIGH_THRESHOLD:
        return ENERGY_PENALTY_MEDIUM
    else:
        return ENERGY_PENALTY_HIGH


def _calc_saturation(x: float, tau: float) -> float:
    """
    Locked exponential saturation function:
    f(x, tau) = 100 * (1 - exp(-x / tau))
    """
    if x <= 0 or tau <= 0:
        return 0.0
    arg = -x / tau
    arg_clamped = max(EXP_CLAMP_MIN, min(0.0, arg))
    return 100.0 * (1.0 - math.exp(arg_clamped))


def _calc_unsaturated_fat(
    normalized: Dict[str, NormalizedNutrient]
) -> Tuple[float, str]:
    """
    Derives unsaturated fat:
    1. If MUFA and PUFA are available: unsat = mufa + pufa
    2. Else if Total Fat is available: unsat = max(0, total_fat - (sat_fat + trans_fat))
    3. Else: 0.0
    """
    mufa = normalized.get("monounsaturated_fat")
    pufa = normalized.get("polyunsaturated_fat")
    
    if mufa and pufa and not mufa.is_missing and not pufa.is_missing:
        return (mufa.amount_per_100g or 0.0) + (pufa.amount_per_100g or 0.0), "mufa_pufa_sum"
        
    total_fat = normalized.get("total_fat")
    if total_fat and not total_fat.is_missing:
        tf_val = total_fat.amount_per_100g or 0.0
        sat_val = 0.0
        sat_item = normalized.get("saturated_fat")
        if sat_item and not sat_item.is_missing and sat_item.amount_per_100g:
            sat_val = sat_item.amount_per_100g
            
        trans_val = 0.0
        trans_item = normalized.get("trans_fat")
        if trans_item and not trans_item.is_missing and trans_item.amount_per_100g:
            trans_val = trans_item.amount_per_100g
            
        unsat = max(0.0, tf_val - (sat_val + trans_val))
        return unsat, "derived_from_total_fat"
        
    return 0.0, "missing"


def _calc_micronutrient_adequacy(
    normalized: Dict[str, NormalizedNutrient]
) -> float:
    """
    Computes micronutrient score Smicro in [0, 100] based on detected essential micronutrients.
    Each qualifying mineral or vitamin present adds adequacy credit.
    """
    qualifying_micros = ["calcium", "iron", "potassium", "zinc", "magnesium", "vitamin_a", "vitamin_c", "vitamin_d"]
    present_count = 0
    
    for m in qualifying_micros:
        item = normalized.get(m)
        if item and not item.is_missing and (item.amount_per_100g or 0.0) > 0:
            present_count += 1
            
    if present_count == 0:
        return 0.0
        
    # Cap at 100.0 (e.g. 4+ significant micronutrients = 100.0)
    return min(100.0, present_count * 25.0)


def _evaluate_completeness(
    normalized: Dict[str, NormalizedNutrient]
) -> Tuple[float, List[str], List[str]]:
    """
    Evaluates presence of 5 core nutrient groups:
    - Energy
    - Protein
    - Carbohydrate / Sugar
    - Fat
    - Sodium
    Returns: (completeness_ratio, detected_core_list, missing_core_list)
    """
    detected = []
    missing = []

    # 1. Energy
    if normalized.get("energy") and not normalized["energy"].is_missing:
        detected.append("energy")
    else:
        missing.append("energy")

    # 2. Protein
    if normalized.get("protein") and not normalized["protein"].is_missing:
        detected.append("protein")
    else:
        missing.append("protein")

    # 3. Carbohydrate / Sugar
    has_carb = (
        (normalized.get("total_carbohydrate") and not normalized["total_carbohydrate"].is_missing)
        or (normalized.get("total_sugars") and not normalized["total_sugars"].is_missing)
        or (normalized.get("added_sugars") and not normalized["added_sugars"].is_missing)
    )
    if has_carb:
        detected.append("carbohydrate")
    else:
        missing.append("carbohydrate")

    # 4. Fat
    has_fat = (
        (normalized.get("total_fat") and not normalized["total_fat"].is_missing)
        or (normalized.get("saturated_fat") and not normalized["saturated_fat"].is_missing)
    )
    if has_fat:
        detected.append("fat")
    else:
        missing.append("fat")

    # 5. Sodium
    if normalized.get("sodium") and not normalized["sodium"].is_missing:
        detected.append("sodium")
    else:
        missing.append("sodium")

    ratio = round(len(detected) / 5.0, 2)
    return ratio, detected, missing


def _check_hydrogenated_oils(ingredient_text: str) -> bool:
    """Checks ingredient text for partially hydrogenated oil keywords."""
    if not ingredient_text:
        return False
    text_lower = ingredient_text.lower()
    return any(kw in text_lower for kw in HYDROGENATED_OIL_KEYWORDS)


def _verify_water_profile(normalized: Dict[str, NormalizedNutrient]) -> bool:
    """Verifies that nutrition values do not contradict pure drinking water."""
    for key in ("energy", "total_sugars", "added_sugars", "total_fat", "saturated_fat"):
        item = normalized.get(key)
        if item and not item.is_missing and item.amount_per_100g and item.amount_per_100g > 0.0:
            return False
    return True
