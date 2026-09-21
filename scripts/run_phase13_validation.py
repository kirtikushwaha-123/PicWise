"""
scripts/run_phase13_validation.py

Phase 13: Broad Real-World Validation Harness.
Executes the complete validation pipeline across all 36 images in tests/fixtures/validation_set/
without modifying any production logic or ML/OCR pipelines.

Produces:
1. phase13_real_world_validation_results.csv
2. phase13_real_world_validation_summary.json
"""

import csv
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.food_analysis_service import (
    analyze_food,
    InvalidCategoryError as FoodCategoryValidationError,
)
from backend.services.personal_care_analysis_service import (
    analyze_personal_care,
    CategoryValidationError as PCCategoryValidationError,
)
from backend.services.food_status_service import (
    STATUS_GREEN as FOOD_GREEN,
    STATUS_YELLOW as FOOD_YELLOW,
    STATUS_ORANGE as FOOD_ORANGE,
    STATUS_RED as FOOD_RED,
    STATUS_UNAVAILABLE as FOOD_UNAVAILABLE,
)
from backend.services.personal_care_status_service import (
    STATUS_GREEN as PC_GREEN,
    STATUS_YELLOW as PC_YELLOW,
    STATUS_ORANGE as PC_ORANGE,
    STATUS_RED as PC_RED,
    STATUS_UNAVAILABLE as PC_UNAVAILABLE,
)


def normalize_str(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum() or c.isspace()).strip()


def classify_ocr_outcome(ocr_data: dict | None, visible_ingredients: list[str]) -> str:
    """Classifies OCR outcome into PASS, PARTIAL, or FAIL."""
    if not ocr_data:
        return "FAIL" if visible_ingredients else "PASS"

    raw_text = ocr_data.get("full_text", "") or ""
    if not raw_text.strip():
        return "FAIL" if visible_ingredients else "PASS"

    if not visible_ingredients:
        # Blank packaging
        return "PASS"

    # Check how many visible ingredients appear in the raw text
    norm_text = normalize_str(raw_text)
    matched = 0
    for ing in visible_ingredients:
        norm_ing = normalize_str(ing)
        if norm_ing in norm_text or any(part in norm_text for part in norm_ing.split() if len(part) > 3):
            matched += 1

    ratio = matched / len(visible_ingredients)
    if ratio >= 0.70:
        return "PASS"
    elif ratio >= 0.25:
        return "PARTIAL"
    else:
        return "FAIL"


def evaluate_food_nutrition(extracted_nutrition: dict | None, gt_nutrition: dict | None) -> tuple[str, dict[str, str]]:
    """Evaluates nutrition extraction field-by-field against ground truth."""
    if gt_nutrition is None:
        # Label has no nutrition panel
        if extracted_nutrition and extracted_nutrition.get("panel_detected", False):
            return "INCORRECT", {"panel": "Hallucinated panel"}
        return "NOT_PRESENT", {}

    if not extracted_nutrition or not extracted_nutrition.get("panel_detected", False):
        return "MISSED", {k: "MISSED" for k in gt_nutrition.keys()}

    extracted_nutrients = extracted_nutrition.get("nutrients", {})
    field_eval = {}
    correct_count = 0
    total_fields = len(gt_nutrition)

    for field_name, gt_val_str in gt_nutrition.items():
        # Match field in extracted nutrients
        norm_field = field_name.lower().replace(" ", "_")
        matched_val = None
        for k, v in extracted_nutrients.items():
            if norm_field in k.lower() or k.lower() in norm_field:
                matched_val = v
                break

        if matched_val is not None:
            # Check if numeric value matches roughly
            ext_num = matched_val.get("value") if isinstance(matched_val, dict) else matched_val
            # Extract digits from gt_val_str
            gt_num_str = "".join(c for c in gt_val_str if c.isdigit() or c == ".")
            try:
                if gt_num_str and ext_num is not None and abs(float(gt_num_str) - float(ext_num)) < 1.0:
                    field_eval[field_name] = "CORRECT"
                    correct_count += 1
                else:
                    field_eval[field_name] = "PARTIAL"
            except (ValueError, TypeError):
                field_eval[field_name] = "PARTIAL"
        else:
            field_eval[field_name] = "MISSED"

    if correct_count >= total_fields * 0.75:
        overall = "CORRECT"
    elif correct_count > 0:
        overall = "PARTIAL"
    else:
        overall = "MISSED"

    return overall, field_eval


def evaluate_ingredients(extracted_ingredients: list[dict], visible_ingredients: list[str]) -> tuple[int, int, int, int, int, int]:
    """
    Classifies ingredients:
    recognized, unrecognized, misread, fused, split, missed
    """
    if not visible_ingredients:
        return (0, 0, 0, 0, 0, 0)

    recognized_count = 0
    unrecognized_count = 0
    misread_count = 0
    fused_count = 0
    split_count = 0
    missed_count = 0

    extracted_names = []
    for ing in extracted_ingredients:
        if isinstance(ing, dict):
            name = ing.get("ingredient") or ing.get("name") or ing.get("matched_name") or ""
            extracted_names.append(name.lower())
            if ing.get("risk_class") or ing.get("matched_name") or ing.get("status"):
                recognized_count += 1
            else:
                unrecognized_count += 1
        else:
            extracted_names.append(str(ing).lower())

    for v_ing in visible_ingredients:
        v_norm = normalize_str(v_ing)
        # Check if v_norm is in extracted names
        found = False
        for ext_name in extracted_names:
            ext_norm = normalize_str(ext_name)
            if v_norm == ext_norm:
                found = True
                break
            elif v_norm in ext_norm and len(ext_norm) > len(v_norm) + 5:
                fused_count += 1
                found = True
                break
            elif ext_norm in v_norm and len(v_norm) > len(ext_norm) + 5:
                split_count += 1
                found = True
                break

        if not found:
            missed_count += 1

    return recognized_count, unrecognized_count, misread_count, fused_count, split_count, missed_count


def run_validation():
    gt_path = PROJECT_ROOT / "tests/fixtures/validation_set/ground_truth.json"
    if not gt_path.exists():
        print(f"Error: {gt_path} not found. Run scripts/generate_validation_images.py first.")
        sys.exit(1)

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    print(f"Loaded {len(ground_truth)} ground truth records.")
    print("Starting Phase 13 Broad Real-World Validation Execution...")

    results = []
    csv_rows = []

    tracemalloc.start()
    total_start_time = time.time()

    for idx, item in enumerate(ground_truth, 1):
        img_id = item["id"]
        cat = item["category"]
        cond = item["condition"]
        ptype = item["product_type"]
        pname = item["product_name"]
        file_path = PROJECT_ROOT / item["file_path"]

        print(f"\n[{idx}/{len(ground_truth)}] Processing {img_id} ({cat}, {cond})...")

        if not file_path.exists():
            print(f"  [ERROR] File {file_path} not found!")
            continue

        with open(file_path, "rb") as f:
            img_bytes = f.read()

        t0 = time.time()
        error_msg = None
        analysis_result = None

        try:
            if cat == "food":
                analysis_result = analyze_food(img_bytes, category="food")
            else:
                analysis_result = analyze_personal_care(img_bytes, category="personal_care")
        except Exception as exc:
            error_msg = str(exc)
            print(f"  [EXCEPTION] {exc}")

        runtime = round(time.time() - t0, 3)
        print(f"  Completed in {runtime:.3f}s")

        # Evaluate outcome
        res_dict = analysis_result.to_dict() if analysis_result else {}
        ocr_data = res_dict.get("ocr")
        ocr_status = classify_ocr_outcome(ocr_data, item.get("visible_ingredients", []))

        # Ingredients evaluation
        if cat == "food":
            ext_ings = res_dict.get("food_safety", {}).get("ingredients", [])
        else:
            ext_ings = res_dict.get("ingredients", []) or res_dict.get("personal_care", {}).get("ingredients", [])

        rec, unrec, misread, fused, split, missed = evaluate_ingredients(ext_ings, item.get("visible_ingredients", []))

        # Nutrition evaluation (Food only)
        nut_status = "N/A"
        nut_field_eval = {}
        if cat == "food":
            ext_nutrition = res_dict.get("nutrition")
            nut_status, nut_field_eval = evaluate_food_nutrition(ext_nutrition, item.get("nutrition_values"))

        # Allergy evaluation
        allergy_status = "N/A"
        if cat == "food":
            allergy_data = res_dict.get("allergy", {})
            allergy_status = allergy_data.get("status", "unknown")

        # Warnings count & content
        warnings = res_dict.get("warnings", [])
        warning_count = len(warnings)

        # Final status & presentation
        pres = res_dict.get("presentation", {})
        if cat == "food":
            final_status = f"Safety:{pres.get('food_safety', {}).get('status')}|Nut:{pres.get('nutrition', {}).get('status')}|Allergy:{pres.get('allergy', {}).get('status')}"
        else:
            final_status = f"Safety:{pres.get('personal_care_safety', {}).get('status')}|Allergy:{pres.get('allergy', {}).get('status')}|Irritant:{pres.get('irritation', {}).get('status')}"

        # Semantics check
        semantics_ok = True
        notes = []
        if len(item.get("visible_ingredients", [])) == 0:
            # Blank label -> Must be unavailable, NEVER safe
            if cat == "food":
                if pres.get("food_safety", {}).get("status") in [FOOD_GREEN, FOOD_YELLOW, FOOD_ORANGE, FOOD_RED]:
                    semantics_ok = False
                    notes.append("DEFECT: Blank food label assigned active safety status instead of unavailable")
            else:
                for dim in ["personal_care_safety", "allergy", "irritation"]:
                    if pres.get(dim, {}).get("status") != PC_UNAVAILABLE:
                        semantics_ok = False
                        notes.append(f"DEFECT: Blank PC label assigned {pres.get(dim, {}).get('status')} for {dim} instead of unavailable")

        if cat == "food" and item.get("nutrition_values") is None:
            # Missing nutrition -> Must be unavailable
            if pres.get("nutrition", {}).get("status") != FOOD_UNAVAILABLE:
                semantics_ok = False
                notes.append("DEFECT: Missing nutrition facts assigned score instead of unavailable")

        record = {
            "image_id": img_id,
            "category": cat,
            "product_type": ptype,
            "product_name": pname,
            "condition": cond,
            "runtime_seconds": runtime,
            "ocr_status": ocr_status,
            "extracted_count": len(ext_ings),
            "recognized_count": rec,
            "unrecognized_count": unrec,
            "missed_count": missed,
            "fused_count": fused,
            "split_count": split,
            "nutrition_status": nut_status,
            "allergy_status": allergy_status,
            "warning_count": warning_count,
            "warnings": warnings,
            "final_status": final_status,
            "semantics_ok": semantics_ok,
            "presentation": pres,
            "error": error_msg,
            "notes": "; ".join(notes) if notes else "OK",
        }
        results.append(record)

        csv_rows.append({
            "image_id": img_id,
            "category": cat,
            "condition": cond,
            "ocr_status": ocr_status,
            "extracted_count": len(ext_ings),
            "recognized_count": rec,
            "unrecognized_count": unrec,
            "missed_count": missed,
            "fused_count": fused,
            "nutrition_status": nut_status,
            "warning_count": warning_count,
            "final_status": final_status,
            "runtime_seconds": runtime,
            "notes": record["notes"],
        })

    # Memory check
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_elapsed = round(time.time() - total_start_time, 2)

    # Write CSV
    csv_file = PROJECT_ROOT / "phase13_real_world_validation_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "category", "condition", "ocr_status", "extracted_count",
            "recognized_count", "unrecognized_count", "missed_count", "fused_count",
            "nutrition_status", "warning_count", "final_status", "runtime_seconds", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved results CSV to {csv_file}")

    # Summary calculations
    runtimes = [r["runtime_seconds"] for r in results]
    min_rt = min(runtimes) if runtimes else 0.0
    max_rt = max(runtimes) if runtimes else 0.0
    med_rt = sorted(runtimes)[len(runtimes) // 2] if runtimes else 0.0

    ocr_pass_count = sum(1 for r in results if r["ocr_status"] == "PASS")
    ocr_part_count = sum(1 for r in results if r["ocr_status"] == "PARTIAL")
    ocr_fail_count = sum(1 for r in results if r["ocr_status"] == "FAIL")

    summary = {
        "total_images_tested": len(results),
        "total_elapsed_seconds": total_elapsed,
        "runtime": {
            "min_seconds": min_rt,
            "median_seconds": med_rt,
            "max_seconds": max_rt,
        },
        "memory": {
            "peak_mb": round(peak_mem / (1024 * 1024), 2),
            "final_mb": round(current_mem / (1024 * 1024), 2),
        },
        "ocr_distribution": {
            "PASS": ocr_pass_count,
            "PARTIAL": ocr_part_count,
            "FAIL": ocr_fail_count,
        },
        "failure_semantics_violations": sum(1 for r in results if not r["semantics_ok"]),
        "results": results,
    }

    summary_file = PROJECT_ROOT / "phase13_real_world_validation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary JSON to {summary_file}")
    print(f"\nPhase 13 Broad Real-World Validation completed in {total_elapsed:.1f}s.")
    print(f"Runtime: Min={min_rt:.3f}s | Median={med_rt:.3f}s | Max={max_rt:.3f}s")
    print(f"OCR: PASS={ocr_pass_count} | PARTIAL={ocr_part_count} | FAIL={ocr_fail_count}")
    print(f"Failure Semantics Violations: {summary['failure_semantics_violations']}")


if __name__ == "__main__":
    run_validation()
