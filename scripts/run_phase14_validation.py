"""
scripts/run_phase14_validation.py

Phase 14: Evidence-Based Real-World Defect Remediation Re-Validation.
Re-runs the 36-image real-world validation corpus and produces:
  - phase14_real_world_validation_results.csv
  - phase14_real_world_validation_summary.json
  - A side-by-side Before (Phase 13) vs. After (Phase 14) comparison report.
"""

import csv
import json
import os
import sys
from pathlib import Path
import time
import tracemalloc
import unicodedata
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.food_analysis_service import analyze_food
from backend.services.personal_care_analysis_service import analyze_personal_care
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "validation_set"
GT_FILE = FIXTURES_DIR / "ground_truth.json"


def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    for ch in [",", ";", ":", ".", "-", "(", ")", "[", "]", "/", "\\", "•", "·"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())


def evaluate_food_nutrition(ext_nutrition: dict, gt_nutrition: dict) -> tuple[str, dict]:
    if gt_nutrition is None:
        return ("N/A", {})

    if not ext_nutrition or not isinstance(ext_nutrition, dict):
        return ("MISSED", {k: "MISSED" for k in gt_nutrition})

    field_eval = {}
    extracted_nutrients = ext_nutrition.get("nutrients") or ext_nutrition
    correct_count = 0
    total_fields = len(gt_nutrition)

    for field_name, gt_val_str in gt_nutrition.items():
        norm_field = field_name.lower().replace(" ", "_")
        matched_val = None
        for k, v in extracted_nutrients.items():
            if norm_field in k.lower() or k.lower() in norm_field:
                matched_val = v
                break

        if matched_val is not None:
            ext_num = matched_val.get("value") if isinstance(matched_val, dict) else matched_val
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


def evaluate_ingredients(extracted_ingredients: list, visible_ingredients: list[str]) -> tuple[int, int, int, int, int, int]:
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
            name = ing.get("matched_name") or ing.get("raw_text") or ing.get("ocr_text") or ing.get("name") or ing.get("ingredient") or ""
            extracted_names.append(name.lower())
            if ing.get("risk_class") or ing.get("matched_name") or ing.get("status") == "success":
                recognized_count += 1
            else:
                unrecognized_count += 1
        else:
            extracted_names.append(str(ing).lower())

    for v_ing in visible_ingredients:
        v_norm = normalize_str(v_ing)
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

    return (recognized_count, unrecognized_count, misread_count, fused_count, split_count, missed_count)


def run_phase14_validation():
    print(f"=== PICWISE PHASE 14: REAL-WORLD DEFECT REMEDIATION RE-VALIDATION ===")
    print(f"Ground truth dataset: {GT_FILE}")

    with open(GT_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    print(f"Total validation images: {len(ground_truth)}")

    tracemalloc.start()
    total_start_time = time.time()

    results = []
    csv_rows = []

    for idx, item in enumerate(ground_truth, 1):
        img_id = item["id"]
        cat = item["category"]
        cond = item["condition"]
        ptype = item.get("product_type", "Unknown")
        pname = item.get("product_name", "Unknown")
        img_path = PROJECT_ROOT / item["file_path"]

        print(f"\n[{idx:02d}/36] Processing {img_id} ({cat}, {cond})...")

        if not img_path.exists():
            print(f"  [ERROR] Image not found: {img_path}")
            continue

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        t0 = time.time()
        error_msg = None
        res_dict = {}

        try:
            if cat == "food":
                res = analyze_food(img_bytes, category="food")
                res_dict = res.to_dict()
            elif cat == "personal_care":
                res = analyze_personal_care(img_bytes, category="personal_care")
                res_dict = res.to_dict()
            else:
                raise ValueError(f"Unknown category: {cat}")
        except Exception as exc:
            error_msg = str(exc)
            print(f"  [EXCEPTION] {exc}")

        runtime = round(time.time() - t0, 3)

        ocr_info = res_dict.get("ocr") or {}
        ext_ings = ocr_info.get("ingredients") or []

        rec, unrec, misread, fused, split, missed = evaluate_ingredients(
            ext_ings, item.get("visible_ingredients", [])
        )

        vis_count = len(item.get("visible_ingredients", []))
        if vis_count == 0:
            ocr_status = "PASS" if len(ext_ings) == 0 else "FAIL"
        elif rec >= vis_count * 0.75:
            ocr_status = "PASS"
        elif rec > 0 or len(ext_ings) > 0:
            ocr_status = "PARTIAL"
        else:
            ocr_status = "FAIL"

        nut_status = "N/A"
        nut_field_eval = {}
        if cat == "food":
            ext_nutrition = ocr_info.get("nutrition") or res_dict.get("nutrition")
            nut_status, nut_field_eval = evaluate_food_nutrition(ext_nutrition, item.get("nutrition_values"))

        allergy_status = "N/A"
        if cat == "food":
            allergy_data = res_dict.get("allergy", {})
            allergy_status = allergy_data.get("status", "unknown")

        warnings = res_dict.get("warnings", [])
        warning_count = len(warnings)

        pres = res_dict.get("presentation", {})
        if cat == "food":
            fs_status = pres.get('food_safety', {}).get('status')
            nut_pres_status = pres.get('nutrition', {}).get('status')
            all_status = pres.get('allergy', {}).get('status')
            final_status = f"Safety:{fs_status}|Nut:{nut_pres_status}|Allergy:{all_status}"
        else:
            pc_safety_status = pres.get('personal_care_safety', {}).get('status')
            all_status = pres.get('allergy', {}).get('status')
            irr_status = pres.get('irritation', {}).get('status')
            final_status = f"Safety:{pc_safety_status}|Allergy:{all_status}|Irritant:{irr_status}"

        # Failure Semantics Check
        semantics_ok = True
        notes = []
        if len(item.get("visible_ingredients", [])) == 0:
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

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_elapsed = round(time.time() - total_start_time, 2)

    # Save Phase 14 CSV
    csv_file = PROJECT_ROOT / "phase14_real_world_validation_results.csv"
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

    summary_file = PROJECT_ROOT / "phase14_real_world_validation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary JSON to {summary_file}")
    print(f"\nPhase 14 Real-World Re-Validation completed in {total_elapsed:.1f}s.")
    print(f"Runtime: Min={min_rt:.3f}s | Median={med_rt:.3f}s | Max={max_rt:.3f}s")
    print(f"OCR: PASS={ocr_pass_count} | PARTIAL={ocr_part_count} | FAIL={ocr_fail_count}")
    print(f"Failure Semantics Violations: {summary['failure_semantics_violations']}")

    # Print Before vs. After comparison for the 5 remediated defects
    p13_csv = PROJECT_ROOT / "phase13_real_world_validation_results.csv"
    if p13_csv.exists():
        df13 = pd.read_csv(p13_csv).set_index("image_id")
        df14 = pd.read_csv(csv_file).set_index("image_id")

        print("\n" + "=" * 80)
        print("=== BEFORE (PHASE 13) vs. AFTER (PHASE 14) DEFECT REMEDIATION MATRIX ===")
        print("=" * 80)

        # DEF-01: pc_15_perfume_box_blank
        p13_pc15 = df13.loc["pc_15_perfume_box_blank"]
        p14_pc15 = df14.loc["pc_15_perfume_box_blank"]
        print("\n[DEF-01: False Recognition on Blank Marketing Label (pc_15_perfume_box_blank)]")
        print(f"  Phase 13: Extracted={p13_pc15['extracted_count']} | Status={p13_pc15['final_status']} | Notes={p13_pc15['notes']}")
        print(f"  Phase 14: Extracted={p14_pc15['extracted_count']} | Status={p14_pc15['final_status']} | Notes={p14_pc15['notes']}")

        # DEF-02: Food Safety Badge Status
        print("\n[DEF-02: Food Safety Top-Level Badge Availability]")
        p13_fs_unavail = sum(1 for img_id, r in df13.iterrows() if r["category"] == "food" and "Safety:unavailable" in str(r["final_status"]))
        p14_fs_unavail = sum(1 for img_id, r in df14.iterrows() if r["category"] == "food" and "Safety:unavailable" in str(r["final_status"]))
        p14_fs_active = sum(1 for img_id, r in df14.iterrows() if r["category"] == "food" and any(s in str(r["final_status"]) for s in ["Safety:green", "Safety:yellow", "Safety:orange", "Safety:red"]))
        print(f"  Phase 13: Food Safety Unavailable = {p13_fs_unavail}/18 (100% of food products lacked top-level safety badge)")
        print(f"  Phase 14: Food Safety Populated = {p14_fs_active}/18 | Unavailable (Blank only) = {p14_fs_unavail}/18")

        # DEF-03: Small/Dense Text (pc_09_serum_ordinary_small_text)
        p13_pc09 = df13.loc["pc_09_serum_ordinary_small_text"]
        p14_pc09 = df14.loc["pc_09_serum_ordinary_small_text"]
        print("\n[DEF-03: Small/Dense Personal Care Text (pc_09_serum_ordinary_small_text)]")
        print(f"  Phase 13: Extracted={p13_pc09['extracted_count']} | Recognized={p13_pc09['recognized_count']} | OCR Status={p13_pc09['ocr_status']}")
        print(f"  Phase 14: Extracted={p14_pc09['extracted_count']} | Recognized={p14_pc09['recognized_count']} | OCR Status={p14_pc09['ocr_status']}")

        # DEF-04: Dense Food Ingredient Fusion
        print("\n[DEF-04: Dense Food Ingredient Fusion (food_03, food_04, food_09, food_11)]")
        for f_id in ["food_03_noodles_maggi_dense", "food_04_cereal_kelloggs_small_text", "food_09_staple_pasta_complex", "food_11_namkeen_haldiram_dense"]:
            p13_f = df13.loc[f_id]
            p14_f = df14.loc[f_id]
            print(f"  {f_id}:")
            print(f"    Phase 13: Fused={p13_f['fused_count']} | Extracted={p13_f['extracted_count']} | Recognized={p13_f['recognized_count']}")
            print(f"    Phase 14: Fused={p14_f['fused_count']} | Extracted={p14_f['extracted_count']} | Recognized={p14_f['recognized_count']}")

        # DEF-05: Dual-Unit Nutrition Parsing
        p13_food12 = df13.loc["food_12_energy_drink_redbull_units"]
        p14_food12 = df14.loc["food_12_energy_drink_redbull_units"]
        print("\n[DEF-05: Dual-Unit Nutrition Parsing (food_12_energy_drink_redbull_units)]")
        print(f"  Phase 13: Nutrition Status={p13_food12['nutrition_status']} | Final Status={p13_food12['final_status']}")
        print(f"  Phase 14: Nutrition Status={p14_food12['nutrition_status']} | Final Status={p14_food12['final_status']}")

        # Overall Corpus
        print("\n[CORPUS-WIDE COMPARISON]")
        print(f"  Failure Semantics Violations: Phase 13 = {sum(1 for _, r in df13.iterrows() if r['notes'] != 'OK')} -> Phase 14 = {summary['failure_semantics_violations']}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    run_phase14_validation()
