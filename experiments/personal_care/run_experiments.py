"""
experiments/personal_care/run_experiments.py

Orchestrates Phase 10A Personal Care Model Validation & Optimization:
- Audit & canonical grouping verification
- Leakage-free StratifiedGroupKFold validation
- Baseline Logistic Regression evaluation & OOF generation
- Reproducibility test
- Controlled hyperparameter optimization (C and class_weight)
- Dangerous confusion and minority recall tracking
- Error analysis and probability confidence profiling
- Artifact persistence
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.preprocessing.personal_care_dataset import (
    DEFAULT_PERSONAL_CARE_DATA_PATH,
    VALID_ALLERGY_RISKS,
    VALID_IRRITATION_RISKS,
    VALID_SAFETY_LEVELS,
    build_feature_pipeline,
    load_personal_care_dataset,
)

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "personal_care"
REPRODUCIBILITY_DIR = EXPERIMENT_DIR / "reproducibility_audit"
BASELINE_DIR = EXPERIMENT_DIR / "baseline"
OPTIMIZATION_DIR = EXPERIMENT_DIR / "optimization"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
ERROR_DIR = EXPERIMENT_DIR / "error_analysis"


def calculate_dangerous_confusions(y_true: List[str], y_pred: List[str], target: str) -> Dict[str, Any]:
    """Computes safety-critical dangerous confusion counts."""
    counts = {}
    details = []

    if target == "Safety_Level":
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            if t == "High Risk" and p in ["Safe", "Very Safe"]:
                details.append({"index": idx, "true": t, "pred": p, "type": "High Risk -> Safe/Very Safe"})
            elif t == "Moderate Risk" and p == "Very Safe":
                details.append({"index": idx, "true": t, "pred": p, "type": "Moderate Risk -> Very Safe"})

        counts["high_risk_as_safe_or_very_safe"] = sum(1 for d in details if d["type"] == "High Risk -> Safe/Very Safe")
        counts["moderate_risk_as_very_safe"] = sum(1 for d in details if d["type"] == "Moderate Risk -> Very Safe")
        counts["total_dangerous_confusions"] = len(details)

    elif target == "Allergy_Risk":
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            if t == "High" and p in ["No Risk", "Low"]:
                details.append({"index": idx, "true": t, "pred": p, "type": "High Allergy -> No Risk/Low"})
            elif t == "Medium" and p == "No Risk":
                details.append({"index": idx, "true": t, "pred": p, "type": "Medium Allergy -> No Risk"})

        counts["high_allergy_as_no_risk_or_low"] = sum(1 for d in details if d["type"] == "High Allergy -> No Risk/Low")
        counts["medium_allergy_as_no_risk"] = sum(1 for d in details if d["type"] == "Medium Allergy -> No Risk")
        counts["total_dangerous_confusions"] = len(details)

    elif target == "Irritation_Risk":
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            if t == "High" and p in ["No Risk", "Low"]:
                details.append({"index": idx, "true": t, "pred": p, "type": "High Irritation -> No Risk/Low"})
            elif t == "Medium" and p == "No Risk":
                details.append({"index": idx, "true": t, "pred": p, "type": "Medium Irritation -> No Risk"})

        counts["high_irritation_as_no_risk_or_low"] = sum(1 for d in details if d["type"] == "High Irritation -> No Risk/Low")
        counts["medium_irritation_as_no_risk"] = sum(1 for d in details if d["type"] == "Medium Irritation -> No Risk")
        counts["total_dangerous_confusions"] = len(details)

    counts["details"] = details
    return counts


def evaluate_model_cv(
    df: pd.DataFrame,
    target: str,
    C: float = 1.0,
    class_weight: Any = None,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs 5-fold StratifiedGroupKFold on canonical_group_id, collecting OOF predictions and metrics."""
    y = df[target].values
    groups = df["canonical_group_id"].values
    classes = sorted(list(set(y)))
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.empty(len(df), dtype=object)
    oof_probs = np.zeros((len(df), len(classes)), dtype=float)
    oof_folds = np.zeros(len(df), dtype=int)

    group_leakages = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, y, groups=groups)):
        # Verify zero group leakage
        tr_groups = set(groups[train_idx])
        val_groups_set = set(groups[val_idx])
        overlap = tr_groups.intersection(val_groups_set)
        if overlap:
            group_leakages.append({"fold": fold, "leakage_groups": list(overlap)})

        X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline = Pipeline([
            ("preprocessor", build_feature_pipeline()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ])

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)
        model_classes = list(pipeline.named_steps["classifier"].classes_)

        # Align probability columns to sorted classes
        for col_idx, cls_name in enumerate(model_classes):
            target_col_idx = classes.index(cls_name)
            oof_probs[val_idx, target_col_idx] = probs[:, col_idx]

        oof_preds[val_idx] = preds
        oof_folds[val_idx] = fold

    # Metrics calculation
    macro_f1 = float(f1_score(y, oof_preds, average="macro"))
    weighted_f1 = float(f1_score(y, oof_preds, average="weighted"))
    acc = float(accuracy_score(y, oof_preds))
    bal_acc = float(balanced_accuracy_score(y, oof_preds))

    # Per-class metrics
    clf_report = classification_report(y, oof_preds, output_dict=True, digits=4)
    conf_mat = confusion_matrix(y, oof_preds, labels=classes).tolist()

    # Dangerous confusions
    dangerous = calculate_dangerous_confusions(list(y), list(oof_preds), target)

    # Probability sanity check
    prob_sums = np.sum(oof_probs, axis=1)
    prob_valid = bool(np.allclose(prob_sums, 1.0, atol=1e-5))

    return {
        "target": target,
        "C": C,
        "class_weight": class_weight,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "classification_report": clf_report,
        "confusion_matrix": conf_mat,
        "classes": classes,
        "dangerous_confusions": dangerous,
        "group_leakage_detected": len(group_leakages) > 0,
        "group_leakages": group_leakages,
        "probability_valid": prob_valid,
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "oof_folds": oof_folds,
    }


def run_full_suite():
    print("=" * 80)
    print("PHASE 10A: PERSONAL CARE MODEL VALIDATION & OPTIMIZATION")
    print("=" * 80)

    # 1. Dataset Audit
    print("\n[Step 1] Loading and Auditing Dataset...")
    df = load_personal_care_dataset()
    row_count = len(df)
    col_count = len(df.columns)
    unique_groups = df["canonical_group_id"].nunique()

    print(f"  Shape: {row_count} rows x {col_count} columns")
    print(f"  Canonical groups: {unique_groups} unique groups across {row_count} rows")
    print(f"  Zero nulls confirmed: {df.isnull().sum().sum() == 0}")

    audit_info = {
        "dataset_path": DEFAULT_PERSONAL_CARE_DATA_PATH,
        "row_count": row_count,
        "column_count": col_count,
        "columns": df.columns.tolist(),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_ingredient_names": int(df["Ingredient_Name"].duplicated().sum()),
        "unique_canonical_groups": unique_groups,
        "target_distributions": {
            "Safety_Level": df["Safety_Level"].value_counts().to_dict(),
            "Allergy_Risk": df["Allergy_Risk"].value_counts().to_dict(),
            "Irritation_Risk": df["Irritation_Risk"].value_counts().to_dict(),
        },
        "alternate_names_count": int((df["Packaging Names / Alternate Names"] != "No Alternate Names").sum()),
    }

    with open(REPRODUCIBILITY_DIR / "audit_summary.json", "w") as f:
        json.dump(audit_info, f, indent=2)

    # 2. Baseline Evaluation (PART 7)
    print("\n[Step 2] Evaluating Baseline Configuration (C=1.0, class_weight=None)...")
    targets = ["Safety_Level", "Allergy_Risk", "Irritation_Risk"]
    baseline_results = {}
    oof_df = df[["Ingredient_Name", "canonical_group_id"]].copy()

    for target in targets:
        res = evaluate_model_cv(df, target, C=1.0, class_weight=None, random_state=42)
        baseline_results[target] = res

        oof_df[f"{target}_true"] = df[target]
        oof_df[f"{target}_pred"] = res["oof_preds"]
        oof_df[f"{target}_fold"] = res["oof_folds"]

        for idx, cls_name in enumerate(res["classes"]):
            oof_df[f"{target}_prob_{cls_name}"] = res["oof_probs"][:, idx]

        print(f"  Target: {target:<16} | Macro F1: {res['macro_f1']:.4f} | Weighted F1: {res['weighted_f1']:.4f} | Acc: {res['accuracy']:.4f} | Leakage: {res['group_leakage_detected']}")

    mean_baseline_macro = float(np.mean([baseline_results[t]["macro_f1"] for t in targets]))
    print(f"  --> Baseline Mean Macro F1: {mean_baseline_macro:.4f}")

    # Save baseline artifacts
    baseline_serializable = {
        "mean_macro_f1": mean_baseline_macro,
        "targets": {
            t: {k: v for k, v in baseline_results[t].items() if k not in ["oof_preds", "oof_probs", "oof_folds"]}
            for t in targets
        },
    }
    with open(BASELINE_DIR / "baseline_metrics.json", "w") as f:
        json.dump(baseline_serializable, f, indent=2)
    oof_df.to_csv(BASELINE_DIR / "oof_predictions.csv", index=False)

    # 3. Reproducibility Check (PART 13)
    print("\n[Step 3] Running Reproducibility Test (2nd identical run)...")
    repro_match = True
    for target in targets:
        res_repeat = evaluate_model_cv(df, target, C=1.0, class_weight=None, random_state=42)
        if not np.array_equal(baseline_results[target]["oof_preds"], res_repeat["oof_preds"]):
            repro_match = False
            print(f"  [ERROR] Predictions non-deterministic for {target}!")
        if not np.allclose(baseline_results[target]["oof_probs"], res_repeat["oof_probs"], atol=1e-7):
            repro_match = False
            print(f"  [ERROR] Probabilities non-deterministic for {target}!")
    print(f"  Reproducibility verification status: {'PASSED (Exact Match)' if repro_match else 'FAILED'}")

    # 4. Controlled Optimization Grid (PART 9)
    print("\n[Step 4] Running Controlled Logistic Regression Optimization Grid...")
    c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    class_weights = [None, "balanced"]

    grid_results = []

    for cw in class_weights:
        for c in c_values:
            cfg_name = f"C={c}_cw={str(cw)}"
            target_res = {}
            macro_f1s = []

            for target in targets:
                res = evaluate_model_cv(df, target, C=c, class_weight=cw, random_state=42)
                target_res[target] = res
                macro_f1s.append(res["macro_f1"])

            mean_f1 = float(np.mean(macro_f1s))

            # Minority recall tracking
            safety_high_recall = target_res["Safety_Level"]["classification_report"]["High Risk"]["recall"]
            allergy_high_recall = target_res["Allergy_Risk"]["classification_report"]["High"]["recall"]
            irritation_high_recall = target_res["Irritation_Risk"]["classification_report"]["High"]["recall"]

            # Dangerous confusions
            safety_dang = target_res["Safety_Level"]["dangerous_confusions"]["total_dangerous_confusions"]
            allergy_dang = target_res["Allergy_Risk"]["dangerous_confusions"]["total_dangerous_confusions"]
            irritation_dang = target_res["Irritation_Risk"]["dangerous_confusions"]["total_dangerous_confusions"]

            grid_entry = {
                "configuration": cfg_name,
                "C": c,
                "class_weight": str(cw),
                "safety_macro_f1": target_res["Safety_Level"]["macro_f1"],
                "allergy_macro_f1": target_res["Allergy_Risk"]["macro_f1"],
                "irritation_macro_f1": target_res["Irritation_Risk"]["macro_f1"],
                "mean_macro_f1": mean_f1,
                "safety_high_recall": safety_high_recall,
                "allergy_high_recall": allergy_high_recall,
                "irritation_high_recall": irritation_high_recall,
                "safety_dangerous_confusions": safety_dang,
                "allergy_dangerous_confusions": allergy_dang,
                "irritation_dangerous_confusions": irritation_dang,
                "total_dangerous_confusions": safety_dang + allergy_dang + irritation_dang,
            }
            grid_results.append(grid_entry)

            print(
                f"  {cfg_name:<18} | Mean Macro F1: {mean_f1:.4f} "
                f"(Safety: {grid_entry['safety_macro_f1']:.4f}, Allergy: {grid_entry['allergy_macro_f1']:.4f}, Irritation: {grid_entry['irritation_macro_f1']:.4f}) | "
                f"High Recalls: [{safety_high_recall:.2f}, {allergy_high_recall:.2f}, {irritation_high_recall:.2f}] | "
                f"Dang: {grid_entry['total_dangerous_confusions']}"
            )

    grid_df = pd.DataFrame(grid_results)
    grid_df = grid_df.sort_values(by="mean_macro_f1", ascending=False).reset_index(drop=True)
    grid_df.to_csv(OPTIMIZATION_DIR / "optimization_comparison_table.csv", index=False)

    best_cfg = grid_df.iloc[0].to_dict()
    print("\n--- Best Configuration by Mean Macro F1 ---")
    print(f"  Configuration: {best_cfg['configuration']}")
    print(f"  Mean Macro F1: {best_cfg['mean_macro_f1']:.4f}")
    print(f"  Safety: {best_cfg['safety_macro_f1']:.4f} | Allergy: {best_cfg['allergy_macro_f1']:.4f} | Irritation: {best_cfg['irritation_macro_f1']:.4f}")
    print(f"  Total Dangerous Confusions: {best_cfg['total_dangerous_confusions']}")

    with open(OPTIMIZATION_DIR / "optimization_summary.json", "w") as f:
        json.dump({"best_configuration": best_cfg, "all_experiments": grid_results}, f, indent=2)

    # 5. Error & Confidence Analysis for Best Configuration (PART 15 & 16)
    print("\n[Step 5] Performing Detailed Error & Confidence Analysis on Best Configuration...")
    best_c = float(best_cfg["C"])
    best_cw = None if best_cfg["class_weight"] == "None" else "balanced"

    error_analysis_data = {}
    best_oof_df = df[["Ingredient_Name", "canonical_group_id", "Primary_Function", "Ingredient_Category", "Origin", "Regulatory_Status"]].copy()

    for target in targets:
        best_res = evaluate_model_cv(df, target, C=best_c, class_weight=best_cw, random_state=42)
        best_oof_df[f"{target}_true"] = df[target]
        best_oof_df[f"{target}_pred"] = best_res["oof_preds"]
        best_oof_df[f"{target}_correct"] = (df[target] == best_res["oof_preds"])

        # Max confidence
        best_oof_df[f"{target}_confidence"] = np.max(best_res["oof_probs"], axis=1)

        # Confidence analysis
        correct_mask = best_oof_df[f"{target}_correct"]
        high_conf_correct = int(((best_oof_df[f"{target}_confidence"] >= 0.8) & correct_mask).sum())
        high_conf_error = int(((best_oof_df[f"{target}_confidence"] >= 0.8) & (~correct_mask)).sum())
        low_conf = int((best_oof_df[f"{target}_confidence"] < 0.5).sum())

        error_analysis_data[target] = {
            "dangerous_confusions": best_res["dangerous_confusions"],
            "high_confidence_correct (>=0.8)": high_conf_correct,
            "high_confidence_incorrect (>=0.8)": high_conf_error,
            "low_confidence (<0.5)": low_conf,
            "average_confidence_correct": float(best_oof_df.loc[correct_mask, f"{target}_confidence"].mean()),
            "average_confidence_error": float(best_oof_df.loc[~correct_mask, f"{target}_confidence"].mean()),
        }

    best_oof_df.to_csv(ERROR_DIR / "best_model_oof_detailed.csv", index=False)
    with open(ERROR_DIR / "error_analysis_summary.json", "w") as f:
        json.dump(error_analysis_data, f, indent=2)

    print("  Error analysis and OOF artifacts saved successfully.")
    print("\n" + "=" * 80)
    print("EXPERIMENT RUN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_full_suite()
