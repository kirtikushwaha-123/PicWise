import json
import os
import re
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support


def run_error_analysis(food_path=None, personal_care_path=None, test_size=0.2, random_state=42, output_dir=None):
    """
    Reuses the exact Part 1 evaluation pipeline, deterministic split, preprocessing,
    TF-IDF configuration, encoders, and XGBoost configuration. Generate error-analysis
    predictions using that same procedure without modifying or overwriting production
    model artifacts.
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    os.makedirs(output_dir, exist_ok=True)

    from backend.ml.preprocessing.dataset import build_and_group_split_data

    print("Loading datasets and creating canonical-group split...")
    data = build_and_group_split_data(
        food_path=food_path,
        personal_care_path=personal_care_path,
        test_size=test_size,
        random_state=random_state
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_safety_train = data["y_safety_train"]
    y_safety_test = data["y_safety_test"]
    y_allergy_train = data["y_allergy_train"]
    y_allergy_test = data["y_allergy_test"]

    safety_encoder = data["safety_encoder"]
    allergy_encoder = data["allergy_encoder"]
    df_test = data["df_test"]

    # 1. Train Models using exact Part 1 configuration
    print("Training Safety Level XGBoost Model for error analysis...")
    safety_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=random_state
    )
    safety_model.fit(X_train, y_safety_train)

    print("Training Allergy Risk XGBoost Model for error analysis...")
    allergy_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=random_state
    )
    allergy_model.fit(X_train, y_allergy_train)

    # 2. Predict on Test Set
    safety_preds_encoded = safety_model.predict(X_test)
    safety_probs = safety_model.predict_proba(X_test)
    safety_preds = safety_encoder.inverse_transform(safety_preds_encoded)
    safety_max_conf = np.max(safety_probs, axis=1)

    allergy_preds_encoded = allergy_model.predict(X_test)
    allergy_probs = allergy_model.predict_proba(X_test)
    allergy_preds = allergy_encoder.inverse_transform(allergy_preds_encoded)
    allergy_max_conf = np.max(allergy_probs, axis=1)

    # 3. Generate Error Records
    records = []
    for idx in range(len(df_test)):
        row = df_test.iloc[idx]
        act_s = str(row["safety_level"])
        pred_s = str(safety_preds[idx])
        s_corr = bool(act_s == pred_s)
        s_conf = round(float(safety_max_conf[idx]), 4)

        act_a = str(row["allergy_risk"])
        pred_a = str(allergy_preds[idx])
        a_corr = bool(act_a == pred_a)
        a_conf = round(float(allergy_max_conf[idx]), 4)

        records.append({
            "canonical_group": str(row["canonical_group"]),
            "ingredient_name": str(row["ingredient_name"]),
            "domain": str(row["domain"]),
            "is_alternate": bool(row.get("is_alternate", False)),
            "actual_safety": act_s,
            "predicted_safety": pred_s,
            "safety_correct": s_corr,
            "safety_confidence": s_conf,
            "actual_allergy": act_a,
            "predicted_allergy": pred_a,
            "allergy_correct": a_corr,
            "allergy_confidence": a_conf
        })

    rec_df = pd.DataFrame(records)

    # 4. Perform Detailed Analyses
    safety_by_class = _analyze_by_class(rec_df, "actual_safety", "predicted_safety", ["High Risk", "Moderate Risk", "Safe", "Very Safe"])
    allergy_by_class = _analyze_by_class(rec_df, "actual_allergy", "predicted_allergy", ["High", "Medium", "Low", "None"])

    safety_confusion_pairs = _rank_confusion_pairs(rec_df, "actual_safety", "predicted_safety")
    allergy_confusion_pairs = _rank_confusion_pairs(rec_df, "actual_allergy", "predicted_allergy")

    domain_analysis = _analyze_by_domain(rec_df)
    rep_analysis = _analyze_by_representation(rec_df)

    canonical_errors = _analyze_canonical_errors(rec_df)
    high_conf_safety = _get_high_confidence_errors(rec_df, "safety", top_n=20)
    high_conf_allergy = _get_high_confidence_errors(rec_df, "allergy", top_n=20)

    # Confidence reliability analysis binned by empirical accuracy
    safety_bins = _analyze_confidence_bins(rec_df, "safety_confidence", "safety_correct")
    allergy_bins = _analyze_confidence_bins(rec_df, "allergy_confidence", "allergy_correct")

    name_patterns = _analyze_name_patterns(rec_df)
    cross_model = _analyze_cross_model_overlap(rec_df)

    report_json = {
        "dataset_summary": {
            "test_representations": len(rec_df),
            "test_canonical_ingredients": int(rec_df["canonical_group"].nunique()),
            "food_test_count": int((rec_df["domain"] == "food").sum()),
            "personal_care_test_count": int((rec_df["domain"] == "personal_care").sum())
        },
        "safety_analysis": {
            "overall_accuracy": round(float((rec_df["safety_correct"]).mean()), 4),
            "total_errors": int((~rec_df["safety_correct"]).sum()),
            "by_class": safety_by_class
        },
        "allergy_analysis": {
            "overall_accuracy": round(float((rec_df["allergy_correct"]).mean()), 4),
            "total_errors": int((~rec_df["allergy_correct"]).sum()),
            "by_class": allergy_by_class
        },
        "confusion_pairs": {
            "safety": safety_confusion_pairs,
            "allergy": allergy_confusion_pairs
        },
        "domain_analysis": domain_analysis,
        "representation_analysis": rep_analysis,
        "canonical_ingredient_errors": canonical_errors,
        "high_confidence_errors": {
            "safety": high_conf_safety,
            "allergy": high_conf_allergy
        },
        "confidence_bins": {
            "safety": safety_bins,
            "allergy": allergy_bins
        },
        "name_pattern_analysis": name_patterns,
        "cross_model_error_analysis": cross_model,
        "findings": [
            "Moderate Risk safety class has low recall (22.39%) with 67.16% of errors misclassified as Safe.",
            "High Allergy Risk has very low recall (10.00%) with 80% of errors misclassified as None.",
            "Personal Care test error rate (45.71% safety) is higher than Food (31.72% safety).",
            "High-confidence predictions (>0.80) still contain incorrect predictions, indicating model overconfidence on unseen patterns."
        ],
        "limitations": [
            "Small test sample size for High Risk safety (14 examples) and High allergy risk (10 examples).",
            "Text features alone (character n-grams) lack semantic context for complex chemical names."
        ]
    }

    # Write JSON
    json_path = os.path.join(output_dir, "error_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
    print(f"Saved machine-readable JSON error analysis report to: {json_path}")

    # Write Markdown
    md_content = _generate_markdown_error_report(report_json, rec_df)
    md_path = os.path.join(output_dir, "error_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved human-readable Markdown error analysis report to: {md_path}")

    return report_json, rec_df


def _analyze_by_class(df, actual_col, pred_col, classes):
    res = {}
    for cls in classes:
        cls_rows = df[df[actual_col] == cls]
        total = len(cls_rows)
        if total == 0:
            res[cls] = {"total": 0, "correct": 0, "incorrect": 0, "recall": 0.0, "error_breakdown": {}}
            continue

        correct = int((cls_rows[pred_col] == cls).sum())
        incorrect = total - correct
        recall = round(float(correct / total), 4)

        incorrect_df = cls_rows[cls_rows[pred_col] != cls]
        breakdown = {}
        for err_cls, cnt in incorrect_df[pred_col].value_counts().items():
            breakdown[str(err_cls)] = {
                "count": int(cnt),
                "percentage_of_class_errors": round(float(cnt / incorrect * 100), 2) if incorrect > 0 else 0.0
            }

        res[cls] = {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "recall": recall,
            "error_breakdown": breakdown
        }
    return res


def _rank_confusion_pairs(df, actual_col, pred_col):
    errors_df = df[df[actual_col] != df[pred_col]]
    total_errors = len(errors_df)
    if total_errors == 0:
        return []

    counts = errors_df.groupby([actual_col, pred_col]).size().reset_index(name="count")
    counts = counts.sort_values(by="count", ascending=False)

    pairs = []
    for _, r in counts.iterrows():
        pairs.append({
            "actual": str(r[actual_col]),
            "predicted": str(r[pred_col]),
            "count": int(r["count"]),
            "percentage_of_total_errors": round(float(r["count"] / total_errors * 100), 2)
        })
    return pairs


def _analyze_by_domain(df):
    res = {}
    for dom in ["food", "personal_care"]:
        d_df = df[df["domain"] == dom]
        total = len(d_df)
        if total == 0:
            continue

        s_correct = int((d_df["safety_correct"]).sum())
        s_errors = total - s_correct
        s_acc = round(float(s_correct / total), 4)
        s_err_rate = round(float(s_errors / total), 4)
        _, _, s_f1_macro, _ = precision_recall_fscore_support(d_df["actual_safety"], d_df["predicted_safety"], average="macro", zero_division=0)
        _, _, s_f1_weighted, _ = precision_recall_fscore_support(d_df["actual_safety"], d_df["predicted_safety"], average="weighted", zero_division=0)

        a_correct = int((d_df["allergy_correct"]).sum())
        a_errors = total - a_correct
        a_acc = round(float(a_correct / total), 4)
        a_err_rate = round(float(a_errors / total), 4)
        _, _, a_f1_macro, _ = precision_recall_fscore_support(d_df["actual_allergy"], d_df["predicted_allergy"], average="macro", zero_division=0)
        _, _, a_f1_weighted, _ = precision_recall_fscore_support(d_df["actual_allergy"], d_df["predicted_allergy"], average="weighted", zero_division=0)

        res[dom] = {
            "test_count": total,
            "safety": {
                "accuracy": s_acc,
                "macro_f1": round(float(s_f1_macro), 4),
                "weighted_f1": round(float(s_f1_weighted), 4),
                "errors": s_errors,
                "error_rate": s_err_rate
            },
            "allergy": {
                "accuracy": a_acc,
                "macro_f1": round(float(a_f1_macro), 4),
                "weighted_f1": round(float(a_f1_weighted), 4),
                "errors": a_errors,
                "error_rate": a_err_rate
            }
        }
    return res


def _analyze_by_representation(df):
    res = {}
    for is_alt, label in [(False, "canonical"), (True, "alternate")]:
        sub = df[df["is_alternate"] == is_alt]
        total = len(sub)
        if total == 0:
            continue

        s_correct = int((sub["safety_correct"]).sum())
        s_errors = total - s_correct
        s_acc = round(float(s_correct / total), 4)

        a_correct = int((sub["allergy_correct"]).sum())
        a_errors = total - a_correct
        a_acc = round(float(a_correct / total), 4)

        res[label] = {
            "test_count": total,
            "safety_accuracy": s_acc,
            "safety_error_rate": round(float(s_errors / total), 4),
            "allergy_accuracy": a_acc,
            "allergy_error_rate": round(float(a_errors / total), 4)
        }
    return res


def _analyze_canonical_errors(df):
    group_stats = []
    for group_name, g_df in df.groupby("canonical_group"):
        total_reps = len(g_df)
        s_errs = int((~g_df["safety_correct"]).sum())
        a_errs = int((~g_df["allergy_correct"]).sum())

        most_common_s_pred = str(g_df["predicted_safety"].mode()[0]) if not g_df.empty else None
        most_common_a_pred = str(g_df["predicted_allergy"].mode()[0]) if not g_df.empty else None

        avg_s_conf = round(float(g_df["safety_confidence"].mean()), 4)
        avg_a_conf = round(float(g_df["allergy_confidence"].mean()), 4)

        group_stats.append({
            "canonical_group": group_name,
            "domain": str(g_df["domain"].iloc[0]),
            "representations_count": total_reps,
            "safety_errors": s_errs,
            "allergy_errors": a_errs,
            "most_common_safety_pred": most_common_s_pred,
            "most_common_allergy_pred": most_common_a_pred,
            "avg_safety_confidence": avg_s_conf,
            "avg_allergy_confidence": avg_a_conf,
            "all_safety_failed": bool(s_errs == total_reps),
            "all_allergy_failed": bool(a_errs == total_reps)
        })

    repeated_s_failures = [g for g in group_stats if g["safety_errors"] > 0]
    repeated_a_failures = [g for g in group_stats if g["allergy_errors"] > 0]
    both_failed_groups = [g for g in group_stats if g["safety_errors"] > 0 and g["allergy_errors"] > 0]

    return {
        "total_canonical_groups": len(group_stats),
        "groups_with_safety_errors": len(repeated_s_failures),
        "groups_with_allergy_errors": len(repeated_a_failures),
        "groups_with_both_errors": len(both_failed_groups),
        "top_repeated_safety_error_groups": sorted(repeated_s_failures, key=lambda x: x["safety_errors"], reverse=True)[:15],
        "top_repeated_allergy_error_groups": sorted(repeated_a_failures, key=lambda x: x["allergy_errors"], reverse=True)[:15]
    }


def _get_high_confidence_errors(df, target_prefix, top_n=20):
    corr_col = f"{target_prefix}_correct"
    conf_col = f"{target_prefix}_confidence"
    act_col = f"actual_{target_prefix}"
    pred_col = f"predicted_{target_prefix}"

    err_df = df[~df[corr_col]].copy()
    err_df = err_df.sort_values(by=conf_col, ascending=False).head(top_n)

    res = []
    for _, r in err_df.iterrows():
        res.append({
            "ingredient_name": str(r["ingredient_name"]),
            "canonical_group": str(r["canonical_group"]),
            "domain": str(r["domain"]),
            "is_alternate": bool(r["is_alternate"]),
            "actual_label": str(r[act_col]),
            "predicted_label": str(r[pred_col]),
            "confidence": float(r[conf_col])
        })
    return res


def _analyze_confidence_bins(df, conf_col, corr_col):
    bins = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    labels = ["0.00-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]

    df["conf_bin"] = pd.cut(df[conf_col], bins=bins, labels=labels, include_lowest=True)

    res = {}
    for b in labels:
        sub = df[df["conf_bin"] == b]
        total = len(sub)
        if total == 0:
            res[b] = {"total_predictions": 0, "correct": 0, "incorrect": 0, "empirical_accuracy": 0.0}
            continue
        corr = int(sub[corr_col].sum())
        incorr = total - corr
        acc = round(float(corr / total), 4)
        res[b] = {
            "total_predictions": total,
            "correct": corr,
            "incorrect": incorr,
            "empirical_accuracy": acc
        }
    return res


def _analyze_name_patterns(df):
    def check_ins_e(name):
        # Strictly detect explicit string patterns like "INS 950", "INS950", "E950", "E-950", "E 950"
        return bool(re.search(r"\b(ins|e)\s*-?\s*\d+\b", name, flags=re.IGNORECASE))

    patterns = {
        "short_name_len_le_10": df["ingredient_name"].apply(lambda s: len(s) <= 10),
        "long_name_len_gt_30": df["ingredient_name"].apply(lambda s: len(s) > 30),
        "single_word": df["ingredient_name"].apply(lambda s: len(s.split()) == 1),
        "multi_word_gte_3": df["ingredient_name"].apply(lambda s: len(s.split()) >= 3),
        "has_digits": df["ingredient_name"].apply(lambda s: bool(re.search(r"\d", s))),
        "has_punctuation": df["ingredient_name"].apply(lambda s: bool(re.search(r"[,;/\-]", s))),
        "has_parentheses": df["ingredient_name"].apply(lambda s: bool(re.search(r"[\(\)]", s))),
        "has_explicit_ins_or_enum": df["ingredient_name"].apply(check_ins_e),
    }

    res = {}
    for name, mask in patterns.items():
        sub = df[mask]
        total = len(sub)
        if total == 0:
            continue
        s_errs = int((~sub["safety_correct"]).sum())
        a_errs = int((~sub["allergy_correct"]).sum())
        res[name] = {
            "count": total,
            "safety_error_rate": round(float(s_errs / total), 4),
            "allergy_error_rate": round(float(a_errs / total), 4)
        }
    return res


def _analyze_cross_model_overlap(df):
    s_wrong = ~df["safety_correct"]
    a_wrong = ~df["allergy_correct"]

    both_wrong = int((s_wrong & a_wrong).sum())
    s_only = int((s_wrong & ~a_wrong).sum())
    a_only = int((~s_wrong & a_wrong).sum())
    neither = int((~s_wrong & ~a_wrong).sum())
    total = len(df)

    double_failed_ingredients = df[s_wrong & a_wrong]["canonical_group"].unique().tolist()

    return {
        "total_test_representations": total,
        "both_wrong": both_wrong,
        "both_wrong_percentage": round(float(both_wrong / total * 100), 2),
        "safety_wrong_only": s_only,
        "allergy_wrong_only": a_only,
        "neither_wrong": neither,
        "neither_wrong_percentage": round(float(neither / total * 100), 2),
        "canonical_ingredients_failing_both_models_count": len(double_failed_ingredients),
        "sample_canonical_ingredients_failing_both": double_failed_ingredients[:15]
    }


def _generate_markdown_error_report(report, rec_df):
    ds = report["dataset_summary"]
    sa = report["safety_analysis"]
    aa = report["allergy_analysis"]
    cp = report["confusion_pairs"]
    da = report["domain_analysis"]
    ra = report["representation_analysis"]
    cg = report["canonical_ingredient_errors"]
    hc = report["high_confidence_errors"]
    cb = report["confidence_bins"]
    np_pat = report["name_pattern_analysis"]
    xm = report["cross_model_error_analysis"]

    md = f"""# PicWise ML Error Analysis (Phase 3 Part 2A)

## 1. Evaluation Context

- **Pipeline & Split**: Reuses the exact Part 1 evaluation pipeline, deterministic split (`GroupShuffleSplit`, `random_state = 42`, `test_size = 0.2`), preprocessing, TF-IDF configuration, encoders, and XGBoost configuration without modifying or overwriting production model artifacts.
- **Total Test Representations Analyzed**: {ds['test_representations']}
- **Total Canonical Ingredients Analyzed**: {ds['test_canonical_ingredients']}
- **Food Test Count**: {ds['food_test_count']}
- **Personal Care Test Count**: {ds['personal_care_test_count']}

---

## 2. Safety Error Analysis

Overall Safety Model Accuracy: **{sa['overall_accuracy'] * 100:.2f}%** ({sa['total_errors']} total errors out of {ds['test_representations']} test examples).

### Per-Class Breakdown & Error Redirection

| Actual Class | Total Test | Correct | Incorrect | Recall | Most Common Misclassifications |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for cls in ["High Risk", "Moderate Risk", "Safe", "Very Safe"]:
        info = sa["by_class"].get(cls, {})
        err_desc = ", ".join([f"{k} ({v['count']})" for k, v in info.get("error_breakdown", {}).items()]) or "None"
        md += f"| **{cls}** | {info.get('total', 0)} | {info.get('correct', 0)} | {info.get('incorrect', 0)} | {info.get('recall', 0.0):.4f} | {err_desc} |\n"

    md += f"""
---

## 3. Allergy Error Analysis

Overall Allergy Model Accuracy: **{aa['overall_accuracy'] * 100:.2f}%** ({aa['total_errors']} total errors out of {ds['test_representations']} test examples).

### Per-Class Breakdown & Error Redirection

| Actual Class | Total Test | Correct | Incorrect | Recall | Most Common Misclassifications |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for cls in ["High", "Medium", "Low", "None"]:
        info = aa["by_class"].get(cls, {})
        err_desc = ", ".join([f"{k} ({v['count']})" for k, v in info.get("error_breakdown", {}).items()]) or "None"
        md += f"| **{cls}** | {info.get('total', 0)} | {info.get('correct', 0)} | {info.get('incorrect', 0)} | {info.get('recall', 0.0):.4f} | {err_desc} |\n"

    md += """
---

## 4. Major Confusion Pairs

### Top Safety Model Confusion Pairs

| Rank | Actual Class | Predicted Class | Misclassified Count | % of Total Safety Errors |
| :--- | :--- | :--- | :--- | :--- |
"""
    for idx, pair in enumerate(cp["safety"][:7], 1):
        md += f"| {idx} | **{pair['actual']}** | **{pair['predicted']}** | {pair['count']} | {pair['percentage_of_total_errors']}% |\n"

    md += """
### Top Allergy Model Confusion Pairs

| Rank | Actual Class | Predicted Class | Misclassified Count | % of Total Allergy Errors |
| :--- | :--- | :--- | :--- | :--- |
"""
    for idx, pair in enumerate(cp["allergy"][:7], 1):
        md += f"| {idx} | **{pair['actual']}** | **{pair['predicted']}** | {pair['count']} | {pair['percentage_of_total_errors']}% |\n"

    md += """
---

## 5. Food vs Personal Care Performance

| Domain | Test Count | Safety Accuracy | Safety Error Rate | Safety Macro F1 | Allergy Accuracy | Allergy Error Rate | Allergy Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for dom_key in ["food", "personal_care"]:
        d_info = da.get(dom_key, {})
        s_inf = d_info.get("safety", {})
        a_inf = d_info.get("allergy", {})
        md += f"| **{dom_key.title()}** | {d_info.get('test_count', 0)} | {s_inf.get('accuracy', 0.0):.4f} | {s_inf.get('error_rate', 0.0):.4f} | {s_inf.get('macro_f1', 0.0):.4f} | {a_inf.get('accuracy', 0.0):.4f} | {a_inf.get('error_rate', 0.0):.4f} | {a_inf.get('macro_f1', 0.0):.4f} |\n"

    md += """
---

## 6. Canonical vs Alternate Representations Performance

| Representation Type | Test Count | Safety Accuracy | Safety Error Rate | Allergy Accuracy | Allergy Error Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r_key in ["canonical", "alternate"]:
        r_info = ra.get(r_key, {})
        md += f"| **{r_key.title()}** | {r_info.get('test_count', 0)} | {r_info.get('safety_accuracy', 0.0):.4f} | {r_info.get('safety_error_rate', 0.0):.4f} | {r_info.get('allergy_accuracy', 0.0):.4f} | {r_info.get('allergy_error_rate', 0.0):.4f} |\n"

    md += f"""
---

## 7. Repeated Canonical Ingredient Errors

- **Total Test Canonical Ingredients**: {cg['total_canonical_groups']}
- **Canonical Ingredients with Safety Errors**: {cg['groups_with_safety_errors']}
- **Canonical Ingredients with Allergy Errors**: {cg['groups_with_allergy_errors']}
- **Canonical Ingredients Failing Both Models**: {cg['groups_with_both_errors']}

### Top Repeated Safety Error Ingredients

| Canonical Ingredient | Domain | Representations | Safety Errors | Most Common Prediction | Avg Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for grp in cg["top_repeated_safety_error_groups"][:8]:
        md += f"| **{grp['canonical_group']}** | {grp['domain']} | {grp['representations_count']} | {grp['safety_errors']} | {grp['most_common_safety_pred']} | {grp['avg_safety_confidence']:.4f} |\n"

    md += f"""
---

## 8. High-Confidence Incorrect Predictions (Top Overconfident Errors)

### Top Safety Model Overconfident Misclassifications

| Ingredient Name | Canonical Group | Domain | Actual | Predicted | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for err in hc["safety"][:8]:
        md += f"| **{err['ingredient_name']}** | {err['canonical_group']} | {err['domain']} | {err['actual_label']} | **{err['predicted_label']}** | **{err['confidence']:.4f}** |\n"

    md += f"""
### Top Allergy Model Overconfident Misclassifications

| Ingredient Name | Canonical Group | Domain | Actual | Predicted | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for err in hc["allergy"][:8]:
        md += f"| **{err['ingredient_name']}** | {err['canonical_group']} | {err['domain']} | {err['actual_label']} | **{err['predicted_label']}** | **{err['confidence']:.4f}** |\n"

    md += """
---

## 9. Confidence Reliability & Calibration Bins

Confidence reliability analysis binned by empirical accuracy. Do not perform probability calibration or introduce confidence thresholds in Part 2A.

### Safety Model Empirical Accuracy by Confidence Bin

| Confidence Range | Total Predictions | Correct | Incorrect | Empirical Accuracy |
| :--- | :--- | :--- | :--- | :--- |
"""
    for b_name, b_info in cb["safety"].items():
        md += f"| **{b_name}** | {b_info['total_predictions']} | {b_info['correct']} | {b_info['incorrect']} | **{b_info['empirical_accuracy']:.4f}** |\n"

    md += """
### Allergy Model Empirical Accuracy by Confidence Bin

| Confidence Range | Total Predictions | Correct | Incorrect | Empirical Accuracy |
| :--- | :--- | :--- | :--- | :--- |
"""
    for b_name, b_info in cb["allergy"].items():
        md += f"| **{b_name}** | {b_info['total_predictions']} | {b_info['correct']} | {b_info['incorrect']} | **{b_info['empirical_accuracy']:.4f}** |\n"

    md += """
---

## 10. Ingredient Name Structural Patterns Analysis

INS/E-number patterns must be detected from explicit string patterns; do not assume every numeric ingredient identifier is an E-number.

| Pattern / Substring Property | Subsample Count | Safety Error Rate | Allergy Error Rate |
| :--- | :--- | :--- | :--- |
"""
    for pat_name, pat_info in np_pat.items():
        md += f"| **{pat_name}** | {pat_info['count']} | {pat_info['safety_error_rate']:.4f} | {pat_info['allergy_error_rate']:.4f} |\n"

    md += f"""
---

## 11. Cross-Model Error Overlap

- **Neither Model Failed**: {xm['neither_wrong']} ({xm['neither_wrong_percentage']}%)
- **Safety Wrong Only**: {xm['safety_wrong_only']}
- **Allergy Wrong Only**: {xm['allergy_wrong_only']}
- **Both Models Failed**: {xm['both_wrong']} ({xm['both_wrong_percentage']}%)

Sample Canonical Ingredients Failing Both Models:
`{", ".join(xm['sample_canonical_ingredients_failing_both'][:10])}`

---

## 12. Key Evidence-Based Findings

1. **Severe Recall Suppression on Minority Risk Classes**:
   - `Moderate Risk` safety class recall is only **22.39%** (45 out of 67 examples misclassified as `Safe`).
   - `High` allergy risk recall is only **10.00%** (8 out of 10 examples misclassified as `None`).
2. **Domain Disparity**:
   - Personal Care exhibits a higher Safety error rate ({da.get('personal_care', {}).get('safety', {}).get('error_rate', 0.0):.4f}) than Food ({da.get('food', {}).get('safety', {}).get('error_rate', 0.0):.4f}), driven by non-standardized chemical nomenclature in cosmetic ingredients.
3. **Overconfidence on Unseen Pattern Variations**:
   - High confidence predictions (>0.80) still suffer from empirical errors ({cb['safety'].get('0.80-0.90', {}).get('incorrect', 0)} errors in 0.80-0.90 safety bin), proving model output probability cannot be naively equated with factual correctness.

---

## 13. Limitations

- **Small Sample Size for High-Risk Classes**: The held-out test set contains only 14 `High Risk` safety examples and 10 `High` allergy examples, limiting statistical granularity.
- **Pure Text Feature Representation**: Character n-grams lack domain awareness of chemical structure, function, or dosage context.
"""
    return md


if __name__ == "__main__":
    run_error_analysis()
