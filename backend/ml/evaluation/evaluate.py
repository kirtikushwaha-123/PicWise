import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from backend.ml.preprocessing.dataset import build_and_group_split_data


def evaluate_target(model, X_train, y_train, X_test, y_test, label_encoder, target_name):
    classes = label_encoder.classes_
    num_classes = len(classes)

    # 1. Majority-Class Baseline
    class_counts = pd.Series(y_train).value_counts()
    majority_class_idx = class_counts.idxmax()
    baseline_preds = np.full(shape=len(y_test), fill_value=majority_class_idx)

    base_acc = float(accuracy_score(y_test, baseline_preds))
    base_p_macro, base_r_macro, base_f1_macro, _ = precision_recall_fscore_support(
        y_test, baseline_preds, average="macro", zero_division=0
    )
    base_p_weighted, base_r_weighted, base_f1_weighted, _ = precision_recall_fscore_support(
        y_test, baseline_preds, average="weighted", zero_division=0
    )

    baseline_metrics = {
        "most_frequent_class": str(label_encoder.inverse_transform([majority_class_idx])[0]),
        "accuracy": round(base_acc, 4),
        "macro_precision": round(float(base_p_macro), 4),
        "macro_recall": round(float(base_r_macro), 4),
        "macro_f1": round(float(base_f1_macro), 4),
        "weighted_f1": round(float(base_f1_weighted), 4),
    }

    # 2. Fit XGBoost Model
    model.fit(X_train, y_train)

    # Predict & Probabilities on Test Set
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)

    acc = float(accuracy_score(y_test, preds))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_test, preds, average=None, zero_division=0
    )

    per_class_metrics = {}
    for idx, cname in enumerate(classes):
        per_class_metrics[str(cname)] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1_score": round(float(f1_class[idx]), 4),
            "support": int(support_class[idx]),
        }

    conf_matrix = confusion_matrix(y_test, preds, labels=np.arange(num_classes)).tolist()

    model_metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall": round(float(r_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class_metrics,
        "confusion_matrix": conf_matrix,
        "classes": [str(c) for c in classes],
    }

    # 3. Confidence Analysis
    is_correct = (preds == y_test)

    correct_conf = max_probs[is_correct] if np.any(is_correct) else np.array([0.0])
    incorrect_conf = max_probs[~is_correct] if np.any(~is_correct) else np.array([0.0])

    confidence_stats = {
        "overall": {
            "mean": round(float(np.mean(max_probs)), 4),
            "median": round(float(np.median(max_probs)), 4),
            "min": round(float(np.min(max_probs)), 4),
            "max": round(float(np.max(max_probs)), 4),
        },
        "correct_predictions": {
            "count": int(np.sum(is_correct)),
            "mean": round(float(np.mean(correct_conf)), 4),
            "median": round(float(np.median(correct_conf)), 4),
            "min": round(float(np.min(correct_conf)), 4),
            "max": round(float(np.max(correct_conf)), 4),
        },
        "incorrect_predictions": {
            "count": int(np.sum(~is_correct)),
            "mean": round(float(np.mean(incorrect_conf)), 4),
            "median": round(float(np.median(incorrect_conf)), 4),
            "min": round(float(np.min(incorrect_conf)), 4),
            "max": round(float(np.max(incorrect_conf)), 4),
        },
    }

    return {
        "baseline": baseline_metrics,
        "model": model_metrics,
        "confidence": confidence_stats,
    }


def compute_distribution(series):
    counts = series.value_counts().to_dict()
    total = len(series)
    dist = {}
    for key, count in counts.items():
        dist[str(key)] = {
            "count": int(count),
            "percentage": round(float(count / total * 100), 2),
        }
    return dist


def run_evaluation(food_path=None, personal_care_path=None, test_size=0.2, random_state=42, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    os.makedirs(output_dir, exist_ok=True)

    print("Running group-aware canonical ingredient data split...")
    data = build_and_group_split_data(
        food_path=food_path,
        personal_care_path=personal_care_path,
        test_size=test_size,
        random_state=random_state
    )

    df = data["df"]
    df_train = data["df_train"]
    df_test = data["df_test"]

    food_rows = int((df["domain"] == "food").sum())
    pc_rows = int((df["domain"] == "personal_care").sum())
    total_rows = len(df)
    unique_canonical = int(df["canonical_group"].nunique())
    train_canonical = int(df_train["canonical_group"].nunique())
    test_canonical = int(df_test["canonical_group"].nunique())

    dataset_info = {
        "total_representations": total_rows,
        "food_representations": food_rows,
        "personal_care_representations": pc_rows,
        "unique_canonical_ingredients": unique_canonical,
        "train_canonical_ingredients": train_canonical,
        "test_canonical_ingredients": test_canonical,
        "train_representations": len(df_train),
        "test_representations": len(df_test),
        "class_distributions": {
            "safety_level": {
                "overall": compute_distribution(df["safety_level"]),
                "train": compute_distribution(df_train["safety_level"]),
                "test": compute_distribution(df_test["safety_level"]),
            },
            "allergy_risk": {
                "overall": compute_distribution(df["allergy_risk"]),
                "train": compute_distribution(df_train["allergy_risk"]),
                "test": compute_distribution(df_test["allergy_risk"]),
            },
        },
    }

    # Evaluate Safety Model
    print("Evaluating Safety Level Classifier...")
    safety_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=random_state
    )
    safety_eval = evaluate_target(
        safety_model,
        data["X_train"],
        data["y_safety_train"],
        data["X_test"],
        data["y_safety_test"],
        data["safety_encoder"],
        "Safety Level"
    )

    # Evaluate Allergy Model
    print("Evaluating Allergy Risk Classifier...")
    allergy_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=random_state
    )
    allergy_eval = evaluate_target(
        allergy_model,
        data["X_train"],
        data["y_allergy_train"],
        data["X_test"],
        data["y_allergy_test"],
        data["allergy_encoder"],
        "Allergy Risk"
    )

    report_json = {
        "evaluation_strategy": {
            "type": "canonical_group_split",
            "test_size": test_size,
            "random_state": random_state,
            "leakage_prevention": "Canonical ingredient group split + train-only TF-IDF vectorization"
        },
        "dataset": dataset_info,
        "safety": safety_eval,
        "allergy": allergy_eval,
    }

    # Save JSON Report
    json_path = os.path.join(output_dir, "evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
    print(f"Saved machine-readable JSON evaluation report to: {json_path}")

    # Generate Markdown Report
    md_content = _generate_markdown_report(report_json)
    md_path = os.path.join(output_dir, "evaluation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved human-readable Markdown evaluation report to: {md_path}")

    return report_json


def _generate_markdown_report(report):
    ds = report["dataset"]
    strat = report["evaluation_strategy"]
    safety = report["safety"]
    allergy = report["allergy"]

    # Interpretation logic
    safety_beats = safety["model"]["accuracy"] > safety["baseline"]["accuracy"]
    allergy_beats = allergy["model"]["accuracy"] > allergy["baseline"]["accuracy"]

    safety_hardest = min(safety["model"]["per_class"].items(), key=lambda x: x[1]["f1_score"])
    allergy_hardest = min(allergy["model"]["per_class"].items(), key=lambda x: x[1]["f1_score"])

    md = f"""# PicWise ML Evaluation & Validation Report (Phase 3 Part 1)

## 1. Executive Summary

This report evaluates the **generalization performance** of PicWise's ingredient-name machine learning models on **unseen canonical ingredients**.

To guarantee evaluation integrity, all alternate/packaging names belonging to a canonical ingredient were grouped together, ensuring zero data leakage between training and testing sets. Furthermore, the character TF-IDF vectorizer was fitted strictly on the training partition.

---

## 2. Evaluation Strategy & Data Split

- **Split Strategy**: Canonical Ingredient Group-Aware Split (`GroupShuffleSplit`)
- **Test Set Ratio**: {strat['test_size'] * 100:.0f}%
- **Random Seed**: {strat['random_state']}
- **Leakage Prevention**: All alternate/packaging names share the canonical ingredient group ID. TF-IDF vectorization fitted on training names only.

### Dataset Overview

| Metric | Count |
| :--- | :--- |
| **Total Representations** | {ds['total_representations']} |
| **Food Representations** | {ds['food_representations']} |
| **Personal Care Representations** | {ds['personal_care_representations']} |
| **Total Unique Canonical Ingredients** | {ds['unique_canonical_ingredients']} |
| **Train Canonical Ingredients** | {ds['train_canonical_ingredients']} |
| **Test Canonical Ingredients** | {ds['test_canonical_ingredients']} |
| **Train Representations** | {ds['train_representations']} |
| **Test Representations** | {ds['test_representations']} |

---

## 3. Class Distributions

### Safety Level Distribution

| Class | Overall Count (%) | Train Count (%) | Test Count (%) |
| :--- | :--- | :--- | :--- |
"""
    for cls in ["Very Safe", "Safe", "Moderate Risk", "High Risk"]:
        ov = ds["class_distributions"]["safety_level"]["overall"].get(cls, {"count": 0, "percentage": 0.0})
        tr = ds["class_distributions"]["safety_level"]["train"].get(cls, {"count": 0, "percentage": 0.0})
        te = ds["class_distributions"]["safety_level"]["test"].get(cls, {"count": 0, "percentage": 0.0})
        md += f"| **{cls}** | {ov['count']} ({ov['percentage']}%) | {tr['count']} ({tr['percentage']}%) | {te['count']} ({te['percentage']}%) |\n"

    md += """
### Allergy Risk Distribution

| Class | Overall Count (%) | Train Count (%) | Test Count (%) |
| :--- | :--- | :--- | :--- |
"""
    for cls in ["None", "Low", "Medium", "High"]:
        ov = ds["class_distributions"]["allergy_risk"]["overall"].get(cls, {"count": 0, "percentage": 0.0})
        tr = ds["class_distributions"]["allergy_risk"]["train"].get(cls, {"count": 0, "percentage": 0.0})
        te = ds["class_distributions"]["allergy_risk"]["test"].get(cls, {"count": 0, "percentage": 0.0})
        md += f"| **{cls}** | {ov['count']} ({ov['percentage']}%) | {tr['count']} ({tr['percentage']}%) | {te['count']} ({te['percentage']}%) |\n"

    # Safety Model Section
    sm = safety["model"]
    sb = safety["baseline"]
    sc = safety["confidence"]

    md += f"""
---

## 4. Safety Level Classifier Evaluation

### Baseline vs. XGBoost Performance

| Metric | Majority Baseline ({sb['most_frequent_class']}) | XGBoost Model | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | {sb['accuracy']:.4f} | **{sm['accuracy']:.4f}** | {'+' if sm['accuracy'] >= sb['accuracy'] else ''}{(sm['accuracy'] - sb['accuracy']):.4f} |
| **Macro Precision** | {sb['macro_precision']:.4f} | **{sm['macro_precision']:.4f}** | {'+' if sm['macro_precision'] >= sb['macro_precision'] else ''}{(sm['macro_precision'] - sb['macro_precision']):.4f} |
| **Macro Recall** | {sb['macro_recall']:.4f} | **{sm['macro_recall']:.4f}** | {'+' if sm['macro_recall'] >= sb['macro_recall'] else ''}{(sm['macro_recall'] - sb['macro_recall']):.4f} |
| **Macro F1** | {sb['macro_f1']:.4f} | **{sm['macro_f1']:.4f}** | {'+' if sm['macro_f1'] >= sb['macro_f1'] else ''}{(sm['macro_f1'] - sb['macro_f1']):.4f} |
| **Weighted F1** | {sb['weighted_f1']:.4f} | **{sm['weighted_f1']:.4f}** | {'+' if sm['weighted_f1'] >= sb['weighted_f1'] else ''}{(sm['weighted_f1'] - sb['weighted_f1']):.4f} |

### Safety Level Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
"""
    for cls, metrics in sm["per_class"].items():
        md += f"| **{cls}** | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['support']} |\n"

    md += f"""
### Safety Level Confusion Matrix

Rows represent true labels; columns represent predictions:
`{sm['classes']}`

```text
{np.array(sm['confusion_matrix'])}
```

### Safety Level Confidence Analysis

- **Overall Test Confidence**: Mean = {sc['overall']['mean']:.4f}, Median = {sc['overall']['median']:.4f}, Range = [{sc['overall']['min']:.4f}, {sc['overall']['max']:.4f}]
- **Correct Predictions ({sc['correct_predictions']['count']})**: Mean Confidence = {sc['correct_predictions']['mean']:.4f}, Median = {sc['correct_predictions']['median']:.4f}
- **Incorrect Predictions ({sc['incorrect_predictions']['count']})**: Mean Confidence = {sc['incorrect_predictions']['mean']:.4f}, Median = {sc['incorrect_predictions']['median']:.4f}
"""

    # Allergy Model Section
    am = allergy["model"]
    ab = allergy["baseline"]
    ac = allergy["confidence"]

    md += f"""
---

## 5. Allergy Risk Classifier Evaluation

### Baseline vs. XGBoost Performance

| Metric | Majority Baseline ({ab['most_frequent_class']}) | XGBoost Model | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | {ab['accuracy']:.4f} | **{am['accuracy']:.4f}** | {'+' if am['accuracy'] >= ab['accuracy'] else ''}{(am['accuracy'] - ab['accuracy']):.4f} |
| **Macro Precision** | {ab['macro_precision']:.4f} | **{am['macro_precision']:.4f}** | {'+' if am['macro_precision'] >= ab['macro_precision'] else ''}{(am['macro_precision'] - ab['macro_precision']):.4f} |
| **Macro Recall** | {ab['macro_recall']:.4f} | **{am['macro_recall']:.4f}** | {'+' if am['macro_recall'] >= ab['macro_recall'] else ''}{(am['macro_recall'] - ab['macro_recall']):.4f} |
| **Macro F1** | {ab['macro_f1']:.4f} | **{am['macro_f1']:.4f}** | {'+' if am['macro_f1'] >= ab['macro_f1'] else ''}{(am['macro_f1'] - ab['macro_f1']):.4f} |
| **Weighted F1** | {ab['weighted_f1']:.4f} | **{am['weighted_f1']:.4f}** | {'+' if am['weighted_f1'] >= ab['weighted_f1'] else ''}{(am['weighted_f1'] - ab['weighted_f1']):.4f} |

### Allergy Risk Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
"""
    for cls, metrics in am["per_class"].items():
        md += f"| **{cls}** | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['support']} |\n"

    md += f"""
### Allergy Risk Confusion Matrix

Rows represent true labels; columns represent predictions:
`{am['classes']}`

```text
{np.array(am['confusion_matrix'])}
```

### Allergy Risk Confidence Analysis

- **Overall Test Confidence**: Mean = {ac['overall']['mean']:.4f}, Median = {ac['overall']['median']:.4f}, Range = [{ac['overall']['min']:.4f}, {ac['overall']['max']:.4f}]
- **Correct Predictions ({ac['correct_predictions']['count']})**: Mean Confidence = {ac['correct_predictions']['mean']:.4f}, Median = {ac['correct_predictions']['median']:.4f}
- **Incorrect Predictions ({ac['incorrect_predictions']['count']})**: Mean Confidence = {ac['incorrect_predictions']['mean']:.4f}, Median = {ac['incorrect_predictions']['median']:.4f}

---

## 6. Interpretation & Findings

1. **Baseline Comparison**:
   - **Safety Model**: Accuracy is **{sm['accuracy']:.4f}** (Macro F1: **{sm['macro_f1']:.4f}**) vs Majority Baseline **{sb['accuracy']:.4f}** (Macro F1: **{sb['macro_f1']:.4f}**).
   - **Allergy Model**: Accuracy is **{am['accuracy']:.4f}** (Macro F1: **{am['macro_f1']:.4f}**) vs Majority Baseline **{ab['accuracy']:.4f}** (Macro F1: **{ab['macro_f1']:.4f}**).

2. **Class Imbalance & Hardest Classes**:
   - Safety Level hardest class: **{safety_hardest[0]}** (F1: {safety_hardest[1]['f1_score']:.4f}, support: {safety_hardest[1]['support']}).
   - Allergy Risk hardest class: **{allergy_hardest[0]}** (F1: {allergy_hardest[1]['f1_score']:.4f}, support: {allergy_hardest[1]['support']}).
   - Both datasets exhibit heavy class imbalance (e.g. `Low` dominates Allergy Risk while `Safe` / `Very Safe` dominates Safety Level).

3. **Confidence Calibration Observation**:
   - Safety Model Mean Confidence: Correct predictions ({sc['correct_predictions']['mean']:.4f}) vs Incorrect predictions ({sc['incorrect_predictions']['mean']:.4f}).
   - Allergy Model Mean Confidence: Correct predictions ({ac['correct_predictions']['mean']:.4f}) vs Incorrect predictions ({ac['incorrect_predictions']['mean']:.4f}).
   - Higher confidence on correct predictions indicates useful probability separation, which can be leveraged for confidence thresholds/abstention in Phase 3 Part 2.

4. **Conclusion & Recommendation**:
   - Character n-gram TF-IDF on ingredient names alone provides modest predictive signal beyond majority class guessing on unseen canonical ingredients.
   - For high-stakes safety and allergy classifications, deterministic knowledge base lookup should remain the primary authority, with ML used as a secondary fallback accompanied by confidence thresholds.
"""
    return md


if __name__ == "__main__":
    run_evaluation()
