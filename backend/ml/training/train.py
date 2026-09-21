import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from backend.ml.preprocessing.dataset import build_and_split_data, DEFAULT_MODEL_DIR


def train_models(food_path=None, personal_care_path=None, model_dir=DEFAULT_MODEL_DIR):
    print("Preparing dataset and engineering features...")
    data = build_and_split_data(food_path, personal_care_path, model_dir=model_dir)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_safety_train = data["y_safety_train"]
    y_safety_test = data["y_safety_test"]
    y_allergy_train = data["y_allergy_train"]
    y_allergy_test = data["y_allergy_test"]

    safety_encoder = data["safety_encoder"]
    allergy_encoder = data["allergy_encoder"]

    os.makedirs(model_dir, exist_ok=True)

    # 1. Train Safety Level XGBoost Model
    print("Training XGBoost Safety Level Classifier...")
    safety_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42
    )
    safety_model.fit(X_train, y_safety_train)

    safety_preds = safety_model.predict(X_test)
    safety_acc = accuracy_score(y_safety_test, safety_preds)
    print("\n===========================================")
    print(f"SAFETY LEVEL MODEL EVALUATION (Accuracy: {safety_acc:.4f})")
    print("===========================================")
    print(classification_report(
        y_safety_test,
        safety_preds,
        target_names=safety_encoder.classes_
    ))

    # Save Safety Model
    safety_model_path = os.path.join(model_dir, "safety_model.joblib")
    joblib.dump(safety_model, safety_model_path)
    print(f"Safety model saved to: {safety_model_path}")

    # 2. Train Allergy Risk XGBoost Model
    print("\nTraining XGBoost Allergy Risk Classifier...")
    allergy_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42
    )
    allergy_model.fit(X_train, y_allergy_train)

    allergy_preds = allergy_model.predict(X_test)
    allergy_acc = accuracy_score(y_allergy_test, allergy_preds)
    print("\n===========================================")
    print(f"ALLERGY RISK MODEL EVALUATION (Accuracy: {allergy_acc:.4f})")
    print("===========================================")
    print(classification_report(
        y_allergy_test,
        allergy_preds,
        target_names=allergy_encoder.classes_
    ))

    # Save Allergy Model
    allergy_model_path = os.path.join(model_dir, "allergy_model.joblib")
    joblib.dump(allergy_model, allergy_model_path)
    print(f"Allergy model saved to: {allergy_model_path}")

    print("\nTraining complete successfully!")
    return {
        "safety_accuracy": safety_acc,
        "allergy_accuracy": allergy_acc,
        "safety_model_path": safety_model_path,
        "allergy_model_path": allergy_model_path
    }


if __name__ == "__main__":
    train_models()
