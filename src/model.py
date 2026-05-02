from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.model_selection import cross_val_score
import joblib, os
import numpy as np

def train_model(X_train, y_train, model_type="random_forest"):
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",   # handles class imbalance
            random_state=42,
            n_jobs=-1
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42
        )

    model.fit(X_train, y_train)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/churn_model.pkl")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc":    round(roc_auc_score(y_test, y_proba), 4),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "cm":     confusion_matrix(y_test, y_pred).tolist(),
        "fpr":    [],
        "tpr":    [],
    }
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    metrics["fpr"] = fpr.tolist()
    metrics["tpr"] = tpr.tolist()
    return metrics

def load_model():
    return joblib.load("models/churn_model.pkl")