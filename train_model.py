from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    CATEGORICAL_FEATURES,
    INPUT_COLUMNS,
    NUMERIC_FEATURES,
    TelcoFeatureEngineer,
    load_raw_telco,
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.08,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(clf) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", TelcoFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("clf", clf),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    pre = pipeline.named_steps["preprocess"]
    return pre.get_feature_names_out()


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_raw_telco(DATA_PATH)
    X = df[INPUT_COLUMNS]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    best_name = None
    best_score = -np.inf
    best_pipeline = None

    for name, clf in candidate_models().items():
        pipe = make_pipeline(clf)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = {"mean_roc_auc": float(scores.mean()), "std_roc_auc": float(scores.std())}
        print(f"{name}: ROC-AUC {scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name
            best_pipeline = pipe

    assert best_pipeline is not None
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    metrics = {
        "best_model": best_name,
        "cv_results": cv_results,
        "test": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        },
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "churn_rate": float(y.mean()),
        "n_rows": int(len(df)),
    }

    feature_names = get_feature_names(best_pipeline)
    clf = best_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.abs(clf.coef_[0])
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    joblib.dump(best_pipeline, MODEL_DIR / "churn_pipeline.joblib")
    importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSelected model: {best_name}")
    print(f"Test ROC-AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"Test Recall:  {metrics['test']['recall']:.4f}")
    print(f"Test F1:      {metrics['test']['f1']:.4f}")
    print(f"Saved artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
