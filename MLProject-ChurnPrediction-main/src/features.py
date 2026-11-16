from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "num_services",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]

INPUT_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean Telco fields and add intermediate-level derived features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
        df["SeniorCitizen"] = (
            pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int).astype(str)
        )

        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
        df["num_services"] = df[SERVICE_COLS].eq("Yes").sum(axis=1)

        bins = [-0.1, 12, 24, 36, 48, 60, 200]
        labels = ["1-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
        df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)
        return df


def load_raw_telco(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df
