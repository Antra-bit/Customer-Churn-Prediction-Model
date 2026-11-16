from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features import INPUT_COLUMNS, load_raw_telco

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E2E8F0"),
    margin=dict(l=20, r=20, t=50, b=20),
)

st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; }
      .kpi-card {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1rem 1.1rem;
      }
      .kpi-label { color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.2rem; }
      .kpi-value { color: #f8fafc; font-size: 1.7rem; font-weight: 700; }
      .risk-high { color: #fb7185; font-size: 1.6rem; font-weight: 700; }
      .risk-med { color: #fbbf24; font-size: 1.6rem; font-weight: 700; }
      .risk-low { color: #34d399; font-size: 1.6rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0)
    return raw


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_importance() -> pd.DataFrame:
    if not IMPORTANCE_PATH.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.read_csv(IMPORTANCE_PATH)


def kpi(label: str, value: str) -> None:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("Chart filters")
    contracts = st.sidebar.multiselect(
        "Contract", sorted(df["Contract"].unique()), default=list(df["Contract"].unique())
    )
    internets = st.sidebar.multiselect(
        "Internet service",
        sorted(df["InternetService"].unique()),
        default=list(df["InternetService"].unique()),
    )
    payments = st.sidebar.multiselect(
        "Payment method",
        sorted(df["PaymentMethod"].unique()),
        default=list(df["PaymentMethod"].unique()),
    )
    tenure_range = st.sidebar.slider(
        "Tenure (months)",
        int(df["tenure"].min()),
        int(df["tenure"].max()),
        (int(df["tenure"].min()), int(df["tenure"].max())),
    )
    filtered = df[
        df["Contract"].isin(contracts)
        & df["InternetService"].isin(internets)
        & df["PaymentMethod"].isin(payments)
        & df["tenure"].between(tenure_range[0], tenure_range[1])
    ]
    return filtered


def page_overview(df: pd.DataFrame) -> None:
    st.title("Telco customer churn dashboard")
    st.caption("Explore churn patterns, then score a customer with the improved pipeline model.")

    filtered = apply_filters(df)
    churned = filtered["Churn"] == "Yes"
    n = len(filtered)
    churn_rate = churned.mean() if n else 0
    revenue_at_risk = filtered.loc[churned, "MonthlyCharges"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Customers in view", f"{n:,}")
    with c2:
        kpi("Churn rate", f"{churn_rate:.1%}")
    with c3:
        kpi("Churned customers", f"{int(churned.sum()):,}")
    with c4:
        kpi("Monthly revenue at risk", f"${revenue_at_risk:,.0f}")

    st.markdown("### Interactive insights")
    left, right = st.columns(2)

    with left:
        mix = filtered["Churn"].value_counts().rename_axis("status").reset_index(name="count")
        fig = px.pie(
            mix,
            names="status",
            values="count",
            hole=0.55,
            color="status",
            color_discrete_map={"Yes": "#F43F5E", "No": "#6366F1"},
            title="Churn mix",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")

    with right:
        by_contract = (
            filtered.groupby("Contract")["Churn"]
            .apply(lambda s: (s == "Yes").mean())
            .reset_index(name="churn_rate")
        )
        fig = px.bar(
            by_contract,
            x="Contract",
            y="churn_rate",
            title="Churn rate by contract",
            color="churn_rate",
            color_continuous_scale="RdPu",
        )
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    dim = st.selectbox(
        "Break down churn by",
        [
            "Contract",
            "PaymentMethod",
            "InternetService",
            "gender",
            "Partner",
            "SeniorCitizen",
            "PaperlessBilling",
            "TechSupport",
            "OnlineSecurity",
        ],
        index=1,
    )
    grouped = (
        filtered.groupby(dim)
        .agg(customers=("Churn", "size"), churn_rate=("Churn", lambda s: (s == "Yes").mean()))
        .reset_index()
    )
    fig = px.bar(
        grouped,
        x=dim,
        y="churn_rate",
        hover_data=["customers"],
        title=f"Churn rate by {dim}",
        color="churn_rate",
        color_continuous_scale="OrRd",
    )
    fig.update_layout(**PLOTLY_LAYOUT, yaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            filtered,
            x="tenure",
            color="Churn",
            barmode="overlay",
            nbins=36,
            color_discrete_map={"Yes": "#F43F5E", "No": "#6366F1"},
            title="Tenure distribution by churn",
            opacity=0.75,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.box(
            filtered,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={"Yes": "#F43F5E", "No": "#6366F1"},
            title="Monthly charges vs churn",
            points="outliers",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")

    fig = px.scatter(
        filtered.sample(min(1500, len(filtered)), random_state=42) if len(filtered) else filtered,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        size="TotalCharges",
        hover_data=["Contract", "InternetService", "PaymentMethod"],
        color_discrete_map={"Yes": "#F43F5E", "No": "#6366F1"},
        title="Tenure vs monthly charges (bubble size = total charges)",
        opacity=0.7,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")


def default_total(tenure: int, monthly: float) -> float:
    return round(max(tenure, 1) * monthly, 2)


def page_predict(model) -> None:
    st.title("Predict customer churn")
    st.caption("Fill in the customer profile. The pipeline encodes features the same way the model was trained.")

    if model is None:
        st.error("Model not found. Run `python train_model.py` first.")
        return

    with st.form("predict_form"):
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", ["Yes", "No"])
        with b:
            multiple = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"] if phone == "Yes" else ["No phone service"],
            )
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            no_inet = "No internet service"
            security = st.selectbox(
                "Online security",
                [no_inet] if internet == "No" else ["No", "Yes", no_inet],
            )
            backup = st.selectbox(
                "Online backup",
                [no_inet] if internet == "No" else ["No", "Yes", no_inet],
            )
            protection = st.selectbox(
                "Device protection",
                [no_inet] if internet == "No" else ["No", "Yes", no_inet],
            )
            support = st.selectbox(
                "Tech support",
                [no_inet] if internet == "No" else ["No", "Yes", no_inet],
            )
        with c:
            tv = st.selectbox(
                "Streaming TV",
                [no_inet] if internet == "No" else ["No", "Yes", no_inet],
            )
            movies = st.selectbox(
                "Streaming movies",
                [no_inet] if internet == "No" else ["No", "Yes", no_inet],
            )
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
            total = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=10000.0,
                value=float(default_total(tenure, monthly)),
                step=10.0,
                help="Defaults to tenure × monthly charges. Override if you have the actual total.",
            )
        submitted = st.form_submit_button("Predict churn", width="stretch")

    if not submitted:
        st.info("Set the customer attributes and click Predict churn.")
        return

    row = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple if phone == "Yes" else "No phone service",
                "InternetService": internet,
                "OnlineSecurity": no_inet if internet == "No" else security,
                "OnlineBackup": no_inet if internet == "No" else backup,
                "DeviceProtection": no_inet if internet == "No" else protection,
                "TechSupport": no_inet if internet == "No" else support,
                "StreamingTV": no_inet if internet == "No" else tv,
                "StreamingMovies": no_inet if internet == "No" else movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ]
    )[INPUT_COLUMNS]

    proba = float(model.predict_proba(row)[0, 1])
    label = int(proba >= 0.5)
    if proba >= 0.65:
        risk, css = "High risk of churn", "risk-high"
        advice = "Offer a longer contract, add tech support or security, or switch off electronic-check billing."
    elif proba >= 0.40:
        risk, css = "Medium risk", "risk-med"
        advice = "Watch this account. A retention offer or service add-on may reduce risk."
    else:
        risk, css = "Likely to stay", "risk-low"
        advice = "Low churn signal. Keep service quality steady; no urgent intervention."

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Outcome</div><div class="{css}">{risk}</div></div>', unsafe_allow_html=True)
    with k2:
        kpi("Churn probability", f"{proba:.1%}")
    with k3:
        kpi("Predicted class", "Churn" if label else "Retain")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            title={"text": "Churn probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#F43F5E" if proba >= 0.5 else "#34d399"},
                "steps": [
                    {"range": [0, 40], "color": "#064e3b"},
                    {"range": [40, 65], "color": "#78350f"},
                    {"range": [65, 100], "color": "#7f1d1d"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
            },
        )
    )
    gauge.update_layout(**PLOTLY_LAYOUT, height=280)
    st.plotly_chart(gauge, width="stretch")
    st.success(advice)

    importance = load_importance().head(12)
    if not importance.empty:
        fig = px.bar(
            importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top model drivers (global)",
            color="importance",
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")


def page_model(metrics: dict | None, importance: pd.DataFrame) -> None:
    st.title("Model performance")
    if metrics is None:
        st.error("Metrics not found. Run `python train_model.py` first.")
        return

    test = metrics["test"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Best model", str(metrics["best_model"]).replace("_", " ").title())
    with c2:
        kpi("Test ROC-AUC", f"{test['roc_auc']:.3f}")
    with c3:
        kpi("Recall", f"{test['recall']:.3f}")
    with c4:
        kpi("Precision", f"{test['precision']:.3f}")
    with c5:
        kpi("F1 score", f"{test['f1']:.3f}")

    st.subheader("Cross-validation (ROC-AUC)")
    cv_df = pd.DataFrame(
        [
            {"model": name, "mean_roc_auc": vals["mean_roc_auc"], "std_roc_auc": vals["std_roc_auc"]}
            for name, vals in metrics["cv_results"].items()
        ]
    )
    fig = px.bar(
        cv_df,
        x="model",
        y="mean_roc_auc",
        error_y="std_roc_auc",
        title="5-fold stratified CV",
        color="mean_roc_auc",
        color_continuous_scale="Blues",
    )
    fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, yaxis_range=[0.7, 0.9])
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        cm = test["confusion_matrix"]
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Purples",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Retain", "Churn"],
            y=["Retain", "Churn"],
            title="Confusion matrix (test set)",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with right:
        roc = metrics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name="Model", line=dict(color="#F43F5E", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="#64748b")))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"ROC curve (AUC {test['roc_auc']:.3f})",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )
        st.plotly_chart(fig, width="stretch")

    if not importance.empty:
        fig = px.bar(
            importance.head(20).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Feature importance",
            color="importance",
            color_continuous_scale="Agsunset",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=620, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
        **What changed vs the original notebook model**
        - Train/test split is stratified *before* any resampling-style tricks (the original SMOTEENN on the full set leaked labels).
        - A sklearn pipeline handles cleaning, tenure groups, service counts, scaling, and one-hot encoding.
        - Class imbalance is handled with `class_weight='balanced'` instead of oversampling the whole dataset.
        - Logistic Regression, Random Forest, and Gradient Boosting are compared on ROC-AUC; the winner is saved.
        """
    )


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Predict", "Model performance"])
    df = load_data()
    model = load_model()
    metrics = load_metrics()
    importance = load_importance()

    if page == "Overview":
        page_overview(df)
    elif page == "Predict":
        page_predict(model)
    else:
        page_model(metrics, importance)


if __name__ == "__main__":
    main()
