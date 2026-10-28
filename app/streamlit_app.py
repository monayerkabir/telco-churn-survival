"""Telco churn explorer: what-if scoring, portfolio targeting and survival by segment.

Runs standalone. It loads the committed model bundle directly (no API needed), so it can
be deployed to Streamlit Community Cloud with ``app/streamlit_app.py`` as entrypoint.

Environment variables:
    CHURN_MODEL_DIR  model directory (default: model_bundle/)
    CHURN_DATA_PATH  raw Telco CSV for the portfolio and segment views; downloaded and
                     checksum-verified on first use if it does not exist
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):  # Streamlit Cloud runs the script without installing it
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import streamlit as st
from lifelines import KaplanMeierFitter

from app import charts
from churn import business, survival
from churn.bundle import BUNDLE_DIR
from churn.config import (
    ADDON_SERVICE_COLS,
    CATEGORY_LEVELS,
    DURATION_COL,
    ID_COL,
    RAW_DATA_PATH,
    TARGET_COL,
    BusinessConfig,
)
from churn.data import clean_data, download_raw_data, load_raw_data, split_train_test
from churn.inference import FEATURE_COLUMNS, ChurnPredictor

MODEL_DIR = os.environ.get("CHURN_MODEL_DIR", str(BUNDLE_DIR))
DATA_PATH = os.environ.get("CHURN_DATA_PATH", str(RAW_DATA_PATH))
MODEL_LABELS = {  # display names for the training run's model keys
    "logreg": "Logistic regression",
    "lgbm_default": "LightGBM",
    "lgbm_tuned": "LightGBM (Optuna-tuned)",
}
SEGMENTS = [  # attributes with at most four levels (the validated palette's limit)
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "PaperlessBilling",
    "SeniorCitizen",
    "Partner",
    "Dependents",
]
DEFAULT_CUSTOMER = {
    "tenure": 5,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": True,
    "MonthlyCharges": 89.5,
    "InternetService": "Fiber optic",
    "addons": ["OnlineBackup", "StreamingTV"],
    "PhoneService": True,
    "MultipleLines": False,
    "gender": "Female",
    "SeniorCitizen": False,
    "Partner": True,
    "Dependents": False,
}


def money(x: float, markdown: bool = False) -> str:
    """$1,234 / -$56. In markdown "$" must be escaped, or two of them start LaTeX math."""
    sign = "\\$" if markdown else "$"
    return f"-{sign}{abs(x):,.0f}" if x < 0 else f"{sign}{x:,.0f}"


# --------------------------------------------------------------------------------------
# Cached loading
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def load_predictor(model_dir: str) -> ChurnPredictor:
    return ChurnPredictor.load(model_dir)


@st.cache_data(show_spinner="Loading customers…")
def load_customers(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        download_raw_data(dest=path)  # verifies the SHA-256
    return clean_data(load_raw_data(path))


@st.cache_data(show_spinner="Scoring held-out customers…")
def score_portfolio(_predictor: ChurnPredictor, model_version: str, data_path: str) -> pd.DataFrame:
    """Hold-out customers (same split as training) with their model scores.

    ``_predictor`` is not hashed by Streamlit (leading underscore), so ``model_version``
    is what invalidates the cache when a different model is loaded.
    """
    _, test = split_train_test(load_customers(data_path))
    scores = _predictor.score(_predictor.prepare(test[FEATURE_COLUMNS]))
    return pd.concat([test, scores], axis=1)


@st.cache_data(show_spinner="Computing survival curves…")
def segment_curves(
    _predictor: ChurnPredictor, model_version: str, data_path: str, segment: str, source: str
) -> pd.DataFrame:
    """Long frame (level, month, survival) of observed or predicted curves per level.

    ``model_version`` is the cache key for the unhashed ``_predictor``.
    """
    customers = load_customers(data_path)
    rows = []
    if source == "observed":
        df = survival.survival_frame(customers)
        for level in CATEGORY_LEVELS[segment]:
            sub = df[df[segment] == level]
            if sub.empty:
                continue
            km = KaplanMeierFitter().fit(sub[DURATION_COL], sub[TARGET_COL])
            curve = km.survival_function_at_times(survival.TIME_GRID).to_numpy()
            rows.append(
                pd.DataFrame(
                    {"level": level, "month": survival.TIME_GRID, "survival": curve, "n": len(sub)}
                )
            )
    else:
        _, test = split_train_test(customers)
        curves = _predictor.survival_curves(_predictor.prepare(test[FEATURE_COLUMNS]))
        for level in CATEGORY_LEVELS[segment]:
            mask = (test[segment] == level).to_numpy()
            if not mask.any():
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "level": level,
                        "month": survival.TIME_GRID,
                        "survival": curves[mask].mean(axis=0),
                        "n": int(mask.sum()),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------
# What-if
# --------------------------------------------------------------------------------------
def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


def customer_form() -> dict:
    """Inputs for one customer, returned in the API/inference record format."""
    for key, value in DEFAULT_CUSTOMER.items():
        st.session_state.setdefault(f"in_{key}", value)

    st.markdown("**Account**")
    tenure = st.slider("Tenure (months)", 0, 72, key="in_tenure")
    contract = st.selectbox("Contract", CATEGORY_LEVELS["Contract"], key="in_Contract")
    payment = st.selectbox(
        "Payment method", CATEGORY_LEVELS["PaymentMethod"], key="in_PaymentMethod"
    )
    paperless = st.toggle("Paperless billing", key="in_PaperlessBilling")
    monthly = st.number_input(
        "Monthly charges ($)", min_value=1.0, max_value=500.0, step=5.0, key="in_MonthlyCharges"
    )

    st.markdown("**Services**")
    internet = st.selectbox(
        "Internet service", CATEGORY_LEVELS["InternetService"], key="in_InternetService"
    )
    no_internet = internet == "No"
    addons = st.multiselect(
        "Add-on services",
        ADDON_SERVICE_COLS,
        key="in_addons",
        disabled=no_internet,
        help="Unavailable without internet service.",
    )
    phone = st.toggle("Phone service", key="in_PhoneService")
    multiple = st.toggle("Multiple lines", key="in_MultipleLines", disabled=not phone)

    st.markdown("**Household**")
    gender = st.radio("Gender", CATEGORY_LEVELS["gender"], horizontal=True, key="in_gender")
    senior = st.toggle("Senior citizen", key="in_SeniorCitizen")
    partner = st.toggle("Has a partner", key="in_Partner")
    dependents = st.toggle("Has dependents", key="in_Dependents")

    record = {
        "gender": gender,
        "SeniorCitizen": yes_no(senior),
        "Partner": yes_no(partner),
        "Dependents": yes_no(dependents),
        "tenure": tenure,
        "PhoneService": yes_no(phone),
        "MultipleLines": yes_no(multiple) if phone else "No phone service",
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": yes_no(paperless),
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
    }
    for col in ADDON_SERVICE_COLS:
        record[col] = "No internet service" if no_internet else yes_no(col in addons)
    return record


def what_if_tab(predictor: ChurnPredictor, pal: charts.Palette) -> None:
    st.caption(
        "Change any attribute and the prediction updates. Pin a scenario to compare against it."
    )
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        customer = customer_form()

    (pred,) = predictor.predict([customer], top_k=8, include_curve=True)
    baseline = st.session_state.get("baseline")

    with right:
        p = pred["churn_probability"]
        surv = pred["survival"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Churn probability",
            f"{p:.0%}",
            delta=f"{(p - baseline['churn_probability']) * 100:+.0f} pts" if baseline else None,
            delta_color="inverse",
            help="Classifier score (calibrated LightGBM), on the dataset's churn label.",
        )
        k2.metric(
            "Revenue at risk",
            money(pred["revenue_at_risk"]["amount"]),
            help="P(churn) × monthly charges × 12 months.",
        )
        p12 = surv["churn_probability_12m"]
        k3.metric(
            "12-month churn",
            "n/a" if p12 is None else f"{p12:.0%}",
            help=(
                "Survival model: P(churn in the next 12 months | active today). It answers a "
                "different question from the classifier, so the two numbers differ; they rank "
                "customers similarly. n/a beyond the 72-month window the data covers."
            ),
        )
        k4.metric(
            "Remaining tenure",
            f"{surv['expected_remaining_tenure_months']:.0f} mo",
            help=(
                "Expected months until churn, restricted to the 72-month observation window "
                "(the curves cannot be extrapolated beyond it)."
            ),
        )

        decision = pred["decision"]
        verdict = (
            "expected profit of the offer: "
            f"**{money(decision['expected_profit_if_targeted'], markdown=True)}** "
            f"(threshold {decision['threshold']:.2f}; breakeven for this customer "
            f"{decision['breakeven_probability']:.0%})"
        )
        if decision["target"]:
            st.success(f"**Target with a retention offer**: {verdict}", icon=":material/campaign:")
        else:
            st.info(f"**Don't target**: {verdict}", icon=":material/do_not_disturb_on:")

        st.markdown("**Survival from today**")
        curve = pd.DataFrame(surv["curve"])
        base_curve = pd.DataFrame(baseline["survival"]["curve"]) if baseline else None
        st.altair_chart(charts.survival_curve(curve, pal, base_curve), width="stretch")

        st.markdown("**Why this score? Top drivers (SHAP)**")
        drivers = pd.DataFrame(pred["explanation"]["top_drivers"])
        st.altair_chart(charts.shap_drivers(drivers, pal), width="stretch")
        st.caption(
            "Contributions to the classifier's log-odds relative to an average customer; "
            f"the remaining attributes add {pred['explanation']['other_features_contribution']:+.2f}."
        )

        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("Pin as baseline", icon=":material/push_pin:"):
            st.session_state["baseline"] = pred
            st.rerun()
        if b2.button("Clear baseline", disabled=baseline is None):
            st.session_state.pop("baseline", None)
            st.rerun()


# --------------------------------------------------------------------------------------
# Portfolio
# --------------------------------------------------------------------------------------
def portfolio_tab(predictor: ChurnPredictor, pal: charts.Palette) -> None:
    try:
        port = score_portfolio(predictor, predictor.version, DATA_PATH)
    except Exception as exc:  # noqa: BLE001 - surface any data problem in the UI
        st.error(
            f"Could not load the customer data ({exc}). Set CHURN_DATA_PATH or allow downloads."
        )
        return

    st.caption(
        f"{len(port):,} held-out customers the models never saw in training. Their outcomes "
        "are known, so the profit below is realised on real churn, not forecast."
    )
    cfg = predictor.business_config
    c1, c2, c3, c4 = st.columns(4)
    threshold = c1.slider("Targeting threshold", 0.0, 1.0, predictor.threshold, 0.01)
    offer_cost = c2.number_input("Offer cost per customer ($)", 0.0, 500.0, cfg.offer_cost, 5.0)
    success = c3.slider("Offer success rate", 0.0, 1.0, cfg.offer_success_rate, 0.05)
    horizon = c4.select_slider("Value horizon (months)", [6, 12, 24], cfg.horizon_months)
    bcfg = BusinessConfig(horizon_months=horizon, offer_cost=offer_cost, offer_success_rate=success)

    y, p, charges = port[TARGET_COL], port["p_churn"], port["MonthlyCharges"]
    curve = business.profit_curve(y, p, charges, cfg=bcfg)
    at = curve.iloc[(curve["threshold"] - threshold).abs().idxmin()]
    everyone = curve.iloc[0]
    best = business.optimal_threshold(curve)
    rar = business.revenue_at_risk(p, charges, horizon)
    targeted = (p >= threshold).to_numpy()
    value_aware = business.value_aware_targets(p, charges, bcfg)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Customers targeted",
        f"{int(at['n_targeted']):,}",
        f"{at['share_targeted']:.0%} of base",
        delta_color="off",
        delta_arrow="off",
    )
    k2.metric(
        "Churners reached",
        f"{int(at['churners_reached']):,} of {int(y.sum()):,}",
        f"{at['churners_reached'] / max(y.sum(), 1):.0%}",
        delta_color="off",
        delta_arrow="off",
    )
    k3.metric(
        "Expected profit",
        money(at["profit"]),
        f"{money(at['profit'] - everyone['profit'])} vs offering everyone",
    )
    k4.metric(
        "Revenue at risk covered",
        f"{rar[targeted].sum() / max(rar.sum(), 1e-9):.0%}",
        f"of {money(rar.sum())} over {horizon} months",
        delta_color="off",
        delta_arrow="off",
    )

    st.altair_chart(charts.profit_curve(curve, threshold, pal), width="stretch")
    value_aware_profit = business.policy_profit(y, value_aware, charges, bcfg)
    st.caption(
        f"Threshold chosen on training data: **{predictor.threshold:.2f}** (for "
        f"{money(cfg.offer_cost, markdown=True)} offers, {cfg.offer_success_rate:.0%} success, "
        f"{cfg.horizon_months}-month value). Best on these customers in hindsight: "
        f"**{best['threshold']:.2f}** ({money(best['profit'], markdown=True)}). Targeting each "
        "customer whose own expected profit is positive would earn "
        f"**{money(value_aware_profit, markdown=True)}** with {int(value_aware.sum()):,} offers."
    )

    st.markdown("**Targeted customers, by revenue at risk**")
    table = (
        port.loc[targeted]
        .assign(revenue_at_risk=rar[targeted])
        .sort_values("revenue_at_risk", ascending=False)
    )
    view = table[
        [
            ID_COL,
            "Contract",
            DURATION_COL,
            "MonthlyCharges",
            "p_churn",
            "revenue_at_risk",
            "p_churn_12m_survival",
            "expected_remaining_tenure",
        ]
    ].rename(
        columns={
            ID_COL: "Customer",
            DURATION_COL: "Tenure (mo)",
            "MonthlyCharges": "Monthly ($)",
            "p_churn": "P(churn)",
            "revenue_at_risk": "Revenue at risk ($)",
            "p_churn_12m_survival": "Churn ≤ 12 mo (survival)",
            "expected_remaining_tenure": "Remaining tenure (mo)",
        }
    )
    st.dataframe(
        view.head(25),
        hide_index=True,
        column_config={
            "P(churn)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Churn ≤ 12 mo (survival)": st.column_config.NumberColumn(format="percent"),
            "Revenue at risk ($)": st.column_config.NumberColumn(format="dollar"),
            "Monthly ($)": st.column_config.NumberColumn(format="dollar"),
            "Remaining tenure (mo)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.download_button(
        f"Download all {int(targeted.sum()):,} targeted customers (CSV)",
        view.to_csv(index=False).encode(),
        file_name=f"retention_targets_t{threshold:.2f}.csv",
        mime="text/csv",
        icon=":material/download:",
    )


# --------------------------------------------------------------------------------------
# Survival by segment
# --------------------------------------------------------------------------------------
def segments_tab(predictor: ChurnPredictor, pal: charts.Palette) -> None:
    c1, c2 = st.columns([1, 2])
    segment = c1.selectbox("Segment by", SEGMENTS)
    source = c2.segmented_control(
        "Curves",
        ["observed", "model"],
        default="observed",
        format_func=lambda s: {
            "observed": "Observed (Kaplan-Meier, all customers)",
            "model": "Model (average prediction, held-out customers)",
        }[s],
    )
    try:
        curves = segment_curves(
            predictor, predictor.version, DATA_PATH, segment, source or "observed"
        )
    except Exception as exc:  # noqa: BLE001 - surface any data problem in the UI
        st.error(
            f"Could not load the customer data ({exc}). Set CHURN_DATA_PATH or allow downloads."
        )
        return

    st.altair_chart(charts.segment_survival(curves, segment, pal), width="stretch")

    table = curves.pivot_table(index="level", columns="month", values="survival", sort=False)
    summary = pd.DataFrame(
        {
            "customers": curves.groupby("level", sort=False)["n"].first(),
            **{f"active at {m} mo": table[m] for m in (12, 24, 48, 72)},
            "median lifetime": [
                next((f"{int(m)} mo" for m, s in row.items() if s <= 0.5), "not reached")
                for _, row in table.iterrows()
            ],
        }
    ).rename_axis(segment)
    st.dataframe(
        summary,
        column_config={
            **{
                f"active at {m} mo": st.column_config.NumberColumn(format="percent")
                for m in (12, 24, 48, 72)
            },
            "median lifetime": st.column_config.TextColumn(
                help="'not reached' when more than half are still active at 72 months."
            ),
        },
    )
    st.caption(
        "Observed curves are Kaplan-Meier estimates (active customers are censored at their "
        f"current tenure). Model curves average the {predictor.survival_model_name} "
        "predictions of held-out customers from month 0."
    )


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------
def sidebar(predictor: ChurnPredictor) -> None:
    m = predictor.metrics
    test = m["classification"]["test"]["final"]
    surv = m["survival"]["models"][predictor.survival_model_name]
    cfg = predictor.business_config
    with st.sidebar:
        st.markdown("### Model card")
        st.markdown(
            f"- **Classifier:** {MODEL_LABELS.get(predictor.champion, predictor.champion)}, test ROC-AUC "
            f"{test['roc_auc']['value']:.3f}\n"
            f"- **Survival:** {predictor.survival_model_name}, C-index {surv['c_index']:.3f}\n"
            f"- **Retention threshold:** {predictor.threshold:.2f}\n"
            f"- **Assumptions:** ${cfg.offer_cost:.0f} offer, {cfg.offer_success_rate:.0%} "
            f"success, {cfg.horizon_months}-month value"
        )
        st.caption(f"Model version `{predictor.version}`")
        st.caption(
            "Data: IBM Telco Customer Churn (7,043 customers). Predictions are for "
            "illustration on a public, semi-synthetic dataset."
        )


def main() -> None:
    st.set_page_config(
        page_title="Telco churn explorer", page_icon=":material/insights:", layout="wide"
    )
    predictor = load_predictor(MODEL_DIR)
    pal = charts.palette(st.context.theme.type)

    st.title("Telco churn explorer")
    st.markdown("Who is likely to churn, **when**, and is a retention offer worth it?")
    sidebar(predictor)

    what_if, portfolio, segments = st.tabs(
        ["Customer what-if", "Portfolio targeting", "Survival by segment"]
    )
    with what_if:
        what_if_tab(predictor, pal)
    with portfolio:
        portfolio_tab(predictor, pal)
    with segments:
        segments_tab(predictor, pal)


main()
