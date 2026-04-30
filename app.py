"""
Bank Marketing – Interactive Prediction Dashboard
==================================================
Launch:  streamlit run app.py
"""

import pathlib, pickle, hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ── Paths ──────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "bank-additional-full.csv"
VIS_DIR = BASE / "visualizations"
CACHE_PKL = BASE / "models_cache.pkl"

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing – ML Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  DATA & MODEL TRAINING  (cached so it runs only once)
# ══════════════════════════════════════════════════════════════════


def _data_hash() -> str:
    return hashlib.md5(DATA_PATH.read_bytes()).hexdigest()


@st.cache_resource(show_spinner="Training all 9 models … please wait ⏳")
def load_and_train(_data_hash: str):
    df = pd.read_csv(DATA_PATH, sep=";")
    df["y_binary"] = df["y"].map({"no": 0, "yes": 1})

    features = df.drop(columns=["y", "y_binary"])
    target = df["y_binary"]

    cat_cols = features.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "enc",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model_defs = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
        "Voting Ensemble": VotingClassifier(
            estimators=[
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=42
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
                ("gb", GradientBoostingClassifier(random_state=42)),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.001,
                        learning_rate="adaptive",
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=42,
                    ),
                ),
            ],
            voting="soft",
            n_jobs=-1,
        ),
        "Stacking Ensemble": StackingClassifier(
            estimators=[
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=42
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
                ("gb", GradientBoostingClassifier(random_state=42)),
            ],
            final_estimator=LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ),
            cv=3,
            n_jobs=-1,
        ),
    }

    pipelines = {}
    metrics = {}
    for name, clf in model_defs.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = (
            pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        )
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1-Score": f1_score(y_test, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba) if proba is not None else 0,
        }
        pipelines[name] = pipe

    metrics_df = pd.DataFrame(metrics).T.sort_values("F1-Score", ascending=False)
    return df, pipelines, metrics_df, cat_cols, num_cols


df, pipelines, metrics_df, cat_cols, num_cols = load_and_train(_data_hash())


# ══════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════
MODEL_COLOURS = {
    name: colour
    for name, colour in zip(
        metrics_df.index,
        [
            "#1b9e77",
            "#d95f02",
            "#7570b3",
            "#e7298a",
            "#66a61e",
            "#e6ab02",
            "#a6761d",
            "#2166ac",
            "#b2182b",
        ],
    )
}


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR – CLIENT INPUT
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=60)
    st.title("🏦 Client Profile")
    st.markdown(
        "Fill in client details to generate a **live prediction** across all 9 models."
    )
    st.divider()

    # Demographic
    st.subheader("👤 Demographics")
    age = st.slider("Age", 17, 98, 38)
    job = st.selectbox("Job", sorted(df["job"].unique()))
    marital = st.selectbox("Marital Status", sorted(df["marital"].unique()))
    education = st.selectbox("Education", sorted(df["education"].unique()))

    # Financial
    st.subheader("💳 Financial Status")
    default = st.selectbox("Credit Default?", ["no", "yes", "unknown"])
    housing = st.selectbox("Housing Loan?", ["no", "yes", "unknown"])
    loan = st.selectbox("Personal Loan?", ["no", "yes", "unknown"])

    # Campaign
    st.subheader("📞 Campaign Details")
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])
    month = st.selectbox(
        "Last Contact Month",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
    )
    day_of_week = st.selectbox("Last Contact Day", ["mon", "tue", "wed", "thu", "fri"])
    duration = st.slider("Call Duration (sec)", 0, 5000, 180)
    campaign = st.slider("Contacts This Campaign", 1, 56, 2)
    pdays = st.slider("Days Since Previous Contact (999 = never)", 0, 999, 999)
    previous = st.slider("Previous Campaign Contacts", 0, 7, 0)
    poutcome = st.selectbox("Previous Outcome", ["nonexistent", "failure", "success"])

    # Economic
    st.subheader("📈 Economic Indicators")
    emp_var_rate = st.number_input(
        "Employment Variation Rate", -3.5, 1.5, 1.1, step=0.1
    )
    cons_price_idx = st.number_input(
        "Consumer Price Index", 92.0, 95.0, 93.75, step=0.01
    )
    cons_conf_idx = st.number_input(
        "Consumer Confidence Index", -51.0, -26.0, -41.8, step=0.1
    )
    euribor3m = st.number_input("Euribor 3-Month Rate", 0.6, 5.1, 4.86, step=0.01)
    nr_employed = st.number_input(
        "No. Employed (quarterly)", 4960.0, 5230.0, 5191.0, step=1.0
    )

# Build a single-row DataFrame matching the training schema
input_row = pd.DataFrame(
    [
        {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "nr.employed": nr_employed,
        }
    ]
)


# ══════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='text-align:center;'>🏦 Bank Marketing – ML Prediction Dashboard</h1>"
    "<p style='text-align:center;color:grey;'>Predict term-deposit subscription using 9 machine-learning models</p>",
    unsafe_allow_html=True,
)

# ── KPI banner ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Dataset Records", f"{len(df):,}")
k2.metric("Features Used", len(num_cols) + len(cat_cols))
k3.metric("Models Trained", len(pipelines))
k4.metric("Best F1 Model", metrics_df.index[0])

st.divider()

# ── Tabs ───────────────────────────────────────────────────────
tab_pred, tab_compare, tab_data = st.tabs(
    ["🔮  Live Prediction", "📊  Model Comparison", "📁  Dataset Explorer"]
)

# ───────────── TAB 1 : LIVE PREDICTION ─────────────────────────
with tab_pred:
    st.subheader("Multi-Model Prediction Results")
    st.markdown(
        "Based on the client profile set in the **sidebar**, each model predicts the probability of subscription."
    )

    predictions = {}
    for name, pipe in pipelines.items():
        prob = pipe.predict_proba(input_row)[0]  # [P(no), P(yes)]
        predictions[name] = {
            "Subscribe %": round(prob[1] * 100, 2),
            "Not Subscribe %": round(prob[0] * 100, 2),
            "Prediction": "✅ Yes" if prob[1] >= 0.5 else "❌ No",
        }

    pred_df = pd.DataFrame(predictions).T
    pred_df = pred_df.sort_values("Subscribe %", ascending=False)

    # Gauge-style horizontal bar
    fig_bar = go.Figure()
    for name in pred_df.index:
        fig_bar.add_trace(
            go.Bar(
                y=[name],
                x=[pred_df.loc[name, "Subscribe %"]],
                orientation="h",
                marker_color=MODEL_COLOURS.get(name, "#888"),
                text=f"{pred_df.loc[name, 'Subscribe %']:.1f}%",
                textposition="auto",
                name=name,
                showlegend=False,
            )
        )
    fig_bar.update_layout(
        title="Subscription Probability by Model",
        xaxis=dict(title="Probability (%)", range=[0, 100]),
        height=420,
        margin=dict(l=10, r=10, t=50, b=30),
        yaxis=dict(autorange="reversed"),
    )
    # Add a 50% decision threshold line
    fig_bar.add_vline(
        x=50,
        line_dash="dash",
        line_color="red",
        annotation_text="50% threshold",
        annotation_position="top right",
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # Summary cards
    yes_count = sum(1 for v in predictions.values() if v["Prediction"] == "✅ Yes")
    no_count = len(predictions) - yes_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Models Predicting YES", f"{yes_count} / {len(predictions)}")
    c2.metric("Models Predicting NO", f"{no_count} / {len(predictions)}")
    majority = "✅ Subscribe" if yes_count > no_count else "❌ Not Subscribe"
    c3.metric("Majority Vote", majority)

    # Detailed table
    with st.expander("📋 Detailed Prediction Table", expanded=False):
        st.dataframe(
            pred_df.style.format(
                {"Subscribe %": "{:.2f}", "Not Subscribe %": "{:.2f}"}
            ),
            use_container_width=True,
        )

    # Radar chart of the top-probability model's test metrics
    st.markdown("---")
    st.subheader("🎯 Test-Set Performance of the Top-Predicting Model")
    top_model = pred_df.index[0]
    top_metrics = metrics_df.loc[top_model]
    radar_metrics = list(top_metrics.index)
    radar_vals = top_metrics.values.tolist() + [top_metrics.values[0]]
    radar_labels = radar_metrics + [radar_metrics[0]]
    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=radar_vals,
            theta=radar_labels,
            fill="toself",
            name=top_model,
            line_color=MODEL_COLOURS.get(top_model, "#1b9e77"),
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1])),
        title=f"Metric Profile: {top_model}",
        height=400,
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ───────────── TAB 2 : MODEL COMPARISON ────────────────────────
with tab_compare:
    st.subheader("📊 Model Performance Comparison (Test Set)")

    # Heatmap
    fig_heat = px.imshow(
        metrics_df.values,
        x=metrics_df.columns.tolist(),
        y=metrics_df.index.tolist(),
        color_continuous_scale="YlGnBu",
        zmin=0,
        zmax=1,
        text_auto=".3f",
        aspect="auto",
        title="Performance Heatmap",
    )
    fig_heat.update_layout(height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Grouped bar
    fig_group = go.Figure()
    for metric in metrics_df.columns:
        fig_group.add_trace(
            go.Bar(
                x=metrics_df.index,
                y=metrics_df[metric],
                name=metric,
            )
        )
    fig_group.update_layout(
        barmode="group",
        title="Metrics Comparison",
        yaxis=dict(range=[0, 1.05]),
        height=450,
    )
    st.plotly_chart(fig_group, use_container_width=True)

    # Scatter F1 vs ROC-AUC
    st.markdown("#### F1-Score vs ROC-AUC")
    scatter_df = metrics_df.reset_index().rename(columns={"index": "Model"})
    fig_sc = px.scatter(
        scatter_df,
        x="ROC-AUC",
        y="F1-Score",
        text="Model",
        color="Model",
        color_discrete_sequence=list(MODEL_COLOURS.values()),
        size="Accuracy",
        size_max=22,
        title="F1-Score vs ROC-AUC (bubble size = Accuracy)",
    )
    fig_sc.update_traces(textposition="top center")
    fig_sc.update_layout(height=480)
    st.plotly_chart(fig_sc, use_container_width=True)

    # Sortable table
    with st.expander("📋 Full Metrics Table", expanded=True):
        st.dataframe(
            metrics_df.style.format("{:.4f}").background_gradient(
                cmap="YlGnBu", axis=None
            ),
            use_container_width=True,
        )

    # Existing PNG charts
    st.markdown("---")
    st.subheader("📸 Training Visualisations")
    img_map = {
        "Target Distribution": "01_target_distribution.png",
        "Numeric Features": "02_numeric_features_distribution.png",
        "Categorical Features": "03_categorical_features_distribution.png",
        "Model Comparison": "04_model_comparison.png",
        "Confusion Matrices": "05_confusion_matrices.png",
        "ROC Curves": "06_roc_curves.png",
        "Feature Importance": "07_feature_importance.png",
        "Correlation Heatmap": "08_correlation_heatmap.png",
        "Class Imbalance": "09_class_imbalance.png",
        "Precision vs Recall": "10_precision_recall_bar.png",
        "F1 vs ROC-AUC Scatter": "11_f1_roc_scatter.png",
        "Radar Chart": "12_radar_chart.png",
    }
    selected_img = st.selectbox("Select a chart", list(img_map.keys()))
    img_path = VIS_DIR / img_map[selected_img]
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info(f"Image not found: {img_path.name}")


# ───────────── TAB 3 : DATASET EXPLORER ────────────────────────
with tab_data:
    st.subheader("📁 Dataset Explorer")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Total Records:** {len(df):,}")
        st.markdown(f"**Total Features:** {df.shape[1] - 1}")
    with col_b:
        vc = df["y"].value_counts()
        st.markdown(
            f"**No (non-subscribe):** {vc['no']:,}  ({vc['no']/len(df)*100:.1f}%)"
        )
        st.markdown(
            f"**Yes (subscribe):** {vc['yes']:,}  ({vc['yes']/len(df)*100:.1f}%)"
        )

    st.dataframe(df.head(200), use_container_width=True, height=400)

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:grey;font-size:0.85em;'>"
    "BM 173 – Applications of Machine Learning | Bank Marketing Classification Dashboard"
    "</p>",
    unsafe_allow_html=True,
)
