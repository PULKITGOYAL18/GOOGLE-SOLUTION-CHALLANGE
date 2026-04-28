import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG — force light theme via CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FairML Pipeline",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — always light, device-responsive
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── force light background everywhere ── */
html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background-color: #F8FAFC !important;
    color: #1E293B !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1E3A5F 0%, #2563EB 100%) !important;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #E0ECFF !important;
    font-weight: 600;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255,255,255,0.15) !important;
    border-radius: 8px;
}

/* ── headings ── */
h1 { color: #1E3A5F !important; font-weight: 800 !important; }
h2 { color: #1E3A5F !important; font-weight: 700 !important; }
h3 { color: #2563EB !important; font-weight: 600 !important; }

/* ── metric cards ── */
[data-testid="stMetricValue"] { color: #1E3A5F !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #64748B !important; }
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

/* ── dataframes ── */
[data-testid="stDataFrame"] { color: #1E293B !important; }
.stDataFrame th { background: #EFF6FF !important; color: #1E3A5F !important; }
.stDataFrame td { color: #334155 !important; }

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 10px rgba(37,99,235,0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important;
    box-shadow: 0 5px 16px rgba(37,99,235,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 10px rgba(5,150,105,0.3) !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #047857, #065F46) !important;
    box-shadow: 0 5px 16px rgba(5,150,105,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── expanders ── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #1E3A5F !important;
    font-weight: 600 !important;
}

/* ── success / warning / error ── */
[data-testid="stAlert"] { border-radius: 10px !important; }
.stSuccess { background: #ECFDF5 !important; color: #065F46 !important; border-left: 4px solid #059669 !important; }
.stWarning { background: #FFFBEB !important; color: #92400E !important; border-left: 4px solid #F59E0B !important; }
.stError   { background: #FEF2F2 !important; color: #991B1B !important; border-left: 4px solid #EF4444 !important; }
.stInfo    { background: #EFF6FF !important; color: #1E40AF !important; border-left: 4px solid #2563EB !important; }

/* ── inputs ── */
[data-testid="stSelectbox"] select,
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput input {
    background: #FFFFFF !important;
    color: #1E293B !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
}
label[data-testid="stWidgetLabel"] {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: #EFF6FF !important;
    border: 2px dashed #93C5FD !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] * { color: #1E3A5F !important; }

/* ── progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #2563EB, #60A5FA) !important;
}

/* ── tabs ── */
[data-testid="stTab"] {
    background: #FFFFFF !important;
    color: #1E3A5F !important;
    font-weight: 600 !important;
}
[data-testid="stTab"][aria-selected="true"] {
    border-bottom: 3px solid #2563EB !important;
    color: #2563EB !important;
}

/* ── cards / info boxes ── */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 10px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    color: #1E293B;
}
.stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin: 12px 0;
}
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.badge-green  { background: #DCFCE7; color: #166534; }
.badge-yellow { background: #FEF9C3; color: #854D0E; }
.badge-red    { background: #FEE2E2; color: #991B1B; }
.badge-blue   { background: #DBEAFE; color: #1E40AF; }

/* ── section divider ── */
.section-header {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border-left: 5px solid #2563EB;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    margin: 20px 0 14px 0;
    color: #1E3A5F;
    font-weight: 700;
    font-size: 1.05rem;
}

/* ── responsive tweaks ── */
@media (max-width: 768px) {
    .main .block-container { padding: 1rem !important; }
    .stat-row { flex-direction: column; }
}

/* ── hide streamlit branding ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section(title, icon=""):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

def badge(text, color="blue"):
    return f'<span class="badge badge-{color}">{text}</span>'

def card(content):
    st.markdown(f'<div class="info-card">{content}</div>', unsafe_allow_html=True)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ FairML Pipeline")
    st.markdown("*Google Solution Challenge*")
    st.markdown('DeepThinkersAI')
    st.markdown("Pulkit Goyal, Mohd Raza, Sudhanshu Bhatt, Pragya Chauhan")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", [
        "🏠 Home",
        "📤 Upload & Configure",
        "🔍 Bias Detection",
        "🤖 Model Training",
        "📊 Fairness Analysis",
        "💡 SHAP Explainability",
        "✨ Gemini AI Assistant",
        "📄 Export Report",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.78rem;opacity:0.85;line-height:1.6">'
        '✅ Auto feature selection<br>'
        '✅ SMOTE decision pipeline<br>'
        '✅ Reweighting fairness<br>'
        '✅ SHAP explainability<br>'
        '✨ Gemini AI insights<br>'
        '✅ PDF report export'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key in ["df", "target", "sensitive_features", "X", "y", "weights",
            "X_train", "X_test", "y_train", "y_test", "w_train", "w_test",
            "X_train_enc", "X_test_enc", "final_model", "y_pred_final",
            "fairness_df", "shap_values", "shap_values_2d", "shap_sample_X",
            "x_train_final", "x_test_final",
            "choice", "accuracy", "precision", "recall", "f1",
            "comparison_df", "y_train_final_smote", "strategy",
            "acc_no", "f1_no", "acc_sm", "f1_sm",
            "gemini_api_key", "chat_history", "gemini_report_text"]:
    if key not in st.session_state:
        st.session_state[key] = None

# chat_history needs to be a list, not None
if st.session_state["chat_history"] is None:
    st.session_state["chat_history"] = []


# ══════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# ⚖️ Fair, Explainable & Optimized ML Pipeline")
    st.markdown(
        "**An end-to-end automated machine learning system with built-in fairness, "
        "explainability, and bias mitigation — built for the Google Solution Challenge.**"
    )
    st.markdown("")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        card("<h3 style='color:#2563EB;margin:0'>🔍</h3><b style='color:#1E3A5F'>Bias Detection</b><br><small style='color:#64748B'>Automatic data-level bias check across sensitive groups</small>")
    with col2:
        card("<h3 style='color:#059669;margin:0'>⚖️</h3><b style='color:#1E3A5F'>Reweighting</b><br><small style='color:#64748B'>Sample reweighting to balance under-represented groups</small>")
    with col3:
        card("<h3 style='color:#D97706;margin:0'>🤖</h3><b style='color:#1E3A5F'>Auto ML</b><br><small style='color:#64748B'>SMOTE decision + feature selection + RF training</small>")
    with col4:
        card("<h3 style='color:#7C3AED;margin:0'>💡</h3><b style='color:#1E3A5F'>SHAP XAI</b><br><small style='color:#64748B'>Model explainability with SHAP values & insights</small>")
    with col5:
        card("<h3 style='color:#0891B2;margin:0'>✨</h3><b style='color:#1E3A5F'>Gemini AI</b><br><small style='color:#64748B'>AI-powered bias explainer, SHAP insights & smart chat</small>")

    st.markdown("")
    st.markdown("### 🗺️ Pipeline Overview")
    steps = [
        ("1", "Upload CSV", "Upload your dataset", "#2563EB"),
        ("2", "Configure", "Set target & sensitive features", "#7C3AED"),
        ("3", "Bias Check", "Pre-training bias analysis", "#D97706"),
        ("4", "Train Model", "SMOTE + feature selection + RF", "#059669"),
        ("5", "Fairness", "Post-training fairness metrics", "#DC2626"),
        ("6", "SHAP", "Explainability analysis", "#0891B2"),
        ("7", "Gemini AI", "AI insights & smart chat", "#0891B2"),
        ("8", "Report", "Export full PDF report", "#65A30D"),
    ]
    cols = st.columns(len(steps))
    for col, (num, title, desc, color) in zip(cols, steps):
        col.markdown(
            f'<div style="text-align:center;background:#FFFFFF;border:1px solid #E2E8F0;'
            f'border-top:4px solid {color};border-radius:10px;padding:14px 8px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.05)">'
            f'<div style="width:32px;height:32px;background:{color};color:#fff;'
            f'border-radius:50%;font-weight:800;line-height:32px;margin:0 auto 8px">{num}</div>'
            f'<b style="color:#1E3A5F;font-size:0.85rem">{title}</b><br>'
            f'<small style="color:#64748B">{desc}</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info("👈 Use the sidebar to navigate. Start with **📤 Upload & Configure**.")


# ══════════════════════════════════════════════
# PAGE: UPLOAD & CONFIGURE
# ══════════════════════════════════════════════
elif page == "📤 Upload & Configure":
    st.markdown("# 📤 Upload & Configure")

    section("Dataset Upload", "📁")
    uploaded = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="The same CSV used in the notebook (e.g., loan approval dataset)"
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["df"] = df
            st.success(f"✅ Dataset loaded — **{df.shape[0]:,} rows × {df.shape[1]} columns**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")

            with st.expander("👀 Preview Dataset (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)

            with st.expander("📊 Column Info"):
                info_df = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str).values,
                    "Nulls": df.isnull().sum().values,
                    "Unique": df.nunique().values
                })
                st.dataframe(info_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading file: {e}")

    if st.session_state["df"] is not None:
        df = st.session_state["df"]

        section("Column Configuration", "⚙️")
        col1, col2 = st.columns(2)

        with col1:
            target = st.selectbox(
                "🎯 Target Column",
                options=df.columns.tolist(),
                index=df.columns.tolist().index("Loan_Status") if "Loan_Status" in df.columns else 0,
                help="Column you want the model to predict"
            )
        with col2:
            default_sensitive = [c for c in ["Gender", "Property_Area", "Education", "Married"] if c in df.columns]
            sensitive_features = st.multiselect(
                "🔒 Sensitive Features",
                options=[c for c in df.columns if c != target],
                default=default_sensitive,
                help="Columns representing demographic attributes (for fairness analysis)"
            )

        if st.button("✅ Confirm Configuration & Preprocess"):
            if not sensitive_features:
                st.warning("Please select at least one sensitive feature.")
            else:
                with st.spinner("Preprocessing data..."):
                    from sklearn.preprocessing import LabelEncoder

                    df_proc = df.copy()

                    # Fill missing
                    for col in df_proc.select_dtypes(include=np.number).columns:
                        df_proc[col] = df_proc[col].fillna(df_proc[col].mean())
                    for col in df_proc.select_dtypes(include="object").columns:
                        df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0])

                    # Encode target
                    le = LabelEncoder()
                    df_proc[target] = le.fit_transform(df_proc[target])

                    X = df_proc.drop(columns=[target])
                    y = df_proc[target]
                    X_fair = X.copy()

                    # Sample weights
                    weights = np.ones(len(X))
                    for col in sensitive_features:
                        if col in X.columns:
                            freq = X[col].value_counts(normalize=True)
                            weights *= X[col].map(lambda x: 1 / freq.get(x, 1))
                    weights = weights / np.mean(weights)

                    st.session_state.update({
                        "df": df_proc,
                        "target": target,
                        "sensitive_features": sensitive_features,
                        "X": X,
                        "y": y,
                        "weights": weights,
                    })

                st.success("✅ Configuration saved. Proceed to **Bias Detection**.")
                col1, col2, col3 = st.columns(3)
                col1.metric("Target", target)
                col2.metric("Sensitive Features", len(sensitive_features))
                col3.metric("Features", X.shape[1])


# ══════════════════════════════════════════════
# PAGE: BIAS DETECTION
# ══════════════════════════════════════════════
elif page == "🔍 Bias Detection":
    st.markdown("# 🔍 Pre-Training Bias Detection")

    if st.session_state["X"] is None:
        st.warning("Please complete **Upload & Configure** first.")
        st.stop()

    X = st.session_state["X"]
    y = st.session_state["y"]
    weights = st.session_state["weights"]
    sensitive_features = st.session_state["sensitive_features"]

    section("Group Distribution Analysis", "📊")
    st.markdown("Showing original vs reweighted distributions for each sensitive feature.")

    for feat in sensitive_features:
        if feat not in X.columns:
            continue

        st.markdown(f"#### 🔍 Feature: `{feat}`")

        before_dist = X[feat].value_counts(normalize=True)
        w_series = pd.Series(weights, index=X.index)
        after_dist = w_series.groupby(X[feat]).sum()
        after_dist = after_dist / after_dist.sum()

        df_compare = pd.DataFrame({
            "Before (Original)": before_dist,
            "After (Reweighted)": after_dist
        }).fillna(0).reset_index()
        df_compare.columns = [feat, "Before (Original)", "After (Reweighted)"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(df_compare.round(4), use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 3), facecolor="white")
            x_pos = np.arange(len(df_compare))
            bars1 = ax.bar(x_pos - 0.2, df_compare["Before (Original)"], 0.38,
                           label="Before", color="#93C5FD", edgecolor="white")
            bars2 = ax.bar(x_pos + 0.2, df_compare["After (Reweighted)"], 0.38,
                           label="After", color="#2563EB", edgecolor="white")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_compare[feat].astype(str), fontsize=9, color="#1E293B")
            ax.set_ylabel("Proportion", color="#1E293B", fontsize=9)
            ax.set_title(f"{feat} — Before vs After Reweighting", color="#1E3A5F", fontweight="bold")
            ax.legend(fontsize=8)
            ax.set_facecolor("#F8FAFC")
            ax.tick_params(colors="#1E293B")
            for spine in ax.spines.values():
                spine.set_color("#E2E8F0")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    section("Class Distribution", "📈")
    class_dist = y.value_counts(normalize=True)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(class_dist.reset_index().rename(columns={"index": "Class", st.session_state["target"]: "Proportion"}),
                     use_container_width=True)
    with col2:
        minority = class_dist.min()
        if minority < 0.25:
            strategy = "SMOTE"
            color, msg = "red", "Severe imbalance → SMOTE recommended"
        elif minority < 0.40:
            strategy = "BOTH"
            color, msg = "yellow", "Moderate imbalance → SMOTE evaluated"
        else:
            strategy = "NO_SMOTE"
            color, msg = "green", "Dataset is balanced → SMOTE not needed"

        st.session_state["strategy"] = strategy
        st.markdown(f'<div class="info-card"><b>Imbalance Strategy:</b> {badge(strategy, color)}<br><small style="color:#64748B">{msg}</small></div>', unsafe_allow_html=True)
        for cls, prop in class_dist.items():
            st.progress(float(prop), text=f"Class {cls}: {prop:.1%}")


# ══════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ══════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("# 🤖 Model Training Pipeline")

    if st.session_state["X"] is None:
        st.warning("Please complete **Upload & Configure** first.")
        st.stop()

    X = st.session_state["X"]
    y = st.session_state["y"]
    weights = st.session_state["weights"]
    strategy = st.session_state["strategy"] or "BOTH"

    if st.button("🚀 Run Full Training Pipeline"):
        progress = st.progress(0, text="Starting pipeline...")

        # ── STEP 1: Train/test split
        from sklearn.model_selection import train_test_split
        progress.progress(8, "Splitting data...")

        X_train_raw, X_test_raw, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42, stratify=y
        )

        # ── STEP 2: Remove useless columns
        progress.progress(15, "Detecting useless columns...")

        def detect_useless_columns(df):
            useless = []
            for col in df.columns:
                if df[col].nunique() == len(df) or df[col].nunique() == 1:
                    useless.append(col)
            return useless

        useless = detect_useless_columns(X_train_raw)
        X_train_raw = X_train_raw.drop(columns=useless)
        X_test_raw = X_test_raw.drop(columns=useless)

        # ── STEP 3: Encoding
        progress.progress(25, "Encoding categorical features...")

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        cat_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_train_raw.select_dtypes(include=np.number).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
            remainder="passthrough"
        )

        X_train_enc = preprocessor.fit_transform(X_train_raw)
        X_test_enc = preprocessor.transform(X_test_raw)
        feature_names = preprocessor.get_feature_names_out()

        X_train_enc = pd.DataFrame(X_train_enc, columns=feature_names, index=X_train_raw.index)
        X_test_enc = pd.DataFrame(X_test_enc, columns=feature_names, index=X_test_raw.index)

        # ── STEP 4: SMOTE decision
        progress.progress(40, "Running SMOTE decision pipeline...")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.base import clone
        from imblearn.over_sampling import SMOTE

        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

        # Without SMOTE
        m_no = clone(rf)
        m_no.fit(X_train_enc, y_train)
        y_p_no = m_no.predict(X_test_enc)
        acc_no = accuracy_score(y_test, y_p_no)
        f1_no = f1_score(y_test, y_p_no, average="weighted")

        # With SMOTE
        smote = SMOTE(random_state=42)
        X_sm, y_sm = smote.fit_resample(X_train_enc, y_train)
        m_sm = clone(rf)
        m_sm.fit(X_sm, y_sm)
        y_p_sm = m_sm.predict(X_test_enc)
        acc_sm = accuracy_score(y_test, y_p_sm)
        f1_sm = f1_score(y_test, y_p_sm, average="weighted")

        if (f1_sm > f1_no) and (acc_sm >= acc_no - 0.02):
            X_train_final_smote, y_train_final_smote, choice = X_sm, y_sm, "SMOTE"
        else:
            X_train_final_smote, y_train_final_smote, choice = X_train_enc, y_train, "ORIGINAL"

        # ── STEP 5: Feature selection
        progress.progress(60, "Running feature selection pipeline...")

        from sklearn.metrics import precision_score, recall_score
        from sklearn.inspection import permutation_importance

        rf_base = clone(rf)
        rf_base.fit(X_train_enc, y_train, sample_weight=w_train)

        importances = rf_base.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": X_train_enc.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        mean_imp = importance_df["Importance"].mean()
        std_imp = importance_df["Importance"].std()
        low_t = mean_imp * 0.5
        high_t = mean_imp + std_imp

        removal_reason = {}
        for _, row in importance_df.iterrows():
            f, imp = row["Feature"], row["Importance"]
            if imp < low_t:
                removal_reason[f] = "Low Importance"
            elif imp > high_t:
                removal_reason[f] = "Dominant"

        corr_matrix = X_train_enc.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper.columns:
            if any(upper[col] > 0.9):
                removal_reason[col] = "High Correlation"

        drop_features = list(removal_reason.keys())

        def eval_model(model, Xt, yt):
            yp = model.predict(Xt)
            return {
                "accuracy": accuracy_score(yt, yp),
                "precision": precision_score(yt, yp, average="weighted", zero_division=0),
                "recall": recall_score(yt, yp, average="weighted", zero_division=0),
                "f1": f1_score(yt, yp, average="weighted", zero_division=0),
            }

        base_metrics = eval_model(rf_base, X_test_enc, y_test)

        X_tr_new = X_train_enc.drop(columns=drop_features, errors="ignore")
        X_te_new = X_test_enc.drop(columns=drop_features, errors="ignore")

        if X_tr_new.shape[1] >= 2:
            new_rf = clone(rf)
            new_rf.fit(X_tr_new, y_train)
            new_metrics = eval_model(new_rf, X_te_new, y_test)

            comparison_df = pd.DataFrame({
                "Metric": list(base_metrics.keys()),
                "Before": list(base_metrics.values()),
                "After": list(new_metrics.values()),
            })
            comparison_df["Drop"] = comparison_df["Before"] - comparison_df["After"]
            use_new = all(comparison_df["Drop"] <= 0.10)
        else:
            use_new = False
            comparison_df = None

        if use_new:
            x_train_final = X_tr_new
            x_test_final = X_te_new
        else:
            x_train_final = X_train_enc
            x_test_final = X_test_enc

        # ── STEP 6: Final model
        progress.progress(80, "Training final model...")

        final_model = clone(rf)
        final_model.fit(x_train_final, y_train_final_smote)
        y_pred_final = final_model.predict(x_test_final)

        from sklearn.metrics import classification_report
        accuracy = accuracy_score(y_test, y_pred_final)
        precision_val = precision_score(y_test, y_pred_final, average="weighted", zero_division=0)
        recall_val = recall_score(y_test, y_pred_final, average="weighted", zero_division=0)
        f1_val = f1_score(y_test, y_pred_final, average="weighted", zero_division=0)

        progress.progress(100, "Done!")

        # Save to session state
        st.session_state.update({
            "X_train": X_train_enc, "X_test": X_test_enc,
            "y_train": y_train, "y_test": y_test,
            "w_train": w_train, "w_test": w_test,
            "x_train_final": x_train_final, "x_test_final": x_test_final,
            "final_model": final_model, "y_pred_final": y_pred_final,
            "accuracy": accuracy, "precision": precision_val,
            "recall": recall_val, "f1": f1_val,
            "choice": choice, "comparison_df": comparison_df,
            "y_train_final_smote": y_train_final_smote,
            "removal_reason": removal_reason,
            "importance_df": importance_df,
            "acc_no": acc_no, "f1_no": f1_no,
            "acc_sm": acc_sm, "f1_sm": f1_sm,
        })

    # ── Display results if trained
    if st.session_state["final_model"] is not None:
        section("SMOTE Decision Results", "🔄")
        choice = st.session_state["choice"]
        acc_no = st.session_state["acc_no"]
        f1_no = st.session_state["f1_no"]
        acc_sm = st.session_state["acc_sm"]
        f1_sm = st.session_state["f1_sm"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Without SMOTE — Accuracy", f"{acc_no:.4f}")
        col2.metric("With SMOTE — Accuracy", f"{acc_sm:.4f}")
        col3.metric("Final Choice", choice, delta="Selected" if choice == "SMOTE" else "")

        section("Feature Selection Results", "🗂️")
        if st.session_state.get("removal_reason"):
            rr = st.session_state["removal_reason"]
            rr_df = pd.DataFrame(list(rr.items()), columns=["Feature", "Removal Reason"])
            st.dataframe(rr_df, use_container_width=True)
        if st.session_state["comparison_df"] is not None:
            comp = st.session_state["comparison_df"]
            st.dataframe(comp.round(4), use_container_width=True)
        else:
            st.info("Original features retained (removing features reduced performance beyond tolerance).")

        section("Final Model Performance", "📊")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{st.session_state['accuracy']:.4f}")
        c2.metric("Precision", f"{st.session_state['precision']:.4f}")
        c3.metric("Recall", f"{st.session_state['recall']:.4f}")
        c4.metric("F1 Score", f"{st.session_state['f1']:.4f}")

        section("Top Feature Importances", "📈")
        imp_df = st.session_state.get("importance_df")
        if imp_df is not None:
            top = imp_df.head(15)
            fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
            bars = ax.barh(top["Feature"][::-1], top["Importance"][::-1],
                           color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top))),
                           edgecolor="white")
            ax.set_xlabel("Importance", color="#1E293B")
            ax.set_title("Top 15 Feature Importances", color="#1E3A5F", fontweight="bold")
            ax.set_facecolor("#F8FAFC")
            ax.tick_params(colors="#1E293B")
            for spine in ax.spines.values():
                spine.set_color("#E2E8F0")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.success("✅ Training complete! Proceed to **Fairness Analysis**.")


# ══════════════════════════════════════════════
# PAGE: FAIRNESS ANALYSIS
# ══════════════════════════════════════════════
elif page == "📊 Fairness Analysis":
    st.markdown("# 📊 Post-Training Fairness Analysis")

    if st.session_state["final_model"] is None:
        st.warning("Please complete **Model Training** first.")
        st.stop()

    X = st.session_state["X"]
    y_test = st.session_state["y_test"]
    y_pred_final = st.session_state["y_pred_final"]
    sensitive_features = st.session_state["sensitive_features"]

    if st.button("🔍 Run Fairness Analysis"):
        try:
            from fairlearn.metrics import (
                MetricFrame, selection_rate,
                equalized_odds_difference, demographic_parity_difference
            )
            from sklearn.metrics import accuracy_score

            fairness_results = []
            progress = st.progress(0, "Computing fairness metrics...")

            for i, feature in enumerate(sensitive_features):
                sf_test = X.loc[y_test.index, feature]
                mf = MetricFrame(
                    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
                    y_true=y_test, y_pred=y_pred_final, sensitive_features=sf_test
                )
                eo = equalized_odds_difference(y_test, y_pred_final, sensitive_features=sf_test)
                dp = demographic_parity_difference(y_test, y_pred_final, sensitive_features=sf_test)

                fairness_results.append({
                    "Feature": feature,
                    "EO": eo,
                    "DP": dp,
                    "Score": (abs(eo) + abs(dp)) / 2,
                    "by_group": mf.by_group
                })
                progress.progress(int((i + 1) / len(sensitive_features) * 100))

            fairness_df = pd.DataFrame([
                {k: v for k, v in r.items() if k != "by_group"}
                for r in fairness_results
            ]).sort_values("Score", ascending=False)

            st.session_state["fairness_df"] = fairness_df
            st.session_state["fairness_results_full"] = fairness_results
            progress.progress(100, "Done!")
        except ImportError:
            st.error("fairlearn not installed. Please install it: `pip install fairlearn`")
            st.stop()

    if st.session_state["fairness_df"] is not None:
        fairness_df = st.session_state["fairness_df"]
        fairness_results_full = st.session_state.get("fairness_results_full", [])

        section("Fairness Score Summary", "⚖️")
        st.dataframe(fairness_df.round(4), use_container_width=True)

        section("Group-wise Metrics", "📋")
        for r in fairness_results_full:
            st.markdown(f"**Feature: `{r['Feature']}`**")
            st.dataframe(r["by_group"].round(4), use_container_width=True)

        section("Bias Level & Recommendations", "💡")
        for _, row in fairness_df.iterrows():
            feat, score = row["Feature"], row["Score"]

            if score > 0.3:
                level = "HIGH BIAS"
                b_color = "red"
                suggestion = "Apply ThresholdOptimizer or remove/transform this feature."
                icon = "🔴"
            elif score > 0.15:
                level = "MODERATE BIAS"
                b_color = "yellow"
                suggestion = "Try reweighting again, feature engineering, or check proxy bias."
                icon = "🟡"
            else:
                level = "LOW BIAS"
                b_color = "green"
                suggestion = "Bias is acceptable. No major intervention needed."
                icon = "🟢"

            card(
                f"<b style='color:#1E3A5F'>{icon} {feat}</b> — {badge(level, b_color)}<br>"
                f"<small>EO Difference: <b>{row['EO']:.4f}</b> &nbsp;|&nbsp; "
                f"DP Difference: <b>{row['DP']:.4f}</b> &nbsp;|&nbsp; "
                f"Score: <b>{score:.4f}</b></small><br>"
                f"<small style='color:#64748B'>➤ {suggestion}</small>"
            )

        section("Bias Score Visualization", "📊")
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        colors_map = {"LOW": "#059669", "MOD": "#D97706", "HIGH": "#DC2626"}
        bar_colors = [
            "#DC2626" if s > 0.3 else "#D97706" if s > 0.15 else "#059669"
            for s in fairness_df["Score"]
        ]
        ax.bar(fairness_df["Feature"], fairness_df["Score"], color=bar_colors, edgecolor="white", width=0.5)
        ax.axhline(0.3, color="#DC2626", linestyle="--", lw=1.5, label="High threshold (0.3)")
        ax.axhline(0.15, color="#D97706", linestyle="--", lw=1.5, label="Moderate threshold (0.15)")
        ax.set_ylabel("Bias Score", color="#1E293B")
        ax.set_title("Fairness Score by Sensitive Feature", color="#1E3A5F", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_facecolor("#F8FAFC")
        ax.tick_params(colors="#1E293B")
        for sp in ax.spines.values():
            sp.set_color("#E2E8F0")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.success("✅ Fairness analysis complete! Proceed to **SHAP Explainability**.")


# ══════════════════════════════════════════════
# PAGE: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════
elif page == "💡 SHAP Explainability":
    st.markdown("# 💡 SHAP Model Explainability")

    if st.session_state["final_model"] is None:
        st.warning("Please complete **Model Training** first.")
        st.stop()

    final_model = st.session_state["final_model"]
    x_train_final = st.session_state["x_train_final"]
    x_test_final = st.session_state["x_test_final"]

    if st.button("🧠 Compute SHAP Values"):
        with st.spinner("Computing SHAP values (this may take a moment)..."):
            try:
                import shap
                sample_X = x_test_final.sample(min(100, len(x_test_final)), random_state=42)
                explainer = shap.TreeExplainer(final_model)
                shap_values_raw = explainer(sample_X)

                # ── Handle 3-D output (multiclass: samples × features × classes)
                # shap_values_raw.values shape can be (n, f) for binary or (n, f, c) for multiclass
                raw_arr = shap_values_raw.values
                if raw_arr.ndim == 3:
                    # Use class index 1 (positive class) for binary, or mean-abs across classes
                    n_classes = raw_arr.shape[2]
                    if n_classes == 2:
                        shap_2d = raw_arr[:, :, 1]          # positive class
                    else:
                        shap_2d = raw_arr.mean(axis=2)       # average across all classes
                else:
                    shap_2d = raw_arr                        # already 2D

                # Store both: raw Explanation object AND the safe 2D numpy array
                st.session_state["shap_values"] = shap_values_raw
                st.session_state["shap_values_2d"] = shap_2d
                st.session_state["shap_sample_X"] = sample_X
                st.success("✅ SHAP values computed!")
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    if st.session_state["shap_values"] is not None:
        import shap
        shap_values = st.session_state["shap_values"]
        shap_2d     = st.session_state["shap_values_2d"]   # always (n_samples, n_features)
        sample_X    = st.session_state["shap_sample_X"]
        fairness_df = st.session_state["fairness_df"]

        section("SHAP Summary Plot", "📊")
        try:
            plt.close("all")
            shap.summary_plot(shap_2d, sample_X, show=False)
            fig_sum = plt.gcf()
            fig_sum.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig_sum)
            plt.close("all")
        except Exception as e:
            st.error(f"Summary plot failed: {e}")

        section("SHAP Bar Plot (Global Importance)", "📈")
        try:
            plt.close("all")
            shap.summary_plot(shap_2d, sample_X, plot_type="bar", show=False)
            fig_bar = plt.gcf()
            fig_bar.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close("all")
        except Exception as e:
            st.error(f"Bar plot failed: {e}")

        if fairness_df is not None:
            section("Fairness × SHAP Combined Insights", "⚠️")
            # shap_2d is always (n_samples, n_features) — safe to take mean(axis=0)
            shap_importance = np.abs(shap_2d).mean(axis=0)
            shap_feat_imp = dict(zip(sample_X.columns, shap_importance))

            sv_list = list(shap_feat_imp.values())
            p75 = np.percentile(sv_list, 75)
            p40 = np.percentile(sv_list, 40)

            for _, row in fairness_df.iterrows():
                feat = row["Feature"]
                score = row["Score"]
                shap_imp = shap_feat_imp.get(feat, 0)
                shap_level = "HIGH" if shap_imp > p75 else "MODERATE" if shap_imp > p40 else "LOW"

                if score > 0.3:
                    level, b_color, suggestion = "HIGH BIAS", "red", "Apply ThresholdOptimizer or remove/transform."
                elif score > 0.15:
                    level, b_color, suggestion = "MODERATE BIAS", "yellow", "Reweight, feature engineering, or check proxy."
                else:
                    level, b_color, suggestion = "LOW BIAS", "green", "Acceptable bias."

                if score > 0.3 and shap_level == "HIGH":
                    insight = "⚠️ CRITICAL: Feature is both biased AND highly influential."
                    i_color = "#991B1B"
                elif score > 0.3 and shap_level == "LOW":
                    insight = "⚠️ Bias exists but model doesn't rely heavily on this feature."
                    i_color = "#92400E"
                elif score <= 0.15 and shap_level == "HIGH":
                    insight = "ℹ️ Important feature but not causing fairness issues."
                    i_color = "#1E40AF"
                else:
                    insight = "✔ Balanced behavior."
                    i_color = "#065F46"

                card(
                    f"<b style='color:#1E3A5F'>{feat}</b> — {badge(level, b_color)} | "
                    f"{badge(f'SHAP: {shap_level}', 'blue')}<br>"
                    f"<small>Fairness Score: <b>{score:.4f}</b> &nbsp;|&nbsp; SHAP Impact: <b>{shap_imp:.4f}</b></small><br>"
                    f"<small>➤ Action: {suggestion}</small><br>"
                    f"<small style='color:{i_color}'><b>{insight}</b></small>"
                )


# ══════════════════════════════════════════════
# PAGE: GEMINI AI ASSISTANT
# ══════════════════════════════════════════════
elif page == "✨ Gemini AI Assistant":
    st.markdown("# ✨ Gemini AI Assistant")
    st.markdown("Powered by **Google Gemini 2.0 Flash** — your intelligent ML fairness co-pilot.")

    # ── API Key — hardcoded, no input needed from any user
    # ✏️  REPLACE the string below with your actual Gemini API key once.
    # ✏️  Every user who opens the app will use it automatically.
    from dotenv import load_dotenv
    import os 
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    st.session_state["gemini_api_key"] = GEMINI_API_KEY

    if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        st.error(
            "⚠️ Gemini API key not set yet.\n\n"
            "Open **app.py**, find the line that says `GEMINI_API_KEY = \"PASTE_YOUR_GEMINI_API_KEY_HERE\"`"
            " and replace it with your actual key from https://aistudio.google.com/app/apikey"
        )
        st.stop()

    # ── Helper: call Gemini
    from google import genai
    def call_gemini(prompt, system="You are an expert ML fairness and explainability scientist. Be clear, concise, and actionable."):
        try:
            import google.generativeai as genai
            genai.configure(api_key=st.session_state["gemini_api_key"])
            model = genai.GenerativeModel(
                "gemini-flash-lite-latest",
                system_instruction=system
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Gemini error: {e}"

    # ── TABS for 4 features
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Bias Explainer",
        "💬 Chat with Dataset",
        "📝 Smart Report",
        "🧠 SHAP Insights",
    ])

    # ─────────────────────────────────────────
    # TAB 1: AI BIAS EXPLAINER
    # ─────────────────────────────────────────
    with tab1:
        st.markdown("### 🔴 AI Bias Explainer")
        st.markdown("Gemini reads your fairness results and explains them in plain English with actionable recommendations.")

        if st.session_state["fairness_df"] is None:
            st.warning("Run **Fairness Analysis** first to unlock this feature.")
        else:
            fairness_df = st.session_state["fairness_df"]
            accuracy   = st.session_state["accuracy"] or 0
            f1_val     = st.session_state["f1"] or 0
            target     = st.session_state["target"] or "target"
            choice     = st.session_state["choice"] or "ORIGINAL"
            strategy   = st.session_state["strategy"] or "BOTH"

            with st.expander("📊 Fairness Data being sent to Gemini"):
                st.dataframe(fairness_df.round(4), use_container_width=True)

            if st.button("✨ Generate AI Bias Explanation", key="bias_explain_btn"):
                with st.spinner("Gemini is analyzing your bias results..."):
                    prompt = f"""
You are analyzing a machine learning fairness report for a model that predicts: **{target}**.

## Model Performance
- Accuracy: {accuracy:.4f}
- F1 Score: {f1_val:.4f}
- Training Data Strategy: {choice} (SMOTE strategy detected: {strategy})

## Fairness Results
{fairness_df.to_string(index=False)}

**Columns explained:**
- EO (Equalized Odds Difference): measures difference in true positive rates across groups. Closer to 0 = fairer.
- DP (Demographic Parity Difference): measures difference in prediction rates across groups. Closer to 0 = fairer.
- Score: average of |EO| and |DP| — the overall bias score.

## Your Task
1. Explain what these fairness numbers mean in simple, non-technical language (2-3 sentences per feature).
2. Identify which feature has the most serious bias and WHY it matters in real-world context.
3. Give 3 specific, practical actions the developer can take to reduce bias for each HIGH or MODERATE feature.
4. End with an overall fairness verdict: Is this model safe to deploy? Why or why not?

Use clear headings, bullet points, and emojis to make the response easy to read.
"""
                    result = call_gemini(prompt)
                    st.session_state["gemini_bias_explanation"] = result

            if st.session_state.get("gemini_bias_explanation"):
                st.markdown(
                    f'<div class="info-card" style="border-left:4px solid #4285F4">'
                    f'{st.session_state["gemini_bias_explanation"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ─────────────────────────────────────────
    # TAB 2: CHAT WITH DATASET
    # ─────────────────────────────────────────
    with tab2:
        st.markdown("### 💬 Chat with Your Dataset & Pipeline")
        st.markdown("Ask anything about your data, model, fairness results, or ML concepts.")

        if st.session_state["df"] is None:
            st.warning("Upload a dataset first to enable chat.")
        else:
            df        = st.session_state["df"]
            target    = st.session_state["target"] or "unknown"
            sens_feat = st.session_state["sensitive_features"] or []
            acc       = st.session_state["accuracy"]
            f1v       = st.session_state["f1"]
            fdf       = st.session_state["fairness_df"]

            # Build context summary
            context = f"""
You are an AI assistant embedded in a Fair ML Pipeline app.
You have full knowledge of the user's dataset and pipeline results.

DATASET CONTEXT:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Target column: {target}
- Sensitive features: {', '.join(sens_feat) if sens_feat else 'not set yet'}
- Numeric columns: {', '.join(df.select_dtypes(include='number').columns.tolist())}
- Categorical columns: {', '.join(df.select_dtypes(include='object').columns.tolist())}
- Missing values: {df.isnull().sum().sum()}
- Class distribution: {df[target].value_counts().to_dict() if target in df.columns else 'N/A'}

MODEL RESULTS:
- Accuracy: {f'{acc:.4f}' if acc else 'not trained yet'}
- F1 Score: {f'{f1v:.4f}' if f1v else 'not trained yet'}
- SMOTE choice: {st.session_state.get('choice', 'not run yet')}

FAIRNESS RESULTS:
{fdf.to_string(index=False) if fdf is not None else 'Fairness analysis not run yet'}

Answer the user's questions helpfully, accurately, and concisely.
If asked about something not in the context, use your general ML knowledge.
"""

            # Display chat history
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state["chat_history"]:
                    if msg["role"] == "user":
                        st.markdown(
                            f'<div style="background:#EFF6FF;border-radius:12px 12px 4px 12px;'
                            f'padding:12px 16px;margin:6px 0;color:#1E3A5F;text-align:right;">'
                            f'<b>You:</b> {msg["content"]}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;'
                            f'border-radius:4px 12px 12px 12px;padding:12px 16px;'
                            f'margin:6px 0;color:#1E293B;">'
                            f'<b>✨ Gemini:</b> {msg["content"]}</div>',
                            unsafe_allow_html=True,
                        )

            # Suggested quick questions
            st.markdown("**💡 Quick Questions:**")
            quick_cols = st.columns(3)
            quick_questions = [
                "Which feature has the most bias?",
                "Why is my model's accuracy at this level?",
                "Should I use SMOTE for this dataset?",
                "What does Equalized Odds mean?",
                "How can I improve model fairness?",
                "Explain SHAP values simply",
            ]
            for i, qq in enumerate(quick_questions):
                with quick_cols[i % 3]:
                    if st.button(qq, key=f"qq_{i}"):
                        with st.spinner("Gemini is thinking..."):
                            reply = call_gemini(qq, system=context)
                            st.session_state["chat_history"].append({"role": "user", "content": qq})
                            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                        st.rerun()

            # Free-form input
            st.markdown("")
            user_input = st.text_input(
                "Ask Gemini anything...",
                placeholder="e.g. Why is Gender showing high bias? What should I do next?",
                key="chat_input"
            )
            col_send, col_clear = st.columns([1, 5])
            with col_send:
                send_clicked = st.button("Send ➤", key="send_chat")
            with col_clear:
                if st.button("🗑️ Clear Chat", key="clear_chat"):
                    st.session_state["chat_history"] = []
                    st.rerun()

            if send_clicked and user_input.strip():
                with st.spinner("Gemini is thinking..."):
                    reply = call_gemini(user_input, system=context)
                    st.session_state["chat_history"].append({"role": "user", "content": user_input})
                    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()

    # ─────────────────────────────────────────
    # TAB 3: SMART REPORT NARRATION
    # ─────────────────────────────────────────
    with tab3:
        st.markdown("### 📝 AI-Generated Smart Report")
        st.markdown("Gemini writes a **custom narrative report** tailored to your specific pipeline results — ready to include in your submission.")

        if st.session_state["final_model"] is None:
            st.warning("Run **Model Training** first to generate a smart report.")
        else:
            target    = st.session_state["target"] or "target"
            sens_feat = st.session_state["sensitive_features"] or []
            acc       = st.session_state["accuracy"] or 0
            prec      = st.session_state["precision"] or 0
            rec       = st.session_state["recall"] or 0
            f1v       = st.session_state["f1"] or 0
            choice    = st.session_state["choice"] or "ORIGINAL"
            strategy  = st.session_state["strategy"] or "BOTH"
            acc_no    = st.session_state["acc_no"] or 0
            f1_no     = st.session_state["f1_no"] or 0
            acc_sm    = st.session_state["acc_sm"] or 0
            f1_sm     = st.session_state["f1_sm"] or 0
            fdf       = st.session_state["fairness_df"]
            comp_df   = st.session_state["comparison_df"]

            report_style = st.selectbox(
                "Report Style",
                ["Technical (for judges/developers)", "Executive Summary (for non-technical audience)", "Academic (for research submission)"],
                key="report_style_select"
            )

            if st.button("✨ Generate Smart Report", key="smart_report_btn"):
                with st.spinner("Gemini is writing your personalized report..."):
                    style_instruction = {
                        "Technical (for judges/developers)": "Write in a technical style with ML terminology, code insights, and specific metric analysis.",
                        "Executive Summary (for non-technical audience)": "Write in plain English, avoid jargon, focus on real-world impact and business value.",
                        "Academic (for research submission)": "Write in formal academic style with sections like Abstract, Methodology, Results, Discussion, and Conclusion.",
                    }[report_style]

                    fairness_section = fdf.to_string(index=False) if fdf is not None else "Fairness analysis not run."
                    feature_section  = comp_df.to_string(index=False) if comp_df is not None else "Original features retained."

                    prompt = f"""
{style_instruction}

Write a comprehensive report for a Fair ML Pipeline project submitted to the Google Solution Challenge.

## Pipeline Details
- **Prediction Target:** {target}
- **Sensitive Features Analyzed:** {', '.join(sens_feat)}
- **Imbalance Strategy Detected:** {strategy}
- **Final Training Data:** {choice}

## SMOTE Comparison
- Without SMOTE → Accuracy: {acc_no:.4f}, F1: {f1_no:.4f}
- With SMOTE → Accuracy: {acc_sm:.4f}, F1: {f1_sm:.4f}
- Decision: Used {choice} dataset

## Feature Selection Results
{feature_section}

## Final Model Performance
- Accuracy: {acc:.4f}
- Precision: {prec:.4f}
- Recall: {rec:.4f}
- F1 Score: {f1v:.4f}

## Fairness Analysis Results
{fairness_section}

## Required Report Sections
1. Project Overview & Problem Statement
2. Methodology (preprocessing, fairness techniques, model training)
3. Results & Analysis (with interpretation of ALL metrics)
4. Fairness Evaluation & Real-World Impact
5. Limitations & Future Work
6. Conclusion

Make it compelling, specific to these results, and submission-ready for Google Solution Challenge.
"""
                    result = call_gemini(prompt)
                    st.session_state["gemini_report_text"] = result

            if st.session_state.get("gemini_report_text"):
                st.markdown(
                    f'<div class="info-card" style="border-left:4px solid #059669;max-height:600px;overflow-y:auto;">'
                    f'{st.session_state["gemini_report_text"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Download as text
                st.download_button(
                    "⬇️ Download Report as .txt",
                    data=st.session_state["gemini_report_text"],
                    file_name="FairML_AI_Report.txt",
                    mime="text/plain",
                    key="dl_report_txt"
                )

    # ─────────────────────────────────────────
    # TAB 4: SHAP INSIGHT EXPLAINER
    # ─────────────────────────────────────────
    with tab4:
        st.markdown("### 🧠 AI SHAP Insight Explainer")
        st.markdown("Gemini reads your SHAP values and explains **why** the model made its decisions in human-readable language.")

        if st.session_state["shap_values_2d"] is None:
            st.warning("Run **SHAP Explainability** first to unlock this feature.")
        else:
            shap_2d   = st.session_state["shap_values_2d"]
            sample_X  = st.session_state["shap_sample_X"]
            target    = st.session_state["target"] or "target"
            fdf       = st.session_state["fairness_df"]

            # Build top SHAP features table
            shap_importance = np.abs(shap_2d).mean(axis=0)
            shap_df = pd.DataFrame({
                "Feature": sample_X.columns,
                "Mean |SHAP|": shap_importance
            }).sort_values("Mean |SHAP|", ascending=False).head(15)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 15 Features by SHAP Importance:**")
                st.dataframe(shap_df.round(5), use_container_width=True)
            with col2:
                fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
                ax.barh(shap_df["Feature"][::-1], shap_df["Mean |SHAP|"][::-1],
                        color=plt.cm.Blues(np.linspace(0.4, 0.9, len(shap_df))),
                        edgecolor="white")
                ax.set_xlabel("Mean |SHAP|", color="#1E293B", fontsize=9)
                ax.set_title("SHAP Feature Importance", color="#1E3A5F", fontweight="bold")
                ax.set_facecolor("#F8FAFC")
                ax.tick_params(colors="#1E293B", labelsize=8)
                for sp in ax.spines.values():
                    sp.set_color("#E2E8F0")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            if st.button("✨ Generate SHAP AI Explanation", key="shap_explain_btn"):
                with st.spinner("Gemini is interpreting your SHAP values..."):

                    fairness_context = ""
                    if fdf is not None:
                        fairness_context = f"\n\nFairness Results:\n{fdf.to_string(index=False)}"

                    prompt = f"""
You are an expert ML explainability scientist analyzing SHAP (SHapley Additive exPlanations) values.

The model predicts: **{target}**

## Top 15 Features by Mean Absolute SHAP Value
{shap_df.to_string(index=False)}
{fairness_context}

## Your Task

**Part 1 — Feature Impact Explanation:**
For the top 5 most important features, explain:
- What does this feature represent in the real world?
- How much does it influence predictions (high/medium/low)?
- Does a higher or lower value push the prediction toward the positive class?

**Part 2 — Model Decision Logic:**
In 3-4 sentences, summarize the overall "story" the model tells — what patterns is it primarily using to make decisions?

**Part 3 — Fairness × SHAP Connection:**
{"Cross-reference the SHAP importance with the fairness results. Flag any features that are BOTH highly important AND biased — these are the most critical to address." if fdf is not None else "Note: Run Fairness Analysis to get a combined fairness × SHAP analysis."}

**Part 4 — Recommendations:**
Give 3 specific actions the developer should take based on these SHAP insights.

Use clear headings, bullet points, and plain language.
"""
                    result = call_gemini(prompt)
                    st.session_state["gemini_shap_explanation"] = result

            if st.session_state.get("gemini_shap_explanation"):
                st.markdown(
                    f'<div class="info-card" style="border-left:4px solid #7C3AED">'
                    f'{st.session_state["gemini_shap_explanation"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════
# PAGE: EXPORT REPORT
# ══════════════════════════════════════════════
elif page == "📄 Export Report":
    st.markdown("# 📄 Export Full Report")

    if st.session_state["final_model"] is None:
        st.warning("Please complete **Model Training** first.")
        st.stop()

    section("Report Preview", "👁️")

    target = st.session_state["target"]
    sensitive_features = st.session_state["sensitive_features"] or []
    X = st.session_state["X"]
    weights = st.session_state["weights"]
    strategy = st.session_state["strategy"] or "BOTH"
    choice = st.session_state["choice"] or "ORIGINAL"
    acc_no = st.session_state["acc_no"] or 0
    f1_no = st.session_state["f1_no"] or 0
    acc_sm = st.session_state["acc_sm"] or 0
    f1_sm = st.session_state["f1_sm"] or 0
    accuracy = st.session_state["accuracy"] or 0
    precision_val = st.session_state["precision"] or 0
    recall_val = st.session_state["recall"] or 0
    f1_val = st.session_state["f1"] or 0
    comparison_df = st.session_state["comparison_df"]
    fairness_df = st.session_state["fairness_df"]
    x_train_final = st.session_state["x_train_final"]
    x_test_final = st.session_state["x_test_final"]
    final_model = st.session_state["final_model"]
    shap_values = st.session_state["shap_values"]
    shap_sample_X = st.session_state.get("shap_sample_X")
    # Gemini-generated content (optional, included if available)
    for _gk in ["gemini_bias_explanation", "gemini_report_text", "gemini_shap_explanation"]:
        if _gk not in st.session_state:
            st.session_state[_gk] = None

    # Show a preview card
    col1, col2 = st.columns(2)
    with col1:
        card(
            f"<b style='color:#1E3A5F'>Pipeline Summary</b><br>"
            f"Target: <b>{target}</b><br>"
            f"Sensitive Features: <b>{', '.join(sensitive_features)}</b><br>"
            f"SMOTE Strategy: <b>{strategy}</b><br>"
            f"Final Dataset: <b>{choice}</b>"
        )
    with col2:
        card(
            f"<b style='color:#1E3A5F'>Model Performance</b><br>"
            f"Accuracy: <b>{accuracy:.4f}</b><br>"
            f"Precision: <b>{precision_val:.4f}</b><br>"
            f"Recall: <b>{recall_val:.4f}</b><br>"
            f"F1 Score: <b>{f1_val:.4f}</b>"
        )

    if st.button("📄 Generate & Download PDF Report"):
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            import shap as shap_lib

            styles = getSampleStyleSheet()
            story = []

            # ── Title
            title_style = ParagraphStyle("title", fontSize=20, fontName="Helvetica-Bold",
                                          textColor=colors.HexColor("#1E3A5F"), spaceAfter=6)
            sub_style = ParagraphStyle("sub", fontSize=11, textColor=colors.HexColor("#64748B"), spaceAfter=14)
            h2_style = ParagraphStyle("h2", fontSize=14, fontName="Helvetica-Bold",
                                       textColor=colors.HexColor("#2563EB"), spaceBefore=14, spaceAfter=6)
            body_style = ParagraphStyle("body", fontSize=10, textColor=colors.HexColor("#1E293B"), spaceAfter=4, leading=14)

            story.append(Paragraph("Fair, Explainable & Optimized ML Pipeline", title_style))
            story.append(Paragraph("Google Solution Challenge — Automated Fairness-Aware Machine Learning", sub_style))
            story.append(Spacer(1, 8))

            # ── Intro
            story.append(Paragraph("Pipeline Overview", h2_style))
            story.append(Paragraph(
                "This project implements an end-to-end ML pipeline with fairness-aware learning, "
                "automated imbalance handling (SMOTE decision), feature optimization, and model "
                "explainability via SHAP. The pipeline ensures transparent and accountable AI.",
                body_style
            ))

            # ── Config
            story.append(Paragraph("Configuration", h2_style))
            story.append(Paragraph(f"Target Column: {target}", body_style))
            story.append(Paragraph(f"Sensitive Features: {', '.join(sensitive_features)}", body_style))
            story.append(Spacer(1, 8))

            # ── Reweighting
            story.append(Paragraph("Reweighting (Fairness Preprocessing)", h2_style))
            story.append(Paragraph("Sample reweighting was applied to balance under-represented sensitive groups.", body_style))

            w_series = pd.Series(weights, index=X.index)
            for col in sensitive_features:
                if col not in X.columns:
                    continue
                before = X[col].value_counts(normalize=True)
                after = w_series.groupby(X[col]).sum()
                after = after / after.sum()
                df_c = pd.DataFrame({"Before": before, "After": after}).fillna(0)
                story.append(Paragraph(f"Feature: {col}", body_style))
                for idx, row in df_c.iterrows():
                    story.append(Paragraph(f"  {idx}: Before={row['Before']:.3f} → After={row['After']:.3f}", body_style))
                story.append(Spacer(1, 4))

            # ── SMOTE
            story.append(Paragraph("Class Imbalance Handling", h2_style))
            story.append(Paragraph(
                f"Strategy detected: {strategy}. "
                f"Without SMOTE → Accuracy: {acc_no:.4f}, F1: {f1_no:.4f}. "
                f"With SMOTE → Accuracy: {acc_sm:.4f}, F1: {f1_sm:.4f}. "
                f"Final Decision: {choice}.",
                body_style
            ))

            # ── Feature selection
            story.append(Paragraph("Feature Selection", h2_style))
            if comparison_df is not None:
                story.append(Paragraph("Metric comparison (before vs after feature removal):", body_style))
                for _, row in comparison_df.iterrows():
                    story.append(Paragraph(
                        f"{row['Metric']}: Before={row['Before']:.4f}, After={row['After']:.4f}, Drop={row['Drop']:.4f}",
                        body_style
                    ))
            else:
                story.append(Paragraph("Original features were retained as performance drop exceeded tolerance.", body_style))

            # ── Model performance
            story.append(Paragraph("Final Model Performance", h2_style))
            story.append(Paragraph(f"Accuracy: {accuracy:.4f}", body_style))
            story.append(Paragraph(f"Precision: {precision_val:.4f}", body_style))
            story.append(Paragraph(f"Recall: {recall_val:.4f}", body_style))
            story.append(Paragraph(f"F1 Score: {f1_val:.4f}", body_style))

            # ── Heatmap image
            story.append(Paragraph("Feature Correlation Heatmap", h2_style))
            try:
                hmap_fig, hax = plt.subplots(figsize=(8, 5), facecolor="white")
                feats = x_train_final.columns[:15]
                sns.heatmap(x_train_final[feats].corr(), annot=True, cmap="Blues",
                            fmt=".2f", ax=hax, annot_kws={"size": 7})
                hax.set_title("Feature Correlation Heatmap", color="#1E3A5F")
                plt.tight_layout()
                hmap_buf = io.BytesIO()
                hmap_fig.savefig(hmap_buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
                hmap_buf.seek(0)
                plt.close(hmap_fig)
                story.append(RLImage(hmap_buf, width=5 * inch, height=3.2 * inch))
            except Exception as e:
                story.append(Paragraph(f"Heatmap not available: {e}", body_style))

            # ── SHAP image
            story.append(Paragraph("SHAP Explainability", h2_style))
            shap_2d_pdf = st.session_state.get("shap_values_2d")
            if shap_2d_pdf is not None and shap_sample_X is not None:
                try:
                    import shap as shap_lib
                    plt.close("all")
                    shap_lib.summary_plot(shap_2d_pdf, shap_sample_X, show=False)
                    shap_buf = io.BytesIO()
                    plt.gcf().savefig(shap_buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
                    shap_buf.seek(0)
                    plt.close("all")
                    story.append(RLImage(shap_buf, width=5 * inch, height=3.2 * inch))
                except Exception as e:
                    story.append(Paragraph(f"SHAP plot not available: {e}", body_style))
            else:
                story.append(Paragraph("SHAP values not computed. Run SHAP Explainability page first.", body_style))

            # ── Fairness
            story.append(Paragraph("Final Fairness Evaluation", h2_style))
            if fairness_df is not None:
                for _, row in fairness_df.iterrows():
                    score = row["Score"]
                    if score > 0.3:
                        level = "HIGH BIAS"
                        suggestion = "Apply ThresholdOptimizer or remove/transform this feature."
                    elif score > 0.15:
                        level = "MODERATE BIAS"
                        suggestion = "Try reweighting, feature engineering, or check proxy bias."
                    else:
                        level = "LOW BIAS"
                        suggestion = "Bias is acceptable. No major intervention needed."

                    story.append(Paragraph(
                        f"<b>{row['Feature']}</b> | EO: {row['EO']:.4f} | DP: {row['DP']:.4f} | "
                        f"Score: {score:.4f} | Level: {level}", body_style))
                    story.append(Paragraph(f"  ➤ {suggestion}", body_style))
                    story.append(Spacer(1, 4))
            else:
                story.append(Paragraph("Fairness analysis not yet run.", body_style))

            # ── Gemini AI Narrative (if available)
            gemini_report = st.session_state.get("gemini_report_text")
            if gemini_report:
                story.append(Paragraph("AI-Generated Narrative (Gemini)", h2_style))
                # Strip markdown bold/headers for PDF
                import re
                clean_report = re.sub(r'\*\*(.*?)\*\*', r'\1', gemini_report)
                clean_report = re.sub(r'#{1,3} ', '', clean_report)
                for line in clean_report.split('\n'):
                    line = line.strip()
                    if line:
                        story.append(Paragraph(line, body_style))
                story.append(Spacer(1, 12))

            # ── Conclusion
            story.append(Paragraph("Conclusion & Future Work", h2_style))
            story.append(Paragraph(
                "The pipeline successfully applied reweighting to reduce bias and evaluated SMOTE and "
                "feature selection. SHAP analysis provided transparency into model decisions. "
                "Fairness metrics revealed areas needing further mitigation. Future work includes "
                "post-processing with ThresholdOptimizer and adversarial debiasing techniques.",
                body_style
            ))

            # ── Build PDF
            pdf_buf = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buf,
                                    leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                                    topMargin=1 * inch, bottomMargin=0.75 * inch)
            doc.build(story)
            pdf_buf.seek(0)

            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_buf,
                file_name="FairML_Pipeline_Report.pdf",
                mime="application/pdf",
            )
            st.success("✅ Report generated! Click above to download.")

        except ImportError:
            st.error("reportlab is required: `pip install reportlab`")
        except Exception as e:
            st.error(f"Report generation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    section("Export Final Dataset", "💾")
    if st.button("💾 Download Final ML Dataset (CSV)"):
        if st.session_state["x_train_final"] is not None:
            target = st.session_state["target"]
            x_train_final = st.session_state["x_train_final"]
            y_train_final_smote = st.session_state["y_train_final_smote"]
            w_train = st.session_state["w_train"]

            final_df = x_train_final.copy()
            final_df[target] = y_train_final_smote.values if hasattr(y_train_final_smote, "values") else y_train_final_smote

            csv_buf = io.StringIO()
            final_df.to_csv(csv_buf, index=False)

            st.download_button(
                label="⬇️ Download final_ml_dataset.csv",
                data=csv_buf.getvalue(),
                file_name="final_ml_dataset.csv",
                mime="text/csv",
            )
        else:
            st.warning("No trained model data found.")