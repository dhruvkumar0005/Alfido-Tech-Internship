import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─── ABSOLUTE BASE DIRECTORY (works regardless of working directory) ────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
IMAGES_DIR  = os.path.join(BASE_DIR, "images")
DATA_PATH   = os.path.join(BASE_DIR, "loan_prediction.csv")
BUNDLE_PATH = os.path.join(MODELS_DIR, "loan_approval_models.pkl")

# ─── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #1a1f36 0%, #0d1117 100%);
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.app-header h1 {
    color: #f1f5f9;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

/* Status Cards */
.card-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.18) 0%, rgba(5,150,105,0.06) 100%);
    border: 2px solid #10b981;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
}
.card-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.18) 0%, rgba(220,38,38,0.06) 100%);
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
}
.card-review {
    background: linear-gradient(135deg, rgba(245,158,11,0.18) 0%, rgba(217,119,6,0.06) 100%);
    border: 2px solid #f59e0b;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-green  { background: #10b981; color: white; }
.badge-red    { background: #ef4444; color: white; }
.badge-yellow { background: #f59e0b; color: white; }

/* Confidence bar */
.conf-bar-bg {
    background: #1e293b;
    border-radius: 8px;
    height: 14px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-bar-fill-green  { height: 100%; background: linear-gradient(90deg,#059669,#10b981); border-radius: 8px; }
.conf-bar-fill-yellow { height: 100%; background: linear-gradient(90deg,#d97706,#f59e0b); border-radius: 8px; }
.conf-bar-fill-red    { height: 100%; background: linear-gradient(90deg,#dc2626,#ef4444); border-radius: 8px; }

/* Insight boxes */
.insight-box {
    background: #1e293b;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #cbd5e1;
    font-size: 0.9rem;
}

/* Data table */
.highlight-best td:first-child { font-weight: 700; color: #f59e0b; }

/* ── Sidebar base ── */
[data-testid="stSidebar"] {
    background: #0f172a;
}

/* All text inside sidebar */
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Section headings (## markdown) */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}

/* Widget labels (Gender, Married, etc.) */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Selectbox / dropdown text */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div {
    color: #f1f5f9 !important;
    background-color: #1e293b !important;
}

/* Selectbox border */
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    border-color: #334155 !important;
    background-color: #1e293b !important;
}

/* Radio button text */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

/* Number input */
[data-testid="stSidebar"] .stNumberInput input {
    color: #f1f5f9 !important;
    background-color: #1e293b !important;
    border-color: #334155 !important;
}

/* Caption / small info text */
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #64748b !important;
}

/* Dividers */
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
}

/* Markdown paragraphs */
[data-testid="stSidebar"] .stMarkdown p {
    color: #cbd5e1 !important;
}

/* Best model caption line */
[data-testid="stSidebar"] .stCaption p {
    color: #7dd3fc !important;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ─── AUTO-TRAIN PIPELINE (runs if pkl is missing) ─────────────────────────────
def train_and_save_bundle():
    """Full training pipeline — runs automatically when pkl is missing."""
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score)
    from imblearn.over_sampling import SMOTE

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Loan_ID"], errors="ignore")

    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History"]
    cat_cols = ["Gender", "Married", "Dependents", "Education",
                "Self_Employed", "Property_Area"]

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"].map({"Y": 1, "N": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False,
                              handle_unknown="ignore"), cat_cols)
    ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_train_proc, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = []
    for name, model in models.items():
        model.fit(X_tr_sm, y_tr_sm)
        yp  = model.predict(X_test_proc)
        ypr = model.predict_proba(X_test_proc)[:, 1]
        results.append({
            "Model":     name,
            "Accuracy":  accuracy_score(y_test, yp),
            "Precision": precision_score(y_test, yp),
            "Recall":    recall_score(y_test, yp),
            "F1 Score":  f1_score(y_test, yp),
            "ROC-AUC":   roc_auc_score(y_test, ypr),
        })

    results_df = (pd.DataFrame(results)
                  .sort_values(["F1 Score", "ROC-AUC"], ascending=False)
                  .reset_index(drop=True))

    best_name  = results_df.iloc[0]["Model"]
    best_model = models[best_name]

    bundle = {
        "models":              models,
        "best_model_name":     best_name,
        "best_model":          best_model,
        "preprocessor":        preprocessor,
        "feature_names":       list(preprocessor.get_feature_names_out()),
        "results_df":          results_df,
        "deployment_threshold": 0.50,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return bundle


# ─── LOAD MODEL BUNDLE ─────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    # Always try the absolute path first
    if os.path.exists(BUNDLE_PATH):
        with open(BUNDLE_PATH, "rb") as f:
            return pickle.load(f)
    # Fallback: try relative path (in case cwd == project root)
    rel_path = os.path.join("models", "loan_approval_models.pkl")
    if os.path.exists(rel_path):
        with open(rel_path, "rb") as f:
            return pickle.load(f)
    # Not found — auto-train
    return None


bundle = load_bundle()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🏦 Loan Approval Prediction & Risk Dashboard</h1>
    <p>Automated ML pipeline with SMOTE balancing · Compare Logistic Regression, Decision Tree, Random Forest & Gradient Boosting</p>
</div>
""", unsafe_allow_html=True)

# Auto-train if bundle is missing
if bundle is None:
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Dataset not found at `{DATA_PATH}`.  "
            "Please make sure `loan_prediction.csv` is in the same folder as `app.py`."
        )
        st.stop()
    with st.spinner("Model not found — training from scratch. This takes ~30 seconds..."):
        try:
            bundle = train_and_save_bundle()
            st.cache_resource.clear()
            st.success("Model trained and saved successfully! Reloading...")
            st.rerun()
        except Exception as e:
            st.error(f"Auto-training failed: {e}")
            st.exception(e)
            st.stop()

models_dict      = bundle["models"]
best_model_name  = bundle["best_model_name"]
preprocessor     = bundle["preprocessor"]
results_df       = bundle.get("results_df", pd.DataFrame())
deployment_thr   = bundle.get("deployment_threshold", 0.50)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Applicant Configuration")

    preset = st.selectbox(
        "Quick-Load Sample Profile",
        ["Custom Input",
         "🟢 Strong Profile (High Approval)",
         "🟡 Borderline Profile (Manual Review)",
         "🔴 Weak Profile (Likely Rejection)"]
    )

    # Defaults
    dv = dict(gender="Male", married="Yes", dependents="0", education="Graduate",
              self_employed="No", applicant_income=6000, coapplicant_income=2000.0,
              loan_amount=140.0, term=360, credit_history=1.0, property_area="Urban")

    if preset == "🟢 Strong Profile (High Approval)":
        dv.update(applicant_income=9500, coapplicant_income=4000.0, loan_amount=110.0,
                  credit_history=1.0, property_area="Semiurban")
    elif preset == "🟡 Borderline Profile (Manual Review)":
        dv.update(applicant_income=3800, coapplicant_income=1200.0, loan_amount=200.0,
                  credit_history=1.0, property_area="Rural")
    elif preset == "🔴 Weak Profile (Likely Rejection)":
        dv.update(applicant_income=2200, coapplicant_income=0.0, loan_amount=280.0,
                  credit_history=0.0, property_area="Rural", married="No")

    selected_model_name = st.selectbox(
        "Classification Model",
        list(models_dict.keys()),
        index=list(models_dict.keys()).index(best_model_name)
    )
    st.caption(f"★ Best performing model: **{best_model_name}** (by F1 + ROC-AUC)")

    st.divider()
    st.markdown("**Demographics**")
    gender        = st.selectbox("Gender", ["Male", "Female"],
                                  index=0 if dv["gender"] == "Male" else 1)
    married       = st.selectbox("Married", ["Yes", "No"],
                                  index=0 if dv["married"] == "Yes" else 1)
    dependents    = st.selectbox("Dependents", ["0", "1", "2", "3+"],
                                  index=["0","1","2","3+"].index(dv["dependents"]))
    education     = st.selectbox("Education", ["Graduate", "Not Graduate"],
                                  index=0 if dv["education"] == "Graduate" else 1)
    self_employed = st.selectbox("Self Employed", ["No", "Yes"],
                                  index=0 if dv["self_employed"] == "No" else 1)
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"],
                                  index=["Urban","Semiurban","Rural"].index(dv["property_area"]))

    st.markdown("**Financial Details**")
    applicant_income    = st.number_input("Applicant Monthly Income ($)", 0, 100_000,
                                           int(dv["applicant_income"]), step=500)
    coapplicant_income  = st.number_input("Co-Applicant Monthly Income ($)", 0.0, 50_000.0,
                                           float(dv["coapplicant_income"]), step=500.0)
    loan_amount         = st.number_input("Loan Amount (× $1,000)", 10.0, 1_000.0,
                                           float(dv["loan_amount"]), step=10.0)
    loan_term           = st.selectbox("Loan Term (months)",
                                        [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                                        index=8)
    credit_history      = st.radio(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda x: "Good (No Defaults)" if x == 1.0 else "Bad (Prior Defaults)",
        index=0 if dv["credit_history"] == 1.0 else 1
    )

# ─── BUILD INPUT ──────────────────────────────────────────────────────────────
input_df = pd.DataFrame([{
    "Gender":             gender,
    "Married":            married,
    "Dependents":         dependents,
    "Education":          education,
    "Self_Employed":      self_employed,
    "ApplicantIncome":    applicant_income,
    "CoapplicantIncome":  coapplicant_income,
    "LoanAmount":         loan_amount,
    "Loan_Amount_Term":   loan_term,
    "Credit_History":     credit_history,
    "Property_Area":      property_area
}])

input_proc   = preprocessor.transform(input_df)
clf          = models_dict[selected_model_name]
prob_approved = clf.predict_proba(input_proc)[0][1]
prob_rejected = 1 - prob_approved


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Eligibility Predictor",
    "📊 Model Performance",
    "📈 Data Insights",
    "📁 Batch Prediction"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ELIGIBILITY PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1.1, 1])

    with left:
        # ── Decision card ──────────────────────────────────────────────────────
        if prob_approved >= 0.65:
            color_cls, badge_cls, label = "approved", "green", "LOAN APPROVED"
            emoji = "✅"
        elif prob_approved >= 0.45:
            color_cls, badge_cls, label = "review",   "yellow", "MANUAL REVIEW"
            emoji = "⚠️"
        else:
            color_cls, badge_cls, label = "rejected", "red",   "LOAN REJECTED"
            emoji = "❌"

        bar_color = {"approved": "green", "review": "yellow", "rejected": "red"}[color_cls]
        bar_pct   = int(prob_approved * 100)

        st.markdown(f"""
        <div class="card-{color_cls}">
            <span class="badge badge-{badge_cls}">{emoji} {label}</span>
            <h3 style="color: {'#10b981' if color_cls=='approved' else '#f59e0b' if color_cls=='review' else '#ef4444'};
                       margin: 14px 0 4px 0; font-size:1.55rem;">
                Approval Probability: {prob_approved*100:.1f}%
            </h3>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill-{bar_color}" style="width:{bar_pct}%"></div>
            </div>
            <p style="color:#94a3b8; margin-top:10px; font-size:0.9rem;">
                Model: <b>{selected_model_name}</b> &nbsp;·&nbsp;
                Threshold: {deployment_thr:.0%}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Financial ratios ──────────────────────────────────────────────────
        st.subheader("Financial Ratio Analysis")
        total_income  = applicant_income + coapplicant_income
        loan_dollars  = loan_amount * 1_000
        r_monthly     = 0.085 / 12
        n             = loan_term
        emi = (loan_dollars * r_monthly * (1 + r_monthly)**n) / ((1 + r_monthly)**n - 1) if n > 0 else 0
        dti = (emi / total_income * 100) if total_income > 0 else 0
        lti = (loan_dollars / (total_income * 12))  if total_income > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Household Income",    f"${total_income:,.0f}/mo")
        c2.metric("Est. Monthly EMI",    f"${emi:,.0f}/mo")
        c3.metric("DTI Ratio",           f"{dti:.1f}%",
                  delta="Safe" if dti < 40 else "High!",
                  delta_color="inverse" if dti >= 40 else "normal")
        c4.metric("Loan-to-Income",      f"{lti:.2f}x",
                  delta="Safe" if lti < 3 else "High!",
                  delta_color="inverse" if lti >= 3 else "normal")

        # ── Probability donut ─────────────────────────────────────────────────
        st.subheader("Approval vs. Rejection Probability")
        fig_donut, ax_d = plt.subplots(figsize=(4.5, 4.5), facecolor='none')
        wedge_colors = (['#10b981', '#ef4444'] if prob_approved >= 0.65
                        else ['#f59e0b', '#ef4444'] if prob_approved >= 0.45
                        else ['#ef4444', '#1e293b'])
        wedge_sizes  = [prob_approved, prob_rejected]
        wedges, _ = ax_d.pie(wedge_sizes, colors=wedge_colors, startangle=90,
                              wedgeprops=dict(width=0.5, edgecolor='#0f172a', linewidth=2))
        ax_d.text(0, 0, f"{prob_approved*100:.1f}%",
                  ha='center', va='center', fontsize=20, fontweight='bold', color='white')
        ax_d.set_facecolor('none')
        fig_donut.patch.set_alpha(0)
        st.pyplot(fig_donut, use_container_width=True)
        plt.close(fig_donut)

    with right:
        st.subheader("Risk Factor Assessment")

        # Credit history
        if credit_history == 1.0:
            st.success("**Credit History**: Good standing — strongest positive predictor of approval.")
        else:
            st.error("**Credit History**: Prior defaults detected — primary rejection trigger.")

        # Income
        if total_income >= 6000:
            st.success(f"**Combined Income**: ${total_income:,.0f}/mo — strong debt-coverage capacity.")
        elif total_income >= 3000:
            st.warning(f"**Combined Income**: ${total_income:,.0f}/mo — moderate, loan amount should stay low.")
        else:
            st.error(f"**Combined Income**: ${total_income:,.0f}/mo — below recommended threshold for this loan size.")

        # DTI
        if dti <= 40:
            st.success(f"**Debt-to-Income (DTI)**: {dti:.1f}% — within acceptable range (< 40%).")
        else:
            st.error(f"**Debt-to-Income (DTI)**: {dti:.1f}% — exceeds 40% threshold. Consider lower loan or longer term.")

        # Property area insight
        area_tips = {
            "Semiurban": "**Property Area**: Semiurban — historically highest approval rates in training data.",
            "Urban":     "**Property Area**: Urban — good approval rates with strong income.",
            "Rural":     "**Property Area**: Rural — slightly lower approval rates; income verification is critical."
        }
        st.info(area_tips.get(property_area, ""))

        # Dependents
        if dependents in ["2", "3+"]:
            st.warning(f"**Dependents**: {dependents} — higher household obligations may reduce effective disposable income.")

        st.divider()
        st.subheader("Underwriting Recommendation")
        recs = []
        if credit_history == 0.0:
            recs.append("• Require creditworthy co-signor or collateral asset before reconsidering.")
        if dti > 45:
            recs.append(f"• Reduce requested loan from ${loan_amount:,.0f}K or extend term to reduce EMI below 40% DTI.")
        if total_income < 3000 and loan_amount > 150:
            recs.append("• Income is insufficient for requested loan amount — suggest smaller loan product.")
        if not recs:
            recs.append("• Application satisfies standard automated credit scoring criteria.")
            recs.append("• Proceed to documentation verification & KYC validation.")

        for r in recs:
            st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Benchmark Comparison")

    if not results_df.empty:
        # Style the dataframe
        def highlight_best(df):
            styled = pd.DataFrame('', index=df.index, columns=df.columns)
            for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']:
                if col in df.columns:
                    best_idx = df[col].idxmax()
                    styled.at[best_idx, col] = 'background-color: #1a4a6b; color: #60a5fa; font-weight: bold'
            styled.at[results_df.index[0], 'Model'] = 'color: #f59e0b; font-weight: 700'
            return styled

        fmt_cols = {c: "{:.3f}" for c in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']}
        st.dataframe(results_df.style.format(fmt_cols).apply(highlight_best, axis=None),
                     use_container_width=True)
        st.caption(f"Blue cells = best per metric · Best overall model: **{best_model_name}** (ranked by F1 Score + ROC-AUC)")
    else:
        st.warning("Model results dataframe not found in bundle. Re-run rebuild_bundle.py.")

    st.divider()

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        for img_name, caption in [
            ("roc_curves_all_models.png", "ROC Curve Comparison"),
            ("model_comparison.png",      "All-Metrics Model Comparison")
        ]:
            img_path = os.path.join(IMAGES_DIR, img_name)
            if os.path.exists(img_path):
                st.image(img_path, caption=caption, use_container_width=True)

    with img_col2:
        for img_name, caption in [
            ("confusion_matrix.png",  f"{best_model_name} - Confusion Matrix"),
            ("feature_importance.png", "Top Feature Contributions")
        ]:
            img_path = os.path.join(IMAGES_DIR, img_name)
            if os.path.exists(img_path):
                st.image(img_path, caption=caption, use_container_width=True)

    st.divider()
    st.subheader("Business-Oriented Interpretation")
    st.markdown("""
| Finding | Business Implication |
|---------|---------------------|
| **Credit History is the #1 predictor** | Automate hard rejection for applicants with no/bad credit history |
| **Random Forest achieved highest F1** | Robust to outlier incomes and complex feature interactions |
| **Logistic Regression has highest ROC-AUC** | Preferred when interpretability & threshold-tuning is critical |
| **SMOTE improved minority-class recall** | Reduces risk of missing creditworthy applicants (false negatives) |
| **Semiurban > Urban > Rural approval rates** | Risk-based pricing should reflect geographic loan risk |
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Exploratory Data Analysis – Key Insights")

    img_pairs = [
        ("target_distribution.png",   "Class Imbalance (Y=Approved, N=Rejected)"),
        ("missing_values_heatmap.png", "Missing Values Heatmap"),
        ("credit_history_impact.png",  "Credit History - Approval Rate"),
        ("property_area_impact.png",   "Property Area - Approval Rate"),
        ("numerical_distributions.png","Numerical Feature Distributions"),
        ("smote_class_balance.png",    "SMOTE Balancing Effect"),
        ("correlation_heatmap.png",    "Correlation Between Numerical Features"),
    ]

    rows = [img_pairs[i:i+2] for i in range(0, len(img_pairs), 2)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (img_name, caption) in zip(cols, row):
            img_path = os.path.join(IMAGES_DIR, img_name)
            if os.path.exists(img_path):
                col.image(img_path, caption=caption, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Batch Loan Application Scoring")
    st.markdown("""
Upload a CSV with the same columns as the training dataset. The app will append
**Approval_Probability** and **Predicted_Status** columns.

**Required columns (case-sensitive):**  
`Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area`
    """)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(raw)}** records. Preview:")
        st.dataframe(raw.head(5), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            try:
                batch = raw.copy()
                loan_ids = batch.pop("Loan_ID") if "Loan_ID" in batch.columns else pd.RangeIndex(1, len(batch)+1)

                for col in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']:
                    if col in batch.columns:
                        batch[col] = batch[col].fillna(batch[col].median())
                for col in ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']:
                    if col in batch.columns:
                        batch[col] = batch[col].fillna(batch[col].mode()[0])

                proc   = preprocessor.transform(batch)
                probs  = clf.predict_proba(proc)[:, 1]
                status = ["Approved" if p >= deployment_thr else "Rejected" for p in probs]

                output = raw.copy()
                output["Approval_Probability (%)"] = (probs * 100).round(2)
                output["Predicted_Status"]          = status

                n_approved = sum(1 for s in status if s == "Approved")
                n_rejected = len(status) - n_approved

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Applications",    len(status))
                c2.metric("Approved",              n_approved, delta=f"{n_approved/len(status)*100:.1f}%", delta_color="normal")
                c3.metric("Rejected",              n_rejected, delta=f"{n_rejected/len(status)*100:.1f}%", delta_color="inverse")

                st.dataframe(output, use_container_width=True)
                csv_bytes = output.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV",
                    data=csv_bytes,
                    file_name="loan_batch_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.exception(e)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#475569; font-size:0.82rem;'>"
    "🏦 Loan Approval ML System &nbsp;·&nbsp; "
    "Scikit-Learn · Imbalanced-Learn (SMOTE) · Streamlit &nbsp;·&nbsp; "
    "Models: Logistic Regression · Decision Tree · Random Forest · Gradient Boosting"
    "</center>",
    unsafe_allow_html=True
)
