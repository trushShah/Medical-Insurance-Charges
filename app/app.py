"""
Medical Insurance Charges — Interactive Streamlit Dashboard
Converted from 202518027_assignment_4.ipynb

Run with:  streamlit run app.py
Requires:  insurance.csv in the same folder as this script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Charges Dashboard",
    layout="wide",
)

NUM_COLS = ["age", "bmi", "children", "charges"]
CAT_COLS = ["sex", "smoker", "region"]
ALPHA = 0.05


# ----------------------------------------------------------------------------
# Data / model loading (cached so the app stays fast on interaction)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df


@st.cache_resource
def fit_model(df: pd.DataFrame):
    formula = "charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)"
    model = smf.ols(formula, data=df).fit()
    return model


@st.cache_data
def compute_vif(df: pd.DataFrame):
    _, X_design = dmatrices(
        "charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)",
        data=df,
        return_type="dataframe",
    )
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_design.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_design.values, i)
        for i in range(X_design.shape[1])
    ]
    return vif_data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Could not find **insurance.csv** in the app folder. "
        "Place the dataset next to app.py and refresh."
    )
    st.stop()

model = fit_model(df)

st.title("🏥 Medical Insurance Charges — Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🧪 Hypothesis Testing Lab", "🔮 Live Prediction & Diagnostics"]
)

# ============================================================================
# TAB 1 — DATA EXPLORATION
# ============================================================================
with tab1:
    st.header("Data Exploration")

    # ---- Sidebar filters (this tab only) -----------------------------------
    st.sidebar.header("🔎 Data Exploration Filters")

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.sidebar.slider(
        "Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max)
    )

    charge_min, charge_max = float(df["charges"].min()), float(df["charges"].max())
    charge_range = st.sidebar.slider(
        "Charges (bill) range",
        min_value=float(np.floor(charge_min)),
        max_value=float(np.ceil(charge_max)),
        value=(float(np.floor(charge_min)), float(np.ceil(charge_max))),
    )

    region_sel = st.sidebar.multiselect(
        "Region", options=sorted(df["region"].unique()), default=sorted(df["region"].unique())
    )
    sex_sel = st.sidebar.multiselect(
        "Sex", options=sorted(df["sex"].unique()), default=sorted(df["sex"].unique())
    )
    smoker_sel = st.sidebar.multiselect(
        "Smoker", options=sorted(df["smoker"].unique()), default=sorted(df["smoker"].unique())
    )

    filtered = df[
        (df["age"].between(age_range[0], age_range[1]))
        & (df["charges"].between(charge_range[0], charge_range[1]))
        & (df["region"].isin(region_sel))
        & (df["sex"].isin(sex_sel))
        & (df["smoker"].isin(smoker_sel))
    ]

    st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** records after filtering.")

    if filtered.empty:
        st.warning("No records match the current filters. Adjust the sidebar controls.")
    else:
        # ---- Summary statistics --------------------------------------------
        st.subheader("Dataset Summary Statistics")
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("**Descriptive metrics (numerical features)**")
            desc_stats = pd.DataFrame(
                {
                    "Mean": filtered[NUM_COLS].mean(),
                    "Median": filtered[NUM_COLS].median(),
                    "Std Dev": filtered[NUM_COLS].std(),
                    "IQR": filtered[NUM_COLS].quantile(0.75) - filtered[NUM_COLS].quantile(0.25),
                    "Skewness": filtered[NUM_COLS].skew(),
                    "Kurtosis": filtered[NUM_COLS].kurtosis(),
                }
            )
            st.dataframe(desc_stats.style.format("{:.3f}"), use_container_width=True)

        with col_b:
            st.markdown("**Categorical breakdown**")
            for c in CAT_COLS:
                st.write(f"*{c}*")
                st.dataframe(
                    filtered[c].value_counts().rename("count").to_frame(),
                    use_container_width=True,
                )

        st.divider()

        # ---- Reactive plots --------------------------------------------------
        st.subheader("Distributions")
        dist_col = st.selectbox("Choose a numerical feature to plot", NUM_COLS, index=3)
        fig_hist = px.histogram(
            filtered, x=dist_col, nbins=30, marginal="box",
            color="smoker", barmode="overlay", opacity=0.7,
            title=f"Distribution of {dist_col}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("**Correlation matrix (numerical features)**")
            corr = filtered[NUM_COLS].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Correlation Matrix",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_d:
            st.markdown("**Charges vs. a numerical feature**")
            x_feat = st.selectbox(
                "X-axis feature", ["age", "bmi", "children"], key="scatter_x"
            )
            fig_scatter = px.scatter(
                filtered, x=x_feat, y="charges", color="smoker",
                hover_data=["sex", "region"],
                title=f"Charges vs. {x_feat}",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("**Average charges by category**")
        cat_choice = st.selectbox("Group by", CAT_COLS, key="bar_cat")
        bar_data = filtered.groupby(cat_choice, as_index=False)["charges"].mean()
        fig_bar = px.bar(
            bar_data, x=cat_choice, y="charges",
            title=f"Average Charges by {cat_choice}",
            color=cat_choice,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# ============================================================================
with tab2:
    st.header("Hypothesis Testing Lab")
    st.write(
        "Pick a categorical factor and a numerical metric. The app automatically "
        "checks the test assumptions (normality, equal variance) and runs the "
        "appropriate statistical test."
    )

    col1, col2 = st.columns(2)
    with col1:
        factor = st.selectbox("Categorical factor", CAT_COLS, index=1)  # default: smoker
    with col2:
        metric = st.selectbox("Numerical metric", NUM_COLS, index=3)  # default: charges

    groups_labels = sorted(df[factor].dropna().unique())
    groups = [df.loc[df[factor] == g, metric].dropna() for g in groups_labels]
    n_groups = len(groups_labels)

    st.markdown(f"**Groups found in `{factor}`:** {', '.join(map(str, groups_labels))}")

    # ---- Assumption checks --------------------------------------------------
    st.subheader("1. Assumption Checks")

    normal_flags = []
    shapiro_rows = []
    for label, g in zip(groups_labels, groups):
        sample = g.sample(min(len(g), 1000), random_state=42) if len(g) > 3 else g
        if len(sample) >= 3:
            stat_sw, p_sw = stats.shapiro(sample)
            normal_flags.append(p_sw > ALPHA)
            shapiro_rows.append({"Group": label, "Shapiro-Wilk p-value": p_sw, "Normal?": p_sw > ALPHA})
        else:
            normal_flags.append(False)
            shapiro_rows.append({"Group": label, "Shapiro-Wilk p-value": np.nan, "Normal?": False})

    shapiro_df = pd.DataFrame(shapiro_rows)
    st.dataframe(shapiro_df.style.format({"Shapiro-Wilk p-value": "{:.3e}"}), use_container_width=True)

    levene_stat, levene_p = stats.levene(*groups)
    equal_var = levene_p > ALPHA
    st.write(f"**Levene's Test for equal variance:** statistic = `{levene_stat:.4f}`, "
             f"p-value = `{levene_p:.3e}` → {'Equal variances ✅' if equal_var else 'Unequal variances ⚠️'}")

    all_normal = all(normal_flags)

    # ---- Test selection & execution -----------------------------------------
    st.subheader("2. Test Selection & Result")

    if n_groups < 2:
        st.warning("Need at least 2 groups to run a hypothesis test.")
    elif n_groups == 2:
        if all_normal and equal_var:
            test_name = "Independent samples t-test (Student's t-test)"
            test_stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=True)
        elif all_normal and not equal_var:
            test_name = "Welch's t-test (unequal variances)"
            test_stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        else:
            test_name = "Mann-Whitney U test (non-parametric)"
            test_stat, p_value = stats.mannwhitneyu(groups[0], groups[1])
    else:
        if all_normal and equal_var:
            test_name = "One-Way ANOVA"
            test_stat, p_value = stats.f_oneway(*groups)
        else:
            test_name = "Kruskal-Wallis H-test (non-parametric)"
            test_stat, p_value = stats.kruskal(*groups)

    decision = "Reject H0" if p_value < ALPHA else "Fail to Reject H0"
    decision_color = "🔴" if p_value < ALPHA else "🟢"

    m1, m2, m3 = st.columns(3)
    m1.metric("Test Used", test_name)
    m2.metric("Test Statistic", f"{test_stat:.4f}")
    m3.metric("p-value", f"{p_value:.3e}")

    st.markdown(
        f"### {decision_color} Conclusion: **{decision}** at α = {ALPHA}"
    )
    if p_value < ALPHA:
        st.success(
            f"There is a statistically significant difference in **{metric}** "
            f"across the groups of **{factor}** (p = {p_value:.3e} < {ALPHA})."
        )
    else:
        st.info(
            f"There is **no** statistically significant difference in **{metric}** "
            f"across the groups of **{factor}** (p = {p_value:.3e} ≥ {ALPHA})."
        )

    st.subheader("3. Visual Comparison")
    fig_box = px.box(df, x=factor, y=metric, color=factor, points="outliers",
                      title=f"{metric} by {factor}")
    st.plotly_chart(fig_box, use_container_width=True)

# ============================================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# ============================================================================
with tab3:
    st.header("Live Prediction & Diagnostics")
    st.write(
        "Model: `charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)` "
        "(OLS regression). Adjust the inputs below to generate a real-time prediction."
    )

    st.subheader("1. Enter Policyholder Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        in_age = st.slider("Age", int(df["age"].min()), int(df["age"].max()), int(df["age"].median()))
        in_sex = st.radio("Sex", sorted(df["sex"].unique()), horizontal=True)
    with c2:
        in_bmi = st.number_input(
            "BMI", min_value=float(df["bmi"].min()), max_value=float(df["bmi"].max()),
            value=float(round(df["bmi"].median(), 1)), step=0.1,
        )
        in_smoker = st.radio("Smoker", sorted(df["smoker"].unique()), horizontal=True)
    with c3:
        in_children = st.number_input(
            "Children", min_value=int(df["children"].min()), max_value=int(df["children"].max()),
            value=int(df["children"].median()), step=1,
        )
        in_region = st.selectbox("Region", sorted(df["region"].unique()))

    new_point = pd.DataFrame(
        {
            "age": [in_age],
            "bmi": [in_bmi],
            "children": [in_children],
            "sex": [in_sex],
            "smoker": [in_smoker],
            "region": [in_region],
        }
    )

    pred = model.get_prediction(new_point).summary_frame(alpha=ALPHA)
    point_pred = pred["mean"].iloc[0]
    ci_low, ci_high = pred["mean_ci_lower"].iloc[0], pred["mean_ci_upper"].iloc[0]
    pi_low, pi_high = pred["obs_ci_lower"].iloc[0], pred["obs_ci_upper"].iloc[0]

    st.subheader("2. Prediction")
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted Charges", f"${point_pred:,.2f}")
    p2.metric(f"{int((1-ALPHA)*100)}% Confidence Interval", f"${ci_low:,.0f} – ${ci_high:,.0f}")
    p3.metric(f"{int((1-ALPHA)*100)}% Prediction Interval", f"${pi_low:,.0f} – ${pi_high:,.0f}")
    st.caption(
        "The confidence interval reflects uncertainty in the *average* charges for this profile; "
        "the (wider) prediction interval reflects uncertainty for a *single new* policyholder."
    )

    st.divider()
    st.subheader("3. Model Diagnostics (Gauss-Markov Checks)")

    fitted_vals = model.fittedvalues
    residuals = model.resid

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Residuals vs. Fitted Values** (Linearity & Homoscedasticity)")
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        ax1.scatter(fitted_vals, residuals, alpha=0.5)
        ax1.axhline(0, color="red", linestyle="--")
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs. Fitted Values")
        st.pyplot(fig1)

    with d2:
        st.markdown("**Q-Q Plot of Residuals** (Normality Check)")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sm.qqplot(residuals, line="45", fit=True, ax=ax2)
        ax2.set_title("Q-Q Plot of Residuals")
        st.pyplot(fig2)

    st.markdown("**Variance Inflation Factor (Multicollinearity Check)**")
    vif_data = compute_vif(df)
    st.dataframe(
        vif_data.style.format({"VIF": "{:.3f}"}).background_gradient(subset=["VIF"], cmap="Reds"),
        use_container_width=True,
    )
    st.caption("Rule of thumb: VIF > 5–10 suggests problematic multicollinearity for that predictor.")

    with st.expander("Show full OLS regression summary"):
        st.text(model.summary())
