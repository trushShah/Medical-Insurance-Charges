# Medical Insurance Charges — Analytics & Prediction Dashboard

An end-to-end data science application bridging rigorous statistical inference and interactive web deployment. This project explores the structural determinants of medical insurance costs through Exploratory Data Analysis (EDA), parametric/non-parametric hypothesis testing, OLS regression modeling, and a deployed Streamlit user interface.

---

## 🌟 Live Demo
Experience the live application deployed on Streamlit Community Cloud:
👉 **[Access Medical Cost Dashboard Here](https://medical-insurance-charges-by-trush.streamlit.app/)**

---

## 🔍 Project Overview & Architecture
Understanding healthcare cost drivers is critical for risk assessment and actuarial modeling. This repository encompasses:
1. **Exploratory Data Analysis (EDA) & Diagnostics:** Evaluating data distributions, missing values, skewness, and multi-collinearity.
2. **Statistical Hypothesis Testing:** Rigorous assumption checks followed by parametric and non-parametric testing to evaluate subgroup differences (e.g., smokers vs. non-smokers, regional cost variations).
3. **Predictive Modeling:** Fitting an Ordinary Least Squares (OLS) regression model complete with comprehensive residual diagnostics (Jarque-Bera, Durbin-Watson).
4. **Interactive Dashboard:** A production-ready web application enabling users to input individual health metrics and receive real-time cost estimations.

---

## 📊 Dataset Summary
The model is trained on structural healthcare parameters including:
* **`age`**: Age of the primary beneficiary.
* **`sex`**: Contractor gender (`female`, `male`).
* **`bmi`**: Body Mass Index, providing understanding of body weight relative to height.
* **`children`**: Number of children/dependents covered by insurance.
* **`smoker`**: Smoking status (`yes`, `no`), serving as a primary cost driver.
* **`region`**: Beneficiary's residential area in the US (`northeast`, `southeast`, `northwest`, `southwest`).
* **`charges`**: Individual medical costs billed by health insurance.

---

## 📈 Synthesis of Statistical Findings
* **Normality & Non-Parametric Testing:** Because medical charges exhibit severe right-skewness, normality assumptions fail under the Shapiro-Wilk test. Consequently, the **Mann-Whitney $U$ test** was deployed to validate significant cost inflation among smokers versus non-smokers.
* **Regional Variance:** One-Way ANOVA tests evaluated regional cost variations, showing structural parity across certain sectors while highlighting localized variance.
* **OLS Regression Performance:** The final Ordinary Least Squares model yielded an $R^2$ score of approximately **0.751**, confirming that variables like smoking status, age, and BMI explain the vast majority of variance in medical billing charges.

---

<table>
  <tr>
    <th align="left"></th>
    <th align="left">Execution & Workflow Details</th>
  </tr>
  <tr>
    <td><b>Local Installation & Setup</b></td>
    <td>
      <b>1. Clone the Repository:</b><br>
      <code>git clone https://github.com/trushShah/Medical-Insurance-Charges.git</code><br>
      <code>cd Medical-Insurance-Charges</code><br><br>
      <b>2. Set Up Virtual Environment:</b><br>
      <code>python -m venv .venv</code><br>
      <code>.venv\Scripts\Activate.ps1  # (Use: source .venv/bin/activate on Mac/Linux)</code><br><br>
      <b>3. Install Dependencies:</b><br>
      <code>pip install -r app/requirements.txt</code><br><br>
      <b>4. Launch Dashboard:</b><br>
      <code>streamlit run app/app.py</code> (Opens automatically at <code>http://localhost:8501</code>)
    </td>
  </tr>
</table>
