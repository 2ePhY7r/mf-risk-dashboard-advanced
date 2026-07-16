import streamlit as st
import pandas as pd
import plotly.express as px

# Setting up the Business-Focused Dashboard
st.set_page_config(page_title="Underwriting Risk Control Panel", layout="wide")

st.title("🛡️ MetLife Underwriting Risk Intelligence")
st.markdown("""
This dashboard translates predictive modeling into actionable underwriting guidelines. 
It enables the optimization of the **Loss Ratio** by balancing risk sensitivity against portfolio growth.
""")

# --- 1. Executive Summary (KPIs) ---
st.header("Executive Portfolio Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Predictive AUC", "0.775", "High Discrimination")
c2.metric("Portfolio High-Risk Exposure", "18.4%") 
c3.metric("Underwriting Strategy", "Conservative", "Optimized Threshold")

# --- 2. Interactive Strategy Simulator ---
st.header("🎯 Policy Decision Simulator")
st.info("Adjust the Risk Probability Threshold to recalibrate the underwriting barrier.")

threshold = st.slider("Select Risk Classification Threshold", 0.0, 1.0, 0.5)

st.markdown(f"""
**Strategic Impact at {threshold*100:.0f}% Threshold:**
* **Risk Mitigation:** Identifying individuals with a high probability of loss.
* **Business Action:** Applicants flagged above this threshold are routed to **Physical Exams** or offered **Rated Policies**.
* **Loss Ratio Control:** By lowering this threshold, the company effectively increases the protective barrier against adverse selection.
""")

# --- 3. Business Drivers (Explainability) ---
st.header("Risk Factor Analysis")
st.write("Top drivers influencing high-risk classification, ranked by Random Forest Importance.")

df_drivers = pd.DataFrame({
    "Feature": ['Age Group', 'Age-BMI Interaction', 'Health Score', 'Chronic Disease Burden'],
    "Importance": [37.69, 24.12, 18.66, 6.27]
})

fig = px.bar(df_drivers, x="Importance", y="Feature", orientation='h', color="Importance",
             title="Primary Risk Drivers for Underwriting")
st.plotly_chart(fig, use_container_width=True)

st.success("The Age-BMI interaction is the most significant compounding risk factor, requiring stricter underwriting scrutiny for older, obese applicants.")
