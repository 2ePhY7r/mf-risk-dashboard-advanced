import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Load Data - Use the actual model output (probabilities)
@st.cache_data
def get_model_inference():
    # Assuming your model output contains 'id', 'actual_risk', and 'predicted_probability'
    df = pd.read_csv("model_results.csv") 
    return df

df = get_model_inference()

# 2. Business Logic: Define the "Impact"
# Add a column for 'potential_loss' (assume an average claim cost for demonstration)
avg_claim_cost = 50000 

st.title("🛡️ Underwriting Profitability Simulator")

# Interactive Sidebar for Business Strategy
st.sidebar.header("Strategy Settings")
threshold = st.sidebar.slider("Acceptance Threshold", 0.0, 1.0, 0.5, help="Applicants above this risk score will be rejected or referred.")

# 3. Dynamic Calculation
# Filter data based on the business user's chosen threshold
high_risk_flagged = df[df['predicted_probability'] > threshold]
potential_savings = len(high_risk_flagged) * avg_claim_cost

# Display Metrics that business people actually care about
col1, col2, col3 = st.columns(3)
col1.metric("Risk Policies Blocked", len(high_risk_flagged))
col2.metric("Potential Claims Avoided", f"${potential_savings:,}")
col3.metric("Retention Rate", f"{ (1 - (len(high_risk_flagged)/len(df)))*100:.1f}%")

# 4. Visualization: "Risk Distribution"
fig = px.histogram(df, x='predicted_probability', nbins=50, title="Population Risk Distribution")
fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
st.plotly_chart(fig)
