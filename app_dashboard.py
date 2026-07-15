import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="MetLife Underwriting & Risk Sandbox v2.0", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ MetLife Premium Risk & Underwriting Optimization Sandbox (v2.0)")
st.markdown("---")

# ==========================================
# 2. 数据加载与处理
# ==========================================
@st.cache_data
def load_and_transform_data():
    df = pd.read_excel("insurance_test_data.xlsx", sheet_name="Sheet1")
    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())
    df['age'] = df['age'].fillna(df['age'].mean())
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    
    df_clean = df.copy()
    df_clean['raw_age'] = df_clean['age']
    df_clean['raw_bmi'] = df_clean['bmi']
    df_clean['policy_type_orig'] = df_clean['policy_type']
    
    # 特征工程
    df_clean['age_group'] = pd.cut(df_clean['age'], bins=[0, 35, 50, 65, 120], labels=[0, 1, 2, 3]).fillna(0).astype(int)
    df_clean['bmi_tier'] = pd.cut(df_clean['bmi'], bins=[0, 25.0, 30.0, 100.0], labels=[0, 1, 2]).fillna(0).astype(int)
    df_clean['health_deficit'] = (100 - df_clean['health_score']) * df_clean['has_chronic_disease']
    df_clean['age_bmi_risk'] = df_clean['age_group'] * df_clean['bmi_tier']
    df_clean['log_annual_income'] = np.log1p(df_clean['annual_income'])
    df_clean['log_past_claims'] = np.log1p(df_clean['past_claims_amount'])
    
    policy_map = {'Basic': 0, 'Premium': 1, 'Platinum': 2}
    df_clean['policy_type_encoded'] = df_clean['policy_type'].map(policy_map).fillna(0).astype(int)
    df_clean['claim_to_income_ratio'] = df_clean['past_claims_amount'] / (df_clean['annual_income'] + 1)
    
    return df_clean

df_fe = load_and_transform_data()
feature_cols = ['age_group', 'bmi_tier', 'health_deficit', 'age_bmi_risk', 'log_annual_income', 'health_score', 'log_past_claims', 'policy_type_encoded', 'claim_to_income_ratio']

# ==========================================
# 3. 模型训练
# ==========================================
@st.cache_resource
def train_models(data):
    X = data[feature_cols]
    y = data['is_high_risk']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    lr = LogisticRegression(random_state=42).fit(X_scaled_df, y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(X_scaled_df, y)
    return lr, rf, scaler, X_scaled_df

lr_model, rf_model, scaler, X_scaled_df = train_models(df_fe)

# ==========================================
# 4. 侧边栏与数据流
# ==========================================
model_choice = st.sidebar.selectbox("Choose Model:", ["Logistic Regression", "Random Forest"])
threshold = st.sidebar.slider("Threshold:", 0.1, 0.9, 0.45)
probs = lr_model.predict_proba(X_scaled_df)[:, 1] if "Logistic" in model_choice else rf_model.predict_proba(X_scaled_df)[:, 1]
df_fe['predicted_prob'] = probs
df_fe['dynamic_pred'] = (df_fe['predicted_prob'] >= threshold).astype(int)

# ==========================================
# 5. 可视化 (已添加唯一 key 修复 ID 冲突，并修正深色主题下对比度)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Interpretability")
    if "Logistic" in model_choice:
        imp_data = pd.DataFrame({'Feature': feature_cols, 'Value': np.exp(lr_model.coef_[0])})
        fig = px.bar(imp_data, x='Value', y='Feature', template="plotly_dark")
    else:
        imp_data = pd.DataFrame({'Feature': feature_cols, 'Value': rf_model.feature_importances_})
        fig = px.bar(imp_data, x='Value', y='Feature', template="plotly_dark")
    
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True, key="imp_chart_unique_001")

with col2:
    st.subheader("Metabolic Risk Compound")
    fig2 = px.scatter(df_fe.sample(min(500, len(df_fe))), x="health_score", y="age_bmi_risk", color="dynamic_pred", template="plotly_dark")
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
    st.plotly_chart(fig2, use_container_width=True, key="scatter_chart_unique_001")

# ==========================================
# 6. 个案下钻门户
# ==========================================
st.markdown("---")
st.subheader("🔍 Automated Underwriting Portal")
search_id = st.selectbox("Select Customer ID for Audit:", options=df_fe['customer_id'].unique(), key="cust_search_id")

customer_row = df_fe[df_fe['customer_id'] == search_id].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customer Age", f"{int(customer_row['raw_age'])} (Tier {int(customer_row['age_group'])})")
c2.metric("Health Score", f"{customer_row['health_score']:.1f}")
c3.metric("BMI Index", f"{customer_row['raw_bmi']:.2f} (Tier {int(customer_row['bmi_tier'])})")
c4.metric("Risk Prob", f"{customer_row['predicted_prob']*100:.1f}%")

if customer_row['predicted_prob'] >= threshold:
    st.error(f"⚠️ System Verdict: DENY / SURCHARGE.")
else:
    st.success(f"✅ System Verdict: AUTO-APPROVE.")