import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 页面配置
st.set_page_config(page_title="大都会核保风险决策看板", layout="wide")

# 1. 业务指标展示区 (Dashboard)
st.title("🛡️ MetLife 核保风险 predictive BI")
st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("AUC 分数", "0.7753", "Logistic Regression")
col2.metric("F1-Score (高风险)", "0.6737", "稳健预测能力")
col3.metric("总样本量", "60,000", "完整数据集")

# 2. 核心逻辑：业务风险驱动力分析 (Insights)
st.subheader("💡 风险关键驱动因子 (Risk Drivers)")
# 使用你模型结果中的特征重要性数据
data_imp = {
    "Feature": ['age_group', 'age_bmi_risk', 'health_score', 'health_deficit', 'log_annual_income', 'log_past_claims', 'claim_to_income_ratio', 'bmi_tier', 'policy_type'],
    "Importance": [0.3769, 0.2412, 0.1866, 0.0627, 0.0380, 0.0373, 0.0372, 0.0134, 0.0067]
}
df_imp = pd.DataFrame(data_imp)

fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                 title="随机森林：特征重要性排序", color='Importance', color_continuous_scale='Viridis')
st.plotly_chart(fig_imp, use_container_width=True)

# 3. 业务 What-if 模拟 (Decision Support)
st.subheader("🎯 核保阈值敏感性模拟")
st.info("通过调整风险判别阈值，观察对‘高风险识别’数量的影响，辅助制定核保策略。")

threshold = st.slider("设定高风险判别概率阈值 (Probability Threshold)", 0.0, 1.0, 0.5)

# 模拟演示：假设我们有一份预测概率列表
st.write(f"当前策略：当模型输出概率 > {threshold} 时，标记为高风险")
st.warning("注：下调阈值将更保守（识别更多风险，但可能增加拒保率）；上调阈值将更开放（降低拒保，但可能漏报风险）。")

# 4. 可解释性输出 (OR 解读)
st.subheader("📊 逻辑回归胜算比 (Odds Ratio) 解读")
or_data = {
    "Factor": ['Age Group', 'Age_BMI_Risk', 'Health_Score'],
    "OR": [1.8841, 1.5938, 0.5825],
    "Insight": ["风险因子 (每级+88%)", "风险因子 (每级+59%)", "保护因子 (每分-42%)"]
}
st.table(pd.DataFrame(or_data))
