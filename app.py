import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG & CUSTOM STYLING
# ============================================================
st.set_page_config(
    page_title="Churn Intelligence Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colorful Professional CSS
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 50%, #6a3093 100%);
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-weight: 500;
        font-size: 16px;
    }
    
    /* Title Styling */
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
    }
    .sub-title {
        color: #2a5298;
        font-size: 1.5em;
        font-weight: 600;
        border-left: 5px solid #667eea;
        padding-left: 15px;
        margin: 20px 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s;
    }
    .metric-card:hover { transform: translateY(-5px); }
    .metric-card h3 { font-size: 2em; margin: 0; }
    .metric-card p { margin: 5px 0 0 0; opacity: 0.9; }
    
    /* Risk Level Cards */
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 25px; border-radius: 15px; color: white;
        box-shadow: 0 6px 15px rgba(17, 153, 142, 0.4);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 25px; border-radius: 15px; color: #333;
        box-shadow: 0 6px 15px rgba(247, 151, 30, 0.4);
    }
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 25px; border-radius: 15px; color: white;
        box-shadow: 0 6px 15px rgba(235, 51, 73, 0.4);
    }
    .risk-critical {
        background: linear-gradient(135deg, #8e0e00 0%, #1f1c18 100%);
        padding: 25px; border-radius: 15px; color: white;
        box-shadow: 0 6px 15px rgba(142, 14, 0, 0.5);
    }
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #667eea;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD PRE-TRAINED ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    models = {
        'Logistic Regression': joblib.load('saved_models/logistic_regression_model.pkl'),
        'KNN': joblib.load('saved_models/knn_model.pkl'),
        'Random Forest': joblib.load('saved_models/random_forest_model.pkl'),
        'XGBoost': joblib.load('saved_models/xgboost_model.pkl'),
        'SVM': joblib.load('saved_models/svm_model.pkl')
    }
    return {
        'models': models,
        'metrics': joblib.load('saved_models/metrics.pkl'),
        'roc_data': joblib.load('saved_models/roc_data.pkl'),
        'preprocessor': joblib.load('saved_models/preprocessor.pkl'),
        'num_features': joblib.load('saved_models/num_features.pkl'),
        'cat_features': joblib.load('saved_models/cat_features.pkl'),
        'X_test': joblib.load('saved_models/X_test.pkl'),
        'y_test': joblib.load('saved_models/y_test.pkl'),
    }

artifacts = load_artifacts()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 🎯 Navigation")
page = st.sidebar.radio(
    "Select a Section:",
    ["🏠 Home", "📊 Model Performance", "🔮 Live Prediction",
     "📈 Feature Importance", "🎯 ROC Curves", "🧠 XAI Explainability",
     "💡 Recommendation Engine"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info(
    "**Churn Intelligence Hub** uses 5 ML models to predict customer attrition "
    "and provide actionable business recommendations."
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_risk_category(prob):
    prob_pct = prob * 100
    if prob_pct <= 30:
        return "Low Risk", "risk-low", "🟢", "Maintain current engagement. Standard service protocols are sufficient."
    elif prob_pct <= 50:
        return "Moderate Risk", "risk-moderate", "🟡", "Monitor behavior. Consider loyalty perks & check-in calls."
    elif prob_pct <= 70:
        return "High Risk", "risk-high", "🟠", "Immediate outreach. Offer personalized retention packages."
    else:
        return "Critical Risk", "risk-critical", "🔴", "URGENT: Executive-level intervention. Premium retention offers required."

# ============================================================
# PAGE: HOME
# ============================================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-title">🎯 Customer Churn Intelligence Hub</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>5</h3><p>ML Models</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>10K</h3><p>Training Records</p></div>', unsafe_allow_html=True)
    with col3:
        best_auc = artifacts['metrics']['ROC-AUC'].max()
        st.markdown(f'<div class="metric-card"><h3>{best_auc:.2%}</h3><p>Best ROC-AUC</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>XAI</h3><p>SHAP Powered</p></div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-title">🚀 Dashboard Capabilities</h2>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-box">
        <h3>📊 Model Comparison</h3>
        Compare 5 state-of-the-art models on accuracy, precision, recall, F1, and ROC-AUC.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <h3>🔮 Live Prediction</h3>
        Enter customer data and get instant churn predictions with probability scores.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-box">
        <h3>🧠 XAI Explainability</h3>
        SHAP-powered explanations showing why each prediction was made.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <h3>💡 Smart Recommendations</h3>
        Tier-based business actions based on churn probability (0-30%, 31-50%, 51-70%, 71-100%).
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "📊 Model Performance":
    st.markdown('<h1 class="main-title">📊 Model Performance Metrics</h1>', unsafe_allow_html=True)
    st.markdown("Comprehensive comparison of all trained models on the test dataset.")
    
    # Metrics Table
    st.markdown('<h2 class="sub-title">📋 Performance Table</h2>', unsafe_allow_html=True)
    df_metrics = artifacts['metrics'].copy()
    
    # Highlight best performers
    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: #38ef7d; color: white; font-weight: bold' if v else '' for v in is_max]
    
    styled = df_metrics.style.apply(highlight_best, subset=['Accuracy','Precision','Recall','F1-Score','ROC-AUC'])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    
    # Visualizations
    st.markdown('<h2 class="sub-title">📈 Visual Comparison</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(df_metrics, x='Model', y=['Accuracy','Precision','Recall','F1-Score'],
                    barmode='group', title="Classification Metrics by Model",
                    color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#38ef7d'])
        fig.update_layout(template='plotly_white', height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = px.bar(df_metrics, x='Model', y='ROC-AUC',
                     title="ROC-AUC Comparison",
                     color='Model',
                     color_discrete_sequence=['#667eea','#764ba2','#f093fb','#38ef7d','#f7971e'])
        fig2.update_layout(template='plotly_white', height=450, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Confusion Matrices
    st.markdown('<h2 class="sub-title">🧮 Confusion Matrices</h2>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (name, model) in enumerate(artifacts['models'].items()):
        with cols[i]:
            y_pred = model.predict(artifacts['X_test'])
            cm = confusion_matrix(artifacts['y_test'], y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(name, fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=8)
            ax.set_ylabel('Actual', fontsize=8)
            st.pyplot(fig)
            plt.close()

# ============================================================
# PAGE: LIVE PREDICTION
# ============================================================
elif page == "🔮 Live Prediction":
    st.markdown('<h1 class="main-title">🔮 Live Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown("Enter customer information below to predict churn probability in real-time.")
    
    # Model Selector
    selected_model = st.selectbox(
        "🤖 Select Prediction Model:",
        list(artifacts['models'].keys()),
        index=3  # XGBoost default (best performer)
    )
    
    st.markdown("---")
    st.markdown('<h2 class="sub-title">📝 Customer Information</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider("Credit Score", 350, 850, 650)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 92, 38)
    with col2:
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        balance = st.number_input("Account Balance (€)", 0.0, 260000.0, 76000.0, step=1000.0)
        num_products = st.slider("Number of Products", 1, 4, 1)
    with col3:
        has_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        is_active = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
        salary = st.number_input("Estimated Salary (€)", 0.0, 200000.0, 100000.0, step=1000.0)
    
    if st.button("🚀 Predict Churn Probability", type="primary"):
        # Build input DataFrame
        input_data = pd.DataFrame({
            'CreditScore': [credit_score], 'Geography': [geography], 'Gender': [gender],
            'Age': [age], 'Tenure': [tenure], 'Balance': [balance],
            'NumOfProducts': [num_products], 'HasCrCard': [1 if has_card=="Yes" else 0],
            'IsActiveMember': [1 if is_active=="Yes" else 0], 'EstimatedSalary': [salary]
        })
        
        model = artifacts['models'][selected_model]
        prob = model.predict_proba(input_data)[0][1]
        pred = 1 if prob > 0.5 else 0
        
        risk_label, risk_class, emoji, recommendation = get_risk_category(prob)
        
        st.markdown("---")
        st.markdown(f'<div class="{risk_class}"><h2>{emoji} {risk_label}</h2>'
                   f'<h1>Churn Probability: {prob:.2%}</h1>'
                   f'<p><b>Prediction:</b> {"⚠️ WILL CHURN" if pred else "✅ WILL STAY"}</p>'
                   f'<p><b>Recommendation:</b> {recommendation}</p></div>',
                   unsafe_allow_html=True)
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=prob*100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "black"},
                'bar': {'color': "#764ba2"},
                'steps': [
                    {'range': [0, 30], 'color': '#38ef7d'},
                    {'range': [30, 50], 'color': '#ffd200'},
                    {'range': [50, 70], 'color': '#f45c43'},
                    {'range': [70, 100], 'color': '#8e0e00'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75, 'value': prob*100
                }
            }
        ))
        fig.update_layout(height=350, title=f"{selected_model} - Churn Gauge")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: FEATURE IMPORTANCE
# ============================================================
elif page == "📈 Feature Importance":
    st.markdown('<h1 class="main-title">📈 Feature Importance Analysis</h1>', unsafe_allow_html=True)
    
    selected_model = st.selectbox("Select Model:", list(artifacts['models'].keys()), key="fi_model")
    model = artifacts['models'][selected_model]
    clf = model.named_steps['classifier']
    
    all_features = artifacts['num_features'] + artifacts['cat_features']
    
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        df_imp = pd.DataFrame({'Feature': all_features, 'Importance': importances})
        df_imp = df_imp.sort_values('Importance', ascending=True)
        
        fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h',
                    title=f"{selected_model} - Feature Importance",
                    color='Importance', color_continuous_scale='viridis')
        fig.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠️ {selected_model} doesn't have native feature_importances_. Using SHAP instead.")
        X_test_proc = model.named_steps['preprocessor'].transform(artifacts['X_test'])
        if selected_model == 'Logistic Regression':
            explainer = shap.LinearExplainer(clf, X_test_proc)
        else:
            explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X_test_proc, 100))
        shap_values = explainer.shap_values(X_test_proc[:100])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, feature_names=all_features, show=False)
        st.pyplot(fig)
        plt.close()

# ============================================================
# PAGE: ROC CURVES
# ============================================================
elif page == "🎯 ROC Curves":
    st.markdown('<h1 class="main-title">🎯 ROC Curves & AUC Analysis</h1>', unsafe_allow_html=True)
    
    fig = go.Figure()
    colors = ['#667eea', '#764ba2', '#f093fb', '#38ef7d', '#f7971e']
    
    for i, (name, data) in enumerate(artifacts['roc_data'].items()):
        fig.add_trace(go.Scatter(
            x=data['fpr'], y=data['tpr'],
            mode='lines', name=f"{name} (AUC = {data['auc']:.4f})",
            line=dict(color=colors[i], width=3)
        ))
    
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                            name='Random (AUC = 0.5)',
                            line=dict(color='gray', dash='dash', width=2)))
    
    fig.update_layout(
        title='ROC Curves - All Models Comparison',
        xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        template='plotly_white', height=600,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # AUC Summary
    st.markdown('<h2 class="sub-title">🏆 AUC Rankings</h2>', unsafe_allow_html=True)
    auc_df = artifacts['metrics'][['Model','ROC-AUC']].sort_values('ROC-AUC', ascending=False)
    auc_df['Rank'] = range(1, len(auc_df)+1)
    auc_df['Rating'] = auc_df['ROC-AUC'].apply(
        lambda x: '🥇 Excellent' if x>0.85 else '🥈 Good' if x>0.75 else '🥉 Fair' if x>0.65 else '⚠️ Poor'
    )
    st.dataframe(auc_df[['Rank','Model','ROC-AUC','Rating']], use_container_width=True, hide_index=True)

# ============================================================
# PAGE: XAI EXPLAINABILITY
# ============================================================
elif page == "🧠 XAI Explainability":
    st.markdown('<h1 class="main-title">🧠 Explainable AI (XAI)</h1>', unsafe_allow_html=True)
    st.markdown("Understanding model decisions using **SHAP** (SHapley Additive exPlanations).")
    
    tab1, tab2 = st.tabs(["🌍 Global Explanations", "👤 Local (Single Customer)"])
    
    with tab1:
        selected_model = st.selectbox("Select Model:", list(artifacts['models'].keys()), key="xai_global")
        model = artifacts['models'][selected_model]
        clf = model.named_steps['classifier']
        preproc = model.named_steps['preprocessor']
        
        X_test_proc = preproc.transform(artifacts['X_test'])
        all_features = artifacts['num_features'] + artifacts['cat_features']
        
        with st.spinner("Computing SHAP values..."):
            if selected_model in ['Random Forest', 'XGBoost']:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_test_proc)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            elif selected_model == 'Logistic Regression':
                explainer = shap.LinearExplainer(clf, X_test_proc)
                shap_values = explainer.shap_values(X_test_proc)
            else:
                explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X_test_proc, 100))
                shap_vals = explainer.shap_values(X_test_proc[:100])
                shap_values = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        
        st.markdown("#### 📊 SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(12, 7))
        if selected_model in ['Random Forest', 'XGBoost', 'Logistic Regression']:
            shap.summary_plot(shap_values, feature_names=all_features, show=False, max_display=10)
        else:
            shap.summary_plot(shap_values, feature_names=all_features[:shap_values.shape[1]], show=False, max_display=10)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("#### 🐝 SHAP Beeswarm Plot")
        fig, ax = plt.subplots(figsize=(12, 7))
        if selected_model in ['Random Forest', 'XGBoost', 'Logistic Regression']:
            shap.summary_plot(shap_values, feature_names=all_features, plot_type="dot", show=False, max_display=10)
        else:
            shap.summary_plot(shap_values, feature_names=all_features[:shap_values.shape[1]], plot_type="dot", show=False, max_display=10)
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        st.markdown("### 🔍 Single Customer Explanation")
        sample_idx = st.slider("Select a test customer index:", 0, len(artifacts['X_test'])-1, 0)
        selected_model = st.selectbox("Select Model:", list(artifacts['models'].keys()), key="xai_local")
        
        model = artifacts['models'][selected_model]
        customer = artifacts['X_test'].iloc[[sample_idx]]
        
        prob = model.predict_proba(customer)[0][1]
        risk_label, risk_class, emoji, rec = get_risk_category(prob)
        
        st.markdown(f'<div class="{risk_class}"><h3>{emoji} {risk_label}: {prob:.2%}</h3></div>', unsafe_allow_html=True)
        
        with st.expander("📋 Customer Details"):
            st.dataframe(customer.T, use_container_width=True)
        
        with st.spinner("Computing local SHAP..."):
            preproc = model.named_steps['preprocessor']
            clf = model.named_steps['classifier']
            X_proc = preproc.transform(customer)
            all_features = artifacts['num_features'] + artifacts['cat_features']
            
            if selected_model in ['Random Forest', 'XGBoost']:
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_proc)
                if isinstance(sv, list): sv = sv[1]
            elif selected_model == 'Logistic Regression':
                explainer = shap.LinearExplainer(clf, preproc.transform(artifacts['X_test']))
                sv = explainer.shap_values(X_proc)
            else:
                explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(preproc.transform(artifacts['X_test']), 50))
                svv = explainer.shap_values(X_proc)
                sv = svv[1] if isinstance(svv, list) else svv
            
            fig, ax = plt.subplots(figsize=(10, 6))
            feats = all_features if selected_model in ['Random Forest', 'XGBoost', 'Logistic Regression'] else all_features[:sv.shape[1]]
            base_val = explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1] if len(explainer.expected_value)>1 else explainer.expected_value[0]
            shap.waterfall_plot(shap.Explanation(values=sv[0], feature_names=feats, base_values=base_val), show=False, max_display=10)
            st.pyplot(fig, bbox_inches='tight')
            plt.close()

# ============================================================
# PAGE: RECOMMENDATION ENGINE
# ============================================================
elif page == "💡 Recommendation Engine":
    st.markdown('<h1 class="main-title">💡 Business Recommendation Engine</h1>', unsafe_allow_html=True)
    st.markdown("Tier-based retention strategies based on churn probability.")
    
    # Strategy Cards
    st.markdown('<h2 class="sub-title">🎯 Retention Strategy Matrix</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="risk-low">
        <h2>🟢 LOW RISK (0% - 30%)</h2>
        <ul>
        <li>✅ Maintain standard service quality</li>
        <li>📧 Regular newsletters & updates</li>
        <li>🎁 Annual loyalty rewards</li>
        <li>📊 Periodic satisfaction surveys</li>
        </ul>
        <b>Cost Impact:</b> Minimal
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="risk-high">
        <h2>🟠 HIGH RISK (51% - 70%)</h2>
        <ul>
        <li>📞 Immediate personal outreach</li>
        <li>💰 Customized retention offers</li>
        <li>🎯 Dedicated account manager</li>
        <li>🔄 Product upgrade incentives</li>
        </ul>
        <b>Cost Impact:</b> Significant
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="risk-moderate">
        <h2>🟡 MODERATE RISK (31% - 50%)</h2>
        <ul>
        <li>👀 Monitor engagement metrics</li>
        <li>🎟️ Targeted promotional offers</li>
        <li>💬 Proactive check-in calls</li>
        <li>📱 Personalized content</li>
        </ul>
        <b>Cost Impact:</b> Moderate
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="risk-critical">
        <h2>🔴 CRITICAL RISK (71% - 100%)</h2>
        <ul>
        <li>🚨 Executive-level intervention</li>
        <li>💎 Premium retention packages</li>
        <li>🤝 Win-back campaigns</li>
        <li>📝 Exit interviews if churned</li>
        </ul>
        <b>Cost Impact:</b> Critical Investment
        </div>
        """, unsafe_allow_html=True)
    
    # Batch Analysis
    st.markdown("---")
    st.markdown('<h2 class="sub-title">📊 Portfolio Risk Distribution</h2>', unsafe_allow_html=True)
    
    selected_model = st.selectbox("Model for analysis:", list(artifacts['models'].keys()), key="rec_model")
    model = artifacts['models'][selected_model]
    
    probs = model.predict_proba(artifacts['X_test'])[:, 1]
    risk_buckets = pd.cut(probs, bins=[0, 0.3, 0.5, 0.7, 1.0],
                         labels=['Low (0-30%)', 'Moderate (31-50%)', 'High (51-70%)', 'Critical (71-100%)'])
    
    dist_df = risk_buckets.value_counts().reset_index()
    dist_df.columns = ['Risk Level', 'Count']
    dist_df['Percentage'] = (dist_df['Count'] / dist_df['Count'].sum() * 100).round(2)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = px.pie(dist_df, values='Count', names='Risk Level', hole=0.4,
                    color='Risk Level',
                    color_discrete_map={
                        'Low (0-30%)': '#38ef7d', 'Moderate (31-50%)': '#ffd200',
                        'High (51-70%)': '#f45c43', 'Critical (71-100%)': '#8e0e00'
                    })
        fig.update_layout(height=450, title='Risk Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
        
        st.info(f"""
        **Key Insights:**
        - Total customers analyzed: **{len(probs):,}**
        - Average churn probability: **{probs.mean():.2%}**
        - Highest risk group needs **immediate attention**
        """)