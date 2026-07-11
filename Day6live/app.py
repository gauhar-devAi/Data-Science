import streamlit as st
import pandas as pd
import joblib

# 1. Page Config (Browser tab name & icon)
st.set_page_config(page_title="Telecom Churn Analyzer", page_icon="📊", layout="wide")

# Load model safely
@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

model = load_model()

# Header Section
st.title('📊 Enterprise Customer Churn Predictor')
st.markdown("Deploys optimized Random Forest machine learning to evaluate retention risks and calculate real-time defection probabilities.")
st.markdown("---")

# 2. Main 2-Column Layout
col1, col2 = st.columns([1.1, 0.9], gap="large") # Column sizes slightly adjusted

with col1:
    st.subheader('👤 Customer Profile Attributes')
    
    # Nested Columns inside inputs for clean layout
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        tenure = st.slider('Tenure (months)', 0, 72, 12, help="Number of months the customer has stayed with the company")
        monthly = st.number_input('Monthly Charges ($)', 0.0, 200.0, 65.0, step=5.0)
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    
    with sub_col2:
        internet = st.selectbox('Internet Service Infrastructure', ['DSL', 'Fiber optic', 'No'])
        payment = st.selectbox('Payment Billing Method', 
            ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
        security = st.selectbox('Online Security Add-on', ['Yes', 'No'])
        support = st.selectbox('Premium Tech Support Add-on', ['Yes', 'No'])

with col2:
    st.subheader('🎯 Real-Time Risk Diagnostics')
    
    # Process prediction only when a logical calculation state is needed
    total = tenure * monthly
    
    features = {
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet == 'No' else 0,
        'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card' else 0,
        'OnlineSecurity_Yes': 1 if security == 'Yes' else 0,
        'TechSupport_Yes': 1 if support == 'Yes' else 0,
    }
    
    input_df = pd.DataFrame([features])
    
    # Feature map alignment
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]
    
    # Probabilities
    prob = model.predict_proba(input_df)[0][1]
    prob_percentage = int(prob * 100)
    
    # Display Progress Visual
    st.markdown(f"**Churn Probability Rate:** `{prob_percentage}%`")
    st.progress(prob)
    
    # Executive Condition Alert
    if prob > 0.7:
        st.error(f'🚨 **CRITICAL RISK ({prob_percentage}%)**')
        st.markdown("""
        **Recommended Retention Actions:**
        * 📞 Trigger immediate manual outreach call within 24 hours.
        * 📉 Offer **Contract Upgrade** discount (Move to 1-Year or 2-Year plan).
        * 🎁 Waive premium tech support fees for the next 3 billing cycles.
        """)
    elif prob > 0.4:
        st.warning(f'⚠️ **MEDIUM RISK ({prob_percentage}%)**')
        st.markdown("""
        **Recommended Retention Actions:**
        * 📧 Dispatch an automated customer appreciation loyalty email.
        * 🔄 Suggest transitioning to credit card automatic payment to reduce billing friction.
        """)
    else:
        st.success(f'✅ **STABLE ACCOUNT ({prob_percentage}%)**')
        st.markdown("""
        **Status:** Normal operations. Customer shows high baseline loyalty indicators. No proactive intervention needed.
        """)

st.markdown("---")
# 3. Model Metadata Expansion
with st.expander('📊 Technical Model Validation Metrics'):
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Model Architecture", "Random Forest")
    meta_col2.metric("Validation AUC Score", "0.8421")
    meta_col3.metric("Training Dataset Pool", "7,043 Accounts")