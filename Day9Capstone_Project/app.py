import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HR Attrition Predictor", layout="centered")

st.title('HR Employee Attrition Risk Predictor')
st.caption('Powered by Random Forest Model · Trained on IBM HR Data Engine')

# Loading serialized engine components
model = joblib.load('attrition_model.pkl')
features = joblib.load('feature_names.pkl')

col1, col2 = st.columns(2)
with col1:
    age = st.slider('Age', 18, 60, 35)
    income = st.number_input('Monthly Income ($)', 1000, 20000, 5000)
    overtime = st.selectbox('OverTime Status', ['No', 'Yes'])
    years = st.slider('Years at Company', 0, 40, 5)
    satisfaction = st.slider('Job Satisfaction Rating (1-4)', 1, 4, 3)

with col2:
    st.subheader("Risk Evaluation Output")
    if st.button('Predict Attrition Risk', use_container_width=True):
        input_dict = {f: 0 for f in features}
        input_dict['Age'] = age
        input_dict['MonthlyIncome'] = income
        input_dict['YearsAtCompany'] = years
        input_dict['JobSatisfaction'] = satisfaction
        input_dict['OverTime'] = 1 if overtime == 'Yes' else 0

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[features] # Maintain rigid matrix formatting constraints
        prob = model.predict_proba(input_df)[0][1]

        st.metric(label='Calculated Attrition Probability', value=f'{prob*100:.1f}%')
        st.progress(int(prob*100))

        # Core Strategic Multi-Risk Evaluation Conditions
        if prob > 0.60:
            st.error('🚨 High Risk — immediate retention and intervention action needed.')
        elif prob > 0.35:
            st.warning('⚠️ Medium Risk — monitor closely')
        else:
            st.success('✅ Low Risk — employee is stable')
