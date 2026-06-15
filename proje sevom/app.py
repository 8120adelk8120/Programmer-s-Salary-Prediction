import streamlit as st
import numpy as np
import pandas as pd
import joblib

# استفاده از دکوراتور کش برای بهینه‌سازی بارگذاری مدل
@st.cache_resource
def load_model():
    return joblib.load('salary_predictor_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("خطا: فایل 'salary_predictor_model.pkl' یافت نشد. ابتدا نوت‌بوک را اجرا کنید.")
    st.stop()

st.title('Salary Prediction (Neural Network Model)')
st.write("""### Predict salary using an optimized Deep Learning structure""")

# لیست کشورهای هماهنگ شده با فیلترها
countries = (
    "Australia", "Austria", "Belgium", "Brazil", "Canada", "Czech Republic", 
    "Denmark", "Finland", "France", "Germany", "Greece", "India", 
    "Iran, Islamic Republic of...", "Israel", "Italy", "Mexico", 
    "Netherlands", "New Zealand", "Norway", "Pakistan", "Poland", 
    "Portugal", "Romania", "Russian Federation", "South Africa", 
    "Spain", "Sweden", "Switzerland", "Turkey", 
    "United Kingdom of Great Britain and Northern Ireland", "United States of America"
)

education = (
    "Less than a Bachelors",
    "Bachelor’s degree",
    "Master’s degree",
    "Post grad"
)

country_input = st.selectbox("Country", countries)
education_input = st.selectbox("Education Level", education)
experience_input = st.slider("Years of Experience", 0, 50, 3)

columns = ['Country', 'EdLevel', 'YearsCodePro']

ok = st.button("Calculate Salary")
if ok:
    X_new_df = pd.DataFrame([[country_input, education_input, float(experience_input)]], columns=columns)
    
    # پیش‌بینی مدل عصبی و تبدیل خروجی لگاریتمی به دلار واقعی
    salary_log = model.predict(X_new_df)
    salary = np.expm1(salary_log)
    
    st.subheader(f"The estimated salary is ${salary[0]:,.2f}")