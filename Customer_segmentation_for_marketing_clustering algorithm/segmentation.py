import numpy as np
import pandas as pd
import streamlit as st
import joblib

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Customer Segmentation App')
st.write('Enter customer details to predict the segment')

Age = st.number_input('Age', min_value=18, max_value=100, value=35)
Income = st.number_input('Income', min_value=0, max_value=200000, value=50000)
Total_spending = st.number_input('Total_spending', min_value=0, )
Num_web_purchases = st.number_input('Number of Web Purchases', min_value=0, max_value=100, value=10)
Num_store_processes = st.number_input('Number of Store Purchases', min_value=0, max_value=100, value=10)
Num_web_visit = st.number_input('Number of Web Visit', min_value=0, max_value=50, value=3)
Recency = st.number_input('Recency (Day since last purchase)', min_value=0, max_value=100, value=10)

input_data = pd.DataFrame({
    'Age': [Age],
    'Income': [Income],
    'Total_Spending': [Total_spending],
    'NumWebPurchases': [Num_web_purchases],
    'NumStorePurchases': [Num_store_processes],
    'NumWebVisitsMonth': [Num_web_visit],
    'Recency': [Recency]
})

feature_order = [
    'Age',
    'Income',
    'Total_Spending',
    'NumWebPurchases',
    'NumStorePurchases',
    'NumWebVisitsMonth',
    'Recency'
]

input_data = input_data[feature_order]
input_scaled = scaler.transform(input_data)
