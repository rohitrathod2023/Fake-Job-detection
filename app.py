# app.py
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
model = joblib.load('random_forest_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app
st.title("Job Posting Fraud Detection")

# Input fields for job posting details
description = st.text_area("Job Description")
company_profile = st.text_area("Company Profile")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")

# Combine text fields
text = description + ' ' + company_profile + ' ' + requirements + ' ' + benefits

# Predict button
if st.button("Predict"):
    # Transform the input text using TF-IDF
    text_tfidf = tfidf.transform([text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)
    
    # Display result
    if prediction[0] == 1:
        st.error("This job posting is likely fraudulent.")
    else:
       st.success("This job posting is likely legitimate.")
