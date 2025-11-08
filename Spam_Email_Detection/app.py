# app.py
import streamlit as st
import pickle
import re
import string
import numpy as np
from scipy.sparse import hstack

# -----------------------
# 1. Load Model and Vectorizer
# -----------------------
with open(r"C:\Users\91990\Projects\Spam_Email_Detection\Models\ensemble_spam_model.pkl", "rb") as f:
    ensemble = pickle.load(f)

with open(r"C:\Users\91990\Projects\Spam_Email_Detection\Models\tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# -----------------------
# 2. Preprocessing and Features
# -----------------------
spam_keywords = [
    'free', 'win', 'prize', 'urgent', 'claim', 'cash', 'offer',
    'click', 'selected', 'congratulations', 'gift', 'winner', 'limited', 'bonus'
]
stop_words = set()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def custom_features(text):
    num_exclamations = text.count('!')
    num_numbers = sum(c.isdigit() for c in text)
    num_keywords = sum(text.count(word) for word in spam_keywords)
    has_money_pattern = int(bool(re.search(r'\$\d+', text)))
    has_earn_sign_pattern = int(bool(re.search(r'(earn|sign up|week)', text)))
    return [num_exclamations, num_numbers, num_keywords, has_money_pattern, has_earn_sign_pattern]

def predict_email(text):
    text_clean = preprocess_text(text)
    tfidf_vect = tfidf.transform([text_clean])
    custom_feat_vect = np.array(custom_features(text_clean)).reshape(1,-1)
    X_vect = hstack([tfidf_vect, custom_feat_vect])
    pred = ensemble.predict(X_vect)
    return "Spam" if pred[0]==1 else "Ham"

# -----------------------
# 3. Streamlit UI
# -----------------------
st.set_page_config(page_title="Spam Email Detector App", page_icon="📩")
st.title("📧 Spam Email Detector App")
st.write("Paste your email below to check if it is **Spam** or **Ham**.")

email_input = st.text_area("Enter Email Text Here:")

if st.button("Predict"):
    if email_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        result = predict_email(email_input)
        if result == "Spam":
            st.error("⚠️ This Email is likely SPAM!")
        else:
            st.success("✅ This Email is HAM.")
