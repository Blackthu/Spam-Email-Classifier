import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("Spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Classifier")
st.write("Enter an email message below to predict whether it is **Spam** or **Ham (Not Spam)**.")

email = st.text_area(
    "Email Message",
    height=200,
    placeholder="Type or paste an email here..."
)

if st.button("Predict"):

    if email.strip() == "":
        st.warning("Please enter an email.")
    else:

        transformed = vectorizer.transform([email])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not Spam (Ham)")