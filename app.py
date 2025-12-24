import streamlit as st
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("spam_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- UI ----------------
st.title("📧 Spam Email Detection System")
st.write("This app uses **TF-IDF + Naive Bayes** to detect spam messages.")

st.markdown("---")

message = st.text_area(
    "✍️ Enter your message below:",
    height=150
)

if st.button("🔍 Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        confidence = model.predict_proba([message]).max() * 100

        st.markdown("### 📊 Result")

        if prediction == 1:
            st.error(
                "🚨 **SPAM MESSAGE**\n\n"
                f"Confidence: **{confidence:.2f}%**"
            )
        else:
            st.success(
                "✅ **NOT SPAM (HAM)**\n\n"
                f"Confidence: **{confidence:.2f}%**"
            )

st.markdown("---")
st.caption("Machine Learning Project | Spam Detection")