import streamlit as st
import pickle
import re
import random
# =========================
# Load Model & Vectorizer
# =========================
model = pickle.load(open("models/tfidf_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

# =========================
# Text Cleaning Function
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# =========================
# Custom UI Styling
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cfcfcf;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 15px;
}
.real {
    background-color: #1f7a1f;
}
.fake {
    background-color: #8b0000;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown('<p class="title">🧠 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a news article and detect if it is Fake or Real</p>', unsafe_allow_html=True)

# =========================
# Example Buttons
# =========================
real_news_list = [
    "The government announced new economic reforms to boost infrastructure and job creation.",
    "Scientists developed a new vaccine that shows high effectiveness against viral infections.",
    "The central bank confirmed stability in the national currency after recent reforms."
]

fake_news_list = [
    "Scientists confirmed that aliens are secretly living among humans and controlling governments.",
    "A new study proves that drinking coffee makes humans invisible for 5 minutes.",
    "NASA announced a hidden planet made entirely of gold near the solar system."
]

st.markdown("### ⚡ Try Random News Examples")

col1, col2 = st.columns(2)

with col1:
    if st.button("📰 Real News Example"):
        st.session_state["news_input"] = random.choice(real_news_list)

with col2:
    if st.button("🚨 Fake News Example"):
        st.session_state["news_input"] = random.choice(fake_news_list)
# =========================
# Input Area
# =========================
user_input = st.text_area(
    "✍ Enter News Text Here:",
    height=200,
    value=st.session_state.get("news_input", "")
)

# =========================
# Buttons
# =========================
col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🚀 Predict")

with col2:
    clear_btn = st.button("🧹 Clear")

if clear_btn:
    st.session_state["news_input"] = ""
    st.rerun()

# =========================
# Prediction
# =========================
if predict_btn:

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text first!")
    
    else:
        cleaned_text = clean_text(user_input)
        vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]
        confidence = max(prob) * 100

        st.markdown("### 📊 Result")

        # Progress Bar
        st.progress(int(confidence))

        # Result Box
        if prediction == 1:
            st.markdown(
                f'<div class="result-box real">✅ REAL NEWS<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box fake">❌ FAKE NEWS<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )