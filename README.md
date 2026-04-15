# 🧠 Fake News Detection App

An end-to-end Natural Language Processing (NLP) project that detects whether a news article is **Fake** or **Real** using Machine Learning.

---

## 🚀 Project Overview

This project covers the full NLP pipeline:

- Data Collection (True.csv & Fake.csv)
- Data Cleaning & Preprocessing
- Feature Engineering (TF-IDF & Bag of Words)
- Model Training (Logistic Regression)
- Model Evaluation
- Deployment using Streamlit

---

## 📊 Dataset

- Two datasets:
  - True News
  - Fake News
- Combined into a single dataset with labels:
  - `1` → Real News
  - `0` → Fake News

---

## ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- NLP Techniques (TF-IDF, BoW)
- Streamlit (for deployment)

---

## 🧠 Model Performance

| Technique       | Accuracy |
|----------------|----------|
| TF-IDF         | ~98.3%   |
| Bag of Words   | ~99.1%   |

---

## 🎯 Features

- Clean and preprocess text data
- Compare multiple feature engineering techniques
- Train and evaluate ML models
- Interactive Streamlit app
- Confidence score for predictions
- Example news testing buttons

---

## 🖥️ How to Run the App

### 1. Install requirements
```bash
pip install -r requirements.txt

### 2. Run Streamlit app
streamlit run app.py

👨‍💻 Author

Mahmoud Alamwy