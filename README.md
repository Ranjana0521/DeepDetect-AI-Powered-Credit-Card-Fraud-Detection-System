# 💳 DeepDetect: AI-Powered Credit Card Fraud Detection System

The system combines **advanced ML models, explainable AI, and an interactive analytics dashboard** to detect fraudulent activity while maintaining transparency in decision-making.

Built with a **production-style architecture**, this project demonstrates how AI can be applied in financial systems for **real-time fraud detection and risk monitoring**.

---

# 🌐 Live Application

🔗 **Live Demo**

https://deepdetect-ai-powered-credit-card-fraud-detection-system-xbrpn.streamlit.app/

---

#  Key Features

## 🔎 Real-Time Fraud Detection

* Predicts fraud probability instantly
* Accepts **transaction amount and time**
* Displays **fraud probability score**
* Automatic **risk classification**

## 📊 Interactive Analytics Dashboard

* Fraud vs Legitimate transaction visualization
* Transaction amount distribution
* Quick insights into transaction behaviour

##  Explainable AI

* Uses **SHAP (SHapley Additive Explanations)**
* Shows which features influenced predictions
* Improves transparency of AI decisions

## 📈 Model Performance Evaluation

* Confusion Matrix
* ROC Curve with AUC score
* Visualization of model effectiveness

## ⚡ Fraud Risk Visualization

* Interactive fraud risk gauge
* Risk categories:

  * LOW
  * MEDIUM
  * HIGH
    
---

# ⚙️ Technology Stack

### Programming

Python

### Machine Learning

Scikit-Learn
XGBoost
SHAP

### Data Processing

Pandas
NumPy

### Visualization

Matplotlib
Seaborn
Plotly

### Web Application

Streamlit

### Deployment

Streamlit Community Cloud
GitHub

---

# 📊 Dataset

The system uses a **Credit Card Fraud Detection Dataset** containing anonymized transaction features.

The dataset includes:

* Transaction Time
* Transaction Amount
* PCA-transformed features (V1 – V28)
* Fraud label (`Class`)

### Class Distribution

Legitimate Transactions ≈ 99.8%
Fraudulent Transactions ≈ 0.2%

This heavy imbalance makes fraud detection a **challenging anomaly detection problem**, making machine learning essential.

---

# 📈 Model Performance

The fraud detection model demonstrates strong predictive capability.

Evaluation metrics include:

Confusion Matrix
ROC Curve
AUC Score

These metrics help evaluate **how effectively the model identifies fraudulent transactions while minimizing false positives**.

---

DeepDetect integrates **SHAP explainability** to interpret predictions.

Benefits include:

* Identifying important features influencing fraud detection
* Improving model transparency
* Helping financial analysts understand model decisions

---

# 🔒 Real-World Applications

This system architecture can be extended for:

Banking fraud monitoring
FinTech payment security
Transaction anomaly detection
Credit card fraud prevention systems
Risk scoring engines

---
