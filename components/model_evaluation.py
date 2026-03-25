import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib


# -------------------------------------------------------
# LOAD DATASET SAFELY (LOCAL OR ONLINE)

@st.cache_data
def load_data():

    # try local dataset first
    if os.path.exists("creditcard.csv"):
        return pd.read_csv("creditcard.csv")

    # fallback dataset (stable public mirror)
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

    return pd.read_csv(url)


# -------------------------------------------------------
# MODEL EVALUATION

def show_model_metrics():

    st.subheader("📊 Model Evaluation")

    df = load_data()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load("model/fraud_model.pkl")

    y_pred = model.predict(X_test)

    # ---------------- CONFUSION MATRIX ----------------

    st.markdown("### Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ---------------- ROC CURVE ----------------

    st.markdown("### ROC Curve")

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()

    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], '--')

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    st.pyplot(fig2)
