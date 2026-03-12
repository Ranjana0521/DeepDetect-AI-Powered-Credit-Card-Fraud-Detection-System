import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# -------------------------------------------------------
# LOAD DATASET (WORKS LOCALLY + STREAMLIT CLOUD)

@st.cache_data
def load_data():

    if os.path.exists("creditcard.csv"):
        return pd.read_csv("creditcard.csv")

    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

    return pd.read_csv(url)


# -------------------------------------------------------
# ANALYTICS DASHBOARD

def show_dashboard():

    st.subheader("📊 Transaction Analytics Dashboard")

    df = load_data()

    col1, col2 = st.columns(2)

    # Fraud vs Legitimate transactions
    with col1:

        st.markdown("### Fraud vs Legit Transactions")

        fraud_counts = df["Class"].value_counts()

        fig, ax = plt.subplots()

        sns.barplot(
            x=fraud_counts.index,
            y=fraud_counts.values,
            palette="viridis",
            ax=ax
        )

        ax.set_xticklabels(["Legit", "Fraud"])

        st.pyplot(fig)

    # Transaction amount distribution
    with col2:

        st.markdown("### Transaction Amount Distribution")

        fig, ax = plt.subplots()

        sns.histplot(df["Amount"], bins=50, kde=True, ax=ax)

        st.pyplot(fig)
