import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():

    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

    df = pd.read_csv(url)

    return df


def show_dashboard():

    st.subheader("📊 Transaction Analytics Dashboard")

    df = load_data()

    col1,col2 = st.columns(2)

    # Fraud vs Legitimate count
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

        ax.set_xticklabels(["Legit","Fraud"])

        st.pyplot(fig)

    # Transaction amount distribution
    with col2:

        st.markdown("### Transaction Amount Distribution")

        fig, ax = plt.subplots()

        sns.histplot(df["Amount"],bins=50,kde=True,ax=ax)

        st.pyplot(fig)
