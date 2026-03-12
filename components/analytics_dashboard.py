import streamlit as st
import pandas as pd
import plotly.express as px

def show_dashboard():

    df = pd.read_csv("data/creditcard.csv")

    st.subheader("Fraud Distribution")

    fig = px.pie(
        df,
        names="Class",
        title="Fraud vs Normal Transactions"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Transaction Amount Distribution")

    fig2 = px.histogram(
        df,
        x="Amount",
        nbins=50
    )

    st.plotly_chart(fig2,use_container_width=True)