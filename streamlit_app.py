import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# importing customer churn data
data = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv", index_col=0)
st.write("Churn Data")
st.write(data)

st.write("How many customers in the dataset churned?")
target_bins = data.loc[:, 'Churn'].value_counts()
st.bar_chart(target_bins)