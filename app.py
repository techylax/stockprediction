import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import tensorflow as tf
from keras.models import load_model
import streamlit as st

st.title("Stock Trend Analysis")
user_input = st.text_input("Enter Stock Ticker", "AAPL")

start = '2015-10-20'
end = '2018-10-20'
df = yf.download(user_input, start, end)
df.head()

# Data Description

st.subheader("Data from 2015-2018")
st.write(df.describe())

# Visualization

st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)
