# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:21:45 2022

@author: Shriprada
"""

import streamlit as st
from datetime import date
import pandas as pd



from statsmodels.tsa.arima.model import ARIMA
#import statsmodels.api as smapi

from plotly import graph_objs as go
#import matplotlib.pyplot as plt




wipro_data = pd.read_csv("F:/Project/Files/wipro_daily_data.csv")
infosys_data = pd.read_csv("F:/Project/Files/infosys_daily_data.csv")
tatamotors_data = pd.read_csv("F:/Project/Files/tatamotors_daily_data.csv")
reliance_data = pd.read_csv("F:/Project/Files/reliance_daily_data.csv")

data_dict = {'Infosys': infosys_data,
              'Reliance': reliance_data,
              'Tatamotors': tatamotors_data,
              'Wipro': wipro_data
              }

wipro_model_arima = ARIMA(wipro_data['Close'], order = (1,2,1)).fit()
infosys_model_arima = ARIMA(infosys_data['Close'], order = (1,1,1)).fit()
TMS_model_arima = ARIMA(tatamotors_data['Close'], order = (1,1,1)).fit()
Reliance_model_arima = ARIMA(reliance_data['Close'], order = (1,2,1)).fit()


model_dict = {'Infosys': infosys_model_arima,
              'Reliance': Reliance_model_arima,
              'Tatamotors': TMS_model_arima,
              'Wipro': wipro_model_arima}

st.title('Stock Forecast App')

st.subheader('Select Stock for Forecasting')
selected_stock = st.selectbox('Stock', data_dict.keys())

n_months = st.slider("Months of Prediction:", 1, 3)
period = n_months * 30


@st.cache(allow_output_mutation=True)
def load_data(x):
    return x

data_load_state = st.text("Load data....")
data = load_data(data_dict[selected_stock])
data_load_state.text("Loading data ...done!")

st.subheader("Raw data")
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'Stock_Close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Forecasting ARIMA
     
df_train = data[["Date","Close"]]
df_train.reset_index(inplace = True)
df_train

#start = len(df_train)
#end = len(df_train)  + (n_months*30)

def arima(ticker):
    df_train = pd.DataFrame(ticker.forecast(steps = 250))
    df_train.reset_index(inplace =True)
    #df_train.rename({'index':'Date','predicted_mean':'Close_forecast'}, axis = 1, inplace = True)
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x = data['index'], y = data['Close'], name = 'Stock_Close'))
    fig1.add_trace(go.Scatter(x = df_train['index'], y = df_train['predicted_mean'], name = 'Forecast_data'))
    fig1.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig1)
    
    st.subheader("Forecast data")
    st.write(df_train.head(n_months*30))


arima(model_dict[selected_stock])


