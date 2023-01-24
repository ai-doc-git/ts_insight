import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.stats.diagnostic as diag
from scipy.stats import kendalltau
from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

from autots import AutoTS

import warnings
warnings.filterwarnings("ignore")

############################################## Title ##############################################

st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> Business Insight </h1>", unsafe_allow_html=True)
st.markdown('----')

############################################## Body ##############################################
df_ad = st.session_state['my_data3']
val_col = st.session_state['val_col']
group_col = st.session_state['group_col']
decomposed_result = st.session_state['decomposed_val']

# df_b = pd.DataFrame(df_tmp)
# df_b.index = df[date_col]
# df_b.index = pd.DatetimeIndex(df_b.index)
df_ad['year'] = df_ad.index.year
df_ad['month'] = df_ad.index.month
df_ad['quarter'] = df_ad.index.quarter

def highlight_survived(val):
    return ['background-color: green'] if val > 0 else ['background-color: red']*len(s)

def color_survived(val):
    color = 'rgb(238, 249, 239)' if val > 0 else 'rgb(253, 238, 237)'
    return f'background-color: {color}'

st.markdown("<h5 style='text-align: left; color: rgb(0, 0, 0);'> Year on Year Growth </h5>", unsafe_allow_html=True)
year_df = df_ad.groupby('year').agg({'Inflow':sum})
year_df['lag'] = year_df.Inflow.shift(1)
year_df['YoY growth'] = ((year_df.Inflow - year_df.lag) / year_df.lag) * 100
year_df.dropna(inplace=True)
year_df['YoY growth'] = [int(item) for item in year_df['YoY growth']]
year_output = year_df[['YoY growth']]
year_output = year_output[1:]
y_out = year_output.transpose().style.applymap(color_survived)
st.dataframe(y_out, use_container_width=True)

st.markdown("<h5 style='text-align: left; color: rgb(0, 0, 0);'> Quarter on Quarter Growth </h5>", unsafe_allow_html=True)
quarter_df = df_ad.groupby(['year','quarter']).agg({'Inflow':sum})
quarter_df['lag'] = quarter_df.Inflow.shift(1)
quarter_df['QoQ growth'] = ((quarter_df.Inflow - quarter_df.lag) / quarter_df.lag) * 100
quarter_df.dropna(inplace=True)
quarter_df['QoQ growth'] = [int(item) for item in quarter_df['QoQ growth']]
quarter_df2 = quarter_df[['QoQ growth']]
quarter_output = quarter_df2[1:]
q_out = quarter_output.transpose().style.applymap(color_survived)
st.dataframe(q_out, use_container_width=True)

st.markdown("<h5 style='text-align: left; color: rgb(0, 0, 0);'> Month on Month Growth </h5>", unsafe_allow_html=True)
month_df = df_ad.groupby(['year','month']).agg({'Inflow':sum})
month_df['lag'] = month_df.Inflow.shift(1)
month_df['MoM growth'] = ((month_df.Inflow - month_df.lag) / month_df.lag) * 100
month_df.dropna(inplace=True)
month_df['MoM growth'] = [int(item) for item in month_df['MoM growth']]
# month_df.fillna('na', inplace=True)
month_df = month_df[['MoM growth']]
month_output = month_df[1:]
m_out = month_output.transpose().style.applymap(color_survived)
st.dataframe(m_out, use_container_width=True)


# Overall Trend of the data
st.markdown('----')
st.markdown("<h3 style='text-align: center; color: rgb(0, 0, 0);'> Overall Trend in Data</h3>", unsafe_allow_html=True)
st.markdown('----')

st.line_chart(data=decomposed_result, x=None, height=200, y='trend', use_container_width=True)
calculate_kendall_tau(df_ad[[val_col]], val_col)