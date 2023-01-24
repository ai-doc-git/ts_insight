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

st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> Forecastability </h1>", unsafe_allow_html=True)
st.markdown('----')

############################################## Body ##############################################
df_ad = st.session_state['my_data3']
val_col = st.session_state['val_col']
group_col = st.session_state['group_col']
decomposed_result = st.session_state['decomposed_val']


#Stationarity Check
st.markdown('----')
st.markdown("<h3 style='text-align: center; color: rgb(0, 0, 0);'> Stationarity Check </h3>", unsafe_allow_html=True)

adfuller_result = adfuller(df_ad[val_col])
st.info('Augmented Dickey_fuller test p-value: %f' % adfuller_result[1])

if adfuller_result[1] > 0.05:
    st.error('Given time series data is non-stationary as p-value is more than 0.05, hence, data is going to be differenced.')
    adf_df = df_ad.copy()
    adf_df['diff1'] = adf_df[val_col].diff(periods=1)
    adf_df.dropna(inplace=True)
    adfuller_result = adfuller(adf_df['diff1'].values)
    st.info('Augmented Dickey_fuller test p-value after differencing: %f' % adfuller_result[1])
else:
    st.success('Given time series data is stationary as p-value is less than 0.05')
    
    
# ACF and PACF plot
rcParams["figure.figsize"] = 6, 3
st.markdown('----')
st.markdown("<h3 style='text-align: center; color: rgb(0, 0, 0);'> ACF and PACF plot </h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
fig = plot_acf(df_ad, lags=30)
col1.pyplot(fig)

fig = plot_pacf(df_ad, lags=30)
col2.pyplot(fig)


# White Noise Test
st.markdown('----')
st.markdown("<h3 style='text-align: center; color: rgb(0, 0, 0);'> White Noise and Random Walk Test </h3>", unsafe_allow_html=True)
st.markdown('----')

colw, colr = st.columns(2)
wn_result = diag.acorr_ljungbox(df_ad, lags=[30], boxpierce=True, model_df=0, period=None, return_df=None)
if wn_result[1] > 0.05 and wn_result[3] > 0.05:
    colw.error('Given time-series is a White Noise.')
else:
    colw.success('Given time-series is not a White Noise.')

# Random Walk Test
# st.markdown('----')
# st.markdown("<h3 style='text-align: center; color: rgb(0, 0, 0);'> Random Walk Test </h3>", unsafe_allow_html=True)
# st.markdown('----')
df_tmp_diff = df_ad.diff()
df_tmp_diff = df_tmp_diff.dropna()
rw_result = diag.acorr_ljungbox(df_tmp_diff, lags=[30], boxpierce=True, model_df=0, period=None, return_df=None)
if wn_result[1] > 0.05 and wn_result[3] > 0.05:
    colr.error('Given time-series is a Random Walk.')
else:
    colr.success('Given time-series is not a Random Walk.')