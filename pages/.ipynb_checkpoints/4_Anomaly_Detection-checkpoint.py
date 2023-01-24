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

def fix_mean(data, val_col, outliers):
    data1 = data.copy()
    data_mu = data1[val_col].mean()
    for item in outliers:
        data1.at[item[0], val_col] = data_mu
    return data1

def fix_median(data, val_col, outliers):
    data1 = data.copy()
    data_mu = data1[val_col].median()
    for item in outliers:
        data1.at[item[0], val_col] = data_mu
    return data1

############################################## Title ##############################################

st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> Anomaly Detection and Handling </h1>", unsafe_allow_html=True)
st.markdown('----')

############################################## Body ##############################################
df_ad = st.session_state['my_data3']
val_col = st.session_state['val_col']
group_col = st.session_state['group_col']
decomposed_result = st.session_state['decomposed_val']


df_mu = df_ad[val_col].mean()
df_std = df_ad[val_col].std()

threshold = st.select_slider("Select a threshold:",['3', '3.5', '4', '4.5', '5'])
# st.markdown('----')

st.line_chart(data=df_ad, x=None, height=200, y=val_col, use_container_width=True)

lower = df_mu - (float(threshold) * df_std)
upper = df_mu + (float(threshold) * df_std)

outlier_df = pd.DataFrame({'residual':decomposed_result.residual})
outlier_df['lower'] = [lower for item in range(len(outlier_df))]
outlier_df['upper'] = [upper for item in range(len(outlier_df))]

st.line_chart(outlier_df)

outliers = outlier_df.loc[(outlier_df['residual'] < outlier_df['lower']) | (outlier_df['residual'] > outlier_df['upper'])]
outliers['Date'] = outliers.index
outliers.index = [item for item in range(len(outliers))]
outlier_tmp = outliers.copy()
outlier_lst = list(outlier_tmp[['Date']].values)
outliers['Date'] = outliers['Date'].dt.strftime('%Y-%m-%d')
st.markdown("<h5 style='text-align: left; color: rgb(0, 0, 0);'> Outliers </h5>", unsafe_allow_html=True)
st.dataframe(outliers[['Date']].transpose(), use_container_width=True)

st.write('Number of Outliers:' + str(len(outliers)))


st.markdown("<h3 style='text-align: center; color: rgb(0, 0, 0);'> Fix Outliers </h3>", unsafe_allow_html=True)
st.markdown('----')

st.write("Select the Outlier handling method:")
ccol1, ccol2, ccol3 = st.columns(3)

method1 = ccol1.checkbox('Mean')
method2 = ccol2.checkbox('Median')
method3 = ccol3.checkbox('No Outliers')

df_mod = pd.DataFrame()
if method1:
    df_mod = fix_mean(df_ad, val_col, outlier_lst)
    st.success('Fixed Outliers using mean of the data.')
    df_new = df_mod.copy()
    df_new['org'] = df_ad[val_col]
    st.line_chart(data=df_new, x=None, height=300, use_container_width=True)
if method2:
    df_mod = fix_median(df_ad, val_col, outlier_lst)
    st.success('Fixed Outliers using median of the data.')
    df_new = df_mod.copy()
    df_new['org'] = df_ad[val_col]
    st.line_chart(data=df_new, x=None, height=300, use_container_width=True)
if method3:
    df_new = df_ad.copy()
    st.success('No Outliers found.')
    st.line_chart(data=df_new, x=None, height=300, use_container_width=True)
    
save_data = st.button('Save Data')
# if save_data and 'my_data3' not in st.session_state:
df_ad_next = df_mod.copy()
st.session_state['my_data4'] = df_ad_next

