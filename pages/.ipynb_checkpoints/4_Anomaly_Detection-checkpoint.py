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
outliers['Date'] = outliers['Date'].dt.strftime('%Y-%m-%d')
st.markdown("<h5 style='text-align: left; color: rgb(0, 0, 0);'> Outliers </h5>", unsafe_allow_html=True)
st.dataframe(outliers[['Date']].transpose(), use_container_width=True)

