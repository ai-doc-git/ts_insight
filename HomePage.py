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

st.set_page_config(
    page_title='TS - INSIGHT', 
    page_icon='🕐', 
    layout='wide'
)

st.sidebar.write('NAVIGATION SIDEBAR')

############################################## Title ##############################################

st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> TS - INSIGHT </h1>", unsafe_allow_html=True)

############################################## Animation ##############################################
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets5.lottiefiles.com/packages/lf20_ecqldsqz.json"
_,img_col, _ = st.columns((0.5,2,0.5))

with img_col:
    lottie_hello = load_lottieurl(lottie_url)
    st_lottie(lottie_hello, key="user")
    
    
    
############################################## Title ##############################################

uploaded_file = st.file_uploader("Upload time series data:")
if uploaded_file is not None and 'my_data' not in st.session_state:
    df = pd.read_csv(uploaded_file)
    # st.session_state['my_data'] = pd.DataFrame()
    
    st.session_state['my_data'] = df