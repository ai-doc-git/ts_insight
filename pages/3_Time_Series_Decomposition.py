import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose

# Title
st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> Time Series Decomposition </h1>", unsafe_allow_html=True)
st.markdown('----')

# Body
df_ad = st.session_state['my_data3']
val_col = st.session_state['val_col']
group_col = st.session_state['group_col']

st.write('Original Time Series:')
st.line_chart(data=df_ad, x=None, height=300, y=val_col, use_container_width=True)

seasonal_period = st.number_input("Enter the seasonal period:")

_, col, _ = st.columns((4,1,4))
decompose_btn = col.button('DECOMPOSE',type='primary')
if decompose_btn:
    
    result = seasonal_decompose(df_ad[val_col], extrapolate_trend='freq', period=int(seasonal_period))
    decomposed_result = pd.DataFrame({'actual':result.observed, 'trend':result.trend, 'seasonal':result.seasonal, 'residual':result.resid})

    if 'decomposed_val' not in st.session_state:
        st.session_state['decomposed_val'] = decomposed_result
        
    st.markdown("<h3 style='text-align: center; color: rgb(0, 0, 0);'> Decomposed Time Series </h3>", unsafe_allow_html=True)
    colar, colts = st.columns(2)
    colar.line_chart(data=decomposed_result, x=None, height=200, y='actual', use_container_width=True)
    colts.line_chart(data=decomposed_result, x=None, height=200, y='trend', use_container_width=True)
    colts.line_chart(data=decomposed_result, x=None, height=200, y='seasonal', use_container_width=True)
    colar.line_chart(data=decomposed_result, x=None, height=200, y='residual', use_container_width=True)
