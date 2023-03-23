# Import python packages
import numpy as np
import pandas as pd
import streamlit as st

# Imputation Methods
def interpolation(df):
    df = df.interpolate()
    return df

def forward_fill(df):
    df = df.ffill()
    return df

def backward_fill(df):
    df = df.bfill()
    return df

# Title
st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> Data Quality Check </h1>", unsafe_allow_html=True)
st.markdown('----')

# Body
df = st.session_state['my_data2']
val_col = st.session_state['val_col']
group_col = st.session_state['group_col']


if group_col:
    df = df.drop([group_col], axis=1)

df = df[[val_col]]

if (df.isnull().sum().values[0]/ len(df) * 100) > 0:
    st.error('Missing values in data: ' + str(round(df.isnull().sum().values[0]/ len(df),2) * 100) + "%")
else:
    st.success('Missing values in data: ' + str(round(df.isnull().sum().values[0]/ len(df),2) * 100) + "%")
    
st.write('Missing Value Imputation:')
ccol1, ccol2, ccol3, ccol4 = st.columns(4)

method1 = ccol1.checkbox('Interpolation')
method2 = ccol2.checkbox('Forward Fill')
method3 = ccol3.checkbox('Backward Fill')
method4 = ccol4.checkbox('No Imputation')

df_mod = pd.DataFrame()

# Perform Imputation as per the selected method
if method1:
    df_mod = interpolation(df)
    st.success('Filled Missing Value in data using Interpolation, Missing values : ' + str(round(df_mod.isnull().sum().values[0]/ len(df_mod),2) * 100) + "%")
    df_new = df_mod.copy()
    df_new['org'] = df[val_col]
    st.line_chart(data=df_new, x=None, height=300, use_container_width=True)
if method2:
    df_mod = forward_fill(df)
    st.success('Filled Missing Value in data with Forward fill, Missing values : ' + str(round(df_mod.isnull().sum().values[0]/ len(df_mod),2) * 100) + "%")
    df_new = df_mod.copy()
    df_new['org'] = df[val_col]
    st.line_chart(data=df_new, x=None, height=300, use_container_width=True)
if method3:
    df_mod = backward_fill(df)
    st.success('Filled Missing Value in data with Backward fill, Missing values : ' + str(round(df_mod.isnull().sum().values[0]/ len(df_mod),2) * 100) + "%")
    df_new = df_mod.copy()
    df_new['org'] = df[val_col]
    st.line_chart(data=df_new, x=None, height=300, use_container_width=True)
if method4:
    df_mod = df.copy()

st.write('Original Dataframe:')
st.dataframe(df.transpose(), use_container_width=True)

st.write('Imputed Dataframe:')
st.dataframe(df_mod.transpose(), use_container_width=True)

df_next = df_mod.copy()
save_data = st.button('Save Data')
st.session_state['my_data3'] = df_next