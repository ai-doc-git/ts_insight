# Import python packages
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import kendalltau

# Method to find trend in data using Kendall's Tau
def calculate_kendall_tau(data,col_name):
    """
    Function to check the trend of any given stock in last year
    """
    index_list = [item for item in range(1,len(data)+1)]
    data['Index'] = index_list

    corr, _ = kendalltau(data['Index'], data[col_name])

    if corr > 0.6:
        st.success("Given time-series data has Strong Positive Trend")
    elif corr > 0 and corr <= 0.6:
        st.success("Given time-series data has Weak Positive Trend")
    elif corr < 0 and corr >= -0.6:
        st.error("Given time-series data has Weak Negative Trend")
    else:
        rst.error("Given time-series data has Strong Negative Trend")


# Title
st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> Business Insight </h1>", unsafe_allow_html=True)
st.markdown('----')

# Body
df_ad = st.session_state['my_data4']
val_col = st.session_state['val_col']
group_col = st.session_state['group_col']
decomposed_result = st.session_state['decomposed_val']

df_ad['year'] = df_ad.index.year
df_ad['month'] = df_ad.index.month
df_ad['quarter'] = df_ad.index.quarter

def highlight_survived(val):
    return ['background-color: green'] if val > 0 else ['background-color: red']*len(s)

def color_survived(val):
    color = 'rgb(238, 249, 239)' if val > 0 else 'rgb(253, 238, 237)'
    return f'background-color: {color}'

# YoY outcome
st.markdown("<h5 style='text-align: left; color: rgb(0, 0, 0);'> Year on Year Growth </h5>", unsafe_allow_html=True)
year_df = df_ad.groupby('year').agg({val_col:sum})
year_df['lag'] = year_df[val_col].shift(1)
year_df['YoY growth'] = ((year_df[val_col] - year_df.lag) / year_df.lag) * 100
year_df.dropna(inplace=True)
year_df['YoY growth'] = [int(item) for item in year_df['YoY growth']]
year_output = year_df[['YoY growth']]
year_output = year_output
y_out = year_output.transpose().style.applymap(color_survived)
st.dataframe(y_out, use_container_width=True)
st.bar_chart(year_output)

# QoQ outcome
st.markdown("<h5 style='text-align: left; color: rgb(0, 0, 0);'> Quarter on Quarter Growth </h5>", unsafe_allow_html=True)
quarter_df = df_ad.groupby(['year','quarter']).agg({val_col:sum})
quarter_df['lag'] = quarter_df[val_col].shift(1)
quarter_df['QoQ growth'] = ((quarter_df[val_col] - quarter_df.lag) / quarter_df.lag) * 100
quarter_df.dropna(inplace=True)
quarter_df['QoQ growth'] = [int(item) for item in quarter_df['QoQ growth']]
quarter_df2 = quarter_df[['QoQ growth']]
quarter_output = quarter_df2[0:]
q_out = quarter_output.transpose().style.applymap(color_survived)
st.dataframe(q_out, use_container_width=True)

quarter_df3 = quarter_df2.reset_index()
index_list = []
for item1, item2, item3 in zip(quarter_df3['year'], quarter_df3['quarter'], quarter_df3['QoQ growth']):
    index_list.append(str(item1) + '_' + str(item2))
quarter_df3 = quarter_df3.drop(['year', 'quarter'], axis=1)
quarter_df3.index = index_list
st.bar_chart(quarter_df3)

# MoM outcome
st.markdown("<h5 style='text-align: left; color: rgb(0, 0, 0);'> Month on Month Growth </h5>", unsafe_allow_html=True)
month_df = df_ad.groupby(['year','month']).agg({val_col:sum})
month_df['lag'] = month_df[val_col].shift(1)
month_df['MoM growth'] = ((month_df[val_col] - month_df.lag) / month_df.lag) * 100
month_df.dropna(inplace=True)
month_df['MoM growth'] = [int(item) for item in month_df['MoM growth']]
month_df = month_df[['MoM growth']]
month_output = month_df[0:]
m_out = month_output.transpose().style.applymap(color_survived)
st.dataframe(m_out, use_container_width=True)

month_df2 = month_df.reset_index()
index_list = []
for item1, item2, item3 in zip(month_df2['year'], month_df2['month'], month_df2['MoM growth']):
    index_list.append(str(item1) + '_' + str(item2))
month_df2 = month_df2.drop(['year', 'month'], axis=1)
month_df2.index = index_list
st.bar_chart(month_df2)


# Overall Trend of the data
st.markdown('----')
st.markdown("<h3 style='text-align: center; color: rgb(0, 0, 0);'> Overall Trend in Data</h3>", unsafe_allow_html=True)
st.markdown('----')

st.line_chart(data=decomposed_result, x=None, height=200, y='trend', use_container_width=True)
calculate_kendall_tau(df_ad[[val_col]], val_col)