# Import python packages
import pandas as pd
import streamlit as st

# Title
st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> Read Data </h1>", unsafe_allow_html=True)
st.markdown('----')

# Body
df = st.session_state['my_data']
st.dataframe(df,height=200,use_container_width=True)

col1, col2, col3 = st.columns(3)
date_col = col1.text_input("Enter Date Column name:")
val_col = col2.text_input("Enter Value Column name:")
group_col = col3.text_input("Enter Group by Column name:(optional)")

dropdown_selected_value = ''
if group_col:
    group_dropdown = st.container()
    with group_dropdown:
        options = df[group_col].unique()
        group_dropdown = st.selectbox('Select a Category:',options)
        dropdown_selected_value = group_dropdown

    df = df.loc[df[group_col] == dropdown_selected_value]

_, col, _ = st.columns((4,1,4))
clicked = col.button('Visualize Data', type='secondary')
if clicked:
    if group_col:
        df = df.loc[df[group_col] == dropdown_selected_value]
    else:
        df = df.copy()
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    df = df.set_index([date_col])
    st.line_chart(data=df, x=None, height=300, y=val_col, use_container_width=True)
    st.session_state['my_data2'] = df
    
    if 'val_col' not in st.session_state:
        st.session_state['val_col'] = val_col
        
    if 'group_col' not in st.session_state:
        st.session_state['group_col'] = group_col