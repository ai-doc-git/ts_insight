# Import python packages
import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Page Congifuration
st.set_page_config(
    page_title='TS - INSIGHT', 
    page_icon='🕐', 
    layout='wide'
)

st.sidebar.write('NAVIGATION SIDEBAR')
st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> TS - INSIGHT </h1>", unsafe_allow_html=True)

# Load Animation
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
    
# Upload Data
uploaded_file = st.file_uploader("Upload time series data:")
if uploaded_file is not None and 'my_data' not in st.session_state:
    df = pd.read_csv(uploaded_file)
    st.session_state['my_data'] = df