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
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from fbprophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
# from skforecast.ForecasterAutoreg import ForecasterAutoreg
# from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom 
# from sklearn.ensemble import RandomForestRegressor
# from skforecast. ForecasterAutoregMultiOutput import ForecasterAutoregMultioutput
# from sklearn.pipeline import make_pipeline
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings("ignore")


############################################## Models ##############################################

def auto_arima_modelling(data, train, test, val_col, fcst_horizon):
    train_ = train.reset_index()
    train_['unique_id'] = 0
    train_.rename(columns={train_.columns[0]:'ds', val_col:'y'}, inplace=True)
    model = StatsForecast(models=[AutoARIMA()], freq = 'D')
    model.fit(train_)
    prediction = model.predict(len(test))
    prediction = pd.DataFrame({val_col:list(prediction.AutoARIMA)})
    datelist = pd.date_range(test.index[0], periods=len(test)).tolist()
    prediction.index = datelist
    
    data_ = data.reset_index ()
    data_['unique_id'] = 0
    data_.rename(columns={data_.columns[0]: 'ds', val_col: 'y'}, inplace=True)
    model = StatsForecast(models=[AutoARIMA()], freq = "D") 
    model.fit(data_)
    forecast = model.predict(fcst_horizon)
    forecast = pd.DataFrame({val_col:list(forecast.AutoARIMA)})
    datelist = pd.date_range(test.index[-1]+timedelta(days=1), periods=fcst_horizon).tolist()
    forecast.index = datelist
    return prediction, forecast
                          

def auto_reg_modelling(data, train, test, val_col, fcst_horizon):                
    params_grid = {'lags': [3, 7, 14, 21, 28, 31, [1,7,14,28,30,31]],
                   'trend': ['n', 'c', 'ct', 't'],
                   'seasonal' : [True, False]}
    grid = hp_combination(params_grid)
    model_obj = AutoReg
    best_param, accuracy = hptuning(grid, model_obj, train, test)
    model = AutoReg(train, **best_param).fit()
    prediction = model.predict(len(train), len(train)+len(test)-1)
    prediction = pd.DataFrame({val_col:prediction})

    fmodel = AutoReg(data, **best_param).fit()
    forecast = fmodel.predict(len(data), len(data)+fcst_horizon-1)
    forecast = pd.DataFrame({val_col:forecast})
    return prediction,forecast
                          
                          
                          
def HWES(data, train, test, val_col, fcst_horizon):
    params_grid = {'trend': ["add","mul","additive", "multiplicative", None],
                   'seasonal': ["add", "mul", "additive", "multiplicative", None],
                   'seasonal_periods': [7, 30, 365]}
    grid = hp_combination(params_grid)
    model_obj = ExponentialSmoothing
    best_param, accuracy = hptuning(grid, model_obj, train, test)
    model = ExponentialSmoothing(train, **best_param).fit()
    prediction = model.predict(len(train), len(train)+len(test)-1)
    prediction = pd.DataFrame({val_col : prediction})
    
    fmodel = ExponentialSmoothing(data, **best_param).fit()
    forecast = fmodel.predict(len(data), len(data)+fcst_horizon-1)
    forecast = pd.DataFrame({val_col: forecast})
    return prediction,forecast
                          
                          
def hp_combination(params_grid):
    grid = ParameterGrid(params_grid)
    param_grid = [p for p in grid]
    return param_grid
                          
                          
                          
def hptuning(grid, model_obj, train, test):
    best_model_dict = {}
    count = 0
    for p in grid:
        try:
            model = model_obj(train, **p).fit()
            forecast = model.predict(len(train), len(train)+len(test)-1)
            forecast = pd.DataFrame({val_col:forecast})
            mae, accuracy, mse, rmse = evaluate_model(test, forecast)
            best_model_dict[count] = mae
            count = count + 1
        except:
            pass
    best_model = sorted(best_model_dict, key=lambda x: best_model_dict[x], reverse=False)
    return grid[best_model[0]], best_model_dict[best_model[0]]
                          

def evaluate_model(test, forecast):
    true_list = [np.sum(item) for item in list(np.array_split(test, len(test)/30))] 
    pred_list = [np. sum(item) for item in list(np.array_split(forecast, len(test)/30))]
    mape = mean_absolute_percentage_error (true_list, pred_list)
    test_mape = mape * 100
    accuracy = 100 - test_mape
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    return mae, accuracy, mse, rmse


def train_test_split(data):
    data_size = len(data)
    train = data[:-int((0.25 * data_size))]
    test = data[-int((0.25 * data_size)):]
    return data, train, test

############################################## Title ##############################################

st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> Forecast </h1>", unsafe_allow_html=True)
st.markdown('----')

############################################## Body ##############################################
df_fcst = st.session_state['my_data4']
val_col = st.session_state['val_col']
group_col = st.session_state['group_col']
decomposed_result = st.session_state['decomposed_val']
                          
                          
                          
model_name = st.selectbox("Select a Model to forecast:",['Select Model','Auto-Arima', 'Holt-Winter-Exponential Smoothing', 'Auto-Reg'])
# st.dataframe(df_fcst)

if model_name == 'Auto-Arima':
    st.write('Auto-Arima Model Selected.')
    data_tmp = df_fcst[[val_col]]
    data, train, test = train_test_split(data_tmp)
    prediction,forecast = auto_arima_modelling(data, train, test, val_col, 180)
    
    full_data = data.append(forecast)
    full_data['train'] = train[val_col]
    full_data['test'] = test[val_col]
    full_data['pred'] = prediction[val_col]
    full_data['fcst'] = forecast[val_col]
    f_data = full_data.drop([val_col], axis=1)
    st.line_chart(f_data)
    
    st.write("Evaluation Result:")
    mae, accuracy, mse, rmse = evaluate_model(test, prediction)
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    
    mcol1.metric("Accuracy", round(accuracy,2))
    mcol2.metric("MAE", round(mae,2))
    mcol3.metric("MSE", round(mse,2))
    mcol4.metric("RMSE", round(rmse,2))
elif model_name == 'Holt-Winter-Exponential Smoothing':
    st.write('Holt-Winter-Exponential Smoothing Model Selected.')
    data_tmp = df_fcst[[val_col]]
    data, train, test = train_test_split(data_tmp)
    prediction,forecast = HWES(data, train, test, val_col, 180)
    
    full_data = data.append(forecast)
    full_data['train'] = train[val_col]
    full_data['test'] = test[val_col]
    full_data['pred'] = prediction[val_col]
    full_data['fcst'] = forecast[val_col]
    f_data = full_data.drop([val_col], axis=1)
    st.line_chart(f_data)
    
    st.write("Evaluation Result:")
    mae, accuracy, mse, rmse = evaluate_model(test, prediction)
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    
    mcol1.metric("Accuracy", round(accuracy,2))
    mcol2.metric("MAE", round(mae,2))
    mcol3.metric("MSE", round(mse,2))
    mcol4.metric("RMSE", round(rmse,2))
    
    
elif model_name == 'Auto-Reg':
    st.write('Auto-Reg Model Selected.')
    data_tmp = df_fcst[[val_col]]
    data, train, test = train_test_split(data_tmp)
    prediction,forecast = auto_reg_modelling(data, train, test, val_col, 180)
    
    full_data = data.append(forecast)
    full_data['train'] = train[val_col]
    full_data['test'] = test[val_col]
    full_data['pred'] = prediction[val_col]
    full_data['fcst'] = forecast[val_col]
    f_data = full_data.drop([val_col], axis=1)
    st.line_chart(f_data)
    
    st.write("Evaluation Result:")
    mae, accuracy, mse, rmse = evaluate_model(test, prediction)
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    
    mcol1.metric("Accuracy", round(accuracy,2))
    mcol2.metric("MAE", round(mae,2))
    mcol3.metric("MSE", round(mse,2))
    mcol4.metric("RMSE", round(rmse,2))              
                          
                          
                          