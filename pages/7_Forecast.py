# Import python packages
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# Models
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
                   'seasonal_periods': [7, 12, 30]}
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
                          

# Hyper-parameter Tuning
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
                          
# Model Evaluation
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

# Train Test Split
def train_test_split(data):
    data_size = len(data)
    train = data[:-int((0.25 * data_size))]
    test = data[-int((0.25 * data_size)):]
    return data, train, test

# Title
st.markdown("<h1 style='text-align: center; color: rgb(0, 0, 0);'> Forecast </h1>", unsafe_allow_html=True)
st.markdown('----')

# Body
df_fcst = st.session_state['my_data4']
val_col = st.session_state['val_col']
group_col = st.session_state['group_col']
decomposed_result = st.session_state['decomposed_val']

forecast_horizon = st.text_input("Enter forecast horizon:")
model_name = st.selectbox("Select a Model to forecast:",['Select Model:', 'Holt-Winter-Exponential Smoothing', 'Auto-Reg'])

if model_name == 'Holt-Winter-Exponential Smoothing':
    st.write('Holt-Winter-Exponential Smoothing Model Selected.')
    data_tmp = df_fcst[[val_col]]
    data, train, test = train_test_split(data_tmp)
    prediction,forecast = HWES(data, train, test, val_col, int(forecast_horizon))
    
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
    
    st.markdown("----")
    csv_data = f_data.to_csv().encode('utf-8')
    st.download_button('Download Report', csv_data, file_name='export.csv', mime='text/csv')
    
elif model_name == 'Auto-Reg':
    st.write('Auto-Reg Model Selected.')
    data_tmp = df_fcst[[val_col]]
    data, train, test = train_test_split(data_tmp)
    prediction,forecast = auto_reg_modelling(data, train, test, val_col, int(forecast_horizon))
    
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
                          
    st.markdown("----")
    csv_data = f_data.to_csv().encode('utf-8')
    st.download_button('Download Report', csv_data, file_name='export.csv', mime='text/csv')                      