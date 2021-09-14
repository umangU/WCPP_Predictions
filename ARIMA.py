import pandas as pd
import numpy as np
import pmdarima as pm
import csv
from pmdarima.arima import ADFTest
from statsmodels.tsa.arima_model import ARIMA

filepath = 'C:/Documents/Codes/Python/WCPP_Benchmark/WCPP_ARIMA_Test.csv'
df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
columns = df.columns
predictions = []
for i in columns:
    df1 = df[i]
    arima_model = pm.auto_arima(df1.values, start_p=0, d=1,
                      start_q=0,
                      max_p=5,
                      max_d=5,
                      max_q=5,
                      start_P=0,
                      D=1,
                      start_Q=0,
                      max_P=5,
                      max_D=5,
                      max_Q=5,
                      m=12,
                      seasonal=True,
                      error_action='warn',
                      trace=True,
                      success_warnings=True,
                      stepwise=True,
                      random_state=20,
                      n_fits=50)

    n_periods = 6
    result = arima_model.predict(n_periods=n_periods)
    df_result = pd.DataFrame(result).T
    df_result.to_csv('C:/Documents/Codes/Python/WCPP_Benchmark/WCPP_ERA2023_Predictions.csv', mode='a',header = False)
