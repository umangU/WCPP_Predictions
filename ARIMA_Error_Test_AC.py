import pandas as pd
import numpy as np
import pmdarima as pm
import csv
from pmdarima.arima import ADFTest
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

filepath = 'C:/Documents/Codes/Python/WCPP_Benchmark/World Article Count/WCPP_Article_Count_Test.csv'
df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
df_train = df[0:3]
df_test = df[3:]

arima_model = pm.auto_arima(df_train, start_p=0, d=1,
                  start_q=0,
                  max_p=5,
                  max_d=5,
                  max_q=5,
                  start_P=0,
                  D=0,
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

n_periods = 3
result = pd.DataFrame(arima_model.predict(n_periods=n_periods), index=df_test.index)
print(mean_absolute_percentage_error(df_test,result))

print(result)