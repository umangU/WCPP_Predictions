import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.arima import ADFTest
from sklearn.metrics import mean_absolute_percentage_error

filepath = 'C:/Documents/Codes/R/World Centile/World_Centile_All.csv'
df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
df_sub = df[['102']]
df_train = df_sub[0:60]
df_test = df_sub[55:]

arima_model = pm.auto_arima(df_train, start_p=0, d=1,
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
                  m=1,
                  seasonal=True,
                  error_action='warn',
                  trace=True,
                  success_warnings=True,
                  stepwise=True,
                  random_state=20,
                  n_fits=50)

n_periods = 10
result = pd.DataFrame(arima_model.predict(n_periods=n_periods), index=df_test.index)
print(mean_absolute_percentage_error(df_test,result))
