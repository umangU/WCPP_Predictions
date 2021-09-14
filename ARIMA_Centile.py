import pandas as pd
import pmdarima as pm
from pmdarima.arima import ADFTest

filepath = 'C:/Documents/Codes/R/World Centile/World_Centile_All.csv'
df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
columns = df.columns
predictions = []
for i in columns:
    df1 = df[i]
    arima_model = pm.auto_arima(df1.values, start_p=0,
                      test='adf',
                      d=None,
                      start_q=0,
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

    n_periods = 5
    result = arima_model.predict(n_periods=n_periods)
    df_result = pd.DataFrame(result).T
    df_result.to_csv('C:/Documents/Codes/R/World Centile/World_ERA2023_Centile_Predictions.csv', mode='a',header = False)