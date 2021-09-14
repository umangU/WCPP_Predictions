#import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
import warnings
import csv
#read the data
warnings.filterwarnings('ignore')

filepath = 'WCPP_Input.csv'
df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')

nobs = 6
df_train, df_test = df[0:-nobs], df[-nobs:]

df_differenced = df_train.diff().dropna()
df_diff = df_differenced.diff().dropna()

model = VAR(df_diff)
model_fit = model.fit()

lag_order = model_fit.k_ar
forecast_input = df_diff.values[-lag_order:]
fc = model_fit.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df[col].iloc[-1]-df[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(df_train, df_forecast, second_diff=True)
final = df_results.loc[:, ['0102_forecast','0103_forecast','0104_forecast','0105_forecast','0201_forecast','0202_forecast',
'0203_forecast','0204_forecast','0205_forecast','0206_forecast','0299_forecast','0301_forecast','0302_forecast','0303_forecast',
'0304_forecast','0305_forecast','0306_forecast','0307_forecast','0399_forecast','0401_forecast','0402_forecast','0403_forecast',
'0404_forecast','0405_forecast','0406_forecast','0499_forecast','0501_forecast','0502_forecast','0503_forecast','0601_forecast',
'0602_forecast','0603_forecast','0604_forecast','0605_forecast','0606_forecast','0607_forecast','0608_forecast','0699_forecast',
'0701_forecast','0702_forecast','0703_forecast','0704_forecast','0705_forecast','0706_forecast','0707_forecast','0799_forecast',
'0901_forecast','0902_forecast','0903_forecast','0904_forecast','0905_forecast','0906_forecast','0907_forecast','0908_forecast',
'0909_forecast','0910_forecast','0911_forecast','0912_forecast','0913_forecast','0914_forecast','0915_forecast','0999_forecast',
'1001_forecast','1002_forecast','1003_forecast','1004_forecast','1007_forecast','1101_forecast','1102_forecast','1103_forecast',
'1104_forecast','1105_forecast','1106_forecast','1107_forecast','1108_forecast','1109_forecast','1110_forecast','1111_forecast',
'1112_forecast','1113_forecast','1114_forecast','1115_forecast','1116_forecast','1117_forecast','1199_forecast','1701_forecast',
'1702_forecast','1799_forecast']]

print(mean_absolute_percentage_error(df_test,final)*100)
# data1 = df.diff().dropna()
# data2 = data1.diff().dropna()
#
# model = VAR(endog = data2)
# model_fit = model.fit()
# yhat = model_fit.forecast(model_fit.y, steps=1)
# df_f = pd.DataFrame(yhat, columns=df.columns + '_2d')
# df_res = invert_transformation(data2,df_f, second_diff=True)
# final = df_res.loc[:, ['0102_forecast','0103_forecast','0104_forecast','0105_forecast','0201_forecast','0202_forecast',
# '0203_forecast','0204_forecast','0205_forecast','0206_forecast','0299_forecast','0301_forecast','0302_forecast','0303_forecast',
# '0304_forecast','0305_forecast','0306_forecast','0307_forecast','0399_forecast','0401_forecast','0402_forecast','0403_forecast',
# '0404_forecast','0405_forecast','0406_forecast','0499_forecast','0501_forecast','0502_forecast','0503_forecast','0601_forecast',
# '0602_forecast','0603_forecast','0604_forecast','0605_forecast','0606_forecast','0607_forecast','0608_forecast','0699_forecast',
# '0701_forecast','0702_forecast','0703_forecast','0704_forecast','0705_forecast','0706_forecast','0707_forecast','0799_forecast',
# '0901_forecast','0902_forecast','0903_forecast','0904_forecast','0905_forecast','0906_forecast','0907_forecast','0908_forecast',
# '0909_forecast','0910_forecast','0911_forecast','0912_forecast','0913_forecast','0914_forecast','0915_forecast','0999_forecast',
# '1001_forecast','1002_forecast','1003_forecast','1004_forecast','1007_forecast','1101_forecast','1102_forecast','1103_forecast',
# '1104_forecast','1105_forecast','1106_forecast','1107_forecast','1108_forecast','1109_forecast','1110_forecast','1111_forecast',
# '1112_forecast','1113_forecast','1114_forecast','1115_forecast','1116_forecast','1117_forecast','1199_forecast','1701_forecast',
# '1702_forecast','1799_forecast']]
# print(final)
# final.to_csv('C:/Documents/Codes/Python/WCPP_Benchmark/second.csv')
