import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from math import sqrt

filepath = 'WCPP_Input.csv'
df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')

train = df.iloc[:18]
test = df.iloc[18:]

scaler = MinMaxScaler()
scaler.fit(train)

df_scaled = scaler.transform(df)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

n_input = 12
n_features = 88

generator = TimeseriesGenerator(scaled_train, scaled_train, length = n_input, batch_size=1)

X, y = generator[0]

model = Sequential()
model.add(LSTM(50, input_shape=(n_input, n_features), return_sequences=True))
model.add(LSTM(128, return_sequences = True))
model.add(LSTM(128, return_sequences = True))
model.add(LSTM(64, return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(scaled_train.shape[1]))
model.compile(optimizer = 'adam', loss='mse')

# early_stop = EarlyStopping(monitor='val_loss',patience=1)
model.fit_generator(generator, epochs=50)

test_predictions = []

first_eval_batch = df_scaled[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(0,6):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]], axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
print(true_predictions)
