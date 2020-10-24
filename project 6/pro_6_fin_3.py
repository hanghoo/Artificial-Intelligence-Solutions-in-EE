## Written by Ricardo Valdez
# City College of New York, CUNY
# Date: October 18, 2020
# --------------------------------------------------------------------------
# | Project -  Apple Stock Prediction                                      |
# --------------------------------------------------------------------------
#
# With an ANN, predict the next day closing price movement for Apple stocks.
# Use the following 6 technical indicators as inputs for your ANN:
#     - Simple moving average (SMA)
#     - Exponential moving average (EMA)
#     - Momentum
#     - Bollinger bands (BB)
#     - Stochastic oscillator (SO)
#     - On balance volume (OBV)
#

import numpy as np
import pandas as pd
import time
from tensorflow import keras
import OBV
import matplotlib.pyplot as plt

def print_weights(weights):
    print('\n******* WEIGHTS OF ANN *******\n')
    for i in range(int(len(weights)/2)):
        print('Weights W%d:\n' %(i), weights[i*2])
        print('Bias b%d:\n' %(i), weights[(i*2)+1])


#%% DATA EXTRACTION
historical_data_file = 'AAPL.csv'

## load the historical stock data
data = pd.read_csv(historical_data_file)

## keep only the closing prices
df = data[['Close']]

## calculate the Simple Moving Average and add it to a new column in df;
## rolling means moving window starting from the current entry and
## the last 5 days (open df in Variable explorer ad see the values)
short_window = 5
long_window = 25

df['SMA'] = df.rolling(short_window).mean()
## calculate the Exponential Moving Average and add it
## to a new column in df;
## ewm means exponential moving average
df['EMA'] = df['Close'].ewm(span=long_window).mean()
## calculate the Momentum and add it to a new column in df;
## Momentum is defined as close [today] / close [5 days ago]
df['Momentum'] = df['Close'] / df['Close'].shift(short_window)
## calculate the standard deviation of SMA over a rolling window
## ddof means degree of freedom, set it to zero to get the population std
df['STD'] = df['Close'].rolling(short_window).std(ddof=0)
## calculate the Bollinger Band for each day
## BB > 1 means the closing price is above the upper band
## BB < -1 means the closing price is below the lower band
df['BB'] = (df['Close'] - df['SMA']) / (2 * df['STD'])
## calculate Stochastic Oscillator (SO) for each day
df['14-low'] = data['Low'].rolling(14).min()
df['14-high'] = data['High'].rolling(14).max()
df['SO'] = (df['Close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])

## calculate On Balance Volume (OBV) for each day
df['OBV'] = OBV.on_balance_volume(data, trend_periods=21, close_col='Close', vol_col='Volume')['obv']

## calculate the percent change of closing price for each day
## shift(1) means yesterday (applies to each element of df)
increase = df['Close'] - df['Close'].shift(1)
df['Percent_Change'] = (increase / df['Close'].shift(1)) * 100

## shift percent change to the previous day so that it now represents the
## percent change for the next day (shift df elements up by 1 using -1)
df['Percent_Change'] = df['Percent_Change'].shift(-1)

#%% ANN TRAINING

## start training ANN using the df file contains historical data and
## information that was just populated above
print('\n\n********* NOW START TRAINING ANN USING', historical_data_file,'*********')
time.sleep(3)

## remove rows with invalid inputs (i.e., nan) and
## create input and output arrays for ANN
## starting from day 25 to te end, but excluding the end (due to -1)
## we will predict the precent change of the last day recorded in csv (today)
X = np.array(df[long_window:-1][['SMA', 'EMA', 'Momentum', 'BB', 'SO', 'OBV']])
Y = np.array(df[long_window:-1]['Percent_Change'])

## create a model for the ANN
model = keras.Sequential()
# Create ANN with 6 inputs + 2 hidden layers + 1 output layer
## first hidden layer that accepts 6 input features
## the hidden layer will have 4 neurons;
## dense means every neuron in the layer connects to every neuron in the
## previous layer;
model.add(keras.layers.Dense(4, activation='relu', input_shape=(6,)))
## add another hidden layer with 3 neurons to the ANN
model.add(keras.layers.Dense(3, activation='relu'))
## add an output layer with a single output (percent change)
model.add(keras.layers.Dense(1, activation='linear'))

## set the optimization algorithm used for minimizing loss function
## use gradient descent (adam) to minimize error (loss)
model.compile(optimizer='adam', loss='mean_squared_error')
## train the ANN model using 200 iterations
model.fit(X, Y, epochs=200)

## record the frequency of the loss
figure, ax = plt.subplots(2, 2, figsize=(10, 5))
for (i, lnrate) in enumerate([0.1, 0.01, 0.001, 0.0001]):
    loss = []
    for j in range(10):
        opt = keras.optimizers.Adam(learning_rate=lnrate)
        model.compile(optimizer=opt, loss='mean_squared_error')
        model.fit(X, Y, epochs=200)
        loss.append(model.evaluate(X, Y))
    print(loss)
    ax[i // 2, i % 2].hist(loss, bins=5, edgecolor='black')
    # display y label
    ax[i // 2, i % 2].set_xlabel("Loss")
    # display x label
    ax[i // 2, i % 2].set_ylabel("Frequency")
    # display title
    ax[i // 2, i % 2].set_title("Learning Rate: " + str(lnrate))

plt.tight_layout()
plt.savefig("Fin_histogram.png")
plt.show()

## training with more iterations will yield better results
## build different ANN configurations for better results
## use different activation functions for better results
## use different optimizers adam or SGD (stochastic gradient descent)
## model.fit(X, Y, epochs=2000)
weights = model.get_weights()
print_weights(weights)
print('\n\n********** ANN training complete **********\n\n')

#%% ANN PREDICTION

## insert the inputs for the latest trading day into an array
latest_SMA = df.iloc[-1]['SMA']
latest_EMA = df.iloc[-1]['EMA']
latest_Momentum = df.iloc[-1]['Momentum']
latest_BB = df.iloc[-1]['BB']
latest_SO = df.iloc[-1]['SO']
latest_OBV = df.iloc[-1]['OBV']
latest_inputs = np.array([[latest_SMA, latest_EMA, latest_Momentum, latest_BB, latest_SO, latest_OBV]])
prediction = model.predict(latest_inputs)[0,0]

print('\n***************************************')
print('ANN Predicted Next Day Stock Movement: %+.2f%%' % (prediction))
print('***************************************')


