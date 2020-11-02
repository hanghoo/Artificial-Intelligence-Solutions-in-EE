# -------------------------------- EE 6530 ---------------------------------
# | Project #7 -  Natural language processing using ANN                    |
# --------------------------------------------------------------------------
#
# Instructor            : Prof. Uyar
#
# Student 1 Name        : Hang Hu
# Student 1 CCNY email  : hhu002
# Student 1 Log In Name : ee6530_03
# Student 2 Name        :
# Student 2 CCNY email  :
# Student 2 Log In Name :
# Student 3 Name        :
# Student 2 CCNY email  :
# Student 3 Log In Name :
# --------------------------------------------------------------------------
# | I UNDERSTAND THAT COPYING PROGRAMS FROM OTHERS WILL BE DEALT           |
# | WITH DISCIPLINARY RULES OF CCNY.                                       |
# --------------------------------------------------------------------------

# Predict the movement of Apple stocks based on key words from Apple twitter
# with the help of an ANN.
# Define your own key words.
# The days where no data was present are given in arrays called missing_days
# and invalid days.

import datetime
from datetime import date
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import matplotlib.pyplot as plt


# %% Function to remove punctuation and web links from tweets
# @\S+|https?://\S+ - matches either a substring which starts with @
# and contains non-whitespace characters \S+ OR a link(url) which
# starts with http(s)://
from typing import Any


def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)',
                           ' ', tweet).split())


def str_to_date(string):
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()


def print_weights(weights):
    # weights = model.get_weights();
    print('\n******* WEIGHTS OF ANN *******\n')
    for i in range(int(len(weights) / 2)):
        print('Weights W%d:\n' % (i), weights[i * 2])
        print('Bias b%d:\n' % (i), weights[(i * 2) + 1])


def normalize_column(dataframe, col_name):
    maximum = max(dataframe[col_name])
    minimum = min(dataframe[col_name])
    dataframe[col_name] = (dataframe[col_name] - minimum) / (maximum - minimum)
    return dataframe[col_name]


# %% Load in file, remove punctuation and weblinks from all tweets and then
#   perform sentiment analysis.
print('\n\n********** CLEANING TWEETS **********\n\n')
filename = "Apple_Blog_tweets.csv"
df = pd.read_csv(filename)
df["Clean Tweet"] = df['text'].apply(lambda x: clean_tweet(x))
df['created_at'] = pd.to_datetime(df['created_at']).dt.date

# %% Data cleaning and processing
# Choose keywords, count how many times each key word was used in tweets
print('\n\n********** IDENTIFYING KEY WORDS **********\n\n')
key_words = ['COVID', 'MacBook', 'iPhone', 'iPad', 'TikTok', 'Watch',
             'Glass', 'AirPods', 'Google', 'Microsoft', 'Facebook', 'President', 'Android', 'AppleTV']
df['noof_keywords'] = np.where(df.text.str.contains('|'.join(key_words)), 1, 0)
for key_word in key_words:
    df[key_word] = np.where(df.text.str.contains(key_word), 1, 0)
# list of invalid days: including federal holidays, days for when we have no data
invalid_days = ['2020-08-03', '2020-08-04', '2020-08-05', '2020-08-06', '2020-08-07',
                '2020-08-10', '2020-08-11', '2020-08-12', '2020-08-13', '2020-08-14',
                '2020-08-17', '2020-08-18', '2020-08-19', '2020-08-20', '2020-08-21',
                '2020-08-24', '2020-08-25', '2020-08-26', '2020-08-27', '2020-08-28']
missing_days = []
noof_missing_days = len(missing_days)
invalid_days = list(map(lambda x: str_to_date(x), invalid_days))

# if the tweets are in invalid days, move them to the next day for prediction.
for index, row in df.iterrows():
    if row['created_at'] in invalid_days:
        new_day = row['created_at'] + datetime.timedelta(days=1)
        df.loc[index, 'created_at'] = new_day
# Move tweets sent on a saturday or sunday to the nearest Monday.
# When using the weekday() function -> 0 is Monday and 6 is Sunday
for index, row in df.iterrows():
    if row['created_at'].weekday() == 5:
        new_day = row['created_at'] + datetime.timedelta(days=2)
        df.loc[index, "created_at"] = new_day
    elif row['created_at'].weekday() == 6:
        new_day = row['created_at'] + datetime.timedelta(days=1)
        df.loc[index, 'created_at'] = new_day
# group the data by date - aggregate all keywords used and sum them up.
# for example, the tweets from the day mention 'dollar' multiple times, add
#   them together.
columns_to_aggregate = {'noof_keywords':'sum'}
for key_word in key_words:
    columns_to_aggregate[key_word] = 'sum'
grouped_df = df.groupby(by='created_at').agg(columns_to_aggregate).reset_index()
# %% Stock prices
filename = "Apple_stocks.csv"
snp_df = pd.read_csv(filename)
# return at close each day
snp_df['percent change'] = snp_df['Close'].pct_change()

missing_days = list(map(lambda x: str_to_date(x), missing_days))
for missing_day in missing_days:
    grouped_df = grouped_df[grouped_df['created_at'] != missing_day]

# %% neural network - IN PROGRESS
# create the input data - last 10 days will be used for predictions at the end
# this window is subject to change
grouped_df['day_opening'] = snp_df['Open']
grouped_df['units_traded'] = snp_df['Volume']
days_to_predict = 0.35 * len(grouped_df) + noof_missing_days

# before putting any data into NN, we normalize the data (between 0 and 1)
# Normailize only the open and volume traded (beacuse they are large compared
# to the other input values)
grouped_df['day_opening'] = normalize_column(grouped_df, 'day_opening')
grouped_df['units_traded'] = normalize_column(grouped_df, 'units_traded')

features = ['noof_keywords', 'day_opening', 'units_traded']
features = features + key_words
num_inputs = len(features)
x = np.array(grouped_df[1:int(-1*days_to_predict)][features])
y = np.array(snp_df[1:(int(-1*days_to_predict)+1)][['percent change']])  # type: Any
#x = np.array(grouped_df[:int(days_to_predict)][features])
#y = np.array(snp_df[:int(days_to_predict)][['percent change']])

neural_net = tf.keras.Sequential()

# define the input layer
## add another hidden layer with 4 neurons to the NN
neural_net.add(tf.keras.layers.Dense(4, activation='relu', input_shape=(num_inputs,)))
## add another hidden layer with 8 neurons to the NN
neural_net.add(tf.keras.layers.Dense(8, activation='relu'))
## add an output layer with a single output (percent change)
neural_net.add(tf.keras.layers.Dense(1, activation='linear'))

neural_net.compile(optimizer='adam', loss='mean_absolute_error')

## train the ANN model using 1200 iterations
print('\n\n********** Begin ANN training **********\n\n')
neural_net.fit(x, y, epochs=1200)

## record the frequency of the loss beginning
figure, ax = plt.subplots(2, 2, figsize=(10, 5))
for (i, lnrate) in enumerate([0.1, 0.01, 0.001, 0.0001]):
    loss = []
    for j in range(10):
        opt = tf.keras.optimizers.Adam(learning_rate=lnrate)
        neural_net.compile(optimizer=opt, loss='mean_squared_error')
        neural_net.fit(x, y, epochs=100)
        loss.append(neural_net.evaluate(x, y))
    print(loss)
    ax[i // 2, i % 2].hist(loss, bins=5, edgecolor='black')
    # display y label
    ax[i // 2, i % 2].set_xlabel("Loss")
    # display x label
    ax[i // 2, i % 2].set_ylabel("Frequency")
    # display title
    ax[i // 2, i % 2].set_title("Learning Rate: " + str(lnrate))

plt.tight_layout()
plt.savefig("histogram.png")
plt.show()
## record the frequency of the loss ending

weights = neural_net.get_weights()
print_weights(weights)
print('\n\n********** ANN training complete **********\n\n')

# %%  make predictions
noof_correct_movement = 0
noof_predictions = 0
diffs = []
input_features = []
print('\n\n********** ANN PREDICTIONS **********\n\n')
for i in range(int(days_to_predict), int(noof_missing_days), -1):
    input_features.clear()
    noof_predictions += 1
    actual_change = snp_df.iloc[(-1 * i) + 1]['percent change']
    for feature in features:
        input_features.append(grouped_df.iloc[-1 * i][feature])
    predicted_change = neural_net.predict(np.array([input_features]))[0, 0]

    print(f"The predicted change for {grouped_df.iloc[(-1 * i) + 1]['created_at']}",
          end='')
    print(f" was: {predicted_change:.5f}")
    print(f"Actual change was for {snp_df.iloc[(-1 * i) + 1]['Date']} was: ",
          end='')
    print(f"{round(actual_change, 4)}\n")

    if (predicted_change * actual_change) > 0:
        diffs.append(math.fabs(predicted_change - actual_change))
        noof_correct_movement += 1

percent_correct = (noof_correct_movement / noof_predictions) * 100
print(f"ANN was correct in predicting the movement ", end = '')
print(f"{((noof_correct_movement / noof_predictions) * 100):.2f}% of the time in",
      end='')
print(f" {noof_predictions} predictions.")
average_diff = round(sum(diffs) / len(diffs), 5)
print(f"The average error of the correct predictions were", end='')
print(f" {average_diff * 100.0:.1f} %")

with open('output.txt', 'a') as output_file:
    output_file.write(f'{date.today()} percentage of times correct prediction:')
    output_file.write(f' {round(percent_correct, 1)} % ')
    output_file.write(f' with error {average_diff * 100.0:.1f}%\n')

grouped_df.to_csv('AppleTweet_Input.csv')