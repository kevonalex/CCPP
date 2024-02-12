import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import sklearn
import yfinance as yf
import time
import keyboard
import sys
import csv
import random

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, concatenate
from tensorflow.keras.models import Sequential

class CryptoPricePredictor:
    def __init__(self, crypto_currency, fiat_currency):
        self.crypto_currency = crypto_currency
        self.fiat_currency = fiat_currency
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = Sequential()
        self.is_running = True

    # def prepare_data(self, data):
    #     scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
    #
    #     x_train, y_train = [], []
    #     prediction_days = 45
    #     future_day = 30
    #
    #     for x in range(prediction_days, len(scaled_data) - future_day):
    #         x_train.append(scaled_data[x - prediction_days:x, 0])
    #         y_train.append(scaled_data[x + future_day, 0])
    #
    #     x_train, y_train = np.array(x_train), np.array(y_train)
    #     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    #
    #     return x_train, y_train

    def prepare_data(self, data, sentiment_data, prediction_days, future_day):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        scaled_sentiment_data = self.scaler.fit_transform(sentiment_data.reshape(-1, 1))

        x_lstm_train, x_sentiment_train, y_train = [], [], []

        for x in range(prediction_days, len(scaled_data) - future_day):
            x_lstm = scaled_data[x - prediction_days:x, 0]
            x_sentiment = scaled_sentiment_data[x - prediction_days:x, 0]

            x_lstm_train.append(x_lstm)
            x_sentiment_train.append(x_sentiment)
            y_train.append(scaled_data[x + future_day, 0])

        x_lstm_train = np.array(x_lstm_train)
        x_sentiment_train = np.array(x_sentiment_train)
        y_train = np.array(y_train)

        x_lstm_train = np.reshape(x_lstm_train, (x_lstm_train.shape[0], x_lstm_train.shape[1], 1))
        x_sentiment_train = np.reshape(x_sentiment_train, (x_sentiment_train.shape[0], x_sentiment_train.shape[1], 1))

        return [x_lstm_train, x_sentiment_train], y_train

    # def create_model(self,x_train):
    #     self.model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    #     self.model.add(Dropout(0.2))
    #     self.model.add(LSTM(units=50, return_sequences=True))
    #     self.model.add(Dropout(0.2))
    #     self.model.add(LSTM(units=50))
    #     self.model.add(Dropout(0.2))
    #     self.model.add(Dense(units=1))

        # self.model.compile(optimizer='adam', loss='mean_squared_error')

    def create_model(self, x_train):
        x_train = np.array(x_train)
        sentiment_input = Input(shape=(x_train[1].shape[1]))  # Input shape: (timesteps, features)
        lstm_input = Input(shape=(x_train[0].shape[1], 1), name='lstm_input')

        lstm_layer = LSTM(units=50, return_sequences=True)(lstm_input)
        lstm_layer = Dropout(0.2)(lstm_layer)

        lstm_layer_reshaped = LSTM(units=50)(lstm_layer)
        lstm_layer_reshaped = Dropout(0.2)(lstm_layer_reshaped)

        # lstm_layer = LSTM(units=50, return_sequences=True)(lstm_layer)
        # lstm_layer = Dropout(0.2)(lstm_layer)
        #
        # lstm_layer = LSTM(units=50)(lstm_layer)
        # lstm_layer = Dropout(0.2)(lstm_layer)

        combined_layer = concatenate([lstm_layer_reshaped, sentiment_input])
        # combined_layer = concatenate([lstm_layer, sentiment_input])

        dense_layer = Dense(units=1)(combined_layer)

        self.model = keras.Model(inputs=[lstm_input, sentiment_input], outputs=dense_layer)

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=25, batch_size=32)#, #verbose=0) # verbose zero will hide output

    def predict_prices(self, model_inputs, prediction_days, future_day):
        x_price_test = []
        x_news_test = []

        for x in range(prediction_days, len(model_inputs)):
            x_price_test.append(model_inputs[0][x - prediction_days:x])
            x_news_test.append(model_inputs[1][x - prediction_days:x])

        x_price_test = np.array(x_price_test, dtype=object)
        x_news_test = np.array(x_news_test, dtype=object)

        # Create empty arrays to hold reshaped inputs
        reshaped_x_price_test = np.empty((len(x_price_test), prediction_days, 1))
        reshaped_x_news_test = np.empty((len(x_news_test), prediction_days, 1))

        for i in range(len(x_price_test)):
            reshaped_x_price_test[i] = np.reshape(x_price_test[i], (prediction_days, 1))
            reshaped_x_news_test[i] = np.reshape(x_news_test[i], (prediction_days, 1))

        # Concatenate price and news inputs
        x_test = np.concatenate((reshaped_x_price_test, reshaped_x_news_test), axis=2)

        prediction_prices = self.model.predict(x_test)
        prediction_prices = self.scaler.inverse_transform(prediction_prices)

        return prediction_prices

        ## OLD CODE HAS TWO ##

        # x_lstm_test, x_sentiment_test = model_inputs[0], model_inputs[1]
        # print("x_lstm_test shape:", x_lstm_test.shape)  # Add this line to check the shape
        #
        # if len(x_lstm_test.shape) < 3 :
        #     x_lstm_test = np.reshape(x_lstm_test, (x_lstm_test.shape[0], x_lstm_test.shape[1], 1))
        #
        # # x_lstm_test = np.reshape(x_lstm_test, (x_lstm_test.shape[0], x_lstm_test.shape[1], 1))
        # x_test = [x_lstm_test, x_sentiment_test]
        #
        # # for x in range(prediction_days, len(model_inputs)):
        # #     x_test.append(model_inputs[x - prediction_days:x, 0])
        #
        # # x_test = np.array(x_test)
        # # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        #
        # prediction_prices = self.model.predict(x_test)
        # prediction_prices = self.scaler.inverse_transform(prediction_prices)
        #
        # return prediction_prices

    # def run_strategy(self, start, end, test_start, test_end, prediction_days, future_day, window_size, stop_loss_percent, take_profit_percent):
    def run_strategy(self, start, end, test_start, test_end, prediction_days, future_day, stop_loss_percent, take_profit_percent):

            historical_data = yf.download(f'{self.crypto_currency}-{self.fiat_currency}', start, end)

            def generate_sequence(n):
                sequence = []
                current_value = random.uniform(1, 2)  # Start with a random value between 1 and 2

                for _ in range(n):
                    sequence.append(current_value)
                    current_value += random.uniform(0.01, 0.1)  # Increase the current value by a random amount

                    if current_value > 7:  # Cap the value at 7
                        current_value = 7

                return sequence

            historical_price_data = historical_data['Close'].values

            n = len(historical_price_data) # Number of values in the sequence
            historical_news_data = generate_sequence(n)
            historical_news_data = np.array(historical_news_data)

            # x_train, y_train = self.prepare_data(historical_data['Close'].values, prediction_days, future_day, window_size)
            x_train, y_train = self.prepare_data(historical_price_data, historical_news_data, prediction_days, future_day)

            self.create_model(x_train)
            self.train_model(x_train, y_train)

            # test_start = dt.datetime(2022, 1, 1)
            # test_end = dt.datetime.now()

            test_data = yf.download(f'{self.crypto_currency}-{self.fiat_currency}', test_start, test_end)
            actual_prices = test_data['Close'].values

            total_dataset = pd.concat((historical_data['Close'], test_data['Close']), axis=0)

            model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
            model_inputs = model_inputs.reshape(-1, 1)
            model_inputs = self.scaler.fit_transform(model_inputs)

            prediction_prices = self.predict_prices(model_inputs, prediction_days, future_day)

            # plot predictions
            # plt.plot(actual_prices, color='black', label='Actual Prices')
            # plt.plot(prediction_prices, color='green', label='Predicted Prices')
            # plt.title(f'{crypto_currency} price prediction')
            # plt.xlabel('Time')
            # plt.ylabel('Price')
            # plt.legend(loc='upper left')
            # plt.show()

            # predict next day

            real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
            real_data = np.array(real_data)
            real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

            prediction = self.model.predict(real_data)
            prediction = self.scaler.inverse_transform(prediction)
            print(prediction)
            # ...

            # predict next day

            real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
            real_data = np.array(real_data)
            real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

            prediction = self.model.predict(real_data)
            prediction = self.scaler.inverse_transform(prediction)

            # Determine buy/sell decision and exit price and stop loss values
            current_price = actual_prices[-1]
            predicted_price = prediction[0][0]
            timestamp = dt.datetime.now()


            if predicted_price > current_price:
                # Buy decision
                exit_price = current_price * (1 + take_profit_percent)
                stop_loss_price = current_price * (1 - stop_loss_percent)
                technical_decision = "Buy"
                print("Decision: " + technical_decision)
                print("Current Price:", current_price)
                print("Exit Price:", exit_price)
                print("Stop Loss Price:", stop_loss_price)
                print("Timestamp: ", timestamp)
                output = [technical_decision, current_price, exit_price, stop_loss_price, end]
                with open(output_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(output)


            else:
                # Sell decision
                exit_price = current_price * (1 - take_profit_percent)
                stop_loss_price = current_price * (1 + stop_loss_percent)
                technical_decision = "Sell"
                print("Decision: " + technical_decision)
                print("Current Price:", current_price)
                print("Exit Price:", exit_price)
                print("Stop Loss Price:", stop_loss_price)
                print("Timestamp: ", timestamp)
                output = [technical_decision, current_price, exit_price, stop_loss_price, end]
                with open(output_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(output)


# run program
# set strategy parameters
crypto_currency = 'BTC'
fiat_currency = 'GBP'
# history_start = dt.datetime(2022, 1, 1)
# history_end = dt.datetime.now()
history_start = dt.datetime(2021, 1, 1)
history_end = dt.datetime.now()
test_start = dt.datetime(2022, 1, 1)
test_end = dt.datetime.now()
prediction_days = 45
future_day = 30
stop_loss_percent = 0.03  # 3%
take_profit_percent = 0.05  # 5%
# window_size = 10

program_active = True
count = 1
output_file = "output.csv"

# Run the strategy continuously until the user decides to stop

count=1
while program_active:
    print(
        "<--------------- Running LSTM prediction algorithm on '" + crypto_currency + "-" + fiat_currency + "' pair - Iteration: " + str(
            count) + " --------------->")
    predictor = CryptoPricePredictor(crypto_currency, fiat_currency)
    predictor.run_strategy(history_start, history_end, test_start, test_end, prediction_days, future_day,
                           stop_loss_percent, take_profit_percent)
    # predictor.run_strategy(history_start, history_end, test_start, test_end, prediction_days, future_day, window_size, stop_loss_percent, take_profit_percent)
    count += 1

    # if keyboard.is_pressed('enter'):
    #     program_active = False
    #     # break

