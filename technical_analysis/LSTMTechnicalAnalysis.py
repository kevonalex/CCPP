import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import pandas_datareader as web
import datetime as dt
import sklearn
import yfinance as yf
import time
import keras
# import keyboard
# import sys
import csv

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


class CryptoPricePredictor:
    def __init__(self, crypto_currency, fiat_currency):
        self.crypto_currency = crypto_currency
        self.fiat_currency = fiat_currency
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = Sequential()
        self.is_running = True

    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        x_train, y_train = [], []
        prediction_days = 45
        future_day = 30

        for x in range(prediction_days, len(scaled_data) - future_day):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x + future_day, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train

    def prepare_data(self, data, prediction_days, future_day):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        x_train, y_train = [], []

        for x in range(prediction_days, len(scaled_data) - future_day):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x + future_day, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train

    # def prepare_data(self, data, prediction_days, future_day, window_size):
    #     scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
    #
    #     x_train, y_train = [], []
    #
    #     for x in range(prediction_days + window_size, len(scaled_data) - future_day):
    #         x_train.append(scaled_data[x - prediction_days - window_size:x - future_day, 0])
    #         y_train.append(scaled_data[x + future_day, 0])
    #
    #     # Calculate moving average
    #     moving_averages = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    #     moving_averages_scaled = self.scaler.transform(moving_averages.reshape(-1, 1))
    #
    #     # Select only the moving averages corresponding to the x_train data
    #     moving_averages_scaled = moving_averages_scaled[window_size:len(x_train) + window_size]
    #
    #     # Combine price data and moving averages
    #     x_train = np.concatenate((x_train, moving_averages_scaled), axis=1)
    #
    #     x_train, y_train = np.array(x_train), np.array(y_train)
    #     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    #
    #     return x_train, y_train

        # ...

    def create_model(self, x_train):
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=35, batch_size=32, verbose=0) # verbose zero will hide output

    def predict_prices(self, model_inputs, prediction_days, future_day):
        x_test = []

        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        prediction_prices = self.model.predict(x_test)
        prediction_prices = self.scaler.inverse_transform(prediction_prices)

        return prediction_prices

    # def run_strategy(self, start, end, test_start, test_end, prediction_days, future_day, window_size, stop_loss_percent, take_profit_percent):
    def run_strategy(self, start, end, test_start, test_end, prediction_days, future_day, stop_loss_percent, take_profit_percent):

            historical_data = yf.download(f'{self.crypto_currency}-{self.fiat_currency}', start, end)

            # x_train, y_train = self.prepare_data(historical_data['Close'].values, prediction_days, future_day, window_size)
            x_train, y_train = self.prepare_data(historical_data['Close'].values, prediction_days, future_day)

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
            plt.plot(actual_prices, color='black', label='Actual Prices')
            plt.plot(prediction_prices, color='green', label='Predicted Prices')
            plt.title(f'{crypto_currency} price prediction')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend(loc='upper left')
            plt.show()

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

#  date time format is yyyy/mm/dd

history_start = dt.datetime(2022, 1, 1)
history_end = dt.datetime(2022, 12, 31)
test_start = dt.datetime(2023, 1, 1)
test_end = dt.datetime.now()
prediction_days = 45
future_day = 30
stop_loss_percent = 0.03  # 3%
take_profit_percent = 0.05  # 5%
# window_size = 10

program_active = True
count=0
output_file = "output.csv"

# Run the strategy continuously until the user decides to stop
while program_active:
    print("<--------------- Running LSTM prediction algorithm on '" + crypto_currency + "-" + fiat_currency + "' pair - Iteration: " + str(count) + " --------------->")
    predictor = CryptoPricePredictor(crypto_currency, fiat_currency)
    predictor.run_strategy(history_start, history_end, test_start, test_end, prediction_days, future_day, stop_loss_percent, take_profit_percent)
    # predictor.run_strategy(history_start, history_end, test_start, test_end, prediction_days, future_day, window_size, stop_loss_percent, take_profit_percent)
    count+=1

    # if keyboard.is_pressed('enter'):
    #     program_active = False
    #     sys.exit()
        # break

