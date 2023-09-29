import torch
import numpy as np
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from plot import plot_normalized_data, plot_stock_price_history, plot_train_test_split
from config import COMPANY_TICKER, PREDICTION_TYPE, TRAIN_SPLIT, PredictionType, WINDOW_SIZE


def normalize(train_data, test_data, plot=False):
    # normalizes the data in separate windows to allow lower price periods to still have a significance
    normalize_window = 2000
    scaler = MinMaxScaler()
    for i in range(0, len(train_data), normalize_window):
        train_data[i:i+normalize_window,
                   :] = scaler.fit_transform(train_data[i:i+normalize_window, :])
    # apply the normalized transformation to the test data afterwards
    test_data = scaler.transform(test_data)

    if plot:
        plot_normalized_data(train_data, test_data)
    return train_data.reshape(-1), test_data.reshape(-1)


def train_test_split(prices, currency, plot=False):
    train_size = int(np.round(TRAIN_SPLIT * len(prices)))
    train_data = prices[:train_size].reshape(-1, 1)
    test_data = prices[train_size:].reshape(-1, 1)
    if plot:
        plot_train_test_split(train_data, test_data, currency)
    return train_data, test_data


def get_train_data_split(train_data):
    # splits out the train data into train_input and train_target
    train_windows = np.empty((len(train_data) - WINDOW_SIZE + 1, WINDOW_SIZE), dtype=np.float32)
    # fill train_windows using a sliding window across the train data
    for i in range(len(train_data) - WINDOW_SIZE + 1):
        train_windows[i] = train_data[i:i+WINDOW_SIZE]

    # all train windows but excludes the final price in each
    train_input = torch.from_numpy(train_windows[:, :-1]).reshape(-1, WINDOW_SIZE-1, 1)
    # all train windows but excludes the first price in each, corresponding to the next timestep in the sequence
    train_target = torch.from_numpy(train_windows[:, 1:]).reshape(-1, WINDOW_SIZE-1, 1)

    return train_input, train_target


def get_test_data_split(test_data):
    # splits out the test data into test_input and test_target
    test_prices = test_data.astype(np.float32)
    match PREDICTION_TYPE:
        case PredictionType.NEXT_POINT:
            # makes predictions using the true test data windows
            test_windows = np.empty(
                (len(test_data) - WINDOW_SIZE + 1, WINDOW_SIZE), dtype=np.float32)
            for i in range(len(test_data) - WINDOW_SIZE + 1):
                test_windows[i] = test_data[i:i+WINDOW_SIZE]
            # all test window excluding the last point
            test_input = torch.from_numpy(test_windows[:, :-1]).reshape(-1, WINDOW_SIZE-1, 1)
            # all values after the first input window - equivalent to the last point of every test_input window
            test_target = torch.from_numpy(test_prices[WINDOW_SIZE-1:])
            return test_input, test_target

        case PredictionType.MULTIPLE_SEQUENCES:
            # full test data since this will be split into sequences when predicting
            test_input = torch.from_numpy(test_prices)
            # starts at future prediction starting point, excluding predictions made with test data as input
            test_target = torch.from_numpy(test_prices[WINDOW_SIZE:])
            return test_input, test_target

        case PredictionType.FULL_SEQUENCE:
            # only use the first input window for full sequence prediction
            test_input = torch.from_numpy(test_prices[:WINDOW_SIZE-1]).reshape(1, -1, 1)
            # starts at future prediction starting point, excluding predictions made with test data as input
            test_target = torch.from_numpy(test_prices[WINDOW_SIZE:])
            return test_input, test_target


def load_data(start='2000-01-01', end='2023-09-01', plot=False):
    stock_df = yf.download(COMPANY_TICKER, start=start, end=end)
    currency = yf.Ticker(COMPANY_TICKER).info['currency']

    if plot:
        plot_stock_price_history(currency=currency, stock_df=stock_df)

    prices = stock_df['Close'].values

    train_data, test_data = train_test_split(prices, currency, plot=True)
    train_data, test_data = normalize(train_data, test_data, plot=True)

    train_input, train_target = get_train_data_split(train_data)
    test_input, test_target = get_test_data_split(test_data)

    return train_input, train_target, test_input, test_target
