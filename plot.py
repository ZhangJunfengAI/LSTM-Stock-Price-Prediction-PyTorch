import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from config import COMPANY_TICKER, PREDICTION_TYPE

FIGURE_SIZE = (13, 8)


def plot_stock_price_history(currency, stock_df):
    plt.figure(figsize=FIGURE_SIZE)
    stock_df['Close'].plot()
    plt.title(f'{COMPANY_TICKER} Stock Price History')
    plt.xlabel('Date')
    plt.ylabel(f'Price ({currency})')
    plt.show()


def plot_train_test_split(train_data, test_data, currency):
    plt.figure(figsize=FIGURE_SIZE)
    plt.title('Train test split')
    plt.plot(np.arange(len(train_data)), train_data, label='Train data')
    plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), test_data,
             label='Test data')
    plt.ylabel(f'Price ({currency})')
    plt.legend()
    plt.show()


def plot_normalized_data(train_data, test_data):
    plt.figure(figsize=FIGURE_SIZE)
    plt.title('Normalized Train and Test data')
    plt.plot(np.arange(len(train_data)), train_data, label='Normalized Train data')
    plt.plot(np.arange(len(train_data), len(
        train_data) + len(test_data)), test_data, label='Normalized Test data')
    plt.legend()
    plt.show()


def plot_multiple_sequence_predictions(
        multiple_sequence_predictions, test_target, step, show=False):
    colours = cm.tab20(np.linspace(0, 1, len(multiple_sequence_predictions)+2))
    plt.figure(figsize=FIGURE_SIZE)
    plt.title(f'{PREDICTION_TYPE} {COMPANY_TICKER} results')
    plt.plot(test_target.detach().numpy(), label='Test data')
    for c, (x, y_pred) in zip(colours[2:], multiple_sequence_predictions):
        plt.plot(x, y_pred, label=f'Prediction', c=c)
    plt.legend()
    save_plot(step, show)


def plot_predictions(prediction, test_target, step, show=False):
    plt.figure(figsize=FIGURE_SIZE)
    plt.title(f'{PREDICTION_TYPE} {COMPANY_TICKER} results')
    plt.plot(test_target.detach().numpy(), label='Test data')
    plt.plot(prediction, label='Prediction')
    plt.legend()
    save_plot(step, show)


def save_plot(step, show):
    path = f'prediction_plots/step_{step}_predictions.png'
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
