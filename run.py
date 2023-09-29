from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import HIDDEN_LAYERS, LEARNING_RATE, PREDICTION_TYPE, STEPS, WINDOW_SIZE, PredictionType
from data_loader import load_data
from model import LSTM, predict_full_sequence, predict_next_point, predict_sequence
from plot import plot_multiple_sequence_predictions, plot_predictions

# create directory for prediction plots
Path("prediction_plots/").mkdir(parents=True, exist_ok=True)

np.random.seed(0)
torch.manual_seed(0)

train_input, train_target, test_input, test_target = load_data(plot=True)

model = LSTM(hidden_layers=HIDDEN_LAYERS)
criterion = nn.MSELoss()
# LBFGS optimizer since it can load the whole data for training
optimizer = optim.LBFGS(model.parameters(), lr=LEARNING_RATE)

for step in range(STEPS):

    def closure():
        optimizer.zero_grad()
        out = model(train_input)
        train_loss = criterion(out, train_target)
        print(f'train loss: {train_loss.item()}')
        train_loss.backward()
        return train_loss
    optimizer.step(closure)

    with torch.no_grad():
        match PREDICTION_TYPE:
            case PredictionType.NEXT_POINT:
                prediction = predict_next_point(
                    test_input, test_target, model, criterion)
                plot_predictions(prediction, test_target, step)
            case PredictionType.MULTIPLE_SEQUENCES:
                multiple_sequence_predictions = []
                # non overlapping windows of test data to use for each sequence prediction
                for i in range(int(len(test_target)/WINDOW_SIZE)):
                    prediction = predict_sequence(
                        test_input, test_target, i, model, criterion)
                    multiple_sequence_predictions.append(
                        (np.arange(i*WINDOW_SIZE, (i+1)*WINDOW_SIZE), prediction))
                plot_multiple_sequence_predictions(
                    multiple_sequence_predictions, test_target, step)
            case PredictionType.FULL_SEQUENCE:
                prediction = predict_full_sequence(
                    test_input, test_target, model, criterion)
                plot_predictions(prediction, test_target, step)
