import torch
import torch.nn as nn

from config import WINDOW_SIZE


class LSTM(nn.Module):
    def __init__(self, hidden_layers=64, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        self.num_layers = num_layers

        # using dropout as a regularization technique to prevent overfitting
        self.lstm = nn.LSTM(1, self.hidden_layers, self.num_layers, batch_first=True, dropout=0.2)
        # self.gru = nn.GRU(1, self.hidden_layers, self.num_layers, batch_first=True, dropout=0.2) # gru

        self.linear = nn.Linear(self.hidden_layers, 1)

    def forward(self, input, future_preds=0):
        outputs = []
        h_t = torch.zeros(self.num_layers, input.size(0), self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(self.num_layers, input.size(0), self.hidden_layers, dtype=torch.float32)

        for input_t in input.split(1, dim=1):
            out, _ = self.lstm(input_t, (h_t, c_t))
            # out, _ = self.gru(input_t, h_t) # gru
            output = self.linear(out)
            outputs.append(output)

        # if future_preds then the model will make predictions using output as input
        for _ in range(future_preds):
            out, _ = self.lstm(output, (h_t, c_t))
            # out, _ = self.gru(output, h_t) # gru
            output = self.linear(out)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs


def predict_next_point(test_input, test_target, model, criterion):
    # predicts the next point for each test data sliding window, only using test data as input
    prediction = model(test_input)
    # the prediction is all the last values for every predicted sliding window
    prediction = prediction[:, -1].reshape(-1)
    test_loss = criterion(prediction, test_target)
    print('test loss:', test_loss.item())
    prediction = prediction.detach().numpy()
    return prediction


def predict_sequence(test_input, test_target, i, model, criterion):
    # predicts a sequence of future steps given the ith test window
    prediction = model(
        test_input[i * WINDOW_SIZE: (i + 1) * WINDOW_SIZE - 1].reshape(1, -1, 1),
        future_preds=WINDOW_SIZE)
    # filter to only include future predictions
    prediction = prediction[:, WINDOW_SIZE-1:].reshape(-1)
    test_loss = criterion(prediction, test_target[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE])
    print('test loss:', test_loss.item())
    prediction = prediction.detach().numpy()
    return prediction


def predict_full_sequence(test_input, test_target, model, criterion):
    # predicts all future steps given the first test window
    prediction = model(test_input, future_preds=len(test_target))
    # filter to only include future predictions
    prediction = prediction[:, WINDOW_SIZE-1:].reshape(-1)
    test_loss = criterion(prediction, test_target)
    print('test loss:', test_loss.item())
    prediction = prediction.detach().numpy()
    return prediction
