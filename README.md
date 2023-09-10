# LSTM-Stock-Price-Prediction-PyTorch
Time Series stock price prediction. Focuses on multi-sequence future predictions, full sequence future predictions and next-day predictions using a PyTorch LSTM implementation. 

The aim of this project is to implement these three prediction methods, building on the ideas from the article: https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks, using PyTorch's LSTM (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) and running it on the FTSE 100 Index price as an example.

## Requirements
* Python 3.10 <br/>
* matplotlib 3.7.2<br/>
* numpy 1.22.3<br/>
* scikit-learn 1.0.2<br/>
* torch 2.0.1<br/>
* yfinance 0.2.28<br/>



## Results
Multiple sequence future prediction:

<img src="https://github.com/RoryCoulson/LSTM-Stock-Price-Prediction-PyTorch/assets/52762734/698041af-7637-4a61-92b4-7960fc538fd0"><br/><br/>

Without using dropout:

<img src="https://github.com/RoryCoulson/LSTM-Stock-Price-Prediction-PyTorch/assets/52762734/471a7cd9-6ebc-4c26-87f0-afc45d377d58"><br/><br/>

Full sequence future prediction:

<img src="https://github.com/RoryCoulson/LSTM-Stock-Price-Prediction-PyTorch/assets/52762734/45961f17-74ba-4107-9f3e-b77f670aa285"><br/><br/>

Next point prediction. Note these predictions use the true test data unlike the others that make predictions from the output of previous predictions, so this can only be used for next-day predictions:

<img src="https://github.com/RoryCoulson/LSTM-Stock-Price-Prediction-PyTorch/assets/52762734/10db9bd6-7fa3-43c1-b38d-4cc3ffb2dade">

## References
* https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction
* https://www.datacamp.com/tutorial/lstm-python-stock-market
* https://github.com/pytorch/examples/blob/main/time_sequence_prediction
