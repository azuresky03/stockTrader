import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from models.LSTM_Univariant import LSTM
from models.Sequence import SequenceDataset
import os


DATA_DIR = "./data"
RES_DIR = "./results"


def train(model: LSTM, n_epochs: int, device, trainloader: DataLoader, testloader: DataLoader, save_model_path: str): #testloader: DataLoader,
    t_losses, v_losses = [], []
    criterion = MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for epoch in range(n_epochs):
      train_loss, valid_loss = 0.0, 0.0

      # train step
      model.train()
      # Loop over train dataset
      for x, y in trainloader:
        optimizer.zero_grad()
        # move inputs to device
        x = x.to(device)
        y  = y.squeeze().to(device)

        # Forward Pass
        preds = model(x).squeeze()

        loss = criterion(preds, y) # compute batch loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
      epoch_loss = train_loss / len(trainloader)
      t_losses.append(epoch_loss)
      
      # validation step
      model.eval()
      # Loop over validation dataset
      for x, y in testloader:
        with torch.no_grad():
          x, y = x.to(device), y.squeeze().to(device)
          preds = model(x).squeeze()
          error = criterion(preds, y)
        valid_loss += error.item()
      valid_loss = valid_loss / len(testloader)
      v_losses.append(valid_loss)
          
      print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
    # plot_losses(t_losses, v_losses)

    torch.save(model.state_dict(), save_model_path)
    print("Model is saved to:", save_model_path)


def make_predictions_from_dataloader(model, unshuffled_dataloader):
  model.eval()
  predictions, actuals = [], []
  for x, y in unshuffled_dataloader:
    with torch.no_grad():
      p = model(x)
      predictions.append(p)
      actuals.append(y.squeeze())
  predictions = torch.cat(predictions).numpy()
  actuals = torch.cat(actuals).numpy()
  return predictions.squeeze(), actuals

def one_step_forecast(model: LSTM, history: np.ndarray):
      '''
      model: PyTorch model object
      history: a sequence of values representing the latest values of the time 
      series, requirement -> len(history.shape) == 2
    
      outputs a single value which is the prediction of the next value in the
      sequence.
      '''
      model.cpu()
      model.eval()
      with torch.no_grad():
        pre = torch.Tensor(history).unsqueeze(0)
        pred = model(pre)
      return pred.detach().numpy().reshape(-1)


def n_step_forecast(model: LSTM, data: pd.DataFrame, true_data: pd.DataFrame, target: str, tw: int, n: int, forecast_from: int=None, plot=False):
    '''
    n: integer defining how many steps to forecast
    forecast_from: integer defining which index to forecast from. None if
    you want to forecast from the end.
    plot: True if you want to output a plot of the forecast, False if not.
    '''
    history = data[target].copy().to_frame()

    if forecast_from:
        pre = list(history[forecast_from - tw : forecast_from][target].values)
    else:
        pre = list(history[target])[-tw:]


    for i, step in enumerate(range(n)):
       pre_ = np.array(pre[-tw:], dtype=np.float64).reshape(-1, 1)
       forcast = one_step_forecast(model, pre_)
       pre.append(forcast)

    res = history.copy()
    ls = [np.nan for i in range(len(history))]

    # Note: I have not handled the edge case where the start index + n is 
    # before the end of the dataset and crosses past it.
    if forecast_from:
        ls[forecast_from : forecast_from + n] = list(np.array(pre[-n:]))
        res['forecast'] = ls
        res.columns = ['actual', 'forecast']
    else:
        fc = ls + list(np.array(pre[-n:]))
        ls = ls + [np.nan for i in range(len(pre[-n:]))]
        ls[:len(history)] = history[target].values
        res = pd.DataFrame([ls, fc], index=['actual', 'forecast']).T
    return res


def main():
  df = pd.read_csv('./data/Train万华化学.csv', sep=',')
  test_df = pd.read_csv('./data/Test万华化学.csv', sep=',')
  n_features= 12          # input features       dropped "adjustflag", "date" - reference_date, "turn"
  nhid = 64               # Number of nodes in the hidden layer
  n_dnn_layers = 5        # Number of hidden fully connected layers
  nout = 1                # Prediction Window
  sequence_len = 100      # Training Window
  num_preds = 30          # predictions 

  USE_CUDA = torch.cuda.is_available()
  device = 'cuda' if USE_CUDA else 'cpu'

  model = LSTM(n_features, nhid, nout, sequence_len, n_deep_layers=n_dnn_layers).to(device)
  # model.load_state_dict(torch.load('./savedModel/万华化学'))

  train_dataloader, test_dataloader = model.load_data(df, isShuffle=False, sequence_len=sequence_len, nout=nout)
  train(model=model, n_epochs = 20, device=device, trainloader=train_dataloader, testloader=test_dataloader, save_model_path='./savedModel/万华化学') # testloader=test_df,

  #-----------------------------------------

  res = n_step_forecast(model=model, target="close", data=df, true_data=test_df, tw=sequence_len, n=num_preds).squeeze()


  # predictions = predictions['close']
  actuals = np.array(test_df["close"].values[:num_preds])


  figure, axes = plt.subplots(figsize=(15, 6))
  axes.xaxis_date()

  dates = np.array(pd.date_range(start="2020-01-04", periods=num_preds, freq="B"))

  predictions = np.array(res.iloc[:num_preds, :]["forecast"])
  actuals = np.array(test_df.iloc[:num_preds, :]["close"])

  axes.plot(dates, actuals, color = 'red', label = 'Real Stock Price')
  axes.plot(dates, predictions, color = 'blue', label = 'Predicted Stock Price')
  #axes.xticks(np.arange(0,394,50))
  plt.title('Stock Price Prediction')
  plt.xlabel('Time')
  plt.ylabel('Stock Price')
  plt.legend()
  plt.savefig('./plots/万华化学.png')
  # plt.show()

  
if __name__ == "__main__":
    main()