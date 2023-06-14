import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('seaborn')
import os
import torch
from models.LSTM_multivariant import LSTM
from sklearn.preprocessing import MinMaxScaler
from torch.nn import MSELoss
from torch.utils.data import DataLoader

DATA_DIR = "./data"
RES_DIR = "./results"


def train(model: LSTM, n_epochs: int, device, scalers: MinMaxScaler, save_model_path: str, trainloader: DataLoader, testloader: DataLoader = None): 
    t_losses, v_losses = [], []
    criterion = MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    n_batches = len(trainloader)

    for epoch in range(n_epochs):
      train_loss, valid_loss, real_loss = 0.0, 0.0, 0.0

      # train step
      model.train()
      # Loop over train dataset
      for x, y in trainloader:
        optimizer.zero_grad()
        # move inputs to device
        x = x.to(device)
        y = y.squeeze().to(device)

        # Forward Pass
        preds = model(x).squeeze()
        pred_numpy = preds.detach().numpy()
        inversed_preds = scalers["close"].inverse_transform(pred_numpy.reshape(-1, 1))
        inversed_y = scalers["close"].inverse_transform(y.detach().numpy().reshape(-1, 1))

        loss = criterion(preds, y)  # compute batch loss
        real_loss += np.sum( np.abs(inversed_y - inversed_preds) )

        train_loss += loss.item()
        loss.backward()
        optimizer.step()
      epoch_loss = train_loss / n_batches
      epoch_mae = real_loss / n_batches
      t_losses.append(epoch_loss)
      
      if testloader:
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
      
      print(f'epoch {epoch + 1} --> train loss: {epoch_loss}, real mae loss: {epoch_mae}')
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


def one_step_forecast(model: LSTM, history: pd.DataFrame):
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
      # Create initial sequence input based on where in the series to forecast 
      # from.
      if forecast_from:
        pre = list(history[forecast_from - tw : forecast_from][target].values)
      else:
        pre = list(history[target].values)[-tw:]

      # TODO: getting this part working
      forecasts = []
      # Call one_step_forecast n times and append prediction to history
      for i, step in enumerate(range(n)):
        pre_ = np.array(pre[-tw:]).reshape(-1, 1)
        print(pre_.shape)
        forecast = one_step_forecast(model, pre_).squeeze()

        forecasts.append(forecast)
        pre.append(true_data.loc[i + forecast_from, "close"])
      
      # The rest of this is just to add the forecast to the correct time of 
      # the history series
      res = history.copy()
      ls = [np.nan for i in range(len(history))]

      if forecast_from:
        ls[forecast_from : forecast_from + n] = list(np.array(pre[-n:]))
        res['forecast'] = ls
        res.columns = ['actual', 'forecast']
      else:
        fc = ls + list(np.array(pre[-n:]))
        ls = ls + [np.nan for i in range(len(pre[-n:]))]
        ls[:len(history)] = history.values
        res = pd.DataFrame([ls, fc], index=['actual', 'forecast']).T
      
      return res, np.array(forecasts)

def process_single_stock(num_epoch: int, num_pred: int, train_file: str, test_file: str, isTrain=True):
  df = pd.read_csv(f'./data/{train_file}.csv', sep=',')
  test_df = pd.read_csv(f'./data/{test_file}.csv', sep=',')
  test_df = pd.concat([df, test_df], axis=0)
  combined_df = test_df.fillna(method='ffill')
  reference_date = pd.to_datetime("2014-06-03")

  n_features = 9
  nhid = 64          # Number of nodes in the hidden layer
  n_lstm_layers = 3
  n_dnn_layers = 3    # Number of hidden fully connected layers
  nout = 1            # Prediction Window
  sequence_len = 60   # Training Window

  USE_CUDA = torch.cuda.is_available()
  device = 'cuda' if USE_CUDA else 'cpu'

  model = LSTM(n_features, nhid, nout, n_lstm_layers=n_lstm_layers, sequence_len=sequence_len, n_deep_layers=n_dnn_layers).to(device)
  if isTrain:
    res = model.load_data(combined_df, isShuffle=False, split=1.0, reference_date=reference_date, sequence_len=sequence_len, nout=nout)
    if len(res) == 1:
      train_dataloader = res[0]
    else:
      scalers, train_dataloader = res
    train(model=model, n_epochs = num_epoch, device=device, scalers=scalers, trainloader=train_dataloader, save_model_path='./savedModel/万华化学.pt') # testloader=test_df,
    # last_epoch = last_epoch.flatten()

    # dates = np.array(pd.date_range(start="2014-06-03", periods=len(last_epoch), freq='B'))

    # new_df = pd.DataFrame(index=dates)
    # # new_df = pd.DataFrame(last_epoch)
    # new_df["close"] = last_epoch

    # new_df.to_csv(f"./predictions/{train_file[5:]}_pred.csv")
    # print(f'{train_file[5:]}_pred.csv')

    # code for plotting graph

    # figure, axes = plt.subplots(figsize=(15, 6))
    # axes.xaxis_date()

    # actuals = np.array(test_df["close"].values[:num_pred])
    # dates = np.array(pd.date_range(start="2020-01-04", periods=len(actuals), freq="B"))

    # axes.plot(dates, actuals, color = 'red', label = 'Real Stock Price')
    # axes.plot(dates, new_df, color = 'blue', label = 'Predicted Stock Price')
    # #  axes.plot(dates, predictions["forecast"].values[-num_pred:], color = 'blue', label = 'Predicted Stock Price')
    #  #axes.xticks(np.arange(0,394,50))
    # plt.title('Stock Price Prediction')
    # plt.xlabel('Time')
    # plt.ylabel('Stock Price')
    # plt.legend()
    # plt.savefig('./plots/万华化学.png')
  else:
    model.load_state_dict(torch.load('./savedModel/万华化学.pt'))
    scalers, dataloader = model.load_data(df, isShuffle=False, sequence_len=sequence_len, nout=nout, isTrain=True)
    
    _, predictions = n_step_forecast(model, df, target="close", tw=sequence_len, true_data=df, n=num_pred, forecast_from=100)

    actuals = np.array(test_df["close"].values[:num_pred])

    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    dates = np.array(pd.date_range(start="2020-01-04", periods=len(actuals), freq="B"))

    axes.plot(dates, actuals, color = 'red', label = 'Real Stock Price')
    axes.plot(dates, predictions, color = 'blue', label = 'Predicted Stock Price')
    # axes.plot(dates, predictions["forecast"].values[-num_pred:], color = 'blue', label = 'Predicted Stock Price')
    # axes.xticks(np.arange(0,394,50))
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('./plots/万华化学.png')
    # plt.show()

def main():
  dir = "./data"
  train_files, test_files = [], []
  for file in os.listdir(dir):
    name, end = file.split(".")
    if end == "csv":
      if name[1] == "e":
        test_files.append(name)
      else:
        train_files.append(name)
  
  for i in range(len(train_files)):
    train_file, test_file = train_files[i], test_files[i]
    process_single_stock(num_epoch=150, num_pred=60, train_file=train_file, test_file=test_file, isTrain=True)

  
if __name__ == "__main__":
    main()