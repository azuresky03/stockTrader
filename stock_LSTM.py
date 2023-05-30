import random

import numpy as np
import pandas as pd
from pylab import mpl, plt

plt.style.use('seaborn')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, num_layers: int, out_dim: int) -> None:
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # TODO: temporal information to capture?
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # added layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # added layer
        self.fc3 = nn.Linear(hidden_dim, out_dim)     # output layer


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        feed data into a 
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        out = self.fc3(out)

        return out
    

    @staticmethod
    def load_data(stock_data: pd.DataFrame, split_rate: float, look_back: int) -> list[np.ndarray]:
        """"
        split data to train and test for a single stock data
        params:
            stock_data -> data frame that needs to process
            split_rate -> determines percentage of given data to be train or test data
            look_back  -> provide time range to allow LSTM to search for pattern  
                          NOTE: this param should be consistent for both training and prediction process
            
        return:
            list of np.ndarray
        """
        # convert to numpy array
        data_raw = stock_data.values
        data = []
        
        # create all possible sequences of length look_back
        for index in range(len(data_raw) - look_back):
            data.append(data_raw[index: index + look_back])
        
        data = np.array(data)
        test_set_size = int(np.round(split_rate * data.shape[0]))
        train_set_size = data.shape[0] - (test_set_size)
        
        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,-1,:]
        
        x_test = data[train_set_size:,:-1]
        y_test = data[train_set_size:,-1,:]
        return [x_train, y_train, x_test, y_test]
    

    @staticmethod
    def stocks_data(symbols: list[str], dates: pd.DatetimeIndex, data_type: str, img_dir: str) -> list[pd.DataFrame]:
        """
        symbols: a list of file name that contains stock data (eg. csv files)
        dates:   date time index that indicates time range of stock data
        """
        datas = []
        scaler = MinMaxScaler(feature_range=(-1, 1))

        for symbol in symbols:
            df = pd.DataFrame(index=dates)
            df_temp = pd.read_csv("../stockTrader/data/{}.csv".format(data_type + symbol), index_col='date',
                    parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
            df_temp = df_temp.rename(columns={'close': symbol})
            df = df.join(df_temp)
            df.fillna(method='ffill')

            df[symbol] = scaler.fit_transform(df[symbol].values.reshape(-1,1))

            df.plot(figsize=(10, 6), subplots=True)
            plt.ylabel("stock_price")
            plt.xlabel("date")
            plt.legend()
            plt.savefig(f"{img_dir}/{symbol}_read.png")
            datas.append(df)
        
        return datas
    

    def train(
        self,
        model_path: str,
        stock_data: pd.DataFrame,
        stock_name: str,
        result_store_dir: str,
        loss_fn = torch.nn.MSELoss,
        optimizer_fn = torch.optim.Adam,
        split_rate: float = 0.2,
        learning_rate: float = 0.02,
        look_back: int = 30,
    ) -> list[torch.Tensor]:
        # load and preprocess data
        x_train, y_train, x_test, y_test = LSTM.load_data(stock_data, split_rate, look_back)

        # convert to tensor to allow back propagate
        x_train_tensor = torch.from_numpy(x_train).type(torch.Tensor)
        x_test_tensor = torch.from_numpy(x_test).type(torch.Tensor)
        y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor)
        y_test_tensor = torch.from_numpy(y_test).type(torch.Tensor)

        # model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=output_dim, num_layers=num_layers)
        # loss_fn = torch.nn.MSELoss()
        optimizer = optimizer_fn(self.parameters(), lr=learning_rate)

        num_epochs = 500
        hist = np.zeros(num_epochs)
        seq_dim =look_back-1 

        for e in range(num_epochs):
            y_train_pred = self(x_train_tensor)

            loss = loss_fn(y_train_pred, y_train_tensor)
            if e % 10 == 0 and e !=0:
                print("Epoch ", e, "MSE: ", loss.item())

            hist[e] = loss.item()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

        plt.plot(hist, label="Training loss")
        plt.legend()
        plt.savefig(f'{result_store_dir}/{stock_name}_training_loss.png')
        plt.show()

        torch.save(self.state_dict(), model_path)
        print("Model has been saved to:", model_path)
        return [x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor]


    def predict(self, data_tensors: list[torch.Tensor], stock_name: str, df: pd.DataFrame):
        """
        predict data based on 
        """
        x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = data_tensors

        with torch.no_grad():
            y_train_pred = self(x_train_tensor)
            y_test_pred = self(x_test_tensor)

        # scaling features to efficiently learn from data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # invert predictions
        y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
        y_train = scaler.inverse_transform(y_train_tensor.detach().numpy())
        y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
        y_test = scaler.inverse_transform(y_test_tensor.detach().numpy())

        figure, axes = plt.subplots(figsize=(15, 6))
        axes.xaxis_date()
        # print(type(np.array(df[len(df)-len(y_test):].index)))

        axes.plot(np.array(df[len(df)-len(y_test):].index), y_test, color = 'red', label = 'Real Stock Price')
        axes.plot(np.array(df[len(df)-len(y_test):].index), y_test_pred, color = 'blue', label = 'Predicted Stock Price')
        # axes.xticks(np.arange(0,394,50))
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(f'{stock_name}_pred.png')
        plt.show()

        return [y_train_pred, y_test_pred]


    def eval_metric(metric_method):
        pass


# TODO: add train and predict into model itself





if __name__ == "__main__":
    pass