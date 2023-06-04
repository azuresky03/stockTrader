import numpy as np
import pandas as pd 
from pylab import plt
plt.style.use('seaborn')
from sklearn.preprocessing import MinMaxScaler
import torch
from stock_LSTM import LSTM
import os


DATA_DIR = "./data"
RES_DIR = "./results"


# def stocks_data(symbols: list[str], dates: pd.DatetimeIndex, img_dir: str) -> list[pd.DataFrame]:
#     """
#     symbols: a list of file name that contains stock data
#     dates:   date time index that indicates time range of stock data
#     """
#     datas = []
#     scaler = MinMaxScaler(feature_range=(-1, 1))

#     for symbol in symbols:
#         df = pd.DataFrame(index=dates)
#         df_temp = pd.read_csv("../stockTrader/data/Train{}.csv".format(symbol), index_col='date',
#                 parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
#         df_temp = df_temp.rename(columns={'close': symbol})
#         df = df.join(df_temp)
#         df.fillna(method='ffill')

#         df[symbol] = scaler.fit_transform(df[symbol].values.reshape(-1,1))

#         df.plot(figsize=(10, 6), subplots=True)
#         plt.ylabel("stock_price")
#         plt.xlabel("date")
#         plt.legend()
#         plt.savefig(f"{img_dir}/{symbol}_read.png")

#         datas.append(df)
    
#     return datas


# def load_data(stock_data: pd.DataFrame, splt_rate: float, look_back: int) -> list[np.ndarray]:
#     """"
#     split data to train and test for a single stock data

#     return:
#         list of np.ndarray
#     """
#      # convert to numpy array
#     data_raw = stock_data.values
#     data = []
    
#     # create all possible sequences of length look_back
#     for index in range(len(data_raw) - look_back):
#         data.append(data_raw[index: index + look_back])
    
#     data = np.array(data)
#     test_set_size = int(np.round(splt_rate * data.shape[0]))
#     train_set_size = data.shape[0] - (test_set_size)
    
#     x_train = data[:train_set_size,:-1,:]
#     y_train = data[:train_set_size,-1,:]
    
#     x_test = data[train_set_size:,:-1]
#     y_test = data[train_set_size:,-1,:]

    
#     return [x_train, y_train, x_test, y_test]


# def train(
#         model_path: str,
#         stock_data: pd.DataFrame,
#         stock_name: str,
#         split_rate: float = 0.2,
#         look_back: int = 30
#     ) -> LSTM:
#     # load and preprocess data

#     x_train, y_train, x_test, y_test = load_data(stock_data, split_rate, look_back)

#     x_train_tensor = torch.from_numpy(x_train).type(torch.Tensor)
#     x_test_tensor = torch.from_numpy(x_test).type(torch.Tensor)
#     y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor)
#     y_test_tensor = torch.from_numpy(y_test).type(torch.Tensor)
    
#     # hyper parameters
#     input_dim: int = 1
#     hidden_dim: int = 12
#     num_layers: int = 2
#     output_dim: int = 1  
#     look_back: int = 30
#     learning_rate: float = 0.02

#     model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=output_dim, num_layers=num_layers)
#     loss_fn = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     num_epochs = 500
#     hist = np.zeros(num_epochs)
#     seq_dim =look_back-1 

#     for e in range(num_epochs):

#         y_train_pred = model(x_train_tensor)

#         loss = loss_fn(y_train_pred, y_train_tensor)
#         if e % 10 == 0 and e !=0:
#             print("Epoch ", e, "MSE: ", loss.item())

#         hist[e] = loss.item()
#         optimizer.zero_grad()

#         loss.backward()
#         optimizer.step()


#     print("Model's state_dict:")
#     for param_tensor in model.state_dict():
#         print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#     print("Optimizer's state_dict:")
#     for var_name in optimizer.state_dict():
#         print(var_name, "\t", optimizer.state_dict()[var_name])

#     plt.plot(hist, label="Training loss")
#     plt.legend()
#     plt.savefig(f'{RES_DIR}/{stock_name}_training_loss.png')
#     plt.show()

#     torch.save(model.state_dict(), model_path)
#     print("Model has been saved to:", model_path)
#     return model, [x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor]


# def predict(model: LSTM, data_tensors: list[torch.Tensor], stock_name: str, df: pd.DataFrame):
#     x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = data_tensors

#     model.eval()
#     y_train_pred = model(x_train_tensor)
#     y_test_pred = model(x_test_tensor)

#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     # invert predictions
#     y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
#     y_train = scaler.inverse_transform(y_train_tensor.detach().numpy())
#     y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
#     y_test = scaler.inverse_transform(y_test_tensor.detach().numpy())

#     figure, axes = plt.subplots(figsize=(15, 6))
#     axes.xaxis_date()

#     # print(type(np.array(df[len(df)-len(y_test):].index)))

#     axes.plot(np.array(df[len(df)-len(y_test):].index), y_test, color = 'red', label = 'Real Stock Price')
#     axes.plot(np.array(df[len(df)-len(y_test):].index), y_test_pred, color = 'blue', label = 'Predicted Stock Price')
#     #axes.xticks(np.arange(0,394,50))
#     plt.title('Stock Price Prediction')
#     plt.xlabel('Time')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.savefig(f'{stock_name}_pred.png')
#     plt.show()

# def train(
#     model,
#     model_path: str,
#     stock_data: pd.DataFrame,
#     stock_name: str,
#     result_store_dir: str,
#     # loss_fn = torch.nn.MSELoss,
#     optimizer_fn = torch.optim.Adam,
#     split_rate: float = 0.2,
#     learning_rate: float = 0.02,
#     look_back: int = 60,
# ) -> list[torch.Tensor]:
#     # load and preprocess data
#     x_train, y_train, x_test, y_test = LSTM.load_data(stock_data, split_rate, look_back)

#     # convert to tensor to allow back propagate
#     x_train_tensor = torch.from_numpy(x_train).type(torch.Tensor)
#     x_test_tensor = torch.from_numpy(x_test).type(torch.Tensor)
#     y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor)
#     y_test_tensor = torch.from_numpy(y_test).type(torch.Tensor)

#     # print(torch.isnan(x_test_tensor))
#     # model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=output_dim, num_layers=num_layers)
#     loss_fn = torch.nn.MSELoss()
#     optimizer = optimizer_fn(model.parameters(), lr=learning_rate)

#     num_epochs = 500
#     hist = np.zeros(num_epochs)
#     seq_dim =look_back-1

#     for e in range(num_epochs):
#         # print(x_train_tensor)
#         y_train_pred = model(x_train_tensor)

#         # print("predicted prices: ", y_train_pred)
#         loss = loss_fn(y_train_pred, y_train_tensor)

#         if e % 10 == 0 and e !=0:
#             print("Epoch ", e, "MSE: ", loss.item())

#         hist[e] = loss.item()
#         optimizer.zero_grad()

#         loss.backward()
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
#         optimizer.step()

#     print("Model's state_dict:")
#     for param_tensor in model.state_dict():
#         print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#     print("Optimizer's state_dict:")
#     for var_name in optimizer.state_dict():
#         print(var_name, "\t", optimizer.state_dict()[var_name])

#     # plt.plot(hist, label="Training loss")
#     # plt.legend()
#     # plt.savefig(f'{result_store_dir}/{stock_name}_training_loss.png')
#     # plt.show()

#     torch.save(model.state_dict(), model_path + stock_name)
#     print("Model has been saved to:", model_path + stock_name)
#     return [x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor]


# def predict(model, data_tensors: list[torch.Tensor], stock_name: str, df: pd.DataFrame):
#     """
#     predict data based on 
#     """
#     x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = data_tensors

#     with torch.no_grad():
#         y_train_pred = model(x_train_tensor)
#         y_test_pred = model(x_test_tensor)

#     # scaling features to efficiently learn from data
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     # invert predictions
#     y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
#     y_train = scaler.inverse_transform(y_train_tensor.detach().numpy())
#     y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
#     y_test = scaler.inverse_transform(y_test_tensor.detach().numpy())

#     figure, axes = plt.subplots(figsize=(15, 6))
#     axes.xaxis_date()
#     # print(type(np.array(df[len(df)-len(y_test):].index)))

#     axes.plot(np.array(df[len(df)-len(y_test):].index), y_test, color = 'red', label = 'Real Stock Price')
#     axes.plot(np.array(df[len(df)-len(y_test):].index), y_test_pred, color = 'blue', label = 'Predicted Stock Price')
    
#     # axes.xticks(np.arange(0,394,50))
#     plt.title('Stock Price Prediction')
#     plt.xlabel('Time')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.savefig(f'{stock_name}_pred.png')
#     plt.show()

#     return [y_train_pred, y_test_pred]


def main(data_dir: str):
    train_files, test_files = [], []
    for file in os.listdir(data_dir):
        if file[-1] != "y":
            if file[:5] == "Train":
                train_files.append(file)
            else:
                test_files.append(file)

    # print(train_files)

    dates = pd.date_range('2014-06-03','2020-12-31',freq='B')
    dates_test = pd.date_range('2021-01-04','2023-04-28',freq='B')
    train_dfs, train_scaler = LSTM.stocks_data(train_files, dates=dates, img_dir="./train_results")
    test_dfs, test_scaler = LSTM.stocks_data(test_files, dates=dates_test, img_dir="./train_results")

    for file_index in range(len(train_files)):
        file_name = train_files[file_index].split(".")[0][5:]
        print(file_name)

        train_df = train_dfs[file_index]
        test_df = test_dfs[file_index]
        
        split_rate = len(test_df) /(len(train_df) + len(test_df))
        # print(split_rate)
        # train_second_col_name = train_df.columns[1]
        # test_second_col_name = test_df.columns[1]
        # test_df = test_df.rename(columns={test_second_col_name: train_second_col_name})
        combined_df = pd.concat([train_df, test_df], axis=0)
        # print(combined_df.isna())
        # combined_df = combined_df.reset_index()

        # train_dfs = LSTM.load_data(stock_data=combined_df, split_rate=split_rate, look_back=60)


        model = LSTM(hidden_dim=32, input_dim=1, num_layers=2, out_dim=1)
        # data_tensors = model.train(
        #     model_path="./savedModel/", 
        #     stock_data=combined_df, 
        #     stock_name=file_name,
        #     split_rate=split_rate,
        #     result_store_dir="./train_result", 
        #     look_back=60,
        #     # optimizer_fn = torch.optim.SGD(model.parameters(), lr=0.02)
        # )

        x_train, y_train, x_test, y_test = LSTM.load_data(combined_df, split_rate, 30)

        x_train_tensor = torch.from_numpy(x_train).type(torch.Tensor)
        x_test_tensor = torch.from_numpy(x_test).type(torch.Tensor)
        y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor)
        y_test_tensor = torch.from_numpy(y_test).type(torch.Tensor)

        data_tensors = (x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)

        y_train_pred, y_test_pred = model.predict(
                            data_tensors=data_tensors, 
                            stock_name=file_name, 
                            df=combined_df, 
                            scaler=test_scaler,
                            saved_model_path=f"./savedModel/{file_name}"
                        )
        mae_loss = LSTM.eval_metric(y_test_pred, data_tensors[-1].detach().numpy())
        # print(f"MAE Loss: {mae_loss}")

        



if __name__ == "__main__":
    main("./data")