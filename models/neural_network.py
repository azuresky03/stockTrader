import torch 
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataset import StockDataset
from sklearn.preprocessing import MinMaxScaler


class NeuralNets(nn.Module):
    def __init__(self, n_features: int, n_out: int, n_layers: int, n_dims: list[int]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers = []

        self.input_layer = nn.Linear(n_features, n_dims[0], dtype=float)
        self.output_layer = nn.Linear(n_dims[-1], n_out, dtype=float)
        assert(len(n_dims) == n_layers)
        for i in range(n_layers - 1):
            input_dim = n_dims[i]
            output_dim = n_dims[i + 1]
            layers.append(nn.Linear(input_dim, output_dim, dtype=float))
            layers.append(nn.ReLU())

        self.dnn = nn.Sequential(*layers)

    
    def forward(self, x):
        out = self.input_layer(x)
        out = self.dnn(out)
        out = self.output_layer(out)
        return out
    

def normalize(df: pd.DataFrame, reference_date: pd.Timestamp):
    # normalize all columnns
    scalars = {}
    prices = np.array(df["close"].values)
    changes = prices[1:] - prices[:-1]

    # remove last row since we don't know the price of the date after
    df = df.iloc[:-1, :]
    df.loc[:, "date"] = [
        delta_time.days for delta_time in ( np.array([pd.to_datetime(date) for date in df["date"].values]) - np.array([reference_date for _ in range(len(df["date"].values))]) ) 
    ]
    df.loc[:, "close"] = changes.flatten()


    norm_df = df.copy()
    for col in df.columns:
        scalars[col] = MinMaxScaler().fit(df[col].values.reshape(-1, 1))
        normalized_val = scalars[col].transform(norm_df[col].values.reshape(-1, 1))
        norm_df[col] = normalized_val

    return norm_df, scalars


def load_data(df: pd.DataFrame, reference_date: pd.Timestamp):
    df = df.drop(labels=["code", "turn", "adjustflag", "isST", "tradestatus"], axis=1)
    df = df.fillna(method='ffill')
    # df["date"] = [ delta_time.days for delta_time in ( np.array([pd.to_datetime(date) for date in df["date"].values]) - np.array([reference_date for _ in range(len(df["date"].values))]) ) ]
    # prices = np.array(df["close"].values)
    # changes = prices[1:] - prices[:-1]
    # df = df.iloc[:-1, :]

    norm_df, scalars = normalize(df, reference_date)
    # scalar = MinMaxScaler()
    # prices_normalized = scalar.fit_transform(changes.reshape(-1, 1))
    # df["close"] = prices_normalized
    # print("normalized shape: ", prices_normalized.shape)
    
    # print(norm_df)
    dataset = StockDataset(norm_df.drop(labels=["close"], axis=1), np.array(norm_df["close"].values.reshape(-1, 1)))
    dataLoader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, drop_last=True)

    return dataLoader, scalars
    

def train(model: NeuralNets, dataloader: DataLoader, n_epoch: int, save_model_path: str, scalars: dict):
    criterion = nn.MSELoss().to(model.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    for e in range(n_epoch):
        loss_ = 0.0
        for x, y in dataloader:
            # print("x shape: ", x.shape)
            # print("y shape: ", y.shape)

            x = x.to(model.device).double()
            y = y.to(model.device).double()
            
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)

            inversed_y = scalars['close'].inverse_transform(y.detach().numpy().reshape(-1, 1))
            inversed_pred = scalars['close'].inverse_transform(output.detach().numpy().reshape(-1, 1))

            real_loss = np.sum( np.abs(inversed_pred.flatten() - inversed_y.flatten()) )

            loss.backward()
            optimizer.step()
            loss_ += loss.item()
        print(f'epoch {e+1} --> loss: {loss_}, real loss: {real_loss} ')
    
    torch.save(model.state_dict(), save_model_path)
    print("Model is saved to:", save_model_path)


def predict(model: NeuralNets, dataloader: DataLoader, scalar: MinMaxScaler, save_path: str):
    preds = []
    dates = pd.date_range(start="2014-06-03", periods=(len(dataloader) * 32), freq="B")

    model.eval()
    i = 0
    with torch.no_grad():
        for x, y in dataloader:
            i += 1
            print(f"batch {i}")
            x = x.to(model.device).double()
            pred = model(x)
            inversed_pred = scalar.inverse_transform(pred.detach().numpy().reshape(-1, 1))
            preds.append(inversed_pred.flatten())
    
    pred_df = pd.DataFrame(index=dates)
    pred_df["pred_close"] = np.array(preds).flatten()
    pred_df.to_csv(save_path)
    return
    


if __name__ == "__main__":
    reference_date = pd.to_datetime("2014-06-03")

    df_train = pd.read_csv("./data/Train万华化学.csv", sep=",")
    df_test = pd.read_csv("./data/Test万华化学.csv", sep=",")

    n_features = 32

    combined_df = pd.concat([df_train, df_test], axis=0)
    model = NeuralNets(n_features=8, n_out=1, n_layers=15, n_dims=[256 for _ in range(15)])
    dataloader, scalar = load_data(combined_df, reference_date)
    train(model=model, dataloader=dataloader, n_epoch=100, scalars=scalar, save_model_path="./savedModel/万华化学.pt")

    # loaded_model = NeuralNets(n_features=32, n_out=32, n_layers=5, n_dims=[128 for _ in range(5)])
    # loaded_model.load_state_dict(torch.load('./savedModel/万华化学.pt'))
    # dates = [date for date in combined_df["date"].index]
    # dataloader, scalar = NeuralNets.load_data(combined_df)

    # # TODO: we are missing 8 
    # predict(model=loaded_model, dataloader=dataloader, scalar=scalar, save_path="./approximation/万华化学.csv")
