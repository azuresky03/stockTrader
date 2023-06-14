from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class StockDataset(Dataset):
    def __init__(self, x: pd.DataFrame, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # print(self.x.iloc[index].values)
        return torch.Tensor(self.x.iloc[index].values), torch.Tensor(self.y[index])
    
    def __len__(self):
        # return len(self.x)
        return self.x.shape[0]
    