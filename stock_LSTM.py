import random

import numpy as np
import pandas as pd
from pylab import mpl, plt

plt.style.use('seaborn')
import datetime
import itertools
# mpl.rcParams['font.family'] = 'serif'
# from pandas import datetime
import math
import time
from math import sqrt
from operator import itemgetter

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, num_layers: int, out_dim: int):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, out_dim)

        # TODO: temporal information to capture?
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # added layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # added layer
        self.fc3 = nn.Linear(hidden_dim, out_dim)     # output layer


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        out = self.fc3(out)

        return out
    
# TODO: add train and predict into model itself





if __name__ == "__main__":
    pass