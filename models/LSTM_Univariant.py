import random

import numpy as np
import pandas as pd
from pylab import mpl, plt

plt.style.use('seaborn')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from models.Sequence import SequenceDataset




class LSTM(nn.Module):
  
  def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, use_cuda=False, dropout=0.2):
    '''
      n_features: number of input features (1 for univariate forecasting)
      n_hidden: number of neurons in each hidden layer
      n_outputs: number of outputs to predict for each training example
      n_deep_layers: number of hidden dense layers after the lstm layer
      sequence_len: number of steps to look back at for prediction
      dropout: float (0 < dropout < 1) dropout ratio between dense layers
    '''
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.n_lstm_layers = n_lstm_layers
    self.nhid = n_hidden
    self.use_cuda = use_cuda # set option for device selection
    self.sequence_len = sequence_len
    # LSTM Layer
    self.lstm = nn.LSTM(n_features,
                        n_hidden,
                        num_layers=n_lstm_layers,
                        batch_first=True) # As we have transformed our data in this way
    
    # first dense after lstm
    self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden) 
    # self.fc1 = nn.Linear(n_hidden, n_hidden)
    # Dropout layer 
    self.dropout = nn.Dropout(p=dropout)

    # Create fully connected layers (n_hidden x n_deep_layers)
    dnn_layers = []
    for i in range(n_deep_layers):
      # Last layer (n_hidden x n_outputs)
      if i == n_deep_layers - 1:
        dnn_layers.append(nn.ReLU())
        dnn_layers.append(nn.Linear(self.nhid, n_outputs))
      # All other layers (n_hidden x n_hidden) with dropout option
      else:
        dnn_layers.append(nn.ReLU())
        dnn_layers.append(nn.Linear(self.nhid, self.nhid))
        if dropout:
          dnn_layers.append(nn.Dropout(p=dropout))

    self.dnn = nn.Sequential(*dnn_layers)

  def forward(self, x):
    # Initialize hidden state
    hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
    cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)

    # move hidden state to device
    if self.use_cuda:
      hidden_state = hidden_state.to(self.device)
      cell_state = cell_state.to(self.device)
        
    self.hidden = (hidden_state, cell_state)
    # self.hidden = (torch.zeros(self.num_layers, 1, self.hidden_layer_size),
    #            torch.zeros(self.num_layers, 1, self.hidden_layer_size))

    # Forward Pass
    x, h = self.lstm(x, self.hidden) # LSTM
    x = self.dropout(x.contiguous().view(x.shape[0], -1)) # Flatten lstm out 
    x = self.fc1(x) # First Dense
    return self.dnn(x) # Pass forward through fully connected DNN
  
  def generate_sequences(self, df: pd.DataFrame, look_back: int, look_forward: int, target_columns, drop_targets=False):
    '''
      df: Pandas DataFrame of the univariate time-series
      tw: Training Window - Integer defining how many steps to look back
      pw: Prediction Window - Integer defining how many steps forward to predict

      returns: dictionary of sequences and targets for all sequences
    '''
    data = dict() # Store results into a dictionary
    L = len(df)
    for i in range(L-look_back):
      # Option to drop target from dataframe
      if drop_targets:
        df.drop(target_columns, axis=1, inplace=True)

      # Get current sequence  
      sequence = df[i:i+look_back].values
      # Get values right after the current sequence
      target = df[i+look_back:i+look_back+look_forward][target_columns].values
      data[i] = {'sequence': sequence, 'target': target}
    return data
  
  def normalize(self, df: pd.DataFrame):
    scalers = {}

    # for x in df.columns:
    scalers["close"] = StandardScaler().fit(df["close"].values.reshape(-1, 1))
  
    # print("111")

    # Transform data via scalers
    norm_df = df.copy()
    # print("there", norm_df.iloc[:, 0].values)
    # for i, key in enumerate(scalers.keys()):
    #   print(norm_df.iloc[:, i].values)
    norm = scalers["close"].transform(norm_df["close"].values.reshape(-1, 1))
    norm_df["close"] = norm

    return norm_df, scalers["close"]
  
  def load_data(self, df: pd.DataFrame, sequence_len: int, nout: int, isShuffle: bool, BATCH_SIZE: int = 16, split: float = 0.8, train: bool = True):

    """
      df:           train data frame
      sequence_len: number of dates for model to look back
      nout:         output dimension (same as nfeatures)
    """
    norm_df, scalar = self.normalize(df)
    sequences = self.generate_sequences(norm_df.close.to_frame(), sequence_len, nout, 'close')
    dataset = SequenceDataset(sequences)

    # Split the data according to our split ratio and load each subset into a
    # separate DataLoader
    train_len = int(len(dataset)*split)
    lens = [train_len, len(dataset)-train_len]
    train_ds, test_ds = random_split(dataset, lens)
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=isShuffle, drop_last=True)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=isShuffle, drop_last=True)
    if train:
      return trainloader, testloader
    return scalar