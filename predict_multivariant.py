import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.LSTM_multivariant import LSTM
from train import process_single_stock

train_file = "Train万华化学"
test_file = "Test万华化学"


process_single_stock(num_epoch=100, num_pred=60, train_file=train_file, test_file=test_file, isTrain=False)