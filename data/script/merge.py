import os

import pandas as pd

file_list = os.listdir('../processed_data')
test_list = []
train_list = []
for file_name in file_list:
    if file_name[:4]=="Test":
        di = test_list
    else:
        di = train_list
    di.append(pd.read_csv("../processed_data/"+file_name))
print(test_list)