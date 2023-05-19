import pandas as pd
import numpy as np

path = "../../FinRL_Example/datasets/processed_data.csv"
df = pd.read_csv(path)

df1 = df.iloc[260:38064,:]
df2 = df.iloc[38064:,:]

df1.to_csv("train.csv",index=False)
df2.to_csv("test.csv",index=False)

print(df1.iloc[260]['rsi'])
STOCK_NUM = 26
INI_ACCOUNT_BALANCE = 500000
FEATURES_LIST = ['close','volume','high','open','low','macd','adx','rsi']
FEATURES_NUM = len(FEATURES_LIST)
cur_price = []
trading = []
other_features = [0] * (FEATURES_NUM - 2) * STOCK_NUM
day = 26*10
for i in range(STOCK_NUM):
    cur_price.append(df.iloc[day + i]['close'])
    trading.append(df.iloc[day + i]['volume'] != 0)
    for j in range(FEATURES_NUM-2):
        other_features[j * STOCK_NUM + i] = df.iloc[day + i][FEATURES_LIST[j + 2]]
arr = np.concatenate([ cur_price, trading, other_features])
print(cur_price)
print(trading)
print(other_features[:26])
print(other_features[26:26*2])
print(other_features[-27:])