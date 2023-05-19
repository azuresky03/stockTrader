TRAIN = False
model_name = "example2.25"

import sys
from copy import copy

import gym
sys.path.append('..')
import stock_env
import pandas as pd
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import A2C
import stable_baselines3.common.env_checker as check

test_df = pd.read_csv("test.csv")

if TRAIN:
    train_df = pd.read_csv("train.csv")
    train_env = gym.make('stock_env/StockTradingEnv-v1',df=train_df)
    check.check_env(train_env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if TRAIN:
    model = A2C("MlpPolicy", train_env, verbose=1,device=device,n_steps=35)
    model.learn(total_timesteps=6000)
else:
    model = A2C.load(model_name)

test_env = gym.make('stock_env/StockTradingEnv-v1',df=test_df)
check.check_env(test_env)
total_reward = 0
obs = test_env.reset()


rates = []
actions = []
for i in range(1000):
    pre_share_hold = copy(test_env.share_hold)
    action, _ = model.predict(obs)
    obs, reward, done , info = test_env.step(action)
    for i in range(len(pre_share_hold)):
        pre_share_hold[i] = test_env.share_hold[i] - pre_share_hold[i]
    actions.append(pre_share_hold)
    total_reward += reward
    rates.append(info['net_worth']/500000)
    if done:
        break
print(info['net_worth'],info['shares'])
interest = total_reward/500000
print(interest)
s_int = str(int(interest*1000)/1000)

if interest > 1 and TRAIN:
    model.save("example"+s_int)
    plt.plot(list(range(len(rates))),rates)
    plt.xlabel('days')
    plt.ylabel('account balance')
    plt.savefig(s_int+'.png')
else:
    act_df = pd.DataFrame(actions,columns=test_df.iloc[:26]['tic'])
    action_dir = "action_of_models/"
    act_df.to_csv(action_dir+s_int+"actions.csv",index=False)