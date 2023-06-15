import numpy as np

TRAIN = True
OUT = False
model_name = "example2.25"
DAY = 252
RF = 0.03/DAY

import sys
from copy import copy
import math
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
for n_step in [50]:
    for time_step in [9000]:
        for ep in range(3):
            if TRAIN:
                model = A2C("MlpPolicy", train_env, verbose=0,device=device,n_steps=n_step)
                model.learn(total_timesteps=time_step)
            else:
                model = A2C.load(model_name)

            test_env = gym.make('stock_env/StockTradingEnv-v1',df=test_df)
            check.check_env(test_env)
            total_reward = 0
            obs = test_env.reset()


            rates = []
            actions = []
            daily_return = []
            info = {'net_worth':500000}
            for i in range(1200):
                pre_share_hold = copy(test_env.share_hold)
                pre_net_worth = info['net_worth']
                action, _ = model.predict(obs)
                obs, reward, done , info = test_env.step(action)
                for i in range(len(pre_share_hold)):
                    pre_share_hold[i] = test_env.share_hold[i] - pre_share_hold[i]
                actions.append(pre_share_hold)
                total_reward += reward
                rates.append(info['net_worth']/500000)
                daily_return.append((info['net_worth']-pre_net_worth)/pre_net_worth)
                if done:
                    break
            print(time_step,n_step,ep)
            print(info['net_worth'],info['shares'])
            interest = total_reward/500000
            print(interest)
            s_int = str(int(interest*1000)/1000)
            daily_return = np.array(daily_return)
            sharpe = (daily_return.mean()-RF)/daily_return.std()*math.sqrt(DAY)
            print(sharpe)

            if interest > 0.8 and sharpe>1.1 and TRAIN:
                model.save("saved_models/A2C"+s_int)
                plt.plot(list(range(len(rates))),rates)
                plt.xlabel('days')
                plt.ylabel('net worth')
                plt.savefig("saved_models/"+s_int+'.png')
            elif not TRAIN and OUT:
                act_df = pd.DataFrame(actions,columns=test_df.iloc[:26]['tic'])
                action_dir = "action_of_models/"
                act_df.to_csv(action_dir+s_int+"actions.csv",index=False)
            del model