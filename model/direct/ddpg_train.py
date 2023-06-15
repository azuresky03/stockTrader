import math

from stable_baselines3.common.noise import NormalActionNoise

TRAIN = True
model_name = "DDPG1.165"
method = "DDPG"
OUT = False
DAY = 252
RF = 0.03/DAY

import sys
from copy import copy

import gym
sys.path.append('..')
import stock_env
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
import stable_baselines3.common.env_checker as check

test_df = pd.read_csv("test.csv")

if TRAIN:
    train_df = pd.read_csv("train.csv")
    train_env = gym.make('stock_env/StockTradingEnv-v1',df=train_df)
    check.check_env(train_env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

for bs in [5,10,30,50,75,100,150]:
    for ep in range(2):
        if TRAIN:
            n_actions = train_env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            model = DDPG("MlpPolicy", train_env, action_noise=None, verbose=1,batch_size=bs,learning_rate=0.0006)
            model.learn(total_timesteps=8000, log_interval=10)
        else:
            model = DDPG.load("saved_models/"+model_name)

        test_env = gym.make('stock_env/StockTradingEnv-v1',df=test_df)
        check.check_env(test_env)



        total_reward = 0
        obs = test_env.reset()

        rates = []
        actions = []
        daily_return = []
        info = {'net_worth':500000}
        for i in range(1000):
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
        print(f"batch_size={bs} ep={ep}")
        print(info['net_worth'],info['shares'])
        interest = total_reward/500000
        s_int = str(int(interest*1000)/1000)
        daily_return = np.array(daily_return)
        sharpe = (daily_return.mean()-RF)/daily_return.std()*math.sqrt(DAY)
        print("interest rate:",interest,"sharpe ratio:",sharpe)


        if interest > 1.5 and sharpe>1.2 and TRAIN:
            model.save("saved_models/DDPG"+s_int)
            plt.plot(list(range(len(rates))),rates)
            plt.xlabel('days')
            plt.ylabel('net worth')
            plt.savefig("saved_models/DDPG"+s_int+'.png')
        elif not TRAIN and OUT:
            act_df = pd.DataFrame(actions,columns=test_df.iloc[:26]['tic'])
            action_dir = "action_of_models/"
            act_df.to_csv(action_dir+method+s_int+"actions.csv",index=False)
        del model