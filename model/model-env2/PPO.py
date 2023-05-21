import numpy as np
import sys
from copy import copy
import math
import pandas as pd
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import stable_baselines3.common.env_checker as check
import gym
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append('..')
import stock_env

DAY = 252
RF = 0.03/DAY

def train(times=1,timesteps=[9000],n_steps=[128,1024]):
    train_df = pd.read_csv("../direct/train.csv")
    test_df = pd.read_csv("../direct/test.csv")
    for time_step in timesteps:
        for n_step in n_steps:
            for bs in[32,64]:
                for ep in range(times):
                    train_env = gym.make('stock_env/StockTradingEnv-v2', df=train_df)
                    check.check_env(train_env)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    model = PPO("MlpPolicy", train_env, verbose=0, device=device)
                    model.learn(total_timesteps=time_step)

                    test_env = gym.make('stock_env/StockTradingEnv-v2', df=test_df)
                    check.check_env(test_env)
                    obs = test_env.reset()

                    rates = []
                    actions = []
                    daily_return = []
                    positions = []
                    info = {'net_worth': 500000}
                    for i in range(1000):
                        pre_share_hold = copy(test_env.share_hold)
                        pre_net_worth = info['net_worth']
                        action, _ = model.predict(obs)
                        obs, reward, done, info = test_env.step(action)
                        for i in range(len(pre_share_hold)):
                            pre_share_hold[i] = test_env.share_hold[i] - pre_share_hold[i]
                        actions.append(pre_share_hold)
                        rates.append(info['net_worth'] / 500000)
                        daily_return.append((info['net_worth'] - pre_net_worth) / pre_net_worth)
                        positions.append(1-info['balance']/info['net_worth'])
                        if done:
                            break
                    # print(info['net_worth'], info['shares'],info['trans_reward'])
                    interest = info['net_worth'] / 500000 - 1
                    print(n_step,bs,ep)
                    s_int = str(int(interest * 1000) / 1000)
                    daily_return = np.array(daily_return)
                    sharpe = (daily_return.mean() - RF) / daily_return.std() * math.sqrt(DAY)
                    print(interest,sharpe)
                    print()

                    if interest > 0.7 and sharpe > 0.75:
                        model.save("saved_models/env2-PPO" + s_int)
                        plt.plot(list(range(len(rates))), rates,label='net worth')
                        plt.plot(positions,label='position')
                        plt.legend()
                        plt.xlabel('days')
                        plt.savefig("saved_models/" + s_int + '.png',dpi=200)
                    del model


if __name__ == '__main__':
    train()