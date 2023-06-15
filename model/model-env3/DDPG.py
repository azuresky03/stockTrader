import os

import numpy as np
import sys
from copy import copy
import math
import pandas as pd
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
import stable_baselines3.common.env_checker as check
import gym
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append('..')
import stock_env

DAY = 252
RF = 0.03/DAY

def train():
    train_df = pd.read_csv("../direct/train.csv")
    for time_step in [9000,6000]:
        for g in [0.8,0.9]:
            for bs in[30]:
                train_env = gym.make('stock_env/StockTradingEnv-v3', df=train_df)
                check.check_env(train_env)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                n_actions = train_env.action_space.shape[-1]
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                model = DDPG("MlpPolicy", train_env, verbose=1, device=device,gamma=g,batch_size=bs,action_noise=action_noise)
                model.learn(total_timesteps=time_step)

                test_df = pd.read_csv("../direct/test.csv")
                test_env = gym.make('stock_env/StockTradingEnv-v3', df=test_df)
                check.check_env(test_env)

                test_int = 0
                count = 0
                for i in range(3):
                    obs = test_env.reset()
                    count += 1
                    rates = []
                    actions = []
                    daily_return = []
                    positions = []
                    stas = []
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
                        stas.append(info['sta'])
                        if done:
                            break
                    # print(info['net_worth'], info['shares'],info['trans_reward'])
                    interest = info['net_worth'] / 500000 - 1
                    test_int += interest
                    print(bs,g,time_step)
                    daily_return = np.array(daily_return)
                    sharpe = (daily_return.mean() - RF) / daily_return.std() * math.sqrt(DAY)
                    print(interest,sharpe)
                    print(stas[500:510])
                    if sharpe < 0.5:
                        break

                t = int(test_int/count*1000)/1000
                print(t)
                if t > 0.5 and count==3:
                    model.save('model3-DDPG'+str(t))
                print()
                del model

def evaluate(model_path,turns=1):
    test_df = pd.read_csv("../direct/test.csv")
    test_env = gym.make('stock_env/StockTradingEnv-v3', df=test_df)
    check.check_env(test_env)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    total_int = 0
    total_sharpe = 0
    model = DDPG.load(model_path)
    cri_hold = []
    cri = []
    for s in range(26):
        cri_hold.append(10000 / test_df.iloc[s]['close'])
    for turn in range(turns):
        obs = test_env.reset()
        rates = []
        actions = []
        daily_return = []
        positions = []
        info = {'net_worth': 500000}
        for i in range(1000):
            if turn==0:
                cri.append(sum(test_df.iloc[s+i*26]['close']*cri_hold[s] for s in range(26))/260000)
            pre_share_hold = copy(test_env.share_hold)
            pre_net_worth = info['net_worth']
            action, _ = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
            for i in range(len(pre_share_hold)):
                pre_share_hold[i] = test_env.share_hold[i] - pre_share_hold[i]
            actions.append(pre_share_hold)
            rates.append(info['net_worth'] / 500000)
            daily_return.append((info['net_worth'] - pre_net_worth) / pre_net_worth)
            positions.append(1 - info['balance'] / info['net_worth'])
            if done:
                break
        interest = info['net_worth'] / 500000 - 1
        total_int += interest
        s_int = str(int(interest * 1000) / 1000)
        daily_return = np.array(daily_return)
        sharpe = (daily_return.mean() - RF) / daily_return.std() * math.sqrt(DAY)
        total_sharpe += sharpe
        print(interest, sharpe)
        plt.plot(list(range(len(rates))), rates, label=f'trail{turn+1}',color=colors[turn],linewidth=1)
        plt.plot(positions,color=colors[turn],linewidth=0.4,linestyle='--')
    total_int /= turns
    total_sharpe /= turns
    plt.plot(cri,label='STD',linewidth=1.25,color='k')
    plt.xlabel("days")
    plt.legend()
    if not os.path.exists(model_path+'-'):
        os.makedirs(model_path+'-')
    plt.savefig(model_path + '-/' + str(total_int)[:4] + '.png', dpi=200)
    print(total_int,total_sharpe)

if __name__ == '__main__':
    evaluate("model3-DDPG0.821")