import sys
import gym
sys.path.append('..')
import stock_env
import pandas as pd
import torch

from stable_baselines3 import A2C
import stable_baselines3.common.env_checker as check

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

train_env = gym.make('stock_env/StockTradingEnv-v1',df=train_df)
check.check_env(train_env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = A2C("MlpPolicy", train_env, verbose=1,device=device)
model.learn(total_timesteps=5000)

test_env = gym.make('stock_env/StockTradingEnv-v1',df=test_df)
check.check_env(test_env)
total_reward = 0
obs = test_env.reset()
for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done , info = test_env.step(action)
    total_reward += reward
    if done:
        break
print(info['net_worth'],info['shares'])
print(total_reward/500000)