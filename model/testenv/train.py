import sys
import gym
sys.path.append('..')
import stock_env
import pandas as pd

from stable_baselines3 import PPO
import stable_baselines3.common.env_checker as check

test_df = pd.read_csv("Test比亚迪.csv")
train_df = pd.read_csv("Train比亚迪.csv")

env = gym.make('stock_env/StockTradingEnv-v0',df=train_df)

# env.reset()
# print(env.step([0.5]))
# print(env.step([-0.5]))
# print(env.reset())

check.check_env(env)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

env = gym.make('stock_env/StockTradingEnv-v0',df=test_df)
total_reward = 0
obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done , info = env.step(action)
    total_reward += reward
    if done:
        break

print(info,total_reward)