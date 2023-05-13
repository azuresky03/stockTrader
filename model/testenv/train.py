import sys
import gym
sys.path.append('..')
import stock_env
import pandas as pd

from stable_baselines3 import A2C
import stable_baselines3.common.env_checker as check

df = pd.read_csv("Test比亚迪.csv")


env = gym.make('stock_env/StockTradingEnv-v0',df=df)

# env.reset()
# print(env.step([0.5]))
# print(env.step([-0.5]))
# print(env.reset())

check.check_env(env)
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)