import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gym import Env
from gym.spaces import Box, Discrete
from pydantic import BaseModel
from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import matplotlib
import matplotlib.pyplot as plt
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
# from finrl.meta.preprocessing.data import data_split


class StockTradingEnv(gym.Env):
    def __init__(
            self, 
            df: pd.DataFrame,
            stock_dim: int,
            hmax: int,
            initial_amount: int,
            num_stock_shares: list[int],
            buy_cost_pct: list[float],
            sell_cost_pct: list[float],
            reward_scaling: float,
            state_space: int,
            action_space: int,
            tech_indicator_list: list[str],
            turbulence_threshold=None,
            risk_indicator_col="turbulence",
            make_plots: bool = False,
            print_verbosity=10,
            day=0,
            initial=True,
            previous_state=[],
            model_name="",
            mode="",
            iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _modify_stocks(self, indexs: np.ndarray, amounts: np.ndarray):
        # amount: < 0, sell; = 0, hold; > 0, buy
        
        return 
