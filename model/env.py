import gym
from gym import spaces
import numpy as np
import pandas as pd

STOCK_NUM = 30
INI_ACCOUNT_BALANCE = 100000
FEATURES_NUM = 8

class StocktradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,df):
        """
        :param df: pandas.dataframe
        row: days
        col: price 1-stock_num + indicatorA 1-stock_num + indicatorB 1-stock_num
        """
        #=number of steps
        self.day = 0
        self.df = df
        self.account_balance = INI_ACCOUNT_BALANCE
        self.share_hold = [0]*STOCK_NUM

        #Action space
        self.action_space = spaces.Box(low=-1,high=1,shape=(STOCK_NUM,))

        #State space
        # shape = current balance 1 + owned shares 30 + other values(first 30 prices) *30
        # low ? 0 / negative infinity
        self.observation_space = spaces.Box(low=np.NINF,high=np.inf,shape=(1+STOCK_NUM+FEATURES_NUM*STOCK_NUM,))

        #initalize state
        self.state = self._get_obs()

    def _get_obs(self):
        """
        ?? read prices from pandas dataframe and convert into np.array
        :return: np.array
        """

        arr = np.concatenate([[self.account_balance],self.share_hold,self.df.iloc[[self.day]].to_numpy()])
        return arr

    def _get_info(self):
        """
        get information of current status
        :return: string
        """
        s = f'day={self.day} balance={self.account_balance} shares={self.share_hold}'
        s += f'prices={self.state[STOCK_NUM+1:2*STOCK_NUM+1]}'
        return s

    def reset(self,new_df=None):
        """
        reset the environment
        :param new_df: pandas.df
        :return:np.array,string
        """
        if new_df:
            self.df = new_df

        self.day = 0
        self.account_balance = INI_ACCOUNT_BALANCE
        self.share_hold = [0] * STOCK_NUM
        self.state = self._get_obs()
        return self.state,self._get_info()

