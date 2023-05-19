import gym
from gym import spaces
import numpy as np
import pandas as pd

STOCK_NUM = 1
INI_ACCOUNT_BALANCE = 100000
FEATURES_NUM = 7
MIN_TRANS_NUM = 100
SELL_FEES = 0.001
STOP_ACCOUNT_BALANCE = 0.3*INI_ACCOUNT_BALANCE

class StockTradingEnv(gym.Env):
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
        self.net_worth = INI_ACCOUNT_BALANCE

        #Action space
        self.action_space = spaces.Box(low=-1,high=1,shape=(STOCK_NUM,))

        #State space
        # shape = current balance 1 + owned shares 30 + other values(first 30 prices) *30
        # low ? 0 / negative infinity
        self.observation_space = spaces.Box(low=0,high=np.inf,shape=(1+STOCK_NUM+FEATURES_NUM*STOCK_NUM,),dtype=np.float64)

        #initalize state
        self.state = self._get_obs()

    def _get_obs(self):
        """
        ?? read prices from pandas dataframe and convert into np.array
        :return: np.array
        """

        arr = np.concatenate([[self.account_balance],self.share_hold,self.df.iloc[self.day].to_numpy()])
        return arr

    def _get_info(self):
        """
        get information of current status
        :return: dict
        """
        d = { "day":self.day, "balance":self.account_balance, "shares":self.share_hold, "net_worth":self.net_worth}
        return d

    def _take_action(self,action):
        assert len(action)==STOCK_NUM
        buy_stock_total = 0
        #sell stocks
        for i in range(STOCK_NUM):
            if action[i] < 0 and self.share_hold[i]>0:
                sell_amount = -int(action[i]*self.share_hold[i])
                if sell_amount:
                    self.share_hold[i] -= sell_amount
                    self.account_balance += MIN_TRANS_NUM * (1-SELL_FEES) * sell_amount * self.state[1+STOCK_NUM+i]
            elif action[i] > 0:
                buy_stock_total += 1
        #buy stocks proportionally
        for i in range(STOCK_NUM):
            if action[i] > 0 and self.account_balance > 0:
                buy_amount = int (self.account_balance * action[i] / buy_stock_total / self.state[1+STOCK_NUM+i] / MIN_TRANS_NUM)
                if buy_amount > 0:
                    self.share_hold[i] += buy_amount
                    self.account_balance -= buy_amount * MIN_TRANS_NUM * self.state[1+STOCK_NUM+i]
        #compute net worth
        new_net_worth = self.account_balance
        for i in range(STOCK_NUM):
            if self.share_hold[i] > 0:
                new_net_worth += self.share_hold[i] * self.state[1+STOCK_NUM+i] * MIN_TRANS_NUM
        self.net_worth = new_net_worth

    def step(self,action):
        pre_net_worth = self.net_worth
        done = False

        self._take_action(action)
        d = self._get_info()

        reward = self.net_worth - pre_net_worth

        if self.net_worth < STOP_ACCOUNT_BALANCE:
            done = True
            d["stop_string"] = "lose too much"
        elif self.day >= self.df.shape[0]-2:
            done = True
            d["stop_string"] = "Reach the end day"

        if not done:
            self.day += 1
            next_obs = self._get_obs()
            self.state = next_obs
        else:
            next_obs = self.state

        return next_obs,reward,done,d

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
        self.net_worth = INI_ACCOUNT_BALANCE
        self.state = self._get_obs()
        return self.state

    def updateDF(self,new_df,begin_day=None):
        self.df = new_df
        if begin_day:
            self.day = begin_day