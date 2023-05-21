import gym
from gym import spaces
import numpy as np
import pandas as pd

STOCK_NUM = 26
INI_ACCOUNT_BALANCE = 500000
FEATURES_LIST = ['close','volume','high','open','low','macd','adx','rsi']
FEATURES_NUM = len(FEATURES_LIST)
MIN_TRANS_NUM = 100
SELL_FEES = 0.0025
BUY_FEES = 0.002
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
        self.cur_price = [0]*STOCK_NUM
        self.trading = [True]*STOCK_NUM
        self.holding_price = [0]*STOCK_NUM
        self.trans_reward = 0
        self.net_worth = INI_ACCOUNT_BALANCE

        #Action space
        self.action_space = spaces.Box(low=-1,high=1,shape=(STOCK_NUM,))

        #State space
        # shape = current balance 1 + owned shares 30 + other values(first 30 prices) *30
        # low ? 0 / negative infinity
        self.observation_space = spaces.Box(low=np.NINF,high=np.inf,shape=(1+2*STOCK_NUM+FEATURES_NUM*STOCK_NUM,),dtype=np.float64)

        #initalize state
        self.state = self._get_obs()

    def _get_obs(self):
        """
        ?? read prices from pandas dataframe and convert into np.array
        :return: np.array
        """
        cur_price = []
        trading = []
        other_features = [0]*(FEATURES_NUM-2)*STOCK_NUM
        for i in range(STOCK_NUM):
            cur_price.append(self.df.iloc[self.day+i]['close'])
            trading.append(self.df.iloc[self.day+i]['volume']!=0)
            for j in range(FEATURES_NUM-2):
                other_features[j*STOCK_NUM+i] = self.df.iloc[self.day+i][FEATURES_LIST[j+2]]
        self.cur_price = cur_price
        self.trading = trading
        arr = np.concatenate([[self.account_balance],self.share_hold,self.cur_price,self.holding_price,self.trading,other_features])
        return arr

    def _get_info(self):
        """
        get information of current status
        :return: dict
        """
        d = { "day":self.day/STOCK_NUM, "balance":self.account_balance, "shares":self.share_hold, "net_worth":self.net_worth,"cur_price":self.cur_price}
        return d

    def _take_action(self,action):
        buy_stock_total = 0
        #sell stocks
        for i in range(STOCK_NUM):
            if self.trading[i] and action[i] < 0 and self.share_hold[i]>0:
                sell_amount = -int(action[i]*self.share_hold[i])
                if sell_amount:
                    self.share_hold[i] -= sell_amount
                    self.account_balance += MIN_TRANS_NUM * (1-SELL_FEES) * sell_amount * self.cur_price[i]
                    self.trans_reward += sell_amount * MIN_TRANS_NUM * (self.cur_price[i]-self.holding_price[i])
            elif self.trading[i] and action[i] > 0:
                buy_stock_total += 1
        #buy stocks proportionally
        for i in range(STOCK_NUM):
            if self.trading[i] and action[i] > 0 and self.account_balance > 0:
                buy_amount = int (self.account_balance * action[i] / buy_stock_total / self.cur_price[i] / MIN_TRANS_NUM)
                if buy_amount > 0:
                    self.holding_price[i] = (self.holding_price[i] * self.share_hold[i] + self.cur_price[i]*buy_amount)/(self.share_hold[i]+buy_amount)
                    self.share_hold[i] += buy_amount
                    self.account_balance -= buy_amount * MIN_TRANS_NUM * self.state[1+STOCK_NUM+i] * (1+BUY_FEES)
        #compute net worth
        new_net_worth = self.account_balance
        for i in range(STOCK_NUM):
            if self.share_hold[i] > 0:
                new_net_worth += self.share_hold[i] * self.cur_price[i] * MIN_TRANS_NUM
        self.net_worth = new_net_worth

    def step(self,action):
        pre_net_worth = self.net_worth
        done = False

        self._take_action(action)
        d = self._get_info()

        reward = (self.net_worth - pre_net_worth)*0.7 + 0.5*self.trans_reward
        d['trans_reward'] = self.trans_reward
        self.trans_reward = 0

        if self.net_worth < STOP_ACCOUNT_BALANCE:
            done = True
            d["stop_string"] = "lose too much"
        elif self.day >= self.df.shape[0]-STOCK_NUM:
            done = True
            d["stop_string"] = "Reach the end day"

        if not done:
            self.day += STOCK_NUM
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
        self.share_hold = [0]*STOCK_NUM
        self.cur_price = [0]*STOCK_NUM
        self.trading = [True]*STOCK_NUM
        self.net_worth = INI_ACCOUNT_BALANCE
        self.holding_price = [0]*STOCK_NUM
        self.trans_reward = 0
        self.state = self._get_obs()
        return self.state

    def updateDF(self,new_df,begin_day=None):
        self.df = new_df
        if begin_day:
            self.day = begin_day