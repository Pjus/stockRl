import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from collections import deque

from render.StockTradingGraph import StockTradingGraph

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5


INITIAL_ACCOUNT_BALANCE = 10000

LOOKBACK_WINDOW_SIZE = 10


def factor_pairs(val):
    return [(i, val / i) for i in range(1, int(val**0.5)+1) if val % i == 0]


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.MAX_STEPS = len(df)

        self.df = self._adjust_prices(df)
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.df.columns) + 4, LOOKBACK_WINDOW_SIZE), dtype=np.float16)

    def _adjust_prices(self, df):
        adjust_ratio = df['Adj Close'] / df['Close']

        df['Open'] = df['Open'] * adjust_ratio
        df['High'] = df['High'] * adjust_ratio
        df['Low'] = df['Low'] * adjust_ratio
        df['Close'] = df['Close'] * adjust_ratio

        return df

    def _next_observation(self):

        frame = np.zeros((11,10))

        opendeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)
        highdeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)
        lowdeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)
        closedeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)
        ajdeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)
        voldeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)
        bladeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)
        netdeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)
        shadeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)
        costdeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)
        totaldeq = deque(maxlen=LOOKBACK_WINDOW_SIZE)

        opendeq.append(self.df.loc[self.current_step, 'Open'] / MAX_SHARE_PRICE)
        highdeq.append(self.df.loc[self.current_step, 'High'] / MAX_SHARE_PRICE)
        lowdeq.append(self.df.loc[self.current_step, 'Low'] / MAX_SHARE_PRICE)
        closedeq.append(self.df.loc[self.current_step, 'Close'] / MAX_SHARE_PRICE)
        ajdeq.append(self.df.loc[self.current_step, 'Adj Close'] / MAX_SHARE_PRICE)
        voldeq.append(self.df.loc[self.current_step, 'Volume'] / MAX_SHARE_PRICE)
        bladeq.append(self.balance / MAX_ACCOUNT_BALANCE)
        netdeq.append(self.max_net_worth / MAX_ACCOUNT_BALANCE)
        shadeq.append(self.shares_held / MAX_NUM_SHARES)
        costdeq.append(self.cost_basis / MAX_SHARE_PRICE)
        totaldeq.append(self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE))

        if self.current_step > 10:
            obs = np.array([opendeq, highdeq, lowdeq, closedeq, ajdeq, voldeq, bladeq, netdeq, shadeq, costdeq, totaldeq])
            return obs
        return frame

    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

            if shares_bought > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': shares_bought, 'total': additional_cost,
                                    'type': "Buy"})
            else:
                self.trades.append({'step': self.current_step,
                                    'shares': 0, 'total': 0,
                                    'type': "Hold"})

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

            if shares_sold > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': shares_sold, 'total': shares_sold * current_price,
                                    'type': "Sell"})

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        if self.current_step > self.df.index[-1]:
            self.current_step = self.df.index[-1]

        delay_modifier = (self.current_step / self.MAX_STEPS)

        reward = self.balance * delay_modifier + self.current_step
        done = self.net_worth <= 0 or self.current_step >= len(self.df)

        obs = self._next_observation()

        
        trade = self.trades
        if len(trade) > 0:
            tra = trade[-1]['type']
            date = self.df['Date'].values[trade[-1]['step']]
        else:
            tra = None
            date = None

        net = self.net_worth

        return obs, reward, done, {"date" : date, "trade" : tra, 'Net_worth' : net}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0
        self.trades = []

        return self._next_observation()

    def _render_to_file(self, filename='render.txt'):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        file = open(filename, 'a+')

        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})\n')
        file.write(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})\n')
        file.write(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})\n')
        file.write(f'Profit: {profit}\n\n')

        file.close()

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))

        elif mode == 'live':
            if self.visualization == None:
                self.visualization = StockTradingGraph(
                    self.df, kwargs.get('title', None))

            if self.current_step > LOOKBACK_WINDOW_SIZE:
                self.visualization.render(
                    self.current_step, self.net_worth, self.trades, window_size=LOOKBACK_WINDOW_SIZE)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None
