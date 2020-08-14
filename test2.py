import warnings 
warnings.filterwarnings("ignore")


import pandas_datareader as pdr
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# from env.testEnv import StockTradingEnv
from env.StockTradingEnv import StockTradingEnv
from env.get_fundamental import getFundament

import pandas as pd

from ml.data_manager import *
from collections import deque
import math

model = PPO2.load("./model/stock_RL5")

# get stock code
stockCode = 'AAPL'
start_date = '2010-01-01'

df = load_data(stockCode, start_date)
df['Date'] = df['Date'].astype('str')
df.index = range(len(df))

predf = preprocess(df)
predf = predf.drop(['OBV'], axis=1)
predf = predf[-101:]
predf.index = range(len(predf))

env = DummyVecEnv([lambda: StockTradingEnv(predf)])
obs = env.reset()
trades = deque(maxlen=len(predf))
for i in range(len(predf['Date'])-1):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    date = info[0]['date']
    trade = info[0]['trade']
    balance = math.trunc(info[0]['balance'])
    num_share = info[0]['num_share']
    price = info[0]['price']
    total_price = math.trunc(info[0]['price'] * num_share)
    shares = info[0]['shares']
    net = round(info[0]['net'], 2)
    trades.append([date, trade, balance, num_share, shares, price, total_price, net])

for i in trades:
    print(i)
