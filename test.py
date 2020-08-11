import pandas_datareader as pdr
import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

import pandas as pd
import joblib
import itertools

model = PPO2.load("./model/stock_RL")

ticker = 'AAPL'
start_date = '2018-01-01'

df = pdr.get_data_yahoo(ticker, start_date)
df['Date'] = df.index
df.index = range(len(df))
df = df.sort_values('Date')
df['Date'] = df['Date'].astype('str')
df = df.tail(1)
df.index = range(len(df))
print(df)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

obs = env.reset()
for i in range(len(df['Date'])):
    action, _states = model.predict(obs)
    # print("action : ", action)
    # print("action type : ",type(action))

    obs, rewards, done, info = env.step(action)
    print("Date : ", info[0]['date'])
    print("Trade : ", info[0]['trade'])
    print("Total_Net : ", info[0]['Net_worth'])
    print('\n')
