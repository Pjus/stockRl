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

# model = PPO2.load("./model/stock_RL5")

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


xpredf = predf[:-101]
ypredf = predf[-101:]
ypredf.index = range(len(ypredf))



env = DummyVecEnv([lambda: StockTradingEnv(xpredf)])

mlpmodel = PPO2(MlpPolicy, env, verbose=1)
mlpmodel.learn(total_timesteps=500)

# lstmmodel = PPO2(MlpLstmPolicy, env, verbose=1, nminibatches=1)
# lstmmodel.learn(total_timesteps=500)

env = DummyVecEnv([lambda: StockTradingEnv(ypredf)])

obs = env.reset()
trades = deque(maxlen=len(ypredf))
for i in range(len(ypredf['Date'])-1):
    action, _states = mlpmodel.predict(obs)
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
    print("Date : ", i[0])
    print("Trade : ", i[1])
    print("Balance : ", i[2])
    print("Num shares : ", i[3])
    print("Total shares  : ", i[4])
    print("Stock price : ", round(i[5], 2))
    print("Total price : ", i[6])
    print("Portfolio Value : ", i[7])





# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(xpredf)])

model = PPO2(MlpLstmPolicy, env, verbose=1, nminibatches=1)
model.learn(total_timesteps=500)
# model.save("./model/stock_RL8")


# print(env.observation_space)
env = DummyVecEnv([lambda: StockTradingEnv(ypredf)])

obs = env.reset()
for i in range(len(ypredf['Date'])-1):
    action, _states = model.predict(obs)
    print("action : ", action)

    obs, rewards, done, info = env.step(action)
    print("obs : ", obs)
    print("Date : ", info[0]['date'])
    print("Trade : ", info[0]['trade'])
    print("Total_Net : ", info[0]['net'])
    print('\n')
