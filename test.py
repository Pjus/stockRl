import pandas_datareader as pdr
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# from env.testEnv import StockTradingEnv
from env.StockTradingEnv import StockTradingEnv
from env.get_fundamental import getFundament

import pandas as pd

from data_manager import *

# model = PPO2.load("./model/stock_RL2")

ticker = 'AAPL'
start_date = '2015-09-30'

df = load_data(ticker, start_date)
df['Date'] = df['Date'].astype('str')

# predf = preprocess(df)
# print(predf.head())
# predf.index = range(len(predf))


# ticker = 'AAPL'
# start_date = '2015-09-30'

# df = pdr.get_data_yahoo(ticker, start_date)
# df['Date'] = df.index
# df.index = range(len(df))
# df = df.sort_values('Date')
# df['Date'] = df['Date'].astype('str')
# temp_df = pd.merge(df, getFundament('AAPL'), on='Date', how='left').fillna(method='ffill')
# print(temp_df.tail())

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=200)
model.save("./model/stock_RL3")


print(env.observation_space)
obs = env.reset()
for i in range(len(df['Date'])-1):
    action, _states = model.predict(obs)
    print("action : ", action)
    # print("action type : ",type(action))

    obs, rewards, done, info = env.step(action)
    # print("obs : ", obs)
    print("Date : ", info[0]['date'])
    print("Trade : ", info[0]['trade'])
    print("Total_Net : ", info[0]['Net_worth'])
    print('\n')
