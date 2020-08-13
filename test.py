import pandas_datareader as pdr
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# from env.testEnv import StockTradingEnv
from env.StockTradingEnv import StockTradingEnv
from env.get_fundamental import getFundament

import pandas as pd

from ml.data_manager import *

# model = PPO2.load("./model/stock_RL2")

ticker = 'AAPL'
start_date = '2015-09-30'

df = load_data(ticker, start_date)
df['Date'] = df['Date'].astype('str')

predf = preprocess(df)
predf = predf.drop(['OBV'], axis=1)
predf.index = range(len(predf))
print(predf.head())
print(predf.columns)


# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(predf)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=500)
model.save("./model/stock_RL4")


print(env.observation_space)
obs = env.reset()
for i in range(len(predf['Date'])-1):
    action, _states = model.predict(obs)
    print("action : ", action)

    obs, rewards, done, info = env.step(action)
    print("obs : ", obs)
    print("Date : ", info[0]['date'])
    print("Trade : ", info[0]['trade'])
    print("Total_Net : ", info[0]['Net_worth'])
    print('\n')
