import flask
from flask import Flask, request, render_template, redirect, url_for
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import tkinter as tk
from flask_restful import Resource, Api
##########################################################
import requests
import csv
from collections import deque
##########################################################
# ML
import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

import pandas as pd
        # df['Date'] = df.index
        # df.index = range(len(df))
        # df = df.sort_values('Date')
        # df['Date'] = df['Date'].astype('str')
        # df = df.tail(1)
        # df.index = range(len(df))
###########################################################
app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# app.config.from_object('satcounter_config')

api = Api(app)

# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/getAction', methods=['POST', 'GET'])
def getAction(stockCode=None, trade=None):
    if request.method == 'POST':
        pass
    elif request.method == 'GET':
        # load model
        model = PPO2.load("./model/stock_RL2")
        # get stock code
        stockCode = request.args.get('ticker')

        # set api key
        api = "7OVAOOPNIMKJKIAX"
        r = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol="+stockCode+"&apikey="+ api +"&datatype=csv")
        decoded_content = r.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)

        df = pd.DataFrame(my_list[1:])
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'dividend_amount', 'split_coefficient']
        df_stock = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        df_stock = df_stock[::-1]
        df_stock.index = range(len(df_stock))

        df_stock['Open'] = df_stock['Open'].astype(float)
        df_stock['High'] = df_stock['High'].astype(float)
        df_stock['Low'] = df_stock['Low'].astype(float)
        df_stock['Close'] = df_stock['Close'].astype(float)
        df_stock['Adj Close'] = df_stock['Adj Close'].astype(float)
        df_stock['Volume'] = df_stock['Volume'].astype(float)


        env = DummyVecEnv([lambda: StockTradingEnv(df_stock)])
        obs = env.reset()
        trades = deque(maxlen=len(df_stock))
        for i in range(len(df_stock)):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            
            date = info[0]['date']
            trade = info[0]['trade']
            trades.append(date)
            trades.append(trade)


    return render_template('index.html', stockCode=stockCode, trade=trades)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
    # app.run(debug=True, threaded=True)
