import flask
from flask import Flask, request, render_template, redirect, url_for
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import tkinter as tk
# from ml.model import export_model
from flask_restful import Resource, Api

##########################################################
# ML
import pandas_datareader as pdr
import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from ml.env.StockTradingEnv import StockTradingEnv

import pandas as pd


# ticker = 'AAPL'
# start_date = '2018-01-01'

# df = pdr.get_data_yahoo(ticker, start_date)
# df['Date'] = df.index
# df.index = range(len(df))
# df = df.sort_values('Date')
# df['Date'] = df['Date'].astype('str')

# model = PPO2.load("./model/stock_RL")

# # The algorithms require a vectorized environment to run
# env = DummyVecEnv([lambda: StockTradingEnv(df)])

# obs = env.reset()
# for i in range(len(df['Date'])):
#     action, _states = model.predict(obs)
#     print("action : ", action)
#     print("action type : ",type(action))

#     obs, rewards, done, info = env.step(action)
#     print("info : ", info)

###########################################################
app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

api = Api(app)

# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index(ticker=None):
    return flask.render_template('index.html', ticker=ticker)


@app.route('/calculate', methods=['POST', 'GET'])
def calculate(num=None):
    ## 어떤 http method를 이용해서 전달받았는지를 아는 것이 필요함
    ## 아래에서 보는 바와 같이 어떤 방식으로 넘어왔느냐에 따라서 읽어들이는 방식이 달라짐
    if request.method == 'POST':
        #temp = request.form['num']
        pass
    elif request.method == 'GET':
        ## 넘겨받은 숫자 
        temp = request.args.get('ticker')
        ## 넘겨받은 문자
        ## 넘겨받은 값을 원래 페이지로 리다이렉트

        return render_template('index.html', num=temp)
    ## else 로 하지 않은 것은 POST, GET 이외에 다른 method로 넘어왔을 때를 구분하기 위함

@app.route('/getAction', methods=['POST', 'GET'])
def getAction(date=None, trade=None):
    if request.method == 'POST':
        pass
    elif request.method == 'GET':
        model = PPO2.load("./model/stock_RL")
        stockCode = request.args.get('ticker')
        start_date = '2018-01-01'
        df = pdr.get_data_yahoo(stockCode, start_date)
        df['Date'] = df.index
        df.index = range(len(df))
        df = df.sort_values('Date')
        df['Date'] = df['Date'].astype('str')
        df = df.tail(1)
        df.index = range(len(df))

        env = DummyVecEnv([lambda: StockTradingEnv(df)])
        obs = env.reset()
        for i in range(len(df['Date'])):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

            # print("Date : ", info[0]['date'])
            # print("Trade : ", info[0]['trade'])
            # print("Total_Net : ", info[0]['Net_worth'])
            # print('\n')
            date = info[0]['date']
            trade = info[0]['trade']

    return render_template('index.html', date=date, trade=trade)


if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    # model = joblib.load('./model/model.pkl')
    # Flask 서비스 스타트
    app.run(host='127.0.0.1', port=8000, debug=True)
    # app.run(debug=True, threaded=True)
