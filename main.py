import flask
from flask import Flask, request, render_template, redirect, url_for
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import tkinter as tk
# from ml.model import export_model
from flask_restful import Resource, Api

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

if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    # model = joblib.load('./model/model.pkl')
    # Flask 서비스 스타트
    app.run(host='127.0.0.1', port=8000, debug=True)
    # app.run(debug=True, threaded=True)
