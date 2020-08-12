# get fundamental data
import requests
import pandas as pd

def findNone(balance_df_):
    col = []
    for i in balance_df_.columns:
        for j in balance_df_[i]:
            if 'None' in j:
                if i in col:
                    pass
                else:
                    col.append(i)
    return col
    

def getFundament(stockCode):

    result = pd.DataFrame()
    stockCode = stockCode
    api = "7OVAOOPNIMKJKIAX"
    function = ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']

    for func in function:
        r = requests.get("https://www.alphavantage.co/query?function="+ func +"&symbol="+stockCode+"&apikey=" + api)
        data = r.json()['quarterlyReports']
        data_df = pd.DataFrame(data)
        dropCol = findNone(data_df)
        dropCol.append('reportedCurrency')
        data_df_ = data_df.drop(dropCol, axis=1)[::-1]
        data_df_.rename(columns = {'fiscalDateEnding' : 'Date_{}'.format(func)}, inplace=True)
        data_df_1 = pd.concat([data_df_[['Date_{}'.format(func)]], data_df_.drop(['Date_{}'.format(func)], axis=1).astype(float) / 10000000000], axis=1)
        result = pd.concat([result, data_df_1], axis=1)

    result.drop(['Date_{}'.format(function[1]), 'Date_{}'.format(function[2])], axis=1, inplace=True)
    result.rename(columns = {'Date_{}'.format(function[0]) : 'Date'}, inplace=True)

    return result

def get_indicator(stockCode):

    indicators = ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'VWAP']