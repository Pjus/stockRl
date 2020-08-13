from pymongo import MongoClient
import pandas_datareader as pdr
from ml.data_manager import *
from bson.json_util import loads, dumps

ticker = 'RUN'
start_date = '2020-01-01'

df = load_data(ticker, start_date)
df['Date'] = df['Date'].astype('str')

print(df)

# my_client = MongoClient("mongodb://localhost:27017/")

# # print(my_client.list_database_names())
# mydb = my_client['mydatabase']
# mycol = mydb['stock']

# # for ind, row in df.iterrows():
# #     # print(row['High'])
# #     x = mycol.insert_one({"Date":row['Date'], "High":row['High'], "Low": row['Low'], "Open":row['Open'], "Close":row['Close'], "Volume":row['Volume'], "Adj Close":row['Adj Close']}) 
# #     print(x.inserted_id)

# cursor = mycol.find({})
# # for i in cursor:
# #     print(i)
# with open('collection.json', 'w') as file:
#     file.write('[')
#     for document in cursor:
#         file.write(dumps(document))
#         file.write(',')
#     file.write(']')