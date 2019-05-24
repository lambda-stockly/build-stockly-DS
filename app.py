from flask import Flask , request
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler
import os
import requests
from google.cloud import language_v1
from google.cloud.language_v1 import enums
import six
from os import getenv
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()




def sample_analyze_sentiment(content):

    client = language_v1.LanguageServiceClient()

    #content = 'Your text to analyze, e.g. Hello, world!'

    if isinstance(content, six.binary_type):
        content = content.decode('utf-8')

    type_ = enums.Document.Type.PLAIN_TEXT
    document = {'type': type_, 'content': content}

    response = client.analyze_sentiment(document)
    sentiment = response.document_sentiment
    #print('Score: {}'.format(sentiment.score))
    #print('Magnitude: {}'.format(sentiment.magnitude))
    return sentiment.score

def get_news(ticker):
    url = "http://finance.yahoo.com/rss/headline?s="
    feed = requests.get(url + ticker)
    soup = BeautifulSoup(feed.text)
    news = soup.find_all('description')
    text = []
    scores =[]
    for new in news:
        text.append(new.text)
    for new in text[1:10]:
        score = sample_analyze_sentiment(new)
        scores.append(score)

    df = pd.DataFrame(text[1:10])
    df['score']=scores
    return df



APP = Flask(__name__)


model = pickle.load( open( "model.p", "rb" ) )




def generate_df(ticker):
    macd = 'https://www.alphavantage.co/query?function=MACD&symbol=' + ticker + '&interval=daily&series_type=open&apikey=SXG08DL4S2EW8SKC'
    response1 = requests.get(macd)
    if "Note" in response1.json().keys():
        return (response1.json()["Note"])
    df_macd = pd.DataFrame.from_dict(response1.json()['Technical Analysis: MACD']).T

    stoch = 'https://www.alphavantage.co/query?function=STOCH&symbol=' + ticker + '&interval=daily&apikey=SXG08DL4S2EW8SKC'
    response2 = requests.get(stoch)
    if "Note" in response2.json().keys():
        return (response2.json()["Note"])
    df_stoch = pd.DataFrame.from_dict(response2.json()['Technical Analysis: STOCH']).T

    # rsi = 'https://www.alphavantage.co/query?function=RSI&symbol='+ticker+'&interval=daily&time_period=10&series_type=open&apikey=NXAA2P2XI1GQSYPG'
    # response3 = requests.get(rsi)
    # df_rsi = pd.DataFrame.from_dict(response3.json()['Technical Analysis: RSI']).T

    aroon = 'https://www.alphavantage.co/query?function=AROONOSC&symbol=' + ticker + '&interval=daily&time_period=10&apikey=SXG08DL4S2EW8SKC'
    response4 = requests.get(aroon)
    if "Note" in response4.json().keys():
        return (response4.json()["Note"])
    df_aroon = pd.DataFrame.from_dict(response4.json()['Technical Analysis: AROONOSC']).T

    dx = 'https://www.alphavantage.co/query?function=DX&symbol=' + ticker + '&interval=daily&time_period=10&apikey=SXG08DL4S2EW8SKC'
    response5 = requests.get(dx)
    if "Note" in response5.json().keys():
        return (response5.json()["Note"])
    df_dx = pd.DataFrame.from_dict(response5.json()['Technical Analysis: DX']).T

    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + ticker + '&interval=5min&outputsize=full&apikey=SXG08DL4S2EW8SKC'
    response6 = requests.get(url)
    if "Note" in response6.json().keys():
        return (response6.json()["Note"])
    df = pd.DataFrame.from_dict(response6.json()['Time Series (Daily)']).T

    # Join all the dataset
    df = df.join(df_macd)
    df = df.join(df_stoch)
    # df = df.join(df_rsi)
    df = df.join(df_aroon)
    df = df.join(df_dx)
    return df

def signal_class(p):
  if p>=0.05:
    r = 1
  elif p<=-0.05:
    r = -1
  else:
    r=0
  return r

def generate_target(df):
  df2 =df.copy()
  df2 = df.astype(float)
  df2['next_10day_close']=df2['4. close'].shift(10)
  df2['percentage_change']=(df2['next_10day_close']-df2['4. close'])/df2['4. close']
  df2['signal']=df2['percentage_change'].apply(signal_class)
  return df2



@APP.route('/')
@APP.route('/api',methods=['POST'])
def hello_world():
    input ="AAPL"
    if request.method == 'POST':
        input = request.values['ticker']
    market_df = generate_df(input)
    if type(market_df)==str:
        return market_df

    market_df = market_df.dropna()

    X = market_df[['5. volume', 'MACD', 'AROONOSC','MACD_Hist', 'MACD_Signal', 'DX', 'SlowD', 'SlowK']]
    #print(X[0])
    sc = StandardScaler()
    X = sc.fit_transform(X)
    #test = np.array([[ 0.84330129, -0.87267448, -1.55021623, -1.14815012, -0.54096114,1.74642336, -1.14298853, -1.59289229]])
    y_prebro = model.predict_proba(X[0].reshape(1, -1))
    #y_prebro = model.predict_proba(test)
    #print(y_prebro)

    #s,t=get_news(input)

    dict1 = {'TA': {'sell':y_prebro[0][0],'hold':y_prebro[0][1],'buy':y_prebro[0][2]}, 'Sentiment':{'sell':0.5,'hold':0.25,'buy':0.25}}
    #dict1 = {'TA': {'sell': y_prebro[0][0], 'hold': y_prebro[0][1], 'buy': y_prebro[0][2]},
    #         'Sentiment': {'score':s,'news':t}}
    #dict1 = {'TA': {'sell': 0.5, 'hold': 0.25, 'buy': 0.25},'Sentiment': {'sell': 0.5, 'hold': 0.25, 'buy': 0.25}}
    json1 = json.dumps(dict1)
    response=json1
    print(response)


    return response


@APP.route('/sentiment')
@APP.route('/sentiment',methods=['POST'])
def sentiment():
    input ="AAPL"
    if request.method == 'POST':
        input = request.values['ticker']
    market_df = generate_df(input)
    if type(market_df)==str:
        return market_df


    #json1 = json.dumps(dict1)
    score=sample_analyze_sentiment("The bad news seems to keep on coming for Tesla Inc., and one expert says this is the year the electric-car company “comes undone” — and maybe gets bought by Apple Inc.")
    json1 = json.dumps({"sentiment":score})

    response=json1


    return response