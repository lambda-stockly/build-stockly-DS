import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import os
import tweepy
import re
import basilica
import gensim
import gensim.models.doc2vec as doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

'''
    This script will be for Twitter Sentiment and News Sentiment.
'''

class TwitterSentiment():
    '''
        TwitterSentiment object with one parameter being the stock ticker to search for.
    '''
    def __init__(self, ticker):
        # tweepy setup
        TWITTER_AUTH = tweepy.OAuthHandler(os.getenv('CONSUMER_KEY'),
                                           os.getenv('CONSUMER_SECRET'))
        TWITTER_AUTH.set_access_token(os.getenv('ACCESS_TOKEN'),
                                      os.getenv('ACCESS_SECRET'))

        self.api = tweepy.API(TWITTER_AUTH)

        # Basilica.ai
        BASILICA = basilica.Connection(os.getenv('BASILICA'))
        self.basilica = BASILICA

        ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
        ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas')

        self.ticker = ticker.upper()
        self.sid = SentimentIntensityAnalyzer()

    def make_df(self):

        max_tweets=1000
        tweets = [status for status in tweepy.Cursor(self.api.search,
                                             q=self.ticker,
                                             result_type="recent",
                                             tweet_mode="extended",
                                             lang="en",
                                             ).items(max_tweets)]
        df_tweets = [tweet.full_text for tweet in tweets]
        df_embedding = []
        for tweet in tweets:
            embedding = self.basilica.embed_sentence(tweet, model='twitter')


        data = pd.DataFrame(data=, columns=['Tweets','Basilica'])

        return data

    def nltk_magic(self):

        nltk.download('vader_lexicon')
        df = self.make_df()
        n_list = []

        for index, row in df.iterrows():
            s_pol = self.sid.polarity_scores(row['Tweets'])
            n_list.append(s_pol)

        series = pd.Series(n_list)

        df['polarity'] = series.values

        return df

    def output_preds(self):

        df = self.nltk_magic()



















