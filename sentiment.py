import pandas as pd
import numpy as np
import os
import tweepy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

'''
    This script will be for Twitter Sentiment and News Sentiment.
'''

class TwitterSentiment():
    '''
        TwitterSentiment object wth one parameter being the stock ticker to search for.
    '''
    def __init__(self, ticker):
        # tweepy setup
        nltk.download('vader_lexicon')

        TWITTER_AUTH = tweepy.OAuthHandler(os.getenv('CONSUMER_KEY'),
                                           os.getenv('CONSUMER_SECRET'))
        TWITTER_AUTH.set_access_token(os.getenv('ACCESS_TOKEN'),
                                      os.getenv('ACCESS_SECRET'))

        self.api = tweepy.API(TWITTER_AUTH)
        self.ticker = ticker.upper()

    def make_df_with_magic(self):

        api = self.api
        symbol = self.ticker
        sid = SentimentIntensityAnalyzer()
        max_tweets = 200

        tweets = [status for status in tweepy.Cursor(api.search,
                                                     q=symbol,
                                                     result_type="recent",
                                                     tweet_mode="extended",
                                                     lang="en").items(max_tweets)]

        data = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns = ['Tweets'])
        n_list = []

        for index, row in data.iterrows():
            s_pol = sid.polarity_scores(row['Tweets'])
            n_list.append(s_pol)

        series = pd.Series(n_list)
        data['polarity'] = series.values

        return data

    def output_twitter(self):

        data = self.make_df_with_magic()

        neg = []
        neu = []
        pos = []
        compound = []

        pol = data['polarity']

        for i in range(0, len(pol)):
            neg.append(pol[i]['neg'])
            neu.append(pol[i]['neu'])
            pos.append(pol[i]['pos'])
            compound.append(pol[i]['compound'])

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        sell = sum(neg)
        # hold = sum(neu)
        buy = sum(pos)
        comp = sum(compound)

        scores = [sell, comp, buy]
        values = softmax(scores)
        keys = ['Sell', 'Hold', 'Buy']

        twitter_sentiment_analysis = dict(zip(keys, values))

        return twitter_sentiment_analysis