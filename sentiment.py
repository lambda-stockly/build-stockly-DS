import pandas as pd
import numpy as np
import os
import tweepy
import re
import basilica
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

        # Basilica.ai
        BASILICA = basilica.Connection(os.getenv('BASILICA'))
        self.basilica = BASILICA
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
        # df_embedding = []
        # for tweet in tweets:
        #     embedding = self.basilica.embed_sentence(tweet, model='twitter')
        #     df_embedding.append(embedding)

        data = pd.DataFrame(data=df_tweets, columns=['Tweets'])
        # data['Embedding'] = df_embedding

        return data

    def nltk_magic(self):

        df = self.make_df()
        n_list = []

        for index, row in df.iterrows():
            s_pol = self.sid.polarity_scores(row['Tweets'])
            s_pol = dict()
            n_list.append(s_pol)

        series = pd.Series(n_list)

        df['polarity'] = series.values

        return df

    def output_twitter(self):

        data = self.nltk_magic()

        neg = []
        neu = []
        pos = []
        compound = []

        pol = data['polarity'].values

        for i in range(0, len(pol)):
            neg.append(pol[i]['neg'])
            neu.append(pol[i]['neu'])
            pos.append(pol[i]['pos'])
            compound.append(pol[i]['compound'])

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        scores = []

        sell = sum(neg)
        hold = sum(neu)
        buy = sum(pos)
        comp = sum(compound)

        scores = [sell,hold,buy]
        values = softmax(scores)
        keys = ['Sell','Hold','Buy']

        twitter_sentiment_analysis = dict(zip(keys,values))

        return twitter_sentiment_analysis





















