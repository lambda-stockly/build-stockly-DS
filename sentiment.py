import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import os
import fbprophet
from sklearn.model_selection import train_test_splits
import tweepy
import re
import basilica
import gensim
import gensim.models.doc2vec as doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

'''
    This script will be for Reddit Sentiment and Twitter Sentiment.

'''

class Reddit():
	def __init__(self):
		'''
			Fill in this area with code from our tests.
		'''