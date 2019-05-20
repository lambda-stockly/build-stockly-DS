import fbprophet
import pandas as pd
import numpy as np
import pytrends
import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries
import os

class DailyPredict():
    '''
        original script from :
        https://github.com/WillKoehrsen/Data-Analysis/blob/master/stocker/stocker.py
        credit goes to this script.
    '''
    #Initialize parameters
    def __init__(self, ticker):

        ticker = ticker.upper()
        self.symbol = ticker
        self.api_key = os.getenv('ALPHAVANTAGE_API_KEY')

        ts = TimeSeries(key=self.api_key, output_format='pandas')

        try:
            data, meta_data = ts.get_daily(self.symbol, outputsize='fill')

        except Exception as e:
            print('Error retrieving Stock Data...')
            print(e)
            return

        data = data.reset_index(level=0)

        data['date'] = data['Date']
        data = data.rename(columns={
                'Date': 'Date', '1. open': 'Open', '2. high': 'High',
                '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'
            })

        if ('Adj. Close' not in data.columns):
            data['Adj. Close'] = data['Close']
            data['Adj. Open'] = data['Open']

        data['y'] = data['Adj. Close']
        data['Daily Change'] = data['Adj. Close'] - data['Adj. Open']

        self.stock = data.copy()

        self.min_date = min(data['Date'])
        self.max_date = max(data['Date'])

        self.max_price = np.max(self.stock['y'])
        self.min_price = np.min(self.stock['y'])

        self.min_price_date = self.stock[self.stock['y'] == self.min_price]['Date']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
        self.max_price_date = self.stock[self.stock['y'] == self.max_price]['Date']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]

        self.starting_price = float(self.stock.loc[0, 'Adj. Open'])
        self.most_recent_price = float(self.stock.loc[self.stock.index[-1], 'y'])

        self.round_dates = True
        self.training_years = 3
        self.changepoint_prior_scale = 0.05
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.changepoints = None

        print('{} Preprocessing Initialized. Data covers {} to {}.'.format(self.symbol,
                                                                     self.min_date,
                                                                     self.max_date))

    def create_model(self):

        # Make the model
        model = fbprophet.Prophet(daily_seasonality=self.daily_seasonality,
                                  weekly_seasonality=self.weekly_seasonality,
                                  yearly_seasonality=self.yearly_seasonality,
                                  changepoint_prior_scale=self.changepoint_prior_scale,
                                  changepoints=self.changepoints)

        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)

        return model

    def create_prophet_model(self, days=0, resample=False):

        model = self.create_model()

        # Fit on the stock history for self.training_years number of years
        stock_history = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years = self.training_years))]

        if resample:
            stock_history = self.resample(stock_history)

        model.fit(stock_history)

        # Make and predict for next year with future dataframe
        future = model.make_future_dataframe(periods = days, freq='D')
        future = model.predict(future)

        if days > 0:
            # Print the predicted price
            print('Predicted Price on {} = ${:.2f}'.format(
                future.loc[future.index[-1], 'ds'], future.loc[future.index[-1], 'yhat']))

        # Set up the plot

        return model, future



