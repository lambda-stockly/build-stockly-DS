import fbprophet
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import os

class Historical():
    '''
        original script from :
        https://github.com/WillKoehrsen/Data-Analysis/blob/master/stocker/stocker.py
        credit goes to this script.
    '''
    #Initialize parameters
    def __init__(self, ticker):

        ALPHAVANTAGE_API_KEY = '0GOYV58FN3FF3CO2'

        ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas')

        ticker = ticker.upper()
        self.symbol = ticker

        try:
            data, meta_data = ts.get_daily(self.symbol, outputsize='full')

        except Exception as e:
            print('Error retrieving Stock Data...')
            print(e)
            return

        data = data.reset_index(level=0)
        data['date'] = pd.to_datetime(data['date'])
        data['ds'] = data['date']
        data = data.rename(columns={
                'date': 'Date', '1. open': 'Open', '2. high': 'High',
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
    """
    Make sure start and end dates are in the range and can be
    converted to pandas datetimes. Returns dates in the correct format
    """
    def handle_dates(self, start_date, end_date):


        # Default start and end date are the beginning and end of data
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date

        try:
            # Convert to pandas datetime for indexing dataframe
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

        except Exception as e:
            print('Enter valid pandas date format.')
            print(e)
            return

        valid_start = False
        valid_end = False

        # User will continue to enter dates until valid dates are met
        while (not valid_start) & (not valid_end):
            valid_end = True
            valid_start = True

            if end_date < start_date:
                print('End Date must be later than start date.')
                start_date = pd.to_datetime(input('Enter a new start date: '))
                end_date= pd.to_datetime(input('Enter a new end date: '))
                valid_end = False
                valid_start = False

            else:
                if end_date > self.max_date:
                    print('End Date exceeds data range')
                    end_date= pd.to_datetime(input('Enter a new end date: '))
                    valid_end = False

                if start_date < self.min_date:
                    print('Start Date is before date range')
                    start_date = pd.to_datetime(input('Enter a new start date: '))
                    valid_start = False


        return start_date, end_date

    def make_a_df(self,start_date=None, end_date=None,df=None):
        '''
            Added by Chris Louie for stockly
        '''
        # Default is to use the object stock data
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
        if not df:
            df = self.stock.copy()

        start_date, end_date = self.handle_dates(start_date, end_date)

        # keep track of whether the start and end dates are in the data
        start_in = True
        end_in = True

        # If user wants to round dates (default behavior)
        if self.round_dates:
            # Record if start and end date are in df
            if (start_date not in list(df['Date'])):
                start_in = False
            if (end_date not in list(df['Date'])):
                end_in = False

            # If both are not in dataframe, round both
            if (not end_in) & (not start_in):
                trim_df = df[(df['Date'] >= start_date) &
                             (df['Date'] <= end_date)]

            else:
                # If both are in dataframe, round neither
                if (end_in) & (start_in):
                    trim_df = df[(df['Date'] >= start_date) &
                                 (df['Date'] <= end_date)]
                else:
                    # If only start is missing, round start
                    if (not start_in):
                        trim_df = df[(df['Date'] > start_date) &
                                     (df['Date'] <= end_date)]
                    # If only end is missing round end
                    elif (not end_in):
                        trim_df = df[(df['Date'] >= start_date) &
                                     (df['Date'] < end_date)]


        else:
            valid_start = False
            valid_end = False
            while (not valid_start) & (not valid_end):
                start_date, end_date = self.handle_dates(start_date, end_date)

                # No round dates, if either data not in, print message and return
                if (start_date in list(df['Date'])):
                    valid_start = True
                if (end_date in list(df['Date'])):
                    valid_end = True

                # Check to make sure dates are in the data
                if (start_date not in list(df['Date'])):
                    print('Start Date not in data (either out of range or not a trading day.)')
                    start_date = pd.to_datetime(input(prompt='Enter a new start date: '))

                elif (end_date not in list(df['Date'])):
                    print('End Date not in data (either out of range or not a trading day.)')
                    end_date = pd.to_datetime(input(prompt='Enter a new end date: ') )

            # Dates are not rounded
            trim_df = df[(df['Date'] >= start_date) &
                         (df['Date'] <= end_date.date)]

        up_days = []
        down_days = []

        for i in range(0,len(trim_df)):
            if trim_df['Daily Change'][i] > 0:
                up_days.append(1)
                down_days.append(0)
            elif trim_df['Daily Change'][i] < 0:
                down_days.append(1)
                up_days.append(0)
            else:
                down_days.append(0)
                up_days.append(0)
        print(len(up_days))
        print(len(down_days))
        trim_df['Up Days'] = up_days
        trim_df['Down Days'] = down_days

        return trim_df

    def resample(self, dataframe):
        # Change the index and resample at daily level
        dataframe = dataframe.set_index('ds')
        dataframe = dataframe.resample('D')

        # Reset the index and interpolate nan values
        dataframe = dataframe.reset_index(level=0)
        dataframe = dataframe.interpolate()
        return dataframe

    def remove_weekends(self, dataframe):

        # Reset index to use ix
        dataframe = dataframe.reset_index(drop=True)

        weekends = []

        # Find all of the weekends
        for i, date in enumerate(dataframe['ds']):
            if (date.weekday()) == 5 | (date.weekday() == 6):
                weekends.append(i)

        # Drop the weekends
        dataframe = dataframe.drop(weekends, axis=0)

        return dataframe

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

    def evaluate_prediction(self, start_date=None, end_date=None, nshares = None):

        # Default start date is one year before end of data
        # Default end date is end date of data
        if start_date is None:
            start_date = self.max_date - pd.DateOffset(years=1)
        if end_date is None:
            end_date = self.max_date

        start_date, end_date = self.handle_dates(start_date, end_date)

        # Training data starts self.training_years years before start date and goes up to start date
        train = self.stock[(self.stock['Date'] < start_date) &
                           (self.stock['Date'] > (start_date - pd.DateOffset(years=self.training_years)))]

        # Testing data is specified in the range
        test = self.stock[(self.stock['Date'] >= start_date) & (self.stock['Date'] <= end_date)]

        # Create and train the model
        model = self.create_model()
        model.fit(train)

        # Make a future dataframe and predictions
        future = model.make_future_dataframe(periods = 365, freq='D')
        future = model.predict(future)

        # Merge predictions with the known values
        test = pd.merge(test, future, on = 'ds', how = 'inner')

        train = pd.merge(train, future, on = 'ds', how = 'inner')

        # Calculate the differences between consecutive measurements
        test['pred_diff'] = test['yhat'].diff()
        test['real_diff'] = test['y'].diff()

        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff'][1:]) == np.sign(test['real_diff'][1:])) * 1

        # Accuracy when we predict increase and decrease
        increase_accuracy = 100 * np.mean(test[test['pred_diff'] > 0]['correct'])
        decrease_accuracy = 100 * np.mean(test[test['pred_diff'] < 0]['correct'])

        # Calculate mean absolute error
        test_errors = abs(test['y'] - test['yhat'])
        test_mean_error = np.mean(test_errors)

        train_errors = abs(train['y'] - train['yhat'])
        train_mean_error = np.mean(train_errors)

        # Calculate percentage of time actual value within prediction range
        test['in_range'] = False

        for i in test.index:
            if (test.loc[i, 'y'] < test.loc[i, 'yhat_upper']) & (test.loc[i, 'y'] > test.loc[i, 'yhat_lower']):
                test.loc[i, 'in_range'] = True

        in_range_accuracy = 100 * np.mean(test['in_range'])

        if not nshares:

            # Date range of predictions
            print('\nPrediction Range: {} to {}.'.format(start_date,
                end_date))

            # Final prediction vs actual value
            print('\nPredicted price on {} = ${:.2f}.'.format(max(future['ds']), future.loc[future.index[-1], 'yhat']))
            print('Actual price on    {} = ${:.2f}.\n'.format(max(test['ds']), test.loc[test.index[-1], 'y']))

            print('Average Absolute Error on Training Data = ${:.2f}.'.format(train_mean_error))
            print('Average Absolute Error on Testing  Data = ${:.2f}.\n'.format(test_mean_error))

            # Direction accuracy
            print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
            print('When the model predicted a  decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))

            print('The actual value was within the {:d}% confidence interval {:.2f}% of the time.'.format(int(100 * model.interval_width), in_range_accuracy))

        # If a number of shares is specified, play the game
        elif nshares:

            # Only playing the stocks when we predict the stock will increase
            test_pred_increase = test[test['pred_diff'] > 0]

            test_pred_increase.reset_index(inplace=True)
            prediction_profit = []

            # Iterate through all the predictions and calculate profit from playing
            for i, correct in enumerate(test_pred_increase['correct']):

                # If we predicted up and the price goes up, we gain the difference
                if correct == 1:
                    prediction_profit.append(nshares * test_pred_increase.loc[i, 'real_diff'])
                # If we predicted up and the price goes down, we lose the difference
                else:
                    prediction_profit.append(nshares * test_pred_increase.loc[i, 'real_diff'])

            test_pred_increase['pred_profit'] = prediction_profit

            # Put the profit into the test dataframe
            test = pd.merge(test, test_pred_increase[['ds', 'pred_profit']], on = 'ds', how = 'left')
            test.loc[0, 'pred_profit'] = 0

            # Profit for either method at all dates
            test['pred_profit'] = test['pred_profit'].cumsum().ffill()
            test['hold_profit'] = nshares * (test['y'] - float(test.loc[0, 'y']))

            # Display information
            print('You played the stock market in {} from {} to {} with {} shares.\n'.format(
                self.symbol, start_date, end_date, nshares))

            print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
            print('When the model predicted a  decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))

            # Display some friendly information about the perils of playing the stock market
            print('The total profit using the Prophet model = ${:.2f}.'.format(np.sum(prediction_profit)))
            print('The Buy and Hold strategy profit =         ${:.2f}.'.format(float(test.loc[test.index[-1], 'hold_profit'])))
            print('\nThanks for playing the stock market!\n')

            # Plot the predicted and actual profits over time

            # Final profit and final smart used for locating text
            final_profit = test.loc[test.index[-1], 'pred_profit']
            final_smart = test.loc[test.index[-1], 'hold_profit']

            # text location
            last_date = test.loc[test.index[-1], 'ds']
            text_location = (last_date - pd.DateOffset(months = 1))

        return test

    def make_a_future_dataframe(self,periods=30,freq='D'):
        '''
            Added by Chris Louie for stockly
        '''
        train = self.stock[self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=self.training_years))]

        model = self.create_model()
        model.fit(train)

        future = model.make_future_dataframe(periods=periods,freq=freq)
        future = model.predict(future)

        preds = future[future['ds'] >= max(self.stock['Date'])]
        preds = self.remove_weekends(preds)
        preds['diff'] = preds['yhat'].diff()
        preds = preds.dropna()
        preds['direction'] = (preds['diff'] > 0) * 1
        preds = preds.rename(columns={
            'ds': 'Date', 'yhat': 'estimate', 'diff': 'change',
            'yhat_upper': 'upper', 'yhat_lower': 'lower'
        })

        preds = preds.reset_index()

        up_days = []
        down_days = []

        for i in range(len(preds)):
            if preds['estimate'][i] > 0:
                up_days.append(1)
                down_days.append(0)
            elif preds['estimate'][i] < 0:
                down_days.append(1)
                up_days.append(0)
            else:
                down_days.append(0)
                up_days.append(0)
        print(len(up_days))
        print(len(down_days))
        preds['Up Days'] = up_days
        preds['Down Days'] = down_days

        return preds

    # Predict the future price for a given range of days
    def predict_future(self, days=30):

        # Use past self.training_years years for training
        train = self.stock[self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=self.training_years))]

        model = self.create_model()

        model.fit(train)

        # Future dataframe with specified number of days to predict
        future = model.make_future_dataframe(periods=days, freq='D')
        future = model.predict(future)

        # Only concerned with future dates
        future = future[future['ds'] >= max(self.stock['Date'])]

        # Remove the weekends
        future = self.remove_weekends(future)

        # Calculate whether increase or not
        future['diff'] = future['yhat'].diff()

        future = future.dropna()

        # Find the prediction direction and create separate dataframes
        future['direction'] = (future['diff'] > 0) * 1

        # Rename the columns for presentation
        future = future.rename(columns={'ds': 'Date', 'yhat': 'estimate', 'diff': 'change',
                                        'yhat_upper': 'upper', 'yhat_lower': 'lower'})

        future_increase = future[future['direction'] == 1]
        future_decrease = future[future['direction'] == 0]

        # Print out the dates
        print('\nPredicted Increase: \n')
        print(future_increase[['Date', 'estimate', 'change', 'upper', 'lower']])

        print('\nPredicted Decrease: \n')
        print(future_decrease[['Date', 'estimate', 'change', 'upper', 'lower']])

        return future

    def output(self):
        '''
            This method is for storing an output for the predict_future method.
            Create softmax probability for whether player should buy hold or sell
        '''

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        future_model = self.predict_future()
        average_delta = np.mean(future_model['change'])

        buy = sum(future_model['direction'] == 1)
        sell = sum(future_model['direction'] == 0)

        if average_delta > 1:
            hold = average_delta
        elif average_delta < -1:
            hold = -average_delta
        else:
            hold = (buy+sell+average_delta)/3

        scores = [sell,hold,buy]
        values = softmax(scores)
        keys = ['Sell','Hold','Buy']

        historical_analysis = dict(zip(keys,values))

        # output = dict(zip(keys,zip(*softmax_scores)))

        return historical_analysis

