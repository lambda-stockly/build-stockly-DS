# build-stockly-DS

### Lambda School Data Scientists 

### @Ruwai 'Chris L.'
### @derek-shing 'Derek S.'

# TODOs:

- Clean up repo
- update repo with app.py
- install dependencies to pipenv
- commit changes to heroku
- deleted Pipfile, install dependencies via `requirements.txt`

`pip install requirements.txt`

## Historical/Future Usage

```python
from preprocess import Magic

# insert stock ticker to instantiate Historical object.

tesla = Magic('TSLA')

# two endpoint methods that return a dictionary of softmax scores in format:
# {'Sell': 0.25, 'Hold': 0.5, 'Buy': 0.25}

# first method :output_historical:

historical = tesla.output_historical()
print(historical)

# second method :output_future:

future = tesla.output_future()
print(future)
```
## TwitterSentiment Usage

```python
from sentiment import TwitterSentiment

# same as above

tesla = TwitterSentiment('TSLA')

twitter_sentiment = tesla.output_twitter()
print(twitter_sentiment)

# should display :
# {'Sell': 0.1, 'Hold': 0.3, 'Buy': 0.6}
```
