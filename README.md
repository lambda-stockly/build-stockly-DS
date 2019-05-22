# build-stockly-DS

# TODOs:

- !!!!!!!!!!! Route methods listed below to the flaskapp
- Sentiment script
- Download tweets/reddit posts/news to train a model
- Pickle said model to be used for the MVP
- Delete useless notebooks
- Create a clean notebook for examples and publications
- Update README with necessary TODOs and whatnot

## Historical/Future Usage

```python
from preprocess import Magic

# insert stock ticker to instantiate Historical object.

tesla = Magic('TSLA')

# two endpoint methods that return a dictionary of softmax scores in format:
# {'Sell': 0.25, 'Hold': 0.5, 'Buy': '0.25'}

# first method :output_historical:

historical = tesla.output_historical()
print(historical)

# second method :output_future:

future = tesla.output_future()
print(future)
```
