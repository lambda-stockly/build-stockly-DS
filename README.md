# build-stockly-DS

## Historical Usage

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
