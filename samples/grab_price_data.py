from pprint import pprint
from fake_useragent import UserAgent
from pyopt.client import PriceHistory

 # Initialize the client.
price_history_client = PriceHistory(
    symbols=['AAPL','MSFT','SQ'],
    user_agent=UserAgent().edge
)

# Dump it to a CSV file.
price_history_client.price_data_frame.to_csv(
    'data/stock_data.csv',
    index=False
)
pprint(price_history_client.price_data_frame)

# Grab the data frame.
price_data_frame = price_history_client.price_data_frame