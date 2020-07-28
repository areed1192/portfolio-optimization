import requests
import pandas as pd

from typing import List
from typing import Dict
from typing import Union
from datetime import date
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import *

from fake_useragent import UserAgent

class PriceHistory():

    def __init__(self, symbols: List[str]):
        self._api_url = 'https://api.nasdaq.com/api/quote'
        self._api_service = 'historical'
        self._symbols = symbols
        self.price_data_frame = self._build_data_frames()

    def _build_url(self, symbol: str) -> str:
        parts = [self._api_url, symbol, self._api_service]
        return '/'.join(parts)

    @property
    def symbols(self) -> List[str]:
        return self._symbol
    
    def _build_data_frames(self) -> pd.DataFrame:

        all_data = []
        to_date = datetime.today().date()

        # Calculate the Start and End Point.
        from_date = to_date - relativedelta(months=6)

        for symbol in self._symbols:

            all_data = self._grab_prices(
                symbol=symbol,
                from_date=from_date,
                to_date=to_date
            ) + all_data
        
        price_data_frame = pd.DataFrame(data=all_data)
        price_data_frame['date'] = pd.to_datetime(price_data_frame['date'])
    
        return price_data_frame

    def _grab_prices(self, symbol: str, from_date: date, to_date: date) -> List[Dict]:
        
        # Build the URL.
        price_url = self._build_url(symbol=symbol)

        # Calculate the limit.
        limit: timedelta = (to_date - from_date)

        # Define the parameters.
        params = {
            'fromdate': from_date.isoformat(),
            'todate': to_date.isoformat(),
            'assetclass': 'stocks',
            'limit': limit.days
        }

        # Fake the headers.
        headers = {
            'user-agent': UserAgent().edge
        }

        # Grab the historical data.
        historical_data = requests.get(
            url=price_url,
            params=params,
            headers=headers,
            verify=True
        )

        # If it's okay parse it.
        if historical_data.ok:
            historical_data = historical_data.json()
            historical_data = historical_data['data']['tradesTable']['rows']

            # Clean the data.
            for table_row in historical_data:
                table_row['symbol'] = symbol
                table_row['close'] = float(table_row['close'].replace('$',''))
                table_row['volume'] = int(table_row['volume'].replace(',',''))
                table_row['open'] = float(table_row['open'].replace('$',''))
                table_row['high'] = float(table_row['high'].replace('$',''))
                table_row['low'] = float(table_row['low'].replace('$',''))

            return historical_data

