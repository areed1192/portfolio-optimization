# Portfolio Optimization in Python

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Support These Projects](#support-these-projects)

## Overview

A simple python project where we use price data from the NASDAQ website to help
optimize our portfolio of stocks using modern portfolio theory.

## Setup

Right now, the library is not hosted on **PyPi** so you will need to do a local
install on your system if you plan to use it in other scrips you use.

First, clone this repo to your local system. After you clone the repo, make sure
to run the `setup.py` file, so you can install any dependencies you may need. To
run the `setup.py` file, run the following command in your terminal.

```console
pip install -e .
```

This will install all the dependencies listed in the `setup.py` file. Once done
you can use the library wherever you want.

## Usage

Here is a simple example of using the `pyopt` library to grab the index
files for specific quarter.

```python
import pandas as pd
from pyopt.client import PriceHistory

# Define the symbols
symbols = ['AAPL', 'MSFT', 'SQ']
number_of_symbols = len(symbols)

# Initialize the client.
price_history_client = PriceHistory(symbols=['AAPL','MSFT','SQ'])

# Dump it to a CSV file.
price_history_client.price_data_frame.to_csv(
    'stock_data.csv',
    index=False
)
pprint(price_history_client.price_data_frame)
```

## Support These Projects

**Patreon:**
Help support this project and future projects by donating to my [Patreon Page](https://www.patreon.com/sigmacoding). I'm always looking to add more content for individuals like yourself, unfortuantely some of the APIs I would require me to pay monthly fees.

**YouTube:**
If you'd like to watch more of my content, feel free to visit my YouTube channel [Sigma Coding](https://www.youtube.com/c/SigmaCoding).
