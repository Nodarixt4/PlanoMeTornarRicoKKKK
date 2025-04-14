# data_fetcher.py
from binance_api import get_klines
from config import PAIR, INTERVAL, LIMIT

def fetch_ohlcv():
    df = get_klines(PAIR, INTERVAL, LIMIT)
    return df