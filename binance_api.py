# binance_api.py
from binance.client import Client
import pandas as pd
import os
import dotenv
import time

dotenv.load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

def get_klines(symbol, interval, limit):
    try:
        start_time = int(time.time() * 1000) - (limit * 60 * 1000)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit, startTime=start_time)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        print(f"Erro ao obter klines: {e}")
        return pd.DataFrame()

def place_order(symbol, side, quantity):
    try:
        # Garantir que quantity seja float com até 8 decimais
        formatted_quantity = "{:.8f}".format(float(quantity)).rstrip('0').rstrip('.')
        print(f"Enviando ordem: symbol={symbol}, side={side}, quantity={formatted_quantity}")
        order = client.create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=formatted_quantity
        )
        return order
    except Exception as e:
        print(f"Erro ao realizar pedido para {symbol}: {e}")
        raise e

def get_balance(asset):
    try:
        balance = client.get_asset_balance(asset=asset)
        return float(balance['free'])
    except Exception as e:
        print(f"Erro ao obter saldo de {asset}: {e}")
        return 0.0

def get_symbol_filters(symbol):
    try:
        info = client.get_symbol_info(symbol)
        return info['filters']
    except Exception as e:
        print(f"Erro ao obter filtros do símbolo {symbol}: {e}")
        return []