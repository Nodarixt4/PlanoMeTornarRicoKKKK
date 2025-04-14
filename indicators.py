# indicators.py
import ta

def add_indicators(df):
    # RSI e SMA
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["sma"] = ta.trend.SMAIndicator(df["close"], window=14).sma_indicator()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    # EMA - Médias móveis exponenciais
    df["ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_cross"] = df["ema_10"] - df["ema_20"]  # cruzamento de EMAs

    # ADX - Índice Direcional Médio
    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()

    # CCI - Commodity Channel Index
    df["cci"] = ta.trend.CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window=20).cci()

    # OBV - On Balance Volume
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()

    # Momentum (Rate of Change)
    df["momentum"] = ta.momentum.ROCIndicator(close=df["close"], window=10).roc()

    # ATR - Average True Range (volatilidade)
    df["atr"] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()

    return df.dropna()
