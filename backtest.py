# backtest.py
def backtest(df, model):
    balance = 1000  # BRL
    position = 0  # Quantidade de BTC
    for i in range(len(df) - 1):
        features = df[["rsi", "sma", "macd", "macd_signal", "bb_upper", "bb_lower"]].iloc[i].values.reshape(1, -1)
        prediction = model.predict(features)[0]

        if prediction == 1 and balance > 0:  # Compra
            position = balance / df["close"].iloc[i]
            balance = 0
            print(f"[{df['timestamp'].iloc[i]}] BUY at {df['close'].iloc[i]}")
        elif prediction == -1 and position > 0:  # Venda
            balance = position * df["close"].iloc[i]
            position = 0
            print(f"[{df['timestamp'].iloc[i]}] SELL at {df['close'].iloc[i]}")

    # Final: vender posição restante, se houver
    if position > 0:
        balance = position * df["close"].iloc[-1]
        print(f"[{df['timestamp'].iloc[-1]}] Final SELL at {df['close'].iloc[-1]}")

    print(f"\nFinal Balance: R${balance:.2f}")