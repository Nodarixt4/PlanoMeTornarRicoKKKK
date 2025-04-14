# main.py
import time
import os
import dotenv
import pandas as pd
from binance_api import get_klines, client, place_order, get_balance, get_symbol_filters
from indicators import add_indicators
from model import train_model, predict_signal, LOOKBACK
from sklearn.metrics import classification_report
from config import PAIR, INTERVAL, LIMIT

dotenv.load_dotenv()

SYMBOL = PAIR

def calcular_min_lot(preco_atual, min_notional, lot_size_filters):
    min_qty = float(lot_size_filters['minQty'])
    step_size = float(lot_size_filters['stepSize'])
    quantity = min_notional / preco_atual
    if quantity < min_qty:
        quantity = min_qty
    quantity = round(quantity / step_size) * step_size
    return round(quantity, 8)

# Obter filtros do símbolo
symbol_filters = get_symbol_filters(SYMBOL)
min_notional = 20.0
lot_size_filters = None
for f in symbol_filters:
    if f['filterType'] == 'MIN_NOTIONAL':
        min_notional = float(f['minNotional'])
    if f['filterType'] == 'LOT_SIZE':
        lot_size_filters = f

if not lot_size_filters:
    raise ValueError("Filtro LOT_SIZE não encontrado para o símbolo")

print(f">> Mínimo NOTIONAL exigido pela Binance: {min_notional} BRL")
print(f">> LOT_SIZE: minQty={lot_size_filters['minQty']}, stepSize={lot_size_filters['stepSize']}")

# Carregar os candles e adicionar indicadores
df = get_klines(SYMBOL, INTERVAL, LIMIT)
df = add_indicators(df)

# Gerar os sinais com base em uma regra inicial (pode ser atualizado conforme necessário)
df['signal'] = 0
df.loc[(df['rsi'] < 40) & (df['macd'] > df['macd_signal']), 'signal'] = 1
df.loc[(df['rsi'] > 60) & (df['macd'] < df['macd_signal']), 'signal'] = -1

print("Distribuição inicial dos sinais:")
print(df['signal'].value_counts())

# Treinar o modelo inicial
model, X_test, y_test = train_model(df)
print(classification_report(y_test, model.predict(X_test), zero_division=0))
print("Bot iniciado. Esperando sinais...")

# Controle de posição: False = sem posição, True = posição aberta
in_position = False
cycle_count = 0
first_iteration = True  # Para forçar sinal neutro na primeira iteração

def print_card(card_data):
    card = (
        "\n" + "="*50 + "\n" +
        f"TIMESTAMP    : {card_data['timestamp']}\n" +
        f"PREÇO        : {card_data['price']:.2f} BRL\n" +
        f"PREDIÇÃO     : {card_data['predicted_signal']} (Probabilidades: " +
            f"-1: {card_data['probabilities'][0]*100:.1f}%, " +
            f"0: {card_data['probabilities'][1]*100:.1f}%, " +
            f"1: {card_data['probabilities'][2]*100:.1f}%)\n" +
        f"POSIÇÃO      : {card_data['position_status']}\n" +
        f"AÇÃO         : {card_data['action']}\n" +
        f"SALDOS       : BTC: {card_data['btc_balance']:.8f} | BRL: {card_data['brl_balance']:.2f}\n" +
        "="*50 + "\n"
    )
    print(card)

while True:
    try:
        cycle_count += 1
        
        if cycle_count % 10 == 0:
            print(">>> Retreinando o modelo...")
            df = get_klines(SYMBOL, INTERVAL, LIMIT)
            df = add_indicators(df)
            df['signal'] = 0
            df.loc[(df['rsi'] < 40) & (df['macd'] > df['macd_signal']), 'signal'] = 1
            df.loc[(df['rsi'] > 60) & (df['macd'] < df['macd_signal']), 'signal'] = -1
            print(">>> Distribuição dos sinais no retreinamento:", df['signal'].value_counts().to_dict())
            model, X_test, y_test = train_model(df)
            print(classification_report(y_test, model.predict(X_test), zero_division=0))
            print(">>> Modelo retreinado com sucesso.")

        df = get_klines(SYMBOL, INTERVAL, LIMIT)
        df = add_indicators(df)

        if len(df) >= LOOKBACK:
            current_data = df.iloc[-LOOKBACK:]
            timestamp = current_data['timestamp'].iloc[-1]

            if first_iteration:
                predicted_signal = 0
                first_iteration = False
                probabilities = [0.0, 1.0, 0.0]
            else:
                required_columns = ["rsi", "sma", "macd", "macd_signal", "bb_upper", "bb_lower", "volume"]
                df_window_input = current_data.tail(LOOKBACK)[required_columns + ['close']]
                window = []
                for col in required_columns:
                    window.extend(df_window_input[col].values)
                delta = df_window_input['close'].iloc[-1] - df_window_input['close'].iloc[0]
                window.append(delta)
                feature_names = [f"{col}_t-{LOOKBACK-1-j}" for col in required_columns for j in range(LOOKBACK)]
                feature_names.append("close_delta")
                df_window = pd.DataFrame([window], columns=feature_names)
                df_scaled =  __import__('model').scaler.transform(df_window.astype('float64'))
                probabilities = model.predict_proba(df_scaled)[0]
                predicted_signal = model.predict(df_scaled)[0]
                print("Features de previsão (primeiros 10 valores):", df_window.iloc[0].values[:10])
                print("Probabilidades (ordem -1, 0, 1):", probabilities)

            price = float(client.get_symbol_ticker(symbol=SYMBOL)["price"])
            btc_balance = get_balance("BTC")
            brl_balance = get_balance("BRL")

            min_lot_dynamic = calcular_min_lot(price, min_notional, lot_size_filters)
            notional = price * min_lot_dynamic
            notional_with_buffer = notional + 0.5

            action = ""
            if not in_position and predicted_signal == 1:
                if brl_balance >= notional_with_buffer:
                    try:
                        place_order(SYMBOL, "BUY", min_lot_dynamic)
                        in_position = True
                        action = f"COMPRA executada: {min_lot_dynamic} BTC"
                    except Exception as e:
                        action = f"Erro na COMPRA: {e}"
                else:
                    action = f"Saldo BRL insuficiente (Nec: {notional_with_buffer:.2f})"
            elif in_position and predicted_signal == -1:
                if btc_balance >= min_lot_dynamic:
                    try:
                        place_order(SYMBOL, "SELL", min_lot_dynamic)
                        in_position = False
                        action = f"VENDA executada: {min_lot_dynamic} BTC"
                    except Exception as e:
                        action = f"Erro na VENDA: {e}"
                else:
                    action = f"Saldo BTC insuficiente (Nec: {min_lot_dynamic:.8f})"
            else:
                action = "Nenhuma operação realizada."

            card_data = {
                "timestamp": str(timestamp),
                "price": price,
                "predicted_signal": predicted_signal,
                "probabilities": probabilities,
                "position_status": "Aberta" if in_position else "Fechada",
                "action": action,
                "btc_balance": get_balance("BTC"),
                "brl_balance": get_balance("BRL")
            }
            print_card(card_data)
        else:
            print(f"Dados insuficientes: {len(df)} candles disponíveis, necessário {LOOKBACK}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

    time.sleep(60)
