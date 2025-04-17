import time
import os
import dotenv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from binance_api import get_klines, client, place_order, get_balance, get_symbol_filters
from indicators import add_indicators
from model import train_model, predict_signal, LOOKBACK
from sklearn.metrics import classification_report
from config import PAIR, INTERVAL, LIMIT
from collections import deque

dotenv.load_dotenv()

SYMBOL = PAIR

# Statistical tracking variables
trade_history = deque(maxlen=1440)  # Store last 1440 trades for 24-hour performance
buy_count = 0
sell_count = 0
highest_buy_prob = 0.0
highest_buy_time = None
highest_sell_prob = 0.0
highest_sell_time = None
initial_portfolio_value = None
prediction_history = deque(maxlen=60)  # Store last 60 minutes of predictions for plotting

def calcular_min_lot(preco_atual, min_notional, lot_size_filters):
    min_qty = float(lot_size_filters['minQty'])
    step_size = float(lot_size_filters['stepSize'])
    quantity = min_notional / preco_atual
    if quantity < min_qty:
        quantity = min_qty
    quantity = round(quantity / step_size) * step_size
    return round(quantity, 8)

# ANSI color codes for enhanced formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_card(card_data):
    signal_color = Colors.GREEN if card_data['predicted_signal'] == 1 else Colors.RED if card_data['predicted_signal'] == -1 else Colors.YELLOW
    position_color = Colors.GREEN if card_data['position_status'] == "Aberta" else Colors.BLUE
    action_color = Colors.GREEN if "executada" in card_data['action'].lower() else Colors.RED if "erro" in card_data['action'].lower() else Colors.YELLOW
    
    card = (
        f"\n{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.END}\n"
        f"{Colors.BOLD}⏰ TIMESTAMP    :{Colors.END} {card_data['timestamp']}\n"
  f"{Colors.BOLD}💰 PREÇO       :{Colors.END} {card_data['price']:.2f} BRL\n"
        f"{Colors.BOLD}📈 PREDIÇÃO    :{Colors.END} {signal_color}{card_data['predicted_signal']}{Colors.END} "
        f"(Probabilidades: {Colors.RED}-1: {card_data['probabilities'][0]*100:.1f}%{Colors.END}, "
        f"{Colors.YELLOW}0: {card_data['probabilities'][1]*100:.1f}%{Colors.END}, "
        f"{Colors.GREEN}1: {card_data['probabilities'][2]*100:.1f}%{Colors.END})\n"
        f"{Colors.BOLD}🔄 POSIÇÃO     :{Colors.END} {position_color}{card_data['position_status']}{Colors.END}\n"
        f"{Colors.BOLD}🛠 AÇÃO        :{Colors.END} {action_color}{card_data['action']}{Colors.END}\n"
        f"{Colors.BOLD}⚖ SALDOS       :{Colors.END} BTC: {card_data['btc_balance']:.8f} | BRL: {card_data['brl_balance']:.2f}\n"
        f"{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.END}\n"
    )
    print(card)

def print_statistics():
    global trade_history, buy_count, sell_count, highest_buy_prob, highest_buy_time, highest_sell_prob, highest_sell_time, initial_portfolio_value
    current_price = float(client.get_symbol_ticker(symbol=SYMBOL)["price"])
    current_btc = get_balance("BTC")
    current_brl = get_balance("BRL")
    current_portfolio_value = current_brl + (current_btc * current_price)
    
    portfolio_change = 0.0
    if initial_portfolio_value is not None and trade_history.maxlen == len(trade_history):
        portfolio_change = current_portfolio_value - initial_portfolio_value
    
    stats = (
        f"\n{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.END}\n"
        f"{Colors.BOLD}📊 ESTATÍSTICAS DO BOT 📊{Colors.END}\n"
        f"{Colors.BOLD}🔥 Maior chance de COMPRA :{Colors.END} {highest_buy_prob*100:.1f}% às {highest_buy_time or 'N/A'}\n"
        f"{Colors.BOLD}❄ Maior chance de VENDA  :{Colors.END} {highest_sell_prob*100:.1f}% às {highest_sell_time or 'N/A'}\n"
        f"{Colors.BOLD}🛒 Compras realizadas    :{Colors.END} {buy_count}\n"
        f"{Colors.BOLD}💸 Vendas realizadas     :{Colors.END} {sell_count}\n"
        f"{Colors.BOLD}💹 Desempenho (24h)      :{Colors.END} {portfolio_change:+.2f} BRL\n"
        f"{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.END}\n"
    )
    print(stats)

def plot_market_movement(df, prediction_history):
    # Filter last 60 minutes (60 candles)
    df_last_hour = df.tail(60)
    if len(df_last_hour) < 1:
        print("Dados insuficientes para plotar o gráfico (necessário pelo menos 1 candle)")
        return
    
    timestamps = df_last_hour['timestamp']
    prices = df_last_hour['close']
    
    # Initialize lists for buy/sell markers
    buy_points_light = []
    buy_points_dark = []
    sell_points_light = []
    sell_points_dark = []
    
    # Align predictions with timestamps
    for pred in prediction_history:
        if pred['timestamp'] in df_last_hour['timestamp'].values:
            idx = df_last_hour[df_last_hour['timestamp'] == pred['timestamp']].index[0]
            price = prices[idx]
            sell_prob = pred['probabilities'][0]
            buy_prob = pred['probabilities'][2]
            
            if buy_prob > 0.25:
                buy_points_dark.append((idx, price))
            elif buy_prob > 0.10:
                buy_points_light.append((idx, price))
            if sell_prob > 0.25:
                sell_points_dark.append((idx, price))
            elif sell_prob > 0.10:
                sell_points_light.append((idx, price))
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, prices, 'b-', label='Preço de Fechamento (BTC/BRL)')
    
    # Plot buy/sell markers
    if buy_points_light:
        x, y = zip(*buy_points_light)
        plt.scatter(x, y, c='#90EE90', marker='^', s=100, label='Compra (>10%)')
    if buy_points_dark:
        x, y = zip(*buy_points_dark)
        plt.scatter(x, y, c='#006400', marker='^', s=150, label='Compra (>25%)')
    if sell_points_light:
        x, y = zip(*sell_points_light)
        plt.scatter(x, y, c='#FF6347', marker='v', s=100, label='Venda (>10%)')
    if sell_points_dark:
        x, y = zip(*sell_points_dark)
        plt.scatter(x, y, c='#8B0000', marker='v', s=150, label='Venda (>25%)')
    
    # Customize the plot
    plt.title('Movimentação do Mercado BTC/BRL (Última Hora)')
    plt.xlabel('Tempo')
    plt.ylabel('Preço (BRL)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save and close
    plt.savefig('market_movement.png')
    plt.close()

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

# Gerar os sinais com base em uma regra inicial
df['signal'] = 0
df.loc[(df['rsi'] < 40) & (df['macd'] > df['macd_signal']), 'signal'] = 1
df.loc[(df['rsi'] > 60) & (df['macd'] < df['macd_signal']), 'signal'] = -1

print("Distribuição inicial dos sinais:")
print(df['signal'].value_counts())

# Treinar o modelo inicial
model, X_test, y_test = train_model(df)
print(classification_report(y_test, model.predict(X_test), zero_division=0))
print("Bot iniciado. Esperando sinais...")

# Controle de posição
in_position = False
cycle_count = 0
first_iteration = True

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
            print_statistics()

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
                df_scaled = __import__('model').scaler.transform(df_window.astype('float64'))
                probabilities = model.predict_proba(df_scaled)[0]
                predicted_signal = model.predict(df_scaled)[0]
                print("Features de previsão (primeiros 10 valores):", df_window.iloc[0].values[:10])
                print("Probabilidades (ordem -1, 0, 1):", probabilities)

                # Update statistical tracking
                if probabilities[2] > highest_buy_prob:
                    highest_buy_prob = probabilities[2]
                    highest_buy_time = str(timestamp)
                if probabilities[0] > highest_sell_prob:
                    highest_sell_prob = probabilities[0]
                    highest_sell_time = str(timestamp)

            price = float(client.get_symbol_ticker(symbol=SYMBOL)["price"])
            btc_balance = get_balance("BTC")
            brl_balance = get_balance("BRL")

            # Update trade history for portfolio performance
            portfolio_value = brl_balance + (btc_balance * price)
            if initial_portfolio_value is None:
                initial_portfolio_value = portfolio_value
            trade_history.append(portfolio_value)

            # Store prediction for plotting
            prediction_history.append({
                'timestamp': timestamp,
                'probabilities': probabilities,
                'predicted_signal': predicted_signal
            })

            # Generate and save the plot
            plot_market_movement(df, prediction_history)

            min_lot_dynamic = calcular_min_lot(price, min_notional, lot_size_filters)
            notional = price * min_lot_dynamic
            notional_with_buffer = notional + 0.5

            action = ""
            if not in_position and predicted_signal == 1:
                if brl_balance >= notional_with_buffer:
                    try:
                        place_order(SYMBOL, "BUY", min_lot_dynamic)
                        in_position = True
                        buy_count += 1
                        action = f"COMPRA executada: {min_lot_dynamic} BTC"
                        os.environ["ULTIMA_COMPRA"] = str(price)
                    except Exception as e:
                        action = f"Erro na COMPRA: {e}"
                else:
                    action = f"Saldo BRL insuficiente (Nec: {notional_with_buffer:.2f})"
            elif in_position and predicted_signal == -1:
                if btc_balance >= min_lot_dynamic:
                    buy_price = float(os.getenv("ULTIMA_COMPRA", "0"))
                    if buy_price > 0:
                        lucro_percent = ((price - buy_price) / buy_price) * 100
                        if lucro_percent > 0.5:
                            try:
                                place_order(SYMBOL, "SELL", min_lot_dynamic)
                                in_position = False
                                sell_count += 1
                                action = f"VENDA executada: {min_lot_dynamic} BTC com lucro de {lucro_percent:.2f}%"
                                os.environ["ULTIMA_COMPRA"] = "0"
                            except Exception as e:
                                action = f"Erro na VENDA: {e}"
                        else:
                            action = f"Lucro abaixo de 0.5% ({lucro_percent:.2f}%), venda abortada"
                    else:
                        action = "Preço de compra anterior não disponível, venda não realizada"
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