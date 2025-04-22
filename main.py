#teste

import time
import os
import dotenv
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from binance_api import get_klines, client, place_order, get_balance, get_symbol_filters
from indicators import add_indicators
from model import train_model, predict_signal, LOOKBACK
from sklearn.metrics import classification_report
from config import PAIR, INTERVAL, LIMIT
from collections import deque

dotenv.load_dotenv()

SYMBOL = PAIR
VALOR_COMPRA = 40.0  # Fixed purchase value in BRL

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
last_bought_quantity = 0.0  # Store the last bought quantity after fees

def calcular_quantidade_compra(preco_atual, valor_compra, lot_size_filters):
    step_size = float(lot_size_filters['stepSize'])
    quantity = valor_compra / preco_atual
    quantity = round(quantity / step_size) * step_size
    # Increase by 2% to account for fees
    quantity_with_buffer = quantity * 1.02
    quantity_with_buffer = round(quantity_with_buffer / step_size) * step_size
    return round(quantity_with_buffer, 8)

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
    try:
        df_last_hour = df.tail(60)
        if len(df_last_hour) < 1:
            print(f"[ERRO] Dados insuficientes para plotar: {len(df_last_hour)} candles disponíveis")
            return
        
        timestamps = df_last_hour['timestamp']
        prices = df_last_hour['close']
        
        buy_points_light = []
        buy_points_dark = []
        sell_points_light = []
        sell_points_dark = []
        
        df_timestamps_naive = df_last_hour['timestamp']
        for pred in prediction_history:
            pred_timestamp_naive = pred['timestamp']
            if pred_timestamp_naive in df_timestamps_naive.values:
                timestamp = pred['timestamp']
                idx = df_timestamps_naive[df_timestamps_naive == pred_timestamp_naive].index[0]
                price = prices[idx]
                sell_prob = pred['probabilities'][0]
                buy_prob = pred['probabilities'][2]
                
                if buy_prob > 0.01:
                    buy_points_dark.append((timestamp, price))
                elif buy_prob > 0.005:
                    buy_points_light.append((timestamp, price))
                if sell_prob > 0.01:
                    sell_points_dark.append((timestamp, price))
                elif sell_prob > 0.005:
                    sell_points_light.append((timestamp, price))
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, prices, 'b-', label='Preço de Fechamento (BTC/BRL)')
        
        if buy_points_light:
            x, y = zip(*buy_points_light)
            plt.scatter(x, y, c='#90EE90', marker='^', s=100, label='Compra (>0.5%)')
        if buy_points_dark:
            x, y = zip(*buy_points_dark)
            plt.scatter(x, y, c='#006400', marker='^', s=150, label='Compra (>1%)')
        if sell_points_light:
            x, y = zip(*sell_points_light)
            plt.scatter(x, y, c='#FF6347', marker='v', s=100, label='Venda (>0.5%)')
        if sell_points_dark:
            x, y = zip(*sell_points_dark)
            plt.scatter(x, y, c='#8B0000', marker='v', s=150, label='Venda (>1%)')
        
        plt.title('Movimentação do Mercado BTC/BRL (Última Hora)')
        plt.xlabel('Tempo')
        plt.ylabel('Preço (BRL)')
        plt.grid(True)
        plt.legend()
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.xticks(rotation=45)
        
        plt.xlim(timestamps.iloc[0], timestamps.iloc[-1])
        
        plt.tight_layout()
        
        try:
            plt.savefig('market_movement.png')
            print(f"[INFO] Gráfico atualizado: market_movement.png às {df_last_hour['timestamp'].iloc[-1]}")
        except Exception as e:
            print(f"[ERRO] Falha ao salvar market_movement.png: {e}")
            fallback_filename = f'market_movement_{df_last_hour["timestamp"].iloc[-1].strftime("%Y%m%d_%H%M%S")}.png'
            try:
                plt.savefig(fallback_filename)
                print(f"[INFO] Gráfico salvo como fallback: {fallback_filename}")
            except Exception as e2:
                print(f"[ERRO] Falha ao salvar fallback {fallback_filename}: {e2}")
        
        plt.close()
    except Exception as e:
        print(f"[ERRO] Erro no plot_market_movement: {e}")

# Fetch symbol filters
symbol_filters = get_symbol_filters(SYMBOL)
lot_size_filters = None
for f in symbol_filters:
    if f['filterType'] == 'LOT_SIZE':
        lot_size_filters = f

if not lot_size_filters:
    raise ValueError("Filtro LOT_SIZE não encontrado para o símbolo")

print(f">> LOT_SIZE: minQty={lot_size_filters['minQty']}, stepSize={lot_size_filters['stepSize']}")

# Load initial data and add indicators
df = get_klines(SYMBOL, INTERVAL, LIMIT)
df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT+4').dt.tz_localize(None)
df = add_indicators(df)

# Generate initial signals
df['signal'] = 0
df.loc[(df['rsi'] < 40) & (df['macd'] > df['macd_signal']), 'signal'] = 1
df.loc[(df['rsi'] > 60) & (df['macd'] < df['macd_signal']), 'signal'] = -1

print("Distribuição inicial dos sinais:")
print(df['signal'].value_counts())

# Train the initial model
model, X_test, y_test = train_model(df)
print(classification_report(y_test, model.predict(X_test), zero_division=0))
print("Bot iniciado. Esperando sinais...")

# Position control
in_position = False
cycle_count = 0
first_iteration = True
buy_price = 0.0  # To track purchase price for profit calculation

while True:
    try:
        cycle_count += 1
        print(f"[DEBUG] Início do ciclo {cycle_count} às {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if cycle_count % 10 == 0:
            print(">>> Retreinando o modelo...")
            df = get_klines(SYMBOL, INTERVAL, LIMIT)
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT+4').dt.tz_localize(None)
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
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT+4').dt.tz_localize(None)
        df = add_indicators(df)
        print(f"[DEBUG] Dados obtidos: {len(df)} candles")

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

                if probabilities[2] > highest_buy_prob:
                    highest_buy_prob = probabilities[2]
                    highest_buy_time = str(timestamp)
                if probabilities[0] > highest_sell_prob:
                    highest_sell_prob = probabilities[0]
                    highest_sell_time = str(timestamp)

            price = float(client.get_symbol_ticker(symbol=SYMBOL)["price"])
            btc_balance = get_balance("BTC")
            brl_balance = get_balance("BRL")

            portfolio_value = brl_balance + (btc_balance * price)
            if initial_portfolio_value is None:
                initial_portfolio_value = portfolio_value
            trade_history.append(portfolio_value)

            prediction_history.append({
                'timestamp': timestamp,
                'probabilities': probabilities,
                'predicted_signal': predicted_signal
            })

            print("[DEBUG] Chamando plot_market_movement")
            plot_market_movement(df, prediction_history)

            action = ""
            if not in_position and predicted_signal == 1:
                quantity_to_buy = calcular_quantidade_compra(price, VALOR_COMPRA, lot_size_filters)
                notional = price * quantity_to_buy
                if brl_balance >= notional * 1.02:  # Account for fees
                    try:
                        place_order(SYMBOL, "BUY", quantity_to_buy)
                        in_position = True
                        buy_count += 1
                        buy_price = price
                        last_bought_quantity = get_balance("BTC")
                        action = f"COMPRA executada: {quantity_to_buy} BTC, quantidade após taxas: {last_bought_quantity:.8f} BTC"
                    except Exception as e:
                        action = f"Erro na COMPRA: {e}"
                else:
                    action = f"Saldo BRL insuficiente (Nec: {notional * 1.02:.2f})"
            elif in_position:
                lucro_percent = ((price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
                if predicted_signal == -1 or lucro_percent > 1.0:
                    sell_quantity = btc_balance
                    if sell_quantity > 0:
                        try:
                            place_order(SYMBOL, "SELL", sell_quantity)
                            in_position = False
                            sell_count += 1
                            action = f"VENDA executada: {sell_quantity} BTC"
                            if lucro_percent > 1.0:
                                action += f" com lucro de {lucro_percent:.2f}%"
                            buy_price = 0.0
                            last_bought_quantity = 0.0
                        except Exception as e:
                            action = f"Erro na VENDA: {e}"
                    else:
                        action = "Nenhuma quantidade de BTC para vender"
                else:
                    action = f"Aguardando venda: lucro {lucro_percent:.2f}% (<1%) e sem sinal de venda"
            else:
                action = "Nenhuma operação realizada (aguardando sinal de compra)"

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
        
        print(f"[DEBUG] Fim do ciclo {cycle_count} às {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"[ERRO] Erro inesperado no loop principal: {e}")

    time.sleep(60)