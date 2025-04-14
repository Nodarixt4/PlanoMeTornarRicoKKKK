# test_trading_simulation.py

import time

# Função auxiliar que calcula o tamanho mínimo do lote (já definida no seu código)
def calcular_min_lot(preco_atual, min_notional, lot_size_filters):
    min_qty = float(lot_size_filters['minQty'])
    step_size = float(lot_size_filters['stepSize'])
    quantity = min_notional / preco_atual
    if quantity < min_qty:
        quantity = min_qty
    quantity = round(quantity / step_size) * step_size
    return round(quantity, 8)

# Função que simula a execução do pedido conforme o estado e o sinal predito.
def simular_pedido(predicted_signal, in_position, brl_balance, btc_balance, min_lot_dynamic, notional_with_buffer):
    action = ""
    # Se estiver com posição aberta, só considerar venda
    if in_position:
        if predicted_signal == -1:
            if btc_balance >= min_lot_dynamic:
                action = f"VENDA executada: {min_lot_dynamic} BTC"
                in_position = False
                btc_balance -= min_lot_dynamic   # simula a saída da posição
                # Supomos que o valor de venda será o preço atual multiplicado pela quantidade
                # Nesse exemplo, vamos simular um saldo BRL incrementado:
                brl_balance += notional_with_buffer  # apenas para simulação
            else:
                action = f"Saldo BTC insuficiente (Nec: {min_lot_dynamic:.8f})"
        else:
            action = "Aguardando sinal de venda..."
    else:
        # Se não estiver em posição, considerar compra
        if predicted_signal == 1:
            if brl_balance >= notional_with_buffer:
                action = f"COMPRA executada: {min_lot_dynamic} BTC"
                in_position = True
                brl_balance -= notional_with_buffer  # reduz o saldo BRL
                btc_balance += min_lot_dynamic          # aumenta o saldo BTC
            else:
                action = f"Saldo BRL insuficiente (Nec: {notional_with_buffer:.2f})"
        else:
            action = "Aguardando sinal de compra..."
    return action, in_position, brl_balance, btc_balance

# Função de impressão do "card" para exibir o status da operação
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

# Função para simular o ambiente de trading
def simular_cenario(test_name, predicted_signal, initial_in_position, initial_brl, initial_btc, preco_atual, min_notional, lot_size_filters):
    print(f"\n>>> Teste: {test_name}")

    # Calcula o tamanho mínimo do lote e o valor notional mínimo com buffer
    min_lot_dynamic = calcular_min_lot(preco_atual, min_notional, lot_size_filters)
    notional = preco_atual * min_lot_dynamic
    notional_with_buffer = notional + 0.5

    # Simulação: exibe o status inicial e tenta executar o pedido
    print(f"Saldo inicial - BRL: {initial_brl:.2f}, BTC: {initial_btc:.8f}")
    print(f"Preço atual: {preco_atual:.2f}, min_lot_dynamic: {min_lot_dynamic}, notional_with_buffer: {notional_with_buffer:.2f}")
    action, final_position, final_brl, final_btc = simular_pedido(
        predicted_signal, 
        initial_in_position, 
        initial_brl, 
        initial_btc, 
        min_lot_dynamic, 
        notional_with_buffer
    )

    # Cria dados para o "card" e exibe-os
    card_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "price": preco_atual,
        "predicted_signal": predicted_signal,
        "probabilities": [-0.0, 0.0, 0.0] if predicted_signal == 0 else ([0.1, 0.2, 0.7] if predicted_signal == 1 else [0.7, 0.2, 0.1]),
        "position_status": "Aberta" if final_position else "Fechada",
        "action": action,
        "btc_balance": final_btc,
        "brl_balance": final_brl
    }
    print_card(card_data)
    return final_position, final_brl, final_btc

if __name__ == "__main__":
    # Parâmetros fixos para a simulação
    min_notional = 20.0
    lot_size_filters = {"minQty": "0.0001", "stepSize": "0.0001"}
    preco_atual = 100000.00  # exemplo: preço em BRL

    # Cenário 1: Sinal de COMPRA quando não há posição e saldo BRL suficiente.
    in_position, brl, btc = simular_cenario(
        test_name="Cenário de Compra",
        predicted_signal=1,
        initial_in_position=False,
        initial_brl=2000.00,    # suficiente para cobrir o notional (exemplo)
        initial_btc=0.0,
        preco_atual=preco_atual,
        min_notional=min_notional,
        lot_size_filters=lot_size_filters
    )

    # Cenário 2: Sinal de VENDA quando já há posição (BTC) e o sinal de venda é dado.
    in_position, brl, btc = simular_cenario(
        test_name="Cenário de Venda",
        predicted_signal=-1,
        initial_in_position=True,
        initial_brl=1000.00,    # saldo BRL qualquer, pois já está em posição
        initial_btc=0.05,       # exemplo de BTC comprado
        preco_atual=preco_atual,
        min_notional=min_notional,
        lot_size_filters=lot_size_filters
    )

    # Cenário 3: Sinal de AGUARDO (nenhum sinal acionado) quando o sinal for 0 ou não adequado.
    # Teste para compra: sem sinal de compra.
    in_position, brl, btc = simular_cenario(
        test_name="Cenário de Aguardando (Compra)",
        predicted_signal=0,
        initial_in_position=False,
        initial_brl=2000.00,
        initial_btc=0.0,
        preco_atual=preco_atual,
        min_notional=min_notional,
        lot_size_filters=lot_size_filters
    )
    # Teste para venda: sinal neutro mesmo tendo posição aberta.
    in_position, brl, btc = simular_cenario(
        test_name="Cenário de Aguardando (Venda)",
        predicted_signal=0,
        initial_in_position=True,
        initial_brl=1000.00,
        initial_btc=0.05,
        preco_atual=preco_atual,
        min_notional=min_notional,
        lot_size_filters=lot_size_filters
    )
