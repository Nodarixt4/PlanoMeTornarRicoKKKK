# model.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Importa o SMOTE para balancear as classes (assegure-se de ter o pacote imbalanced-learn instalado)
from imblearn.over_sampling import SMOTE

scaler = StandardScaler()
# Aumentamos o lookback para considerar mais candles (contexto histórico ampliado)
LOOKBACK = 20

def create_windowed_features(df, columns, lookback=LOOKBACK):
    """
    Cria uma janela de features a partir das colunas indicadas e adiciona
    a variação (delta) do preço de fechamento ao longo da janela.
    """
    windowed_data = []
    for i in range(lookback, len(df)):
        window = []
        for col in columns:
            window.extend(df[col].iloc[i-lookback:i].values)
        # Adiciona o delta (diferença) entre o último e o primeiro valor do fechamento na janela
        delta = df['close'].iloc[i] - df['close'].iloc[i - lookback]
        window.append(delta)
        windowed_data.append(window)
    
    # Gera nomes para as features dinamicamente
    feature_names = []
    for col in columns:
        for j in range(lookback):
            feature_names.append(f"{col}_t-{lookback-1-j}")
    feature_names.append("close_delta")
    
    return pd.DataFrame(windowed_data, columns=feature_names, index=df.index[lookback:])

def train_model(df):
    """
    Treina um modelo de RandomForest utilizando um conjunto mais rico de features.
    Esta versão utiliza SMOTE para balancear os sinais de compra (1), espera (0)
    e venda (-1).
    """
    # Incluímos as features necessárias, adicionando também volume
    required_columns = ["rsi", "sma", "macd", "macd_signal", "bb_upper", "bb_lower", "volume"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coluna {col} está faltando no DataFrame")
    
    df.dropna(subset=required_columns + ["signal"], inplace=True)

    # Cria as features em janela e ajusta o target para cortar os primeiros LOOKBACK registros
    X = create_windowed_features(df, required_columns)
    y = df["signal"].iloc[LOOKBACK:]
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Dados insuficientes após criar a janela de features")

    from collections import Counter

    # Conta quantas amostras existem em cada classe
    counter = Counter(y)
    minority_class_count = min(counter.values())

    try:
        if minority_class_count > 5:
            smote = SMOTE(k_neighbors=5, random_state=42)
        else:
            smote = SMOTE(k_neighbors=minority_class_count - 1, random_state=42)
            print(f"[AVISO] Poucas amostras na classe minoritária. Usando k_neighbors={minority_class_count - 1} no SMOTE.")

        X_resampled, y_resampled = smote.fit_resample(X, y)

    except ValueError as e:
        print(f"[ERRO] SMOTE falhou: {e}")
        print("[INFO] Continuando sem SMOTE.")
        X_resampled, y_resampled = X, y

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, shuffle=True)

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    global scaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Configuração aprimorada do modelo com mais árvores e profundidade para capacidade de discriminação
    # O balanceamento das classes é agora feito pelo SMOTE, mas mantemos os pesos para reforçar ainda mais a sensibilidade
    class_weights = {-1: 2.5, 0: 1.0, 1: 2.5}
    model = RandomForestClassifier(
        n_estimators=1500,
        max_depth=25,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    predictions, counts = np.unique(model.predict(X_test), return_counts=True)
    print("Previsões no conjunto de teste:", dict(zip(predictions, counts)))

    print("Distribuição de classes (após SMOTE):", pd.Series(y_resampled).value_counts())
    print("Médias e desvios das features no conjunto de treino:")
    print(pd.DataFrame(X_train).describe().loc[['mean', 'std']])


    return model, X_test, y_test

def predict_signal(model, current_data):
    """
    Realiza a previsão do sinal utilizando a mesma estratégia de janela aplicada no treinamento,
    incluindo as features adicionais e a variação (delta) do fechamento.
    """
    required_columns = ["rsi", "sma", "macd", "macd_signal", "bb_upper", "bb_lower", "volume"]
    
    if len(current_data) < LOOKBACK:
        raise ValueError(f"São necessários pelo menos {LOOKBACK} candles para previsão")
    
    # Seleciona as features necessárias e inclui a coluna close para calcular o delta
    df_window_input = current_data.tail(LOOKBACK)[required_columns + ['close']]
    
    window = []
    for col in required_columns:
        window.extend(df_window_input[col].values)
    delta = df_window_input['close'].iloc[-1] - df_window_input['close'].iloc[0]
    window.append(delta)
    
    # Cria os nomes das features de acordo com a janela utilizada
    feature_names = [f"{col}_t-{LOOKBACK-1-j}" for col in required_columns for j in range(LOOKBACK)]
    feature_names.append("close_delta")
    
    df_window = pd.DataFrame([window], columns=feature_names)
    
    print("Features de previsão (primeiros 10 valores):", df_window.iloc[0].values[:10])
    df_scaled = scaler.transform(df_window.astype(np.float64))
    probs = model.predict_proba(df_scaled)[0]
    print("Probabilidades (ordem -1, 0, 1):", probs)
    
    return model.predict(df_scaled)[0]
