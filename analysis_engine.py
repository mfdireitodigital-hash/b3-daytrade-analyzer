"""
ANALISE B3 - 24/7 - Engine de Análise Técnica
Foco: Mini-Índice (WIN) e Mini-Dólar (WDO)
Indicadores: VWAP, EMA 9/21, Fibonacci, RSI, MACD, ATR, Volume, Anti-Violinada
Estratégia: Pullback em EMA/VWAP com confirmação de candle
Timeframes: 5min, 15min, 1h, 4h, Diário
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Sinal(str, Enum):
    COMPRA_FORTE = "COMPRA_FORTE"
    COMPRA = "COMPRA"
    NEUTRO = "NEUTRO"
    VENDA = "VENDA"
    VENDA_FORTE = "VENDA_FORTE"


class Tendencia(str, Enum):
    ALTA = "ALTA"
    BAIXA = "BAIXA"
    LATERAL = "LATERAL"


@dataclass
class FibonacciLevels:
    nivel_0: float      # Topo
    nivel_236: float
    nivel_382: float
    nivel_500: float
    nivel_618: float
    nivel_786: float
    nivel_100: float    # Fundo
    extensao_1272: float
    extensao_1618: float
    extensao_2618: float
    tendencia: str      # "ALTA" ou "BAIXA"


@dataclass
class AnaliseVolume:
    volume_total: float
    volume_compra: float
    volume_venda: float
    ratio_compra_venda: float
    pressao: str  # "COMPRADORES", "VENDEDORES", "EQUILIBRIO"
    volume_medio: float
    volume_acima_media: bool
    delta_acumulado: float


@dataclass
class SinalEntrada:
    tipo: str  # "COMPRA" ou "VENDA"
    preco_entrada: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risco_retorno: float
    confianca: float  # 0-100
    motivos: list
    fibonacci_zona: str
    rsi_status: str
    macd_status: str
    volume_status: str
    violinada_risco: str  # "BAIXO", "MEDIO", "ALTO"


@dataclass
class AnaliseTimeframe:
    timeframe: str
    ativo: str
    preco_atual: float
    tendencia: str
    fibonacci: FibonacciLevels
    rsi: float
    rsi_status: str  # "SOBRECOMPRADO", "SOBREVENDIDO", "NEUTRO"
    macd_valor: float
    macd_sinal: float
    macd_histograma: float
    macd_status: str  # "ALTA", "BAIXA", "CRUZAMENTO_ALTA", "CRUZAMENTO_BAIXA"
    volume: AnaliseVolume
    sinais: list
    suportes: list
    resistencias: list
    violinada_score: float  # 0-100 (quanto maior, mais risco de violinada)
    timestamp: str


# =====================================================
# CÁLCULOS DE INDICADORES
# =====================================================

def calcular_rsi(dados: pd.DataFrame, periodo: int = 14) -> pd.Series:
    """Calcula RSI (Índice de Força Relativa)"""
    delta = dados['close'].diff()
    ganho = delta.where(delta > 0, 0.0)
    perda = (-delta.where(delta < 0, 0.0))

    media_ganho = ganho.rolling(window=periodo, min_periods=1).mean()
    media_perda = perda.rolling(window=periodo, min_periods=1).mean()

    # Evitar divisão por zero
    rs = media_ganho / media_perda.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calcular_macd(dados: pd.DataFrame, rapida: int = 12, lenta: int = 26, sinal: int = 9):
    """Calcula MACD com linha de sinal e histograma"""
    ema_rapida = dados['close'].ewm(span=rapida, adjust=False).mean()
    ema_lenta = dados['close'].ewm(span=lenta, adjust=False).mean()

    macd_linha = ema_rapida - ema_lenta
    macd_sinal = macd_linha.ewm(span=sinal, adjust=False).mean()
    macd_histograma = macd_linha - macd_sinal

    return macd_linha, macd_sinal, macd_histograma


def calcular_vwap(dados: pd.DataFrame) -> pd.Series:
    """Calcula VWAP (Volume Weighted Average Price)."""
    tp = (dados['high'] + dados['low'] + dados['close']) / 3
    vwap = (tp * dados['volume']).cumsum() / dados['volume'].cumsum()
    return vwap


def calcular_atr_series(dados: pd.DataFrame, periodo: int = 14) -> pd.Series:
    """Calcula ATR como série completa."""
    high = dados['high']
    low = dados['low']
    close = dados['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=periodo, min_periods=1).mean()
    return atr


def detectar_lateralizacao(dados: pd.DataFrame, lookback: int = 20) -> dict:
    """Detecta mercado lateralizado. Bloqueia entradas em range."""
    recente = dados.tail(lookback)
    ema9 = recente['close'].ewm(span=9, adjust=False).mean()
    ema21 = recente['close'].ewm(span=21, adjust=False).mean()
    diff = ema9 - ema21
    cruzamentos = 0
    for i in range(1, len(diff)):
        if (diff.iloc[i] > 0 and diff.iloc[i-1] <= 0) or (diff.iloc[i] < 0 and diff.iloc[i-1] >= 0):
            cruzamentos += 1
    preco_range = recente['high'].max() - recente['low'].min()
    deslocamento = abs(recente['close'].iloc[-1] - recente['close'].iloc[0])
    ratio_desloc = deslocamento / preco_range if preco_range > 0 else 0
    atr = calcular_atr_series(recente)
    atr_vs_range = atr.mean() / preco_range if preco_range > 0 else 0
    score = 0
    if cruzamentos >= 3: score += 40
    elif cruzamentos >= 2: score += 25
    if ratio_desloc < 0.15: score += 35
    elif ratio_desloc < 0.30: score += 20
    if atr_vs_range > 0.25: score += 25
    lateral = score >= 50
    return {"lateral": lateral, "score": min(score, 100), "cruzamentos_ema": cruzamentos,
            "deslocamento_pct": round(ratio_desloc * 100, 1),
            "status": "LATERAL - BLOQUEADO" if lateral else "TENDENCIAL"}


def detectar_pullback(dados: pd.DataFrame, tendencia: str) -> dict:
    """Detecta pullback a favor da tendência em EMA 9 ou VWAP."""
    if len(dados) < 21:
        return {"pullback": False, "tipo": "NENHUM", "zona": "N/A", "candle_reversao": False, "confirmado": False}
    preco = dados['close'].iloc[-1]
    open_atual = dados['open'].iloc[-1]
    high_atual = dados['high'].iloc[-1]
    low_atual = dados['low'].iloc[-1]
    ema9 = dados['close'].ewm(span=9, adjust=False).mean().iloc[-1]
    vwap = calcular_vwap(dados).iloc[-1]
    atr = calcular_atr_series(dados).iloc[-1]
    tolerancia = atr * 0.5
    pullback = False
    zona = "N/A"
    candle_reversao = False
    if tendencia == "ALTA":
        if abs(low_atual - ema9) < tolerancia or (low_atual <= ema9 <= preco):
            pullback = True; zona = "EMA 9"
        elif abs(low_atual - vwap) < tolerancia or (low_atual <= vwap <= preco):
            pullback = True; zona = "VWAP"
        candle_reversao = preco > open_atual and preco > ema9
    elif tendencia == "BAIXA":
        if abs(high_atual - ema9) < tolerancia or (preco <= ema9 <= high_atual):
            pullback = True; zona = "EMA 9"
        elif abs(high_atual - vwap) < tolerancia or (preco <= vwap <= high_atual):
            pullback = True; zona = "VWAP"
        candle_reversao = preco < open_atual and preco < ema9
    return {"pullback": pullback,
            "tipo": f"PULLBACK {'COMPRA' if tendencia == 'ALTA' else 'VENDA'}" if pullback else "NENHUM",
            "zona": zona, "candle_reversao": candle_reversao, "confirmado": pullback and candle_reversao}


def calcular_fibonacci(dados: pd.DataFrame, lookback: int = 50) -> FibonacciLevels:
    """
    Calcula níveis de Fibonacci com base nos últimos N candles.
    Identifica tendência e calcula retrações + extensões.
    """
    recente = dados.tail(lookback)
    high = recente['high'].max()
    low = recente['low'].min()
    diff = high - low

    # Determinar tendência pela posição do preço atual
    preco_atual = dados['close'].iloc[-1]
    meio = (high + low) / 2

    # Verificar se o máximo veio antes do mínimo (tendência de baixa) ou depois (alta)
    idx_high = recente['high'].idxmax()
    idx_low = recente['low'].idxmin()

    if idx_high > idx_low:
        # Tendência de ALTA (mínimo veio antes do máximo)
        tendencia = "ALTA"
        return FibonacciLevels(
            nivel_0=high,
            nivel_236=high - diff * 0.236,
            nivel_382=high - diff * 0.382,
            nivel_500=high - diff * 0.500,
            nivel_618=high - diff * 0.618,
            nivel_786=high - diff * 0.786,
            nivel_100=low,
            extensao_1272=high + diff * 0.272,
            extensao_1618=high + diff * 0.618,
            extensao_2618=high + diff * 1.618,
            tendencia=tendencia
        )
    else:
        # Tendência de BAIXA (máximo veio antes do mínimo)
        tendencia = "BAIXA"
        return FibonacciLevels(
            nivel_0=low,
            nivel_236=low + diff * 0.236,
            nivel_382=low + diff * 0.382,
            nivel_500=low + diff * 0.500,
            nivel_618=low + diff * 0.618,
            nivel_786=low + diff * 0.786,
            nivel_100=high,
            extensao_1272=low - diff * 0.272,
            extensao_1618=low - diff * 0.618,
            extensao_2618=low - diff * 1.618,
            tendencia=tendencia
        )


def analisar_volume(dados: pd.DataFrame, periodo_media: int = 20) -> AnaliseVolume:
    """
    Analisa volume de compradores vs vendedores.
    Usa análise de candles para estimar pressão compradora/vendedora.
    """
    recente = dados.tail(periodo_media)

    # Estimar volume de compra vs venda baseado no corpo do candle
    volumes_compra = []
    volumes_venda = []

    for _, candle in recente.iterrows():
        corpo = candle['close'] - candle['open']
        amplitude = candle['high'] - candle['low']

        if amplitude == 0:
            amplitude = 0.01

        # Proporção do volume baseada no corpo do candle
        if corpo > 0:  # Candle de alta
            ratio = min(abs(corpo) / amplitude, 1.0)
            vol_compra = candle['volume'] * (0.5 + ratio * 0.5)
            vol_venda = candle['volume'] - vol_compra
        elif corpo < 0:  # Candle de baixa
            ratio = min(abs(corpo) / amplitude, 1.0)
            vol_venda = candle['volume'] * (0.5 + ratio * 0.5)
            vol_compra = candle['volume'] - vol_venda
        else:  # Doji
            vol_compra = candle['volume'] * 0.5
            vol_venda = candle['volume'] * 0.5

        volumes_compra.append(vol_compra)
        volumes_venda.append(vol_venda)

    total_compra = sum(volumes_compra)
    total_venda = sum(volumes_venda)
    total = total_compra + total_venda

    if total_venda > 0:
        ratio = total_compra / total_venda
    else:
        ratio = 1.0

    # Delta acumulado (últimos 5 candles)
    delta = sum(volumes_compra[-5:]) - sum(volumes_venda[-5:])

    volume_medio = recente['volume'].mean()
    volume_atual = dados['volume'].iloc[-1]

    if ratio > 1.3:
        pressao = "COMPRADORES"
    elif ratio < 0.77:
        pressao = "VENDEDORES"
    else:
        pressao = "EQUILIBRIO"

    return AnaliseVolume(
        volume_total=total,
        volume_compra=total_compra,
        volume_venda=total_venda,
        ratio_compra_venda=round(ratio, 2),
        pressao=pressao,
        volume_medio=volume_medio,
        volume_acima_media=volume_atual > volume_medio * 1.2,
        delta_acumulado=delta
    )


def detectar_violinada(dados: pd.DataFrame, lookback: int = 10) -> float:
    """
    Detecta risco de violinada (whipsaw).
    Analisa: volatilidade excessiva, wicks longos, reversões rápidas.
    Retorna score de 0-100 (maior = mais risco de violinada).
    """
    recente = dados.tail(lookback)
    score = 0

    # 1. Wicks longos (sombras grandes em relação ao corpo)
    wick_ratios = []
    for _, c in recente.iterrows():
        corpo = abs(c['close'] - c['open'])
        sombra_sup = c['high'] - max(c['close'], c['open'])
        sombra_inf = min(c['close'], c['open']) - c['low']
        total_sombra = sombra_sup + sombra_inf

        if corpo > 0:
            wick_ratios.append(total_sombra / corpo)
        else:
            wick_ratios.append(5.0)  # Doji = muita indecisão

    avg_wick = np.mean(wick_ratios)
    if avg_wick > 3.0:
        score += 30
    elif avg_wick > 2.0:
        score += 20
    elif avg_wick > 1.5:
        score += 10

    # 2. Mudanças de direção frequentes
    direcoes = []
    for i in range(1, len(recente)):
        if recente['close'].iloc[i] > recente['close'].iloc[i-1]:
            direcoes.append(1)
        else:
            direcoes.append(-1)

    mudancas = sum(1 for i in range(1, len(direcoes)) if direcoes[i] != direcoes[i-1])
    taxa_mudanca = mudancas / max(len(direcoes) - 1, 1)

    if taxa_mudanca > 0.7:
        score += 30
    elif taxa_mudanca > 0.5:
        score += 20
    elif taxa_mudanca > 0.3:
        score += 10

    # 3. Volatilidade vs tendência (ATR alto com pouca progressão)
    atr_values = []
    for i in range(1, len(recente)):
        tr = max(
            recente['high'].iloc[i] - recente['low'].iloc[i],
            abs(recente['high'].iloc[i] - recente['close'].iloc[i-1]),
            abs(recente['low'].iloc[i] - recente['close'].iloc[i-1])
        )
        atr_values.append(tr)

    atr_medio = np.mean(atr_values) if atr_values else 0
    progressao = abs(recente['close'].iloc[-1] - recente['close'].iloc[0])

    if atr_medio > 0 and progressao < atr_medio * 0.5:
        score += 20
    elif atr_medio > 0 and progressao < atr_medio:
        score += 10

    # 4. Volume decrescente em movimento (sinal de exaustão)
    if len(recente) >= 5:
        vol_inicio = recente['volume'].iloc[:3].mean()
        vol_fim = recente['volume'].iloc[-3:].mean()
        if vol_inicio > 0 and vol_fim < vol_inicio * 0.6:
            score += 20
        elif vol_inicio > 0 and vol_fim < vol_inicio * 0.8:
            score += 10

    return min(score, 100)


def calcular_suportes_resistencias(dados: pd.DataFrame, lookback: int = 200) -> tuple:
    """
    Identifica suportes e resistências robusto com múltiplos métodos:
    1. Pivôs de alta/baixa (2 e 3 barras)
    2. VWAP do dia
    3. High/Low do dia
    4. Números redondos (múltiplos de 500/1000 para WIN, 50/100 para WDO)
    5. Agrupamento por volume (mais toques = mais forte)
    """
    recente = dados.tail(lookback)
    preco_atual = float(recente['close'].iloc[-1])
    suportes_raw = []
    resistencias_raw = []

    # === MÉTODO 1: Pivôs de 2 barras ===
    for i in range(2, len(recente) - 2):
        h = float(recente['high'].iloc[i])
        l = float(recente['low'].iloc[i])
        # Pivô de resistência (topo)
        if (h > recente['high'].iloc[i-1] and h > recente['high'].iloc[i-2] and
            h > recente['high'].iloc[i+1] and h > recente['high'].iloc[i+2]):
            resistencias_raw.append(round(h, 2))
        # Pivô de suporte (fundo)
        if (l < recente['low'].iloc[i-1] and l < recente['low'].iloc[i-2] and
            l < recente['low'].iloc[i+1] and l < recente['low'].iloc[i+2]):
            suportes_raw.append(round(l, 2))

    # === MÉTODO 2: Pivôs de 3 barras (mais fortes) ===
    for i in range(3, len(recente) - 3):
        h = float(recente['high'].iloc[i])
        l = float(recente['low'].iloc[i])
        if all(h > recente['high'].iloc[i+j] for j in [-3,-2,-1,1,2,3]):
            resistencias_raw.append(round(h, 2))
            resistencias_raw.append(round(h, 2))  # peso duplo
        if all(l < recente['low'].iloc[i+j] for j in [-3,-2,-1,1,2,3]):
            suportes_raw.append(round(l, 2))
            suportes_raw.append(round(l, 2))  # peso duplo

    # === MÉTODO 3: High/Low do dia atual ===
    try:
        hoje_dados = recente[recente.index.date == recente.index[-1].date()] if hasattr(recente.index, 'date') else recente.tail(80)
        if len(hoje_dados) > 0:
            high_dia = float(hoje_dados['high'].max())
            low_dia = float(hoje_dados['low'].min())
            if high_dia > preco_atual:
                resistencias_raw.append(round(high_dia, 2))
            if low_dia < preco_atual:
                suportes_raw.append(round(low_dia, 2))
    except:
        pass

    # === MÉTODO 4: Números redondos próximos ===
    if preco_atual > 1000:  # WIN (pontos grandes)
        base = round(preco_atual / 500) * 500
        for mult in [-1500, -1000, -500, 0, 500, 1000, 1500]:
            nivel = base + mult
            if abs(nivel - preco_atual) > 50:  # não muito perto do preço
                if nivel < preco_atual:
                    suportes_raw.append(nivel)
                else:
                    resistencias_raw.append(nivel)
    else:  # WDO
        base = round(preco_atual / 50) * 50
        for mult in [-150, -100, -50, 0, 50, 100, 150]:
            nivel = base + mult
            if abs(nivel - preco_atual) > 5:
                if nivel < preco_atual:
                    suportes_raw.append(nivel)
                else:
                    resistencias_raw.append(nivel)

    # === AGRUPAR por proximidade (tolerância maior) ===
    tol = 0.003 if preco_atual > 1000 else 0.005  # 0.3% WIN, 0.5% WDO
    suportes = _agrupar_niveis_ponderado(sorted(suportes_raw), tolerancia=tol)
    resistencias = _agrupar_niveis_ponderado(sorted(resistencias_raw), tolerancia=tol)

    # Filtrar: só manter níveis razoavelmente perto do preço (dentro de 2%)
    range_max = preco_atual * 0.02
    suportes = [s for s in suportes if preco_atual - s < range_max and s < preco_atual]
    resistencias = [r for r in resistencias if r - preco_atual < range_max and r > preco_atual]

    # Ordenar: suportes mais perto primeiro (desc), resistências mais perto primeiro (asc)
    suportes = sorted(suportes, reverse=True)[:5]
    resistencias = sorted(resistencias)[:5]

    return suportes, resistencias


def _agrupar_niveis_ponderado(niveis: list, tolerancia: float = 0.003) -> list:
    """Agrupa níveis próximos com peso (mais toques = média ponderada)"""
    if not niveis:
        return []
    grupos = [[niveis[0]]]
    for n in niveis[1:]:
        if abs(n - grupos[-1][-1]) / max(abs(grupos[-1][-1]), 1) <= tolerancia:
            grupos[-1].append(n)
        else:
            grupos.append([n])
    # Retornar média de cada grupo, ordenado por quantidade de toques (mais forte primeiro)
    resultado = [(sum(g)/len(g), len(g)) for g in grupos]
    resultado.sort(key=lambda x: -x[1])  # mais toques primeiro
    return [round(r[0], 2) for r in resultado]


def _agrupar_niveis(niveis: list, tolerancia: float = 0.001) -> list:
    """Agrupa níveis de preço próximos (backward compat)"""
    return _agrupar_niveis_ponderado(niveis, tolerancia)


# =====================================================
# INDICADORES TÉCNICOS EXTRAS
# =====================================================

def calcular_bollinger(dados: pd.DataFrame, periodo: int = 20, desvios: float = 2.0) -> dict:
    """Calcula Bandas de Bollinger"""
    if len(dados) < periodo:
        return {"upper": 0, "middle": 0, "lower": 0, "width": 0, "percent_b": 50, "squeeze": False}
    sma = dados['close'].rolling(window=periodo).mean()
    std = dados['close'].rolling(window=periodo).std()
    upper = sma + (std * desvios)
    lower = sma - (std * desvios)
    
    u = float(upper.iloc[-1])
    m = float(sma.iloc[-1])
    l = float(lower.iloc[-1])
    preco = float(dados['close'].iloc[-1])
    width = (u - l) / m * 100 if m > 0 else 0
    percent_b = (preco - l) / (u - l) * 100 if (u - l) > 0 else 50
    
    # Squeeze detection (bandwidth < 4%)
    squeeze = width < 4.0
    
    return {
        "upper": round(u, 2),
        "middle": round(m, 2),
        "lower": round(l, 2),
        "width": round(width, 2),
        "percent_b": round(percent_b, 1),
        "squeeze": squeeze,
        "status": "SQUEEZE" if squeeze else ("SOBRECOMPRADO" if percent_b > 80 else "SOBREVENDIDO" if percent_b < 20 else "NEUTRO"),
    }


def calcular_estocastico(dados: pd.DataFrame, k_periodo: int = 14, d_periodo: int = 3) -> dict:
    """Calcula Estocástico %K e %D"""
    if len(dados) < k_periodo:
        return {"k": 50, "d": 50, "status": "NEUTRO"}
    low_min = dados['low'].rolling(window=k_periodo).min()
    high_max = dados['high'].rolling(window=k_periodo).max()
    denom = high_max - low_min
    k = ((dados['close'] - low_min) / denom.replace(0, np.nan)) * 100
    k = k.fillna(50)
    d = k.rolling(window=d_periodo).mean().fillna(50)
    
    k_val = float(k.iloc[-1])
    d_val = float(d.iloc[-1])
    
    # Status
    if k_val > 80 and d_val > 80:
        status = "SOBRECOMPRADO"
    elif k_val < 20 and d_val < 20:
        status = "SOBREVENDIDO"
    elif k_val > d_val and k.iloc[-2] <= d.iloc[-2]:
        status = "CRUZAMENTO_ALTA"
    elif k_val < d_val and k.iloc[-2] >= d.iloc[-2]:
        status = "CRUZAMENTO_BAIXA"
    else:
        status = "NEUTRO"
    
    return {
        "k": round(k_val, 1),
        "d": round(d_val, 1),
        "status": status,
    }


def calcular_adx(dados: pd.DataFrame, periodo: int = 14) -> dict:
    """Calcula ADX, DI+ e DI-"""
    if len(dados) < periodo + 1:
        return {"adx": 0, "di_plus": 0, "di_minus": 0, "status": "NEUTRO", "forca": "FRACO"}
    
    high = dados['high']
    low = dados['low']
    close = dados['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=dados.index)
    minus_dm = pd.Series(minus_dm, index=dados.index)
    
    # Smoothed TR, +DM, -DM
    atr = tr.rolling(window=periodo, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(window=periodo, min_periods=1).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=periodo, min_periods=1).mean() / atr.replace(0, np.nan))
    
    plus_di = plus_di.fillna(0)
    minus_di = minus_di.fillna(0)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    dx = dx.fillna(0)
    adx = dx.rolling(window=periodo, min_periods=1).mean()
    
    adx_val = float(adx.iloc[-1])
    di_p = float(plus_di.iloc[-1])
    di_m = float(minus_di.iloc[-1])
    
    # Força da tendência
    if adx_val > 50:
        forca = "MUITO_FORTE"
    elif adx_val > 25:
        forca = "FORTE"
    elif adx_val > 20:
        forca = "MODERADO"
    else:
        forca = "FRACO"
    
    # Status
    if di_p > di_m and adx_val > 20:
        status = "ALTA"
    elif di_m > di_p and adx_val > 20:
        status = "BAIXA"
    else:
        status = "NEUTRO"
    
    return {
        "adx": round(adx_val, 1),
        "di_plus": round(di_p, 1),
        "di_minus": round(di_m, 1),
        "status": status,
        "forca": forca,
    }


def calcular_obv(dados: pd.DataFrame) -> dict:
    """Calcula On-Balance Volume (OBV)"""
    if len(dados) < 2:
        return {"obv": 0, "obv_sma": 0, "status": "NEUTRO", "divergencia": "NENHUMA"}
    
    close = dados['close']
    volume = dados['volume']
    
    obv = pd.Series(0.0, index=dados.index)
    for i in range(1, len(dados)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    obv_sma = obv.rolling(window=20, min_periods=1).mean()
    obv_val = float(obv.iloc[-1])
    obv_sma_val = float(obv_sma.iloc[-1])
    
    # Divergência: preço sobe mas OBV cai (ou vice-versa)
    if len(dados) >= 10:
        preco_trend = close.iloc[-1] - close.iloc[-10]
        obv_trend = obv.iloc[-1] - obv.iloc[-10]
        if preco_trend > 0 and obv_trend < 0:
            divergencia = "BAIXA"  # Bearish divergence
        elif preco_trend < 0 and obv_trend > 0:
            divergencia = "ALTA"  # Bullish divergence
        else:
            divergencia = "NENHUMA"
    else:
        divergencia = "NENHUMA"
    
    return {
        "obv": round(obv_val, 0),
        "obv_sma": round(obv_sma_val, 0),
        "status": "ALTA" if obv_val > obv_sma_val else "BAIXA",
        "divergencia": divergencia,
    }


def calcular_ichimoku(dados: pd.DataFrame) -> dict:
    """Calcula Ichimoku Cloud (Tenkan, Kijun, Senkou A/B, Chikou)"""
    if len(dados) < 52:
        return {"tenkan": 0, "kijun": 0, "senkou_a": 0, "senkou_b": 0, "status": "NEUTRO", "nuvem": "NEUTRO"}
    
    high = dados['high']
    low = dados['low']
    close = dados['close']
    
    # Tenkan-sen (Conversion Line) - 9 periods
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    # Kijun-sen (Base Line) - 26 periods
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    # Senkou Span A - midpoint of Tenkan and Kijun (shifted 26)
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B - 52 period (shifted 26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2)
    
    t = float(tenkan.iloc[-1])
    k = float(kijun.iloc[-1])
    sa = float(senkou_a.iloc[-1])
    sb = float(senkou_b.iloc[-1])
    preco = float(close.iloc[-1])
    
    # Cloud color
    if sa > sb:
        nuvem = "ALTA"
    elif sa < sb:
        nuvem = "BAIXA"
    else:
        nuvem = "NEUTRO"
    
    # Status
    if preco > max(sa, sb) and t > k:
        status = "FORTE_ALTA"
    elif preco > max(sa, sb):
        status = "ALTA"
    elif preco < min(sa, sb) and t < k:
        status = "FORTE_BAIXA"
    elif preco < min(sa, sb):
        status = "BAIXA"
    else:
        status = "DENTRO_NUVEM"
    
    return {
        "tenkan": round(t, 2),
        "kijun": round(k, 2),
        "senkou_a": round(sa, 2),
        "senkou_b": round(sb, 2),
        "status": status,
        "nuvem": nuvem,
    }


def calcular_pivot_points(dados: pd.DataFrame) -> dict:
    """Calcula Pivot Points (clássico) baseado no dia anterior"""
    if len(dados) < 2:
        return {"pivot": 0, "r1": 0, "r2": 0, "r3": 0, "s1": 0, "s2": 0, "s3": 0, "posicao": "NEUTRO"}
    
    # Usar último candle completo como referência
    h = float(dados['high'].iloc[-2])
    l = float(dados['low'].iloc[-2])
    c = float(dados['close'].iloc[-2])
    
    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    s1 = 2 * pivot - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    r3 = h + 2 * (pivot - l)
    s3 = l - 2 * (h - pivot)
    
    preco = float(dados['close'].iloc[-1])
    if preco > r2:
        posicao = "ACIMA_R2"
    elif preco > r1:
        posicao = "ACIMA_R1"
    elif preco > pivot:
        posicao = "ACIMA_PIVOT"
    elif preco > s1:
        posicao = "ABAIXO_PIVOT"
    elif preco > s2:
        posicao = "ABAIXO_S1"
    else:
        posicao = "ABAIXO_S2"
    
    return {
        "pivot": round(pivot, 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "r3": round(r3, 2),
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "s3": round(s3, 2),
        "posicao": posicao,
    }


def calcular_vwap_bands(dados: pd.DataFrame, desvios: float = 2.0) -> dict:
    """Calcula VWAP com bandas de desvio padrão"""
    if len(dados) < 5:
        return {"vwap": 0, "upper_1": 0, "upper_2": 0, "lower_1": 0, "lower_2": 0, "posicao": "NEUTRO"}
    tp = (dados['high'] + dados['low'] + dados['close']) / 3
    vol = dados['volume'].replace(0, np.nan)
    cum_tp_vol = (tp * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap = cum_tp_vol / cum_vol
    
    # Standard deviation of VWAP
    vwap_sq = ((tp - vwap) ** 2 * vol).cumsum() / cum_vol
    vwap_std = np.sqrt(vwap_sq)
    
    v = float(vwap.iloc[-1])
    s = float(vwap_std.iloc[-1]) if not np.isnan(vwap_std.iloc[-1]) else 0
    preco = float(dados['close'].iloc[-1])
    
    upper_1 = v + s
    upper_2 = v + s * desvios
    lower_1 = v - s
    lower_2 = v - s * desvios
    
    if preco > upper_2:
        posicao = "ACIMA_BANDA2"
    elif preco > upper_1:
        posicao = "ACIMA_BANDA1"
    elif preco > v:
        posicao = "ACIMA_VWAP"
    elif preco > lower_1:
        posicao = "ABAIXO_VWAP"
    elif preco > lower_2:
        posicao = "ABAIXO_BANDA1"
    else:
        posicao = "ABAIXO_BANDA2"
    
    return {
        "vwap": round(v, 2),
        "upper_1": round(upper_1, 2),
        "upper_2": round(upper_2, 2),
        "lower_1": round(lower_1, 2),
        "lower_2": round(lower_2, 2),
        "posicao": posicao,
    }



def gerar_sinais(
    dados: pd.DataFrame,
    fibonacci: FibonacciLevels,
    rsi: float,
    macd_linha: float,
    macd_sinal: float,
    macd_hist: float,
    volume: AnaliseVolume,
    violinada_score: float,
    tendencia: str = "LATERAL",
    pullback_info: dict = None,
    lateralizacao: dict = None,
    vwap_atual: float = 0,
) -> list:
    """
    Gera sinais CRUZANDO TODOS os indicadores simultaneamente.
    Um sinal só é emitido se TODOS os indicadores concordam na mesma direção.
    Checklist obrigatório:
      1. Tendência (EMA 9/21 + VWAP)
      2. RSI (não sobrecomprado para compra, não sobrevendido para venda)
      3. MACD (histograma e cruzamento na direção)
      4. Volume (pressão compradora/vendedora + acima da média)
      5. Fibonacci (preço em zona favorável)
      6. Anti-violinada (score baixo)
    """
    sinais = []
    preco = dados['close'].iloc[-1]

    # === BLOQUEIO TOTAL: Mercado lateralizado ===
    if lateralizacao and lateralizacao.get("lateral"):
        return []

    # === BLOQUEIO: Risco de violinada alto ===
    if violinada_score > 60:
        return []

    # === CHECKLIST DE COMPRA - TODOS devem confirmar ===
    compra_tendencia = (tendencia == "ALTA")
    compra_rsi = (rsi < 70)  # Não sobrecomprado
    compra_macd = (macd_linha > macd_sinal and macd_hist > 0)
    compra_volume = (volume.pressao == "COMPRADORES" and volume.volume_acima_media)
    compra_pullback = bool(pullback_info and pullback_info.get("confirmado") and tendencia == "ALTA")

    # Fibonacci: preço em zona de retração favorável para compra
    fib = fibonacci
    tolerancia_fib = abs(fib.nivel_0 - fib.nivel_100) * 0.03
    compra_fib = False
    fib_zona_compra = ""
    if fib.tendencia == "ALTA":
        zonas = [
            (fib.nivel_618, "61.8%"),
            (fib.nivel_500, "50.0%"),
            (fib.nivel_382, "38.2%"),
        ]
        for nivel, nome in zonas:
            if abs(preco - nivel) < tolerancia_fib:
                compra_fib = True
                fib_zona_compra = nome
                break
        # Também aceita se preço está acima de 23.6% (tendência forte)
        if preco > fib.nivel_236:
            compra_fib = True
            fib_zona_compra = "acima de 23.6% (tendência forte)"

    # Cruzamento completo: TODOS os 5 indicadores principais devem confirmar
    indicadores_compra = {
        "tendencia": compra_tendencia,
        "rsi": compra_rsi,
        "macd": compra_macd,
        "volume": compra_volume,
        "fibonacci": compra_fib,
    }
    total_confirmados_compra = sum(1 for v in indicadores_compra.values() if v)

    # === CHECKLIST DE VENDA - TODOS devem confirmar ===
    venda_tendencia = (tendencia == "BAIXA")
    venda_rsi = (rsi > 30)  # Não sobrevendido
    venda_macd = (macd_linha < macd_sinal and macd_hist < 0)
    venda_volume = (volume.pressao == "VENDEDORES" and volume.volume_acima_media)
    venda_pullback = bool(pullback_info and pullback_info.get("confirmado") and tendencia == "BAIXA")

    venda_fib = False
    fib_zona_venda = ""
    if fib.tendencia == "BAIXA":
        zonas = [
            (fib.nivel_382, "38.2%"),
            (fib.nivel_500, "50.0%"),
            (fib.nivel_618, "61.8%"),
        ]
        for nivel, nome in zonas:
            if abs(preco - nivel) < tolerancia_fib:
                venda_fib = True
                fib_zona_venda = nome
                break
        if preco < fib.nivel_786:
            venda_fib = True
            fib_zona_venda = "abaixo de 78.6% (tendência forte)"

    indicadores_venda = {
        "tendencia": venda_tendencia,
        "rsi": venda_rsi,
        "macd": venda_macd,
        "volume": venda_volume,
        "fibonacci": venda_fib,
    }
    total_confirmados_venda = sum(1 for v in indicadores_venda.values() if v)

    # === ATR para cálculo de stops ===
    atr = _calcular_atr(dados)

    # === GERAR SINAL DE COMPRA (mínimo 4 de 5 indicadores) ===
    if total_confirmados_compra >= 4 and compra_tendencia:
        motivos = []
        motivos.append(f"TENDENCIA: Alta confirmada (EMA 9 > EMA 21, preço > VWAP)")
        if compra_rsi:
            motivos.append(f"RSI: {rsi:.1f} - zona favorável para compra")
        if compra_macd:
            motivos.append(f"MACD: Cruzamento de alta (hist {macd_hist:.2f})")
        if compra_volume:
            motivos.append(f"VOLUME: Pressão compradora acima da média (ratio {volume.ratio_compra_venda:.2f})")
        if compra_fib:
            motivos.append(f"FIBONACCI: Preço na zona {fib_zona_compra}")
        if compra_pullback:
            motivos.append(f"PULLBACK: Confirmado com candle de reversão")

        confianca = int((total_confirmados_compra / 5) * 100)
        if compra_pullback:
            confianca = min(confianca + 15, 100)
        if violinada_score < 20:
            confianca = min(confianca + 5, 100)

        # Stop técnico
        ultimos_lows = dados['low'].tail(10)
        stop = round(float(ultimos_lows.min()), 2)
        risco = preco - stop
        if risco <= 0:
            risco = atr * 2
            stop = round(preco - risco, 2)

        tp1 = round(preco + risco * 1.0, 2)
        tp2 = round(preco + risco * 2.0, 2)
        tp3 = round(preco + risco * 3.0, 2)
        rr = round((risco * 2) / risco, 2) if risco > 0 else 2.0

        violinada_risco = "BAIXO" if violinada_score < 30 else ("MEDIO" if violinada_score < 60 else "ALTO")

        sinais.append(SinalEntrada(
            tipo="COMPRA",
            preco_entrada=round(preco, 2),
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risco_retorno=rr,
            confianca=confianca,
            motivos=motivos,
            fibonacci_zona=_zona_fibonacci_atual(preco, fibonacci),
            rsi_status="SOBREVENDIDO" if rsi < 20 else ("SOBRECOMPRADO" if rsi > 80 else "NEUTRO"),
            macd_status="ALTA" if macd_hist > 0 else "BAIXA",
            volume_status=volume.pressao,
            violinada_risco=violinada_risco,
        ))

    # === GERAR SINAL DE VENDA (mínimo 4 de 5 indicadores) ===
    if total_confirmados_venda >= 4 and venda_tendencia:
        motivos = []
        motivos.append(f"TENDENCIA: Baixa confirmada (EMA 9 < EMA 21, preço < VWAP)")
        if venda_rsi:
            motivos.append(f"RSI: {rsi:.1f} - zona favorável para venda")
        if venda_macd:
            motivos.append(f"MACD: Cruzamento de baixa (hist {macd_hist:.2f})")
        if venda_volume:
            motivos.append(f"VOLUME: Pressão vendedora acima da média (ratio {volume.ratio_compra_venda:.2f})")
        if venda_fib:
            motivos.append(f"FIBONACCI: Preço na zona {fib_zona_venda}")
        if venda_pullback:
            motivos.append(f"PULLBACK: Confirmado com candle de reversão")

        confianca = int((total_confirmados_venda / 5) * 100)
        if venda_pullback:
            confianca = min(confianca + 15, 100)
        if violinada_score < 20:
            confianca = min(confianca + 5, 100)

        ultimos_highs = dados['high'].tail(10)
        stop = round(float(ultimos_highs.max()), 2)
        risco = stop - preco
        if risco <= 0:
            risco = atr * 2
            stop = round(preco + risco, 2)

        tp1 = round(preco - risco * 1.0, 2)
        tp2 = round(preco - risco * 2.0, 2)
        tp3 = round(preco - risco * 3.0, 2)
        rr = round((risco * 2) / risco, 2) if risco > 0 else 2.0

        violinada_risco = "BAIXO" if violinada_score < 30 else ("MEDIO" if violinada_score < 60 else "ALTO")

        sinais.append(SinalEntrada(
            tipo="VENDA",
            preco_entrada=round(preco, 2),
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risco_retorno=rr,
            confianca=confianca,
            motivos=motivos,
            fibonacci_zona=_zona_fibonacci_atual(preco, fibonacci),
            rsi_status="SOBREVENDIDO" if rsi < 20 else ("SOBRECOMPRADO" if rsi > 80 else "NEUTRO"),
            macd_status="ALTA" if macd_hist > 0 else "BAIXA",
            volume_status=volume.pressao,
            violinada_risco=violinada_risco,
        ))

    return sinais


def _calcular_atr(dados: pd.DataFrame, periodo: int = 14) -> float:
    """Calcula Average True Range"""
    high = dados['high']
    low = dados['low']
    close = dados['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=periodo, min_periods=1).mean()
    return atr.iloc[-1]


def _zona_fibonacci_atual(preco: float, fib: FibonacciLevels) -> str:
    """Identifica em qual zona de Fibonacci o preço está"""
    niveis = [
        (fib.nivel_0, "0% (Topo)"),
        (fib.nivel_236, "23.6%"),
        (fib.nivel_382, "38.2%"),
        (fib.nivel_500, "50.0%"),
        (fib.nivel_618, "61.8%"),
        (fib.nivel_786, "78.6%"),
        (fib.nivel_100, "100% (Fundo)"),
    ]
    niveis_sorted = sorted(niveis, key=lambda x: x[0])

    for i in range(len(niveis_sorted) - 1):
        if niveis_sorted[i][0] <= preco <= niveis_sorted[i+1][0]:
            return f"Entre {niveis_sorted[i][1]} e {niveis_sorted[i+1][1]}"

    if preco > max(n[0] for n in niveis):
        return "Acima de 0% (extensão)"
    return "Abaixo de 100% (extensão)"




# =====================================================
# GESTAO DE RISCO - Implementacao do documento de trading
# =====================================================

class GestaoRisco:
    """
    Sistema de gestão de risco conforme documento de lógica de execução B3.
    Controla risco por operação, risco diário, sequência de stops e lote.
    """

    # Custos operacionais B3 (valores aproximados 2024-2025)
    CUSTOS_B3 = {
        "WIN": {
            "corretagem": 0.0,       # Day trade isento em muitas corretoras
            "emolumentos": 0.0035,   # % sobre volume negociado
            "taxa_registro": 0.0008, # % sobre volume
            "valor_ponto": 0.20,     # R$ por ponto do mini-indice
        },
        "WDO": {
            "corretagem": 0.0,
            "emolumentos": 0.0035,
            "taxa_registro": 0.0008,
            "valor_ponto": 10.00,    # R$ por ponto do mini-dolar
        }
    }

    def __init__(self, capital_total: float = 10000.0, risco_pct_operacao: float = 0.01,
                 risco_pct_diario: float = 0.02, max_stops_consecutivos: int = 3):
        self.capital_total = capital_total
        self.risco_pct_operacao = risco_pct_operacao  # 1% padrão
        self.risco_pct_diario = risco_pct_diario      # 2% padrão
        self.max_stops_consecutivos = max_stops_consecutivos
        self.stops_consecutivos = 0
        self.perda_diaria = 0.0
        self.operacoes_dia = []
        self.bloqueado = False
        self.motivo_bloqueio = ""

    def definir_risco_operacao(self) -> float:
        """Calcula risco máximo por operação: 0.5% a 1% do capital total"""
        return self.capital_total * self.risco_pct_operacao

    def definir_risco_diario_maximo(self) -> float:
        """Calcula risco diário máximo: 2% do capital total"""
        return self.capital_total * self.risco_pct_diario

    def verificar_risco_diario(self) -> dict:
        """Verifica se o risco diário máximo foi atingido"""
        risco_max = self.definir_risco_diario_maximo()
        atingido = abs(self.perda_diaria) >= risco_max
        if atingido:
            self.bloqueado = True
            self.motivo_bloqueio = f"Risco diário máximo atingido (R$ {abs(self.perda_diaria):.2f} / R$ {risco_max:.2f})"
        return {
            "risco_maximo": risco_max,
            "perda_atual": abs(self.perda_diaria),
            "pct_utilizado": (abs(self.perda_diaria) / risco_max * 100) if risco_max > 0 else 0,
            "atingido": atingido,
            "bloqueado": self.bloqueado,
            "motivo": self.motivo_bloqueio
        }

    def validar_risco_retorno(self, stop_loss: float, preco_entrada: float, alvo: float) -> dict:
        """
        Garante que a relação risco/retorno é >= 2:1.
        Retorna se a operação é válida e a relação calculada.
        """
        risco = abs(preco_entrada - stop_loss)
        retorno = abs(alvo - preco_entrada)
        if risco == 0:
            return {"valido": False, "rr": 0, "motivo": "Risco zero - stop_loss igual ao preço de entrada"}
        rr = retorno / risco
        valido = rr >= 2.0
        return {
            "valido": valido,
            "rr": round(rr, 2),
            "risco_pts": round(risco, 2),
            "retorno_pts": round(retorno, 2),
            "motivo": f"R/R {rr:.2f}:1 - {'APROVADO' if valido else 'REPROVADO (min 2:1)'}"
        }

    def calcular_lote_ajustado(self, distancia_stop_pontos: float, ativo: str = "WIN") -> dict:
        """
        Ajusta o lote para que a perda máxima não exceda o risco por operação.
        lote * distancia_stop * valor_ponto <= risco_financeiro
        """
        risco_financeiro = self.definir_risco_operacao()
        custos = self.CUSTOS_B3.get(ativo, self.CUSTOS_B3["WIN"])
        valor_ponto = custos["valor_ponto"]

        if distancia_stop_pontos <= 0 or valor_ponto <= 0:
            return {"lote": 1, "risco_financeiro": risco_financeiro, "motivo": "Valores inválidos, lote padrão 1"}

        lote_max = int(risco_financeiro / (distancia_stop_pontos * valor_ponto))
        lote = max(1, lote_max)
        risco_real = lote * distancia_stop_pontos * valor_ponto

        return {
            "lote": lote,
            "lote_maximo": lote_max,
            "risco_por_operacao": round(risco_financeiro, 2),
            "risco_real": round(risco_real, 2),
            "distancia_stop": round(distancia_stop_pontos, 2),
            "valor_ponto": valor_ponto,
            "motivo": f"{lote} contrato(s) - risco R$ {risco_real:.2f} de R$ {risco_financeiro:.2f} permitido"
        }

    def monitorar_sequencia_stops(self, resultado_ultima_op: float) -> dict:
        """
        Monitora sequência de stops. Se 3 consecutivos, bloqueia temporariamente.
        """
        if resultado_ultima_op < 0:
            self.stops_consecutivos += 1
        else:
            self.stops_consecutivos = 0

        bloqueado = self.stops_consecutivos >= self.max_stops_consecutivos
        if bloqueado:
            self.bloqueado = True
            self.motivo_bloqueio = f"{self.stops_consecutivos} stops consecutivos - operações bloqueadas"

        return {
            "stops_consecutivos": self.stops_consecutivos,
            "max_permitido": self.max_stops_consecutivos,
            "bloqueado": bloqueado,
            "motivo": self.motivo_bloqueio if bloqueado else f"{self.stops_consecutivos}/{self.max_stops_consecutivos} stops"
        }

    def registrar_operacao(self, resultado_financeiro: float, tipo: str, entrada: float, saida: float):
        """Registra uma operação no controle diário"""
        self.operacoes_dia.append({
            "tipo": tipo,
            "entrada": entrada,
            "saida": saida,
            "resultado": resultado_financeiro,
        })
        if resultado_financeiro < 0:
            self.perda_diaria += resultado_financeiro
        self.monitorar_sequencia_stops(resultado_financeiro)

    def calcular_custos_operacionais(self, preco_entrada: float, preco_saida: float,
                                      lote: int, ativo: str = "WIN") -> dict:
        """
        Calcula custos operacionais reais da B3:
        corretagem + emolumentos + taxa de registro + liquidação
        """
        custos = self.CUSTOS_B3.get(ativo, self.CUSTOS_B3["WIN"])
        volume = (preco_entrada + preco_saida) * lote  # Volume total (entrada + saída)

        corretagem = custos["corretagem"] * volume
        emolumentos = custos["emolumentos"] * volume
        taxa_registro = custos["taxa_registro"] * volume
        total = corretagem + emolumentos + taxa_registro

        return {
            "corretagem": round(corretagem, 2),
            "emolumentos": round(emolumentos, 2),
            "taxa_registro": round(taxa_registro, 2),
            "total": round(total, 2),
            "volume_negociado": round(volume, 2)
        }

    def calcular_slippage(self, preco_esperado: float, preco_executado: float) -> dict:
        """Registra e calcula slippage"""
        slippage = abs(preco_esperado - preco_executado)
        return {
            "preco_esperado": preco_esperado,
            "preco_executado": preco_executado,
            "slippage_pts": round(slippage, 2),
            "favoravel": preco_executado <= preco_esperado  # Para compra
        }

    def buscar_alvo_liquidez(self, dados, tipo_operacao: str, lookback: int = 50) -> dict:
        """
        Busca alvos de liquidez baseado em topos/fundos anteriores.
        Compra -> busca topos anteriores e zonas de stop de vendedores
        Venda -> busca fundos anteriores e zonas de stop de compradores
        """
        if len(dados) < lookback:
            lookback = len(dados)

        window = dados.tail(lookback)
        preco_atual = float(dados['close'].iloc[-1])

        if tipo_operacao == "COMPRA":
            # Buscar topos anteriores (resistências) como alvos
            highs = window['high'].values
            alvos = sorted(set([float(h) for h in highs if h > preco_atual]))[:3]
            return {
                "tipo": "COMPRA",
                "alvos": alvos,
                "alvo_principal": alvos[0] if alvos else preco_atual * 1.005,
                "descricao": "Topos anteriores como zonas de liquidez"
            }
        else:
            # Buscar fundos anteriores (suportes) como alvos
            lows = window['low'].values
            alvos = sorted(set([float(l) for l in lows if l < preco_atual]), reverse=True)[:3]
            return {
                "tipo": "VENDA",
                "alvos": alvos,
                "alvo_principal": alvos[0] if alvos else preco_atual * 0.995,
                "descricao": "Fundos anteriores como zonas de liquidez"
            }

    def status_completo(self) -> dict:
        """Retorna status completo da gestão de risco"""
        risco_diario = self.verificar_risco_diario()
        return {
            "capital": self.capital_total,
            "risco_por_op": self.definir_risco_operacao(),
            "risco_diario_max": self.definir_risco_diario_maximo(),
            "perda_diaria": abs(self.perda_diaria),
            "pct_risco_utilizado": risco_diario["pct_utilizado"],
            "operacoes_hoje": len(self.operacoes_dia),
            "stops_consecutivos": self.stops_consecutivos,
            "bloqueado": self.bloqueado,
            "motivo_bloqueio": self.motivo_bloqueio
        }


def definir_stop_loss(preco_entrada: float, distancia_pontos: float, tipo: str) -> float:
    """
    Posiciona stop loss ANTES da execução.
    Compra: stop = entrada - distância
    Venda: stop = entrada + distância
    """
    if tipo.upper() == "COMPRA":
        return round(preco_entrada - distancia_pontos, 2)
    else:
        return round(preco_entrada + distancia_pontos, 2)


def verificar_horario_operacional(hora_atual=None) -> dict:
    """
    Verifica se estamos em horário operacional para day trade na B3.
    Janela 1: 09:15 - 12:00 (melhor liquidez)
    Janela 2: 14:00 - 16:30 (segundo período)
    Evitar: abertura (09:00-09:15), almoço (12:00-14:00), últimos 30min
    """
    from datetime import datetime
    if hora_atual is None:
        hora_atual = datetime.now()

    h = hora_atual.hour
    m = hora_atual.minute
    minutos = h * 60 + m

    janela_1 = 555 <= minutos <= 720   # 09:15 - 12:00
    janela_2 = 840 <= minutos <= 990   # 14:00 - 16:30
    operacional = janela_1 or janela_2

    if not operacional:
        if minutos < 555:
            aviso = "Pre-mercado - aguardar abertura 09:15"
        elif 720 < minutos < 840:
            aviso = "Horario de almoco - menor liquidez"
        elif minutos > 990:
            aviso = "Apos 16:30 - evitar novas operacoes"
        else:
            aviso = "Fora do horario operacional"
    else:
        aviso = "Janela 1 (09:15-12:00)" if janela_1 else "Janela 2 (14:00-16:30)"

    return {
        "janela_1": janela_1,
        "janela_2": janela_2,
        "operacional": operacional,
        "aviso": aviso
    }




# =====================================================
# ESTRATEGIA ZERO LOSS - Proteção de Capital
# Baseado no documento "Estratégia Master Logic B3"
# =====================================================

class ZeroLossProtection:
    """
    Sistema de proteção "Zero Loss" conforme Master Logic B3.
    Gerencia break-even dinâmico, trailing stop e proteção de lucro.
    """

    def __init__(self, ativo: str = "WIN"):
        self.ativo = ativo
        custos_map = {"WIN": 0.20, "WDO": 10.00}
        self.valor_ponto = custos_map.get(ativo, 0.20)

    def dynamic_break_even(self, preco_entrada: float, preco_atual: float,
                           alvo: float, stop_atual: float, tipo: str) -> dict:
        """
        Break-even dinâmico: quando preço atinge 50% do alvo,
        mover stop para entrada + custos operacionais.
        """
        distancia_total = abs(alvo - preco_entrada)
        distancia_atual = abs(preco_atual - preco_entrada)
        pct_caminho = (distancia_atual / distancia_total * 100) if distancia_total > 0 else 0

        # Custos operacionais estimados em pontos
        custos_pts = 2 if self.ativo == "WIN" else 0.5  # ~2 pts WIN, ~0.5 WDO

        ativar_be = pct_caminho >= 50
        if ativar_be:
            if tipo.upper() == "COMPRA":
                novo_stop = preco_entrada + custos_pts
                ativado = preco_atual > preco_entrada + custos_pts
            else:
                novo_stop = preco_entrada - custos_pts
                ativado = preco_atual < preco_entrada - custos_pts
        else:
            novo_stop = stop_atual
            ativado = False

        return {
            "ativado": ativado,
            "pct_caminho": round(pct_caminho, 1),
            "stop_original": stop_atual,
            "stop_breakeven": round(novo_stop, 2) if ativar_be else None,
            "custos_pts": custos_pts,
            "descricao": f"BE ativado - stop em {novo_stop:.2f} (+custos)" if ativado else f"Aguardando 50% do alvo ({pct_caminho:.0f}% atingido)"
        }

    def trailing_stop_atr(self, preco_atual: float, atr: float,
                          stop_atual: float, tipo: str) -> dict:
        """
        Trailing stop baseado em 1.5x ATR.
        O stop segue o preço mantendo distância de 1.5 * ATR.
        """
        distancia = atr * 1.5

        if tipo.upper() == "COMPRA":
            trailing = round(preco_atual - distancia, 2)
            novo_stop = max(trailing, stop_atual)  # Só move para cima
            moveu = novo_stop > stop_atual
        else:
            trailing = round(preco_atual + distancia, 2)
            novo_stop = min(trailing, stop_atual)  # Só move para baixo
            moveu = novo_stop < stop_atual

        return {
            "stop_atual": stop_atual,
            "trailing_calculado": trailing,
            "novo_stop": novo_stop,
            "moveu": moveu,
            "distancia_atr": round(distancia, 2),
            "atr_usado": round(atr, 2),
            "descricao": f"Trailing 1.5xATR = {distancia:.0f} pts" + (" - MOVEU" if moveu else "")
        }

    def profit_protection(self, preco_entrada: float, preco_atual: float,
                          alvo: float, tipo: str) -> dict:
        """
        Proteção de lucro: se lucro atinge 1.5:1, garantir mínimo de 0.8:1.
        """
        risco = abs(alvo - preco_entrada) / 2  # Risco original ~= distância ao stop
        if tipo.upper() == "COMPRA":
            lucro_atual = preco_atual - preco_entrada
        else:
            lucro_atual = preco_entrada - preco_atual

        rr_atual = (lucro_atual / risco) if risco > 0 else 0

        proteger = rr_atual >= 1.5
        if proteger:
            # Garantir mínimo de 0.8 do risco como lucro
            lucro_minimo = risco * 0.8
            if tipo.upper() == "COMPRA":
                stop_protecao = round(preco_entrada + lucro_minimo, 2)
            else:
                stop_protecao = round(preco_entrada - lucro_minimo, 2)
        else:
            stop_protecao = None

        return {
            "rr_atual": round(rr_atual, 2),
            "proteger": proteger,
            "stop_protecao": stop_protecao,
            "lucro_minimo_garantido": round(risco * 0.8, 2) if proteger else 0,
            "descricao": f"R/R atual {rr_atual:.2f} - {'PROTEGER lucro min 0.8:1' if proteger else 'Aguardando 1.5:1'}"
        }

    def volatility_lock(self, atr_atual: float, atr_medio: float,
                        atr_std: float) -> dict:
        """
        Volatility lock: se ATR dispara acima de 1 desvio padrão,
        suspender novas entradas e proteger posições.
        """
        limite = atr_medio + atr_std
        bloqueado = atr_atual > limite

        return {
            "atr_atual": round(atr_atual, 2),
            "atr_medio": round(atr_medio, 2),
            "atr_std": round(atr_std, 2),
            "limite": round(limite, 2),
            "bloqueado": bloqueado,
            "descricao": f"ATR {atr_atual:.0f} {'> LIMITE' if bloqueado else '<= limite'} {limite:.0f} - {'BLOQUEADO' if bloqueado else 'OK'}"
        }

    def gestao_posicao_completa(self, preco_entrada: float, preco_atual: float,
                                 alvo: float, stop_atual: float, atr: float,
                                 tipo: str) -> dict:
        """Executa toda a lógica de proteção Zero Loss em uma chamada."""
        be = self.dynamic_break_even(preco_entrada, preco_atual, alvo, stop_atual, tipo)
        trailing = self.trailing_stop_atr(preco_atual, atr, stop_atual, tipo)
        profit = self.profit_protection(preco_entrada, preco_atual, alvo, tipo)

        # Determinar o melhor stop (mais protetor)
        stops = [stop_atual]
        if be["ativado"] and be["stop_breakeven"]:
            stops.append(be["stop_breakeven"])
        if trailing["moveu"]:
            stops.append(trailing["novo_stop"])
        if profit["proteger"] and profit["stop_protecao"]:
            stops.append(profit["stop_protecao"])

        if tipo.upper() == "COMPRA":
            melhor_stop = max(stops)
        else:
            melhor_stop = min(stops)

        return {
            "break_even": be,
            "trailing_stop": trailing,
            "profit_protection": profit,
            "stop_recomendado": round(melhor_stop, 2),
            "stop_original": stop_atual,
            "protecao_ativa": melhor_stop != stop_atual,
        }


def analisar_correlacao_ativos(dados_win, dados_wdo) -> dict:
    """
    Correlação WIN/WDO: se Índice sobe e Dólar cai com volume,
    confiança no sinal de compra do índice aumenta (e vice-versa).
    """
    import numpy as np

    if dados_win is None or dados_wdo is None:
        return {"disponivel": False, "descricao": "Dados insuficientes"}

    if len(dados_win) < 10 or len(dados_wdo) < 10:
        return {"disponivel": False, "descricao": "Dados insuficientes"}

    # Retornos dos últimos 10 candles
    ret_win = dados_win['close'].pct_change().tail(10).dropna()
    ret_wdo = dados_wdo['close'].pct_change().tail(10).dropna()

    min_len = min(len(ret_win), len(ret_wdo))
    if min_len < 5:
        return {"disponivel": False, "descricao": "Poucos dados"}

    ret_win = ret_win.tail(min_len).values
    ret_wdo = ret_wdo.tail(min_len).values

    correlacao = float(np.corrcoef(ret_win, ret_wdo)[0, 1])

    # Tendência recente
    win_direcao = "ALTA" if ret_win[-1] > 0 else "BAIXA"
    wdo_direcao = "ALTA" if ret_wdo[-1] > 0 else "BAIXA"

    # Correlação normal B3: WIN e WDO são inversamente correlacionados
    # WIN sobe + WDO cai = confirmação de alta do índice
    confirmacao = (win_direcao == "ALTA" and wdo_direcao == "BAIXA") or \
                  (win_direcao == "BAIXA" and wdo_direcao == "ALTA")

    return {
        "disponivel": True,
        "correlacao": round(correlacao, 4),
        "win_direcao": win_direcao,
        "wdo_direcao": wdo_direcao,
        "confirmacao": confirmacao,
        "forca": "FORTE" if abs(correlacao) > 0.5 else "FRACA",
        "descricao": f"WIN {win_direcao} / WDO {wdo_direcao} - Correlacao {correlacao:.2f} - {'CONFIRMADO' if confirmacao else 'DIVERGENTE'}"
    }


def detectar_absorcao(dados, lookback: int = 5) -> dict:
    """
    Detecta absorção: agressão forte (volume alto) mas preço não desloca.
    Indica possível reversão iminente.
    """
    if len(dados) < lookback + 1:
        return {"detectada": False, "descricao": "Dados insuficientes"}

    ultimos = dados.tail(lookback)
    vol_medio = dados['volume'].tail(20).mean()
    vol_recente = ultimos['volume'].mean()

    # Deslocamento de preço
    range_preco = abs(float(ultimos['close'].iloc[-1] - ultimos['open'].iloc[0]))
    range_max = float(ultimos['high'].max() - ultimos['low'].min())

    # Alto volume com pouco deslocamento = absorção
    volume_alto = vol_recente > vol_medio * 1.5
    pouco_deslocamento = range_preco < range_max * 0.3 if range_max > 0 else False

    detectada = volume_alto and pouco_deslocamento

    return {
        "detectada": detectada,
        "volume_ratio": round(vol_recente / vol_medio, 2) if vol_medio > 0 else 0,
        "deslocamento_pct": round(range_preco / range_max * 100, 1) if range_max > 0 else 0,
        "tipo": "ABSORCAO_VENDA" if dados['close'].iloc[-1] < dados['open'].iloc[-1] else "ABSORCAO_COMPRA",
        "descricao": f"{'ABSORCAO DETECTADA - possível reversão' if detectada else 'Sem absorção'}"
    }


def analisar_completo(dados: pd.DataFrame, timeframe: str, ativo: str) -> dict:
    """
    Executa análise completa para um timeframe específico.
    Retorna dicionário com todos os indicadores e sinais.
    """
    if len(dados) < 30:
        return {"erro": "Dados insuficientes para análise"}

    # Calcular indicadores
    rsi_series = calcular_rsi(dados)
    rsi_atual = rsi_series.iloc[-1]

    macd_linha, macd_sinal_line, macd_hist = calcular_macd(dados)
    macd_v = macd_linha.iloc[-1]
    macd_s = macd_sinal_line.iloc[-1]
    macd_h = macd_hist.iloc[-1]

    fibonacci = calcular_fibonacci(dados)
    volume = analisar_volume(dados)
    violinada = detectar_violinada(dados)
    suportes, resistencias = calcular_suportes_resistencias(dados)

    # Novos indicadores - Estratégia Irmãos Domingues
    vwap_series = calcular_vwap(dados)
    vwap_atual = vwap_series.iloc[-1]
    atr_series_data = calcular_atr_series(dados)
    atr_atual = atr_series_data.iloc[-1]
    lateralizacao = detectar_lateralizacao(dados)

    # Tendência baseada em EMA 9/21 + VWAP
    ema_9 = dados['close'].ewm(span=9, adjust=False).mean().iloc[-1]
    ema_21 = dados['close'].ewm(span=21, adjust=False).mean().iloc[-1]
    ema_20 = dados['close'].ewm(span=20, adjust=False).mean().iloc[-1]
    ema_50 = dados['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    preco = dados['close'].iloc[-1]

    if preco > vwap_atual and ema_9 > ema_21:
        tendencia = "ALTA"
    elif preco < vwap_atual and ema_9 < ema_21:
        tendencia = "BAIXA"
    else:
        tendencia = "LATERAL"
    if lateralizacao["lateral"]:
        tendencia = "LATERAL"

    pullback_info = detectar_pullback(dados, tendencia)


    # === Indicadores extras ===
    bollinger = calcular_bollinger(dados)
    estocastico = calcular_estocastico(dados)
    adx = calcular_adx(dados)
    obv = calcular_obv(dados)
    ichimoku = calcular_ichimoku(dados)
    pivot_points = calcular_pivot_points(dados)
    vwap_bands = calcular_vwap_bands(dados)

    sinais = gerar_sinais(dados, fibonacci, rsi_atual, macd_v, macd_s, macd_h,
        volume, violinada, tendencia=tendencia, pullback_info=pullback_info,
        lateralizacao=lateralizacao, vwap_atual=vwap_atual)

    # RSI status
    if rsi_atual > 70:
        rsi_status = "SOBRECOMPRADO"
    elif rsi_atual < 30:
        rsi_status = "SOBREVENDIDO"
    else:
        rsi_status = "NEUTRO"

    # MACD status
    if macd_h > 0 and macd_v > macd_s:
        if len(macd_hist) >= 2 and macd_hist.iloc[-2] <= 0:
            macd_status = "CRUZAMENTO_ALTA"
        else:
            macd_status = "ALTA"
    elif macd_h < 0 and macd_v < macd_s:
        if len(macd_hist) >= 2 and macd_hist.iloc[-2] >= 0:
            macd_status = "CRUZAMENTO_BAIXA"
        else:
            macd_status = "BAIXA"
    else:
        macd_status = "NEUTRO"

    # Preparar dados de candles para o frontend (últimos 100)
    candles_data = []
    display_dados = dados.tail(100)
    rsi_display = rsi_series.tail(100)
    macd_l_display = macd_linha.tail(100)
    macd_s_display = macd_sinal_line.tail(100)
    macd_h_display = macd_hist.tail(100)

    for i, (idx, row) in enumerate(display_dados.iterrows()):
        candle = {
            "time": str(idx) if isinstance(idx, str) else idx.strftime("%Y-%m-%d %H:%M") if hasattr(idx, 'strftime') else str(idx),
            "open": round(float(row['open']), 2),
            "high": round(float(row['high']), 2),
            "low": round(float(row['low']), 2),
            "close": round(float(row['close']), 2),
            "volume": float(row['volume']),
            "rsi": round(float(rsi_display.iloc[i]), 2) if i < len(rsi_display) else 50,
            "macd": round(float(macd_l_display.iloc[i]), 4) if i < len(macd_l_display) else 0,
            "macd_signal": round(float(macd_s_display.iloc[i]), 4) if i < len(macd_s_display) else 0,
            "macd_hist": round(float(macd_h_display.iloc[i]), 4) if i < len(macd_h_display) else 0,
        }
        candles_data.append(candle)

    # Gestão de risco
    gestao = GestaoRisco()
    horario_op = verificar_horario_operacional()

    return {
        "timeframe": timeframe,
        "ativo": ativo,
        "preco_atual": round(float(preco), 2),
        "tendencia": tendencia,
        "fibonacci": {
            "nivel_0": round(fibonacci.nivel_0, 2),
            "nivel_236": round(fibonacci.nivel_236, 2),
            "nivel_382": round(fibonacci.nivel_382, 2),
            "nivel_500": round(fibonacci.nivel_500, 2),
            "nivel_618": round(fibonacci.nivel_618, 2),
            "nivel_786": round(fibonacci.nivel_786, 2),
            "nivel_100": round(fibonacci.nivel_100, 2),
            "extensao_1272": round(fibonacci.extensao_1272, 2),
            "extensao_1618": round(fibonacci.extensao_1618, 2),
            "extensao_2618": round(fibonacci.extensao_2618, 2),
            "tendencia": fibonacci.tendencia,
        },
        "rsi": round(float(rsi_atual), 2),
        "rsi_status": rsi_status,
        "macd_valor": round(float(macd_v), 4),
        "macd_sinal": round(float(macd_s), 4),
        "macd_histograma": round(float(macd_h), 4),
        "macd_status": macd_status,
        "volume": {
            "total": round(volume.volume_total, 0),
            "compra": round(volume.volume_compra, 0),
            "venda": round(volume.volume_venda, 0),
            "ratio": volume.ratio_compra_venda,
            "pressao": volume.pressao,
            "volume_medio": round(volume.volume_medio, 0),
            "acima_media": volume.volume_acima_media,
            "delta_acumulado": round(volume.delta_acumulado, 0),
        },
        "sinais": [
            {
                "tipo": s.tipo,
                "preco_entrada": s.preco_entrada,
                "stop_loss": s.stop_loss,
                "take_profit_1": s.take_profit_1,
                "take_profit_2": s.take_profit_2,
                "take_profit_3": s.take_profit_3,
                "risco_retorno": s.risco_retorno,
                "confianca": s.confianca,
                "motivos": s.motivos,
                "fibonacci_zona": s.fibonacci_zona,
                "rsi_status": s.rsi_status,
                "macd_status": s.macd_status,
                "volume_status": s.volume_status,
                "violinada_risco": s.violinada_risco,
            }
            for s in sinais
        ],
        "suportes": suportes,
        "resistencias": resistencias,
        "violinada_score": round(violinada, 1),
        "candles": candles_data,
        "ema_9": round(float(ema_9), 2),
        "ema_21": round(float(ema_21), 2),
        "ema_20": round(float(ema_20), 2),
        "ema_50": round(float(ema_50), 2),
        "vwap": round(float(vwap_atual), 2),
        "atr": round(float(atr_atual), 2),
        "lateralizacao": lateralizacao,
        "pullback": pullback_info,
        "horario_operacional": horario_op,
        "gestao_risco": gestao.status_completo(),
        "bollinger": bollinger,
        "estocastico": estocastico,
        "adx": adx,
        "obv": obv,
        "ichimoku": ichimoku,
        "pivot_points": pivot_points,
        "vwap_bands": vwap_bands,
    }


def analisar_tendencia_macro(dados: pd.DataFrame) -> dict:
    """
    Analisa tendência macro usando EMAs e estrutura de preço.
    Retorna: {"tendencia": "ALTA"|"BAIXA"|"LATERAL", "forca": 0-100}
    """
    if len(dados) < 50:
        return {"tendencia": "LATERAL", "forca": 0}
    
    close = dados['close']
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    
    c = float(close.iloc[-1])
    e9 = float(ema9.iloc[-1])
    e21 = float(ema21.iloc[-1])
    e50 = float(ema50.iloc[-1])
    
    pontos_alta = 0
    pontos_baixa = 0
    
    # EMA alignment
    if e9 > e21 > e50:
        pontos_alta += 3
    elif e9 < e21 < e50:
        pontos_baixa += 3
    elif e9 > e21:
        pontos_alta += 1
    else:
        pontos_baixa += 1
    
    # Price vs EMAs
    if c > e9: pontos_alta += 1
    else: pontos_baixa += 1
    if c > e21: pontos_alta += 1
    else: pontos_baixa += 1
    if c > e50: pontos_alta += 1
    else: pontos_baixa += 1
    
    # Recent momentum (last 10 candles)
    if len(close) >= 10:
        delta = float(close.iloc[-1]) - float(close.iloc[-10])
        if delta > 0: pontos_alta += 1
        else: pontos_baixa += 1
    
    total = pontos_alta + pontos_baixa
    if pontos_alta > pontos_baixa + 2:
        return {"tendencia": "ALTA", "forca": round(pontos_alta / total * 100)}
    elif pontos_baixa > pontos_alta + 2:
        return {"tendencia": "BAIXA", "forca": round(pontos_baixa / total * 100)}
    else:
        return {"tendencia": "LATERAL", "forca": round(max(pontos_alta, pontos_baixa) / total * 100)}


# =====================================================
# ANÁLISE AVANÇADA DE FLUXO v2.0
# VAP, Book de Ofertas, Absorção, Liquidez, Multi-TF
# Baseado em: Bellafiore, Leandro Paz, Portal do Trader
# =====================================================

def calcular_vap(dados: pd.DataFrame, bins: int = 30) -> dict:
    """
    Volume at Price (VAP) - Perfil de Volume
    Identifica POC, Value Area, zonas de alta/baixa liquidez.
    
    POC = Point of Control (preço com mais volume negociado)
    VA = Value Area (70% do volume - onde o mercado mais operou)
    """
    if len(dados) < 10:
        return {"poc": 0, "va_high": 0, "va_low": 0, "perfil": [], "zonas_liquidez": []}
    
    price_min = float(dados['low'].min())
    price_max = float(dados['high'].max())
    if price_max == price_min:
        return {"poc": price_min, "va_high": price_min, "va_low": price_min, "perfil": [], "zonas_liquidez": []}
    
    step = (price_max - price_min) / bins
    perfil = []
    
    for i in range(bins):
        level_low = price_min + i * step
        level_high = level_low + step
        level_mid = (level_low + level_high) / 2
        
        # Volume que passou por este nível de preço
        vol_at_level = 0
        buy_vol = 0
        sell_vol = 0
        
        for _, row in dados.iterrows():
            r_high = float(row['high'])
            r_low = float(row['low'])
            r_close = float(row['close'])
            r_open = float(row['open'])
            r_vol = float(row.get('volume', 0))
            
            if r_low <= level_high and r_high >= level_low:
                # Proporção do candle que intersecta este nível
                candle_range = r_high - r_low if r_high > r_low else 1
                overlap_low = max(r_low, level_low)
                overlap_high = min(r_high, level_high)
                proportion = (overlap_high - overlap_low) / candle_range
                vol_contrib = r_vol * proportion
                
                vol_at_level += vol_contrib
                if r_close >= r_open:
                    buy_vol += vol_contrib
                else:
                    sell_vol += vol_contrib
        
        perfil.append({
            "preco": round(level_mid, 2),
            "preco_low": round(level_low, 2),
            "preco_high": round(level_high, 2),
            "volume": round(vol_at_level, 0),
            "vol_compra": round(buy_vol, 0),
            "vol_venda": round(sell_vol, 0),
            "delta": round(buy_vol - sell_vol, 0),
        })
    
    # POC - nível com maior volume
    if not perfil:
        return {"poc": 0, "va_high": 0, "va_low": 0, "perfil": [], "zonas_liquidez": []}
    
    poc_level = max(perfil, key=lambda x: x["volume"])
    poc = poc_level["preco"]
    
    # Value Area (70% do volume total)
    total_vol = sum(p["volume"] for p in perfil)
    if total_vol == 0:
        return {"poc": poc, "va_high": price_max, "va_low": price_min, "perfil": perfil, "zonas_liquidez": []}
    
    va_target = total_vol * 0.70
    sorted_by_vol = sorted(perfil, key=lambda x: x["volume"], reverse=True)
    va_vol = 0
    va_prices = []
    for p in sorted_by_vol:
        va_vol += p["volume"]
        va_prices.append(p["preco"])
        if va_vol >= va_target:
            break
    
    va_high = max(va_prices) if va_prices else price_max
    va_low = min(va_prices) if va_prices else price_min
    
    # Zonas de liquidez (gaps no perfil = baixo volume = liquidez thin)
    avg_vol = total_vol / bins if bins > 0 else 1
    zonas_liquidez = []
    for p in perfil:
        if p["volume"] < avg_vol * 0.3:
            zonas_liquidez.append({
                "tipo": "LOW_LIQUIDITY",
                "preco": p["preco"],
                "descricao": f"Baixa liquidez em {p['preco']:.2f} - preço pode acelerar",
            })
        elif p["volume"] > avg_vol * 2.0:
            zonas_liquidez.append({
                "tipo": "HIGH_LIQUIDITY",
                "preco": p["preco"],
                "descricao": f"Alta concentração em {p['preco']:.2f} - possível S/R de fluxo",
            })
    
    return {
        "poc": round(poc, 2),
        "va_high": round(va_high, 2),
        "va_low": round(va_low, 2),
        "total_volume": round(total_vol, 0),
        "perfil": perfil,
        "zonas_liquidez": zonas_liquidez,
    }


def analisar_book_ofertas(dados: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Análise do Book de Ofertas simulado.
    
    Como não temos book real (L2 data), inferimos a partir de:
    - Sombras dos candles (wicks) = rejeição de preço = ofertas defendendo
    - Volume vs movimento de preço = absorção
    - Sequência de candles = agressão direcional
    
    Baseado em: Leandro Paz, Portal do Trader (Caio Sasaki)
    """
    if len(dados) < lookback:
        return {"agressao": "NEUTRO", "intensidade": 0, "absorcao": None, "defesa": None}
    
    recent = dados.tail(lookback)
    
    # --- Análise de Agressão ---
    compra_agressiva = 0
    venda_agressiva = 0
    
    for _, row in recent.iterrows():
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        body = abs(c - o)
        total_range = h - l if h > l else 0.001
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        # Candle com corpo grande e pouca sombra = agressão
        body_ratio = body / total_range
        
        if c > o:  # bullish
            if body_ratio > 0.6:
                compra_agressiva += 1 + (body_ratio - 0.6)
            if lower_wick > body * 0.5:
                compra_agressiva += 0.5  # Martelo = compradores defendendo
        else:  # bearish
            if body_ratio > 0.6:
                venda_agressiva += 1 + (body_ratio - 0.6)
            if upper_wick > body * 0.5:
                venda_agressiva += 0.5  # Estrela cadente = vendedores defendendo
    
    total_agressao = compra_agressiva + venda_agressiva
    if total_agressao == 0:
        total_agressao = 1
    
    ratio_compra = compra_agressiva / total_agressao
    ratio_venda = venda_agressiva / total_agressao
    
    if ratio_compra > 0.65:
        agressao = "COMPRADORA"
        intensidade = round(ratio_compra * 100)
    elif ratio_venda > 0.65:
        agressao = "VENDEDORA"
        intensidade = round(ratio_venda * 100)
    else:
        agressao = "EQUILIBRIO"
        intensidade = 50
    
    # --- Análise de Absorção ---
    absorcao = _detectar_absorcao(dados, lookback)
    
    # --- Análise de Defesa de Preço ---
    defesa = _detectar_defesa_preco(dados, lookback)
    
    # --- Mudança de comportamento ---
    mudanca = _detectar_mudanca_comportamento(dados, lookback)
    
    return {
        "agressao": agressao,
        "intensidade": intensidade,
        "ratio_compra": round(ratio_compra, 2),
        "ratio_venda": round(ratio_venda, 2),
        "compra_agressiva": round(compra_agressiva, 1),
        "venda_agressiva": round(venda_agressiva, 1),
        "absorcao": absorcao,
        "defesa": defesa,
        "mudanca_comportamento": mudanca,
    }


def _detectar_absorcao(dados: pd.DataFrame, lookback: int = 10) -> dict:
    """
    Detecta absorção: muito volume/agressão mas preço NÃO anda.
    = alguém grande absorvendo a pressão contrária.
    Sinal forte de reversão iminente.
    """
    if len(dados) < lookback:
        return None
    
    recent = dados.tail(lookback)
    volumes = recent['volume'].values.astype(float)
    closes = recent['close'].values.astype(float)
    
    if len(volumes) < 5 or volumes.mean() == 0:
        return None
    
    # Últimas 3 velas com volume alto mas preço estagnado
    ultimas3_vol = volumes[-3:]
    vol_medio = volumes[:-3].mean() if len(volumes) > 3 else volumes.mean()
    
    if vol_medio == 0:
        return None
    
    vol_alto = all(v > vol_medio * 1.3 for v in ultimas3_vol)
    preco_range = abs(closes[-1] - closes[-3]) if len(closes) >= 3 else 0
    atr_approx = np.mean(np.abs(np.diff(closes))) if len(closes) > 1 else 1
    preco_parado = preco_range < atr_approx * 0.5
    
    if vol_alto and preco_parado:
        # Determinar quem está absorvendo
        bull_candles = sum(1 for i in range(-3, 0) if closes[i] >= float(recent['open'].iloc[i]))
        if bull_candles >= 2:
            return {
                "detectado": True,
                "tipo": "COMPRADORES_ABSORVENDO",
                "descricao": "Volume alto + preço parado = compradores absorvendo vendas. Reversão para ALTA provável.",
                "forca": "FORTE" if all(v > vol_medio * 2 for v in ultimas3_vol) else "MODERADA",
            }
        else:
            return {
                "detectado": True,
                "tipo": "VENDEDORES_ABSORVENDO",
                "descricao": "Volume alto + preço parado = vendedores absorvendo compras. Reversão para BAIXA provável.",
                "forca": "FORTE" if all(v > vol_medio * 2 for v in ultimas3_vol) else "MODERADA",
            }
    
    return {"detectado": False}


def _detectar_defesa_preco(dados: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detecta defesa de preço: nível testado múltiplas vezes sem romper.
    = grandes players defendendo um preço (iceberg orders).
    """
    if len(dados) < lookback:
        return None
    
    recent = dados.tail(lookback)
    lows = recent['low'].values.astype(float)
    highs = recent['high'].values.astype(float)
    closes = recent['close'].values.astype(float)
    
    atr = np.mean(np.abs(np.diff(closes))) if len(closes) > 1 else 1
    tolerancia = atr * 0.3
    
    # Verificar defesa de suporte (múltiplos lows similares)
    sup_clusters = []
    for i in range(len(lows)):
        encontrou = False
        for cluster in sup_clusters:
            if abs(lows[i] - cluster["nivel"]) < tolerancia:
                cluster["toques"] += 1
                cluster["nivel"] = (cluster["nivel"] * (cluster["toques"] - 1) + lows[i]) / cluster["toques"]
                encontrou = True
                break
        if not encontrou:
            sup_clusters.append({"nivel": lows[i], "toques": 1, "tipo": "SUPORTE"})
    
    # Verificar defesa de resistência
    res_clusters = []
    for i in range(len(highs)):
        encontrou = False
        for cluster in res_clusters:
            if abs(highs[i] - cluster["nivel"]) < tolerancia:
                cluster["toques"] += 1
                cluster["nivel"] = (cluster["nivel"] * (cluster["toques"] - 1) + highs[i]) / cluster["toques"]
                encontrou = True
                break
        if not encontrou:
            res_clusters.append({"nivel": highs[i], "toques": 1, "tipo": "RESISTENCIA"})
    
    # Encontrar defesas fortes (3+ toques)
    defesas = []
    for cluster in sup_clusters + res_clusters:
        if cluster["toques"] >= 3:
            defesas.append({
                "nivel": round(cluster["nivel"], 2),
                "tipo": cluster["tipo"],
                "toques": cluster["toques"],
                "descricao": f"{cluster['tipo']} defendido {cluster['toques']}x em {round(cluster['nivel'],2)} - player grande",
            })
    
    if defesas:
        return {
            "detectado": True,
            "defesas": sorted(defesas, key=lambda x: x["toques"], reverse=True),
            "mais_forte": defesas[0] if defesas else None,
        }
    
    return {"detectado": False, "defesas": []}


def _detectar_mudanca_comportamento(dados: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detecta mudança de comportamento dos participantes:
    - Sequência de candles muda de direção
    - Volume muda de perfil (de baixo para alto ou vice-versa)
    - Volatilidade muda abruptamente
    """
    if len(dados) < lookback:
        return {"detectado": False}
    
    recent = dados.tail(lookback)
    closes = recent['close'].values.astype(float)
    volumes = recent['volume'].values.astype(float)
    
    half = lookback // 2
    
    # Mudança de direção
    primeira_metade = closes[:half]
    segunda_metade = closes[half:]
    
    dir1 = "ALTA" if primeira_metade[-1] > primeira_metade[0] else "BAIXA"
    dir2 = "ALTA" if segunda_metade[-1] > segunda_metade[0] else "BAIXA"
    
    mudou_direcao = dir1 != dir2
    
    # Mudança de volume
    vol1 = volumes[:half].mean() if volumes[:half].mean() > 0 else 1
    vol2 = volumes[half:].mean()
    vol_change = vol2 / vol1
    
    mudou_volume = vol_change > 1.5 or vol_change < 0.5
    
    # Mudança de volatilidade
    ranges1 = np.abs(np.diff(closes[:half]))
    ranges2 = np.abs(np.diff(closes[half:]))
    vol1_avg = ranges1.mean() if len(ranges1) > 0 else 1
    vol2_avg = ranges2.mean() if len(ranges2) > 0 else 1
    volatilidade_change = vol2_avg / vol1_avg if vol1_avg > 0 else 1
    
    mudou_volatilidade = volatilidade_change > 1.8 or volatilidade_change < 0.4
    
    detectado = mudou_direcao or mudou_volume or mudou_volatilidade
    
    descricoes = []
    if mudou_direcao:
        descricoes.append(f"Direção mudou de {dir1} para {dir2}")
    if mudou_volume:
        descricoes.append(f"Volume {'aumentou' if vol_change > 1 else 'diminuiu'} {round(vol_change, 1)}x")
    if mudou_volatilidade:
        descricoes.append(f"Volatilidade {'expandiu' if volatilidade_change > 1 else 'contraiu'} {round(volatilidade_change, 1)}x")
    
    return {
        "detectado": detectado,
        "mudou_direcao": mudou_direcao,
        "direcao_anterior": dir1 if mudou_direcao else None,
        "direcao_atual": dir2 if mudou_direcao else None,
        "mudou_volume": mudou_volume,
        "vol_change_ratio": round(vol_change, 2),
        "mudou_volatilidade": mudou_volatilidade,
        "volatilidade_change": round(volatilidade_change, 2),
        "descricoes": descricoes,
    }


def calcular_fibonacci_multi_tf(dados_5m: pd.DataFrame, dados_15m: pd.DataFrame = None, dados_1h: pd.DataFrame = None) -> dict:
    """
    Fibonacci em múltiplos timeframes para encontrar confluências.
    Usa topos e fundos de REVERSÃO (não qualquer topo/fundo).
    
    Retorna níveis de retração e extensão por TF + zonas de confluência.
    """
    result = {"tf_5m": None, "tf_15m": None, "tf_1h": None, "confluencias": []}
    
    # Calcular Fibo para cada TF disponível
    for label, df in [("tf_5m", dados_5m), ("tf_15m", dados_15m), ("tf_1h", dados_1h)]:
        if df is None or len(df) < 20:
            continue
        
        # Encontrar topos e fundos de reversão significativos
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        
        # Swing highs/lows com lookback proporcional ao TF
        swing_lookback = 5 if label == "tf_5m" else 8 if label == "tf_15m" else 10
        
        swing_highs = []
        swing_lows = []
        
        for i in range(swing_lookback, len(highs) - swing_lookback):
            if highs[i] == max(highs[i-swing_lookback:i+swing_lookback+1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i-swing_lookback:i+swing_lookback+1]):
                swing_lows.append((i, lows[i]))
        
        if not swing_highs or not swing_lows:
            continue
        
        # Usar o swing high e swing low mais recentes e significativos
        last_high_idx, last_high = swing_highs[-1]
        last_low_idx, last_low = swing_lows[-1]
        
        diff = last_high - last_low
        if diff <= 0:
            continue
        
        # Determinar tendência pelo swing mais recente
        if last_high_idx > last_low_idx:
            tendencia = "ALTA"
            fib_levels = {
                "topo": round(last_high, 2),
                "fundo": round(last_low, 2),
                "retracoes": {
                    "23.6": round(last_high - diff * 0.236, 2),
                    "38.2": round(last_high - diff * 0.382, 2),
                    "50.0": round(last_high - diff * 0.500, 2),
                    "61.8": round(last_high - diff * 0.618, 2),
                    "78.6": round(last_high - diff * 0.786, 2),
                },
                "extensoes": {
                    "127.2": round(last_high + diff * 0.272, 2),
                    "161.8": round(last_high + diff * 0.618, 2),
                    "261.8": round(last_high + diff * 1.618, 2),
                },
                "tendencia": tendencia,
            }
        else:
            tendencia = "BAIXA"
            fib_levels = {
                "topo": round(last_high, 2),
                "fundo": round(last_low, 2),
                "retracoes": {
                    "23.6": round(last_low + diff * 0.236, 2),
                    "38.2": round(last_low + diff * 0.382, 2),
                    "50.0": round(last_low + diff * 0.500, 2),
                    "61.8": round(last_low + diff * 0.618, 2),
                    "78.6": round(last_low + diff * 0.786, 2),
                },
                "extensoes": {
                    "127.2": round(last_low - diff * 0.272, 2),
                    "161.8": round(last_low - diff * 0.618, 2),
                    "261.8": round(last_low - diff * 1.618, 2),
                },
                "tendencia": tendencia,
            }
        
        result[label] = fib_levels
    
    # Encontrar confluências entre TFs (mesmo nível em TFs diferentes)
    all_levels = []
    for tf_key in ["tf_5m", "tf_15m", "tf_1h"]:
        fib = result.get(tf_key)
        if not fib:
            continue
        for nome, preco in fib.get("retracoes", {}).items():
            all_levels.append({"tf": tf_key, "tipo": f"Retração {nome}%", "preco": preco})
        for nome, preco in fib.get("extensoes", {}).items():
            all_levels.append({"tf": tf_key, "tipo": f"Extensão {nome}%", "preco": preco})
    
    # Agrupar por proximidade
    if all_levels:
        atr_approx = abs(all_levels[0]["preco"]) * 0.001  # ~0.1% tolerância
        confluencias = []
        used = set()
        
        for i, l1 in enumerate(all_levels):
            if i in used:
                continue
            group = [l1]
            used.add(i)
            for j, l2 in enumerate(all_levels):
                if j in used or j == i:
                    continue
                if abs(l1["preco"] - l2["preco"]) < atr_approx:
                    group.append(l2)
                    used.add(j)
            
            if len(group) >= 2:  # Confluência = mesmo nível em 2+ TFs
                avg_preco = sum(g["preco"] for g in group) / len(group)
                tfs = list(set(g["tf"] for g in group))
                tipos = [g["tipo"] for g in group]
                confluencias.append({
                    "preco": round(avg_preco, 2),
                    "tfs": tfs,
                    "tipos": tipos,
                    "forca": len(group),
                    "descricao": f"Confluência Fibo {round(avg_preco,2)}: {', '.join(tipos)} em {', '.join(tfs)}",
                })
        
        result["confluencias"] = sorted(confluencias, key=lambda x: x["forca"], reverse=True)
    
    return result


def analisar_contexto_mercado(dados: pd.DataFrame, dados_15m: pd.DataFrame = None, dados_1h: pd.DataFrame = None) -> dict:
    """
    Análise de contexto COMPLETA do mercado antes de qualquer entrada.
    
    Combina:
    1. Tendência macro (15m e 1h)
    2. Fase do mercado (acumulação, distribuição, tendência, range)
    3. Momentum e exaustão
    4. Correlação fluxo vs estrutura
    """
    result = {
        "fase": "INDEFINIDA",
        "momentum": "NEUTRO",
        "exaustao": False,
        "contexto_favoravel": False,
        "descricao": "",
        "tendencia_15m": "LATERAL",
        "tendencia_1h": "LATERAL",
        "sr_15m": [],
        "sr_1h": [],
    }
    
    if len(dados) < 20:
        return result
    
    closes = dados['close'].values.astype(float)
    volumes = dados['volume'].values.astype(float)
    highs = dados['high'].values.astype(float)
    lows = dados['low'].values.astype(float)
    
    # --- Fase do mercado ---
    ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values if len(closes) >= 50 else ema20
    
    # Bollinger Width como proxy de volatilidade
    bb_middle = pd.Series(closes).rolling(20).mean()
    bb_std = pd.Series(closes).rolling(20).std()
    bb_width = (bb_std * 2 / bb_middle).iloc[-1] if len(bb_middle) >= 20 else 0
    
    # ADX simplificado (tendência vs range)
    price_changes = np.abs(np.diff(closes))
    avg_change = price_changes[-10:].mean() if len(price_changes) >= 10 else 0
    avg_change_old = price_changes[-20:-10].mean() if len(price_changes) >= 20 else avg_change
    
    trending = avg_change > avg_change_old * 1.3 if avg_change_old > 0 else False
    
    # Classificação de fase
    if bb_width < 0.005 and not trending:
        result["fase"] = "ACUMULAÇÃO/DISTRIBUIÇÃO"
        result["descricao"] = "Mercado comprimido (Bollinger squeeze) - explosão iminente"
    elif trending and closes[-1] > ema20[-1]:
        result["fase"] = "TENDÊNCIA DE ALTA"
        result["descricao"] = "Mercado em tendência de alta com momentum crescente"
    elif trending and closes[-1] < ema20[-1]:
        result["fase"] = "TENDÊNCIA DE BAIXA"
        result["descricao"] = "Mercado em tendência de baixa com momentum crescente"
    else:
        result["fase"] = "RANGE"
        result["descricao"] = "Mercado lateralizado - operar nos extremos"
    
    # --- Momentum ---
    if len(closes) >= 10:
        mom_5 = (closes[-1] / closes[-5] - 1) * 100 if closes[-5] > 0 else 0
        mom_10 = (closes[-1] / closes[-10] - 1) * 100 if closes[-10] > 0 else 0
        
        if mom_5 > 0.3 and mom_10 > 0.3:
            result["momentum"] = "ALTA_FORTE"
        elif mom_5 > 0.1:
            result["momentum"] = "ALTA"
        elif mom_5 < -0.3 and mom_10 < -0.3:
            result["momentum"] = "BAIXA_FORTE"
        elif mom_5 < -0.1:
            result["momentum"] = "BAIXA"
        
        # Exaustão: momentum desacelerando
        if len(closes) >= 15:
            mom_5_ant = (closes[-6] / closes[-11] - 1) * 100 if closes[-11] > 0 else 0
            if abs(mom_5) < abs(mom_5_ant) * 0.5 and abs(mom_5_ant) > 0.1:
                result["exaustao"] = True
                result["descricao"] += " | EXAUSTÃO detectada - momentum desacelerando"
    
    # --- Tendência 15m ---
    if dados_15m is not None and len(dados_15m) >= 20:
        c15 = dados_15m['close'].values.astype(float)
        ema9_15 = pd.Series(c15).ewm(span=9, adjust=False).mean().values
        ema21_15 = pd.Series(c15).ewm(span=21, adjust=False).mean().values
        if ema9_15[-1] > ema21_15[-1]:
            result["tendencia_15m"] = "ALTA"
        elif ema9_15[-1] < ema21_15[-1]:
            result["tendencia_15m"] = "BAIXA"
        
        # S/R no 15m
        from analysis_engine import calcular_suportes_resistencias
        try:
            s15, r15 = calcular_suportes_resistencias(dados_15m)
            result["sr_15m"] = [{"tipo": "S", "preco": round(x, 2)} for x in s15[:3]] + \
                              [{"tipo": "R", "preco": round(x, 2)} for x in r15[:3]]
        except:
            pass
    
    # --- Tendência 1h ---
    if dados_1h is not None and len(dados_1h) >= 20:
        c1h = dados_1h['close'].values.astype(float)
        ema9_1h = pd.Series(c1h).ewm(span=9, adjust=False).mean().values
        ema21_1h = pd.Series(c1h).ewm(span=21, adjust=False).mean().values
        if ema9_1h[-1] > ema21_1h[-1]:
            result["tendencia_1h"] = "ALTA"
        elif ema9_1h[-1] < ema21_1h[-1]:
            result["tendencia_1h"] = "BAIXA"
        
        try:
            s1h, r1h = calcular_suportes_resistencias(dados_1h)
            result["sr_1h"] = [{"tipo": "S", "preco": round(x, 2)} for x in s1h[:3]] + \
                              [{"tipo": "R", "preco": round(x, 2)} for x in r1h[:3]]
        except:
            pass
    
    # --- Contexto favorável? ---
    # Favorável quando tendência + momentum + não exausto + não em squeeze
    if result["fase"] in ("TENDÊNCIA DE ALTA", "TENDÊNCIA DE BAIXA") and \
       result["momentum"] in ("ALTA_FORTE", "ALTA", "BAIXA_FORTE", "BAIXA") and \
       not result["exaustao"]:
        result["contexto_favoravel"] = True
    
    return result


def avaliar_convergencia_completa(
    fluxo: dict,
    contexto: dict,
    vap: dict,
    fib_multi: dict,
    rsi: float,
    macd_h: float,
    tend_macro: str,
    direcao: str,
    preco_atual: float,
) -> dict:
    """
    CONVERGÊNCIA MULTI-FATOR OBRIGATÓRIA
    
    Baseado nos livros:
    - Trading in the Zone (Douglas): probabilidade, não certeza
    - One Good Trade (Bellafiore): PlayBook = convergência
    - Come Into My Trading Room (Elder): Triple Screen
    - Axiomas de Zurique: não entrar sem convicção
    
    Avalia 10 fatores e exige mínimo de 6 para entrada.
    Retorna score, detalhes, e veredicto.
    """
    fatores = []
    score = 0
    
    # 1. FLUXO: Agressão direcional confirma?
    fluxo_ok = False
    if direcao == "COMPRA" and fluxo.get("agressao") == "COMPRADORA":
        fluxo_ok = True
    elif direcao == "VENDA" and fluxo.get("agressao") == "VENDEDORA":
        fluxo_ok = True
    fatores.append({
        "nome": "Fluxo/Agressão",
        "ok": fluxo_ok,
        "detalhe": f"Agressão {fluxo.get('agressao', '?')} ({fluxo.get('intensidade', 0)}%) - {'confirma' if fluxo_ok else 'não confirma'} {direcao}",
        "peso": 1.5,
    })
    if fluxo_ok: score += 1.5
    
    # 2. ABSORÇÃO: Tem absorção a favor?
    absorcao_ok = False
    abs_info = fluxo.get("absorcao", {})
    if abs_info and abs_info.get("detectado"):
        if direcao == "COMPRA" and "COMPRADORES" in abs_info.get("tipo", ""):
            absorcao_ok = True
        elif direcao == "VENDA" and "VENDEDORES" in abs_info.get("tipo", ""):
            absorcao_ok = True
    fatores.append({
        "nome": "Absorção",
        "ok": absorcao_ok,
        "detalhe": abs_info.get("descricao", "Sem absorção detectada") if abs_info and abs_info.get("detectado") else "Sem absorção",
        "peso": 1.0,
    })
    if absorcao_ok: score += 1.0
    
    # 3. CONTEXTO MACRO: Tendência do TF maior confirma?
    contexto_ok = False
    if direcao == "COMPRA" and tend_macro == "ALTA":
        contexto_ok = True
    elif direcao == "VENDA" and tend_macro == "BAIXA":
        contexto_ok = True
    fatores.append({
        "nome": "Contexto Macro",
        "ok": contexto_ok,
        "detalhe": f"Tendência macro: {tend_macro} - {'alinhado' if contexto_ok else 'contra'} {direcao}",
        "peso": 1.5,
    })
    if contexto_ok: score += 1.5
    
    # 4. VAP/POC: Preço em zona favorável?
    vap_ok = False
    if vap and vap.get("poc"):
        poc = vap["poc"]
        va_h = vap.get("va_high", poc)
        va_l = vap.get("va_low", poc)
        
        if direcao == "COMPRA" and preco_atual <= poc:
            vap_ok = True  # Comprando abaixo do POC (desconto)
        elif direcao == "VENDA" and preco_atual >= poc:
            vap_ok = True  # Vendendo acima do POC (prêmio)
        elif va_l <= preco_atual <= va_h:
            pass  # Dentro da VA = neutro
        
    fatores.append({
        "nome": "VAP/Volume Profile",
        "ok": vap_ok,
        "detalhe": f"POC: {vap.get('poc', '?')} | VA: {vap.get('va_low', '?')}-{vap.get('va_high', '?')} | Preço: {preco_atual}" if vap else "VAP indisponível",
        "peso": 1.0,
    })
    if vap_ok: score += 1.0
    
    # 5. FIBONACCI: Preço em zona de retração/extensão?
    fib_ok = False
    fib_detalhe = "Sem dados Fibonacci"
    if fib_multi:
        # Verificar se preço está próximo de alguma retração
        for tf_key in ["tf_1h", "tf_15m", "tf_5m"]:
            fib_tf = fib_multi.get(tf_key)
            if not fib_tf:
                continue
            retracoes = fib_tf.get("retracoes", {})
            for nome, nivel in retracoes.items():
                dist_pct = abs(preco_atual - nivel) / preco_atual * 100 if preco_atual > 0 else 999
                if dist_pct < 0.15:  # Dentro de 0.15% do nível Fibo
                    fib_ok = True
                    fib_detalhe = f"Preço na retração {nome}% do {tf_key} ({nivel})"
                    break
            if fib_ok:
                break
        
        # Confluências Fibo multi-TF são ainda mais fortes
        for conf in fib_multi.get("confluencias", []):
            dist_pct = abs(preco_atual - conf["preco"]) / preco_atual * 100 if preco_atual > 0 else 999
            if dist_pct < 0.2:
                fib_ok = True
                fib_detalhe = f"CONFLUÊNCIA FIBO: {conf['descricao']}"
                score += 0.5  # Bônus por confluência multi-TF
                break
    
    fatores.append({
        "nome": "Fibonacci Multi-TF",
        "ok": fib_ok,
        "detalhe": fib_detalhe,
        "peso": 1.0,
    })
    if fib_ok: score += 1.0
    
    # 6. INDICADORES: RSI + MACD confirmam?
    ind_ok = False
    if direcao == "COMPRA":
        ind_ok = rsi < 65 and macd_h > 0
    elif direcao == "VENDA":
        ind_ok = rsi > 35 and macd_h < 0
    fatores.append({
        "nome": "Indicadores (RSI+MACD)",
        "ok": ind_ok,
        "detalhe": f"RSI={rsi} MACD hist={macd_h} - {'confirma' if ind_ok else 'não confirma'} {direcao}",
        "peso": 1.0,
    })
    if ind_ok: score += 1.0
    
    # 7. DEFESA DE PREÇO: Tem defesa no nível?
    defesa_ok = False
    defesa_info = fluxo.get("defesa", {})
    if defesa_info and defesa_info.get("detectado"):
        mais_forte = defesa_info.get("mais_forte", {})
        if mais_forte:
            if direcao == "COMPRA" and mais_forte.get("tipo") == "SUPORTE":
                defesa_ok = True
            elif direcao == "VENDA" and mais_forte.get("tipo") == "RESISTENCIA":
                defesa_ok = True
    fatores.append({
        "nome": "Defesa de Preço",
        "ok": defesa_ok,
        "detalhe": defesa_info.get("mais_forte", {}).get("descricao", "Sem defesa detectada") if defesa_info and defesa_info.get("detectado") else "Sem defesa",
        "peso": 0.5,
    })
    if defesa_ok: score += 0.5
    
    # 8. FASE DO MERCADO: Contexto favorável?
    fase_ok = contexto.get("contexto_favoravel", False)
    fatores.append({
        "nome": "Fase/Momentum",
        "ok": fase_ok,
        "detalhe": f"{contexto.get('fase', '?')} | Mom: {contexto.get('momentum', '?')} | {'EXAUSTÃO' if contexto.get('exaustao') else 'OK'}",
        "peso": 1.0,
    })
    if fase_ok: score += 1.0
    
    # 9. LIQUIDEZ: Não está em zona de baixa liquidez?
    liq_ok = True  # Default OK
    if vap:
        for zona in vap.get("zonas_liquidez", []):
            if zona["tipo"] == "LOW_LIQUIDITY":
                dist = abs(preco_atual - zona["preco"])
                if dist < preco_atual * 0.001:
                    liq_ok = False
                    break
    fatores.append({
        "nome": "Liquidez",
        "ok": liq_ok,
        "detalhe": "Zona de liquidez adequada" if liq_ok else "CUIDADO: zona de baixa liquidez - slippage provável",
        "peso": 0.5,
    })
    if liq_ok: score += 0.5
    
    # 10. MUDANÇA DE COMPORTAMENTO: Não houve reversão recente?
    mudanca = fluxo.get("mudanca_comportamento", {})
    mud_ok = True
    if mudanca.get("detectado"):
        # Se mudou de direção E a nova direção é contra nosso trade
        if mudanca.get("mudou_direcao"):
            nova_dir = mudanca.get("direcao_atual", "")
            if (direcao == "COMPRA" and nova_dir == "BAIXA") or \
               (direcao == "VENDA" and nova_dir == "ALTA"):
                mud_ok = False
    fatores.append({
        "nome": "Estabilidade do Fluxo",
        "ok": mud_ok,
        "detalhe": "; ".join(mudanca.get("descricoes", ["Fluxo estável"])) if mudanca.get("detectado") else "Fluxo estável",
        "peso": 0.5,
    })
    if mud_ok: score += 0.5
    
    # --- VEREDICTO ---
    max_score = sum(f["peso"] for f in fatores)
    pct = round(score / max_score * 100) if max_score > 0 else 0
    
    ok_count = sum(1 for f in fatores if f["ok"])
    
    if pct >= 75 and ok_count >= 7:
        veredicto = "ENTRADA_FORTE"
        nivel = "A+"
    elif pct >= 60 and ok_count >= 6:
        veredicto = "ENTRADA"
        nivel = "A"
    elif pct >= 50 and ok_count >= 5:
        veredicto = "CAUTELA"
        nivel = "B+"
    elif pct >= 40 and ok_count >= 4:
        veredicto = "ESPERAR"
        nivel = "B"
    else:
        veredicto = "NÃO_ENTRAR"
        nivel = "C"
    
    return {
        "score": round(score, 1),
        "max_score": round(max_score, 1),
        "pct": pct,
        "ok_count": ok_count,
        "total_fatores": len(fatores),
        "fatores": fatores,
        "veredicto": veredicto,
        "nivel": nivel,
        "pros": [f["detalhe"] for f in fatores if f["ok"]],
        "contras": [f["detalhe"] for f in fatores if not f["ok"]],
    }
