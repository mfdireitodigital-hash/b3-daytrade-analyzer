"""
PRO TRADER ANALYSIS - Sistema de análise profissional
Baseado no Trader Pro Skill: Triple Screen (Elder), Confluence Checklist,
PlayBook (Bellafiore), Price Action, Fibonacci, SMC, Volume.

O trader profissional NÃO entra em qualquer sinal.
Ele espera o setup aparecer no GRÁFICO - confluência REAL.

"One Good Trade" - Bellafiore: Foque em QUALIDADE, não quantidade.
"Trading in the Zone" - Douglas: Cada momento é único, pense em probabilidades.
"Come Into My Trading Room" - Elder: Triple Screen = tendência + sinal + entrada.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def calcular_tendencia_macro(dados_window, ativo: str = "WIN") -> Dict:
    """
    TELA 1 - Elder Triple Screen: Tendência do timeframe MAIOR.
    Usa toda a janela disponível (100+ velas de 5min = ~8h).
    
    Critérios (Murphy + Elder):
    - EMA50 slope: inclinação da média longa
    - Estrutura de topos/fundos: ascendentes=ALTA, descendentes=BAIXA
    - Preço vs EMA50: acima=viés comprador, abaixo=viés vendedor
    """
    result = {
        "tendencia": "LATERAL",
        "forca": 0,  # -3 a +3
        "ema50_slope": 0,
        "estrutura": "indefinida",
        "descricao": ""
    }
    
    if len(dados_window) < 50:
        return result
    
    closes = dados_window['close'].astype(float).values
    highs = dados_window['high'].astype(float).values
    lows = dados_window['low'].astype(float).values
    
    # EMA50
    ema50 = _ema(closes, 50)
    if len(ema50) < 20:
        return result
    
    # Slope da EMA50 (últimas 20 velas)
    slope = (ema50[-1] - ema50[-20]) / ema50[-20] * 100  # % change
    result["ema50_slope"] = round(slope, 3)
    
    # Estrutura de topos e fundos (swing points nos últimos 50 candles)
    swing_highs = []
    swing_lows = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append((i, highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append((i, lows[i]))
    
    # Verificar estrutura
    topos_asc = False
    fundos_asc = False
    if len(swing_highs) >= 2:
        topos_asc = swing_highs[-1][1] > swing_highs[-2][1]
    if len(swing_lows) >= 2:
        fundos_asc = swing_lows[-1][1] > swing_lows[-2][1]
    
    forca = 0
    # Preço vs EMA50
    preco_acima_ema50 = closes[-1] > ema50[-1]
    if preco_acima_ema50:
        forca += 1
    else:
        forca -= 1
    
    # Slope
    if ativo == "WDO":
        slope_threshold = 0.01  # WDO move menos
    else:
        slope_threshold = 0.05
    
    if slope > slope_threshold:
        forca += 1
    elif slope < -slope_threshold:
        forca -= 1
    
    # Estrutura
    if topos_asc and fundos_asc:
        forca += 1
        result["estrutura"] = "topos_fundos_ascendentes"
    elif not topos_asc and not fundos_asc:
        forca -= 1
        result["estrutura"] = "topos_fundos_descendentes"
    else:
        result["estrutura"] = "mista"
    
    result["forca"] = forca
    if forca >= 2:
        result["tendencia"] = "ALTA"
        result["descricao"] = "Tendência ALTA: EMA50 subindo + estrutura compradora"
    elif forca <= -2:
        result["tendencia"] = "BAIXA"
        result["descricao"] = "Tendência BAIXA: EMA50 caindo + estrutura vendedora"
    elif forca == 1:
        result["tendencia"] = "ALTA"
        result["descricao"] = "Viés de ALTA moderado"
    elif forca == -1:
        result["tendencia"] = "BAIXA"
        result["descricao"] = "Viés de BAIXA moderado"
    else:
        result["tendencia"] = "LATERAL"
        result["descricao"] = "Mercado LATERAL - sem tendência definida"
    
    return result


def detectar_setup_profissional(
    w,  # window de dados (últimas 100 velas)
    vela,  # vela atual (Series)
    pos_idx: int,  # índice no DataFrame
    dados,  # DataFrame completo
    day_indices: List[int],  # índices do dia
    ativo: str,
    tend_macro: Dict,  # resultado de calcular_tendencia_macro
    rsi_v: float,
    macd_h: float,
    ema9: float,
    ema21: float,
    atr_v: float,
    operacoes_anteriores: List[Dict],  # operações já feitas (para evitar repetição)
) -> Dict:
    """
    Análise completa de confluência profissional.
    
    CONFLUENCE CHECKLIST (Trader Pro - 7 fatores):
    1. Tendência TF maior confirma? (Elder T1)
    2. Preço em nível S/R relevante? (Murphy)
    3. Volume confirma? (Bellafiore - "In Play")
    4. Indicadores confirmam? (RSI + MACD + VWAP alinhados)
    5. Price Action confirma? (Candlestick pattern)
    6. Risco definido R:R >= 1:2? (Elder 3Ms)
    7. Fibonacci confirma? (Retracement/Extension)
    
    REGRA: 4+ = operar. 5-7 = A+ setup. 3 = C+ (reduzido). <3 = NÃO.
    """
    
    o = float(vela['open']); h = float(vela['high'])
    l = float(vela['low']); c = float(vela['close'])
    vol = int(vela.get('volume', 0))
    
    closes = w['close'].astype(float).values if len(w) > 0 else []
    highs = w['high'].astype(float).values if len(w) > 0 else []
    lows = w['low'].astype(float).values if len(w) > 0 else []
    
    # ===== DETERMINAR DIREÇÃO DO SINAL =====
    # Primeiro: qual direção o mercado indica?
    direcao = None
    motivos_direcao = []
    motivos_direcao_extra = []  # motivos extras da definição de direção
    
    # EMA alignment
    if ema9 > ema21 and c > ema9:
        motivos_direcao.append("COMPRA")
    elif ema9 < ema21 and c < ema9:
        motivos_direcao.append("VENDA")
    
    # RSI zones (Elder: osciladores em tendência)
    if rsi_v < 35 and tend_macro["tendencia"] != "BAIXA":
        motivos_direcao.append("COMPRA")  # Sobrevendido contra tendência = pullback compra
    elif rsi_v > 65 and tend_macro["tendencia"] != "ALTA":
        motivos_direcao.append("VENDA")  # Sobrecomprado contra tendência = pullback venda
    elif rsi_v < 45 and tend_macro["tendencia"] == "BAIXA":
        motivos_direcao.append("VENDA")  # RSI fraco em baixa = continuação
    elif rsi_v > 55 and tend_macro["tendencia"] == "ALTA":
        motivos_direcao.append("COMPRA")  # RSI forte em alta = continuação
    
    # MACD histogram
    if macd_h > 0:
        motivos_direcao.append("COMPRA")
    elif macd_h < 0:
        motivos_direcao.append("VENDA")
    
    # Count direction votes
    compra_votes = sum(1 for m in motivos_direcao if m == "COMPRA")
    venda_votes = sum(1 for m in motivos_direcao if m == "VENDA")
    
    if compra_votes > venda_votes:
        direcao = "COMPRA"
    elif venda_votes > compra_votes:
        direcao = "VENDA"
    elif compra_votes == venda_votes and compra_votes > 0:
        # Empate - usar tendência macro como desempate (Elder: Tela 1 decide)
        if tend_macro["tendencia"] == "ALTA":
            direcao = "COMPRA"
            motivos_direcao_extra.append("Empate indicadores - macro ALTA desempata para COMPRA")
        elif tend_macro["tendencia"] == "BAIXA":
            direcao = "VENDA"
            motivos_direcao_extra.append("Empate indicadores - macro BAIXA desempata para VENDA")
        # LATERAL = sem desempate, fica None
    
    # Fallback: nenhum indicador votou mas macro tem direção clara
    if direcao is None and tend_macro["tendencia"] in ("ALTA", "BAIXA"):
        direcao = "COMPRA" if tend_macro["tendencia"] == "ALTA" else "VENDA"
        motivos_direcao_extra.append(f"Indicadores neutros - macro {tend_macro['tendencia']} define direção (Elder: Tela 1 prevalece)")
    
    # ===== CONFLUENCE CHECKLIST =====
    confluencia = {}
    motivos_operar = list(motivos_direcao_extra)  # manter motivos de direção
    motivos_nao_operar = []
    
    # --- FATOR 1: Tendência TF maior (Elder T1) ---
    if direcao:
        if (direcao == "COMPRA" and tend_macro["tendencia"] == "ALTA") or \
           (direcao == "VENDA" and tend_macro["tendencia"] == "BAIXA"):
            confluencia["tendencia_tf_maior"] = True
            motivos_operar.append(f"Triple Screen T1: {direcao} alinhado com tendência {tend_macro['tendencia']} ({tend_macro['descricao']})")
        elif tend_macro["tendencia"] == "LATERAL":
            confluencia["tendencia_tf_maior"] = False
            motivos_nao_operar.append(f"T1 LATERAL - sem tendência definida no TF maior")
        else:
            confluencia["tendencia_tf_maior"] = False
            motivos_nao_operar.append(f"CONTRA TENDÊNCIA: {direcao} contra {tend_macro['tendencia']} (Elder proíbe)")
    else:
        confluencia["tendencia_tf_maior"] = False
        motivos_nao_operar.append("Sem direção clara nos indicadores")
    
    # --- FATOR 2: S/R relevante (Murphy) ---
    suporte = None
    resistencia = None
    confluencia["sr_relevante"] = False
    
    if len(w) >= 20:
        # Encontrar S/R dos últimos 50 candles (mais significativos)
        lookback = min(50, len(w) - 1)
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # S/R por swing points (mais preciso que min/max simples)
        sr_levels = _encontrar_sr_levels(highs[-lookback:], lows[-lookback:], closes[-lookback:])
        
        # Também usar min/max como fallback
        resistencia = float(max(recent_highs))
        suporte = float(min(recent_lows))
        
        # Verificar se preço está PERTO de algum nível S/R
        dist_threshold = atr_v * 0.5  # dentro de 0.5x ATR
        
        proximo_sr = False
        sr_descricao = ""
        
        for level in sr_levels:
            dist = abs(c - level["preco"])
            if dist < dist_threshold:
                proximo_sr = True
                sr_descricao = f"Preço em {level['tipo']} {round(level['preco'], 2)} ({level['toques']} toques)"
                break
        
        # Fallback: distância do suporte/resistência clássico
        if not proximo_sr:
            dist_sup = abs(c - suporte)
            dist_res = abs(resistencia - c)
            if direcao == "COMPRA" and dist_sup < dist_threshold:
                proximo_sr = True
                sr_descricao = f"Preço próximo suporte {round(suporte, 2)}"
            elif direcao == "VENDA" and dist_res < dist_threshold:
                proximo_sr = True
                sr_descricao = f"Preço próximo resistência {round(resistencia, 2)}"
        
        if proximo_sr:
            confluencia["sr_relevante"] = True
            motivos_operar.append(f"S/R Murphy: {sr_descricao}")
        else:
            motivos_nao_operar.append(f"Preço longe de S/R relevante (S:{round(suporte,2)} R:{round(resistencia,2)})")
    
    # --- FATOR 3: Volume confirma (Bellafiore - "In Play") ---
    confluencia["volume_confirma"] = False
    vol_ratio = 0
    
    if vol > 0 and len(w) >= 10:
        avg_vol = float(w['volume'].iloc[-10:].replace(0, np.nan).mean())
        if avg_vol and avg_vol > 0:
            vol_ratio = vol / avg_vol
            if vol_ratio > 1.3:  # Volume 30%+ acima da média
                confluencia["volume_confirma"] = True
                motivos_operar.append(f"Volume {round(vol_ratio, 1)}x acima da média - institucional confirmando")
            elif vol_ratio < 0.5:
                motivos_nao_operar.append(f"Volume fraco ({round(vol_ratio, 1)}x) - sem participação")
            # Volume normal (0.5-1.3): neutro, não penaliza nem confirma
    else:
        # Sem dados de volume (yfinance às vezes não traz)
        # NÃO penalizar - muitos ativos não têm volume confiável no yfinance
        confluencia["volume_confirma"] = True  # Dar benefício da dúvida
        motivos_operar.append("Volume: dados indisponíveis (neutro)")
    
    # --- FATOR 4: Indicadores confirmam (RSI + MACD + VWAP) ---
    # Precisamos pelo menos 2 de 3 indicadores alinhados
    ind_confirmados = 0
    ind_total = 0
    
    # RSI
    if direcao == "COMPRA":
        if 30 <= rsi_v <= 65:  # Espaço pra subir, não sobrecomprado
            ind_confirmados += 1
            motivos_operar.append(f"RSI {rsi_v} - espaço para alta")
        elif rsi_v < 30:  # Sobrevendido = oportunidade de compra
            ind_confirmados += 1
            motivos_operar.append(f"RSI sobrevendido ({rsi_v}) - pullback de compra")
        else:
            motivos_nao_operar.append(f"RSI {rsi_v} sobrecomprado para compra")
    elif direcao == "VENDA":
        if 35 <= rsi_v <= 70:  # Espaço pra cair
            ind_confirmados += 1
            motivos_operar.append(f"RSI {rsi_v} - espaço para queda")
        elif rsi_v > 70:
            ind_confirmados += 1
            motivos_operar.append(f"RSI sobrecomprado ({rsi_v}) - pullback de venda")
        else:
            motivos_nao_operar.append(f"RSI {rsi_v} sobrevendido para venda")
    ind_total += 1
    
    # MACD
    if (direcao == "COMPRA" and macd_h > 0) or (direcao == "VENDA" and macd_h < 0):
        ind_confirmados += 1
        motivos_operar.append(f"MACD histograma {macd_h} confirma {direcao}")
    else:
        motivos_nao_operar.append(f"MACD {macd_h} diverge do sinal")
    ind_total += 1
    
    # VWAP
    vwap = None
    if len(w) >= 10:
        try:
            typical = (w['high'].astype(float) + w['low'].astype(float) + w['close'].astype(float)) / 3
            vol_s = w['volume'].replace(0, 1).astype(float)
            vwap = float((typical * vol_s).cumsum().iloc[-1] / vol_s.cumsum().iloc[-1])
            
            if (direcao == "COMPRA" and c > vwap) or (direcao == "VENDA" and c < vwap):
                ind_confirmados += 1
                motivos_operar.append(f"VWAP {round(vwap, 2)} - preço na direção correta")
            else:
                motivos_nao_operar.append(f"Preço {'abaixo' if direcao == 'COMPRA' else 'acima'} do VWAP {round(vwap, 2)}")
        except:
            pass
    ind_total += 1
    
    confluencia["indicadores_confirmam"] = ind_confirmados >= 2
    
    # --- FATOR 5: Price Action confirma (Candlestick + Estrutura) ---
    confluencia["price_action_confirma"] = False
    price_action = {}
    
    if len(w) >= 3:
        pa_result = _analisar_price_action(w, c, o, h, l, direcao, ativo)
        price_action = pa_result["detalhes"]
        if pa_result["confirma"]:
            confluencia["price_action_confirma"] = True
            motivos_operar.extend(pa_result["motivos"])
        else:
            motivos_nao_operar.extend(pa_result.get("motivos_contra", []))
    
    # --- FATOR 6: Risco definido R:R >= 1:2 ---
    # Sempre verdadeiro pois configuramos R:R 1:2 fixo
    confluencia["risco_definido"] = True
    motivos_operar.append("R:R mínimo 1:2 (Van Tharp: expectativa positiva)")
    
    # --- FATOR 7: Fibonacci ---
    confluencia["fibonacci_confirma"] = False
    fib_level = None
    fib_data = {}
    
    if len(w) >= 30:
        fib_result = _analisar_fibonacci(w, c, atr_v, direcao, ativo)
        fib_data = fib_result.get("data", {})
        if fib_result["confirma"]:
            confluencia["fibonacci_confirma"] = True
            fib_level = fib_result["nivel"]
            motivos_operar.append(f"Fibonacci {fib_level}: zona de reversão em {round(fib_result['preco'], 2)}")
    
    # ===== CLASSIFICAÇÃO DO SETUP (Bellafiore PlayBook) =====
    total_confluencia = sum(1 for v in confluencia.values() if v)
    
    # Qualidade (Bellafiore adaptado: analista sênior decide com qualquer nível)
    if total_confluencia >= 6:
        qualidade = "A+"
        confianca = 5
    elif total_confluencia >= 5:
        qualidade = "A"
        confianca = 5
    elif total_confluencia >= 4:
        qualidade = "B+"
        confianca = 4
    elif total_confluencia >= 3:
        qualidade = "C+"
        confianca = 3
    elif total_confluencia >= 2:
        qualidade = "C"
        confianca = 2
    else:
        qualidade = "D"
        confianca = 1
    
    # ===== FILTRO CONTRA-TENDÊNCIA =====
    # Elder: cuidado contra Tela 1, mas NÃO bloquear totalmente
    contra_tendencia = False
    if direcao and confluencia.get("tendencia_tf_maior") is False:
        if tend_macro["tendencia"] != "LATERAL":
            if total_confluencia < 3:
                contra_tendencia = True
                motivos_nao_operar.append(f"Contra tendência macro ({tend_macro['tendencia']}) com apenas {total_confluencia}/7 - risco alto")
            elif total_confluencia < 5:
                # Permite mas avisa - sizing reduzido
                contra_tendencia = False
                motivos_operar.append(f"Contra tendência macro ({tend_macro['tendencia']}) mas {total_confluencia}/7 confluências = PERMITIDO (sizing -50%)")
            else:
                contra_tendencia = False
                motivos_operar.append("Contra tendência macro mas confluência forte (5+) = tamanho cheio")
    
    # ===== FILTRO REPETIÇÃO DE ENTRADA =====
    # Bellafiore: "Second chance" - não repita no mesmo nível
    entrada_repetida = False
    if direcao and operacoes_anteriores:
        for op_ant in operacoes_anteriores:
            dist = abs(c - op_ant.get("preco_entrada", 0))
            if dist < atr_v * 1.5 and op_ant.get("tipo") == direcao:
                entrada_repetida = True
                motivos_nao_operar.append(f"Entrada similar já feita em {op_ant['preco_entrada']} ({op_ant['hora_entrada']})")
                break
    
    # ===== DECISÃO FINAL =====
    # Analista sênior: opera com 2+ confluências se tiver direção
    # Não precisa 7/7 - nenhum trade real tem tudo alinhado
    operar = (
        direcao is not None
        and qualidade in ("A+", "A", "B+", "C+", "C")
        and not contra_tendencia
        and not entrada_repetida
        and total_confluencia >= 2
    )
    
    # Ajuste de sizing por qualidade (Bellafiore)
    # A+/A = tamanho cheio, B+ = normal, C+ = reduzido mas OPERA
    if qualidade in ("C+", "C") and operar:
        motivos_operar.append(f"{qualidade} ({total_confluencia}/7) = entrada válida com sizing reduzido")
    
    return {
        "direcao": direcao,
        "confluencia": confluencia,
        "total_confluencia": total_confluencia,
        "qualidade": qualidade,
        "confianca": confianca,
        "operar": operar,
        "contra_tendencia": contra_tendencia,
        "entrada_repetida": entrada_repetida,
        "motivos_operar": motivos_operar,
        "motivos_nao_operar": motivos_nao_operar,
        "suporte": suporte,
        "resistencia": resistencia,
        "vwap": vwap,
        "fib_level": fib_level,
        "fib_data": fib_data,
        "price_action": price_action,
        "vol_ratio": vol_ratio,
        "tend_macro": tend_macro,
    }


def gerar_analise_completa(setup: Dict, vela_info: Dict, ativo: str) -> str:
    """
    Gera a narrativa completa da análise como um mentor profissional faria.
    Baseado em Bellafiore (PlayBook), Elder (Triple Screen), Douglas (probabilidades).
    """
    d = setup["direcao"] or "INDEFINIDO"
    c = vela_info
    
    txt = f"{'='*60}\n"
    txt += f"ANÁLISE PROFISSIONAL - {d} às {c.get('hora', '?')}\n"
    txt += f"PlayBook: Setup {setup['qualidade']} ({setup['total_confluencia']}/7 confluências)\n"
    txt += f"{'='*60}\n\n"
    
    # Triple Screen
    tm = setup["tend_macro"]
    txt += f"1. TRIPLE SCREEN (Elder):\n"
    txt += f"   T1 (Macro): {tm['tendencia']} - {tm['descricao']}\n"
    txt += f"   T2 (Sinal): {d}\n"
    txt += f"   Alinhamento: {'✓ CONFIRMADO' if setup['confluencia'].get('tendencia_tf_maior') else '✗ DIVERGENTE'}\n\n"
    
    # Confluência detalhada
    txt += f"2. CHECKLIST DE CONFLUÊNCIA (7 fatores):\n"
    labels = {
        "tendencia_tf_maior": "Tendência TF maior",
        "sr_relevante": "S/R relevante",
        "volume_confirma": "Volume confirma",
        "indicadores_confirmam": "Indicadores (RSI+MACD+VWAP)",
        "price_action_confirma": "Price Action confirma",
        "risco_definido": "Risco R:R definido",
        "fibonacci_confirma": "Fibonacci confirma",
    }
    for key, label in labels.items():
        status = "✓" if setup["confluencia"].get(key) else "✗"
        txt += f"   {status} {label}\n"
    txt += f"   TOTAL: {setup['total_confluencia']}/7\n\n"
    
    # Motivos
    txt += f"3. FATORES A FAVOR:\n"
    for m in setup["motivos_operar"]:
        txt += f"   + {m}\n"
    txt += f"\n4. FATORES CONTRA:\n"
    for m in setup["motivos_nao_operar"]:
        txt += f"   - {m}\n"
    
    txt += f"\n5. DECISÃO: {'OPERAR' if setup['operar'] else 'NÃO OPERAR'} - {setup['qualidade']}\n"
    if setup["contra_tendencia"]:
        txt += f"   ⚠ CONTRA TENDÊNCIA - Elder proíbe\n"
    if setup["entrada_repetida"]:
        txt += f"   ⚠ ENTRADA REPETIDA - Bellafiore: não repetir no mesmo nível\n"
    
    return txt


# ===== HELPERS =====

def _ema(data, period):
    """Calcula EMA simples"""
    if len(data) < period:
        return data
    multiplier = 2 / (period + 1)
    ema = [float(data[0])]
    for i in range(1, len(data)):
        ema.append((float(data[i]) - ema[-1]) * multiplier + ema[-1])
    return ema


def _encontrar_sr_levels(highs, lows, closes) -> List[Dict]:
    """
    Encontra níveis de S/R significativos por zona de preço.
    Murphy: quanto mais toques, mais forte o nível.
    """
    levels = []
    all_prices = list(highs) + list(lows)
    if not all_prices:
        return levels
    
    price_range = max(all_prices) - min(all_prices)
    if price_range == 0:
        return levels
    
    # Divide em zonas de 1% do range
    zone_size = price_range * 0.01
    if zone_size == 0:
        zone_size = 1
    
    # Conta toques por zona
    zones = {}
    for p in all_prices:
        zone_key = round(p / zone_size) * zone_size
        zones[zone_key] = zones.get(zone_key, 0) + 1
    
    # Zonas com 3+ toques = S/R significativo
    current_price = float(closes[-1]) if len(closes) > 0 else 0
    for preco, toques in sorted(zones.items(), key=lambda x: -x[1]):
        if toques >= 3:
            tipo = "SUPORTE" if preco < current_price else "RESISTÊNCIA"
            levels.append({"preco": preco, "toques": toques, "tipo": tipo})
    
    return levels[:6]  # Top 6 níveis


def _analisar_price_action(w, c, o, h, l, direcao, ativo) -> Dict:
    """
    Análise completa de Price Action:
    - Candlestick patterns (Nison)
    - LTA/LTB (Tendência)
    - Rompimento de topo/fundo
    - Quebra de estrutura (BOS)
    """
    result = {
        "confirma": False,
        "motivos": [],
        "motivos_contra": [],
        "detalhes": {}
    }
    
    closes = w['close'].astype(float).values
    highs = w['high'].astype(float).values
    lows = w['low'].astype(float).values
    opens = w['open'].astype(float).values
    
    body = abs(c - o)
    upper_shadow = h - max(c, o)
    lower_shadow = min(c, o) - l
    total_range = h - l if h > l else 0.0001
    
    patterns_found = []
    
    # ===== CANDLESTICK PATTERNS (Nison) =====
    if len(w) >= 3:
        prev_c = float(w['close'].iloc[-2])
        prev_o = float(w['open'].iloc[-2])
        prev_body = abs(prev_c - prev_o)
        prev_h = float(w['high'].iloc[-2])
        prev_l = float(w['low'].iloc[-2])
        
        # Martelo (Hammer) - reversão altista
        if lower_shadow > body * 2 and upper_shadow < body * 0.5 and body > 0:
            if direcao == "COMPRA":
                patterns_found.append("Martelo (reversão altista)")
        
        # Estrela Cadente (Shooting Star) - reversão baixista
        if upper_shadow > body * 2 and lower_shadow < body * 0.5 and body > 0:
            if direcao == "VENDA":
                patterns_found.append("Estrela Cadente (reversão baixista)")
        
        # Engolfo de Alta
        if c > o and prev_c < prev_o and body > prev_body * 1.1 and c > prev_o and o < prev_c:
            if direcao == "COMPRA":
                patterns_found.append("Engolfo de Alta (forte reversão)")
        
        # Engolfo de Baixa
        if o > c and prev_c > prev_o and body > prev_body * 1.1 and o > prev_c and c < prev_o:
            if direcao == "VENDA":
                patterns_found.append("Engolfo de Baixa (forte reversão)")
        
        # Doji (indecisão)
        if body < total_range * 0.1:
            if upper_shadow > total_range * 0.3 and lower_shadow > total_range * 0.3:
                patterns_found.append("Doji - indecisão forte")
        
        # Pin Bar (rejeição de nível)
        if (lower_shadow > body * 3 or upper_shadow > body * 3) and body > 0:
            if lower_shadow > body * 3 and direcao == "COMPRA":
                patterns_found.append("Pin Bar altista (rejeição de fundo)")
            elif upper_shadow > body * 3 and direcao == "VENDA":
                patterns_found.append("Pin Bar baixista (rejeição de topo)")
        
        # Inside Bar (compressão antes de movimento)
        if h <= prev_h and l >= prev_l:
            patterns_found.append("Inside Bar (compressão = movimento iminente)")
    
    # ===== LTA/LTB =====
    if len(w) >= 10:
        # Swing points para LTA/LTB
        sw_highs = []
        sw_lows = []
        for i in range(1, min(len(highs) - 1, 20)):
            if i > 0 and i < len(highs) - 1:
                if highs[-(i+1)] > highs[-i] and highs[-(i+1)] > highs[-(i+2)]:
                    sw_highs.append(float(highs[-(i+1)]))
                if lows[-(i+1)] < lows[-i] and lows[-(i+1)] < lows[-(i+2)]:
                    sw_lows.append(float(lows[-(i+1)]))
        
        if len(sw_lows) >= 2 and sw_lows[0] > sw_lows[1]:
            result["detalhes"]["lta"] = True
            result["detalhes"]["lta_pontos"] = [sw_lows[1], sw_lows[0]]
            if direcao == "COMPRA":
                patterns_found.append("LTA ativa (fundos ascendentes)")
        
        if len(sw_highs) >= 2 and sw_highs[0] < sw_highs[1]:
            result["detalhes"]["ltd"] = True
            result["detalhes"]["ltd_pontos"] = [sw_highs[1], sw_highs[0]]
            if direcao == "VENDA":
                patterns_found.append("LTB ativa (topos descendentes)")
        
        # Rompimento de topo/fundo
        if sw_highs and c > max(sw_highs):
            result["detalhes"]["rompimento_topo"] = True
            result["detalhes"]["topo_rompido"] = max(sw_highs)
            if direcao == "COMPRA":
                patterns_found.append(f"ROMPIMENTO DE TOPO em {round(max(sw_highs), 2)}")
        
        if sw_lows and c < min(sw_lows):
            result["detalhes"]["rompimento_fundo"] = True
            result["detalhes"]["fundo_rompido"] = min(sw_lows)
            if direcao == "VENDA":
                patterns_found.append(f"ROMPIMENTO DE FUNDO em {round(min(sw_lows), 2)}")
        
        # Estrutura (BOS - Break of Structure)
        if len(sw_highs) >= 2 and len(sw_lows) >= 2:
            topos_asc = sw_highs[0] > sw_highs[1]
            fundos_asc = sw_lows[0] > sw_lows[1]
            if topos_asc and fundos_asc:
                result["detalhes"]["estrutura"] = "ALTA"
            elif not topos_asc and not fundos_asc:
                result["detalhes"]["estrutura"] = "BAIXA"
            else:
                result["detalhes"]["estrutura"] = "QUEBRA"
                patterns_found.append("QUEBRA DE ESTRUTURA - mudança de direção possível")
    

    # ===== PULLBACK ANALYSIS =====
    if len(w) >= 15:
        # Pullback = preço corrige contra tendência e depois retoma
        # Detecta via EMAs: preço toca/cruza EMA e volta
        ema9_vals = w['close'].ewm(span=9, adjust=False).mean().values
        ema21_vals = w['close'].ewm(span=21, adjust=False).mean().values
        
        # Tendência definida por EMAs
        ema_trend_up = float(ema9_vals[-1]) > float(ema21_vals[-1])
        ema_trend_down = float(ema9_vals[-1]) < float(ema21_vals[-1])
        
        # Pullback de alta: tendência de alta, preço tocou/cruzou EMA21 e fechou acima
        if ema_trend_up and len(lows) >= 3:
            # Vela anterior tocou EMA21 (low <= ema21) e vela atual fechou acima EMA9
            prev_touched_ema = float(lows[-2]) <= float(ema21_vals[-2]) * 1.001
            curr_above_ema = c > float(ema9_vals[-1])
            if prev_touched_ema and curr_above_ema:
                result["detalhes"]["pullback"] = "ALTA"
                result["detalhes"]["pullback_nivel"] = round(float(ema21_vals[-1]), 2)
                if direcao == "COMPRA":
                    patterns_found.append(f"PULLBACK DE ALTA na EMA21 ({round(float(ema21_vals[-1]),2)}) - retomou tendência")
        
        # Pullback de baixa: tendência de baixa, preço tocou/cruzou EMA21 e fechou abaixo
        if ema_trend_down and len(highs) >= 3:
            prev_touched_ema = float(highs[-2]) >= float(ema21_vals[-2]) * 0.999
            curr_below_ema = c < float(ema9_vals[-1])
            if prev_touched_ema and curr_below_ema:
                result["detalhes"]["pullback"] = "BAIXA"
                result["detalhes"]["pullback_nivel"] = round(float(ema21_vals[-1]), 2)
                if direcao == "VENDA":
                    patterns_found.append(f"PULLBACK DE BAIXA na EMA21 ({round(float(ema21_vals[-1]),2)}) - retomou tendência")
        
        # Pullback em Fibonacci (38.2% ou 50% ou 61.8%)
        if len(w) >= 20:
            recent_high = float(max(highs[-20:]))
            recent_low = float(min(lows[-20:]))
            fib_range = recent_high - recent_low
            if fib_range > 0:
                fib_382 = recent_high - fib_range * 0.382
                fib_500 = recent_high - fib_range * 0.500
                fib_618 = recent_high - fib_range * 0.618
                
                tolerance = fib_range * 0.02  # 2% tolerance
                
                if ema_trend_up and direcao == "COMPRA":
                    for fib_name, fib_val in [("38.2%", fib_382), ("50%", fib_500), ("61.8%", fib_618)]:
                        if abs(l - fib_val) < tolerance and c > o:
                            patterns_found.append(f"Pullback em Fibo {fib_name} ({round(fib_val,2)}) - rejeitou e subiu")
                            result["detalhes"]["pullback_fibo"] = fib_name
                            break
                
                if ema_trend_down and direcao == "VENDA":
                    fib_382_inv = recent_low + fib_range * 0.382
                    fib_500_inv = recent_low + fib_range * 0.500
                    fib_618_inv = recent_low + fib_range * 0.618
                    for fib_name, fib_val in [("38.2%", fib_382_inv), ("50%", fib_500_inv), ("61.8%", fib_618_inv)]:
                        if abs(h - fib_val) < tolerance and c < o:
                            patterns_found.append(f"Pullback em Fibo {fib_name} ({round(fib_val,2)}) - rejeitou e caiu")
                            result["detalhes"]["pullback_fibo"] = fib_name
                            break
    
    # ===== FALSO ROMPIMENTO (Fakeout / Bull Trap / Bear Trap) =====
    if len(w) >= 10 and len(closes) >= 5:
        # Falso rompimento de TOPO: preço rompeu high anterior mas fechou ABAIXO
        recent_highs = [float(x) for x in highs[-10:-1]]  # últimas 9 highs (sem a atual)
        recent_lows = [float(x) for x in lows[-10:-1]]
        
        if recent_highs:
            max_recent = max(recent_highs)
            min_recent = min(recent_lows)
            
            # Bull Trap: rompeu topo mas fechou abaixo
            if h > max_recent and c < max_recent and c < o:
                result["detalhes"]["falso_rompimento"] = "TOPO"
                result["detalhes"]["falso_rompimento_nivel"] = round(max_recent, 2)
                if direcao == "VENDA":
                    patterns_found.append(f"FALSO ROMPIMENTO DE TOPO (Bull Trap) em {round(max_recent,2)} - entrada de VENDA")
                elif direcao == "COMPRA":
                    result["motivos_contra"].append(f"CUIDADO: Possível Bull Trap em {round(max_recent,2)}")
            
            # Bear Trap: rompeu fundo mas fechou acima
            if l < min_recent and c > min_recent and c > o:
                result["detalhes"]["falso_rompimento"] = "FUNDO"
                result["detalhes"]["falso_rompimento_nivel"] = round(min_recent, 2)
                if direcao == "COMPRA":
                    patterns_found.append(f"FALSO ROMPIMENTO DE FUNDO (Bear Trap) em {round(min_recent,2)} - entrada de COMPRA")
                elif direcao == "VENDA":
                    result["motivos_contra"].append(f"CUIDADO: Possível Bear Trap em {round(min_recent,2)}")
    
    # ===== CAPTURA DE LIQUIDEZ (Liquidity Sweep/Grab) =====
    if len(w) >= 15:
        # Captura = preço varre stops (abaixo de suporte ou acima de resistência) e volta rápido
        # Conceito SMC: smart money captura liquidez dos retail traders
        
        # Encontra swing points recentes
        sw_hi = []
        sw_lo = []
        for i in range(2, min(len(highs) - 1, 15)):
            if highs[-(i+1)] > highs[-i] and highs[-(i+1)] > highs[-(i+2)]:
                sw_hi.append(float(highs[-(i+1)]))
            if lows[-(i+1)] < lows[-i] and lows[-(i+1)] < lows[-(i+2)]:
                sw_lo.append(float(lows[-(i+1)]))
        
        if sw_lo:
            # Captura de liquidez de baixo: wick violou fundo anterior mas corpo fechou acima
            nearest_low = min(sw_lo)
            if l < nearest_low and c > nearest_low and c > o:
                result["detalhes"]["captura_liquidez"] = "COMPRA"
                result["detalhes"]["captura_liquidez_nivel"] = round(nearest_low, 2)
                if direcao == "COMPRA":
                    patterns_found.append(f"CAPTURA DE LIQUIDEZ abaixo de {round(nearest_low,2)} - smart money comprando")
        
        if sw_hi:
            # Captura de liquidez de cima: wick violou topo anterior mas corpo fechou abaixo
            nearest_high = max(sw_hi)
            if h > nearest_high and c < nearest_high and c < o:
                result["detalhes"]["captura_liquidez"] = "VENDA"
                result["detalhes"]["captura_liquidez_nivel"] = round(nearest_high, 2)
                if direcao == "VENDA":
                    patterns_found.append(f"CAPTURA DE LIQUIDEZ acima de {round(nearest_high,2)} - smart money vendendo")
    
    # ===== S/R EM TIMEFRAME SUPERIOR (usando janela maior) =====
    if len(w) >= 50:
        # Simula TF superior usando janelas de 30-50 candles (equivale a olhar H1 em M5)
        tf_sup_highs = highs[-50:]
        tf_sup_lows = lows[-50:]
        tf_sup_closes = closes[-50:]
        
        # Agrupa em blocos de 6 candles (simula 30min com candles de 5min)
        block_size = 6
        tf_blocks_hi = []
        tf_blocks_lo = []
        for b in range(0, len(tf_sup_highs) - block_size + 1, block_size):
            tf_blocks_hi.append(float(max(tf_sup_highs[b:b+block_size])))
            tf_blocks_lo.append(float(min(tf_sup_lows[b:b+block_size])))
        
        # Swing points no TF superior
        tf_swing_highs = []
        tf_swing_lows = []
        for i in range(1, len(tf_blocks_hi) - 1):
            if tf_blocks_hi[i] > tf_blocks_hi[i-1] and tf_blocks_hi[i] > tf_blocks_hi[i+1]:
                tf_swing_highs.append(tf_blocks_hi[i])
            if tf_blocks_lo[i] < tf_blocks_lo[i-1] and tf_blocks_lo[i] < tf_blocks_lo[i+1]:
                tf_swing_lows.append(tf_blocks_lo[i])
        
        if tf_swing_highs or tf_swing_lows:
            result["detalhes"]["sr_tf_superior"] = {
                "topos": [round(x, 2) for x in tf_swing_highs[-3:]],
                "fundos": [round(x, 2) for x in tf_swing_lows[-3:]],
            }
            
            # Verifica se preço está próximo de S/R do TF superior
            atr_proxy = float(max(highs[-20:]) - min(lows[-20:])) / 10 if len(highs) >= 20 else 50
            
            for sr in tf_swing_lows:
                if abs(c - sr) < atr_proxy:
                    patterns_found.append(f"Próximo de SUPORTE TF superior ({round(sr,2)})")
                    result["detalhes"]["perto_sr_sup"] = {"tipo": "SUPORTE", "nivel": round(sr, 2)}
                    break
            
            for sr in tf_swing_highs:
                if abs(c - sr) < atr_proxy:
                    patterns_found.append(f"Próximo de RESISTÊNCIA TF superior ({round(sr,2)})")
                    result["detalhes"]["perto_sr_sup"] = {"tipo": "RESISTÊNCIA", "nivel": round(sr, 2)}
                    break


    # ===== RESULTADO =====
    if patterns_found:
        result["confirma"] = True
        result["motivos"] = [f"Price Action: {p}" for p in patterns_found]
    else:
        result["motivos_contra"] = ["Sem padrão de candlestick confirmando entrada"]
    
    result["detalhes"]["patterns"] = patterns_found
    
    return result


def _analisar_fibonacci(w, c, atr_v, direcao, ativo) -> Dict:
    """
    Fibonacci retracement do swing recente.
    Murphy: 38.2%, 50%, 61.8% são os principais.
    """
    result = {"confirma": False, "nivel": None, "preco": 0, "data": {}}
    
    closes = w['close'].astype(float).values
    highs = w['high'].astype(float).values
    lows = w['low'].astype(float).values
    
    # Usar últimos 30-50 candles para o swing
    lookback = min(50, len(highs))
    swing_high = float(max(highs[-lookback:]))
    swing_low = float(min(lows[-lookback:]))
    fib_range = swing_high - swing_low
    
    if fib_range <= 0:
        return result
    
    # Níveis de Fibonacci
    fibs = {
        "23.6%": swing_high - fib_range * 0.236,
        "38.2%": swing_high - fib_range * 0.382,
        "50.0%": swing_high - fib_range * 0.500,
        "61.8%": swing_high - fib_range * 0.618,
        "78.6%": swing_high - fib_range * 0.786,
    }
    
    result["data"] = {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "levels": {k: round(v, 2) for k, v in fibs.items()}
    }
    
    # Threshold: preço dentro de 0.3x ATR do nível Fibonacci
    threshold = atr_v * 0.3
    if ativo == "WDO":
        threshold = max(threshold, 0.002)  # mínimo 2 pips para WDO
    
    # Procurar nível mais próximo
    for nome, preco_fib in fibs.items():
        if abs(c - preco_fib) < threshold:
            result["confirma"] = True
            result["nivel"] = nome
            result["preco"] = preco_fib
            break
    
    return result
