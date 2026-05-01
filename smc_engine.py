"""
SMC ENGINE - Smart Money Concepts
Detecta padrões institucionais: FVG, Liquidity Sweep, Order Blocks, BOS, CHoCH.

Baseado em:
- ICT (Inner Circle Trader) concepts
- FVG detection (apolo_cdaudt)  
- Liquidity Sweep + SMC analysis (Vortex Trade / antonyburse)
- Order Flow institutional patterns
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def detectar_fvg(dados, pos_idx, tipo_sinal=None):
    """
    Fair Value Gap (FVG) - Gap institucional.
    bullFVG: high[2] < low[0] (gap para cima = compra institucional)
    bearFVG: low[2] > high[0] (gap para baixo = venda institucional)
    
    Retorna: (tem_fvg, tipo_fvg, detalhes)
    """
    try:
        if pos_idx < 2:
            return False, None, ""
        
        v0 = dados.iloc[pos_idx]      # vela atual
        v2 = dados.iloc[pos_idx - 2]  # 2 velas atrás
        
        h0 = float(v0['high']); l0 = float(v0['low'])
        h2 = float(v2['high']); l2 = float(v2['low'])
        
        # Bull FVG: high de 2 velas atrás < low da vela atual
        # = gap de preço para cima, institucionais compraram forte
        bull_fvg = h2 < l0
        
        # Bear FVG: low de 2 velas atrás > high da vela atual
        # = gap de preço para baixo, institucionais venderam forte
        bear_fvg = l2 > h0
        
        if bull_fvg:
            gap_size = l0 - h2
            return True, "BULL_FVG", f"Fair Value Gap ALTISTA detectado ({round(gap_size,1)}pts). Institucionais compraram forte - preco saltou deixando gap. Suporte em {round(h2,0)}-{round(l0,0)}"
        
        if bear_fvg:
            gap_size = l2 - h0
            return True, "BEAR_FVG", f"Fair Value Gap BAIXISTA detectado ({round(gap_size,1)}pts). Institucionais venderam forte - preco caiu deixando gap. Resistencia em {round(h0,0)}-{round(l2,0)}"
        
        return False, None, ""
    except Exception as e:
        logger.error(f"Erro FVG: {e}")
        return False, None, ""


def detectar_liquidity_sweep(dados, pos_idx, lookback=20):
    """
    Liquidity Sweep - Varredura de liquidez.
    O preço varre acima da máxima recente (sweep high) ou abaixo da mínima (sweep low)
    e depois REVERTE. Isso indica que smart money pegou a liquidez dos stops.
    
    Retorna: (tem_sweep, tipo_sweep, detalhes)
    """
    try:
        if pos_idx < lookback + 1:
            return False, None, ""
        
        v = dados.iloc[pos_idx]
        v_prev = dados.iloc[pos_idx - 1]
        
        h = float(v['high']); l = float(v['low'])
        c = float(v['close']); o = float(v['open'])
        
        # Calcular máxima e mínima das últimas lookback velas (excluindo a atual)
        recent = dados.iloc[max(0, pos_idx - lookback):pos_idx]
        recent_high = float(recent['high'].max())
        recent_low = float(recent['low'].min())
        
        # Sweep High: high da vela atual > máxima recente, mas close ABAIXO dela
        # = varreu os stops dos vendidos, pegou liquidez, e reverteu
        sweep_high = h > recent_high and c < recent_high and c < o
        
        # Sweep Low: low da vela atual < mínima recente, mas close ACIMA dela
        # = varreu os stops dos comprados, pegou liquidez, e reverteu
        sweep_low = l < recent_low and c > recent_low and c > o
        
        if sweep_high:
            return True, "SWEEP_HIGH", f"Liquidity Sweep HIGH detectado! Preco varou maxima {round(recent_high,0)} (ate {round(h,0)}) e REVERTEU fechando em {round(c,0)}. Smart Money vendeu no topo - sinal VENDA"
        
        if sweep_low:
            return True, "SWEEP_LOW", f"Liquidity Sweep LOW detectado! Preco varou minima {round(recent_low,0)} (ate {round(l,0)}) e REVERTEU fechando em {round(c,0)}. Smart Money comprou no fundo - sinal COMPRA"
        
        return False, None, ""
    except Exception as e:
        logger.error(f"Erro Liquidity Sweep: {e}")
        return False, None, ""


def detectar_order_block(dados, pos_idx, lookback=10):
    """
    Order Block - Última vela contrária antes de um movimento forte.
    É onde os institucionais acumularam posições.
    
    Bull OB: última vela vermelha antes de uma sequência de altas
    Bear OB: última vela verde antes de uma sequência de baixas
    
    Retorna: (tem_ob, tipo_ob, ob_zone, detalhes)
    """
    try:
        if pos_idx < lookback:
            return False, None, None, ""
        
        v = dados.iloc[pos_idx]
        c = float(v['close']); o = float(v['open'])
        h = float(v['high']); l = float(v['low'])
        
        # Procurar order blocks recentes
        for i in range(max(1, pos_idx - lookback), pos_idx - 2):
            vi = dados.iloc[i]
            vi_next = dados.iloc[i + 1]
            vi_next2 = dados.iloc[i + 2] if i + 2 < pos_idx else None
            
            oi = float(vi['open']); ci = float(vi['close'])
            hi = float(vi['high']); li = float(vi['low'])
            
            c_next = float(vi_next['close']); o_next = float(vi_next['open'])
            
            # Bull Order Block: vela vermelha seguida de 2+ velas verdes fortes
            is_bearish = ci < oi
            next_bullish = c_next > o_next
            
            if is_bearish and next_bullish:
                # Verificar se a vela seguinte foi forte (corpo > 70% range)
                next_body = abs(c_next - o_next)
                next_range = float(vi_next['high']) - float(vi_next['low'])
                
                if next_range > 0 and next_body / next_range > 0.6:
                    # Order block zone = range da vela bearish
                    ob_zone = (li, hi)
                    
                    # Preço atual está próximo da zona do OB?
                    if li <= c <= hi or abs(c - hi) < (hi - li) * 0.5:
                        return True, "BULL_OB", ob_zone, f"Order Block ALTISTA em {round(li,0)}-{round(hi,0)} (zona institucional de compra). Preco retornando ao OB = oportunidade COMPRA"
            
            # Bear Order Block: vela verde seguida de 2+ velas vermelhas fortes
            is_bullish = ci > oi
            next_bearish = c_next < o_next
            
            if is_bullish and next_bearish:
                next_body = abs(c_next - o_next)
                next_range = float(vi_next['high']) - float(vi_next['low'])
                
                if next_range > 0 and next_body / next_range > 0.6:
                    ob_zone = (li, hi)
                    
                    if li <= c <= hi or abs(c - li) < (hi - li) * 0.5:
                        return True, "BEAR_OB", ob_zone, f"Order Block BAIXISTA em {round(li,0)}-{round(hi,0)} (zona institucional de venda). Preco retornando ao OB = oportunidade VENDA"
        
        return False, None, None, ""
    except Exception as e:
        logger.error(f"Erro Order Block: {e}")
        return False, None, None, ""


def detectar_bos_choch(dados, pos_idx, lookback=20):
    """
    BOS (Break of Structure) e CHoCH (Change of Character).
    
    BOS: preço rompe último high/low na MESMA direção da tendência (continuação)
    CHoCH: preço rompe último high/low na direção CONTRÁRIA (reversão potencial)
    
    Retorna: (tipo, detalhes)
    """
    try:
        if pos_idx < lookback:
            return None, ""
        
        recent = dados.iloc[max(0, pos_idx - lookback):pos_idx + 1]
        v = dados.iloc[pos_idx]
        c = float(v['close']); h = float(v['high']); l = float(v['low'])
        
        # Encontrar swing highs e swing lows recentes
        highs = []
        lows = []
        for i in range(2, len(recent) - 2):
            vi = recent.iloc[i]
            vi_h = float(vi['high']); vi_l = float(vi['low'])
            
            # Swing high: high > vizinhos
            if vi_h > float(recent.iloc[i-1]['high']) and vi_h > float(recent.iloc[i-2]['high']) and \
               vi_h > float(recent.iloc[i+1]['high']):
                highs.append(vi_h)
            
            # Swing low: low < vizinhos
            if vi_l < float(recent.iloc[i-1]['low']) and vi_l < float(recent.iloc[i-2]['low']) and \
               vi_l < float(recent.iloc[i+1]['low']):
                lows.append(vi_l)
        
        if not highs or not lows:
            return None, ""
        
        last_high = highs[-1]
        last_low = lows[-1]
        
        # Tendência recente: highs subindo = alta, lows descendo = baixa
        trend_up = len(highs) >= 2 and highs[-1] > highs[-2]
        trend_down = len(lows) >= 2 and lows[-1] < lows[-2]
        
        # BOS em alta: rompe último swing high na tendência de alta
        if h > last_high and trend_up:
            return "BOS_ALTA", f"Break of Structure ALTISTA! Rompeu swing high {round(last_high,0)} - continuação da tendência de alta (SMC)"
        
        # BOS em baixa: rompe último swing low na tendência de baixa
        if l < last_low and trend_down:
            return "BOS_BAIXA", f"Break of Structure BAIXISTA! Rompeu swing low {round(last_low,0)} - continuação da tendência de baixa (SMC)"
        
        # CHoCH: rompe na direção CONTRÁRIA
        if l < last_low and trend_up:
            return "CHOCH_BAIXA", f"Change of Character! Mercado era ALTA mas rompeu low {round(last_low,0)} - possível REVERSÃO para baixa (SMC)"
        
        if h > last_high and trend_down:
            return "CHOCH_ALTA", f"Change of Character! Mercado era BAIXA mas rompeu high {round(last_high,0)} - possível REVERSÃO para alta (SMC)"
        
        return None, ""
    except Exception as e:
        logger.error(f"Erro BOS/CHoCH: {e}")
        return None, ""


def aplicar_smc_scoring(dados, pos_idx, tipo_sinal, tend):
    """
    Aplica TODOS os conceitos SMC ao scoring.
    Retorna: (score_extra, motivos, smc_data)
    """
    score_extra = 0
    motivos = []
    smc_data = {}
    
    # 1. FVG (Fair Value Gap)
    tem_fvg, tipo_fvg, fvg_detalhe = detectar_fvg(dados, pos_idx, tipo_sinal)
    if tem_fvg:
        smc_data["fvg"] = {"tipo": tipo_fvg, "detalhe": fvg_detalhe}
        if (tipo_fvg == "BULL_FVG" and tipo_sinal == "COMPRA") or \
           (tipo_fvg == "BEAR_FVG" and tipo_sinal == "VENDA"):
            score_extra += 1
            motivos.append(f"FVG {tipo_fvg} confirma {tipo_sinal} (SMC)")
        elif (tipo_fvg == "BULL_FVG" and tipo_sinal == "VENDA") or \
             (tipo_fvg == "BEAR_FVG" and tipo_sinal == "COMPRA"):
            score_extra -= 1
            motivos.append(f"FVG {tipo_fvg} CONTRA {tipo_sinal} - cuidado (SMC)")
    
    # 2. Liquidity Sweep
    tem_sweep, tipo_sweep, sweep_detalhe = detectar_liquidity_sweep(dados, pos_idx)
    if tem_sweep:
        smc_data["liquidity_sweep"] = {"tipo": tipo_sweep, "detalhe": sweep_detalhe}
        if (tipo_sweep == "SWEEP_LOW" and tipo_sinal == "COMPRA") or \
           (tipo_sweep == "SWEEP_HIGH" and tipo_sinal == "VENDA"):
            score_extra += 2  # Sweep é sinal FORTE
            motivos.append(f"Liquidity Sweep {tipo_sweep} - Smart Money ativo! (SMC +2)")
    
    # 3. Order Block
    tem_ob, tipo_ob, ob_zone, ob_detalhe = detectar_order_block(dados, pos_idx)
    if tem_ob:
        smc_data["order_block"] = {"tipo": tipo_ob, "zona": ob_zone, "detalhe": ob_detalhe}
        if (tipo_ob == "BULL_OB" and tipo_sinal == "COMPRA") or \
           (tipo_ob == "BEAR_OB" and tipo_sinal == "VENDA"):
            score_extra += 1
            motivos.append(f"Order Block {tipo_ob} confirma (SMC)")
    
    # 4. BOS / CHoCH
    bos_tipo, bos_detalhe = detectar_bos_choch(dados, pos_idx)
    if bos_tipo:
        smc_data["estrutura"] = {"tipo": bos_tipo, "detalhe": bos_detalhe}
        if bos_tipo == "BOS_ALTA" and tipo_sinal == "COMPRA":
            score_extra += 1
            motivos.append("Break of Structure ALTA - tendência confirmada (SMC)")
        elif bos_tipo == "BOS_BAIXA" and tipo_sinal == "VENDA":
            score_extra += 1
            motivos.append("Break of Structure BAIXA - tendência confirmada (SMC)")
        elif bos_tipo == "CHOCH_BAIXA" and tipo_sinal == "COMPRA":
            score_extra -= 1
            motivos.append("CHoCH BAIXA detectado - reversão possível, COMPRA arriscada (SMC)")
        elif bos_tipo == "CHOCH_ALTA" and tipo_sinal == "VENDA":
            score_extra -= 1
            motivos.append("CHoCH ALTA detectado - reversão possível, VENDA arriscada (SMC)")
    
    return score_extra, motivos, smc_data
