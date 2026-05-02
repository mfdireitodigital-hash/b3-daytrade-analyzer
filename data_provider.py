"""
B3 Day Trade Analyzer - Provedor de Dados v5.0 (Profit API + yfinance)
Sistema híbrido otimizado:
- Profit API Quote: preço real-time preciso (BVSP.INDX / USDBRL.FOREX)
- Profit API Economic Calendar: calendário econômico para Notícias de Impacto
- yfinance: candles OHLCV históricos 5min (^BVSP / BRL=X) - dados recentes grátis

Auth Profit: query param ?token=API_KEY
"""

import pandas as pd
import numpy as np
import httpx
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

BRT = timezone(timedelta(hours=-3))

# ========================
# Configuração Profit API
# ========================
PROFIT_BASE_URL = "https://api.profit.com"
PROFIT_API_TOKEN = os.getenv("PROFIT_API_TOKEN", "")

# Tickers na Profit API
PROFIT_QUOTE_TICKERS = {
    "WIN": "BVSP.INDX",
    "WDO": "USDBRL.FOREX",
}

# Mapeamento de meses B3
MESES_B3 = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
}
VENCIMENTOS_WIN = [2, 4, 6, 8, 10, 12]

# yfinance config
YFINANCE_TICKERS = {
    "WIN": "^BVSP",
    "WDO": "BRL=X",
}
TIMEFRAME_MAP = {
    "5m": {"interval": "5m", "period": "5d"},
    "15m": {"interval": "15m", "period": "5d"},
    "1h": {"interval": "1h", "period": "1mo"},
    "4h": {"interval": "1h", "period": "3mo"},
    "1d": {"interval": "1d", "period": "6mo"},
}


def obter_contrato_vigente(ativo: str) -> dict:
    """Detecta automaticamente o contrato vigente baseado na data atual e calendário B3"""
    hoje = datetime.now(BRT)
    mes = hoje.month
    ano = hoje.year % 100

    if ativo == "WIN":
        proximo_venc = None
        for m in VENCIMENTOS_WIN:
            if m >= mes:
                if m == mes and hoje.day > 18:
                    continue
                proximo_venc = m
                break
        if proximo_venc is None:
            proximo_venc = 2
            ano += 1
        letra = MESES_B3[proximo_venc]
        ticker_b3 = f"WIN{letra}{ano}"
        return {
            "ticker_b3": ticker_b3,
            "profit_quote": PROFIT_QUOTE_TICKERS["WIN"],
            "yfinance": "^BVSP",
            "nome": f"Mini-Índice ({ticker_b3})",
            "tick": 5,
            "valor_tick": 0.20,
            "contrato": ticker_b3,
            "vencimento_mes": proximo_venc,
            "vencimento_ano": 2000 + ano,
        }
    elif ativo == "WDO":
        proximo_mes = mes
        proximo_ano = ano
        if hoje.day > 3:
            proximo_mes = mes + 1
            if proximo_mes > 12:
                proximo_mes = 1
                proximo_ano += 1
        letra = MESES_B3[proximo_mes]
        ticker_b3 = f"WDO{letra}{proximo_ano}"
        return {
            "ticker_b3": ticker_b3,
            "profit_quote": PROFIT_QUOTE_TICKERS["WDO"],
            "yfinance": "BRL=X",
            "nome": f"Mini-Dólar ({ticker_b3})",
            "tick": 0.5,
            "valor_tick": 10.00,
            "contrato": ticker_b3,
            "vencimento_mes": proximo_mes,
            "vencimento_ano": 2000 + proximo_ano,
        }
    return {"ticker_b3": ativo, "profit_quote": ativo, "yfinance": ativo,
            "nome": ativo, "tick": 1, "valor_tick": 1.0, "contrato": ativo}


class DataProvider:
    """Provedor de dados — Profit API (quote/calendar) + yfinance (candles)"""

    def __init__(self, source: str = "yfinance"):
        self.source = source
        self.cache = {}
        self.cache_ttl = 60
        self.profit_token = PROFIT_API_TOKEN
        self.contratos = {}
        self.preco_realtime = {}
        for ativo in ["WIN", "WDO"]:
            self.contratos[ativo] = obter_contrato_vigente(ativo)
            logger.info(f"Contrato vigente {ativo}: {self.contratos[ativo]['ticker_b3']}")

        if self.profit_token:
            logger.info(f"Profit API configurada (quote + calendar). Token: {self.profit_token[:8]}...")
        else:
            logger.warning("PROFIT_API_TOKEN não configurado. Usando fallbacks para preço real-time.")

    def get_contrato_info(self, ativo: str) -> dict:
        return self.contratos.get(ativo, obter_contrato_vigente(ativo))

    # ========================================
    # PROFIT API — Quote (Preço Real-time)
    # ========================================
    async def obter_preco_realtime(self) -> dict:
        """
        Obtém preço em tempo real.
        Prioridade: Profit API Quote > TradingView Scanner > HG Brasil
        """
        precos = {}

        # === SOURCE 0: Profit API Quote (PRINCIPAL — mais preciso) ===
        if self.profit_token:
            for ativo in ["WIN", "WDO"]:
                ticker = PROFIT_QUOTE_TICKERS.get(ativo)
                if not ticker:
                    continue
                try:
                    url = f"{PROFIT_BASE_URL}/data-api/market-data/quote/{ticker}"
                    async with httpx.AsyncClient(timeout=10) as client:
                        resp = await client.get(url, params={"token": self.profit_token})
                        if resp.status_code == 200:
                            data = resp.json()
                            price = data.get("price")
                            if price:
                                # WDO: USDBRL.FOREX retorna ~5.65, precisamos * 1000 para pontos
                                preco_final = float(price) * 1000 if ativo == "WDO" else float(price)
                                
                                ohlc = data.get("ohlc_week", {})
                                precos[ativo] = {
                                    "preco": preco_final,
                                    "variacao": round(float(data.get("daily_percentage_change", 0)), 2),
                                    "variacao_pts": float(data.get("daily_price_change", 0)) * (1000 if ativo == "WDO" else 1),
                                    "high": float(ohlc.get("high", price)) * (1000 if ativo == "WDO" else 1),
                                    "low": float(ohlc.get("low", price)) * (1000 if ativo == "WDO" else 1),
                                    "open": float(ohlc.get("open", price)) * (1000 if ativo == "WDO" else 1),
                                    "volume": int(data.get("volume", 0)),
                                    "fonte": "Profit API (real-time)",
                                }
                                logger.info(f"Profit Quote {ativo} ({ticker}): {preco_final}")
                except Exception as e:
                    logger.warning(f"Erro Profit quote {ativo}: {e}")

        # === SOURCE 1: TradingView Scanner API (fallback) ===
        ativos_faltando = [a for a in ["WIN", "WDO"] if a not in precos]
        if ativos_faltando:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        "https://scanner.tradingview.com/futures/scan",
                        headers={"Content-Type": "application/json"},
                        json={
                            "symbols": {"tickers": ["BMFBOVESPA:WIN1!", "BMFBOVESPA:WDO1!"]},
                            "columns": ["close", "change", "change_abs", "high", "low", "open", "volume"]
                        }
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("data", []):
                            ticker = item["s"]
                            cols = item["d"]
                            if "WIN" in ticker and "WIN" in ativos_faltando:
                                precos["WIN"] = {
                                    "preco": float(cols[0]),
                                    "variacao": round(float(cols[1]), 2),
                                    "variacao_pts": float(cols[2]),
                                    "high": float(cols[3]),
                                    "low": float(cols[4]),
                                    "open": float(cols[5]),
                                    "volume": int(cols[6]) if cols[6] else 0,
                                    "fonte": "TradingView (futuro real)",
                                }
                            elif "WDO" in ticker and "WDO" in ativos_faltando:
                                precos["WDO"] = {
                                    "preco": float(cols[0]),
                                    "variacao": round(float(cols[1]), 2),
                                    "variacao_pts": round(float(cols[2]), 1),
                                    "high": float(cols[3]),
                                    "low": float(cols[4]),
                                    "open": float(cols[5]),
                                    "volume": int(cols[6]) if cols[6] else 0,
                                    "fonte": "TradingView (futuro real)",
                                }
            except Exception as e:
                logger.warning(f"Erro TradingView Scanner: {e}")

        # === SOURCE 2: HG Brasil (último fallback) ===
        ativos_faltando = [a for a in ["WIN", "WDO"] if a not in precos]
        if ativos_faltando:
            try:
                hg_key = os.getenv("HG_API_KEY", "demo")
                async with httpx.AsyncClient(timeout=8) as client:
                    resp = await client.get(
                        "https://api.hgbrasil.com/finance",
                        params={"format": "json", "key": hg_key}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", {})
                        if "WIN" in ativos_faltando:
                            ibov = results.get("stocks", {}).get("IBOVESPA", {})
                            if ibov and ibov.get("points"):
                                precos["WIN"] = {
                                    "preco": float(ibov["points"]),
                                    "variacao": float(ibov.get("variation", 0)),
                                    "fonte": "HG Brasil (IBOVESPA spot - fallback)",
                                }
                        if "WDO" in ativos_faltando:
                            usd = results.get("currencies", {}).get("USD", {})
                            if usd and usd.get("buy"):
                                precos["WDO"] = {
                                    "preco": round(float(usd["buy"]) * 1000, 1),
                                    "variacao": float(usd.get("variation", 0)),
                                    "fonte": "HG Brasil (USD spot - fallback)",
                                }
            except Exception as e:
                logger.error(f"Erro HG Brasil fallback: {e}")

        if precos:
            self.preco_realtime = precos
            for ativo, info in precos.items():
                logger.info(f"Preço {ativo}: {info['preco']} via {info['fonte']}")

        return self.preco_realtime

    # ========================================
    # PROFIT API — Economic Calendar
    # ========================================
    async def obter_calendario_economico(self, dias: int = 7) -> list:
        """Obtém calendário econômico da Profit API para Notícias de Impacto"""
        if not self.profit_token:
            return []

        eventos = []
        try:
            # Buscar eventos forex (inclui todos os macro: payroll, FOMC, CPI, etc)
            end_ts = int(datetime.now().timestamp())
            start_ts = end_ts - (dias * 86400)

            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{PROFIT_BASE_URL}/data-api/economic_calendar/forex",
                    params={
                        "token": self.profit_token,
                        "start_date": start_ts,
                        "end_date": end_ts + 86400,  # +1 dia para pegar eventos futuros
                        "limit": 100,
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        for ev in data:
                            eventos.append({
                                "nome": ev.get("name", ""),
                                "pais": ev.get("country_iso", ""),
                                "moeda": ev.get("currency", ""),
                                "hora": ev.get("time", 0),
                                "impacto": ev.get("impact", "low"),
                                "atual": ev.get("actual"),
                                "estimativa": ev.get("estimate"),
                                "anterior": ev.get("previous"),
                            })
                        logger.info(f"Profit Calendar: {len(eventos)} eventos")
        except Exception as e:
            logger.error(f"Erro Profit Calendar: {e}")

        return eventos

    # ========================================
    # Interface Principal — obter_dados (yfinance)
    # ========================================
    async def obter_candles_json(self, ativo: str, timeframe: str) -> list:
        """Retorna candles em formato JSON para gráficos frontend"""
        dados = await self.obter_dados(ativo, timeframe)
        if dados is None or dados.empty:
            return []

        candles = []
        for idx, row in dados.iterrows():
            if hasattr(idx, 'isoformat'):
                ts = idx.isoformat()
            else:
                ts = str(idx)
            candles.append({
                "time": ts,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row.get("volume", 0)),
            })
        return candles

    async def obter_dados(self, ativo: str, timeframe: str) -> pd.DataFrame:
        """Obtém dados OHLCV via yfinance (candles históricos)"""
        cache_key = f"{ativo}_{timeframe}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data

        dados = None
        try:
            dados = self._obter_yfinance(ativo, timeframe)
        except Exception as e:
            logger.error(f"Erro ao obter dados {ativo}/{timeframe}: {e}")

        if dados is not None and len(dados) > 0:
            if timeframe in ["5m", "15m", "1h"]:
                dados = self._filtrar_horario_b3(dados)

            if len(dados) > 0:
                self.cache[cache_key] = (datetime.now(), dados)
                return dados

        # Fallback: cache existente
        if cache_key in self.cache:
            _, cached_data = self.cache[cache_key]
            logger.info(f"Usando cache para {ativo}/{timeframe}")
            return cached_data

        # Último recurso: dados simulados
        logger.warning(f"Sem dados para {ativo}/{timeframe}, usando simulados")
        return self._gerar_dados_simulados(ativo, timeframe)

    def _obter_yfinance(self, ativo: str, timeframe: str) -> pd.DataFrame:
        """Obtém dados via Yahoo Finance"""
        import yfinance as yf

        symbol = YFINANCE_TICKERS.get(ativo, "^BVSP" if ativo == "WIN" else "BRL=X")
        tf_config = TIMEFRAME_MAP.get(timeframe, TIMEFRAME_MAP["5m"])
        is_wdo = (ativo == "WDO")

        try:
            dados = yf.download(
                symbol,
                period=tf_config["period"],
                interval=tf_config["interval"],
                progress=False,
                timeout=20
            )

            if dados is None or dados.empty:
                logger.warning(f"yfinance retornou vazio para {symbol}")
                return None

            if hasattr(dados.columns, 'nlevels') and dados.columns.nlevels > 1:
                dados.columns = dados.columns.get_level_values(0)
            dados.columns = [c.lower() for c in dados.columns]

            for col in ['adj close', 'dividends', 'stock splits', 'capital gains']:
                if col in dados.columns:
                    dados = dados.drop(columns=[col])

            # WDO: USD/BRL * 1000
            if is_wdo:
                for col in ['open', 'high', 'low', 'close']:
                    if col in dados.columns:
                        dados[col] = dados[col] * 1000

            if timeframe == "4h":
                dados = self._agregar_timeframe(dados, "4h")

            logger.info(f"yfinance {symbol}: {len(dados)} candles, last close={dados['close'].iloc[-1]:.2f}")
            return dados

        except Exception as e:
            logger.error(f"Erro yfinance {symbol}: {e}")
            return None

    def _filtrar_horario_b3(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Filtra candles para horário B3: 09:00-18:00 BRT (UTC-3)"""
        if dados.empty:
            return dados

        try:
            idx = dados.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx_brt = idx.tz_convert(BRT)
            else:
                idx_brt = idx.tz_localize('UTC').tz_convert(BRT)

            mask = (idx_brt.hour >= 9) & (idx_brt.hour < 18) & (idx_brt.weekday < 5)
            dados_filtrados = dados[mask]

            if len(dados_filtrados) > 0:
                logger.info(f"Filtro B3: {len(dados)} -> {len(dados_filtrados)} candles")
                return dados_filtrados
            else:
                logger.warning("Filtro B3 removeu todos os candles, retornando originais")
                return dados
        except Exception as e:
            logger.error(f"Erro filtro B3: {e}")
            return dados

    def _agregar_timeframe(self, dados: pd.DataFrame, target: str) -> pd.DataFrame:
        """Agrega dados de timeframe menor para maior"""
        if target == "4h":
            return dados.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        return dados

    def _gerar_dados_simulados(self, ativo: str, timeframe: str) -> pd.DataFrame:
        """Gera dados simulados realistas como ÚLTIMO recurso."""
        np.random.seed(42)

        rt = self.preco_realtime.get(ativo, {})
        if ativo == "WIN":
            base_price = rt.get("preco", 187000)
            volatilidade = 120
        else:
            base_price = rt.get("preco", 4955)
            volatilidade = 8

        n_candles = 200
        tf_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = tf_minutes.get(timeframe, 5)

        agora = datetime.now(BRT)
        timestamps = []
        t = agora
        count = 0
        while count < n_candles:
            t = t - timedelta(minutes=minutes)
            if t.weekday() >= 5:
                continue
            if minutes < 1440:
                if t.hour < 9 or t.hour >= 18:
                    continue
            timestamps.insert(0, t)
            count += 1

        prices = [base_price]
        trend = np.random.choice([-1, 1]) * volatilidade * 0.05
        for i in range(1, len(timestamps)):
            change = np.random.normal(trend, volatilidade * 0.3)
            prices.append(prices[-1] + change)

        data = []
        for i, (t, close_p) in enumerate(zip(timestamps, prices)):
            hl_range = abs(np.random.normal(0, volatilidade * 0.3))
            body = np.random.normal(0, volatilidade * 0.15)
            open_p = data[-1]['close'] if i > 0 else close_p - body
            high = max(open_p, close_p) + abs(np.random.normal(0, hl_range * 0.3))
            low = min(open_p, close_p) - abs(np.random.normal(0, hl_range * 0.3))
            base_vol = 5000 if ativo == "WIN" else 3000
            volume = max(100, int(np.random.lognormal(np.log(base_vol), 0.6)))
            data.append({
                'open': round(open_p, 2), 'high': round(high, 2),
                'low': round(low, 2), 'close': round(close_p, 2), 'volume': volume
            })

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
        return df
