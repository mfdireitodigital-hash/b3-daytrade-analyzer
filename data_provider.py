"""
B3 Day Trade Analyzer - Provedor de Dados v3.0
Multi-source: yfinance (candles) + HG Brasil (preço tempo real) + fallback simulado.

Estratégia de dados:
1. yfinance: candles OHLCV intraday (5m, 15m, 1h) - delay ~15min
2. HG Brasil API: preço atual em tempo real (IBOVESPA e USD/BRL)
3. Fallback simulado: apenas quando todas as fontes falham

Rolagem automática de contratos B3 (WIN/WDO) por mês.
Filtro estrito de horário B3: 09:00-18:00 BRT (UTC-3).
"""

import pandas as pd
import numpy as np
import yfinance as yf
import httpx
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

BRT = timezone(timedelta(hours=-3))

# Mapeamento de meses B3
MESES_B3 = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
}

VENCIMENTOS_WIN = [2, 4, 6, 8, 10, 12]  # Meses pares


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
            "yfinance": "BRL=X",
            "nome": f"Mini-Dólar ({ticker_b3})",
            "tick": 0.5,
            "valor_tick": 10.00,
            "contrato": ticker_b3,
            "vencimento_mes": proximo_mes,
            "vencimento_ano": 2000 + proximo_ano,
        }
    return {"ticker_b3": ativo, "yfinance": ativo, "nome": ativo, "tick": 1, "valor_tick": 1.0, "contrato": ativo}


# Timeframes para yfinance
TIMEFRAME_MAP = {
    "5m": {"interval": "5m", "period": "5d"},
    "15m": {"interval": "15m", "period": "5d"},
    "1h": {"interval": "1h", "period": "1mo"},
    "4h": {"interval": "1h", "period": "3mo"},
    "1d": {"interval": "1d", "period": "6mo"},
}


class DataProvider:
    """Provedor de dados multi-source para B3"""

    def __init__(self, source: str = "yfinance"):
        self.source = source
        self.cache = {}
        self.cache_ttl = 60  # segundos
        self.bridge_url = os.getenv("BRIDGE_URL", "http://localhost:8081")
        self.hg_api_key = os.getenv("HG_API_KEY", "demo")
        self.contratos = {}
        self.preco_realtime = {}  # Cache de preço tempo real do HG Brasil
        for ativo in ["WIN", "WDO"]:
            self.contratos[ativo] = obter_contrato_vigente(ativo)
            logger.info(f"Contrato vigente {ativo}: {self.contratos[ativo]['ticker_b3']}")

    def get_contrato_info(self, ativo: str) -> dict:
        return self.contratos.get(ativo, obter_contrato_vigente(ativo))

    async def obter_preco_realtime(self) -> dict:
        """Obtém preço em tempo real via HG Brasil API (gratuita, sem delay)"""
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(
                    f"https://api.hgbrasil.com/finance",
                    params={"format": "json", "key": self.hg_api_key}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", {})
                    precos = {}
                    # IBOVESPA -> WIN
                    ibov = results.get("stocks", {}).get("IBOVESPA", {})
                    if ibov and ibov.get("points"):
                        precos["WIN"] = {
                            "preco": float(ibov["points"]),
                            "variacao": float(ibov.get("variation", 0)),
                            "fonte": "HG Brasil (tempo real)",
                        }
                    # USD/BRL -> WDO
                    usd = results.get("currencies", {}).get("USD", {})
                    if usd and usd.get("buy"):
                        precos["WDO"] = {
                            "preco": round(float(usd["buy"]) * 1000, 1),
                            "variacao": float(usd.get("variation", 0)),
                            "fonte": "HG Brasil (tempo real)",
                        }
                    self.preco_realtime = precos
                    logger.info(f"Preço realtime: WIN={precos.get('WIN',{}).get('preco','N/A')} WDO={precos.get('WDO',{}).get('preco','N/A')}")
                    return precos
        except Exception as e:
            logger.error(f"Erro HG Brasil API: {e}")
        return self.preco_realtime

    async def obter_candles_json(self, ativo: str, timeframe: str) -> list:
        """Retorna candles em formato JSON para gráficos frontend"""
        dados = await self.obter_dados(ativo, timeframe)
        if dados is None or dados.empty:
            return []

        candles = []
        for idx, row in dados.iterrows():
            # Converter timestamp para ISO com timezone
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
        """Obtém dados de mercado com fallback multi-source"""
        cache_key = f"{ativo}_{timeframe}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data

        dados = None
        try:
            if self.source == "bridge":
                dados = await self._obter_bridge(ativo, timeframe)
            else:
                dados = self._obter_yfinance(ativo, timeframe)
        except Exception as e:
            logger.error(f"Erro ao obter dados {ativo}/{timeframe}: {e}")

        if dados is not None and len(dados) > 0:
            # Filtrar horário B3 para intraday
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
        """Obtém dados via Yahoo Finance - yf.download()"""
        contrato = self.contratos.get(ativo, {})
        symbol = contrato.get("yfinance", "^BVSP" if ativo == "WIN" else "BRL=X")
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

            # Normalizar colunas (MultiIndex)
            if hasattr(dados.columns, 'nlevels') and dados.columns.nlevels > 1:
                dados.columns = dados.columns.get_level_values(0)
            dados.columns = [c.lower() for c in dados.columns]

            # Remover colunas extras
            for col in ['adj close', 'dividends', 'stock splits', 'capital gains']:
                if col in dados.columns:
                    dados = dados.drop(columns=[col])

            # WDO: USD/BRL * 1000
            if is_wdo:
                for col in ['open', 'high', 'low', 'close']:
                    if col in dados.columns:
                        dados[col] = dados[col] * 1000

            # Agregar para 4h se necessário
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
                # Converter para BRT
                idx_brt = idx.tz_convert(BRT)
            else:
                # Assumir UTC e converter
                idx_brt = idx.tz_localize('UTC').tz_convert(BRT)

            # Filtrar: 09:00-18:00 BRT, seg-sex
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

    async def _obter_bridge(self, ativo: str, timeframe: str) -> pd.DataFrame:
        """Obtém dados via bridge HTTP do Profit Pro / Tryd"""
        try:
            contrato = self.contratos.get(ativo, {})
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.bridge_url}/api/candles",
                    params={
                        "symbol": contrato.get("contrato", ativo),
                        "timeframe": timeframe,
                        "count": 300
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data["candles"])
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
                    return df
        except Exception as e:
            logger.error(f"Erro bridge: {e}")
        return None

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
        """
        Gera dados simulados realistas como ÚLTIMO recurso.
        Usa preços reais do HG Brasil quando disponível.
        """
        np.random.seed(42)

        # Usar preço real se disponível
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

        # Gerar timestamps estritamente dentro do pregão B3 (09:00-18:00 BRT)
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

        # Gerar preços com transição suave
        prices = [base_price]
        trend = np.random.choice([-1, 1]) * volatilidade * 0.05

        for i in range(1, len(timestamps)):
            change = np.random.normal(trend, volatilidade * 0.3)
            new_price = prices[-1] + change
            prices.append(new_price)

        data = []
        for i, (t, close_p) in enumerate(zip(timestamps, prices)):
            hl_range = abs(np.random.normal(0, volatilidade * 0.3))
            body = np.random.normal(0, volatilidade * 0.15)

            if i > 0:
                open_p = data[-1]['close']  # Continuidade: open = close anterior
            else:
                open_p = close_p - body

            high = max(open_p, close_p) + abs(np.random.normal(0, hl_range * 0.3))
            low = min(open_p, close_p) - abs(np.random.normal(0, hl_range * 0.3))

            base_vol = 5000 if ativo == "WIN" else 3000
            volume = max(100, int(np.random.lognormal(np.log(base_vol), 0.6)))

            data.append({
                'open': round(open_p, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_p, 2),
                'volume': volume
            })

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
        return df
