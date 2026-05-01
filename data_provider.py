"""
B3 Day Trade Analyzer - Provedor de Dados
Camada modular para obter dados de mercado da B3.

Fontes suportadas:
1. yfinance (padrão - gratuito, delay ~15min)
2. Brapi (API brasileira - gratuito com limites)
3. Profit Pro / Tryd (via bridge HTTP - requer setup local)
4. CSV/JSON manual upload

Para day trade real, recomenda-se configurar o bridge do Profit Pro.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import httpx
import os
import json
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Mapeamento de ativos B3
ATIVOS_B3 = {
    "WIN": {
        "yfinance": "^BVSP",  # Usa Ibovespa como proxy
        "nome": "Mini-Índice (WIN)",
        "tick": 5,  # Tamanho do tick em pontos
        "valor_tick": 0.20,  # Valor financeiro por tick
        "contrato": "WINFUT",
    },
    "WDO": {
        "yfinance": "BRL=X",  # Usa USD/BRL como proxy
        "nome": "Mini-Dólar (WDO)",
        "tick": 0.5,
        "valor_tick": 5.00,
        "contrato": "WDOFUT",
    }
}

# Mapeamento de timeframes para yfinance
TIMEFRAME_MAP = {
    "5m": {"interval": "5m", "period": "5d"},
    "15m": {"interval": "15m", "period": "5d"},
    "1h": {"interval": "1h", "period": "1mo"},
    "4h": {"interval": "1h", "period": "3mo"},  # Agregar para 4h
    "1d": {"interval": "1d", "period": "6mo"},
}


class DataProvider:
    """Provedor de dados abstrato com múltiplas fontes"""

    def __init__(self, source: str = "yfinance"):
        self.source = source
        self.cache = {}
        self.cache_ttl = 60  # segundos
        self.bridge_url = os.getenv("BRIDGE_URL", "http://localhost:8081")
        self.brapi_token = os.getenv("BRAPI_TOKEN", "")

    async def obter_dados(self, ativo: str, timeframe: str) -> pd.DataFrame:
        """Obtém dados de mercado para o ativo e timeframe especificados"""

        cache_key = f"{ativo}_{timeframe}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data

        try:
            if self.source == "bridge":
                dados = await self._obter_bridge(ativo, timeframe)
            elif self.source == "brapi":
                dados = await self._obter_brapi(ativo, timeframe)
            else:
                dados = self._obter_yfinance(ativo, timeframe)

            if dados is not None and len(dados) > 0:
                self.cache[cache_key] = (datetime.now(), dados)
                return dados

        except Exception as e:
            logger.error(f"Erro ao obter dados {ativo}/{timeframe}: {e}")

            # Fallback para cache em memoria se disponivel
            if cache_key in self.cache:
                _, cached_data = self.cache[cache_key]
                logger.info(f"Usando cache em memoria para {ativo}/{timeframe}")
                return cached_data

        # Sem dados disponiveis - usar dados simulados como fallback
        logger.warning(f"Sem dados disponiveis para {ativo}/{timeframe}, usando simulados")
        return self._gerar_dados_simulados(ativo, timeframe)

    def _obter_yfinance(self, ativo: str, timeframe: str) -> pd.DataFrame:
        """Obtém dados via yfinance"""
        config = ATIVOS_B3.get(ativo, {})
        symbol = config.get("yfinance", ativo)
        tf_config = TIMEFRAME_MAP.get(timeframe, TIMEFRAME_MAP["5m"])

        try:
            ticker = yf.Ticker(symbol)
            dados = ticker.history(
                interval=tf_config["interval"],
                period=tf_config["period"]
            )

            if dados.empty:
                return None

            # Normalizar colunas
            dados.columns = [c.lower() for c in dados.columns]
            if 'adj close' in dados.columns:
                dados = dados.drop(columns=['adj close'])
            if 'dividends' in dados.columns:
                dados = dados.drop(columns=['dividends'])
            if 'stock splits' in dados.columns:
                dados = dados.drop(columns=['stock splits'])
            if 'capital gains' in dados.columns:
                dados = dados.drop(columns=['capital gains'])

            # Agregar para 4h se necessário
            if timeframe == "4h":
                dados = self._agregar_timeframe(dados, "4h")

            # Ajustar valores para parecer com mini-índice/mini-dólar
            if ativo == "WIN":
                # Ibovespa já tem valores similares ao WIN
                pass
            elif ativo == "WDO":
                # USD/BRL - multiplicar por 1000 para simular pontos WDO
                dados['open'] = dados['open'] * 1000
                dados['high'] = dados['high'] * 1000
                dados['low'] = dados['low'] * 1000
                dados['close'] = dados['close'] * 1000

            return dados

        except Exception as e:
            logger.error(f"Erro yfinance: {e}")
            return None

    async def _obter_bridge(self, ativo: str, timeframe: str) -> pd.DataFrame:
        """
        Obtém dados via bridge HTTP do Profit Pro / Tryd.
        O bridge é um serviço local que expõe dados DDE via HTTP.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.bridge_url}/api/candles",
                    params={
                        "symbol": ATIVOS_B3[ativo]["contrato"],
                        "timeframe": timeframe,
                        "count": 200
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

    async def _obter_brapi(self, ativo: str, timeframe: str) -> pd.DataFrame:
        """Obtém dados via Brapi API"""
        try:
            symbol_map = {"WIN": "IBOV", "WDO": "USDBRL"}
            symbol = symbol_map.get(ativo, ativo)

            range_map = {
                "5m": "5d", "15m": "5d", "1h": "1mo",
                "4h": "3mo", "1d": "6mo"
            }

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"https://brapi.dev/api/v2/crypto",
                    params={
                        "coin": symbol,
                        "range": range_map.get(timeframe, "5d"),
                        "interval": timeframe,
                        "token": self.brapi_token
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    # Parse response into DataFrame
                    # (format depends on actual Brapi response)
                    return self._parse_brapi_response(data)
        except Exception as e:
            logger.error(f"Erro brapi: {e}")
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
        Gera dados simulados realistas para desenvolvimento.
        NÃO usar para operações reais!
        """
        np.random.seed(42)

        # Parâmetros por ativo
        if ativo == "WIN":
            base_price = 128500
            volatilidade = 150
            n_candles = 200
        else:  # WDO
            base_price = 5650
            volatilidade = 15
            n_candles = 200

        # Parâmetros por timeframe
        tf_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = tf_minutes.get(timeframe, 5)

        # Gerar timestamps
        now = datetime.now()
        timestamps = []
        for i in range(n_candles, 0, -1):
            t = now - timedelta(minutes=minutes * i)
            # Filtrar horário de pregão (9h-18h) para intraday
            if minutes < 1440:
                if t.weekday() >= 5:  # Pular fim de semana
                    continue
                if t.hour < 9 or t.hour >= 18:
                    continue
            timestamps.append(t)

        if not timestamps:
            timestamps = [now - timedelta(minutes=minutes * i) for i in range(n_candles, 0, -1)]

        # Gerar preços com random walk + tendência
        prices = [base_price]
        trend = np.random.choice([-1, 1]) * volatilidade * 0.1

        for i in range(1, len(timestamps)):
            change = np.random.normal(trend, volatilidade)
            # Adicionar ciclos para simular movimentos de mercado
            cycle = np.sin(i / 20) * volatilidade * 2
            new_price = prices[-1] + change + cycle * 0.1
            prices.append(new_price)

        # Gerar OHLCV
        data = []
        for i, (t, close) in enumerate(zip(timestamps, prices)):
            hl_range = abs(np.random.normal(0, volatilidade * 0.5))
            body = np.random.normal(0, volatilidade * 0.3)

            open_p = close - body
            high = max(open_p, close) + abs(np.random.normal(0, hl_range * 0.5))
            low = min(open_p, close) - abs(np.random.normal(0, hl_range * 0.5))

            # Volume com variação realista
            base_vol = 5000 if ativo == "WIN" else 3000
            volume = max(100, int(np.random.lognormal(np.log(base_vol), 0.8)))

            data.append({
                'open': round(open_p, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
        return df

    def _parse_brapi_response(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse resposta da Brapi API"""
        try:
            if "results" in data and len(data["results"]) > 0:
                hist = data["results"][0].get("historicalDataPrice", [])
                if hist:
                    df = pd.DataFrame(hist)
                    df['date'] = pd.to_datetime(df['date'], unit='s')
                    df = df.set_index('date')
                    df = df.rename(columns={'adjustedClose': 'close'})
                    return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.error(f"Erro parsing brapi: {e}")
        return None
