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

# Mapeamento de meses B3 (código de vencimento)
MESES_B3 = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
}

# WIN vence na quarta-feira mais próxima do dia 15 dos meses pares (G, J, M, Q, V, Z)
# WDO vence no 1o dia útil de cada mês
VENCIMENTOS_WIN = ["G", "J", "M", "Q", "V", "Z"]  # Fev, Abr, Jun, Ago, Out, Dez


def obter_contrato_vigente(ativo: str) -> dict:
    """Detecta automaticamente o contrato vigente baseado na data atual e calendário B3"""
    from datetime import datetime, timedelta
    hoje = datetime.now()
    mes = hoje.month
    ano = hoje.year % 100  # 26 para 2026

    if ativo == "WIN":
        # WIN vence nos meses pares (G=Fev, J=Abr, M=Jun, Q=Ago, V=Out, Z=Dez)
        # Após o vencimento do mês par atual, rola para o próximo
        meses_venc = [2, 4, 6, 8, 10, 12]
        proximo_venc = None
        for m in meses_venc:
            if m >= mes:
                # Se estamos no mês de vencimento, verificar se já passou dia 15
                if m == mes and hoje.day > 18:  # margem de segurança após vencimento
                    continue
                proximo_venc = m
                break
        if proximo_venc is None:
            # Passou dezembro, vai para fevereiro do próximo ano
            proximo_venc = 2
            ano += 1

        letra = MESES_B3[proximo_venc]
        ticker_b3 = f"WIN{letra}{ano}"
        ticker_yf = "^BVSP"  # Ibovespa como proxy
        return {
            "ticker_b3": ticker_b3,
            "yfinance": ticker_yf,
            "nome": f"Mini-Índice ({ticker_b3})",
            "tick": 5,
            "valor_tick": 0.20,
            "contrato": ticker_b3,
            "vencimento_mes": proximo_venc,
            "vencimento_ano": 2000 + ano,
        }

    elif ativo == "WDO":
        # WDO vence no 1o dia útil de cada mês
        # Se já passou o 1o dia útil, rola para o próximo mês
        proximo_mes = mes
        proximo_ano = ano
        if hoje.day > 3:  # margem - após dia 3 já rolou
            proximo_mes = mes + 1
            if proximo_mes > 12:
                proximo_mes = 1
                proximo_ano += 1

        letra = MESES_B3[proximo_mes]
        ticker_b3 = f"WDO{letra}{proximo_ano}"
        ticker_yf = "BRL=X"  # USD/BRL como proxy
        return {
            "ticker_b3": ticker_b3,
            "yfinance": ticker_yf,
            "nome": f"Mini-Dólar ({ticker_b3})",
            "tick": 0.5,
            "valor_tick": 10.00,
            "contrato": ticker_b3,
            "vencimento_mes": proximo_mes,
            "vencimento_ano": 2000 + proximo_ano,
        }

    return {"ticker_b3": ativo, "yfinance": ativo, "nome": ativo, "tick": 1, "valor_tick": 1.0, "contrato": ativo}


# Mapeamento de ativos B3 (agora dinâmico via obter_contrato_vigente)
ATIVOS_B3 = {
    "WIN": {
        "yfinance": "^BVSP",
        "nome": "Mini-Índice (WIN)",
        "tick": 5,
        "valor_tick": 0.20,
        "contrato": "WINFUT",
    },
    "WDO": {
        "yfinance": "BRL=X",
        "nome": "Mini-Dólar (WDO)",
        "tick": 0.5,
        "valor_tick": 10.00,
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
        # Detectar contratos vigentes automaticamente
        self.contratos = {}
        for ativo in ["WIN", "WDO"]:
            self.contratos[ativo] = obter_contrato_vigente(ativo)
            logger.info(f"Contrato vigente {ativo}: {self.contratos[ativo]['ticker_b3']}")

    def get_contrato_info(self, ativo: str) -> dict:
        """Retorna informações do contrato vigente"""
        return self.contratos.get(ativo, obter_contrato_vigente(ativo))

    async def obter_candles_json(self, ativo: str, timeframe: str) -> list:
        """Retorna candles em formato JSON para gráficos frontend"""
        dados = await self.obter_dados(ativo, timeframe)
        if dados is None or dados.empty:
            return []
        candles = []
        for idx, row in dados.iterrows():
            ts = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
            candles.append({
                "time": ts,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row.get("volume", 0)),
            })
        # Filtrar candles fora do horário B3 (09:00-18:00) para intraday
        if candles:
            filtered = []
            for c in candles:
                try:
                    t = c["time"]
                    if "T" in t:
                        hour = int(t.split("T")[1].split(":")[0])
                        if 9 <= hour < 18:
                            filtered.append(c)
                    else:
                        filtered.append(c)
                except:
                    filtered.append(c)
            candles = filtered if filtered else candles
        return candles

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
                # Filtrar dados fora do horário B3 (09:00-18:00 BRT)
                if timeframe not in ["1d", "4h"]:
                    try:
                        from datetime import timezone
                        BRT = timezone(timedelta(hours=-3))
                        # Converter para BRT e filtrar
                        if dados.index.tz is not None:
                            dados_brt = dados.index.tz_convert(BRT)
                        else:
                            dados_brt = dados.index.tz_localize('UTC').tz_convert(BRT)
                        mask = (dados_brt.hour >= 9) & (dados_brt.hour < 18)
                        dados = dados[mask]
                    except Exception as e:
                        logger.warning(f"Erro ao filtrar horario B3: {e}")
                
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
        RESPEITA RIGOROSAMENTE horario B3: 09:00-18:00 BRT (UTC-3).
        NÃO usar para operações reais!
        """
        from datetime import timezone
        BRT = timezone(timedelta(hours=-3))
        
        # Seed baseado no dia para dados consistentes no mesmo dia
        agora_brt = datetime.now(BRT)
        np.random.seed(int(agora_brt.strftime('%Y%m%d')))

        # Parâmetros por ativo
        if ativo == "WIN":
            base_price = 128500
            volatilidade = 80  # Volatilidade realista para 5min
        else:  # WDO
            base_price = 5650
            volatilidade = 4

        # Parâmetros por timeframe
        tf_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = tf_minutes.get(timeframe, 5)

        # Gerar timestamps RIGOROSAMENTE dentro do horário B3 (09:00-18:00 BRT)
        timestamps = []
        
        if minutes >= 1440:  # Diário
            for i in range(120, 0, -1):
                t = agora_brt - timedelta(days=i)
                if t.weekday() < 5:  # Só dias úteis
                    # Usar 17:00 BRT como timestamp do candle diário
                    t = t.replace(hour=17, minute=0, second=0, microsecond=0)
                    timestamps.append(t)
        else:
            # Intraday: gerar candles de 09:00 até 17:55 BRT (último candle antes de 18:00)
            # Gerar últimos 5 dias úteis
            dias = []
            d = agora_brt.replace(hour=0, minute=0, second=0, microsecond=0)
            while len(dias) < 5:
                if d.weekday() < 5:  # Só dias úteis
                    dias.append(d)
                d = d - timedelta(days=1)
            dias.reverse()
            
            for dia in dias:
                hora_inicio = 9  # 09:00 BRT
                hora_fim = 18    # 18:00 BRT
                minuto = 0
                h = hora_inicio
                m = 0
                while True:
                    t = dia.replace(hour=h, minute=m, second=0, microsecond=0)
                    total_min = h * 60 + m
                    if total_min >= hora_fim * 60:
                        break
                    timestamps.append(t)
                    m += minutes
                    if m >= 60:
                        h += m // 60
                        m = m % 60
                    if h >= hora_fim:
                        break
            
            # Se mercado está aberto agora, cortar no minuto atual
            if agora_brt.weekday() < 5 and 9 <= agora_brt.hour < 18:
                timestamps = [t for t in timestamps if t <= agora_brt]

        if not timestamps:
            # Fallback: gerar pelo menos algo
            timestamps = [agora_brt - timedelta(minutes=minutes * i) for i in range(100, 0, -1)]

        # Gerar preços com random walk SUAVE (sem gaps entre candles)
        prices = [base_price]
        trend = np.random.choice([-0.5, 0.5]) * volatilidade * 0.05

        for i in range(1, len(timestamps)):
            # Verificar se é um novo dia (gap overnight permitido)
            if timestamps[i].date() != timestamps[i-1].date():
                # Gap de abertura: pequeno, realista
                gap = np.random.normal(0, volatilidade * 0.3)
                new_price = prices[-1] + gap
            else:
                # Dentro do mesmo dia: movimentos suaves
                change = np.random.normal(trend, volatilidade * 0.3)
                cycle = np.sin(i / 30) * volatilidade * 0.1
                new_price = prices[-1] + change + cycle
            prices.append(new_price)

        # Gerar OHLCV com candles COERENTES (open = close anterior dentro do dia)
        data = []
        prev_close = None
        prev_date = None
        
        for i, (t, close) in enumerate(zip(timestamps, prices)):
            # Open = close do candle anterior (dentro do mesmo dia)
            if prev_close is not None and prev_date == t.date():
                open_p = prev_close
            else:
                # Primeiro candle do dia ou gap overnight
                open_p = close + np.random.normal(0, volatilidade * 0.1)
            
            # High e low realistas
            body_size = abs(close - open_p)
            wick_up = abs(np.random.normal(0, volatilidade * 0.15))
            wick_down = abs(np.random.normal(0, volatilidade * 0.15))
            
            high = max(open_p, close) + wick_up
            low = min(open_p, close) - wick_down

            # Volume com variação realista
            base_vol = 5000 if ativo == "WIN" else 3000
            # Mais volume na abertura e fechamento
            hour = t.hour
            vol_mult = 1.0
            if hour == 9 or hour == 17:
                vol_mult = 2.0
            elif hour == 10 or hour == 16:
                vol_mult = 1.5
            volume = max(100, int(np.random.lognormal(np.log(base_vol * vol_mult), 0.5)))

            data.append({
                'open': round(open_p, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
            
            prev_close = close
            prev_date = t.date()

        # Remover timezone info para o pandas index (armazenar como naive mas representando BRT)
        naive_timestamps = [t.replace(tzinfo=None) for t in timestamps]
        df = pd.DataFrame(data, index=pd.DatetimeIndex(naive_timestamps))
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
