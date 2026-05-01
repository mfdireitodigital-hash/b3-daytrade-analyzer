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
import re

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

# URLs Investing.com para preço real dos futuros B3
INVESTING_URLS = {
    "WIN": "https://br.investing.com/indices/bovespa-win-futures",
    "WDO": "https://br.investing.com/currencies/usd-brl-mini-futures",
}
INVESTING_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
}


# Cálculo do basis (prêmio do futuro) para quando scraping falha
def calcular_preco_futuro(preco_spot: float, ativo: str) -> float:
    """
    Estima o preço do contrato futuro a partir do spot.
    WIN: Spot * (1 + (SELIC - DivYield) * dias_uteis/252)
    WDO: Spot * 1000 * (1 + cupom_cambial)
    """
    from datetime import date
    hoje = date.today()
    
    if ativo == "WIN":
        # Vencimento WINM26 = ~3a quarta de junho = 17/jun/2026
        # Pegar o mês de vencimento do contrato vigente
        vencimentos_win = {2: 18, 4: 15, 6: 17, 8: 19, 10: 14, 12: 16}  # datas aprox 2026
        mes_atual = hoje.month
        ano = hoje.year
        
        # Achar próximo vencimento
        for m in [2, 4, 6, 8, 10, 12]:
            if m >= mes_atual:
                venc_mes = m
                break
        else:
            venc_mes = 2
            ano += 1
        
        dia_venc = vencimentos_win.get(venc_mes, 17)
        try:
            vencimento = date(ano, venc_mes, dia_venc)
        except:
            vencimento = date(ano, venc_mes, 15)
        
        dias_corridos = max(1, (vencimento - hoje).days)
        dias_uteis = max(1, int(dias_corridos * 5 / 7))
        
        # SELIC 14.25% - Dividend yield ~3.5% = ~10.75% net
        taxa_net = 0.1075
        taxa_periodo = taxa_net * dias_uteis / 252
        return round(preco_spot * (1 + taxa_periodo), 2)
    
    elif ativo == "WDO":
        # Mini-dólar: USD/BRL spot * 1000 com cupom cambial (~1% ao ano)
        # O prêmio é pequeno, tipicamente 0.3-1.0% acima do spot
        return round(preco_spot * 1000 * 1.005, 1)
    
    return preco_spot


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
        """
        Obtém preço em tempo real dos FUTUROS B3 (WIN/WDO).
        Estratégia multi-source com fallback:
        1. Investing.com (scraping HTML) - preço real do contrato futuro
        2. Google Finance (JSON embedded) - preço real do futuro
        3. HG Brasil API (fallback) - IBOVESPA à vista / USD spot
        """
        precos = {}

        # === SOURCE 1: Investing.com ===
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.google.com/",
                "DNT": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "cross-site",
            }
            async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
                for ativo, url in INVESTING_URLS.items():
                    try:
                        resp = await client.get(url, headers=headers)
                        if resp.status_code == 200:
                            html = resp.text
                            # Method 1: data-test attribute
                            price_match = re.search(
                                r'data-test="instrument-price-last">([\d.,]+)',
                                html
                            )
                            # Method 2: JSON embedded current_price
                            if not price_match:
                                if ativo == "WIN":
                                    price_match = re.search(r'"pair_id":941613[^}]*?"current_price":(\d+)', html)
                                else:
                                    price_match = re.search(r'"pair_id":996708[^}]*?"current_price":(\d+)', html)

                            if price_match:
                                price_str = price_match.group(1)
                                # Parse price based on format
                                if ',' in price_str and '.' in price_str:
                                    price_str = price_str.replace('.', '').replace(',', '.')
                                elif '.' in price_str:
                                    parts = price_str.split('.')
                                    if len(parts) == 2 and len(parts[1]) == 3:
                                        price_str = price_str.replace('.', '')
                                preco = float(price_str)

                                var_match = re.search(
                                    r'data-test="instrument-price-change-percent">\(?([+-]?[\d.,]+)%?\)?',
                                    html
                                )
                                variacao = 0.0
                                if var_match:
                                    var_str = var_match.group(1).replace(',', '.')
                                    try:
                                        variacao = float(var_str)
                                    except:
                                        pass

                                precos[ativo] = {
                                    "preco": preco,
                                    "variacao": variacao,
                                    "fonte": "Investing.com (futuro real)",
                                }
                                logger.info(f"Investing.com {ativo}: {preco} ({variacao:+.2f}%)")
                    except Exception as e:
                        logger.warning(f"Erro Investing.com {ativo}: {e}")
        except Exception as e:
            logger.warning(f"Erro geral Investing.com: {e}")

        # === SOURCE 2: Google Finance (alternative) ===
        ativos_faltando = [a for a in ["WIN", "WDO"] if a not in precos]
        if ativos_faltando:
            try:
                gf_urls = {
                    "WIN": "https://www.google.com/finance/quote/WINM26:BVMF",
                    "WDO": "https://www.google.com/finance/quote/WDOK26:BVMF",
                }
                headers_gf = {
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                    "Accept": "text/html",
                    "Accept-Language": "pt-BR,pt;q=0.9",
                }
                async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                    for ativo in ativos_faltando:
                        if ativo in gf_urls:
                            try:
                                resp = await client.get(gf_urls[ativo], headers=headers_gf)
                                if resp.status_code == 200:
                                    # Google Finance stores price in data-last-price
                                    m = re.search(r'data-last-price="([\d.]+)"', resp.text)
                                    if m:
                                        preco_gf = float(m.group(1))
                                        precos[ativo] = {
                                            "preco": preco_gf,
                                            "variacao": 0,
                                            "fonte": "Google Finance (futuro)",
                                        }
                                        logger.info(f"Google Finance {ativo}: {preco_gf}")
                            except Exception as e:
                                logger.warning(f"Erro Google Finance {ativo}: {e}")
            except Exception as e:
                logger.warning(f"Erro geral Google Finance: {e}")

        # === SOURCE 2.5: Cálculo do Basis (se scraping falhou, estimar futuro a partir do spot) ===
        ativos_faltando = [a for a in ["WIN", "WDO"] if a not in precos]
        if ativos_faltando:
            # Tentar obter spot do HG Brasil e calcular o futuro
            try:
                async with httpx.AsyncClient(timeout=8) as client:
                    resp = await client.get(
                        "https://api.hgbrasil.com/finance",
                        params={"format": "json", "key": self.hg_api_key}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", {})
                        if "WIN" in ativos_faltando:
                            ibov = results.get("stocks", {}).get("IBOVESPA", {})
                            if ibov and ibov.get("points"):
                                spot = float(ibov["points"])
                                futuro_est = calcular_preco_futuro(spot, "WIN")
                                precos["WIN"] = {
                                    "preco": futuro_est,
                                    "variacao": float(ibov.get("variation", 0)),
                                    "fonte": "HG Brasil + Basis estimado (futuro)",
                                }
                                logger.info(f"Basis WIN: spot={spot} -> futuro={futuro_est}")
                        if "WDO" in ativos_faltando:
                            usd = results.get("currencies", {}).get("USD", {})
                            if usd and usd.get("buy"):
                                spot = float(usd["buy"])
                                futuro_est = calcular_preco_futuro(spot, "WDO")
                                precos["WDO"] = {
                                    "preco": futuro_est,
                                    "variacao": float(usd.get("variation", 0)),
                                    "fonte": "HG Brasil + Basis estimado (futuro)",
                                }
                                logger.info(f"Basis WDO: spot={spot} -> futuro={futuro_est}")
            except Exception as e:
                logger.error(f"Erro cálculo basis: {e}")

                # === SOURCE 3: HG Brasil (último fallback) ===
        ativos_faltando = [a for a in ["WIN", "WDO"] if a not in precos]
        if ativos_faltando:
            try:
                async with httpx.AsyncClient(timeout=8) as client:
                    resp = await client.get(
                        "https://api.hgbrasil.com/finance",
                        params={"format": "json", "key": self.hg_api_key}
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
