"""
B3 Day Trade Analyzer - API Principal
FastAPI backend com analise automatica a cada 5 minutos.
Cache persistente: mostra dados do ultimo pregao fora do horario.

Endpoints:
  GET /              -> Dashboard web
  GET /api/analise   -> Analise completa (ativo, timeframe)
  GET /api/painel    -> Painel multi-timeframe
  GET /api/sinais    -> Sinais ativos de entrada
  GET /api/status    -> Status do sistema
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from analysis_engine import analisar_completo
from data_provider import DataProvider

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fuso horario de Brasilia (UTC-3)
BRT = timezone(timedelta(hours=-3))

# Caminho do cache persistente
APP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = APP_DIR / "cache_analises.json"

# Estado global da aplicacao
app_state = {
    "ultima_atualizacao": None,
    "analises": {},
    "provider": None,
    "auto_refresh_task": None,
    "usando_cache": False,
    "cache_data_pregao": None,
}

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
ATIVOS = ["WIN", "WDO"]


def mercado_aberto() -> bool:
    """Verifica se o mercado B3 esta aberto (9:00-18:00 BRT, seg-sex)"""
    agora = datetime.now(BRT)
    if agora.weekday() >= 5:
        return False
    if agora.hour < 9 or agora.hour >= 18:
        return False
    return True


def salvar_cache(analises: dict, timestamp: str):
    """Salva analises bem-sucedidas em arquivo JSON para persistencia"""
    try:
        cache_data = {
            "timestamp": timestamp,
            "data_pregao": datetime.now(BRT).strftime("%d/%m/%Y"),
            "analises": analises,
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, default=str)
        logger.info(f"Cache salvo com sucesso: {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Erro ao salvar cache: {e}")


def carregar_cache() -> bool:
    """Carrega analises do cache em disco. Retorna True se carregou."""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            app_state["analises"] = cache_data.get("analises", {})
            app_state["ultima_atualizacao"] = cache_data.get("timestamp")
            app_state["cache_data_pregao"] = cache_data.get("data_pregao")
            app_state["usando_cache"] = True
            logger.info(f"Cache carregado: pregao de {app_state['cache_data_pregao']}")
            return True
    except Exception as e:
        logger.error(f"Erro ao carregar cache: {e}")
    return False


async def atualizar_analises():
    """Atualiza todas as analises para todos os ativos e timeframes"""
    provider = app_state["provider"]
    resultados = {}
    tem_dados_validos = False

    for ativo in ATIVOS:
        resultados[ativo] = {}
        for tf in TIMEFRAMES:
            try:
                dados = await provider.obter_dados(ativo, tf)
                if dados is not None and len(dados) >= 30:
                    analise = analisar_completo(dados, tf, ativo)
                    resultados[ativo][tf] = analise
                    tem_dados_validos = True
                    logger.info(f"Analise atualizada: {ativo}/{tf}")
                else:
                    resultados[ativo][tf] = {"erro": "Dados insuficientes"}
            except Exception as e:
                logger.error(f"Erro na analise {ativo}/{tf}: {e}")
                resultados[ativo][tf] = {"erro": str(e)}

    if tem_dados_validos:
        app_state["analises"] = resultados
        app_state["ultima_atualizacao"] = datetime.now(BRT).strftime("%Y-%m-%d %H:%M:%S")
        app_state["usando_cache"] = False
        app_state["cache_data_pregao"] = None
        salvar_cache(resultados, app_state["ultima_atualizacao"])
        logger.info(f"Analises atualizadas em {app_state['ultima_atualizacao']}")
    else:
        if not app_state["analises"] or app_state["usando_cache"]:
            carregar_cache()
            logger.info("Sem dados novos, mantendo cache do ultimo pregao")


async def auto_refresh_loop():
    """Loop de atualizacao automatica - inteligente com horario de mercado"""
    while True:
        try:
            if mercado_aberto():
                await atualizar_analises()
                logger.info("Mercado aberto - dados atualizados")
            else:
                if not app_state["analises"]:
                    carregar_cache()
                logger.info("Mercado fechado - usando cache do ultimo pregao")
        except Exception as e:
            logger.error(f"Erro no auto-refresh: {e}")
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup e shutdown do app"""
    source = os.getenv("DATA_SOURCE", "yfinance")
    app_state["provider"] = DataProvider(source=source)
    logger.info(f"Data provider inicializado: {source}")

    cache_ok = carregar_cache()

    if mercado_aberto():
        logger.info("Mercado aberto, buscando dados atualizados...")
        await atualizar_analises()
    elif cache_ok:
        logger.info(f"Mercado fechado - exibindo dados do pregao de {app_state['cache_data_pregao']}")
    else:
        logger.info("Sem cache disponivel, tentando buscar dados historicos...")
        await atualizar_analises()

    app_state["auto_refresh_task"] = asyncio.create_task(auto_refresh_loop())
    logger.info("Auto-refresh iniciado (intervalo: 5 minutos)")

    yield

    if app_state["auto_refresh_task"]:
        app_state["auto_refresh_task"].cancel()


app = FastAPI(
    title="B3 Day Trade Analyzer",
    description="Analise tecnica em tempo real para Mini-Indice e Mini-Dolar da B3",
    version="1.1.0",
    lifespan=lifespan
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Pagina principal do dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/analise")
async def get_analise(
    ativo: str = Query("WIN", description="Ativo: WIN ou WDO"),
    timeframe: str = Query("5m", description="Timeframe: 5m, 15m, 1h, 4h, 1d")
):
    """Retorna analise completa para um ativo e timeframe"""
    ativo = ativo.upper()
    if ativo not in ATIVOS:
        return JSONResponse({"erro": f"Ativo invalido. Use: {ATIVOS}"}, status_code=400)
    if timeframe not in TIMEFRAMES:
        return JSONResponse({"erro": f"Timeframe invalido. Use: {TIMEFRAMES}"}, status_code=400)

    analise = app_state["analises"].get(ativo, {}).get(timeframe)
    if not analise:
        try:
            dados = await app_state["provider"].obter_dados(ativo, timeframe)
            if dados is not None and len(dados) >= 30:
                analise = analisar_completo(dados, timeframe, ativo)
        except Exception as e:
            return JSONResponse({"erro": str(e)}, status_code=500)

    return JSONResponse({
        "ultima_atualizacao": app_state["ultima_atualizacao"],
        "analise": analise,
        "usando_cache": app_state["usando_cache"],
        "cache_data_pregao": app_state["cache_data_pregao"],
        "mercado_aberto": mercado_aberto(),
    })


@app.get("/api/painel")
async def get_painel(
    ativo: str = Query("WIN", description="Ativo: WIN ou WDO")
):
    """Retorna painel multi-timeframe completo para um ativo"""
    ativo = ativo.upper()
    if ativo not in ATIVOS:
        return JSONResponse({"erro": "Ativo invalido"}, status_code=400)

    painel = app_state["analises"].get(ativo, {})

    resumo = {
        "ativo": ativo,
        "tendencia_geral": _calcular_tendencia_geral(painel),
        "sinal_principal": _sinal_principal(painel),
        "timeframes": {}
    }

    for tf in TIMEFRAMES:
        analise = painel.get(tf, {})
        if "erro" not in analise and analise:
            resumo["timeframes"][tf] = {
                "tendencia": analise.get("tendencia", "N/A"),
                "rsi": analise.get("rsi", 0),
                "rsi_status": analise.get("rsi_status", "N/A"),
                "macd_status": analise.get("macd_status", "N/A"),
                "volume_pressao": analise.get("volume", {}).get("pressao", "N/A"),
                "violinada_score": analise.get("violinada_score", 0),
                "sinais": analise.get("sinais", []),
                "preco_atual": analise.get("preco_atual", 0),
            }

    return JSONResponse({
        "ultima_atualizacao": app_state["ultima_atualizacao"],
        "painel": resumo,
        "usando_cache": app_state["usando_cache"],
        "cache_data_pregao": app_state["cache_data_pregao"],
        "mercado_aberto": mercado_aberto(),
    })


@app.get("/api/sinais")
async def get_sinais(
    ativo: str = Query("WIN", description="Ativo: WIN ou WDO")
):
    """Retorna sinais ativos de entrada - foco no timeframe de 5 minutos"""
    ativo = ativo.upper()
    analise_5m = app_state["analises"].get(ativo, {}).get("5m", {})

    sinais = analise_5m.get("sinais", [])

    confirmacoes = {}
    for tf in ["15m", "1h", "4h", "1d"]:
        analise_tf = app_state["analises"].get(ativo, {}).get(tf, {})
        if analise_tf and "erro" not in analise_tf:
            confirmacoes[tf] = {
                "tendencia": analise_tf.get("tendencia", "N/A"),
                "rsi_status": analise_tf.get("rsi_status", "N/A"),
                "macd_status": analise_tf.get("macd_status", "N/A"),
            }

    return JSONResponse({
        "ultima_atualizacao": app_state["ultima_atualizacao"],
        "ativo": ativo,
        "timeframe_operacao": "5m",
        "sinais": sinais,
        "confirmacoes_timeframes": confirmacoes,
        "preco_atual": analise_5m.get("preco_atual", 0),
    })


@app.get("/api/status")
async def get_status():
    """Status do sistema"""
    return JSONResponse({
        "status": "online",
        "ultima_atualizacao": app_state["ultima_atualizacao"],
        "data_source": os.getenv("DATA_SOURCE", "yfinance"),
        "ativos_monitorados": ATIVOS,
        "timeframes": TIMEFRAMES,
        "intervalo_refresh": "5 minutos",
        "versao": "1.1.0",
        "mercado_aberto": mercado_aberto(),
        "usando_cache": app_state["usando_cache"],
        "cache_data_pregao": app_state["cache_data_pregao"],
    })


@app.post("/api/forcar-atualizacao")
async def forcar_atualizacao():
    """Forca uma atualizacao imediata de todas as analises"""
    await atualizar_analises()
    return JSONResponse({
        "mensagem": "Analises atualizadas com sucesso",
        "ultima_atualizacao": app_state["ultima_atualizacao"]
    })


def _calcular_tendencia_geral(painel: dict) -> str:
    """Calcula tendencia geral baseada em multiplos timeframes"""
    pesos = {"5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5}
    score = 0
    total_peso = 0

    for tf, peso in pesos.items():
        analise = painel.get(tf, {})
        if analise and "erro" not in analise:
            tendencia = analise.get("tendencia", "LATERAL")
            if tendencia == "ALTA":
                score += peso
            elif tendencia == "BAIXA":
                score -= peso
            total_peso += peso

    if total_peso == 0:
        return "INDEFINIDO"

    ratio = score / total_peso
    if ratio > 0.3:
        return "ALTA"
    elif ratio < -0.3:
        return "BAIXA"
    return "LATERAL"


def _sinal_principal(painel: dict) -> dict:
    """Extrai o sinal principal do timeframe de 5 minutos"""
    analise_5m = painel.get("5m", {})
    sinais = analise_5m.get("sinais", [])

    if sinais:
        melhor = max(sinais, key=lambda s: s.get("confianca", 0))
        return melhor

    return {"tipo": "NEUTRO", "confianca": 0, "motivos": ["Sem sinais claros no momento"]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
