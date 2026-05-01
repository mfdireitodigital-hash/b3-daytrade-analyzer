"""
B3 Day Trade Analyzer - API Principal
FastAPI backend com análise automática a cada 5 minutos.

Endpoints:
  GET /              → Dashboard web
  GET /api/analise   → Análise completa (ativo, timeframe)
  GET /api/painel    → Painel multi-timeframe
  GET /api/sinais    → Sinais ativos de entrada
  GET /api/status    → Status do sistema
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager

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

# Estado global da aplicação
app_state = {
    "ultima_atualizacao": None,
    "analises": {},
    "provider": None,
    "auto_refresh_task": None,
}

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
ATIVOS = ["WIN", "WDO"]


async def atualizar_analises():
    """Atualiza todas as análises para todos os ativos e timeframes"""
    provider = app_state["provider"]
    resultados = {}

    for ativo in ATIVOS:
        resultados[ativo] = {}
        for tf in TIMEFRAMES:
            try:
                dados = await provider.obter_dados(ativo, tf)
                if dados is not None and len(dados) >= 30:
                    analise = analisar_completo(dados, tf, ativo)
                    resultados[ativo][tf] = analise
                    logger.info(f"Análise atualizada: {ativo}/{tf}")
                else:
                    resultados[ativo][tf] = {"erro": "Dados insuficientes"}
            except Exception as e:
                logger.error(f"Erro na análise {ativo}/{tf}: {e}")
                resultados[ativo][tf] = {"erro": str(e)}

    app_state["analises"] = resultados
    app_state["ultima_atualizacao"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Todas as análises atualizadas em {app_state['ultima_atualizacao']}")


async def auto_refresh_loop():
    """Loop de atualização automática a cada 5 minutos"""
    while True:
        try:
            await atualizar_analises()
        except Exception as e:
            logger.error(f"Erro no auto-refresh: {e}")
        await asyncio.sleep(300)  # 5 minutos


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup e shutdown do app"""
    source = os.getenv("DATA_SOURCE", "yfinance")
    app_state["provider"] = DataProvider(source=source)
    logger.info(f"Data provider inicializado: {source}")

    # Primeira atualização
    await atualizar_analises()

    # Iniciar loop de auto-refresh
    app_state["auto_refresh_task"] = asyncio.create_task(auto_refresh_loop())
    logger.info("Auto-refresh iniciado (intervalo: 5 minutos)")

    yield

    # Cleanup
    if app_state["auto_refresh_task"]:
        app_state["auto_refresh_task"].cancel()


app = FastAPI(
    title="B3 Day Trade Analyzer",
    description="Análise técnica em tempo real para Mini-Índice e Mini-Dólar da B3",
    version="1.0.0",
    lifespan=lifespan
)

# Servir arquivos estáticos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Página principal do dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/analise")
async def get_analise(
    ativo: str = Query("WIN", description="Ativo: WIN ou WDO"),
    timeframe: str = Query("5m", description="Timeframe: 5m, 15m, 1h, 4h, 1d")
):
    """Retorna análise completa para um ativo e timeframe"""
    ativo = ativo.upper()
    if ativo not in ATIVOS:
        return JSONResponse({"erro": f"Ativo inválido. Use: {ATIVOS}"}, status_code=400)
    if timeframe not in TIMEFRAMES:
        return JSONResponse({"erro": f"Timeframe inválido. Use: {TIMEFRAMES}"}, status_code=400)

    analise = app_state["analises"].get(ativo, {}).get(timeframe)
    if not analise:
        # Tentar atualizar sob demanda
        try:
            dados = await app_state["provider"].obter_dados(ativo, timeframe)
            if dados is not None and len(dados) >= 30:
                analise = analisar_completo(dados, timeframe, ativo)
        except Exception as e:
            return JSONResponse({"erro": str(e)}, status_code=500)

    return JSONResponse({
        "ultima_atualizacao": app_state["ultima_atualizacao"],
        "analise": analise
    })


@app.get("/api/painel")
async def get_painel(
    ativo: str = Query("WIN", description="Ativo: WIN ou WDO")
):
    """Retorna painel multi-timeframe completo para um ativo"""
    ativo = ativo.upper()
    if ativo not in ATIVOS:
        return JSONResponse({"erro": f"Ativo inválido"}, status_code=400)

    painel = app_state["analises"].get(ativo, {})

    # Resumo consolidado
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
        "painel": resumo
    })


@app.get("/api/sinais")
async def get_sinais(
    ativo: str = Query("WIN", description="Ativo: WIN ou WDO")
):
    """Retorna sinais ativos de entrada - foco no timeframe de 5 minutos"""
    ativo = ativo.upper()
    analise_5m = app_state["analises"].get(ativo, {}).get("5m", {})

    sinais = analise_5m.get("sinais", [])

    # Enriquecer com confirmação de outros timeframes
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
        "versao": "1.0.0"
    })


@app.post("/api/forcar-atualizacao")
async def forcar_atualizacao():
    """Força uma atualização imediata de todas as análises"""
    await atualizar_analises()
    return JSONResponse({
        "mensagem": "Análises atualizadas com sucesso",
        "ultima_atualizacao": app_state["ultima_atualizacao"]
    })


def _calcular_tendencia_geral(painel: dict) -> str:
    """Calcula tendência geral baseada em múltiplos timeframes"""
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
        # Retorna o sinal com maior confiança
        melhor = max(sinais, key=lambda s: s.get("confianca", 0))
        return melhor

    return {"tipo": "NEUTRO", "confianca": 0, "motivos": ["Sem sinais claros no momento"]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
