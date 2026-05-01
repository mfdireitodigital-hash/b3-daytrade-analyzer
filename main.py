"""
B3 Day Trade Analyzer - API Principal
FastAPI backend com análise automática a cada 5 minutos.
Cache persistente: mostra dados do último pregão fora do horário.

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
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np

from analysis_engine import analisar_completo
from data_provider import DataProvider

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fuso horário de Brasília (UTC-3)
BRT = timezone(timedelta(hours=-3))

# Caminho do cache persistente
APP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = APP_DIR / "cache_analises.json"

# Estado global da aplicação
app_state = {
    "ultima_atualizacao": None,
    "analises": {},
    "provider": None,
    "auto_refresh_task": None,
    "usando_cache": False,           # True quando exibindo dados do cache
    "cache_data_pregao": None,       # Data/hora do pregão cacheado
}

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
ATIVOS = ["WIN", "WDO"]

# Usuários autorizados
USUARIOS = {
    "fabianodomingues": "123@mudar",
    "fabiodomingues": "123@mudar",
}


class LoginRequest(BaseModel):
    usuario: str
    senha: str


def converter_numpy(obj):
    """Converte tipos numpy para tipos Python nativos para serialização JSON"""
    if isinstance(obj, dict):
        return {k: converter_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [converter_numpy(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def mercado_aberto() -> bool:
    """Verifica se o mercado B3 está aberto (9:00-18:00 BRT, seg-sex)"""
    agora = datetime.now(BRT)
    # Fim de semana
    if agora.weekday() >= 5:
        return False
    # Fora do horário (9h às 18h)
    if agora.hour < 9 or agora.hour >= 18:
        return False
    return True


def salvar_cache(analises: dict, timestamp: str):
    """Salva análises bem-sucedidas em arquivo JSON para persistência"""
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
    """Carrega análises do cache em disco. Retorna True se carregou."""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            app_state["analises"] = cache_data.get("analises", {})
            app_state["ultima_atualizacao"] = cache_data.get("timestamp")
            app_state["cache_data_pregao"] = cache_data.get("data_pregao")
            app_state["usando_cache"] = True
            logger.info(f"Cache carregado: pregão de {app_state['cache_data_pregao']}")
            return True
    except Exception as e:
        logger.error(f"Erro ao carregar cache: {e}")
    return False


async def atualizar_analises():
    """Atualiza todas as análises para todos os ativos e timeframes"""
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
                    analise = converter_numpy(analise)
                    resultados[ativo][tf] = analise
                    tem_dados_validos = True
                    logger.info(f"Análise atualizada: {ativo}/{tf}")
                else:
                    resultados[ativo][tf] = {"erro": "Dados insuficientes"}
            except Exception as e:
                logger.error(f"Erro na análise {ativo}/{tf}: {e}")
                resultados[ativo][tf] = {"erro": str(e)}

    if tem_dados_validos:
        app_state["analises"] = resultados
        app_state["ultima_atualizacao"] = datetime.now(BRT).strftime("%Y-%m-%d %H:%M:%S")
        app_state["usando_cache"] = False
        app_state["cache_data_pregao"] = None
        # Salvar cache para uso fora do horário
        salvar_cache(resultados, app_state["ultima_atualizacao"])
        logger.info(f"Análises atualizadas em {app_state['ultima_atualizacao']}")
    else:
        # Sem dados novos - usar cache se disponível
        if not app_state["analises"] or app_state["usando_cache"]:
            carregar_cache()
            logger.info("Sem dados novos, mantendo cache do último pregão")


async def auto_refresh_loop():
    """Loop de atualização automática - inteligente com horário de mercado"""
    while True:
        try:
            if mercado_aberto():
                await atualizar_analises()
                logger.info("Mercado aberto - dados atualizados")
            else:
                # Fora do horário: garantir que temos cache carregado
                if not app_state["analises"]:
                    carregar_cache()
                logger.info("Mercado fechado - usando cache do último pregão")
        except Exception as e:
            logger.error(f"Erro no auto-refresh: {e}")
        await asyncio.sleep(300)  # 5 minutos


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup e shutdown do app"""
    source = os.getenv("DATA_SOURCE", "yfinance")
    app_state["provider"] = DataProvider(source=source)
    logger.info(f"Data provider inicializado: {source}")

    # Carregar cache primeiro (garante dados imediatos)
    cache_ok = carregar_cache()

    if mercado_aberto():
        # Mercado aberto: tentar dados frescos
        logger.info("Mercado aberto, buscando dados atualizados...")
        await atualizar_analises()
    elif cache_ok:
        logger.info(f"Mercado fechado - exibindo dados do pregão de {app_state['cache_data_pregao']}")
    else:
        # Sem cache e mercado fechado: tentar buscar mesmo assim (yfinance pode ter dados históricos)
        logger.info("Sem cache disponível, tentando buscar dados históricos...")
        await atualizar_analises()

    # Iniciar loop de auto-refresh em background
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


@app.post("/api/login")
async def login(req: LoginRequest):
    """Autenticação de usuário"""
    usuario = req.usuario.strip().lower()
    senha = req.senha.strip()
    if usuario in USUARIOS and USUARIOS[usuario] == senha:
        return JSONResponse({"sucesso": True, "usuario": usuario, "mensagem": "Login realizado com sucesso"})
    return JSONResponse({"sucesso": False, "mensagem": "Usuário ou senha inválidos"}, status_code=401)


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
            dados = await app_state["provider"].obter_dados(ati