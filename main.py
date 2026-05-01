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
import random
import hashlib

import numpy as np

from analysis_engine import analisar_completo
from data_provider import DataProvider, obter_contrato_vigente

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
    "usando_cache": False,
    "cache_data_pregao": None,
}

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
ATIVOS = ["WIN", "WDO"]

# Arquivo de usuários persistente
USERS_FILE = APP_DIR / "usuarios.json"

# Usuários admin (podem criar novos users)
ADMIN_USERS = ["fabianodomingues", "fabiodomingues"]


def _hash_senha(senha: str) -> str:
    return hashlib.sha256(senha.encode()).hexdigest()


def carregar_usuarios() -> dict:
    """Carrega usuários do arquivo JSON ou cria com defaults"""
    try:
        if USERS_FILE.exists():
            with open(USERS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    # Defaults
    users = {
        "fabianodomingues": {"senha_hash": _hash_senha("123@mudar"), "admin": True},
        "fabiodomingues": {"senha_hash": _hash_senha("123@mudar"), "admin": True},
    }
    salvar_usuarios(users)
    return users


def salvar_usuarios(users: dict):
    """Salva usuários no arquivo JSON"""
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erro ao salvar usuarios: {e}")


class LoginRequest(BaseModel):
    usuario: str
    senha: str


class CriarUsuarioRequest(BaseModel):
    usuario: str
    senha: str
    admin_user: str


class AlterarSenhaRequest(BaseModel):
    usuario: str
    senha_atual: str
    senha_nova: str


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
    if agora.weekday() >= 5:
        return False
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
        salvar_cache(resultados, app_state["ultima_atualizacao"])
        logger.info(f"Análises atualizadas em {app_state['ultima_atualizacao']}")
    else:
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
                if not app_state["analises"]:
                    carregar_cache()
                logger.info("Mercado fechado - usando cache do último pregão")
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
        logger.info(f"Mercado fechado - exibindo dados do pregão de {app_state['cache_data_pregao']}")
    else:
        logger.info("Sem cache disponível, tentando buscar dados históricos...")
        await atualizar_analises()

    app_state["auto_refresh_task"] = asyncio.create_task(auto_refresh_loop())
    logger.info("Auto-refresh iniciado (intervalo: 5 minutos)")

    yield

    if app_state["auto_refresh_task"]:
        app_state["auto_refresh_task"].cancel()


app = FastAPI(
    title="ANALISE B3 - 24/7",
    description="ANALISE TECNICA EM TEMPO REAL - MINI-INDICE E MINI-DOLAR DA B3",
    version="2.0.0",
    lifespan=lifespan
)

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
        try:
            dados = await app_state["provider"].obter_dados(ativo, timeframe)
            if dados is not None and len(dados) >= 30:
                analise = analisar_completo(dados, timeframe, ativo)
                analise = converter_numpy(analise)
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
        return JSONResponse({"erro": f"Ativo inválido"}, status_code=400)

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
        "versao": "2.0.0",
        "mercado_aberto": mercado_aberto(),
        "usando_cache": app_state["usando_cache"],
        "cache_data_pregao": app_state["cache_data_pregao"],
    })


@app.post("/api/forcar-atualizacao")
async def forcar_atualizacao():
    """Força uma atualização imediata de todas as análises"""
    await atualizar_analises()
    return JSONResponse({
        "mensagem": "Análises atualizadas com sucesso",
        "ultima_atualizacao": app_state["ultima_atualizacao"]
    })


@app.post("/api/login")
async def login(req: LoginRequest):
    """Autenticação de usuários"""
    usuario = req.usuario.strip().lower()
    senha = req.senha.strip()
    users = carregar_usuarios()
    if usuario in users:
        user_data = users[usuario]
        # Suporte legacy (senha em texto) e novo (hash)
        if isinstance(user_data, dict):
            if user_data.get("senha_hash") == _hash_senha(senha):
                is_admin = user_data.get("admin", False)
                return JSONResponse({"sucesso": True, "usuario": usuario, "admin": is_admin, "mensagem": "Login realizado com sucesso"})
        elif user_data == senha:  # legacy
            return JSONResponse({"sucesso": True, "usuario": usuario, "admin": usuario in ADMIN_USERS, "mensagem": "Login realizado com sucesso"})
    return JSONResponse({"sucesso": False, "mensagem": "Usuário ou senha inválidos"}, status_code=401)


@app.post("/api/usuarios/criar")
async def criar_usuario(req: CriarUsuarioRequest):
    """Cria novo usuário (apenas admins)"""
    admin = req.admin_user.strip().lower()
    users = carregar_usuarios()
    user_data = users.get(admin, {})
    is_admin = user_data.get("admin", False) if isinstance(user_data, dict) else admin in ADMIN_USERS
    if not is_admin:
        return JSONResponse({"sucesso": False, "mensagem": "Sem permissão"}, status_code=403)
    novo_user = req.usuario.strip().lower()
    if novo_user in users:
        return JSONResponse({"sucesso": False, "mensagem": "Usuário já existe"}, status_code=400)
    users[novo_user] = {"senha_hash": _hash_senha(req.senha.strip()), "admin": False}
    salvar_usuarios(users)
    return JSONResponse({"sucesso": True, "mensagem": f"Usuário {novo_user} criado com sucesso"})


@app.post("/api/usuarios/alterar-senha")
async def alterar_senha(req: AlterarSenhaRequest):
    """Altera senha do próprio usuário"""
    usuario = req.usuario.strip().lower()
    users = carregar_usuarios()
    if usuario not in users:
        return JSONResponse({"sucesso": False, "mensagem": "Usuário não encontrado"}, status_code=404)
    user_data = users[usuario]
    # Verificar senha atual
    if isinstance(user_data, dict):
        if user_data.get("senha_hash") != _hash_senha(req.senha_atual.strip()):
            return JSONResponse({"sucesso": False, "mensagem": "Senha atual incorreta"}, status_code=401)
    elif user_data != req.senha_atual.strip():
        return JSONResponse({"sucesso": False, "mensagem": "Senha atual incorreta"}, status_code=401)
    # Atualizar
    users[usuario] = {"senha_hash": _hash_senha(req.senha_nova.strip()), "admin": user_data.get("admin", usuario in ADMIN_USERS) if isinstance(user_data, dict) else usuario in ADMIN_USERS}
    salvar_usuarios(users)
    return JSONResponse({"sucesso": True, "mensagem": "Senha alterada com sucesso"})


@app.get("/api/candles")
async def get_candles(
    ativo: str = Query("WIN", description="Ativo: WIN ou WDO"),
    timeframe: str = Query("5m", description="Timeframe")
):
    """Retorna dados OHLCV para gráficos de velas"""
    ativo = ativo.upper()
    if ativo not in ATIVOS:
        return JSONResponse({"erro": "Ativo inválido"}, status_code=400)
    candles = await app_state["provider"].obter_candles_json(ativo, timeframe)
    contrato = app_state["provider"].get_contrato_info(ativo)
    return JSONResponse({
        "candles": candles,
        "contrato": contrato.get("ticker_b3", ativo),
        "nome": contrato.get("nome", ativo),
        "timeframe": timeframe,
    })


@app.get("/api/contrato")
async def get_contrato(ativo: str = Query("WIN")):
    """Retorna informações do contrato vigente"""
    ativo = ativo.upper()
    contrato = obter_contrato_vigente(ativo)
    return JSONResponse(contrato)


@app.get("/api/simulacao-capital")
async def get_simulacao_capital(
    ativo: str = Query("WIN"),
    timeframe: str = Query("5m")
):
    """Simula resultado financeiro para diferentes quantidades de contratos"""
    ativo = ativo.upper()
    analise = app_state["analises"].get(ativo, {}).get(timeframe, {})
    contrato_info = app_state["provider"].get_contrato_info(ativo)
    valor_pt = contrato_info.get("valor_tick", 0.20)

    # Pegar sinais do dia
    sinais = analise.get("sinais", [])
    resultados = []
    for n_contratos in [1, 2, 3, 5, 10, 20, 50]:
        total_pts = 0
        ops = 0
        for sinal in sinais:
            pts = sinal.get("pts_estimados", 0)
            if pts == 0:
                atr = analise.get("atr", 50 if "WIN" in ativo else 5)
                pts = atr * 0.5  # estimativa conservadora
            total_pts += pts
            ops += 1
        resultado_fin = round(total_pts * valor_pt * n_contratos, 2)
        resultados.append({
            "contratos": n_contratos,
            "pontos": round(total_pts, 1),
            "resultado": resultado_fin,
            "operacoes": ops,
        })

    return JSONResponse({
        "ativo": ativo,
        "contrato": contrato_info.get("ticker_b3", ativo),
        "valor_ponto": valor_pt,
        "simulacoes": resultados,
    })


@app.get("/api/demo")
async def get_demo(ativo: str = Query("WIN", description="Ativo: WIN ou WDO")):
    """Gera simulação de operações de day trade para modo demo"""
    ativo_upper = ativo.upper()
    operacoes = []
    contrato_win = obter_contrato_vigente("WIN")["ticker_b3"]
    contrato_wdo = obter_contrato_vigente("WDO")["ticker_b3"]
    ativos_demo = [contrato_win, contrato_wdo] if "WIN" in ativo_upper else [contrato_wdo, contrato_win]
    tipos = ["COMPRA", "VENDA"]
    total_resultado = 0

    for i in range(random.randint(5, 10)):
        ativo = random.choice(ativos_demo)
        tipo = random.choice(tipos)
        if "WIN" in ativo:
            entrada = round(random.uniform(125000, 130000), 0)
            variacao = random.uniform(-200, 300)
        else:
            entrada = round(random.uniform(5600, 5900), 1)
            variacao = random.uniform(-15, 20)

        saida = round(entrada + variacao, 1)
        pts = round(variacao, 1)
        valor_pt = 0.20 if "WIN" in ativo else 10.0
        contratos = random.randint(1, 5)
        resultado = round(pts * valor_pt * contratos, 2)
        total_resultado += resultado
        win = resultado > 0
        hora_base = 9 + i
        if hora_base > 17:
            hora_base = 17

        # Calcular stop e alvo para exibicao
        atr_ref = 100 if "WIN" in ativo else 8
        stop_dist = round(random.uniform(0.3, 0.8) * atr_ref, 1)
        alvo_dist = round(random.uniform(1.0, 2.5) * stop_dist, 1)
        if tipo == "COMPRA":
            stop = round(entrada - stop_dist, 1)
            alvo = round(entrada + alvo_dist, 1)
        else:
            stop = round(entrada + stop_dist, 1)
            alvo = round(entrada - alvo_dist, 1)
        rr_val = round(alvo_dist / stop_dist, 1) if stop_dist > 0 else 1.0
        horario = f"{hora_base:02d}:{random.randint(0,59):02d}"

        operacoes.append({
            "id": i + 1,
            "ativo": ativo,
            "tipo": tipo,
            "entrada": entrada,
            "stop": stop,
            "alvo": alvo,
            "saida": saida,
            "pts": pts,
            "rr": f"{rr_val}:1",
            "contratos": contratos,
            "resultado": resultado,
            "win": win,
            "status": "WIN" if win else "LOSS",
            "hora": horario,
            "horario": horario,
            "motivo": random.choice([
                "Pullback na EMA 9 com volume",
                "Rompimento de resistencia com VWAP",
                "Divergencia RSI + suporte",
                "Sinal MACD com confirmacao",
                "Teste de VWAP com rejeicao",
                "Fibonacci 61.8% com volume",
            ]),
        })

    wins = sum(1 for op in operacoes if op["win"])
    losses = len(operacoes) - wins
    total_pts = sum(op["pts"] for op in operacoes)

    data_hoje = datetime.now(BRT).strftime("%d/%m/%Y")

    return JSONResponse({
        "ativo": ativo_upper,
        "data": data_hoje,
        "operacoes": operacoes,
        "resumo": {
            "total": len(operacoes),
            "total_operacoes": len(operacoes),
            "wins": wins,
            "losses": losses,
            "taxa_acerto": round(wins / len(operacoes) * 100, 1) if operacoes else 0,
            "saldo_pontos": round(total_pts, 1),
            "saldo_financeiro": round(total_resultado, 2),
            "resultado_total": round(total_resultado, 2),
            "melhor_op": round(max((op["resultado"] for op in operacoes), default=0), 2),
            "pior_op": round(min((op["resultado"] for op in operacoes), default=0), 2),
        }
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
        melhor = max(sinais, key=lambda s: s.get("confianca", 0))
        return melhor

    return {"tipo": "NEUTRO", "confianca": 0, "motivos": ["Sem sinais claros no momento"]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
