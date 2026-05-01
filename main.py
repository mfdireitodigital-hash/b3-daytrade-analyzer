"""
B3 Day Trade Analyzer - API Principal
FastAPI backend com análise automática a cada 5 minutos.
Cache persistente: mostra dados do último pregão fora do horário.
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

BRT = timezone(timedelta(hours=-3))

APP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = APP_DIR / "cache_analises.json"

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

USERS_FILE = APP_DIR / "usuarios.json"
ADMIN_USERS = ["fabianodomingues", "fabiodomingues"]


def _hash_senha(senha: str) -> str:
    return hashlib.sha256(senha.encode()).hexdigest()


def carregar_usuarios() -> dict:
    try:
        if USERS_FILE.exists():
            with open(USERS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    users = {
        "fabianodomingues": {"senha_hash": _hash_senha("123@mudar"), "admin": True},
        "fabiodomingues": {"senha_hash": _hash_senha("123@mudar"), "admin": True},
    }
    salvar_usuarios(users)
    return users


def salvar_usuarios(users: dict):
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
    agora = datetime.now(BRT)
    if agora.weekday() >= 5:
        return False
    if agora.hour < 9 or agora.hour >= 18:
        return False
    return True


def salvar_cache(analises: dict, timestamp: str):
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
    version="3.0.0",
    lifespan=lifespan
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/analise")
async def get_analise(
    ativo: str = Query("WIN"),
    timeframe: str = Query("5m")
):
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
async def get_painel(ativo: str = Query("WIN")):
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
async def get_sinais(ativo: str = Query("WIN")):
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
    # Include contract info
    contratos = {}
    for ativo in ATIVOS:
        c = obter_contrato_vigente(ativo)
        contratos[ativo] = c.get("ticker_b3", ativo)

    return JSONResponse({
        "status": "online",
        "ultima_atualizacao": app_state["ultima_atualizacao"],
        "data_source": os.getenv("DATA_SOURCE", "yfinance"),
        "ativos_monitorados": ATIVOS,
        "contratos_vigentes": contratos,
        "timeframes": TIMEFRAMES,
        "intervalo_refresh": "5 minutos",
        "versao": "3.0.0",
        "mercado_aberto": mercado_aberto(),
        "usando_cache": app_state["usando_cache"],
        "cache_data_pregao": app_state["cache_data_pregao"],
    })


@app.post("/api/forcar-atualizacao")
async def forcar_atualizacao():
    await atualizar_analises()
    return JSONResponse({
        "mensagem": "Análises atualizadas com sucesso",
        "ultima_atualizacao": app_state["ultima_atualizacao"]
    })


@app.post("/api/login")
async def login(req: LoginRequest):
    usuario = req.usuario.strip().lower()
    senha = req.senha.strip()
    users = carregar_usuarios()
    if usuario in users:
        user_data = users[usuario]
        if isinstance(user_data, dict):
            if user_data.get("senha_hash") == _hash_senha(senha):
                is_admin = user_data.get("admin", False)
                return JSONResponse({"sucesso": True, "usuario": usuario, "admin": is_admin, "mensagem": "Login realizado com sucesso"})
        elif user_data == senha:
            return JSONResponse({"sucesso": True, "usuario": usuario, "admin": usuario in ADMIN_USERS, "mensagem": "Login realizado com sucesso"})
    return JSONResponse({"sucesso": False, "mensagem": "Usuário ou senha inválidos"}, status_code=401)


@app.post("/api/usuarios/criar")
async def criar_usuario(req: CriarUsuarioRequest):
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
    usuario = req.usuario.strip().lower()
    users = carregar_usuarios()
    if usuario not in users:
        return JSONResponse({"sucesso": False, "mensagem": "Usuário não encontrado"}, status_code=404)
    user_data = users[usuario]
    if isinstance(user_data, dict):
        if user_data.get("senha_hash") != _hash_senha(req.senha_atual.strip()):
            return JSONResponse({"sucesso": False, "mensagem": "Senha atual incorreta"}, status_code=401)
    elif user_data != req.senha_atual.strip():
        return JSONResponse({"sucesso": False, "mensagem": "Senha atual incorreta"}, status_code=401)
    users[usuario] = {"senha_hash": _hash_senha(req.senha_nova.strip()), "admin": user_data.get("admin", usuario in ADMIN_USERS) if isinstance(user_data, dict) else usuario in ADMIN_USERS}
    salvar_usuarios(users)
    return JSONResponse({"sucesso": True, "mensagem": "Senha alterada com sucesso"})


@app.get("/api/candles")
async def get_candles(
    ativo: str = Query("WIN"),
    timeframe: str = Query("5m")
):
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
    ativo = ativo.upper()
    contrato = obter_contrato_vigente(ativo)
    return JSONResponse(contrato)


@app.get("/api/simulacao-capital")
async def get_simulacao_capital(
    ativo: str = Query("WIN"),
    timeframe: str = Query("5m"),
    contratos: int = Query(1, description="Quantidade de contratos do usuário")
):
    """Simula resultado financeiro para a quantidade de contratos definida pelo usuário"""
    ativo = ativo.upper()
    analise = app_state["analises"].get(ativo, {}).get(timeframe, {})
    contrato_info = app_state["provider"].get_contrato_info(ativo)

    # Valores corretos B3
    if ativo == "WIN":
        valor_pt = 0.20  # R$ 0.20 por ponto por contrato de mini-índice
    else:
        valor_pt = 10.00  # R$ 10.00 por ponto por contrato de mini-dólar

    tick = contrato_info.get("tick", 5 if ativo == "WIN" else 0.5)

    # Pegar sinais e ATR
    sinais = analise.get("sinais", [])
    atr = analise.get("atr", 100 if ativo == "WIN" else 8)

    total_pts = 0
    ops = 0

    if sinais:
        for sinal in sinais:
            pts = sinal.get("pts_estimados", 0)
            if pts == 0:
                pts = atr * 0.5
            total_pts += pts
            ops += 1
    else:
        # Estimativa baseada em ATR
        random.seed(hash(f"{ativo}_{datetime.now(BRT).strftime('%Y%m%d')}"))
        n_ops = random.randint(4, 8)
        taxa_acerto = 0.60
        tendencia = analise.get("tendencia", "LATERAL")
        if tendencia != "LATERAL":
            taxa_acerto = 0.65
        for i in range(n_ops):
            alvo_pts = atr * random.uniform(0.4, 1.2)
            stop_pts = atr * random.uniform(0.3, 0.6)
            if random.random() < taxa_acerto:
                total_pts += alvo_pts
            else:
                total_pts -= stop_pts
            ops += 1

    # Cálculo para a quantidade do usuário
    n = max(1, contratos)
    resultado_fin = round(total_pts * valor_pt * n, 2)

    return JSONResponse({
        "ativo": ativo,
        "contrato": contrato_info.get("ticker_b3", ativo),
        "contratos": n,
        "valor_ponto": valor_pt,
        "tick": tick,
        "pontos_estimados": round(total_pts, 1),
        "resultado_financeiro": resultado_fin,
        "operacoes_estimadas": ops,
        "info_contrato": {
            "nome": contrato_info.get("nome", ativo),
            "tick": tick,
            "valor_tick": valor_pt,
            "explicacao": f"Cada ponto = R$ {valor_pt:.2f} por contrato. {n} contrato(s) = R$ {valor_pt * n:.2f}/ponto"
        }
    })


@app.get("/api/demo")
async def get_demo(ativo: str = Query("WIN"), contratos: int = Query(1)):
    """Gera simulação de operações de day trade para modo demo"""
    ativo_upper = ativo.upper()
    n_contratos = max(1, contratos)

    # Pegar preço real da análise
    analise_5m = app_state["analises"].get(ativo_upper, {}).get("5m", {})
    preco_real = analise_5m.get("preco_atual", 0)
    atr_real = analise_5m.get("atr", 0)

    contrato_info = obter_contrato_vigente(ativo_upper)
    ticker_b3 = contrato_info["ticker_b3"]

    if ativo_upper == "WIN":
        valor_pt = 0.20
        if preco_real == 0:
            preco_real = 130000
        if atr_real == 0:
            atr_real = 150
    else:
        valor_pt = 10.00
        if preco_real == 0:
            preco_real = 5700
        if atr_real == 0:
            atr_real = 12

    operacoes = []
    total_resultado = 0
    random.seed(hash(f"demo_{ativo_upper}_{datetime.now(BRT).strftime('%Y%m%d')}"))

    # Horários operacionais B3 realistas
    horarios_entrada = [
        "09:18", "09:35", "09:52", "10:15", "10:42",
        "11:05", "11:28", "14:10", "14:38", "15:05",
        "15:32", "16:00", "16:25"
    ]

    n_ops = random.randint(5, 10)
    tipos_motivos = {
        "COMPRA": [
            "RSI sobrevendido + MACD alta + Volume comprador + Pullback EMA 9",
            "Fibonacci 61.8% + MACD cruzamento alta + Volume acima media",
            "Teste VWAP com rejeição + RSI favorável + MACD positivo",
            "Pullback EMA 21 + Volume comprador crescente + Fibonacci 50%",
            "Rompimento resistência + MACD hist crescente + Volume alto",
        ],
        "VENDA": [
            "RSI sobrecomprado + MACD baixa + Volume vendedor + Rejeição VWAP",
            "Fibonacci 38.2% baixa + MACD cruzamento baixa + Volume alto",
            "Teste resistência com rejeição + RSI zona venda + MACD negativo",
            "Pullback EMA 9 em baixa + Volume vendedor + Fibonacci 50%",
            "Rompimento suporte + MACD hist decrescente + Pressão vendedora",
        ]
    }

    for i in range(n_ops):
        tipo = random.choice(["COMPRA", "VENDA"])
        variacao_entrada = random.uniform(-atr_real * 1.5, atr_real * 1.5)
        entrada = round(preco_real + variacao_entrada, 1 if ativo_upper == "WDO" else 0)

        stop_dist = round(atr_real * random.uniform(0.5, 1.0), 1)
        alvo_dist = round(stop_dist * random.uniform(2.0, 3.0), 1)

        if tipo == "COMPRA":
            stop = round(entrada - stop_dist, 1)
            alvo = round(entrada + alvo_dist, 1)
        else:
            stop = round(entrada + stop_dist, 1)
            alvo = round(entrada - alvo_dist, 1)

        # Resultado: 60% taxa de acerto
        win = random.random() < 0.60
        if win:
            pts = round(random.uniform(alvo_dist * 0.6, alvo_dist), 1)
            if tipo == "COMPRA":
                saida = round(entrada + pts, 1)
            else:
                saida = round(entrada - pts, 1)
        else:
            pts = -stop_dist
            if tipo == "COMPRA":
                saida = round(entrada - stop_dist, 1)
            else:
                saida = round(entrada + stop_dist, 1)

        resultado = round(pts * valor_pt * n_contratos, 2)
        total_resultado += resultado

        horario_idx = min(i, len(horarios_entrada) - 1)
        horario = horarios_entrada[horario_idx]

        # Horário de saída (5-25 min depois)
        h_parts = horario.split(":")
        h_min = int(h_parts[0]) * 60 + int(h_parts[1]) + random.randint(5, 25)
        h_saida = f"{h_min // 60:02d}:{h_min % 60:02d}"

        rr_val = round(alvo_dist / stop_dist, 1) if stop_dist > 0 else 2.0
        motivo = random.choice(tipos_motivos[tipo])

        operacoes.append({
            "id": i + 1,
            "ativo": ticker_b3,
            "tipo": tipo,
            "entrada": entrada,
            "stop": stop,
            "alvo": alvo,
            "saida": saida,
            "pts": round(pts, 1),
            "rr": f"{rr_val}:1",
            "contratos": n_contratos,
            "resultado": resultado,
            "win": resultado > 0,
            "status": "WIN" if resultado > 0 else "LOSS",
            "hora_entrada": horario,
            "hora_saida": h_saida,
            "horario": f"{horario} → {h_saida}",
            "motivo": motivo,
            "indicadores_cruzados": "Todos os 5 indicadores confirmaram" if resultado > 0 else "4 de 5 indicadores confirmaram",
        })

    wins = sum(1 for op in operacoes if op["win"])
    losses = len(operacoes) - wins
    total_pts = sum(op["pts"] for op in operacoes)

    data_hoje = datetime.now(BRT).strftime("%d/%m/%Y")

    return JSONResponse({
        "ativo": ativo_upper,
        "contrato": ticker_b3,
        "data": data_hoje,
        "contratos_operados": n_contratos,
        "valor_ponto": valor_pt,
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
            "contratos": n_contratos,
            "valor_ponto": valor_pt,
        }
    })


def _calcular_tendencia_geral(painel: dict) -> str:
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
