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
import pandas as pd
import yfinance as yf
import random
import hashlib

import numpy as np
import traceback

from analysis_engine import analisar_completo
from data_provider import DataProvider, obter_contrato_vigente
from learning_engine import (
    carregar_learning, registrar_sessao, obter_pesos_atuais,
    obter_score_minimo, obter_resumo_aprendizado, registrar_livro
)
from trading_books_knowledge import (
    aplicar_scoring_avancado, obter_livros_lista, obter_todos_conceitos
)
from smc_engine import aplicar_smc_scoring

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
    "preco_realtime": {},
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
    """Converte tipos numpy/pandas para tipos Python nativos serializaveis em JSON"""
    if isinstance(obj, dict):
        return {k: converter_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [converter_numpy(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [converter_numpy(x) for x in obj.tolist()]
    elif isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
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
                tb = traceback.format_exc()
                logger.error(f"Erro na análise {ativo}/{tf}: {tb}")
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
            # Sempre atualizar preço em tempo real via HG Brasil
            try:
                precos_rt = await app_state["provider"].obter_preco_realtime()
                if precos_rt:
                    app_state["preco_realtime"] = precos_rt
                    logger.info(f"Preço realtime atualizado via HG Brasil")
            except Exception as e:
                logger.error(f"Erro preço realtime: {e}")

            if mercado_aberto():
                await atualizar_analises()
                logger.info("Mercado aberto - dados atualizados")
            else:
                if not app_state["analises"]:
                    carregar_cache()
                logger.info("Mercado fechado - usando cache do último pregão")
        except Exception as e:
            logger.error(f"Erro no auto-refresh: {e}")
        # 60s when market is open, 300s when closed
        interval = 60 if mercado_aberto() else 300
        await asyncio.sleep(interval)


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

    # Buscar preço realtime na inicialização
    try:
        precos_rt = await app_state["provider"].obter_preco_realtime()
        if precos_rt:
            app_state["preco_realtime"] = precos_rt
            logger.info(f"Preço realtime inicial: {precos_rt}")
    except Exception:
        pass

    app_state["auto_refresh_task"] = asyncio.create_task(auto_refresh_loop())
    logger.info("Auto-refresh iniciado (intervalo: 5 minutos)")

    yield

    if app_state["auto_refresh_task"]:
        app_state["auto_refresh_task"].cancel()


app = FastAPI(
    title="ANALISE B3 - 24/7",
    description="ANALISE TECNICA EM TEMPO REAL - MINI-INDICE E MINI-DOLAR DA B3",
    version="4.0.0",
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
    try:
        ativo = ativo.upper()
        if ativo not in ATIVOS:
            return JSONResponse({"erro": f"Ativo invalido. Use: {ATIVOS}"}, status_code=400)
        if timeframe not in TIMEFRAMES:
            return JSONResponse({"erro": f"Timeframe invalido. Use: {TIMEFRAMES}"}, status_code=400)

        analise = app_state["analises"].get(ativo, {}).get(timeframe)
        if not analise:
            try:
                logger.info(f"Analise on-the-fly para {ativo}/{timeframe}...")
                dados = await app_state["provider"].obter_dados(ativo, timeframe)
                if dados is not None and len(dados) >= 30:
                    analise = analisar_completo(dados, timeframe, ativo)
                    analise = converter_numpy(analise)
                else:
                    analise = {"erro": "Dados insuficientes"}
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Erro analise on-the-fly: {tb}")
                return JSONResponse({"erro": str(e), "traceback": tb}, status_code=500)

        response_data = {
            "ultima_atualizacao": app_state["ultima_atualizacao"],
            "analise": analise,
            "usando_cache": app_state["usando_cache"],
            "cache_data_pregao": app_state["cache_data_pregao"],
            "mercado_aberto": mercado_aberto(),
            "preco_realtime": app_state.get("preco_realtime", {}).get(ativo, {}),
        }
        try:
            json.dumps(response_data, default=str)
        except Exception:
            response_data["analise"] = json.loads(json.dumps(converter_numpy(analise), default=str))
        return JSONResponse(response_data)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Erro geral /api/analise: {tb}")
        return JSONResponse({"erro": str(e), "traceback": tb}, status_code=500)




@app.get("/api/book")
async def get_book(ativo: str = Query("WIN")):
    """Simula Book de Ofertas (DOM) baseado em dados reais de preço e volume"""
    try:
        provider = app_state["provider"]
        preco_rt = provider.preco_realtime.get(ativo, {})
        preco = preco_rt.get("preco", 0)
        
        if not preco:
            dados = await provider.obter_dados(ativo, "5m")
            if dados is not None and len(dados) > 0:
                preco = float(dados['close'].iloc[-1])
        
        if not preco:
            return JSONResponse({"erro": "Preço não disponível"})
        
        contrato = provider.get_contrato_info(ativo)
        tick = contrato.get("tick", 5 if ativo == "WIN" else 0.5)
        
        # Gerar book simulado realista baseado no preço real
        import random
        random.seed(int(preco * 100) % 10000)  # Seed consistente por preço
        
        niveis_compra = []
        niveis_venda = []
        
        for i in range(10):
            # Compradores (abaixo do preço)
            nivel_preco = preco - (i + 1) * tick
            # Volume maior perto do preço, menor longe
            base_vol = max(50, int(800 * (1 - i * 0.08) + random.randint(-100, 200)))
            qtd_ordens = max(1, int(15 * (1 - i * 0.06) + random.randint(-3, 5)))
            niveis_compra.append({
                "preco": round(nivel_preco, 2),
                "quantidade": base_vol,
                "ordens": qtd_ordens,
            })
            
            # Vendedores (acima do preço)
            nivel_preco = preco + (i + 1) * tick
            base_vol = max(50, int(750 * (1 - i * 0.08) + random.randint(-100, 200)))
            qtd_ordens = max(1, int(14 * (1 - i * 0.06) + random.randint(-3, 5)))
            niveis_venda.append({
                "preco": round(nivel_preco, 2),
                "quantidade": base_vol,
                "ordens": qtd_ordens,
            })
        
        total_compra = sum(n["quantidade"] for n in niveis_compra)
        total_venda = sum(n["quantidade"] for n in niveis_venda)
        ratio = total_compra / total_venda if total_venda > 0 else 1
        
        if ratio > 1.15:
            pressao = "COMPRADORES"
        elif ratio < 0.85:
            pressao = "VENDEDORES"
        else:
            pressao = "EQUILIBRIO"
        
        return JSONResponse(converter_numpy({
            "ativo": ativo,
            "preco_referencia": round(preco, 2),
            "tick": tick,
            "compra": niveis_compra,
            "venda": niveis_venda,
            "total_compra": total_compra,
            "total_venda": total_venda,
            "ratio": round(ratio, 2),
            "pressao": pressao,
            "timestamp": datetime.now(BRT).strftime("%H:%M:%S"),
        }))
    except Exception as e:
        return JSONResponse({"erro": str(e)})


@app.get("/api/tape")
async def get_tape(ativo: str = Query("WIN")):
    """Simula Times & Trades (Tape Reading) baseado em dados reais"""
    try:
        provider = app_state["provider"]
        preco_rt = provider.preco_realtime.get(ativo, {})
        preco = preco_rt.get("preco", 0)
        
        if not preco:
            dados = await provider.obter_dados(ativo, "5m")
            if dados is not None and len(dados) > 0:
                preco = float(dados['close'].iloc[-1])
        
        if not preco:
            return JSONResponse({"erro": "Preço não disponível"})
        
        contrato = provider.get_contrato_info(ativo)
        tick = contrato.get("tick", 5 if ativo == "WIN" else 0.5)
        
        import random
        agora = datetime.now(BRT)
        
        trades = []
        preco_ref = preco
        total_compra = 0
        total_venda = 0
        
        for i in range(30):
            segundos_atras = i * random.randint(2, 15)
            ts = agora - timedelta(seconds=segundos_atras)
            
            # Tipo: agressão compradora ou vendedora
            tipo = random.choice(["COMPRA", "VENDA"])
            
            # Preço varia em torno do preço atual
            variacao = random.randint(-3, 3) * tick
            trade_preco = round(preco_ref + variacao, 2)
            
            # Volume (lotes) - distribuição lognormal
            qtd = max(1, int(random.lognormvariate(2, 1.2)))
            is_grande = qtd >= 20
            
            if tipo == "COMPRA":
                total_compra += qtd
            else:
                total_venda += qtd
            
            trades.append({
                "hora": ts.strftime("%H:%M:%S"),
                "preco": trade_preco,
                "quantidade": qtd,
                "tipo": tipo,
                "grande": is_grande,
            })
        
        total = total_compra + total_venda
        pct_compra = round(total_compra / total * 100, 1) if total > 0 else 50
        
        if pct_compra > 60:
            agressao = "COMPRADORA"
        elif pct_compra < 40:
            agressao = "VENDEDORA"
        else:
            agressao = "EQUILIBRIO"
        
        return JSONResponse(converter_numpy({
            "ativo": ativo,
            "trades": trades,
            "total_compra": total_compra,
            "total_venda": total_venda,
            "pct_compra": pct_compra,
            "pct_venda": round(100 - pct_compra, 1),
            "agressao": agressao,
            "lotes_grandes": sum(1 for t in trades if t["grande"]),
            "timestamp": agora.strftime("%H:%M:%S"),
        }))
    except Exception as e:
        return JSONResponse({"erro": str(e)})

@app.get("/api/debug")
async def debug_analise():
    resultado = {"steps": [], "errors": []}
    try:
        resultado["steps"].append("1. Provider: " + str(app_state["provider"] is not None))
        dados = None
        try:
            dados = await app_state["provider"].obter_dados("WIN", "5m")
            if dados is not None:
                resultado["steps"].append(f"2. Dados: {len(dados)} candles")
                resultado["steps"].append(f"   Close dtype: {dados['close'].dtype}, last={dados['close'].iloc[-1]}")
            else:
                resultado["steps"].append("2. Dados: None")
        except Exception:
            resultado["errors"].append(f"Step 2: {traceback.format_exc()}")
        if dados is not None and len(dados) >= 30:
            try:
                analise = analisar_completo(dados, "5m", "WIN")
                resultado["steps"].append(f"3. Analise OK: preco={analise.get('preco_atual')}")
                converted = converter_numpy(analise)
                resultado["steps"].append("4. converter_numpy OK")
                json_str = json.dumps(converted, default=str)
                resultado["steps"].append(f"5. JSON OK ({len(json_str)} bytes)")
            except Exception:
                resultado["errors"].append(f"Step 3-5: {traceback.format_exc()}")
        cached = app_state["analises"]
        resultado["steps"].append(f"6. Cache: {list(cached.keys())}")
    except Exception:
        resultado["errors"].append(f"General: {traceback.format_exc()}")
    return JSONResponse(resultado)


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
        "versao": "4.0.0",
        "fontes_dados": ["TradingView Scanner (futuro real WIN1!/WDO1!)",
            "yfinance (candles OHLCV)",
            "Book de Ofertas (simulado)",
            "Tape Reading (simulado)", "HG Brasil + Basis (fallback)"],
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




@app.get("/api/replay")
async def get_replay(ativo: str = Query("WIN"), contratos: int = Query(1)):
    """
    Replay automatico do pregao anterior com analise multi-timeframe.
    Cruza 5m (execucao) + 15m (confirmacao) + 1h (tendencia macro).
    """
    from analysis_engine import (
        calcular_rsi, calcular_macd, calcular_vwap, calcular_atr_series,
        detectar_lateralizacao, detectar_pullback, calcular_fibonacci,
        analisar_volume, detectar_violinada, gerar_sinais
    )
    import traceback

    ativo_upper = ativo.upper()
    valor_ponto = 0.20 if ativo_upper == "WIN" else 10.00

    try:
        # Pegar velas em 3 timeframes
        provider = app_state["provider"]
        dados_5m = await provider.obter_dados(ativo_upper, "5m")
        dados_15m = await provider.obter_dados(ativo_upper, "15m")
        dados_1h = await provider.obter_dados(ativo_upper, "1h")

        if dados_5m is None or len(dados_5m) < 50:
            return JSONResponse({"erro": "Dados insuficientes para replay (5m)"})

        brt = timezone(timedelta(hours=-3))

        def to_brt(idx):
            if hasattr(idx, 'tz') and idx.tz is not None:
                return idx.tz_convert(brt)
            try:
                return idx.tz_localize('UTC').tz_convert(brt)
            except:
                return idx

        idx_brt = to_brt(dados_5m.index)
        datas_unicas = sorted(set(idx_brt.date))
        if len(datas_unicas) < 2:
            return JSONResponse({"erro": "Precisa de pelo menos 2 dias de dados"})

        dia_anterior = datas_unicas[-2]
        dia_str = dia_anterior.strftime("%d/%m/%Y")

        # Helper: analisa um timeframe numa janela
        def analisar_tf(window):
            if len(window) < 20:
                return None
            try:
                rsi_s = calcular_rsi(window)
                rsi_v = float(rsi_s.iloc[-1])
                ml, ms, mh = calcular_macd(window)
                mv = float(ml.iloc[-1])
                msv = float(ms.iloc[-1])
                mhv = float(mh.iloc[-1])
                fib = calcular_fibonacci(window)
                vol = analisar_volume(window)
                viol = detectar_violinada(window)
                vwap_s = calcular_vwap(window)
                vwap_v = float(vwap_s.iloc[-1])
                lat = detectar_lateralizacao(window)
                preco = float(window['close'].iloc[-1])
                ema9 = float(window['close'].ewm(span=9, adjust=False).mean().iloc[-1])
                ema21 = float(window['close'].ewm(span=21, adjust=False).mean().iloc[-1])
                # Detect trend: if VWAP is NaN (no volume data), use only EMAs
                import math
                vwap_valid = not (math.isnan(vwap_v) if isinstance(vwap_v, float) else False)
                if vwap_valid:
                    if preco > vwap_v and ema9 > ema21:
                        tend = "ALTA"
                    elif preco < vwap_v and ema9 < ema21:
                        tend = "BAIXA"
                    else:
                        tend = "LATERAL"
                else:
                    # No VWAP: use EMA crossover + price position
                    if ema9 > ema21 and preco > ema9:
                        tend = "ALTA"
                    elif ema9 < ema21 and preco < ema9:
                        tend = "BAIXA"
                    elif ema9 > ema21:
                        tend = "ALTA"  # EMA bullish even if price pulled back
                    elif ema9 < ema21:
                        tend = "BAIXA"  # EMA bearish even if price bounced
                    else:
                        tend = "LATERAL"
                tend_raw = tend  # Save raw trend for fallback
                if lat.get("lateral"):
                    tend = "LATERAL"
                pb = detectar_pullback(window, tend)
                sinais = gerar_sinais(window, fib, rsi_v, mv, msv, mhv,
                    vol, viol, tendencia=tend, pullback_info=pb,
                    lateralizacao=lat, vwap_atual=vwap_v)
                return {
                    "tendencia": tend,
                    "tendencia_raw": tend_raw,
                    "rsi": round(rsi_v, 1),
                    "macd_hist": round(mhv, 2),
                    "volume_pressao": vol.pressao,
                    "sinais": sinais,
                    "preco": preco,
                }
            except:
                return None

        # Helper: get window of data up to a timestamp for a given timeframe
        def get_window(dados_tf, ts_limit, max_bars=100):
            if dados_tf is None or len(dados_tf) == 0:
                return None
            tf_brt = to_brt(dados_tf.index)
            mask = tf_brt <= ts_limit
            window = dados_tf[mask].tail(max_bars)
            return window if len(window) >= 20 else None

        # ========================================
        # REPLAY: vela a vela no 5m
        # ========================================
        operacoes = []
        posicao_aberta = None
        sinais_gerados = []
        velas_info = []
        dia_indices = [i for i, d in enumerate(idx_brt.date) if d == dia_anterior]

        for pos_idx in dia_indices:
            ts = idx_brt[pos_idx]
            hora_str = ts.strftime("%H:%M")
            vela = dados_5m.iloc[pos_idx]
            preco_close = float(vela['close'])
            preco_high = float(vela['high'])
            preco_low = float(vela['low'])

            vela_data = {
                "hora": hora_str,
                "open": round(float(vela['open']), 2),
                "high": round(preco_high, 2),
                "low": round(preco_low, 2),
                "close": round(preco_close, 2),
                "volume": int(vela.get('volume', 0)),
            }

            # === Check stop/alvo da posicao aberta ===
            if posicao_aberta:
                op = posicao_aberta
                hit = None
                if op["tipo"] == "COMPRA":
                    if preco_low <= op["stop_loss"]:
                        hit = ("LOSS", op["stop_loss"], op["stop_loss"] - op["entrada"])
                    elif preco_high >= op["alvo"]:
                        hit = ("WIN", op["alvo"], op["alvo"] - op["entrada"])
                else:
                    if preco_high >= op["stop_loss"]:
                        hit = ("LOSS", op["stop_loss"], op["entrada"] - op["stop_loss"])
                    elif preco_low <= op["alvo"]:
                        hit = ("WIN", op["alvo"], op["entrada"] - op["alvo"])
                if hit:
                    status, saida, pts = hit
                    op["saida"] = saida
                    op["hora_saida"] = hora_str
                    op["pts"] = round(pts, 1)
                    op["resultado"] = round(pts * valor_ponto * contratos, 2)
                    op["status"] = status
                    operacoes.append(op)
                    posicao_aberta = None

            # === Multi-timeframe analysis ===
            sinal_aqui = None
            if not posicao_aberta:
                # Window for 5m
                w5 = dados_5m.iloc[max(0, pos_idx - 100):pos_idx + 1]
                a5 = analisar_tf(w5) if len(w5) >= 30 else None

                if a5 and (a5["sinais"] or a5.get("tendencia_raw", a5.get("tendencia")) in ("ALTA", "BAIXA")):
                    # Confirm with 15m and 1h
                    w15 = get_window(dados_15m, ts)
                    w1h = get_window(dados_1h, ts)
                    a15 = analisar_tf(w15) if w15 is not None else None
                    a1h = analisar_tf(w1h) if w1h is not None else None

                    # Use sinal formal if available, otherwise use multiple strategies
                    if a5["sinais"]:
                        s5 = a5["sinais"][0]
                        tipo_5m = s5.tipo
                    else:
                        from types import SimpleNamespace
                        rsi5 = a5.get("rsi", 50)
                        macd5 = a5.get("macd_hist", 0)
                        tend5 = a5.get("tendencia_raw", a5.get("tendencia"))
                        p = a5["preco"]
                        atr_s = calcular_atr_series(w5)
                        atr_val = float(atr_s.iloc[-1]) if len(atr_s) > 0 else 100
                        ema9_v = float(w5['close'].ewm(span=9, adjust=False).mean().iloc[-1])
                        ema21_v = float(w5['close'].ewm(span=21, adjust=False).mean().iloc[-1])
                        prev_close = float(w5['close'].iloc[-2]) if len(w5) >= 2 else p
                        prev_low = float(w5['low'].iloc[-2]) if len(w5) >= 2 else p
                        prev_high = float(w5['high'].iloc[-2]) if len(w5) >= 2 else p
                        curr_high = float(w5['high'].iloc[-1])
                        curr_low = float(w5['low'].iloc[-1])

                        tipo_5m = None
                        motivo_entrada = []
                        confianca_v = 60

                        # --- ESTRATEGIA 1: Tendencia + RSI + MACD (relaxado) ---
                        if tend5 == "ALTA" and rsi5 < 70 and macd5 > 0:
                            tipo_5m = "COMPRA"
                            motivo_entrada = [f"Tendencia ALTA + RSI {rsi5:.0f} + MACD hist {macd5:.1f}"]
                            confianca_v = 65
                        elif tend5 == "BAIXA" and rsi5 > 30 and macd5 < 0:
                            tipo_5m = "VENDA"
                            motivo_entrada = [f"Tendencia BAIXA + RSI {rsi5:.0f} + MACD hist {macd5:.1f}"]
                            confianca_v = 65

                        # --- ESTRATEGIA 2: Pullback em EMA9 ---
                        if not tipo_5m and tend5 == "ALTA":
                            if curr_low <= ema9_v * 1.001 and p > ema9_v and p > prev_close:
                                tipo_5m = "COMPRA"
                                motivo_entrada = [f"Pullback EMA9 ({ema9_v:.0f}) + reversao alta"]
                                confianca_v = 70
                        if not tipo_5m and tend5 == "BAIXA":
                            if curr_high >= ema9_v * 0.999 and p < ema9_v and p < prev_close:
                                tipo_5m = "VENDA"
                                motivo_entrada = [f"Pullback EMA9 ({ema9_v:.0f}) + reversao baixa"]
                                confianca_v = 70

                        # --- ESTRATEGIA 3: Rompimento de high/low anterior ---
                        if not tipo_5m and len(w5) >= 3:
                            last3_high = float(w5['high'].iloc[-3:-1].max())
                            last3_low = float(w5['low'].iloc[-3:-1].min())
                            if p > last3_high and ema9_v > ema21_v:
                                tipo_5m = "COMPRA"
                                motivo_entrada = [f"Rompimento alta {last3_high:.0f} + EMA9>EMA21"]
                                confianca_v = 60
                            elif p < last3_low and ema9_v < ema21_v:
                                tipo_5m = "VENDA"
                                motivo_entrada = [f"Rompimento baixa {last3_low:.0f} + EMA9<EMA21"]
                                confianca_v = 60

                        # --- ESTRATEGIA 4: RSI extremo (reversao) ---
                        if not tipo_5m:
                            if rsi5 < 25 and macd5 > float(calcular_macd(w5)[2].iloc[-2] if len(w5) > 2 else 0):
                                tipo_5m = "COMPRA"
                                motivo_entrada = [f"RSI sobrevendido ({rsi5:.0f}) + MACD virando"]
                                confianca_v = 55
                            elif rsi5 > 75 and macd5 < float(calcular_macd(w5)[2].iloc[-2] if len(w5) > 2 else 0):
                                tipo_5m = "VENDA"
                                motivo_entrada = [f"RSI sobrecomprado ({rsi5:.0f}) + MACD virando"]
                                confianca_v = 55

                        if tipo_5m:
                            if tipo_5m == "COMPRA":
                                stop_v = round(p - atr_val * 1.2, 2)
                                alvo_v = round(p + atr_val * 1.8, 2)
                            else:
                                stop_v = round(p + atr_val * 1.2, 2)
                                alvo_v = round(p - atr_val * 1.8, 2)
                            motivo_entrada.append(f"ATR: {atr_val:.0f} pts | Stop: {abs(p-stop_v):.0f} | Alvo: {abs(p-alvo_v):.0f} | RR: 1:{abs(p-alvo_v)/abs(p-stop_v):.1f}")
                            s5 = SimpleNamespace(
                                tipo=tipo_5m, preco_entrada=p,
                                stop_loss=stop_v, take_profit_1=alvo_v,
                                confianca=confianca_v, motivos=motivo_entrada
                            )
                        else:
                            s5 = None
                    
                    if not s5:
                        vela_data["sinal"] = None
                        vela_data["posicao_aberta"] = False
                        velas_info.append(vela_data)
                        continue
                    
                    # Cooldown: espera pelo menos 3 velas apos fechar uma operacao
                    if operacoes and len(operacoes) > 0:
                        last_op = operacoes[-1]
                        last_exit_hora = last_op.get("hora_saida", "")
                        if last_exit_hora:
                            exit_idx = next((i for i, v in enumerate(velas_info) if v["hora"] == last_exit_hora), -1)
                            if exit_idx >= 0 and (len(velas_info) - exit_idx) < 3:
                                vela_data["sinal"] = None
                                vela_data["posicao_aberta"] = False
                                velas_info.append(vela_data)
                                continue
                    
                    tipo_5m = s5.tipo

                    # Confluencia: 15m e 1h devem ter mesma tendencia
                    conf_15m = False
                    conf_1h = False
                    motivo_conf = []

                    if a15:
                        t15 = a15.get("tendencia_raw", a15["tendencia"])
                        if tipo_5m == "COMPRA" and t15 in ("ALTA",):
                            conf_15m = True
                            motivo_conf.append(f"15m: Tendencia {t15} | RSI {a15['rsi']}")
                        elif tipo_5m == "VENDA" and t15 in ("BAIXA",):
                            conf_15m = True
                            motivo_conf.append(f"15m: Tendencia {t15} | RSI {a15['rsi']}")
                        elif t15 == "LATERAL":
                            conf_15m = True  # Lateral nao contradiz
                            motivo_conf.append(f"15m: Lateral (nao contradiz)")
                    else:
                        conf_15m = True  # Sem dados = nao bloqueia
                        motivo_conf.append("15m: Sem dados suficientes")

                    if a1h:
                        t1h = a1h.get("tendencia_raw", a1h["tendencia"])
                        if tipo_5m == "COMPRA" and t1h in ("ALTA", "LATERAL"):
                            conf_1h = True
                            motivo_conf.append(f"1h: Tendencia {t1h} | RSI {a1h['rsi']}")
                        elif tipo_5m == "VENDA" and t1h in ("BAIXA", "LATERAL"):
                            conf_1h = True
                            motivo_conf.append(f"1h: Tendencia {t1h} | RSI {a1h['rsi']}")
                    else:
                        conf_1h = True
                        motivo_conf.append("1h: Sem dados suficientes")

                    # Confluencia ponderada: 5m obrigatorio, 15m e 1h sao bonus
                    # Entra se: ambos confirmam, OU 1 confirma e confianca >= 60, OU 5m forte (confianca >= 70)
                    conf_score = (1 if conf_15m else 0) + (1 if conf_1h else 0)
                    entra = conf_score == 2 or (conf_score >= 1 and s5.confianca >= 60) or s5.confianca >= 70
                    if entra:
                        sinal_aqui = {
                            "hora": hora_str,
                            "tipo": tipo_5m,
                            "preco": s5.preco_entrada,
                            "stop": s5.stop_loss,
                            "alvo": s5.take_profit_1,
                            "confianca": s5.confianca,
                            "rsi": a5["rsi"],
                            "tendencia_5m": a5["tendencia"],
                            "tendencia_15m": a15["tendencia"] if a15 else "N/A",
                            "tendencia_1h": a1h["tendencia"] if a1h else "N/A",
                            "confluencia": motivo_conf,
                            "motivos": s5.motivos[:3],
                        }
                        sinais_gerados.append(sinal_aqui)

                        posicao_aberta = {
                            "tipo": tipo_5m,
                            "entrada": s5.preco_entrada,
                            "stop_loss": s5.stop_loss,
                            "alvo": s5.take_profit_1,
                            "hora_entrada": hora_str,
                            "confianca": s5.confianca,
                            "motivos": s5.motivos[:3] + motivo_conf,
                        }

            vela_data["sinal"] = sinal_aqui
            vela_data["posicao_aberta"] = bool(posicao_aberta)
            velas_info.append(vela_data)

        # Fechar posicao aberta no fim do dia
        if posicao_aberta and velas_info:
            op = posicao_aberta
            preco_fech = velas_info[-1]["close"]
            pts = (preco_fech - op["entrada"]) if op["tipo"] == "COMPRA" else (op["entrada"] - preco_fech)
            op["saida"] = preco_fech
            op["hora_saida"] = velas_info[-1]["hora"]
            op["pts"] = round(pts, 1)
            op["resultado"] = round(pts * valor_ponto * contratos, 2)
            op["status"] = "WIN" if pts > 0 else "LOSS"
            op["fechamento_forcado"] = True
            operacoes.append(op)

        # ========================================
        # RELATORIO
        # ========================================
        total_ops = len(operacoes)
        wins = [op for op in operacoes if op["status"] == "WIN"]
        losses = [op for op in operacoes if op["status"] == "LOSS"]
        resultado_bruto = sum(op["resultado"] for op in operacoes)
        resultado_pts = sum(op["pts"] for op in operacoes)

        equity = []
        running = 0
        peak = 0
        max_dd = 0
        for op in operacoes:
            running += op["resultado"]
            equity.append(round(running, 2))
            peak = max(peak, running)
            max_dd = max(max_dd, peak - running)

        max_seq_w = max_seq_l = seq_w = seq_l = 0
        for op in operacoes:
            if op["status"] == "WIN":
                seq_w += 1; seq_l = 0; max_seq_w = max(max_seq_w, seq_w)
            else:
                seq_l += 1; seq_w = 0; max_seq_l = max(max_seq_l, seq_l)

        taxa = round(len(wins) / total_ops * 100, 1) if total_ops > 0 else 0
        media_w = round(sum(o["resultado"] for o in wins) / len(wins), 2) if wins else 0
        media_l = round(sum(o["resultado"] for o in losses) / len(losses), 2) if losses else 0
        sum_losses = abs(sum(o["resultado"] for o in losses))
        fl = round(sum(o["resultado"] for o in wins) / sum_losses, 2) if sum_losses > 0 else 999

        dados_dia = dados_5m.iloc[[i for i, d in enumerate(idx_brt.date) if d == dia_anterior]]
        abertura = round(float(dados_dia.iloc[0]['open']), 2)
        fechamento = round(float(dados_dia.iloc[-1]['close']), 2)
        high_dia = round(float(dados_dia['high'].max()), 2)
        low_dia = round(float(dados_dia['low'].min()), 2)
        amp = round(high_dia - low_dia, 0)
        var_dia = round(fechamento - abertura, 2)
        var_pct = round(var_dia / abertura * 100, 2) if abertura > 0 else 0

        return JSONResponse({
            "dia": dia_str,
            "ativo": ativo_upper,
            "contrato": provider.get_contrato_info(ativo_upper).get("ticker_b3", ativo_upper),
            "contratos": contratos,
            "valor_ponto": valor_ponto,
            "mercado": {
                "abertura": abertura, "fechamento": fechamento,
                "high": high_dia, "low": low_dia,
                "amplitude_pts": amp, "variacao": var_dia, "variacao_pct": var_pct,
                "total_velas": len(dados_dia),
            },
            "resumo": {
                "total_operacoes": total_ops,
                "wins": len(wins), "losses": len(losses),
                "taxa_acerto": taxa,
                "resultado_bruto": round(resultado_bruto, 2),
                "resultado_pts": round(resultado_pts, 1),
                "maior_win": round(max((o["resultado"] for o in wins), default=0), 2),
                "maior_loss": round(min((o["resultado"] for o in losses), default=0), 2),
                "maior_win_pts": round(max((o["pts"] for o in wins), default=0), 1),
                "maior_loss_pts": round(min((o["pts"] for o in losses), default=0), 1),
                "media_win": media_w, "media_loss": media_l,
                "fator_lucro": fl,
                "max_drawdown": round(max_dd, 2),
                "max_sequencia_win": max_seq_w,
                "max_sequencia_loss": max_seq_l,
                "equity_curve": equity,
            },
            "operacoes": operacoes,
            "sinais_gerados": sinais_gerados,
            "total_sinais": len(sinais_gerados),
            "velas": velas_info,
        })

    except Exception as e:
        logger.error(f"Erro replay: {e}\n{traceback.format_exc()}")
        return JSONResponse({"erro": f"Erro no replay: {str(e)}"}, status_code=500)




@app.get("/api/replay-velas")
async def replay_velas(ativo: str = "WIN"):
    """Retorna todas as velas do dia anterior com indicadores para simulacao manual"""
    try:
        provider = app_state["provider"]
        ticker = "^BVSP" if ativo == "WIN" else "USDBRL=X"
        from data_provider import obter_contrato_vigente as _ocv
        contrato_info = _ocv(ativo)
        contrato_nome = contrato_info.get('ticker_b3', ativo)
        valor_ponto = 0.20 if ativo == "WIN" else 10.00

        dados = yf.download(ticker, period="5d", interval="5m", progress=False)
        if dados.empty:
            return JSONResponse({"erro": "Sem dados do yfinance"})
        
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
        dados.columns = [c.lower() for c in dados.columns]
        
        from datetime import timezone, timedelta
        BRT = timezone(timedelta(hours=-3))
        dados.index = dados.index.tz_convert(BRT)
        
        hoje = datetime.now(BRT).date()
        dates = sorted(set(dados.index.date))
        dia_anterior = None
        for d in reversed(dates):
            if d < hoje:
                dia_anterior = d
                break
        if not dia_anterior:
            return JSONResponse({"erro": "Dia anterior nao encontrado"})
        
        # Get day data
        day_mask = dados.index.date == dia_anterior
        day_data = dados[day_mask]
        
        # Calculate indicators for each candle
        from analysis_engine import calcular_rsi, calcular_macd, calcular_atr_series
        import math
        
        velas = []
        all_indices = list(range(len(dados)))
        day_indices = [i for i, d in enumerate(dados.index.date) if d == dia_anterior]
        
        for pos_idx in day_indices:
            w = dados.iloc[max(0, pos_idx - 100):pos_idx + 1]
            vela = dados.iloc[pos_idx]
            ts = dados.index[pos_idx]
            
            rsi_v = None
            macd_v = None
            ema9_v = None
            ema21_v = None
            atr_v = None
            tend = "N/A"
            
            if len(w) >= 20:
                try:
                    rsi_s = calcular_rsi(w)
                    rsi_v = round(float(rsi_s.iloc[-1]), 1)
                    ml, ms, mh = calcular_macd(w)
                    macd_v = round(float(mh.iloc[-1]), 1)
                    ema9_v = round(float(w['close'].ewm(span=9, adjust=False).mean().iloc[-1]), 0)
                    ema21_v = round(float(w['close'].ewm(span=21, adjust=False).mean().iloc[-1]), 0)
                    atr_s = calcular_atr_series(w)
                    atr_v = round(float(atr_s.iloc[-1]), 0) if len(atr_s) > 0 else None
                    
                    p = float(vela['close'])
                    if ema9_v > ema21_v and p > ema9_v:
                        tend = "ALTA"
                    elif ema9_v < ema21_v and p < ema9_v:
                        tend = "BAIXA"
                    elif ema9_v > ema21_v:
                        tend = "ALTA"
                    elif ema9_v < ema21_v:
                        tend = "BAIXA"
                    else:
                        tend = "LATERAL"
                except:
                    pass
            
            velas.append({
                "idx": len(velas),
                "hora": ts.strftime("%H:%M"),
                "open": round(float(vela['open']), 2),
                "high": round(float(vela['high']), 2),
                "low": round(float(vela['low']), 2),
                "close": round(float(vela['close']), 2),
                "volume": int(vela.get('volume', 0)),
                "rsi": rsi_v,
                "macd_hist": macd_v,
                "ema9": ema9_v,
                "ema21": ema21_v,
                "atr": atr_v,
                "tendencia": tend,
                "variacao": round(float(vela['close']) - float(vela['open']), 2),
            })
        
        mercado = {
            "abertura": velas[0]["open"] if velas else 0,
            "fechamento": velas[-1]["close"] if velas else 0,
            "high": max(v["high"] for v in velas) if velas else 0,
            "low": min(v["low"] for v in velas) if velas else 0,
        }
        mercado["amplitude_pts"] = round(mercado["high"] - mercado["low"], 0)
        mercado["variacao_pct"] = round((mercado["fechamento"] / mercado["abertura"] - 1) * 100, 2)
        
        return JSONResponse({
            "dia": dia_anterior.strftime("%d/%m/%Y"),
            "ativo": ativo,
            "contrato": contrato_nome,
            "valor_ponto": valor_ponto,
            "mercado": mercado,
            "total_velas": len(velas),
            "velas": velas,
        })
    except Exception as e:
        logger.error(f"Erro replay-velas: {e}")
        return JSONResponse({"erro": str(e)}, status_code=500)


@app.post("/api/simular-entrada")
async def simular_entrada(request: Request):
    """Simula uma entrada manual do usuario: recebe vela, tipo, stop, alvo e calcula resultado"""
    try:
        body = await request.json()
        ativo = body.get("ativo", "WIN")
        vela_idx = body.get("vela_idx")  # indice da vela no dia
        tipo = body.get("tipo")  # COMPRA ou VENDA
        stop_pts = body.get("stop_pts", 200)  # stop em pontos
        alvo_pts = body.get("alvo_pts", 300)  # alvo em pontos
        contratos = body.get("contratos", 1)
        
        if vela_idx is None or tipo not in ("COMPRA", "VENDA"):
            return JSONResponse({"erro": "Parametros invalidos: vela_idx e tipo (COMPRA/VENDA) obrigatorios"})
        
        valor_ponto = 0.20 if ativo == "WIN" else 10.00
        ticker = "^BVSP" if ativo == "WIN" else "USDBRL=X"
        
        dados = yf.download(ticker, period="5d", interval="5m", progress=False)
        if dados.empty:
            return JSONResponse({"erro": "Sem dados"})
        
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
        dados.columns = [c.lower() for c in dados.columns]
        
        from datetime import timezone, timedelta
        BRT = timezone(timedelta(hours=-3))
        dados.index = dados.index.tz_convert(BRT)
        
        hoje = datetime.now(BRT).date()
        dates = sorted(set(dados.index.date))
        dia_anterior = None
        for d in reversed(dates):
            if d < hoje:
                dia_anterior = d
                break
        
        day_indices = [i for i, d in enumerate(dados.index.date) if d == dia_anterior]
        
        if vela_idx < 0 or vela_idx >= len(day_indices):
            return JSONResponse({"erro": f"vela_idx invalido: {vela_idx}, max: {len(day_indices)-1}"})
        
        # Entry candle
        entry_global_idx = day_indices[vela_idx]
        entry_vela = dados.iloc[entry_global_idx]
        preco_entrada = float(entry_vela['close'])
        hora_entrada = dados.index[entry_global_idx].strftime("%H:%M")
        
        # Calculate stop and target
        if tipo == "COMPRA":
            stop_loss = round(preco_entrada - stop_pts, 2)
            take_profit = round(preco_entrada + alvo_pts, 2)
        else:
            stop_loss = round(preco_entrada + stop_pts, 2)
            take_profit = round(preco_entrada - alvo_pts, 2)
        
        # Simulate forward vela by vela
        resultado = None
        hora_saida = None
        preco_saida = None
        velas_na_operacao = 0
        max_favoravel = 0
        max_adverso = 0
        caminho = []
        
        for future_idx in day_indices[vela_idx + 1:]:
            vela_f = dados.iloc[future_idx]
            h = float(vela_f['high'])
            l = float(vela_f['low'])
            c = float(vela_f['close'])
            hora_f = dados.index[future_idx].strftime("%H:%M")
            velas_na_operacao += 1
            
            if tipo == "COMPRA":
                favoravel = h - preco_entrada
                adverso = preco_entrada - l
                if l <= stop_loss:
                    resultado = "LOSS"
                    preco_saida = stop_loss
                    hora_saida = hora_f
                    break
                elif h >= take_profit:
                    resultado = "WIN"
                    preco_saida = take_profit
                    hora_saida = hora_f
                    break
            else:
                favoravel = preco_entrada - l
                adverso = h - preco_entrada
                if h >= stop_loss:
                    resultado = "LOSS"
                    preco_saida = stop_loss
                    hora_saida = hora_f
                    break
                elif l <= take_profit:
                    resultado = "WIN"
                    preco_saida = take_profit
                    hora_saida = hora_f
                    break
            
            max_favoravel = max(max_favoravel, favoravel)
            max_adverso = max(max_adverso, adverso)
            caminho.append({"hora": hora_f, "close": round(c, 2), "favoravel": round(favoravel, 1), "adverso": round(adverso, 1)})
        
        # Se nao bateu stop nem alvo, fecha no ultimo preco
        if resultado is None:
            last_vela = dados.iloc[day_indices[-1]]
            preco_saida = float(last_vela['close'])
            hora_saida = dados.index[day_indices[-1]].strftime("%H:%M")
            if tipo == "COMPRA":
                pts = preco_saida - preco_entrada
            else:
                pts = preco_entrada - preco_saida
            resultado = "WIN" if pts > 0 else "LOSS"
            fechou_forcado = True
        else:
            fechou_forcado = False
        
        if tipo == "COMPRA":
            pts = round(preco_saida - preco_entrada, 1)
        else:
            pts = round(preco_entrada - preco_saida, 1)
        
        resultado_rs = round(pts * valor_ponto * contratos, 2)
        rr = round(alvo_pts / stop_pts, 1) if stop_pts > 0 else 0
        
        return JSONResponse({
            "entrada": {
                "tipo": tipo,
                "preco": preco_entrada,
                "hora": hora_entrada,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "stop_pts": stop_pts,
                "alvo_pts": alvo_pts,
                "rr": rr,
            },
            "saida": {
                "preco": preco_saida,
                "hora": hora_saida,
                "resultado": resultado,
                "pts": pts,
                "resultado_rs": resultado_rs,
                "velas_na_operacao": velas_na_operacao,
                "fechamento_forcado": fechou_forcado,
            },
            "detalhes": {
                "max_favoravel_pts": round(max_favoravel, 1),
                "max_adverso_pts": round(max_adverso, 1),
                "caminho": caminho[-10:] if len(caminho) > 10 else caminho,
            },
            "contratos": contratos,
            "valor_ponto": valor_ponto,
        })
    except Exception as e:
        logger.error(f"Erro simular-entrada: {e}")
        return JSONResponse({"erro": str(e)}, status_code=500)


@app.get("/api/preco-realtime")
async def get_preco_realtime():
    """Retorna preço em tempo real dos futuros B3 (Investing.com + HG Brasil fallback)"""
    try:
        precos = await app_state["provider"].obter_preco_realtime()
        # Detectar fonte principal dinamicamente
        fontes = set()
        for info in precos.values():
            if isinstance(info, dict) and "fonte" in info:
                fontes.add(info["fonte"])
        fonte_str = " + ".join(sorted(fontes)) if fontes else "Multi-source"
        return JSONResponse({
            "precos": precos,
            "fonte": fonte_str,
            "timestamp": datetime.now(BRT).strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception as e:
        return JSONResponse({"erro": str(e)}, status_code=500)


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




@app.get("/api/sinais-ia")
async def get_sinais_ia(ativo: str = Query("WIN")):
    """Sinais IA - Triple Screen (Elder) + Confluencia 7 pontos + Escala Confianca"""
    ativo = ativo.upper()
    try:
        analise_5m = app_state["analises"].get(ativo, {}).get("5m", {})
        analise_15m = app_state["analises"].get(ativo, {}).get("15m", {})
        analise_1h = app_state["analises"].get(ativo, {}).get("1h", {})
        analise_4h = app_state["analises"].get(ativo, {}).get("4h", {})
        analise_1d = app_state["analises"].get(ativo, {}).get("1d", {})
        
        # === HELPERS ===
        def get_val(analise, key, default=None):
            v = analise.get(key, default) if analise else default
            if isinstance(v, dict): return v.get("valor", v.get("status", default))
            return v
        
        def safe_float(v, default=0):
            try: return float(v) if v is not None else default
            except (ValueError, TypeError): return default
        
        preco = safe_float(analise_5m.get("preco_atual", 0))
        if not preco:
            rt = app_state.get("preco_realtime", {}).get(ativo, {})
            preco = safe_float(rt.get("preco", 0))
        
        spec = {"WIN": {"tick": 5, "vp": 0.20, "atr_mult": 1.5}, "WDO": {"tick": 0.5, "vp": 10.0, "atr_mult": 1.2}}
        s = spec.get(ativo, spec["WIN"])
        
        # === EXTRAIR INDICADORES DE TODOS OS TIMEFRAMES ===
        
        rsi_5m = safe_float(get_val(analise_5m, "rsi", 50), 50)
        rsi_15m = safe_float(get_val(analise_15m, "rsi", 50), 50)
        tendencia_5m = get_val(analise_5m, "tendencia", "LATERAL")
        tendencia_15m = get_val(analise_15m, "tendencia", "LATERAL")
        tendencia_1h = get_val(analise_1h, "tendencia", "LATERAL")
        tendencia_4h = get_val(analise_4h, "tendencia", "LATERAL")
        tendencia_1d = get_val(analise_1d, "tendencia", "LATERAL")
        
        macd_5m = get_val(analise_5m, "macd", "NEUTRO")
        if isinstance(macd_5m, dict): macd_5m = macd_5m.get("status", "NEUTRO")
        macd_15m = get_val(analise_15m, "macd", "NEUTRO")
        if isinstance(macd_15m, dict): macd_15m = macd_15m.get("status", "NEUTRO")
        
        ema9 = safe_float(get_val(analise_5m, "ema9", 0))
        ema21 = safe_float(get_val(analise_5m, "ema21", 0))
        ema50 = safe_float(get_val(analise_5m, "ema50", 0))
        ema200 = safe_float(get_val(analise_5m, "ema200", 0))
        vwap = safe_float(get_val(analise_5m, "vwap", 0))
        
        atr = safe_float(get_val(analise_5m, "atr", 150 if ativo == "WIN" else 15), 150 if ativo == "WIN" else 15)
        
        # Bollinger
        bb = analise_5m.get("bollinger", {}) if analise_5m else {}
        bb_upper = safe_float(bb.get("upper", 0)) if isinstance(bb, dict) else 0
        bb_lower = safe_float(bb.get("lower", 0)) if isinstance(bb, dict) else 0
        
        # ADX
        adx_val = get_val(analise_5m, "adx", 0)
        if isinstance(adx_val, dict): adx_val = adx_val.get("adx", 0)
        adx_val = safe_float(adx_val)
        
        # === TRIPLE SCREEN (Elder) ===
        # Tela 1 (longo): 1h/4h/1d - TENDENCIA
        tela1_score = 0
        tela1_detail = []
        for tf_name, tf_tend in [("1h", tendencia_1h), ("4h", tendencia_4h), ("1d", tendencia_1d)]:
            if tf_tend == "ALTA": tela1_score += 1; tela1_detail.append(f"{tf_name} ALTA")
            elif tf_tend == "BAIXA": tela1_score -= 1; tela1_detail.append(f"{tf_name} BAIXA")
            else: tela1_detail.append(f"{tf_name} LATERAL")
        tela1_dir = "ALTA" if tela1_score > 0 else "BAIXA" if tela1_score < 0 else "LATERAL"
        
        # Tela 2 (medio): 15m - SINAL contra tendencia (pullback)
        tela2_sinal = "NEUTRO"
        tela2_detail = []
        if rsi_15m < 35 and tela1_dir == "ALTA":
            tela2_sinal = "COMPRA"; tela2_detail.append(f"RSI 15m sobrevendido ({rsi_15m:.0f}) em tendencia de alta")
        elif rsi_15m > 65 and tela1_dir == "BAIXA":
            tela2_sinal = "VENDA"; tela2_detail.append(f"RSI 15m sobrecomprado ({rsi_15m:.0f}) em tendencia de baixa")
        elif tendencia_15m == tela1_dir:
            tela2_sinal = "COMPRA" if tela1_dir == "ALTA" else "VENDA" if tela1_dir == "BAIXA" else "NEUTRO"
            tela2_detail.append(f"15m confirmando {tela1_dir}")
        
        # Tela 3 (curto): 5m - ENTRADA precisa
        tela3_entrada = "NEUTRO"
        tela3_detail = []
        if ema9 > 0 and ema21 > 0:
            if ema9 > ema21 and tela2_sinal == "COMPRA":
                tela3_entrada = "COMPRA"; tela3_detail.append("EMA9 > EMA21 confirmando compra")
            elif ema9 < ema21 and tela2_sinal == "VENDA":
                tela3_entrada = "VENDA"; tela3_detail.append("EMA9 < EMA21 confirmando venda")
        if rsi_5m < 30 and tela2_sinal == "COMPRA":
            tela3_entrada = "COMPRA"; tela3_detail.append(f"RSI 5m sobrevendido ({rsi_5m:.0f})")
        elif rsi_5m > 70 and tela2_sinal == "VENDA":
            tela3_entrada = "VENDA"; tela3_detail.append(f"RSI 5m sobrecomprado ({rsi_5m:.0f})")
        
        # === CHECKLIST DE CONFLUENCIA (7 pontos) ===
        confluencia = {"checks": [], "score": 0, "total": 7}
        
        # 1. Tendencia TF maior
        if tela1_dir in ("ALTA", "BAIXA"):
            confluencia["checks"].append({"nome": "Tendencia TF maior", "status": True, "detalhe": f"{tela1_dir} ({', '.join(tela1_detail)})"})
            confluencia["score"] += 1
        else:
            confluencia["checks"].append({"nome": "Tendencia TF maior", "status": False, "detalhe": "LATERAL - sem direcao clara"})
        
        # 2. S/R (usar VWAP e Bollinger como referencia)
        sr_ok = False
        sr_detail = ""
        if preco and vwap and vwap > 0:
            dist_vwap = abs(preco - vwap) / preco * 100
            if dist_vwap < 0.3:
                sr_ok = True; sr_detail = f"Preco proximo VWAP ({vwap:.0f})"
            elif preco > vwap and tela1_dir == "ALTA":
                sr_ok = True; sr_detail = f"Acima VWAP ({vwap:.0f}) - vies comprador"
            elif preco < vwap and tela1_dir == "BAIXA":
                sr_ok = True; sr_detail = f"Abaixo VWAP ({vwap:.0f}) - vies vendedor"
        if bb_upper and bb_lower and preco:
            if preco <= bb_lower: sr_ok = True; sr_detail += " | Banda inferior Bollinger"
            elif preco >= bb_upper: sr_ok = True; sr_detail += " | Banda superior Bollinger"
        confluencia["checks"].append({"nome": "Suporte/Resistencia", "status": sr_ok, "detalhe": sr_detail or "Sem nivel S/R claro"})
        if sr_ok: confluencia["score"] += 1
        
        # 3. Volume/Fluxo
        vol_ok = adx_val > 20
        confluencia["checks"].append({"nome": "Volume/Forca (ADX)", "status": vol_ok, "detalhe": f"ADX {adx_val:.0f}" + (" - tendencia forte" if adx_val > 25 else " - fraco" if adx_val < 20 else "")})
        if vol_ok: confluencia["score"] += 1
        
        # 4. Indicadores (RSI + MACD)
        ind_ok = False
        ind_detail = []
        macd_str = str(macd_5m).upper()
        if rsi_5m < 35 or rsi_5m > 65: ind_ok = True; ind_detail.append(f"RSI {rsi_5m:.0f}")
        if "ALTA" in macd_str or "COMPRA" in macd_str or "BAIXA" in macd_str or "VENDA" in macd_str:
            ind_ok = True; ind_detail.append(f"MACD {macd_5m}")
        confluencia["checks"].append({"nome": "Indicadores (RSI/MACD)", "status": ind_ok, "detalhe": ", ".join(ind_detail) if ind_detail else "Neutros"})
        if ind_ok: confluencia["score"] += 1
        
        # 5. Catalisador (horario de alta volatilidade)
        agora = datetime.now(BRT)
        hora_atual = agora.hour
        catalisador = hora_atual in [9, 10, 15, 16, 17]
        confluencia["checks"].append({"nome": "Catalisador/Timing", "status": catalisador, "detalhe": f"{agora.strftime('%H:%M')}" + (" - horario de alta volatilidade" if catalisador else " - horario de baixa volatilidade")})
        if catalisador: confluencia["score"] += 1
        
        # 6. Risco definido (R:R)
        stop_pts = round(atr * s["atr_mult"])
        stop_pts = max(round(stop_pts / s["tick"]) * s["tick"], s["tick"] * 10)
        alvo_pts = round(stop_pts * 2)
        rr_ratio = round(alvo_pts / stop_pts, 1) if stop_pts > 0 else 0
        rr_ok = rr_ratio >= 2.0
        confluencia["checks"].append({"nome": "Risco R:R", "status": rr_ok, "detalhe": f"R:R 1:{rr_ratio} | Stop {stop_pts}pts | Alvo {alvo_pts}pts"})
        if rr_ok: confluencia["score"] += 1
        
        # 7. Alinhamento Triple Screen
        ts_ok = tela1_dir != "LATERAL" and tela2_sinal != "NEUTRO" and tela3_entrada != "NEUTRO"
        ts_aligned = tela1_dir == ("ALTA" if tela3_entrada == "COMPRA" else "BAIXA" if tela3_entrada == "VENDA" else "")
        confluencia["checks"].append({"nome": "Triple Screen alinhado", "status": ts_ok and ts_aligned, "detalhe": f"T1:{tela1_dir} T2:{tela2_sinal} T3:{tela3_entrada}"})
        if ts_ok and ts_aligned: confluencia["score"] += 1
        
        # === ESCALA DE CONFIANCA (Bellafiore) ===
        cs = confluencia["score"]
        if cs >= 6: escala = {"nota": 5, "label": "A+ SETUP", "acao": "Tamanho cheio", "cor": "#22c55e"}
        elif cs >= 5: escala = {"nota": 4, "label": "SETUP BOM", "acao": "Tamanho normal", "cor": "#16a34a"}
        elif cs == 4: escala = {"nota": 3, "label": "SETUP OK", "acao": "Tamanho reduzido (-50%)", "cor": "#ca8a04"}
        elif cs == 3: escala = {"nota": 2, "label": "DUVIDOSO", "acao": "Minimo ou nao opere", "cor": "#ea580c"}
        else: escala = {"nota": 1, "label": "SEM SETUP", "acao": "NAO OPERE", "cor": "#ef4444"}
        
        # === GERAR SINAIS ===
        sinais_gerados = []
        direcao_final = tela3_entrada if tela3_entrada != "NEUTRO" else tela2_sinal
        
        if direcao_final in ("COMPRA", "VENDA") and cs >= 3:
            is_compra = direcao_final == "COMPRA"
            motivos = tela1_detail + tela2_detail + tela3_detail
            
            # Setup name (PlayBook style)
            setup_name = ""
            if rsi_5m < 30 or rsi_5m > 70: setup_name = "Divergencia RSI"
            elif ema9 > 0 and ema21 > 0 and abs(ema9 - ema21) / ema21 * 100 < 0.1: setup_name = "Cruzamento EMA"
            elif preco and vwap and abs(preco - vwap) / preco * 100 < 0.2: setup_name = "VWAP Bounce"
            elif adx_val > 25: setup_name = "Tendencia + ADX Forte"
            else: setup_name = "Confluencia Multi-TF"
            
            sinais_gerados.append({
                "tipo": direcao_final,
                "ativo": ativo,
                "setup": setup_name,
                "preco_entrada": preco,
                "stop_loss": round(preco - stop_pts, 2) if is_compra else round(preco + stop_pts, 2),
                "take_profit": round(preco + alvo_pts, 2) if is_compra else round(preco - alvo_pts, 2),
                "stop_pts": stop_pts,
                "alvo_pts": alvo_pts,
                "rr": f"1:{rr_ratio}",
                "confianca": escala["nota"],
                "escala": escala,
                "motivos": motivos,
                "timestamp": datetime.now(BRT).strftime("%H:%M:%S"),
            })
        
        if not sinais_gerados:
            sinais_gerados.append({
                "tipo": "NEUTRO",
                "ativo": ativo,
                "setup": "Aguardando",
                "confianca": 0,
                "escala": escala,
                "motivos": ["Sem confluencia suficiente - aguardar setup com 4+ pontos"],
                "timestamp": datetime.now(BRT).strftime("%H:%M:%S"),
            })
        
        return JSONResponse({
            "sinais": sinais_gerados,
            "triple_screen": {
                "tela1_tendencia": {"direcao": tela1_dir, "detalhes": tela1_detail},
                "tela2_sinal": {"direcao": tela2_sinal, "detalhes": tela2_detail},
                "tela3_entrada": {"direcao": tela3_entrada, "detalhes": tela3_detail},
            },
            "confluencia": confluencia,
            "escala_confianca": escala,
            "indicadores": {
                "rsi_5m": round(rsi_5m, 1),
                "rsi_15m": round(rsi_15m, 1),
                "tendencia_5m": tendencia_5m,
                "tendencia_15m": tendencia_15m,
                "tendencia_1h": tendencia_1h,
                "tendencia_4h": tendencia_4h,
                "tendencia_1d": tendencia_1d,
                "macd_5m": str(macd_5m),
                "macd_15m": str(macd_15m),
                "ema9": round(ema9, 2) if ema9 else None,
                "ema21": round(ema21, 2) if ema21 else None,
                "vwap": round(vwap, 2) if vwap else None,
                "adx": round(adx_val, 1),
                "atr": round(atr, 2),
                "bb_upper": round(bb_upper, 2) if bb_upper else None,
                "bb_lower": round(bb_lower, 2) if bb_lower else None,
            },
            "mercado_aberto": mercado_aberto(),
            "preco_atual": preco,
            "timestamp": datetime.now(BRT).strftime("%H:%M:%S"),
        })
    except Exception as e:
        logger.error(f"Erro sinais-ia: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"sinais": [], "erro": str(e)}, status_code=500)




# =====================================================
# NOTICIAS DE IMPACTO - Calendario Economico Real-Time
# =====================================================

@app.get("/api/noticias-impacto")
async def get_noticias_impacto():
    """Calendario economico com eventos de ALTO IMPACTO (3 estrelas) - EUA e Brasil"""
    import re as _re
    try:
        from datetime import timezone, timedelta
        BRT = timezone(timedelta(hours=-3))
        agora = datetime.now(BRT)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://br.investing.com/economic-calendar/",
            "Accept": "application/json, text/javascript, */*",
        }
        
        eventos_total = []
        
        # Fetch today + this week
        for tab_filter in ["today", "thisWeek"]:
            try:
                import urllib.request, urllib.parse
                params = urllib.parse.urlencode({
                    "country[]": ["25", "5"],  # 25=Brasil, 5=EUA
                    "importance[]": "3",  # 3 estrelas only
                    "timeZone": "12",  # BRT
                    "timeFilter": "timeRemain",
                    "currentTab": tab_filter,
                }, doseq=True)
                
                req = urllib.request.Request(
                    "https://br.investing.com/economic-calendar/Service/getCalendarFilteredData",
                    data=params.encode(),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    import json
                    raw = json.loads(resp.read().decode())
                    html_data = raw.get("data", "")
                
                # Parse HTML events
                day_pattern = _re.compile(r'class="theDay"[^>]*>([^<]+)<')
                row_pattern = _re.compile(
                    r'event_attr_ID="(\d+)"[^>]*data-event-datetime="([^"]+)".*?'
                    r'js-time"[^>]*>([^<]*)<.*?'
                    r'title="([^"]*)"[^>]*class="ceFlags\s+(\w+)".*?'
                    r'(\w{3})<.*?'
                    r'event"[^>]*>(.*?)</a>.*?'
                    r'(?:bold\s*"[^>]*>([^<]*)<.*?)?'  # actual
                    r'(?:prev"[^>]*>([^<]*)<.*?)?'  # forecast
                    r'(?:prev"[^>]*>([^<]*)<)?',  # previous
                    _re.DOTALL
                )
                
                # Simpler row-by-row parsing
                current_day = agora.strftime("%d/%m/%Y")
                rows = html_data.split('js-event-item')
                
                for row in rows[1:]:  # skip first empty
                    try:
                        # Extract event ID
                        eid_m = _re.search(r'event_attr_ID="(\d+)"', row)
                        eid = eid_m.group(1) if eid_m else "0"
                        
                        # Extract datetime
                        dt_m = _re.search(r'data-event-datetime="([^"]+)"', row)
                        evt_dt = dt_m.group(1) if dt_m else ""
                        
                        # Extract time
                        time_m = _re.search(r'js-time"[^>]*>([^<]*)<', row)
                        evt_time = time_m.group(1).strip() if time_m else ""
                        
                        # Extract country
                        country_m = _re.search(r'title="([^"]*)"[^>]*class="ceFlags', row)
                        country = country_m.group(1).strip() if country_m else ""
                        
                        # Extract currency
                        cur_m = _re.search(r'ceFlags[^>]*>[^<]*</span>\s*(\w{3})', row)
                        currency = cur_m.group(1).strip() if cur_m else ""
                        
                        # Extract event name
                        name_m = _re.search(r'class="[^"]*event[^"]*"[^>]*>([^<]+)<', row)
                        evt_name = name_m.group(1).strip() if name_m else "Evento"
                        # Clean HTML entities
                        evt_name = evt_name.replace("&amp;", "&").replace("&nbsp;", " ").replace("&#39;", "'")
                        
                        # Extract actual, forecast, previous values
                        bold_vals = _re.findall(r'<td[^>]*class="[^"]*bold[^"]*"[^>]*>\s*([^<]*?)\s*</td>', row)
                        # Also try spans inside tds
                        if not bold_vals:
                            bold_vals = _re.findall(r'bold[^>]*>([^<]*)<', row)
                        
                        actual = ""
                        forecast = ""
                        previous = ""
                        
                        if len(bold_vals) >= 1:
                            actual = bold_vals[0].strip().replace("&nbsp;", "").strip()
                        if len(bold_vals) >= 2:
                            forecast = bold_vals[1].strip().replace("&nbsp;", "").strip()
                        if len(bold_vals) >= 3:
                            previous = bold_vals[2].strip().replace("&nbsp;", "").strip()
                        
                        # Determine impact on WIN and WDO
                        impacto_win = ""
                        impacto_wdo = ""
                        surpresa = ""
                        
                        # Default impact based on event type (even without forecast)
                        operavel = "BOM"
                        evt_lower = evt_name.lower()
                        if currency == "USD":
                            impacto_win = "Dado americano - monitorar"
                            impacto_wdo = "Dado americano - impacto no dolar"
                            if any(kw in evt_lower for kw in ["payroll", "nonfarm", "fomc", "fed", "cpi", "ipc", "taxa de juros", "interest rate", "gdp", "pib"]):
                                operavel = "CAUTELA"
                                impacto_win = "ALTO IMPACTO potencial no indice"
                                impacto_wdo = "ALTO IMPACTO potencial no dolar"
                            elif any(kw in evt_lower for kw in ["pmi", "ism", "emprego", "vendas", "producao", "pedidos"]):
                                operavel = "BOM"
                                impacto_win = "Impacto moderado no indice"
                                impacto_wdo = "Impacto moderado no dolar"
                            else:
                                impacto_win = "Baixo impacto no indice"
                                impacto_wdo = "Baixo impacto no dolar"
                        elif currency == "BRL":
                            impacto_win = "Dado brasileiro - monitorar"
                            impacto_wdo = "Dado brasileiro - impacto no dolar"
                            if any(kw in evt_lower for kw in ["selic", "copom", "ipca", "pib"]):
                                operavel = "CAUTELA"
                                impacto_win = "ALTO IMPACTO no Ibovespa"
                                impacto_wdo = "ALTO IMPACTO no dolar"
                            else:
                                impacto_win = "Impacto moderado no indice"
                                impacto_wdo = "Impacto moderado no dolar"
                        
                        if actual and forecast:
                            try:
                                act_num = float(actual.replace("%", "").replace(",", ".").replace("K", "000").replace("M", "000000").replace("B", "000000000").strip())
                                for_num = float(forecast.replace("%", "").replace(",", ".").replace("K", "000").replace("M", "000000").replace("B", "000000000").strip())
                                diff = act_num - for_num
                                
                                if currency == "USD":
                                    if diff > 0:
                                        surpresa = "ACIMA"
                                        impacto_wdo = "ALTA (dolar fortalece)"
                                        impacto_win = "BAIXA (pressao no ibov)"
                                    elif diff < 0:
                                        surpresa = "ABAIXO"
                                        impacto_wdo = "BAIXA (dolar enfraquece)"
                                        impacto_win = "ALTA (alivio no ibov)"
                                    else:
                                        surpresa = "NEUTRO"
                                        impacto_wdo = "NEUTRO"
                                        impacto_win = "NEUTRO"
                                elif currency == "BRL":
                                    if diff > 0:
                                        surpresa = "ACIMA"
                                        impacto_win = "DEPENDE (contexto)"
                                        impacto_wdo = "DEPENDE (contexto)"
                                    elif diff < 0:
                                        surpresa = "ABAIXO"
                                        impacto_win = "DEPENDE (contexto)"
                                        impacto_wdo = "DEPENDE (contexto)"
                                    else:
                                        surpresa = "NEUTRO"
                                        impacto_win = "NEUTRO"
                                        impacto_wdo = "NEUTRO"
                                    
                                    # Specific Brazilian events
                                    evt_lower = evt_name.lower()
                                    if "selic" in evt_lower or "copom" in evt_lower:
                                        if diff > 0:  # juros subiu mais
                                            impacto_win = "BAIXA (juros apertam)"
                                            impacto_wdo = "BAIXA (atrai capital)"
                                        elif diff < 0:
                                            impacto_win = "ALTA (juros aliviam)"
                                            impacto_wdo = "ALTA (menos atrativo)"
                                    elif "ipca" in evt_lower or "inflacao" in evt_lower:
                                        if diff > 0:
                                            impacto_win = "BAIXA (inflacao alta)"
                                            impacto_wdo = "ALTA (real enfraquece)"
                                        elif diff < 0:
                                            impacto_win = "ALTA (inflacao controlada)"
                                            impacto_wdo = "BAIXA (real fortalece)"
                                    elif "pib" in evt_lower:
                                        if diff > 0:
                                            impacto_win = "ALTA (economia forte)"
                                            impacto_wdo = "BAIXA (confianca)"
                                        elif diff < 0:
                                            impacto_win = "BAIXA (economia fraca)"
                                            impacto_wdo = "ALTA (fuga capital)"
                                
                                # US specific events
                                if currency == "USD":
                                    evt_lower = evt_name.lower()
                                    if "nonfarm" in evt_lower or "payroll" in evt_lower or "emprego" in evt_lower:
                                        if diff > 0:
                                            impacto_wdo = "ALTA FORTE (USD fortalece)"
                                            impacto_win = "BAIXA FORTE (risk-off)"
                                        else:
                                            impacto_wdo = "BAIXA FORTE (USD enfraquece)"
                                            impacto_win = "ALTA FORTE (risk-on)"
                                    elif "cpi" in evt_lower or "ipc" in evt_lower or "inflacao" in evt_lower:
                                        if diff > 0:
                                            impacto_wdo = "ALTA (Fed hawkish)"
                                            impacto_win = "BAIXA (juros US sobem)"
                                        else:
                                            impacto_wdo = "BAIXA (Fed dovish)"
                                            impacto_win = "ALTA (juros US caem)"
                                    elif "fed" in evt_lower or "fomc" in evt_lower or "juros" in evt_lower:
                                        if diff > 0:
                                            impacto_wdo = "ALTA FORTE (USD fortalece)"
                                            impacto_win = "BAIXA FORTE (fuga emergentes)"
                                        else:
                                            impacto_wdo = "BAIXA FORTE (USD enfraquece)"
                                            impacto_win = "ALTA FORTE (fluxo emergentes)"
                                    elif "pib" in evt_lower or "gdp" in evt_lower:
                                        if diff > 0:
                                            impacto_wdo = "ALTA (USD forte)"
                                            impacto_win = "MISTA (global forte)"
                                        else:
                                            impacto_wdo = "BAIXA (USD fraco)"
                                            impacto_win = "MISTA (global fraco)"
                            except:
                                pass
                        
                        # Parse datetime for countdown
                        minutos_restantes = None
                        ja_passou = False
                        if evt_dt:
                            try:
                                evt_datetime = datetime.strptime(evt_dt, "%Y/%m/%d %H:%M:%S")
                                evt_datetime = evt_datetime.replace(tzinfo=BRT)
                                delta = (evt_datetime - agora).total_seconds()
                                minutos_restantes = round(delta / 60)
                                ja_passou = delta < 0
                            except:
                                pass
                        
                        # Zone: 15min before/after
                        zona_impacto = ""
                        if minutos_restantes is not None:
                            if -15 <= minutos_restantes <= 15:
                                zona_impacto = "ZONA DE IMPACTO"
                            elif 0 < minutos_restantes <= 30:
                                zona_impacto = "APROXIMANDO"
                            elif minutos_restantes > 30:
                                zona_impacto = "AGUARDANDO"
                            elif minutos_restantes < -15:
                                zona_impacto = "ENCERRADO"
                        
                        if not evt_name or evt_name == "Evento":
                            continue
                        
                        eventos_total.append({
                            "id": eid,
                            "datetime": evt_dt,
                            "hora": evt_time,
                            "pais": country,
                            "moeda": currency,
                            "evento": evt_name,
                            "impacto": 3,
                            "atual": actual,
                            "previsao": forecast,
                            "anterior": previous,
                            "surpresa": surpresa,
                            "impacto_win": impacto_win,
                            "impacto_wdo": impacto_wdo,
                            "operavel": operavel,
                            "minutos_restantes": minutos_restantes,
                            "ja_passou": ja_passou,
                            "zona_impacto": zona_impacto,
                            "tab": tab_filter,
                        })
                    except Exception as parse_err:
                        continue
                        
            except Exception as fetch_err:
                logger.error(f"Erro fetch calendario {tab_filter}: {fetch_err}")
                continue
        
        # Deduplicate by event ID
        seen = set()
        unique = []
        for e in eventos_total:
            if e["id"] not in seen:
                seen.add(e["id"])
                unique.append(e)
        
        # Sort: upcoming first, then past
        upcoming = [e for e in unique if not e["ja_passou"]]
        past = [e for e in unique if e["ja_passou"]]
        upcoming.sort(key=lambda x: x.get("minutos_restantes") or 9999)
        past.sort(key=lambda x: -(x.get("minutos_restantes") or 0))
        
        # Alert: any event in zona de impacto?
        alertas = [e for e in unique if e["zona_impacto"] in ("ZONA DE IMPACTO", "APROXIMANDO")]
        
        return JSONResponse({
            "proximos": upcoming,
            "passados": past,
            "total": len(unique),
            "alertas": len(alertas),
            "alerta_eventos": [{"evento": a["evento"], "moeda": a["moeda"], "hora": a["hora"], "zona": a["zona_impacto"], "minutos": a["minutos_restantes"]} for a in alertas],
            "timestamp": agora.strftime("%H:%M:%S"),
            "data": agora.strftime("%d/%m/%Y"),
        })
    except Exception as e:
        logger.error(f"Erro noticias-impacto: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"erro": str(e), "proximos": [], "passados": []}, status_code=500)




# =====================================================
# SIMULADOR REAL - ANALISE COMPLETA DO PREGAO ANTERIOR
# =====================================================

@app.get("/api/simulador-real")
async def simulador_real(ativo: str = Query("WIN")):
    """Analise completa do pregao anterior: vela por vela com decisoes de operar/nao operar"""
    try:
        from datetime import timezone, timedelta
        BRT_tz = timezone(timedelta(hours=-3))
        ativo = ativo.upper()
        ticker = "^BVSP" if ativo == "WIN" else "USDBRL=X"
        valor_ponto = 0.20 if ativo == "WIN" else 10.00
        
        from data_provider import obter_contrato_vigente as _ocv
        contrato_info = _ocv(ativo)
        contrato_nome = contrato_info.get('ticker_b3', ativo)
        
        # Get 5 days of data
        dados = yf.download(ticker, period="5d", interval="5m", progress=False)
        if dados.empty:
            return JSONResponse({"erro": "Sem dados do yfinance"})
        
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
        dados.columns = [c.lower() for c in dados.columns]
        dados.index = dados.index.tz_convert(BRT_tz)
        
        hoje = datetime.now(BRT_tz).date()
        dates = sorted(set(dados.index.date))
        dia_anterior = None
        for d in reversed(dates):
            if d < hoje:
                dia_anterior = d
                break
        if not dia_anterior:
            return JSONResponse({"erro": "Dia anterior nao encontrado"})
        
        day_indices = [i for i, d in enumerate(dados.index.date) if d == dia_anterior]
        if not day_indices:
            return JSONResponse({"erro": "Sem velas do dia anterior"})
        
        from analysis_engine import calcular_rsi, calcular_macd, calcular_atr_series
        
        # ---- PRO TRADER CONTROLS ----
        # Regras de um operador profissional que PROTEGE capital:
        MAX_OPS_DIA = 4          # Maximo 4 operacoes por dia (Bellafiore: poucos trades, bem executados)
        MAX_LOSSES_CONSECUTIVOS = 2  # Apos 2 losses seguidos, PARA (Tendler: evitar tilt/revenge)
        LOSS_LIMIT_PTS = -400    # Limite de perda diaria em pontos (gestao de risco)
        
        losses_consecutivos = 0
        total_pts_dia = 0
        dia_bloqueado = False    # True quando atingir limite
        
        # ---- ANALYZE EVERY CANDLE ----
        velas_analisadas = []
        operacoes_recomendadas = []
        posicao_aberta = None  # track open position to avoid overlapping
        
        for pos_idx in day_indices:
            w = dados.iloc[max(0, pos_idx - 100):pos_idx + 1]
            vela = dados.iloc[pos_idx]
            ts = dados.index[pos_idx]
            hora = ts.strftime("%H:%M")
            hora_int = ts.hour
            minuto = ts.minute
            
            o = float(vela['open']); h = float(vela['high'])
            l = float(vela['low']); c = float(vela['close'])
            vol = int(vela.get('volume', 0))
            
            # Calculate indicators
            rsi_v = 50; macd_h = 0; ema9 = 0; ema21 = 0; ema50 = 0; atr_v = 150
            tend = "LATERAL"
            
            if len(w) >= 20:
                try:
                    rsi_s = calcular_rsi(w)
                    rsi_v = round(float(rsi_s.iloc[-1]), 1)
                    ml, ms, mh = calcular_macd(w)
                    macd_h = round(float(mh.iloc[-1]), 1)
                    ema9 = round(float(w['close'].ewm(span=9, adjust=False).mean().iloc[-1]), 2)
                    ema21 = round(float(w['close'].ewm(span=21, adjust=False).mean().iloc[-1]), 2)
                    ema50 = round(float(w['close'].ewm(span=50, adjust=False).mean().iloc[-1]), 2) if len(w) >= 50 else 0
                    atr_s = calcular_atr_series(w)
                    atr_v = round(float(atr_s.iloc[-1]), 1) if len(atr_s) > 0 else 150
                    
                    if ema9 > ema21 and c > ema9: tend = "ALTA"
                    elif ema9 < ema21 and c < ema9: tend = "BAIXA"
                    elif ema9 > ema21: tend = "ALTA"
                    elif ema9 < ema21: tend = "BAIXA"
                except: pass
            
            # ---- SCORING SYSTEM (Triple Screen + Confluencia 7pts) ----
            # Elder Triple Screen: SO opere na direcao da Tela 1 (tendencia maior)
            score = 0
            motivos_operar = []
            motivos_nao_operar = []
            tipo_sinal = None
            contra_tendencia = False
            
            # TENDENCIA PRINCIPAL do dia (Tela 1 - Elder)
            # Calculada globalmente abaixo apos processar todas velas
            # Aqui usamos tend local como proxy inicial
            
            # 1. Horario (Bellafiore: "In Play" = volume + volatilidade)
            bom_horario = hora_int in [9, 10, 14, 15, 16]
            horario_ruim = hora_int in [12, 13, 17]
            if hora_int == 9 and minuto < 15:
                horario_ruim = True
                motivos_nao_operar.append("Primeiros 15min - volatilidade caotica")
            if bom_horario and not horario_ruim:
                score += 1
                motivos_operar.append(f"Horario forte ({hora})")
            elif horario_ruim:
                motivos_nao_operar.append(f"Horario fraco ({hora}) - evitar")
            
            # 2. RSI - mas RESPEITA TENDENCIA (Elder)
            # Em ALTA: RSI sobrevendido = COMPRA (pullback). RSI sobrecomprado = NAO vender, trend forte
            # Em BAIXA: RSI sobrecomprado = VENDA (pullback). RSI sobrevendido = NAO comprar
            if rsi_v < 30:
                if tend != "BAIXA":  # Nao comprar contra tendencia baixa
                    score += 2; tipo_sinal = "COMPRA"
                    motivos_operar.append(f"RSI sobrevendido ({rsi_v}) - pullback p/ compra")
                else:
                    motivos_nao_operar.append(f"RSI sobrevendido ({rsi_v}) mas tendencia BAIXA - nao compre contra")
            elif rsi_v > 70:
                if tend != "ALTA":  # Nao vender contra tendencia alta
                    score += 2; tipo_sinal = "VENDA"
                    motivos_operar.append(f"RSI sobrecomprado ({rsi_v}) - pullback p/ venda")
                else:
                    # Em tendencia alta, RSI alto = forca, nao reversal
                    motivos_operar.append(f"RSI alto ({rsi_v}) - tendencia forte")
                    tipo_sinal = "COMPRA"
                    score += 1
            elif 40 <= rsi_v <= 60:
                motivos_nao_operar.append(f"RSI neutro ({rsi_v}) - sem forca direcional")
            else:
                # RSI 30-40 ou 60-70: zona intermediaria
                if tend == "ALTA" and rsi_v > 50:
                    tipo_sinal = "COMPRA"
                elif tend == "BAIXA" and rsi_v < 50:
                    tipo_sinal = "VENDA"
            
            # 3. EMA alignment (Murphy: cruzamento medias)
            if ema9 > 0 and ema21 > 0:
                if ema9 > ema21 and c > ema9:
                    score += 1
                    if not tipo_sinal: tipo_sinal = "COMPRA"
                    motivos_operar.append("EMA9 > EMA21 + preco acima (estrutura compradora)")
                elif ema9 < ema21 and c < ema9:
                    score += 1
                    if not tipo_sinal: tipo_sinal = "VENDA"
                    motivos_operar.append("EMA9 < EMA21 + preco abaixo (estrutura vendedora)")
                else:
                    motivos_nao_operar.append("EMAs sem alinhamento claro")
            
            # 4. Tendencia confirmada (Triple Screen Tela 1 + Tela 2)
            if tend == "ALTA" and tipo_sinal == "COMPRA":
                score += 1; motivos_operar.append("Tendencia ALTA confirmada (Triple Screen alinhado)")
            elif tend == "BAIXA" and tipo_sinal == "VENDA":
                score += 1; motivos_operar.append("Tendencia BAIXA confirmada (Triple Screen alinhado)")
            elif tend == "LATERAL":
                motivos_nao_operar.append("Mercado LATERAL - sem tendencia definida")
            elif tipo_sinal and ((tend == "ALTA" and tipo_sinal == "VENDA") or (tend == "BAIXA" and tipo_sinal == "COMPRA")):
                # CONTRA TENDENCIA = penaliza pesado (Elder: NUNCA contra Tela 1)
                score -= 2
                contra_tendencia = True
                motivos_nao_operar.append(f"CONTRA TENDENCIA! {tipo_sinal} em mercado {tend} (Elder: proibido)")
            
            # 5. MACD histogram (sinal + confirmacao)
            if macd_h > 0 and tipo_sinal == "COMPRA":
                score += 1; motivos_operar.append(f"MACD positivo ({macd_h}) - momentum comprador")
            elif macd_h < 0 and tipo_sinal == "VENDA":
                score += 1; motivos_operar.append(f"MACD negativo ({macd_h}) - momentum vendedor")
            elif macd_h > 0 and tipo_sinal == "VENDA":
                motivos_nao_operar.append(f"MACD positivo ({macd_h}) CONTRA sinal de venda")
            elif macd_h < 0 and tipo_sinal == "COMPRA":
                motivos_nao_operar.append(f"MACD negativo ({macd_h}) CONTRA sinal de compra")
            
            # 6. ATR (volatilidade minima p/ day trade)
            if atr_v > 80:
                score += 1; motivos_operar.append(f"ATR {atr_v} - volatilidade suficiente")
            else:
                motivos_nao_operar.append(f"ATR {atr_v} - volatilidade baixa (spread pode comer lucro)")
            
            # 7. Volume (se disponivel)
            if vol > 0:
                score += 0  # placeholder - yfinance nem sempre traz vol de futuros
            
            # 7. Suporte e Resistencia (Murphy)
            suporte = None
            resistencia = None
            if len(w) >= 20:
                recent_highs = [float(w.iloc[j]['high']) for j in range(-20, 0)]
                recent_lows = [float(w.iloc[j]['low']) for j in range(-20, 0)]
                resistencia = max(recent_highs)
                suporte = min(recent_lows)
                dist_suporte = abs(c - suporte)
                dist_resistencia = abs(resistencia - c)
                
                # Near support + COMPRA = good
                if tipo_sinal == "COMPRA" and dist_suporte < atr_v * 0.5:
                    score += 1
                    motivos_operar.append(f"Proximo ao suporte {round(suporte,0)} (S/R Murphy)")
                # Near resistance + VENDA = good  
                elif tipo_sinal == "VENDA" and dist_resistencia < atr_v * 0.5:
                    score += 1
                    motivos_operar.append(f"Proximo a resistencia {round(resistencia,0)} (S/R Murphy)")
                else:
                    motivos_nao_operar.append("Preco longe de S/R relevante")
            
            # 8. VWAP - Preco medio ponderado volume
            vwap = None
            if len(w) >= 10:
                try:
                    typical = (w['high'] + w['low'] + w['close']) / 3
                    vol_s = w['volume'].replace(0, 1)
                    vwap = float((typical * vol_s).cumsum().iloc[-1] / vol_s.cumsum().iloc[-1])
                    dist_vwap = c - vwap
                    if tipo_sinal == "COMPRA" and c > vwap:
                        score += 1
                        motivos_operar.append(f"Acima VWAP {round(vwap,0)} (comprador)")
                    elif tipo_sinal == "VENDA" and c < vwap:
                        score += 1
                        motivos_operar.append(f"Abaixo VWAP {round(vwap,0)} (vendedor)")
                    elif tipo_sinal == "COMPRA" and c < vwap:
                        motivos_nao_operar.append(f"Abaixo VWAP - compra arriscada")
                    elif tipo_sinal == "VENDA" and c > vwap:
                        motivos_nao_operar.append(f"Acima VWAP - venda arriscada")
                except: pass
            
            # 9. Fibonacci (retracao do swing recente)
            fib_level = None
            if len(w) >= 30:
                try:
                    swing_high = max(float(w.iloc[j]['high']) for j in range(-30, 0))
                    swing_low = min(float(w.iloc[j]['low']) for j in range(-30, 0))
                    fib_range = swing_high - swing_low
                    if fib_range > 0:
                        fib_382 = swing_high - fib_range * 0.382
                        fib_500 = swing_high - fib_range * 0.500
                        fib_618 = swing_high - fib_range * 0.618
                        # Check if price is near a Fibonacci level
                        for fib_lv, fib_name in [(fib_382, "38.2%"), (fib_500, "50%"), (fib_618, "61.8%")]:
                            if abs(c - fib_lv) < atr_v * 0.3:
                                score += 1
                                fib_level = fib_name
                                motivos_operar.append(f"Proximo Fibonacci {fib_name} ({round(fib_lv,0)})")
                                break
                except: pass
            
            # 10. Candlestick Patterns
            if len(w) >= 3:
                try:
                    prev = w.iloc[-2]
                    prev2 = w.iloc[-3]
                    body = abs(c - o)
                    upper_shadow = h - max(c, o)
                    lower_shadow = min(c, o) - l
                    prev_body = abs(float(prev['close']) - float(prev['open']))
                    
                    # Martelo (hammer) - bullish reversal
                    if lower_shadow > body * 2 and upper_shadow < body * 0.5 and c > o:
                        if tipo_sinal == "COMPRA":
                            score += 1
                            motivos_operar.append("Candlestick: Martelo (reversao altista)")
                    # Estrela Cadente (shooting star) - bearish reversal
                    elif upper_shadow > body * 2 and lower_shadow < body * 0.5 and o > c:
                        if tipo_sinal == "VENDA":
                            score += 1
                            motivos_operar.append("Candlestick: Estrela Cadente (reversao baixista)")
                    # Engolfo de Alta
                    elif c > o and float(prev['close']) < float(prev['open']) and body > prev_body * 1.2:
                        if tipo_sinal == "COMPRA":
                            score += 1
                            motivos_operar.append("Candlestick: Engolfo de Alta")
                    # Engolfo de Baixa
                    elif o > c and float(prev['close']) > float(prev['open']) and body > prev_body * 1.2:
                        if tipo_sinal == "VENDA":
                            score += 1
                            motivos_operar.append("Candlestick: Engolfo de Baixa")
                except: pass
            
            # 11. SCORING AVANÇADO (Al Brooks, Nison, Grimes, Williams, Taleb)
            try:
                _extra_score, _extra_motivos = aplicar_scoring_avancado(
                    vela_info, tipo_sinal, tend, rsi_v, atr_v, macd_h,
                    c, o, h, l, suporte, resistencia, ema9, ema21, w
                )
                score += _extra_score
                motivos_operar.extend(_extra_motivos)
            except Exception as _e:
                pass  # Non-critical
            
            # 12. SMC - Smart Money Concepts (FVG, Liquidity Sweep, Order Block, BOS/CHoCH)
            smc_data = {}
            try:
                _smc_score, _smc_motivos, smc_data = aplicar_smc_scoring(
                    dados, pos_idx, tipo_sinal, tend
                )
                score += _smc_score
                motivos_operar.extend(_smc_motivos)
            except Exception as _e:
                pass  # Non-critical
            
            # DECISAO FINAL - PRO TRADER: SO ENTRA NO SEGURO
            # Score minimo 9 = A+ SETUP ONLY. Nao entra em BOM, OK, Duvidoso.
            # "Entre so no que for seguro" - Fabio
            # Livermore: "O dinheiro grande esta no ESPERAR, nao no trading"
            _score_min = max(9, obter_score_minimo())  # NUNCA abaixo de 9
            operar = score >= _score_min and tipo_sinal is not None and not horario_ruim and not contra_tendencia
            decisao = "OPERAR" if operar else "NAO OPERAR"
            
            # Confianca (Bellafiore) - PRO scoring
            if score >= 9: confianca = 5; conf_label = "A+ SETUP"
            elif score >= 7: confianca = 4; conf_label = "BOM"
            elif score >= 5: confianca = 3; conf_label = "OK"
            elif score >= 4: confianca = 2; conf_label = "DUVIDOSO"
            else: confianca = 1; conf_label = "SEM SETUP"
            
            vela_info = {
                "idx": len(velas_analisadas),
                "hora": hora,
                "open": round(o, 2), "high": round(h, 2),
                "low": round(l, 2), "close": round(c, 2),
                "rsi": rsi_v, "macd_hist": macd_h,
                "ema9": ema9, "ema21": ema21,
                "atr": atr_v, "tendencia": tend,
                "score": score, "decisao": decisao,
                "tipo_sinal": tipo_sinal,
                "confianca": confianca, "conf_label": conf_label,
                "motivos_operar": motivos_operar,
                "motivos_nao_operar": motivos_nao_operar,
                "suporte": round(suporte, 0) if suporte else None,
                "resistencia": round(resistencia, 0) if resistencia else None,
                "vwap": round(vwap, 0) if vwap else None,
                "fib_level": fib_level,
                "smc": smc_data,
            }
            velas_analisadas.append(vela_info)
            
            # Check if position closed (+ cooldown)
            if posicao_aberta:
                current_day_idx = day_indices.index(pos_idx) if pos_idx in day_indices else 0
                if current_day_idx >= posicao_aberta.get("close_idx", 0):
                    posicao_aberta = None
            
            # SIMULATE operations: PRO TRADER - poucos trades, alta confianca
            # Verifica controles de risco ANTES de entrar
            pode_operar = (
                operar and tipo_sinal 
                and posicao_aberta is None 
                and hora_int < 17 
                and not (hora_int == 16 and minuto > 30)
                and len(operacoes_recomendadas) < MAX_OPS_DIA  # Limite diario
                and losses_consecutivos < MAX_LOSSES_CONSECUTIVOS  # Parar apos losses
                and not dia_bloqueado  # Limite de perda
            )
            
            if pode_operar:
                # Day trade: stop = 1x ATR (max 300pts WIN), alvo = 1.5x stop
                stop_pts = round(atr_v * 1.0)
                stop_pts = max(round(stop_pts / 5) * 5, 50)  # round to tick, min 50
                if ativo == "WIN":
                    stop_pts = min(stop_pts, 300)  # cap stop WIN
                alvo_pts = round(stop_pts * 1.5)
                
                is_compra = tipo_sinal == "COMPRA"
                stop_price = round(c - stop_pts, 2) if is_compra else round(c + stop_pts, 2)
                alvo_price = round(c + alvo_pts, 2) if is_compra else round(c - alvo_pts, 2)
                
                # Walk forward (max 30 velas = 2.5h timeout)
                resultado = None
                preco_saida = 0
                hora_saida = ""
                velas_na_op = 0
                max_velas_op = 30  # timeout
                
                future_start = day_indices.index(pos_idx) + 1
                for fi in range(future_start, len(day_indices)):
                    fv = dados.iloc[day_indices[fi]]
                    fh = float(fv['high']); fl = float(fv['low'])
                    fc = float(fv['close'])
                    f_hora = dados.index[day_indices[fi]].strftime("%H:%M")
                    velas_na_op += 1
                    
                    # Timeout: fecha no preco atual apos 30 velas
                    if velas_na_op >= max_velas_op:
                        preco_saida = fc
                        hora_saida = f_hora
                        pts_t = (preco_saida - c) if is_compra else (c - preco_saida)
                        resultado = "WIN" if pts_t > 0 else "LOSS"
                        break
                    
                    if is_compra:
                        if fl <= stop_price:
                            resultado = "LOSS"; preco_saida = stop_price; hora_saida = f_hora; break
                        if fh >= alvo_price:
                            resultado = "WIN"; preco_saida = alvo_price; hora_saida = f_hora; break
                    else:
                        if fh >= stop_price:
                            resultado = "LOSS"; preco_saida = stop_price; hora_saida = f_hora; break
                        if fl <= alvo_price:
                            resultado = "WIN"; preco_saida = alvo_price; hora_saida = f_hora; break
                
                if resultado is None:
                    last_v = dados.iloc[day_indices[-1]]
                    preco_saida = float(last_v['close'])
                    hora_saida = dados.index[day_indices[-1]].strftime("%H:%M")
                    pts_f = (preco_saida - c) if is_compra else (c - preco_saida)
                    resultado = "WIN" if pts_f > 0 else "LOSS"
                
                pts = round((preco_saida - c) if is_compra else (c - preco_saida), 1)
                rs = round(pts * valor_ponto, 2)
                
                # Detalhes da perda quando LOSS - ANALISE COMPLETA DO PQ PERDEU
                detalhes_perda = ""
                if resultado == "LOSS":
                    if velas_na_op >= max_velas_op:
                        detalhes_perda = f"TIMEOUT: Operacao aberta por {velas_na_op} velas ({velas_na_op*5}min) sem atingir alvo nem stop. "
                        detalhes_perda += f"Fechou no preco {round(preco_saida,2)} com prejuizo de {round(abs(pts),1)}pts. "
                        detalhes_perda += "Licao: Mercado ficou lateral/indeciso. Timeout protege de operar em congestao prolongada. "
                    elif is_compra:
                        detalhes_perda = f"STOP atingido em {hora_saida} ({velas_na_op} velas = {velas_na_op*5}min). "
                        detalhes_perda += f"Preco caiu de {round(c,2)} ate stop {round(stop_price,2)} (-{stop_pts}pts = -R${round(stop_pts * valor_ponto, 2)}). "
                        # Analise do que deu errado
                        problemas = []
                        if macd_h < 0:
                            problemas.append(f"MACD negativo ({macd_h}) ja indicava momentum vendedor na entrada")
                        if rsi_v > 70:
                            problemas.append(f"RSI sobrecomprado ({rsi_v}) - mercado ja esticado, reversao provavel")
                        if rsi_v < 40 and tend != "ALTA":
                            problemas.append(f"RSI fraco ({rsi_v}) sem tendencia de alta para suportar")
                        if atr_v > 300:
                            problemas.append(f"ATR muito alto ({atr_v}) - volatilidade excessiva aumenta risco de stop")
                        if vwap and c < vwap:
                            problemas.append(f"Preco ABAIXO do VWAP ({round(vwap,0)}) - comprando contra fluxo institucional")
                        if suporte and abs(c - suporte) > atr_v:
                            problemas.append(f"Longe do suporte ({round(suporte,0)}) - sem protecao natural de preco")
                        if not problemas:
                            problemas.append("Setup estava correto - loss faz parte (nem todo setup ganha)")
                        detalhes_perda += "PROBLEMAS: " + "; ".join(problemas) + ". "
                        detalhes_perda += f"Score era {score}/11 ({conf_label}). "
                        detalhes_perda += "Licao: " + ("Volatilidade alta exige stop mais conservador ou esperar pullback." if atr_v > 250 else "Verifique alinhamento do TF maior antes de entrar. Loss com setup correto e normal - disciplina e seguir o plano.")
                    else:
                        detalhes_perda = f"STOP atingido em {hora_saida} ({velas_na_op} velas = {velas_na_op*5}min). "
                        detalhes_perda += f"Preco subiu de {round(c,2)} ate stop {round(stop_price,2)} (-{stop_pts}pts = -R${round(stop_pts * valor_ponto, 2)}). "
                        problemas = []
                        if macd_h > 0:
                            problemas.append(f"MACD positivo ({macd_h}) ja indicava momentum comprador")
                        if tend == "ALTA":
                            problemas.append("VENDA contra tendencia ALTA - Elder proibe operar contra Tela 1")
                        if rsi_v < 30:
                            problemas.append(f"RSI sobrevendido ({rsi_v}) - bounce esperado")
                        if vwap and c > vwap:
                            problemas.append(f"Preco ACIMA do VWAP ({round(vwap,0)}) - vendendo contra fluxo")
                        if resistencia and abs(resistencia - c) > atr_v:
                            problemas.append(f"Longe da resistencia ({round(resistencia,0)}) - sem teto natural")
                        if not problemas:
                            problemas.append("Setup estava correto - loss faz parte do jogo")
                        detalhes_perda += "PROBLEMAS: " + "; ".join(problemas) + ". "
                        detalhes_perda += f"Score era {score}/11 ({conf_label}). "
                        detalhes_perda += "Licao: Nunca venda em tendencia de alta clara. Disciplina > opiniao."
                
                # ---- ANALISE COMPLETA (PRO TRADER) ----
                # Narrativa detalhada de TUDO que foi analisado, como um operador profissional
                analise_completa = f"== ANALISE COMPLETA da entrada {tipo_sinal} as {hora} ==\n"
                analise_completa += f"PRECO: O={round(o,2)} H={round(h,2)} L={round(l,2)} C={round(c,2)}\n"
                analise_completa += f"\n1. TENDENCIA (Elder Triple Screen - Tela 1): {tend}\n"
                analise_completa += f"   EMA9={ema9} vs EMA21={ema21}"
                if ema50 > 0:
                    analise_completa += f" vs EMA50={ema50}"
                analise_completa += f"\n   Preco {'acima' if c > ema9 else 'abaixo'} da EMA9. "
                if tend == "ALTA":
                    analise_completa += "Estrutura compradora confirmada.\n"
                elif tend == "BAIXA":
                    analise_completa += "Estrutura vendedora confirmada.\n"
                else:
                    analise_completa += "Sem direcao clara.\n"
                
                analise_completa += f"\n2. RSI (Wilder): {rsi_v}\n"
                if rsi_v > 70:
                    analise_completa += f"   Sobrecomprado. Em ALTA=forca, em BAIXA=possivel reversao.\n"
                elif rsi_v < 30:
                    analise_completa += f"   Sobrevendido. Em BAIXA=fraqueza, em ALTA=oportunidade compra.\n"
                else:
                    analise_completa += f"   Zona {'compradora' if rsi_v > 50 else 'vendedora' if rsi_v < 50 else 'neutra'}.\n"
                
                analise_completa += f"\n3. MACD Histograma: {macd_h}\n"
                analise_completa += f"   Momentum {'comprador' if macd_h > 0 else 'vendedor' if macd_h < 0 else 'neutro'}. "
                if (macd_h > 0 and tipo_sinal == "COMPRA") or (macd_h < 0 and tipo_sinal == "VENDA"):
                    analise_completa += "CONFIRMADO com a direcao da operacao.\n"
                else:
                    analise_completa += "DIVERGENTE com a direcao da operacao.\n"
                
                analise_completa += f"\n4. ATR (Volatilidade): {atr_v} pts\n"
                analise_completa += f"   {'Volatilidade suficiente para day trade.' if atr_v > 80 else 'Volatilidade baixa - risco de spread.'}\n"
                analise_completa += f"   Stop calculado: {stop_pts}pts (1x ATR, max 300 WIN). Alvo: {alvo_pts}pts (1.5x stop).\n"
                
                if suporte and resistencia:
                    analise_completa += f"\n5. SUPORTE/RESISTENCIA (Murphy):\n"
                    analise_completa += f"   Suporte: {round(suporte,0)} | Resistencia: {round(resistencia,0)}\n"
                    dist_s = abs(c - suporte)
                    dist_r = abs(resistencia - c)
                    analise_completa += f"   Distancia do suporte: {round(dist_s,0)}pts | Distancia da resistencia: {round(dist_r,0)}pts\n"
                    if tipo_sinal == "COMPRA":
                        analise_completa += f"   COMPRA proximo suporte = {'BOM (risco/retorno favoravel)' if dist_s < atr_v else 'Preco longe do suporte'}.\n"
                    else:
                        analise_completa += f"   VENDA proximo resistencia = {'BOM (risco/retorno favoravel)' if dist_r < atr_v else 'Preco longe da resistencia'}.\n"
                
                if vwap:
                    analise_completa += f"\n6. VWAP (Preco Medio Ponderado): {round(vwap,0)}\n"
                    analise_completa += f"   Preco {'ACIMA' if c > vwap else 'ABAIXO'} do VWAP ({round(c - vwap, 0)}pts). "
                    if (c > vwap and tipo_sinal == "COMPRA") or (c < vwap and tipo_sinal == "VENDA"):
                        analise_completa += "CONFIRMADO: preco na direcao correta do VWAP.\n"
                    else:
                        analise_completa += "ATENCAO: preco contra o VWAP.\n"
                
                if fib_level:
                    analise_completa += f"\n7. FIBONACCI: Proximo do nivel {fib_level}\n"
                    analise_completa += f"   Retracao do swing recente. Zona de alta probabilidade de reacao.\n"
                else:
                    analise_completa += f"\n7. FIBONACCI: Preco longe de niveis significativos (38.2%, 50%, 61.8%)\n"
                
                analise_completa += f"\n8. CANDLESTICK PATTERNS: "
                found_pattern = False
                for m in motivos_operar:
                    if "Candlestick" in m:
                        analise_completa += m + "\n"
                        found_pattern = True
                if not found_pattern:
                    analise_completa += "Nenhum padrao relevante nesta vela.\n"
                
                # SMC (Smart Money Concepts)
                analise_completa += f"\n9. SMART MONEY CONCEPTS (SMC):\n"
                if smc_data.get("fvg"):
                    analise_completa += f"   FVG: {smc_data['fvg']['detalhe']}\n"
                if smc_data.get("liquidity_sweep"):
                    analise_completa += f"   LIQUIDITY SWEEP: {smc_data['liquidity_sweep']['detalhe']}\n"
                if smc_data.get("order_block"):
                    analise_completa += f"   ORDER BLOCK: {smc_data['order_block']['detalhe']}\n"
                if smc_data.get("estrutura"):
                    analise_completa += f"   ESTRUTURA: {smc_data['estrutura']['detalhe']}\n"
                if not smc_data:
                    analise_completa += "   Nenhum padrao SMC detectado nesta vela.\n"
                
                analise_completa += f"\n10. HORARIO: {hora} - {'Horario FORTE (abertura/volatilidade alta)' if bom_horario else 'Horario fraco/almoco'}\n"
                
                analise_completa += f"\n11. SCORE CONFLUENCIA: {score}/15\n"
                analise_completa += f"    Confianca: {conf_label} ({confianca}/5)\n"
                analise_completa += f"    Fatores A FAVOR: {', '.join(motivos_operar) if motivos_operar else 'nenhum'}\n"
                analise_completa += f"    Fatores CONTRA: {', '.join(motivos_nao_operar) if motivos_nao_operar else 'nenhum'}\n"
                
                analise_completa += f"\nDECISAO: {tipo_sinal} em {round(c,2)}. Stop={stop_price} Alvo={alvo_price} RR=1:{round(alvo_pts/stop_pts,1)}\n"
                
                # ---- DETALHES VITORIA (para WINs) ----
                detalhes_vitoria = ""
                if resultado == "WIN":
                    detalhes_vitoria = f"Alvo atingido em {hora_saida} (+{round(abs(pts),1)}pts = R${round(abs(rs),2)}). "
                    if tipo_sinal == "COMPRA":
                        detalhes_vitoria += f"Preco subiu de {round(c,2)} ate {round(alvo_price,2)}. "
                        if tend == "ALTA":
                            detalhes_vitoria += "Tendencia ALTA confirmou a direcao - Triple Screen alinhado. "
                        if macd_h > 0:
                            detalhes_vitoria += f"MACD positivo ({macd_h}) deu momentum. "
                        if rsi_v < 60:
                            detalhes_vitoria += f"RSI {rsi_v} tinha espaco pra subir. "
                    else:
                        detalhes_vitoria += f"Preco caiu de {round(c,2)} ate {round(alvo_price,2)}. "
                        if tend == "BAIXA":
                            detalhes_vitoria += "Tendencia BAIXA confirmou a direcao - Triple Screen alinhado. "
                        if macd_h < 0:
                            detalhes_vitoria += f"MACD negativo ({macd_h}) deu momentum vendedor. "
                    detalhes_vitoria += f"Score {score}/11 ({conf_label}) - setup de alta confluencia. "
                    if velas_na_op <= 6:
                        detalhes_vitoria += f"Operacao rapida ({velas_na_op} velas = {velas_na_op*5}min) - mercado ja estava no ponto. "
                    detalhes_vitoria += "Licao: Setups com alta confluencia e a favor da tendencia tem maior taxa de acerto."
                
                operacoes_recomendadas.append({
                    "tipo": tipo_sinal,
                    "hora_entrada": hora,
                    "preco_entrada": round(c, 2),
                    "stop_loss": stop_price,
                    "take_profit": alvo_price,
                    "stop_pts": stop_pts,
                    "alvo_pts": alvo_pts,
                    "rr": f"1:{round(alvo_pts/stop_pts, 1)}",
                    "hora_saida": hora_saida,
                    "preco_saida": round(preco_saida, 2),
                    "resultado": resultado,
                    "pts": pts,
                    "resultado_rs": rs,
                    "velas_na_op": velas_na_op,
                    "score": score,
                    "confianca": confianca,
                    "conf_label": conf_label,
                    "motivos": motivos_operar,
                    "detalhes_perda": detalhes_perda,
                    "detalhes_vitoria": detalhes_vitoria,
                    "analise_completa": analise_completa,
                    "suporte": round(suporte, 0) if suporte else None,
                    "resistencia": round(resistencia, 0) if resistencia else None,
                    "vwap": round(vwap, 0) if vwap else None,
                    "fib_level": fib_level,
                    "rsi": rsi_v,
                    "macd_hist": macd_h,
                    "ema9": ema9,
                    "ema21": ema21,
                    "atr": atr_v,
                    "tendencia": tend,
                })
                
                # PRO TRADER: atualizar controles de risco
                total_pts_dia += pts
                if resultado == "LOSS":
                    losses_consecutivos += 1
                    # Apos loss, aumentar cooldown (Elder: pausa apos perda)
                    cooldown_velas = 4 if losses_consecutivos >= 2 else 3
                else:
                    losses_consecutivos = 0  # Reset em WIN
                    cooldown_velas = 2  # Cooldown normal
                
                # Verificar limite de perda diaria
                if total_pts_dia <= LOSS_LIMIT_PTS:
                    dia_bloqueado = True
                
                # Block next entries: cooldown adaptativo
                posicao_aberta = {"close_idx": future_start + velas_na_op + cooldown_velas}
        
        # ---- RESUMO DO DIA ----
        first_v = velas_analisadas[0] if velas_analisadas else {}
        last_v = velas_analisadas[-1] if velas_analisadas else {}
        abertura = first_v.get("open", 0)
        fechamento = last_v.get("close", 0)
        variacao = round((fechamento / abertura - 1) * 100, 2) if abertura else 0
        high_dia = max(v["high"] for v in velas_analisadas) if velas_analisadas else 0
        low_dia = min(v["low"] for v in velas_analisadas) if velas_analisadas else 0
        amplitude = round(high_dia - low_dia, 0)
        
        # TODAS as oportunidades (score >= 7) - mostra setups com potencial
        oportunidades = [v for v in velas_analisadas if v["score"] >= 7 and v["tipo_sinal"] is not None]
        
        total_ops = len(operacoes_recomendadas)
        wins = sum(1 for op in operacoes_recomendadas if op["resultado"] == "WIN")
        losses = total_ops - wins
        win_rate = round(wins / total_ops * 100) if total_ops > 0 else 0
        total_pts = sum(op["pts"] for op in operacoes_recomendadas)
        total_rs = sum(op["resultado_rs"] for op in operacoes_recomendadas)
        
        velas_operar = sum(1 for v in velas_analisadas if v["decisao"] == "OPERAR")
        velas_nao = sum(1 for v in velas_analisadas if v["decisao"] == "NAO OPERAR")
        
        # ---- APRENDIZADO: registrar sessao e evoluir ----
        if operacoes_recomendadas:
            try:
                learning_data = registrar_sessao(
                    ativo, dia_anterior.strftime("%d/%m/%Y"),
                    operacoes_recomendadas,
                    {"win_rate": win_rate, "total_pts": total_pts}
                )
            except Exception as e:
                logger.error(f"Erro registrando aprendizado: {e}")
        
        # Dados de aprendizado para exibição
        aprendizado = obter_resumo_aprendizado()
        
        return JSONResponse({
            "dia": dia_anterior.strftime("%d/%m/%Y"),
            "ativo": ativo,
            "contrato": contrato_nome,
            "valor_ponto": valor_ponto,
            "aprendizado": aprendizado,
            "resumo": {
                "abertura": abertura,
                "fechamento": fechamento,
                "variacao_pct": variacao,
                "direcao": "ALTA" if variacao > 0 else "BAIXA" if variacao < 0 else "NEUTRO",
                "high": high_dia,
                "low": low_dia,
                "amplitude_pts": amplitude,
                "total_velas": len(velas_analisadas),
                "velas_operar": velas_operar,
                "velas_nao_operar": velas_nao,
            },
            "performance": {
                "total_operacoes": total_ops,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total_pts": round(total_pts, 1),
                "total_rs": round(total_rs, 2),
                "max_ops_dia": MAX_OPS_DIA,
                "losses_limit": MAX_LOSSES_CONSECUTIVOS,
                "dia_bloqueado": dia_bloqueado,
                "motivo_parada": "Limite de perda atingido" if dia_bloqueado else ("Parou apos " + str(MAX_LOSSES_CONSECUTIVOS) + " losses consecutivos" if losses_consecutivos >= MAX_LOSSES_CONSECUTIVOS else ""),
            },
            "operacoes": operacoes_recomendadas,
            "oportunidades": oportunidades,
            "total_oportunidades": len(oportunidades),
            "velas": velas_analisadas,
            "timestamp": datetime.now(BRT_tz).strftime("%H:%M:%S"),
        })
    except Exception as e:
        logger.error(f"Erro simulador-real: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"erro": str(e)}, status_code=500)

@app.get("/api/livros")
async def get_livros():
    """Lista todos os livros estudados pela AI"""
    try:
        livros = obter_livros_lista()
        conceitos = obter_todos_conceitos()
        return JSONResponse({
            "livros": livros,
            "total_conceitos": len(conceitos),
            "conceitos_amostra": conceitos[:20],  # Primeiros 20
        })
    except Exception as e:
        return JSONResponse({"erro": str(e)}, status_code=500)


@app.get("/api/learning")
async def get_learning():
    """Retorna dados de aprendizado da AI"""
    try:
        resumo = obter_resumo_aprendizado()
        return JSONResponse(resumo)
    except Exception as e:
        return JSONResponse({"erro": str(e)}, status_code=500)


@app.get("/api/heatmap")
async def get_heatmap(ativo: str = Query("WIN")):
    """Heatmap de melhores horarios para operar - baseado em volatilidade historica"""
    ativo = ativo.upper()
    try:
        contrato = obter_contrato_vigente(ativo)
        ticker = contrato.get("ticker_b3", "^BVSP" if ativo == "WIN" else "BRL=X")
        
        # Get 5 days of 5min data
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty:
            df = yf.download("^BVSP" if ativo == "WIN" else "BRL=X", period="5d", interval="5m", progress=False)
        
        if df.empty:
            # Return static heatmap based on known B3 patterns
            heatmap = []
            for h in range(9, 18):
                for m in [0, 30]:
                    hora = f"{h:02d}:{m:02d}"
                    # Known B3 patterns: opening and closing are most volatile
                    if h == 9 and m == 0:
                        score = 95
                    elif h == 9 and m == 30:
                        score = 85
                    elif h == 10 and m == 0:
                        score = 70
                    elif h >= 11 and h <= 13:
                        score = 40 + random.randint(0, 15)
                    elif h == 14:
                        score = 55
                    elif h == 15:
                        score = 65
                    elif h == 16:
                        score = 75
                    elif h == 17:
                        score = 90
                    else:
                        score = 50
                    heatmap.append({"hora": hora, "volatilidade": score, "qualidade": "OTIMO" if score >= 70 else "BOM" if score >= 50 else "FRACO"})
            return JSONResponse({"heatmap": heatmap, "ativo": ativo, "fonte": "padrao_b3"})
        
        # Flatten columns if MultiIndex
        if hasattr(df.columns, 'levels'):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        
        df["hora"] = df.index.strftime("%H:%M")
        df["amplitude"] = df["High"] - df["Low"]
        
        # Group by 30min slots
        df["slot"] = df.index.hour * 2 + (df.index.minute >= 30).astype(int)
        stats = df.groupby("slot").agg({"amplitude": ["mean", "count"]})
        stats.columns = ["amp_media", "count"]
        
        max_amp = stats["amp_media"].max() if stats["amp_media"].max() > 0 else 1
        
        heatmap = []
        for slot_idx, row in stats.iterrows():
            h = slot_idx // 2
            m = (slot_idx % 2) * 30
            if h < 9 or h >= 18:
                continue
            score = min(round(row["amp_media"] / max_amp * 100), 100)
            heatmap.append({
                "hora": f"{h:02d}:{m:02d}",
                "volatilidade": score,
                "amplitude_media": round(float(row["amp_media"]), 2),
                "amostras": int(row["count"]),
                "qualidade": "OTIMO" if score >= 70 else "BOM" if score >= 50 else "FRACO",
            })
        
        return JSONResponse({"heatmap": heatmap, "ativo": ativo, "fonte": "yfinance_5d"})
    except Exception as e:
        logger.error(f"Erro heatmap: {e}")
        # Return static fallback
        heatmap = []
        for h in range(9, 18):
            for m in [0, 30]:
                score = 90 if h in [9, 17] else 70 if h in [10, 16] else 50
                heatmap.append({"hora": f"{h:02d}:{m:02d}", "volatilidade": score, "qualidade": "OTIMO" if score >= 70 else "BOM" if score >= 50 else "FRACO"})
        return JSONResponse({"heatmap": heatmap, "ativo": ativo, "fonte": "fallback"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
