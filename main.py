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
import pandas as pd
import traceback

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
        dados_5m = await data_provider.obter_dados(ativo_upper, "5m")
        dados_15m = await data_provider.obter_dados(ativo_upper, "15m")
        dados_1h = await data_provider.obter_dados(ativo_upper, "1h")

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
                if preco > vwap_v and ema9 > ema21:
                    tend = "ALTA"
                elif preco < vwap_v and ema9 < ema21:
                    tend = "BAIXA"
                else:
                    tend = "LATERAL"
                if lat.get("lateral"):
                    tend = "LATERAL"
                pb = detectar_pullback(window, tend)
                sinais = gerar_sinais(window, fib, rsi_v, mv, msv, mhv,
                    vol, viol, tendencia=tend, pullback_info=pb,
                    lateralizacao=lat, vwap_atual=vwap_v)
                return {
                    "tendencia": tend,
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

                if a5 and a5["sinais"]:
                    # Confirm with 15m and 1h
                    w15 = get_window(dados_15m, ts)
                    w1h = get_window(dados_1h, ts)
                    a15 = analisar_tf(w15) if w15 is not None else None
                    a1h = analisar_tf(w1h) if w1h is not None else None

                    s5 = a5["sinais"][0]
                    tipo_5m = s5.tipo

                    # Confluencia: 15m e 1h devem ter mesma tendencia
                    conf_15m = False
                    conf_1h = False
                    motivo_conf = []

                    if a15:
                        if tipo_5m == "COMPRA" and a15["tendencia"] in ("ALTA",):
                            conf_15m = True
                            motivo_conf.append(f"15m: Tendencia {a15['tendencia']} | RSI {a15['rsi']}")
                        elif tipo_5m == "VENDA" and a15["tendencia"] in ("BAIXA",):
                            conf_15m = True
                            motivo_conf.append(f"15m: Tendencia {a15['tendencia']} | RSI {a15['rsi']}")
                        elif a15["tendencia"] == "LATERAL":
                            conf_15m = True  # Lateral nao contradiz
                            motivo_conf.append(f"15m: Lateral (nao contradiz)")
                    else:
                        conf_15m = True  # Sem dados = nao bloqueia
                        motivo_conf.append("15m: Sem dados suficientes")

                    if a1h:
                        if tipo_5m == "COMPRA" and a1h["tendencia"] in ("ALTA", "LATERAL"):
                            conf_1h = True
                            motivo_conf.append(f"1h: Tendencia {a1h['tendencia']} | RSI {a1h['rsi']}")
                        elif tipo_5m == "VENDA" and a1h["tendencia"] in ("BAIXA", "LATERAL"):
                            conf_1h = True
                            motivo_conf.append(f"1h: Tendencia {a1h['tendencia']} | RSI {a1h['rsi']}")
                    else:
                        conf_1h = True
                        motivo_conf.append("1h: Sem dados suficientes")

                    # So entra se ambos confirmam
                    if conf_15m and conf_1h:
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
            "contrato": data_provider.get_contrato_info(ativo_upper).get("ticker_b3", ativo_upper),
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
