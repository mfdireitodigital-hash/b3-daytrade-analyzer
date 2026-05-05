"""
# Build: 1777916428
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
    carregar_learning, registrar_sessao, registrar_trade_replay, obter_pesos_atuais,
    obter_score_minimo, obter_resumo_aprendizado, registrar_livro, consultar_memoria,
    registrar_historico_completo, obter_historico
)
from trading_books_knowledge import (
    aplicar_scoring_avancado, obter_livros_lista, obter_todos_conceitos
)
from smc_engine import aplicar_smc_scoring
from pro_trader_analysis import calcular_tendencia_macro, detectar_setup_profissional, gerar_analise_completa
from news_impact import avaliar_impacto_noticias, obter_noticias_do_dia

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
    # Operador Senior LIVE state
    "operador_live": {
        "WIN": {"operacoes": [], "total_pts": 0, "losses_consecutivos": 0, "dia_bloqueado": False, "ultimo_trade_hora": None, "cooldown_ate": None, "dia": None, "trade_ativo": None, "aguardando_entrada": False},
        "WDO": {"operacoes": [], "total_pts": 0, "losses_consecutivos": 0, "dia_bloqueado": False, "ultimo_trade_hora": None, "cooldown_ate": None, "dia": None, "trade_ativo": None, "aguardando_entrada": False},
    },
    # Memória de erros do operador (persiste entre dias)
    "operador_erros": [],  # [{tecnica, condicoes, motivo_erro, data, ativo}]
}


# ===== PERSISTÊNCIA DO ESTADO DO OPERADOR =====
STATE_FILE = Path("/tmp/operador_state.json")

def salvar_estado_operador():
    """Salva estado do operador em arquivo para sobreviver restarts"""
    try:
        state_to_save = {
            "WIN": dict(app_state["operador_live"]["WIN"]),
            "WDO": dict(app_state["operador_live"]["WDO"]),
            "timestamp": datetime.now(TZ_BR).isoformat(),
        }
        STATE_FILE.write_text(json.dumps(state_to_save, default=str, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Erro ao salvar estado: {e}")

def carregar_estado_operador():
    """Carrega estado do operador de arquivo"""
    try:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            hoje = str(datetime.now(TZ_BR).date())
            for ativo in ["WIN", "WDO"]:
                if ativo in data and data[ativo].get("dia") == hoje:
                    # Mesmo dia - restaurar estado completo
                    app_state["operador_live"][ativo].update(data[ativo])
                    logger.info(f"Estado restaurado para {ativo}: {len(data[ativo].get('operacoes',[]))} ops, trade_ativo={bool(data[ativo].get('trade_ativo'))}")
                else:
                    logger.info(f"Estado de {ativo} é de outro dia, ignorando")
    except Exception as e:
        logger.error(f"Erro ao carregar estado: {e}")

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

    # Restaurar estado do operador (sobrevive restart/F5)
    carregar_estado_operador()
    
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


@app.get("/api/version")
async def api_version():
    return {"version": "3.7.3", "build": "20260505f", "changes": "memoria_inteligente_bloqueio,aprender_erros,regras_auto"}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    resp = templates.TemplateResponse("dashboard.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


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
        
        # Quando mercado aberto e preço RT disponível, enriquecer análise com dados live
        if mercado_aberto() and response_data.get("preco_realtime"):
            rt = response_data["preco_realtime"]
            rt_preco = rt.get("preco", 0)
            if rt_preco and isinstance(analise, dict) and analise.get("preco_atual"):
                # Se preço RT diverge >0.5% do yfinance, dados estão defasados
                yf_preco = analise.get("preco_atual", 0)
                if yf_preco and abs(rt_preco - yf_preco) / yf_preco > 0.005:
                    analise["preco_atual"] = rt_preco
                    analise["_dados_defasados"] = True
                    analise["_fonte_preco"] = rt.get("fonte", "TradingView")
                    # Usar variação RT
                    if rt.get("variacao"):
                        analise["variacao_pct"] = rt["variacao"]
                    if rt.get("variacao_pts"):
                        analise["variacao"] = rt["variacao_pts"]
                    if rt.get("high"):
                        analise["high_dia"] = rt["high"]
                    if rt.get("low"):
                        analise["low_dia"] = rt["low"]
                    if rt.get("open"):
                        analise["abertura"] = rt["open"]
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
    
    # Se painel vazio e mercado aberto, forçar análise on-the-fly
    if (not painel or all("erro" in painel.get(tf, {}) for tf in TIMEFRAMES)) and mercado_aberto():
        try:
            provider = app_state["provider"]
            painel = {}
            for tf in TIMEFRAMES:
                try:
                    dados = await provider.obter_dados(ativo, tf)
                    if dados is not None and len(dados) >= 20:
                        analise = analisar_completo(dados, tf, ativo)
                        analise = converter_numpy(analise)
                        painel[tf] = analise
                except Exception as e:
                    painel[tf] = {"erro": str(e)}
            # Update app_state
            if ativo not in app_state["analises"]:
                app_state["analises"][ativo] = {}
            app_state["analises"][ativo].update(painel)
            logger.info(f"Painel {ativo}: análise on-the-fly atualizada")
        except Exception as e:
            logger.error(f"Erro painel on-the-fly {ativo}: {e}")
    
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
                # Verify this day actually has market-hours data
                test_mask = (dados.index.date == d) & (dados.index.hour >= 9) & (dados.index.hour < 18)
                if dados[test_mask].shape[0] >= 5:  # At least 5 candles in market hours
                    dia_anterior = d
                    break
        if not dia_anterior:
            return JSONResponse({"erro": "Nenhum dia util com dados encontrado nos ultimos 5 dias"})
        
        # Get day data
        day_mask = (dados.index.date == dia_anterior) & (dados.index.hour >= 9) & (dados.index.hour < 18)
        day_data = dados[day_mask]
        
        # Calculate indicators for each candle
        from analysis_engine import calcular_rsi, calcular_macd, calcular_atr_series
        import math
        
        velas = []
        all_indices = list(range(len(dados)))
        day_indices = [i for i, d in enumerate(dados.index.date) if d == dia_anterior and 9 <= dados.index[i].hour < 18]
        
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
        mercado["variacao_pct"] = round((mercado["fechamento"] / mercado["abertura"] - 1) * 100, 2) if mercado["abertura"] != 0 else 0
        
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
                test_mask = (dados.index.date == d) & (dados.index.hour >= 9) & (dados.index.hour < 18)
                if dados[test_mask].shape[0] >= 5:
                    dia_anterior = d
                    break
        if not dia_anterior:
            return JSONResponse({"erro": "Nenhum dia util com dados encontrado"})
        
        day_indices = [i for i, d in enumerate(dados.index.date) if d == dia_anterior and 9 <= dados.index[i].hour < 18]
        
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
        
        # ===== ANALISE EDUCATIVA =====
        from analysis_engine import calcular_rsi
        educativa = []
        try:
            # Contexto da vela de entrada
            entry_rsi = None
            rsi_series = calcular_rsi(dados, 14)
            if entry_global_idx < len(rsi_series):
                entry_rsi = round(float(rsi_series.iloc[entry_global_idx]), 1)
            
            # Tendencia na entrada
            ema9 = dados['close'].ewm(span=9).mean()
            ema21 = dados['close'].ewm(span=21).mean()
            e9 = float(ema9.iloc[entry_global_idx])
            e21 = float(ema21.iloc[entry_global_idx])
            tend = "ALTA" if e9 > e21 else "BAIXA" if e9 < e21 else "LATERAL"
            
            # A favor ou contra tendencia?
            a_favor = (tipo == "COMPRA" and tend == "ALTA") or (tipo == "VENDA" and tend == "BAIXA")
            
            if resultado == "WIN":
                educativa.append(f"Operacao a {'FAVOR' if a_favor else 'CONTRA'} da tendencia ({tend})")
                if a_favor:
                    educativa.append("Regra aplicada: Operar a favor da tendencia macro (Elder - Triple Screen)")
                else:
                    educativa.append("Atencao: Ganhou contra a tendencia - isso e mais arriscado e menos consistente")
                if entry_rsi:
                    if tipo == "COMPRA" and entry_rsi < 40:
                        educativa.append(f"RSI estava em {entry_rsi} (sobrevendido) - bom ponto de compra")
                    elif tipo == "VENDA" and entry_rsi > 60:
                        educativa.append(f"RSI estava em {entry_rsi} (sobrecomprado) - bom ponto de venda")
                    else:
                        educativa.append(f"RSI na entrada: {entry_rsi}")
                educativa.append(f"EMA9 {'acima' if e9>e21 else 'abaixo'} da EMA21 - confirmacao de tendencia")
                educativa.append(f"R:R configurado {rr}:1 - {'excelente' if rr >= 2 else 'aceitavel' if rr >= 1.5 else 'baixo, busque minimo 1:2'}")
                if max_adverso > 0:
                    educativa.append(f"Maximo adverso: {round(max_adverso,0)}pts - {'operacao tranquila' if max_adverso < stop_pts*0.5 else 'chegou perto do stop' if max_adverso > stop_pts*0.7 else 'oscilou mas segurou'}")
            else:
                educativa.append(f"Operacao a {'FAVOR' if a_favor else 'CONTRA'} da tendencia ({tend})")
                if not a_favor:
                    educativa.append("LICAO: Entrou CONTRA a tendencia - principal causa de loss (Elder: so opere na direcao da Tela 1)")
                if entry_rsi:
                    if tipo == "COMPRA" and entry_rsi > 70:
                        educativa.append(f"RSI estava em {entry_rsi} - SOBRECOMPRADO! Nao compre com RSI alto (divergencia)")
                    elif tipo == "VENDA" and entry_rsi < 30:
                        educativa.append(f"RSI estava em {entry_rsi} - SOBREVENDIDO! Nao venda com RSI baixo")
                    else:
                        educativa.append(f"RSI na entrada: {entry_rsi}")
                educativa.append(f"Stop de {stop_pts}pts foi {'adequado' if stop_pts >= round(max_adverso,0) else 'curto demais - considere stop maior baseado no ATR'}")
                if max_favoravel > 0:
                    educativa.append(f"Chegou a ter {round(max_favoravel,0)}pts a favor antes de stopar - {'considere parcial ou trailing stop' if max_favoravel > alvo_pts*0.5 else 'nao teve forca na direcao'}")
                educativa.append(f"Axioma de Zurique #3: Corte perdas rapido. Stop bem posicionado protege o capital.")
        except Exception as ex:
            logger.error(f"Erro na analise educativa: {ex}"); educativa.append(f"Analise educativa indisponivel: {str(ex)[:100]}")
        
        # ===== GRAVAR NA MEMORIA PERSISTENTE =====
        memoria_resultado = None
        alerta_memoria = None
        try:
            # Montar objeto de operação para learning_engine
            op_learning = {
                "tipo": tipo,
                "hora_entrada": hora_entrada,
                "resultado": resultado,
                "pts": pts,
                "resultado_rs": resultado_rs,
                "tendencia": tend if 'tend' in locals() else "LATERAL",
                "rsi": entry_rsi if 'entry_rsi' in locals() and entry_rsi else 50,
                "macd_hist": 0,
                "score": rr,  # usar R:R como proxy de qualidade
                "conf_label": f"R:R {rr}:1",
                "motivos": educativa[:3] if educativa else [],
                "detalhes_perda": "; ".join(educativa) if resultado == "LOSS" else "",
            }
            
            # Consultar memória ANTES (alertas de erros passados similares)
            try:
                alerta_check = consultar_memoria(op_learning)
                if alerta_check and alerta_check.get("tem_alerta"):
                    alerta_memoria = alerta_check
            except Exception as mem_err:
                logger.error(f"Erro consultando memoria: {mem_err}")
            
            # Gravar trade na memória
            memoria_resultado = registrar_trade_replay(ativo, op_learning)
            logger.info(f"Trade replay gravado na memoria: {resultado} {pts}pts - total ops: {memoria_resultado.get('total_operacoes')}")
        except Exception as learn_err:
            logger.error(f"Erro gravando na memoria: {learn_err}")
        
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
            "educativa": educativa,
            "contratos": contratos,
            "valor_ponto": valor_ponto,
            "memoria": memoria_resultado,
            "alerta_memoria": alerta_memoria,
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
        if cs >= 6: escala = {"nota": 5, "label": "A+ SETUP", "acao": "Tamanho cheio - confluência máxima", "cor": "#22c55e"}
        elif cs >= 5: escala = {"nota": 4, "label": "SETUP BOM", "acao": "Tamanho normal - boa confluência", "cor": "#16a34a"}
        elif cs == 4: escala = {"nota": 3, "label": "SETUP OK", "acao": "Pode operar - sizing reduzido", "cor": "#ca8a04"}
        elif cs == 3: escala = {"nota": 2, "label": "VIÁVEL", "acao": "Entrada cautelosa - 3 fatores", "cor": "#f59e0b"}
        elif cs == 2: escala = {"nota": 2, "label": "POSSÍVEL", "acao": "Entrada possível - sizing mínimo", "cor": "#ea580c"}
        elif cs == 1: escala = {"nota": 1, "label": "ARRISCADO", "acao": "Alto risco - 1 fator apenas", "cor": "#ef4444"}
        else: escala = {"nota": 0, "label": "SEM BASE", "acao": "Nenhum fator confirmando", "cor": "#ef4444"}
        
        # === GERAR SINAIS ===
        sinais_gerados = []
        direcao_final = tela3_entrada if tela3_entrada != "NEUTRO" else tela2_sinal
        
        # Sempre dar parecer se tiver direção, mesmo com poucos fatores
        if direcao_final in ("COMPRA", "VENDA") and cs >= 1:
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
                "motivos": ["Sem direção clara - indicadores divergentes"],
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
# Fonte: Profit API Economic Calendar + Investing.com fallback
# =====================================================

# Cache global de noticias (evita rate limit da API)
_noticias_cache = {"data": None, "timestamp": None, "ttl": 1800}  # 30min cache

async def _buscar_noticias_profit():
    """Busca eventos da Profit API Economic Calendar"""
    token = os.getenv("PROFIT_API_TOKEN", "")
    if not token:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.profit.com/data-api/economic_calendar/forex",
                params={"token": token, "limit": 50}
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "message" in data:
                    logger.warning(f"Profit Calendar: {data['message']}")
                    return []
            return []
    except Exception as e:
        logger.error(f"Erro Profit Calendar: {e}")
        return []

def _buscar_noticias_forexfactory():
    """Busca eventos via ForexFactory JSON API (fallback confiavel, sem scraping)"""
    try:
        import urllib.request
        from datetime import timezone, timedelta
        BRT = timezone(timedelta(hours=-3))
        
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        
        if not isinstance(data, list):
            return []
        
        eventos = []
        for evt in data:
            country = evt.get("country", "")
            if country not in ("USD", "BRL"):
                continue
            
            impact = evt.get("impact", "")
            if impact in ("Holiday", "Non-Economic"):
                continue
            
            title = evt.get("title", "")
            if not title:
                continue
            
            # Parse date: "2026-05-01T09:30:00-04:00"
            date_str = evt.get("date", "")
            evt_time = ""
            evt_dt = ""
            try:
                if "T" in date_str:
                    dt_part = date_str[:19]
                    tz_part = date_str[19:]
                    base_dt = datetime.strptime(dt_part, "%Y-%m-%dT%H:%M:%S")
                    if tz_part and tz_part != "Z":
                        sign = 1 if tz_part[0] == "+" else -1
                        tz_h = int(tz_part[1:3])
                        tz_m = int(tz_part[4:6]) if len(tz_part) > 4 else 0
                        offset = timedelta(hours=tz_h, minutes=tz_m) * sign
                        base_dt = base_dt.replace(tzinfo=timezone(offset))
                    else:
                        base_dt = base_dt.replace(tzinfo=timezone.utc)
                    base_dt = base_dt.astimezone(BRT)
                    evt_time = base_dt.strftime("%H:%M")
                    evt_dt = base_dt.strftime("%Y/%m/%d %H:%M:%S")
                else:
                    continue
            except Exception:
                continue
            
            pais = "EUA" if country == "USD" else "Brasil"
            impacto_map = {"High": 3, "Medium": 2, "Low": 1}
            
            eventos.append({
                "id": f"ff_{hash(title + date_str) % 100000}",
                "datetime": evt_dt,
                "hora": evt_time,
                "pais": pais,
                "moeda": country,
                "evento": title,
                "actual": "",
                "previsao": evt.get("forecast", ""),
                "anterior": evt.get("previous", ""),
                "impacto_nivel": impacto_map.get(impact, 1),
                "fonte": "forexfactory",
            })
        
        logger.info(f"ForexFactory: {len(eventos)} eventos USD/BRL")
        return eventos
    except Exception as e:
        logger.error(f"Erro ForexFactory: {e}")
        return []

def _processar_evento(evt, agora, fonte="profit"):
    """Processa um evento raw em formato padrao para o frontend"""
    import re as _re
    from datetime import timezone, timedelta
    BRT = timezone(timedelta(hours=-3))
    
    if fonte == "profit":
        # Profit API format: {name, time (unix), impact, currency, actual, estimate, previous, country_iso}
        ts = evt.get("time", 0)
        try:
            evt_datetime = datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc).astimezone(BRT)
        except:
            return None
        
        evt_name = evt.get("name", "")
        currency = evt.get("currency", "")
        impact = evt.get("impact", "low")
        actual = str(evt.get("actual", "")) if evt.get("actual") is not None else ""
        forecast = str(evt.get("estimate", "")) if evt.get("estimate") is not None else ""
        previous = str(evt.get("previous", "")) if evt.get("previous") is not None else ""
        country = evt.get("country_iso", "")
        evt_time = evt_datetime.strftime("%H:%M")
        evt_dt = evt_datetime.strftime("%Y/%m/%d %H:%M:%S")
        eid = f"p_{ts}_{currency}"
        
        # Filtrar: apenas high impact OU USD/BRL
        if impact.lower() not in ("high",) and currency not in ("USD", "BRL"):
            return None
    else:
        # ForexFactory / Investing format (already parsed)
        evt_name = evt.get("evento", "")
        currency = evt.get("moeda", "")
        actual = evt.get("actual", "")
        forecast = evt.get("previsao", "")
        previous = evt.get("anterior", "")
        country = evt.get("pais", "")
        evt_time = evt.get("hora", "")
        evt_dt = evt.get("datetime", "")
        eid = evt.get("id", "0")
        
        try:
            if evt_dt:
                evt_datetime = datetime.strptime(evt_dt, "%Y/%m/%d %H:%M:%S").replace(tzinfo=BRT)
            elif evt_time and ":" in evt_time:
                h, m = map(int, evt_time.split(":"))
                evt_datetime = agora.replace(hour=h, minute=m, second=0)
            else:
                return None
        except:
            return None
    
    # Filter: only USD and BRL
    if currency not in ("USD", "BRL"):
        return None
    
    # Calculate time difference
    delta = (evt_datetime - agora).total_seconds()
    minutos_restantes = round(delta / 60)
    ja_passou = delta < 0
    
    # Zone calculation
    zona_impacto = ""
    if -15 <= minutos_restantes <= 15:
        zona_impacto = "ZONA DE IMPACTO"
    elif 0 < minutos_restantes <= 30:
        zona_impacto = "APROXIMANDO"
    elif minutos_restantes > 30:
        zona_impacto = "AGUARDANDO"
    elif minutos_restantes < -15:
        zona_impacto = "ENCERRADO"
    
    # Impact analysis
    evt_lower = evt_name.lower()
    impacto_win = ""
    impacto_wdo = ""
    surpresa = ""
    operavel = "BOM"
    
    # Check if critical event
    criticos = ["payroll", "nonfarm", "non-farm", "fomc", "fed fund", "taxa de juros", "interest rate", "cpi", "ipc", "selic", "copom", "gdp", "pib"]
    altos = ["pmi", "ism", "emprego", "employment", "unemployment", "vendas no varejo", "retail sales", "producao industrial", "ipca", "inflacao"]
    
    is_critico = any(kw in evt_lower for kw in criticos)
    is_alto = any(kw in evt_lower for kw in altos)
    
    if currency == "USD":
        if is_critico:
            operavel = "CAUTELA"
            impacto_win = "ALTO IMPACTO potencial no indice"
            impacto_wdo = "ALTO IMPACTO potencial no dolar"
        elif is_alto:
            impacto_win = "Impacto moderado no indice"
            impacto_wdo = "Impacto moderado no dolar"
        else:
            impacto_win = "Baixo impacto no indice"
            impacto_wdo = "Baixo impacto no dolar"
    elif currency == "BRL":
        if is_critico:
            operavel = "CAUTELA"
            impacto_win = "ALTO IMPACTO no Ibovespa"
            impacto_wdo = "ALTO IMPACTO no dolar"
        elif is_alto:
            impacto_win = "Impacto moderado no indice"
            impacto_wdo = "Impacto moderado no dolar"
        else:
            impacto_win = "Baixo impacto no indice"
            impacto_wdo = "Baixo impacto no dolar"
    
    # Surpresa analysis
    if actual and forecast:
        try:
            act_num = float(str(actual).replace("%","").replace(",",".").replace("K","000").replace("M","000000").strip())
            for_num = float(str(forecast).replace("%","").replace(",",".").replace("K","000").replace("M","000000").strip())
            if act_num > for_num:
                surpresa = "ACIMA"
                if currency == "USD":
                    impacto_wdo = "ALTA (dolar fortalece)"
                    impacto_win = "BAIXA (pressao no ibov)"
                    if "nonfarm" in evt_lower or "payroll" in evt_lower:
                        impacto_wdo = "ALTA FORTE (USD fortalece)"
                        impacto_win = "BAIXA FORTE (risk-off)"
                elif currency == "BRL":
                    if "selic" in evt_lower or "copom" in evt_lower:
                        impacto_win = "BAIXA (juros apertam)"
                        impacto_wdo = "BAIXA (atrai capital)"
                    elif "ipca" in evt_lower:
                        impacto_win = "BAIXA (inflacao alta)"
                        impacto_wdo = "ALTA (real enfraquece)"
                    elif "pib" in evt_lower:
                        impacto_win = "ALTA (economia forte)"
                        impacto_wdo = "BAIXA (confianca)"
            elif act_num < for_num:
                surpresa = "ABAIXO"
                if currency == "USD":
                    impacto_wdo = "BAIXA (dolar enfraquece)"
                    impacto_win = "ALTA (alivio no ibov)"
                    if "nonfarm" in evt_lower or "payroll" in evt_lower:
                        impacto_wdo = "BAIXA FORTE (USD enfraquece)"
                        impacto_win = "ALTA FORTE (risk-on)"
                elif currency == "BRL":
                    if "selic" in evt_lower or "copom" in evt_lower:
                        impacto_win = "ALTA (juros aliviam)"
                        impacto_wdo = "ALTA (menos atrativo)"
                    elif "ipca" in evt_lower:
                        impacto_win = "ALTA (inflacao controlada)"
                        impacto_wdo = "BAIXA (real fortalece)"
                    elif "pib" in evt_lower:
                        impacto_win = "BAIXA (economia fraca)"
                        impacto_wdo = "ALTA (fuga capital)"
            else:
                surpresa = "NEUTRO"
        except:
            pass
    
    return {
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
    }


@app.get("/api/noticias-impacto")
async def get_noticias_impacto():
    """Calendario economico — Profit API (principal) + Investing.com (fallback)"""
    try:
        from datetime import timezone, timedelta
        BRT = timezone(timedelta(hours=-3))
        agora = datetime.now(BRT)
        
        eventos_processados = []
        fonte_usada = "nenhuma"
        
        # Check cache first
        cache = _noticias_cache
        if cache["data"] and cache["timestamp"]:
            age = (datetime.now() - cache["timestamp"]).total_seconds()
            if age < cache["ttl"]:
                logger.info(f"Noticias: usando cache ({int(age)}s old)")
                return JSONResponse(cache["data"])
        
        # === FONTE 1: Profit API Economic Calendar ===
        raw_profit = await _buscar_noticias_profit()
        if raw_profit:
            fonte_usada = "Profit API"
            for evt in raw_profit:
                processed = _processar_evento(evt, agora, fonte="profit")
                if processed:
                    eventos_processados.append(processed)
            logger.info(f"Profit Calendar: {len(eventos_processados)} eventos relevantes de {len(raw_profit)} total")
        
        # === FONTE 2: ForexFactory JSON (sempre buscar para complementar) ===
        if len(eventos_processados) < 5:  # complementar se Profit retornou poucos
            raw_inv = _buscar_noticias_forexfactory()
            if raw_inv:
                fonte_usada = "ForexFactory" if not eventos_processados else fonte_usada + " + ForexFactory"
                for evt in raw_inv:
                    processed = _processar_evento(evt, agora, fonte="forexfactory")
                    if processed:
                        eventos_processados.append(processed)
                logger.info(f"ForexFactory: {len(eventos_processados)} eventos")
        
        # Deduplicate by similar name+time
        seen_keys = set()
        unique = []
        for e in eventos_processados:
            key = f"{e['evento'][:20]}_{e['hora']}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append(e)
        
        # Split upcoming vs past
        upcoming = [e for e in unique if not e["ja_passou"]]
        past = [e for e in unique if e["ja_passou"]]
        upcoming.sort(key=lambda x: x.get("minutos_restantes") or 9999)
        past.sort(key=lambda x: -(x.get("minutos_restantes") or 0))
        
        # Alerts
        alertas = [e for e in unique if e["zona_impacto"] in ("ZONA DE IMPACTO", "APROXIMANDO")]
        
        result = {
            "proximos": upcoming,
            "passados": past,
            "total": len(unique),
            "alertas": len(alertas),
            "alerta_eventos": [{"evento": a["evento"], "moeda": a["moeda"], "hora": a["hora"], "zona": a["zona_impacto"], "minutos": a["minutos_restantes"]} for a in alertas],
            "timestamp": agora.strftime("%H:%M:%S"),
            "data": agora.strftime("%d/%m/%Y"),
            "fonte": fonte_usada,
        }
        
        # Save to cache (only if we have events)
        if len(unique) > 0:
            _noticias_cache["data"] = result
            _noticias_cache["timestamp"] = datetime.now()
        else:
            # Don't cache empty results - retry next time
            logger.warning("Noticias: 0 eventos, não cacheando")
        
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Erro noticias-impacto: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"erro": str(e), "proximos": [], "passados": [], "total": 0, "alertas": 0, "alerta_eventos": [], "timestamp": "", "data": ""}, status_code=500)



# ===== OPERADOR SENIOR - MEMÓRIA DE ERROS =====
ERROS_FILE = APP_DIR / "operador_erros.json"

def _carregar_erros_operador():
    """Carrega memória de erros do operador"""
    try:
        if ERROS_FILE.exists():
            with open(ERROS_FILE, "r") as f:
                erros = json.load(f)
                app_state["operador_erros"] = erros
                return erros
    except:
        pass
    return app_state.get("operador_erros", [])

def _salvar_erros_operador():
    """Persiste memória de erros"""
    try:
        with open(ERROS_FILE, "w") as f:
            json.dump(app_state.get("operador_erros", []), f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Erro salvando memória: {e}")

def _registrar_erro_operador(ativo, tecnica, condicoes, motivo_erro):
    """Registra um erro para não repeti-lo"""
    erro = {
        "ativo": ativo,
        "tecnica": tecnica,
        "condicoes": condicoes,  # {tendencia, rsi_faixa, janela, score}
        "motivo_erro": motivo_erro,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "count": 1,
    }
    erros = app_state.get("operador_erros", [])
    # Verificar se erro similar já existe
    for e in erros:
        if (e["tecnica"] == tecnica and e["ativo"] == ativo 
            and e["condicoes"].get("tendencia") == condicoes.get("tendencia")
            and e["condicoes"].get("janela") == condicoes.get("janela")):
            e["count"] += 1
            e["data"] = erro["data"]
            _salvar_erros_operador()
            return
    erros.append(erro)
    # Manter últimos 50 erros
    if len(erros) > 50:
        erros = erros[-50:]
    app_state["operador_erros"] = erros
    _salvar_erros_operador()

def _verificar_erro_similar(ativo, tecnica, condicoes):
    """Verifica se há erro similar na memória. Retorna (bloquear, motivo)"""
    erros = app_state.get("operador_erros", [])
    for e in erros:
        if e["ativo"] == ativo and e["tecnica"] == tecnica:
            # Mesma técnica + mesma tendência + mesmo tipo de janela = erro similar
            if (e["condicoes"].get("tendencia") == condicoes.get("tendencia")
                and e["condicoes"].get("janela_tipo") == condicoes.get("janela_tipo")):
                if e["count"] >= 2:
                    return True, f"MEMÓRIA: {tecnica} já falhou {e['count']}x em {e['condicoes'].get('tendencia')} / {e['condicoes'].get('janela_tipo')}. Evitando repetição."
                elif e["count"] >= 1:
                    return False, f"ALERTA: {tecnica} falhou 1x em condições similares. Cautela extra."
    return False, None


@app.post("/api/operador-live/entrar")
async def operador_entrar(request: Request):
    """
    Usuário clicou ENTRADA AGORA - operador entra no próximo setup válido.
    Registra o trade como ativo e começa a monitorar.
    """
    try:
        data = await request.json()
        ativo = data.get("ativo", "WIN").upper()
        forcar = data.get("forcar", False)
        op_state = app_state["operador_live"][ativo]
        
        # Marcar como aguardando entrada
        op_state["aguardando_entrada"] = True
        if forcar:
            op_state["forcar_proxima"] = True
            logger.info(f"FORÇAR PROXIMA ativado para {ativo} - background vai entrar na próxima")
        salvar_estado_operador()
        
        return JSONResponse({"ok": True, "forcar": forcar, "msg": f"Operador {'FORÇANDO' if forcar else 'aguardando'} próximo setup para {ativo}..."})
    except Exception as e:
        return JSONResponse({"ok": False, "erro": str(e)})


@app.get("/api/operador-live/monitor")
async def operador_monitor(ativo: str = Query("WIN")):
    """
    Monitora trade ativo - verifica se stop ou alvo foram atingidos.
    Chamado pelo frontend a cada refresh.
    """
    try:
        from datetime import timezone, timedelta
        BRT_tz = timezone(timedelta(hours=-3))
        ativo = ativo.upper()
        ticker = "^BVSP" if ativo == "WIN" else "USDBRL=X"
        valor_ponto = 0.20 if ativo == "WIN" else 10.00
        
        op_state = app_state["operador_live"][ativo]
        trade = op_state.get("trade_ativo")
        
        if not trade:
            return JSONResponse({"status": "SEM_TRADE", "msg": "Nenhum trade ativo"})
        
        # Buscar preço atual
        preco_atual = None
        try:
            from data_provider import DataProvider
            dp = DataProvider()
            rt = dp.obter_preco_realtime(ativo)
            if rt and rt.get("preco"):
                preco_atual = rt["preco"]
        except:
            pass
        
        if not preco_atual:
            # Fallback: yfinance last close
            try:
                dados = yf.download(ticker, period="1d", interval="5m", progress=False)
                if not dados.empty:
                    if isinstance(dados.columns, pd.MultiIndex):
                        dados.columns = dados.columns.get_level_values(0)
                    dados.columns = [c.lower() for c in dados.columns]
                    preco_atual = float(dados['close'].iloc[-1])
            except:
                return JSONResponse({"status": "ERRO", "msg": "Sem preço disponível"})
        
        is_compra = trade["tipo"] == "COMPRA"
        stop = trade["stop_loss"]
        alvo = trade["take_profit"]
        entrada = trade["preco_entrada"]
        
        # Verificar resultado
        resultado = None
        if is_compra:
            if preco_atual >= alvo:
                resultado = "WIN"
            elif preco_atual <= stop:
                resultado = "LOSS"
        else:
            if preco_atual <= alvo:
                resultado = "WIN"
            elif preco_atual >= stop:
                resultado = "LOSS"
        
        # P&L parcial
        if is_compra:
            pnl_pts = round(preco_atual - entrada, 1)
        else:
            pnl_pts = round(entrada - preco_atual, 1)
        
        if ativo == "WDO":
            pnl_pts = round(pnl_pts * 1000, 1)
        
        pnl_rs = round(pnl_pts * valor_ponto, 2)
        
        agora = datetime.now(BRT_tz)
        duracao_min = 0
        try:
            h_ent, m_ent = trade["hora_entrada"].split(":")
            entrada_time = agora.replace(hour=int(h_ent), minute=int(m_ent), second=0)
            duracao_min = int((agora - entrada_time).total_seconds() / 60)
        except:
            pass
        
        response = {
            "status": "MONITORANDO" if not resultado else "FECHADO",
            "trade": trade,
            "preco_atual": preco_atual,
            "pnl_pts": pnl_pts,
            "pnl_rs": pnl_rs,
            "duracao_min": duracao_min,
            "resultado": resultado,
        }
        
        # Se trade fechou, registrar resultado
        if resultado:
            hora_saida = agora.strftime("%H:%M")
            trade["resultado"] = resultado
            trade["pts"] = pnl_pts - (5 if ativo == "WIN" else 1)  # custos
            trade["hora_saida"] = hora_saida
            trade["preco_saida"] = preco_atual
            trade["resultado_rs"] = round(trade["pts"] * valor_ponto, 2)
            trade["duracao_min"] = duracao_min
            
            # Registrar na lista de operações
            op_state["operacoes"].append(dict(trade))
            op_state["total_pts"] += trade["pts"]
            op_state["trade_ativo"] = None
            salvar_estado_operador()
            
            if resultado == "LOSS":
                op_state["losses_consecutivos"] += 1
                cooldown_min = 15 if op_state["losses_consecutivos"] >= 2 else 10
                op_state["cooldown_ate"] = (agora + timedelta(minutes=cooldown_min)).isoformat()
                
                # ===== APRENDER COM O ERRO =====
                _registrar_erro_operador(
                    ativo=ativo,
                    tecnica=trade.get("estrategia", "Desconhecida"),
                    condicoes={
                        "tendencia": trade.get("tend_macro", "?"),
                        "janela_tipo": trade.get("janela_qualidade", "?"),
                        "janela": trade.get("janela", "?"),
                        "score": trade.get("score", 0),
                        "rsi_faixa": "sobrevendido" if trade.get("rsi", 50) < 30 else "sobrecomprado" if trade.get("rsi", 50) > 70 else "normal",
                    },
                    motivo_erro=f"LOSS de {abs(trade['pts'])}pts. Stop atingido em {duracao_min}min."
                )
                response["aprendizado"] = f"ERRO REGISTRADO: {trade.get('estrategia')} falhou em {trade.get('janela')}. Operador não repetirá esta técnica nas mesmas condições."
            else:
                op_state["losses_consecutivos"] = 0
                op_state["cooldown_ate"] = None
                response["aprendizado"] = f"WIN registrado! {trade.get('estrategia')} funcionou em {trade.get('janela')}."
            
            LOSS_LIMIT = -400 if ativo == "WIN" else -40
            if op_state["total_pts"] <= LOSS_LIMIT:
                op_state["dia_bloqueado"] = True
            
            response["resultado"] = resultado
            response["pts_final"] = trade["pts"]
            response["rs_final"] = trade["resultado_rs"]
        
        return JSONResponse(response)
    except Exception as e:
        logger.error(f"Erro monitor: {e}")
        return JSONResponse({"status": "ERRO", "msg": str(e)})


@app.get("/api/operador-live")
async def operador_live(ativo: str = Query("WIN"), max_entradas: int = Query(10), forcar_entrada: bool = Query(False)):
    """
    OPERADOR SENIOR LIVE - Análise em TEMPO REAL
    
    Diferente do simulador-real (que analisa o dia inteiro de uma vez),
    este endpoint analisa o MOMENTO ATUAL e decide:
    - ENTRAR AGORA (setup encontrado)
    - ESPERAR (sem setup, aguardando oportunidade)
    - PARAR (limite atingido ou horário ruim)
    
    Mantém estado das operações do dia em app_state.
    Auto-refresh a cada 2 minutos quando mercado aberto.
    
    forcar_entrada=True: "Próxima Entrada" - analisa com filtros mais abertos
    """
    try:
        from datetime import timezone, timedelta
        BRT_tz = timezone(timedelta(hours=-3))
        ativo = ativo.upper()
        ticker = "^BVSP" if ativo == "WIN" else "USDBRL=X"
        valor_ponto = 0.20 if ativo == "WIN" else 10.00
        max_entradas = max(1, max_entradas)
        agora = datetime.now(BRT_tz)
        hoje = agora.date()
        hora_atual = agora.strftime("%H:%M")
        hora_int = agora.hour
        minuto = agora.minute
        
        from data_provider import obter_contrato_vigente as _ocv
        contrato_info = _ocv(ativo)
        contrato_nome = contrato_info.get('ticker_b3', ativo)
        
        # ===== RESETAR ESTADO SE DIA MUDOU =====
        op_state = app_state["operador_live"][ativo]
        if op_state["dia"] != str(hoje):
            op_state.update({
                "operacoes": [], "total_pts": 0, "losses_consecutivos": 0,
                "dia_bloqueado": False, "ultimo_trade_hora": None,
                "cooldown_ate": None, "dia": str(hoje), "trade_ativo": None, "aguardando_entrada": False,
                "forcar_proxima": False
            })
            salvar_estado_operador()
        
        # ===== TRADE ATIVO? =====
        trade_ativo_info = op_state.get("trade_ativo")
        
        # ===== CARREGAR MEMÓRIA DE ERROS =====
        if not app_state.get("operador_erros"):
            _carregar_erros_operador()
        
        # ===== VERIFICAR MERCADO =====
        _mercado_aberto = mercado_aberto()
        if not _mercado_aberto:
            return JSONResponse({
                "status": "MERCADO_FECHADO",
                "ativo": ativo,
                "contrato": contrato_nome,
                "hora_atual": hora_atual,
                "mensagem": "Mercado fechado. O operador começa às 9:15.",
                "operacoes_dia": op_state["operacoes"],
                "total_pts_dia": op_state["total_pts"],
                "modo": "OFFLINE",
            })
        
        # ===== JANELAS DE OPERAÇÃO =====
        t_min = hora_int * 60 + minuto
        janela_nome = "Normal"
        janela_qual = "NORMAL"
        pode_operar_janela = True
        
        JANELAS = [
            (9*60, 9*60+15, "Leilão/Primeiros 15min", "PROIBIDO", False),
            (9*60+15, 10*60+30, "Abertura Pós-Leilão", "PRIME", True),
            (10*60+30, 11*60+30, "Manhã Institucional", "BOA", True),
            (11*60+30, 13*60+30, "Almoço", "RUIM", True),
            (12*60, 13*60, "Almoço Morto", "PROIBIDO", False),
            (13*60+30, 14*60, "Pré-NY", "NORMAL", True),
            (14*60, 15*60, "Retomada NY Open", "PRIME", True),
            (15*60, 16*60+30, "Tarde Institucional", "BOA", True),
            (16*60+30, 17*60, "Pré-Fechamento", "RUIM", True),
            (17*60, 18*60, "Leilão Fechamento", "PROIBIDO", False),
        ]
        for t_ini, t_fim, nome, qual, pode in JANELAS:
            if t_ini <= t_min < t_fim:
                janela_nome = nome
                janela_qual = qual
                pode_operar_janela = pode
                break
        
        # ===== VERIFICAR LIMITES =====
        MAX_LOSSES_CONSEC = 3
        LOSS_LIMIT = -400 if ativo == "WIN" else -40
        
        motivo_bloqueio = None
        if op_state["dia_bloqueado"]:
            motivo_bloqueio = "Limite de perda diária atingido. Tendler: parar, não forçar."
        elif op_state["losses_consecutivos"] >= MAX_LOSSES_CONSEC:
            motivo_bloqueio = f"{MAX_LOSSES_CONSEC} losses consecutivos. Tendler: identificar tilt."
        elif len(op_state["operacoes"]) >= max_entradas:
            motivo_bloqueio = f"Limite de {max_entradas} entradas atingido."
        elif not pode_operar_janela:
            motivo_bloqueio = f"Janela {janela_nome} ({janela_qual}) - operador não opera neste horário."
        elif hora_int >= 17:
            motivo_bloqueio = "Após 17h - leilão de fechamento."
        
        # Cooldown check
        em_cooldown = False
        if op_state["cooldown_ate"]:
            cooldown_time = datetime.fromisoformat(op_state["cooldown_ate"])
            if agora < cooldown_time:
                em_cooldown = True
                mins_restantes = int((cooldown_time - agora).total_seconds() / 60)
                motivo_bloqueio = f"Cooldown ativo: aguardar {mins_restantes}min (até {cooldown_time.strftime('%H:%M')})"
        
        # Se bloqueado e NÃO forçando, marcar mas CONTINUAR a análise para mostrar na UI
        bloqueio_ativo = motivo_bloqueio and not forcar_entrada
        if forcar_entrada and em_cooldown:
            # Forçar entrada bypass cooldown
            op_state["cooldown_ate"] = None
            motivo_bloqueio = None
            em_cooldown = False
        
        # ===== BUSCAR DADOS ATUAIS =====
        from pro_trader_analysis import calcular_tendencia_macro, detectar_setup_profissional
        from analysis_engine import calcular_rsi, calcular_macd, calcular_atr_series
        
        dados = yf.download(ticker, period="5d", interval="5m", progress=False)
        if dados.empty:
            return JSONResponse({"status": "ERRO", "mensagem": "Sem dados do yfinance"})
        
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
        dados.columns = [c.lower() for c in dados.columns]
        dados.index = dados.index.tz_convert(BRT_tz)
        
        # Pegar as últimas 100 velas (contexto)
        if len(dados) < 20:
            return JSONResponse({"status": "ERRO", "mensagem": "Dados insuficientes"})
        
        # ===== TENDÊNCIA MACRO (Elder Tela 1) =====
        tend_macro = calcular_tendencia_macro(dados, ativo)
        
        # ===== ANÁLISE DA VELA ATUAL =====
        w = dados.iloc[-100:] if len(dados) >= 100 else dados
        vela = dados.iloc[-1]
        pos_idx = len(dados) - 1
        ts = dados.index[-1]
        
        o = float(vela['open']); h = float(vela['high'])
        l = float(vela['low']); c = float(vela['close'])
        vol = int(vela.get('volume', 0))
        
        # Indicadores
        rsi_v = 50; macd_h = 0; ema9 = 0; ema21 = 0; atr_v = 150
        tend = "LATERAL"
        
        if len(w) >= 20:
            try:
                rsi_s = calcular_rsi(w)
                rsi_v = round(float(rsi_s.iloc[-1]), 1)
                ml, ms, mh = calcular_macd(w)
                macd_h = round(float(mh.iloc[-1]), 1)
                ema9 = round(float(w['close'].ewm(span=9, adjust=False).mean().iloc[-1]), 2)
                ema21 = round(float(w['close'].ewm(span=21, adjust=False).mean().iloc[-1]), 2)
                atr_s = calcular_atr_series(w)
                atr_v = round(float(atr_s.iloc[-1]), 4) if len(atr_s) > 0 else (150 if ativo == 'WIN' else 0.01)
                if ema9 > ema21 and c > ema9: tend = "ALTA"
                elif ema9 < ema21 and c < ema9: tend = "BAIXA"
                elif ema9 > ema21: tend = "ALTA"
                elif ema9 < ema21: tend = "BAIXA"
            except: pass
        
        # ===== ANÁLISE PRO (Confluence Checklist) =====
        day_indices = list(range(max(0, len(dados)-100), len(dados)))
        
        setup = detectar_setup_profissional(
            w=w, vela=vela, pos_idx=pos_idx, dados=dados,
            day_indices=day_indices, ativo=ativo,
            tend_macro=tend_macro, rsi_v=rsi_v, macd_h=macd_h,
            ema9=ema9, ema21=ema21, atr_v=atr_v,
            operacoes_anteriores=op_state["operacoes"],
        )
        
        tipo_sinal = setup["direcao"]
        operar = setup["operar"]
        score = setup["total_confluencia"]
        conf_label = setup["qualidade"]
        motivos_operar = list(setup["motivos_operar"])
        motivos_nao_operar = list(setup["motivos_nao_operar"])
        suporte = setup["suporte"]
        resistencia = setup["resistencia"]
        vwap = setup["vwap"]
        
        # ===== NOTÍCIAS =====
        try:
            noticias_dia = obter_noticias_do_dia()
            news_impact = avaliar_impacto_noticias(hora_atual, ativo, tipo_sinal, noticias_dia)
            if news_impact["bloquear"]:
                # Notícia não cancela - avisa e reduz sizing
                motivos_operar.append(f"⚠️ NOTÍCIA: {news_impact['motivo']} - SIZING REDUZIDO")
            if news_impact["modificador_score"] < 0:
                score = max(1, score + news_impact["modificador_score"])
                motivos_operar.append(f"Score ajustado {score}/7 (notícia)")
        except: pass
        
        # ===== IDENTIFICAR SETUP DO PLAYBOOK =====
        setup_playbook = _identificar_playbook_live(setup, rsi_v, macd_h, ema9, ema21, c, vwap, atr_v, tend_macro["tendencia"])
        
        # ===== FILTRO DE JANELA =====
        if janela_qual == "RUIM" and operar:
            if conf_label == "SKIP":
                operar = False
                motivos_nao_operar.append(f"Setup SKIP em {janela_nome} - sem condições mínimas")
            elif conf_label in ("C+", "C"):
                motivos_operar.append(f"⚠️ {janela_nome} ({janela_qual}) + {conf_label} = sizing mínimo (1 contrato)")
            elif conf_label == "B+" and score < 4:
                motivos_operar.append(f"⚠️ B+ em {janela_nome} com {score}/7 - cautela no sizing")
        
        # ===== FORÇAR ENTRADA (botão FORÇAR ENTRADA) =====
        # Quando forcar=True, entra AGORA se tiver qualquer direção detectada
        # O usuário decidiu forçar - relaxar TODOS os filtros exceto horário proibido
        # Check persistent forcar flag (set by FORÇAR ENTRADA button, survives tab switch)
        if not forcar_entrada and op_state.get("forcar_proxima"):
            forcar_entrada = True
            logger.info(f"FORÇAR PROXIMA ativo (flag persistente) - aplicando na análise atual")
        
        if forcar_entrada and not operar:
            if tipo_sinal and score >= 1 and janela_qual != "PROIBIDO":
                operar = True
                conf_label = f"{conf_label} (FORÇADO)"
                motivos_nao_operar.clear()  # Limpar bloqueios
                motivos_operar.append(f"ENTRADA FORÇADA - {score}/7 confluências - risco assumido pelo operador")
                if setup.get("contra_tendencia"):
                    motivos_operar.append("⚠️ Contra tendência macro - sizing mínimo recomendado")
            elif not tipo_sinal:
                # Sem direção nenhuma - usar tendência macro como fallback
                if tend_macro["tendencia"] in ("ALTA", "BAIXA"):
                    tipo_sinal = "COMPRA" if tend_macro["tendencia"] == "ALTA" else "VENDA"
                    operar = True
                    conf_label = f"MACRO (FORÇADO)"
                    motivos_nao_operar.clear()
                    motivos_operar.append(f"ENTRADA FORÇADA pela tendência macro ({tend_macro['tendencia']})")
                    motivos_operar.append("⚠️ Poucos fatores - use stop curto e sizing mínimo")
        
        # ===== SMC complementar =====
        smc_data = {}
        try:
            _smc_score, _smc_motivos, smc_data = aplicar_smc_scoring(dados, pos_idx, tipo_sinal, tend)
            if _smc_motivos:
                motivos_operar.extend(_smc_motivos)
        except: pass
        
        # ===== VERIFICAR MEMÓRIA DE ERROS =====
        erro_similar = False
        alerta_erro = None
        if operar and tipo_sinal and setup_playbook["nome"]:
            janela_tipo = "PRIME" if janela_qual == "PRIME" else "BOA" if janela_qual == "BOA" else "RUIM"
            condicoes_check = {
                "tendencia": tend_macro["tendencia"],
                "janela_tipo": janela_tipo,
            }
            erro_similar, alerta_erro = _verificar_erro_similar(ativo, setup_playbook["nome"], condicoes_check)
            if erro_similar:
                operar = False
                motivos_nao_operar.append(alerta_erro)
            elif alerta_erro:
                motivos_nao_operar.append(alerta_erro)
        
        # ===== DECISÃO DO OPERADOR =====
        preco_atual = c
        
        # Calcular stop e alvo
        stop_pts = 0; alvo_pts = 0; stop_price = 0; alvo_price = 0; rr_ratio = 2.0
        if tipo_sinal:
            stop_pts = round(atr_v * 1.5, 4) if ativo != 'WIN' else round(atr_v * 1.5)
            if ativo == "WIN":
                stop_pts = max(round(stop_pts / 5) * 5, 80)
                stop_pts = min(stop_pts, 350)
            else:
                stop_pts = max(round(stop_pts * 200) / 200, 0.015)
                stop_pts = min(stop_pts, 0.08)
            
            rr_ratio = 2.5 if conf_label.startswith("A") else 2.0 if conf_label.startswith("B") else 1.8
            alvo_pts = round(stop_pts * rr_ratio, 4)
            is_compra = tipo_sinal == "COMPRA"
            stop_price = round(preco_atual - stop_pts, 2) if is_compra else round(preco_atual + stop_pts, 2)
            alvo_price = round(preco_atual + alvo_pts, 2) if is_compra else round(preco_atual - alvo_pts, 2)
        
        # ===== RACIOCÍNIO DO OPERADOR =====
        # Se bloqueio ativo (cooldown/dia bloqueado), não operar mas mostrar análise
        if bloqueio_ativo:
            operar = False
            motivos_nao_operar.insert(0, motivo_bloqueio)
        
        raciocinio = ""
        if operar and tipo_sinal:
            raciocinio = (
                f"ENTRADA RECOMENDADA: {tipo_sinal} agora às {hora_atual}\n"
                f"Setup: {setup_playbook['nome'] or 'Confluência'} | {conf_label} ({score}/7)\n"
                f"Janela: {janela_nome} ({janela_qual})\n"
                f"Tendência macro: {tend_macro['tendencia']} (força {tend_macro['forca']})\n"
                f"RSI={rsi_v} | MACD={macd_h} | EMA9{'>' if ema9>ema21 else '<'}EMA21\n"
                f"Stop: {stop_pts}pts | Alvo: {alvo_pts}pts | R:R 1:{rr_ratio}\n"
                f"Preço: {round(preco_atual,2)} | Stop: {stop_price} | Alvo: {alvo_price}\n"
                f"Operação #{len(op_state['operacoes'])+1} de {max_entradas}"
            )
            status = "ENTRADA"
            # Clear forcar flag after successful entry
            if op_state.get("forcar_proxima"):
                op_state["forcar_proxima"] = False
                salvar_estado_operador()
        else:
            # Dar parecer mesmo sem operar - NUNCA ficar mudo
            motivo_espera = motivos_nao_operar[0] if motivos_nao_operar else "Aguardando melhores condições"
            prox_janela = _proxima_janela_boa(t_min)
            
            # Analista sênior - SEMPRE dá opinião clara e direta
            if score >= 5 and tipo_sinal:
                parecer = f"FORTE sinal de {tipo_sinal} ({score}/7 {conf_label}) - Bloqueio: {motivo_espera}"
            elif score >= 3 and tipo_sinal:
                parecer = f"Sinal de {tipo_sinal} ({score}/7 {conf_label}) - Bloqueio: {motivo_espera}"
            elif score >= 2 and tipo_sinal:
                parecer = f"Possível {tipo_sinal} ({score}/7) - Cautela: {motivo_espera}"
            elif score >= 1 and tipo_sinal:
                parecer = f"Indício de {tipo_sinal} ({score}/7) - {motivo_espera}"
            elif tipo_sinal:
                parecer = f"Tendência de {tipo_sinal} fraca - {motivo_espera}"
            else:
                parecer = f"Sem direção - indicadores divergentes"
            
            raciocinio = (
                f"PARECER: {parecer}\n"
                f"Preço: {round(preco_atual,2)} | Tendência macro: {tend_macro['tendencia']}\n"
                f"Score: {score}/7 ({conf_label})\n"
                f"RSI={rsi_v} | MACD={macd_h} | EMA9{'>' if ema9>ema21 else '<'}EMA21\n"
                f"{'Bloqueio: ' + motivo_espera if motivos_nao_operar else 'Sem bloqueio - faltam confluências'}\n"
                f"Janela: {janela_nome} ({janela_qual})\n"
                f"{prox_janela}"
            )
            status = "PARECER" if score >= 1 else "AGUARDANDO"
        
        # ===== PRÓXIMAS OPORTUNIDADES (olhar 3 velas anteriores para contexto) =====
        ultimas_velas = []
        for i in range(-min(6, len(dados)), 0):
            v = dados.iloc[i]
            v_ts = dados.index[i]
            ultimas_velas.append({
                "hora": v_ts.strftime("%H:%M"),
                "open": round(float(v['open']), 2),
                "high": round(float(v['high']), 2),
                "low": round(float(v['low']), 2),
                "close": round(float(v['close']), 2),
            })
        
        # ===== PREÇO REALTIME =====
        preco_rt = None
        try:
            from data_provider import DataProvider
            dp = DataProvider()
            rt = dp.obter_preco_realtime(ativo)
            if rt and rt.get("preco"):
                preco_rt = rt
        except: pass
        
        response = {
            "status": status,
            "ativo": ativo,
            "contrato": contrato_nome,
            "hora_atual": hora_atual,
            "preco_atual": round(preco_atual, 2),
            "preco_realtime": preco_rt,
            "modo": "LIVE",
            "mercado_aberto": True,
            "janela": janela_nome,
            "janela_qualidade": janela_qual,
            "max_entradas": max_entradas,
            # Tendência
            "tend_macro": {
                "tendencia": tend_macro["tendencia"],
                "forca": tend_macro["forca"],
                "descricao": tend_macro["descricao"],
            },
            # Análise atual
            "analise_atual": {
                "direcao": tipo_sinal,
                "score": score,
                "qualidade": conf_label,
                "operar": operar,
                "setup_playbook": setup_playbook["nome"],
                "setup_desc": setup_playbook["desc"],
                "motivos_operar": motivos_operar[:5],
                "motivos_nao_operar": motivos_nao_operar[:5],
                "confluencia": setup["confluencia"],
                "rsi": rsi_v,
                "macd_hist": macd_h,
                "ema9": ema9,
                "ema21": ema21,
                "atr": atr_v,
                "tendencia_curta": tend,
                "suporte": round(suporte, 0) if suporte else None,
                "resistencia": round(resistencia, 0) if resistencia else None,
                "vwap": round(vwap, 0) if vwap else None,
            },
            # Trade proposto (se ENTRADA)
            "trade_proposto": {
                "tipo": tipo_sinal,
                "preco_entrada": round(preco_atual, 2),
                "stop_loss": stop_price,
                "take_profit": alvo_price,
                "stop_pts": round(stop_pts * 1000, 1) if ativo == "WDO" else stop_pts,
                "alvo_pts": round(alvo_pts * 1000, 1) if ativo == "WDO" else alvo_pts,
                "rr": f"1:{round(rr_ratio, 1)}",
            } if operar and tipo_sinal else None,
            # Raciocínio
            "raciocinio": raciocinio,
            # Estado do dia
            "operacoes_dia": op_state["operacoes"],
            "total_ops": len(op_state["operacoes"]),
            "total_pts_dia": round(op_state["total_pts"], 1),
            "total_rs_dia": round(op_state["total_pts"] * valor_ponto, 2),
            "losses_consecutivos": op_state["losses_consecutivos"],
            "dia_bloqueado": op_state["dia_bloqueado"],
            "win_rate_dia": round(sum(1 for op in op_state["operacoes"] if op.get("resultado") == "WIN") / len(op_state["operacoes"]) * 100) if op_state["operacoes"] else 0,
            # Contexto
            "ultimas_velas": ultimas_velas,
            "proxima_janela": _proxima_janela_boa(t_min),
            "timestamp": agora.strftime("%H:%M:%S"),
            "trade_ativo": op_state.get("trade_ativo"),
            "aguardando_entrada": op_state.get("aguardando_entrada", False),
            "erros_memoria": len(app_state.get("operador_erros", [])),
            "alerta_erro": alerta_erro,
            "motivo": motivos_nao_operar[0] if motivos_nao_operar else ("Entrada recomendada" if operar else "Sem sinal"),
        }
        
        # Se trade ativo, enriquecer raciocínio com detalhes do trade
        if trade_ativo_info:
            ta = trade_ativo_info
            response["raciocinio"] = (
                f"TRADE ABERTO: {ta.get('tipo','')} as {ta.get('hora_entrada','')}" + "\n"
                + f"Estrategia: {ta.get('estrategia','')} | {ta.get('conf_label','')} ({ta.get('score',0)}/7)" + "\n"
                + f"Entrada: {ta.get('preco_entrada',0)} | Stop: {ta.get('stop_loss',0)} | Alvo: {ta.get('take_profit',0)}" + "\n"
                + f"Janela: {ta.get('janela','')} ({ta.get('janela_qualidade','')})" + "\n"
                + f"Monitorando... alvo em {ta.get('alvo_pts',0)}pts, stop em {ta.get('stop_pts',0)}pts"
            )
        
        # ===== AUTO-ENTRAR se aguardando_entrada OU forçar_entrada =====
        deve_entrar = (op_state.get("aguardando_entrada") or forcar_entrada) and operar and tipo_sinal and not op_state.get("trade_ativo")
        if deve_entrar:
            # Entrar automaticamente!
            trade_novo = {
                "tipo": tipo_sinal,
                "hora_entrada": hora_atual,
                "preco_entrada": round(preco_atual, 2),
                "stop_loss": stop_price,
                "take_profit": alvo_price,
                "stop_pts": round(stop_pts * 1000, 1) if ativo == "WDO" else stop_pts,
                "alvo_pts": round(alvo_pts * 1000, 1) if ativo == "WDO" else alvo_pts,
                "rr": f"1:{round(rr_ratio, 1)}",
                "estrategia": setup_playbook["nome"] or "Confluência",
                "janela": janela_nome,
                "janela_qualidade": janela_qual,
                "conf_label": conf_label,
                "score": score,
                "rsi": rsi_v,
                "tend_macro": tend_macro["tendencia"],
                "motivos": motivos_operar[:5],
            }
            op_state["trade_ativo"] = trade_novo
            op_state["aguardando_entrada"] = False
            response["status"] = "TRADE_ABERTO"
            response["trade_ativo"] = trade_novo
            response["raciocinio"] = (
                f"TRADE ABERTO: {tipo_sinal} às {hora_atual}\n"
                f"Estratégia: {trade_novo['estrategia']} | {conf_label} ({score}/7)\n"
                f"Entrada: {round(preco_atual,2)} | Stop: {stop_price} | Alvo: {alvo_price}\n"
                f"Janela: {janela_nome} ({janela_qual})\n"
                f"Monitorando... alvo em {alvo_pts}pts, stop em {stop_pts}pts"
            )
            logger.info(f"OPERADOR: Trade aberto {tipo_sinal} {ativo} @ {preco_atual}")
            salvar_estado_operador()
        
        return JSONResponse(response)
        
    except Exception as e:
        logger.error(f"Erro operador-live: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"status": "ERRO", "mensagem": str(e)}, status_code=500)


@app.post("/api/operador-live/registrar")
async def operador_registrar_trade(request: Request):
    """
    Registrar resultado de um trade executado pelo operador.
    O frontend chama isso quando o trade fecha (alvo ou stop).
    """
    try:
        data = await request.json()
        ativo = data.get("ativo", "WIN").upper()
        valor_ponto = 0.20 if ativo == "WIN" else 10.00
        
        op_state = app_state["operador_live"][ativo]
        
        trade = {
            "tipo": data.get("tipo"),
            "hora_entrada": data.get("hora_entrada"),
            "preco_entrada": data.get("preco_entrada"),
            "stop_loss": data.get("stop_loss"),
            "take_profit": data.get("take_profit"),
            "estrategia": data.get("estrategia"),
            "janela": data.get("janela"),
            "conf_label": data.get("conf_label"),
            "score": data.get("score"),
            "resultado": data.get("resultado"),  # "WIN" ou "LOSS"
            "pts": data.get("pts", 0),
            "hora_saida": data.get("hora_saida"),
            "preco_saida": data.get("preco_saida"),
            "resultado_rs": round(data.get("pts", 0) * valor_ponto, 2),
        }
        
        op_state["operacoes"].append(trade)
        op_state["total_pts"] += trade["pts"]
        op_state["ultimo_trade_hora"] = trade["hora_entrada"]
        
        if trade["resultado"] == "LOSS":
            op_state["losses_consecutivos"] += 1
            cooldown_min = 15 if op_state["losses_consecutivos"] >= 2 else 10
            from datetime import timezone, timedelta
            BRT_tz = timezone(timedelta(hours=-3))
            op_state["cooldown_ate"] = (datetime.now(BRT_tz) + timedelta(minutes=cooldown_min)).isoformat()
        else:
            op_state["losses_consecutivos"] = 0
            op_state["cooldown_ate"] = None
        
        LOSS_LIMIT = -400 if ativo == "WIN" else -40
        if op_state["total_pts"] <= LOSS_LIMIT:
            op_state["dia_bloqueado"] = True
        
        # Salvar no histórico persistente
        try:
            from datetime import timezone, timedelta
            BRT_tz2 = timezone(timedelta(hours=-3))
            hoje_str = datetime.now(BRT_tz2).strftime("%d/%m/%Y")
            registrar_historico_completo(
                ativo=ativo,
                data_sessao=hoje_str,
                modo="OPERADOR",
                operacoes=[trade],
                performance={
                    "total_operacoes": len(op_state["operacoes"]),
                    "wins": sum(1 for t in op_state["operacoes"] if t.get("resultado") == "WIN"),
                    "losses": sum(1 for t in op_state["operacoes"] if t.get("resultado") != "WIN"),
                    "win_rate": round(sum(1 for t in op_state["operacoes"] if t.get("resultado") == "WIN") / max(len(op_state["operacoes"]),1) * 100),
                    "total_pts": op_state["total_pts"],
                    "total_rs": round(op_state["total_pts"] * valor_ponto, 2),
                    "fator_lucro": 0,
                },
            )
            logger.info(f"Operador trade salvo no histórico: {trade['resultado']} {trade['pts']}pts")
        except Exception as he:
            logger.error(f"Erro salvando trade do operador no histórico: {he}")
        
        return JSONResponse({"ok": True, "total_ops": len(op_state["operacoes"]), "total_pts": op_state["total_pts"]})
    except Exception as e:
        return JSONResponse({"ok": False, "erro": str(e)}, status_code=500)


def _proxima_janela_boa(t_min_atual):
    """Retorna texto sobre a próxima janela boa de operação"""
    janelas_boas = [
        (9*60+15, "Abertura Pós-Leilão (9:15)"),
        (10*60+30, "Manhã Institucional (10:30)"),
        (14*60, "Retomada NY Open (14:00)"),
        (15*60, "Tarde Institucional (15:00)"),
    ]
    for t, nome in janelas_boas:
        if t > t_min_atual:
            mins = t - t_min_atual
            return f"Próxima janela boa: {nome} (em {mins}min)"
    return "Sem mais janelas boas hoje"


def _identificar_playbook_live(setup_data, rsi_v, macd_h, ema9, ema21, c, vwap, atr_v, tend_macro_dir):
    """Identifica setup do PlayBook para o endpoint live"""
    pa = setup_data.get("price_action", {})
    patterns = pa.get("patterns", []) if isinstance(pa, dict) else []
    confl = setup_data.get("confluencia", {})
    direcao = setup_data.get("direcao")
    
    if any("ROMPIMENTO" in str(p) for p in patterns) and confl.get("volume_confirma"):
        return {"nome": "Rompimento S/R + Volume", "desc": "Nível S/R rompido com volume. Pullback = entrada."}
    
    if vwap and direcao and abs(c - vwap) < atr_v * 0.5:
        if any(p for p in patterns if any(k in str(p) for k in ["Martelo","Engolfo","Pin Bar"])):
            return {"nome": "VWAP Bounce", "desc": f"Preço retornou à VWAP ({round(vwap,0)}) com reversão."}
    
    if (direcao == "COMPRA" and rsi_v < 35) or (direcao == "VENDA" and rsi_v > 65):
        if confl.get("price_action_confirma"):
            return {"nome": "Divergência RSI", "desc": f"RSI extremo ({rsi_v}) com confirmação PA."}
    
    if pa.get("captura_liquidez") or pa.get("falso_rompimento"):
        return {"nome": "Absorção / Smart Money", "desc": "Captura de liquidez detectada."}
    
    if pa.get("pullback") and confl.get("tendencia_tf_maior"):
        return {"nome": "EMA 21 + Tendência", "desc": f"Pullback na EMA21 a favor de {tend_macro_dir}."}
    
    if setup_data.get("total_confluencia", 0) >= 2:
        return {"nome": "Confluência Técnica", "desc": f"{setup_data['total_confluencia']}/7 fatores alinhados."}
    
    return {"nome": None, "desc": None}



@app.get("/api/simulador-real")
async def simulador_real(ativo: str = Query("WIN"), max_entradas: int = Query(5)):
    """
    OPERADOR SENIOR AUTÔNOMO - Simulador Real PRO
    
    Baseado nos frameworks:
    - Triple Screen (Elder): Tela 1 (macro) → Tela 2 (sinal) → Tela 3 (entrada)
    - Confluence Checklist (7 fatores): 4+ para operar
    - PlayBook (Bellafiore): 5 setups nomeados, classificação A+/B+/C+
    - Risk Management: 1-2% por trade, R:R mínimo 1:2, max losses consecutivos
    - Mental Game (Tendler): Controle de tilt, paradas obrigatórias
    - Axiomas de Zurique: Corte perdas rápido, realize lucros
    
    O operador decide AUTONOMAMENTE:
    - Quais horários entrar (janelas ótimas: 9:15-11:30, 14:00-16:30)
    - Qual estratégia usar (dos 5 setups do PlayBook)
    - Tamanho da posição baseado na qualidade do setup
    - Quando parar (limite de perdas, tilt, horário ruim)
    """
    try:
        from datetime import timezone, timedelta
        BRT_tz = timezone(timedelta(hours=-3))
        ativo = ativo.upper()
        ticker = "^BVSP" if ativo == "WIN" else "USDBRL=X"
        valor_ponto = 0.20 if ativo == "WIN" else 10.00
        max_entradas = max(1, max_entradas)
        
        from data_provider import obter_contrato_vigente as _ocv
        contrato_info = _ocv(ativo)
        contrato_nome = contrato_info.get('ticker_b3', ativo)
        
        from pro_trader_analysis import (
            calcular_tendencia_macro, detectar_setup_profissional,
            gerar_analise_completa
        )
        
        # ===== DADOS DO MERCADO =====
        dados = yf.download(ticker, period="5d", interval="5m", progress=False)
        if dados.empty:
            return JSONResponse({"erro": "Sem dados do yfinance"})
        
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
        dados.columns = [c.lower() for c in dados.columns]
        dados.index = dados.index.tz_convert(BRT_tz)
        
        hoje = datetime.now(BRT_tz).date()
        dates = sorted(set(dados.index.date))
        
        # ===== MODO LIVE vs REPLAY =====
        aviso_replay = None
        modo_live = False
        dia_analise = None
        dia_anterior = hoje  # default para modo live
        
        if mercado_aberto():
            today_indices = [i for i, d in enumerate(dados.index.date) if d == hoje and 9 <= dados.index[i].hour < 18]
            if len(today_indices) >= 2:
                dia_analise = hoje
                day_indices = today_indices
                modo_live = True
                logger.info(f"SimReal LIVE: {len(day_indices)} candles de hoje {hoje}")
            else:
                aviso_replay = f"Mercado aberto mas dados de hoje ainda não disponíveis no yfinance. Mostrando replay do último pregão."
        
        if not modo_live:
            dia_anterior = None
            for d in reversed(dates):
                if d < hoje:
                    test_mask = (dados.index.date == d) & (dados.index.hour >= 9) & (dados.index.hour < 18)
                    if dados[test_mask].shape[0] >= 5:
                        dia_anterior = d
                        break
            if not dia_anterior:
                return JSONResponse({"erro": "Nenhum dia util com dados encontrado nos ultimos dias"})
            dia_analise = dia_anterior
            day_indices = [i for i, d in enumerate(dados.index.date) if d == dia_analise and 9 <= dados.index[i].hour < 18]
            if not day_indices:
                return JSONResponse({"erro": "Sem velas do dia anterior"})
        
        from analysis_engine import calcular_rsi, calcular_macd, calcular_atr_series
        
        # ================================================================
        # OPERADOR SENIOR - CONTROLES DE RISCO (Elder + Tendler + Axiomas)
        # ================================================================
        MAX_OPS_DIA = max_entradas
        MAX_LOSSES_CONSECUTIVOS = 3
        LOSS_LIMIT_PTS = -400 if ativo == "WIN" else -40  # limite de perda diária
        losses_consecutivos = 0
        total_pts_dia = 0
        dia_bloqueado = False
        motivo_parada = ""
        
        # ================================================================
        # JANELAS DE OPERAÇÃO DO OPERADOR SENIOR
        # Baseado em Bellafiore ("In Play" = horários com volume/catalista)
        # e experiência de traders BR (Stormer, Andre Moraes)
        # ================================================================
        JANELAS_OTIMAS = [
            # (hora_ini, min_ini, hora_fim, min_fim, nome, qualidade)
            (9, 15, 10, 30, "Abertura Pós-Leilão", "PRIME"),      # Volatilidade pós-abertura
            (10, 30, 11, 30, "Manhã Institucional", "BOA"),        # Fluxo institucional
            (14, 0, 15, 0, "Retomada NY Open", "PRIME"),           # NY abre + fluxo novo
            (15, 0, 16, 30, "Tarde Institucional", "BOA"),         # Ajuste de posições
        ]
        JANELAS_RUINS = [
            (9, 0, 9, 15, "Leilão/Primeiros 15min", "PROIBIDO"),  # Caótico demais
            (11, 30, 13, 30, "Almoço", "RUIM"),                    # Sem volume
            (12, 0, 13, 0, "Almoço Morto", "PROIBIDO"),            # Volume mínimo
            (16, 30, 18, 0, "Pré-Fechamento", "RUIM"),             # Spreads altos
            (17, 0, 18, 0, "Leilão Fechamento", "PROIBIDO"),       # Não operar
        ]
        
        def classificar_janela(hora_int, minuto):
            """Retorna (nome_janela, qualidade, pode_operar)"""
            for h_ini, m_ini, h_fim, m_fim, nome, qual in JANELAS_RUINS:
                t = hora_int * 60 + minuto
                t_ini = h_ini * 60 + m_ini
                t_fim = h_fim * 60 + m_fim
                if t_ini <= t < t_fim:
                    return nome, qual, qual != "PROIBIDO"
            for h_ini, m_ini, h_fim, m_fim, nome, qual in JANELAS_OTIMAS:
                t = hora_int * 60 + minuto
                t_ini = h_ini * 60 + m_ini
                t_fim = h_fim * 60 + m_fim
                if t_ini <= t < t_fim:
                    return nome, qual, True
            return "Horário Normal", "NORMAL", True
        
        # ================================================================
        # PLAYBOOK - 5 SETUPS NOMEADOS (Bellafiore)
        # Cada setup tem: nome, condições, descrição
        # ================================================================
        def identificar_setup_playbook(setup_data, rsi_v, macd_h, ema9, ema21, c, vwap, atr_v, tend_macro_dir):
            """
            Identifica qual dos 5 setups do PlayBook está presente.
            Retorna (nome_setup, descricao, bonus_score)
            """
            pa = setup_data.get("price_action", {})
            patterns = pa.get("patterns", [])
            confl = setup_data.get("confluencia", {})
            direcao = setup_data.get("direcao")
            
            # 1. ROMPIMENTO S/R COM VOLUME
            sr_rompido = any("ROMPIMENTO" in p for p in patterns)
            vol_ok = confl.get("volume_confirma", False)
            if sr_rompido and vol_ok:
                return ("Rompimento S/R + Volume", 
                        "Nível de S/R rompido com volume confirmando. Pullback para reteste = entrada.",
                        1)
            
            # 2. VWAP BOUNCE
            if vwap and direcao:
                dist_vwap = abs(c - vwap)
                if dist_vwap < atr_v * 0.5:
                    has_reversal = any(p for p in patterns if "Martelo" in p or "Engolfo" in p or "Pin Bar" in p)
                    if has_reversal:
                        return ("VWAP Bounce",
                                f"Preço retornou à VWAP ({round(vwap,0)}) com candle de reversão. A favor da tendência.",
                                1)
            
            # 3. DIVERGÊNCIA RSI
            rsi_diverge = False
            if direcao == "COMPRA" and rsi_v < 35:
                rsi_diverge = True
            elif direcao == "VENDA" and rsi_v > 65:
                rsi_diverge = True
            if rsi_diverge and confl.get("price_action_confirma"):
                return ("Divergência RSI",
                        f"RSI em extremo ({rsi_v}) com confirmação de Price Action. Reversão provável.",
                        1)
            
            # 4. ABSORÇÃO / CAPTURA LIQUIDEZ (SMC)
            captura = pa.get("captura_liquidez")
            falso_romp = pa.get("falso_rompimento")
            if captura or falso_romp:
                desc = "Captura de liquidez" if captura else "Falso rompimento"
                return ("Absorção / Smart Money",
                        f"{desc} detectado. Institucional operando contra o varejo.",
                        1)
            
            # 5. EMA 21 + TENDÊNCIA (Pullback)
            pullback = pa.get("pullback")
            if pullback and confl.get("tendencia_tf_maior"):
                return ("EMA 21 + Tendência",
                        f"Pullback na EMA21 a favor da tendência macro ({tend_macro_dir}). Setup clássico Stormer.",
                        1)
            
            # Setup genérico de confluência
            if setup_data.get("total_confluencia", 0) >= 2:
                return ("Confluência Técnica",
                        f"Múltiplos fatores alinhados ({setup_data['total_confluencia']}/7). Entrada por confluência.",
                        0)
            
            return (None, None, 0)
        
        # ================================================================
        # TENDÊNCIA MACRO (Elder Tela 1)
        # ================================================================
        macro_window = dados.iloc[:day_indices[-1]+1]
        tend_macro = calcular_tendencia_macro(macro_window, ativo)

        # ================================================================
        # BRIEFING PRÉ-MERCADO (como operador senior faria)
        # ================================================================
        # Dados do dia anterior para context
        prev_day_data = None
        for d in reversed(dates):
            if d < dia_analise:
                prev_indices = [i for i, dd in enumerate(dados.index.date) if dd == d]
                if prev_indices:
                    prev_day_data = dados.iloc[prev_indices]
                break
        
        briefing = {
            "tendencia_macro": tend_macro["tendencia"],
            "forca_macro": tend_macro["forca"],
            "estrutura": tend_macro["estrutura"],
            "descricao_macro": tend_macro["descricao"],
            "regra_elder": f"Só operar {('COMPRA' if tend_macro['tendencia'] == 'ALTA' else 'VENDA' if tend_macro['tendencia'] == 'BAIXA' else 'ambos com cautela')} - Elder Tela 1",
            "max_entradas_config": max_entradas,
            "risco_por_trade": "1-2% do capital",
            "rr_minimo": "1:2",
            "janelas_operacao": "9:15-11:30 e 14:00-16:30",
            "plano": f"Operar máximo {max_entradas} entradas. "
                     f"Só setups A+ e B+ a favor de {tend_macro['tendencia']}. "
                     f"Stop máximo {'350pts' if ativo == 'WIN' else '8pts WDO'}. "
                     f"Parar após {MAX_LOSSES_CONSECUTIVOS} losses consecutivos ou {abs(LOSS_LIMIT_PTS)}pts de prejuízo.",
        }
        
        if prev_day_data is not None and len(prev_day_data) > 0:
            prev_close = float(prev_day_data['close'].iloc[-1])
            prev_high = float(prev_day_data['high'].max())
            prev_low = float(prev_day_data['low'].min())
            briefing["dia_anterior"] = {
                "fechamento": round(prev_close, 2),
                "high": round(prev_high, 2),
                "low": round(prev_low, 2),
                "amplitude": round(prev_high - prev_low, 0),
                "nota": f"S/R importantes do dia anterior: Suporte {round(prev_low,0)} / Resistência {round(prev_high,0)}"
            }

        # ================================================================
        # NOTÍCIAS DE IMPACTO
        # ================================================================
        try:
            noticias_dia = obter_noticias_do_dia()
            briefing["noticias_count"] = len(noticias_dia)
            # Identificar horários perigosos por notícias
            horarios_noticias = []
            for n in noticias_dia:
                if n.get("impact", "").lower() in ["high", "alto"]:
                    horarios_noticias.append(f"{n.get('time', '?')} - {n.get('title', n.get('evento', '?'))}")
            briefing["noticias_alto_impacto"] = horarios_noticias[:5]
        except Exception as _ne:
            logger.warning(f"Falha ao buscar noticias: {_ne}")
            noticias_dia = []
            briefing["noticias_count"] = 0
        
        # ================================================================
        # ANÁLISE VELA A VELA - OPERADOR AUTÔNOMO
        # ================================================================
        velas_analisadas = []
        operacoes_recomendadas = []
        posicao_aberta = None
        decisoes_operador = []  # Log de decisões do operador
        
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
            
            # ---- INDICADORES ----
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
                    atr_v = round(float(atr_s.iloc[-1]), 4) if len(atr_s) > 0 else (150 if ativo == 'WIN' else 0.01)
                    
                    if ema9 > ema21 and c > ema9: tend = "ALTA"
                    elif ema9 < ema21 and c < ema9: tend = "BAIXA"
                    elif ema9 > ema21: tend = "ALTA"
                    elif ema9 < ema21: tend = "BAIXA"
                except: pass
            
            # ---- ANÁLISE PRO (Confluence Checklist) ----
            setup = detectar_setup_profissional(
                w=w, vela=vela, pos_idx=pos_idx, dados=dados,
                day_indices=day_indices, ativo=ativo,
                tend_macro=tend_macro, rsi_v=rsi_v, macd_h=macd_h,
                ema9=ema9, ema21=ema21, atr_v=atr_v,
                operacoes_anteriores=operacoes_recomendadas,
            )
            
            tipo_sinal = setup["direcao"]
            operar = setup["operar"]
            score = setup["total_confluencia"]
            confianca = setup["confianca"]
            conf_label = setup["qualidade"]
            motivos_operar = list(setup["motivos_operar"])
            motivos_nao_operar = list(setup["motivos_nao_operar"])
            suporte = setup["suporte"]
            resistencia = setup["resistencia"]
            vwap = setup["vwap"]
            fib_level = setup["fib_level"]
            price_action = setup["price_action"]
            
            # ================================================================
            # OPERADOR SENIOR - DECISÃO DE JANELA DE TEMPO
            # ================================================================
            janela_nome, janela_qual, janela_pode = classificar_janela(hora_int, minuto)
            
            if not janela_pode and operar:
                operar = False
                motivos_nao_operar.append(f"OPERADOR: Janela {janela_nome} ({janela_qual}) - não opero neste horário")
            elif janela_qual == "RUIM" and operar and conf_label in ("C+", "B+"):
                # Em horário ruim, só entra A+ 
                if conf_label == "C+":
                    operar = False
                    motivos_nao_operar.append(f"OPERADOR: Horário {janela_nome} exige setup A+ (atual: {conf_label})")
                elif conf_label == "B+":
                    # B+ em horário ruim: só se tiver 5+ confluências
                    if score < 5:
                        operar = False
                        motivos_nao_operar.append(f"OPERADOR: B+ em horário ruim precisa 5+ confluências (atual: {score})")
            
            # ================================================================
            # IMPACTO DE NOTÍCIAS
            # ================================================================
            news_impact = avaliar_impacto_noticias(hora, ativo, tipo_sinal, noticias_dia)
            if news_impact["bloquear"] and operar:
                operar = False
                motivos_nao_operar.append(f"NOTÍCIA: {news_impact['motivo']}")
            elif news_impact["modificador_score"] != 0:
                score = max(0, min(7, score + news_impact["modificador_score"]))
                if news_impact["modificador_score"] < 0:
                    motivos_nao_operar.append(f"Notícia: {news_impact['alerta']}")
                    if score < 4 and operar:
                        operar = False
                        motivos_nao_operar.append(f"Score caiu para {score}/7 por notícia")
                elif news_impact["modificador_score"] > 0:
                    motivos_operar.append(f"Notícia favorável: {news_impact['alerta']}")
            
            # ================================================================
            # IDENTIFICAR SETUP DO PLAYBOOK
            # ================================================================
            setup_playbook_nome, setup_playbook_desc, bonus = identificar_setup_playbook(
                setup, rsi_v, macd_h, ema9, ema21, c, vwap, atr_v, tend_macro["tendencia"]
            )
            if bonus > 0 and setup_playbook_nome:
                score = min(7, score + bonus)
                motivos_operar.append(f"PlayBook: {setup_playbook_nome}")
            
            # SMC complementar
            smc_data = {}
            try:
                _smc_score, _smc_motivos, smc_data = aplicar_smc_scoring(dados, pos_idx, tipo_sinal, tend)
                if _smc_motivos:
                    motivos_operar.extend(_smc_motivos)
                if _smc_score >= 2 and conf_label == "C+" and not setup["contra_tendencia"]:
                    conf_label = "B+ (SMC)"
                    confianca = 4
                    if not operar and not setup["entrada_repetida"]:
                        operar = True
                        motivos_operar.append("SMC forte promoveu C+ para B+")
            except: pass
            
            # Books scoring complementar
            try:
                vela_info_temp = {
                    "open": o, "high": h, "low": l, "close": c,
                    "rsi": rsi_v, "macd_hist": macd_h, "tendencia": tend,
                    "ema9": ema9, "ema21": ema21, "atr": atr_v,
                    "suporte": suporte, "resistencia": resistencia,
                }
                _extra_score, _extra_motivos = aplicar_scoring_avancado(
                    vela_info_temp, tipo_sinal, tend, rsi_v, atr_v, macd_h,
                    c, o, h, l, suporte, resistencia, ema9, ema21, w
                )
                if _extra_motivos:
                    motivos_operar.extend(_extra_motivos)
            except: pass
            
            decisao = "OPERAR" if operar else "NAO OPERAR"
            
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
                "price_action": price_action,
                "confluencia": setup["confluencia"],
                "qualidade_setup": conf_label,
                "janela": janela_nome,
                "janela_qualidade": janela_qual,
                "setup_playbook": setup_playbook_nome,
            }
            velas_analisadas.append(vela_info)
            
            # Cooldown check
            if posicao_aberta:
                current_day_idx = day_indices.index(pos_idx) if pos_idx in day_indices else 0
                if current_day_idx >= posicao_aberta.get("close_idx", 0):
                    posicao_aberta = None
            
            # ================================================================
            # OPERADOR SENIOR - DECISÃO DE ENTRADA
            # ================================================================
            pode_operar = (
                operar and tipo_sinal
                and posicao_aberta is None
                and hora_int < 17
                and not (hora_int == 16 and minuto > 30)
                and len(operacoes_recomendadas) < MAX_OPS_DIA
                and losses_consecutivos < MAX_LOSSES_CONSECUTIVOS
                and not dia_bloqueado
            )
            
            # ===== RACIOCÍNIO DO OPERADOR =====
            raciocinio_operador = ""
            if pode_operar:
                raciocinio_operador = (
                    f"[{hora}] DECISÃO: ENTRAR {tipo_sinal}\n"
                    f"Janela: {janela_nome} ({janela_qual})\n"
                    f"Setup: {setup_playbook_nome or 'Confluência'} | {conf_label} ({score}/7)\n"
                    f"Tendência macro: {tend_macro['tendencia']} (força {tend_macro['forca']})\n"
                    f"Indicadores: RSI={rsi_v} MACD={macd_h} EMA9={'>' if ema9>ema21 else '<'}EMA21\n"
                    f"Operação #{len(operacoes_recomendadas)+1} de {MAX_OPS_DIA} permitidas\n"
                    f"Losses consecutivos: {losses_consecutivos}/{MAX_LOSSES_CONSECUTIVOS}\n"
                    f"P&L dia: {round(total_pts_dia,1)}pts"
                )
            elif operar and tipo_sinal:
                # Queria operar mas não pode
                bloqueios = []
                if posicao_aberta: bloqueios.append("posição aberta/cooldown")
                if len(operacoes_recomendadas) >= MAX_OPS_DIA: bloqueios.append(f"limite {MAX_OPS_DIA} ops atingido")
                if losses_consecutivos >= MAX_LOSSES_CONSECUTIVOS: bloqueios.append(f"{MAX_LOSSES_CONSECUTIVOS} losses consecutivos")
                if dia_bloqueado: bloqueios.append("dia bloqueado por perda")
                if hora_int >= 17: bloqueios.append("horário de fechamento")
                raciocinio_operador = f"[{hora}] BLOQUEADO: Setup {conf_label} presente mas {', '.join(bloqueios)}"
            
            if raciocinio_operador:
                decisoes_operador.append(raciocinio_operador)
            
            if not pode_operar:
                continue
            
            # ================================================================
            # MEMÓRIA INTELIGENTE - CONSULTA ANTES DE ENTRAR (Douglas + Tendler)
            # "Quem não aprende com os erros está condenado a repeti-los"
            # ================================================================
            alerta_memoria = None
            memoria_bloqueou = False
            try:
                mem_op = {
                    "tipo": tipo_sinal,
                    "hora_entrada": hora,
                    "tendencia": tend,
                    "rsi": rsi_v,
                    "macd_hist": macd_h,
                    "score": score,
                    "conf_label": conf_label,
                    "motivos": motivos_operar[:5],
                }
                mem_check = consultar_memoria(mem_op)
                if mem_check and mem_check.get("tem_alerta"):
                    alerta_memoria = mem_check
                    alertas = mem_check.get("alertas", [])
                    # Similaridade >= 85%: BLOQUEIA a entrada (já errou assim antes)
                    if alertas and alertas[0]["similaridade"] >= 85:
                        memoria_bloqueou = True
                        motivos_nao_operar.append(
                            f"🧠 MEMÓRIA BLOQUEOU: {alertas[0]['similaridade']}% similar a erro anterior - "
                            f"{alertas[0].get('licao', 'padrão já deu loss')}"
                        )
                        decisoes_operador.append(
                            f"[{hora}] 🧠 MEMÓRIA INTELIGENTE BLOQUEOU: "
                            f"{tipo_sinal} {conf_label} {score}/7 - "
                            f"Similaridade {alertas[0]['similaridade']}% com erro de {alertas[0].get('erro_data', '?')}. "
                            f"Lição: {alertas[0].get('licao', '?')}"
                        )
                        continue  # PULA esta entrada
                    # Similaridade 70-84%: ALERTA mas permite (com score reduzido)
                    elif alertas and alertas[0]["similaridade"] >= 70:
                        score = max(score - 1, 0)
                        motivos_nao_operar.append(
                            f"⚠ MEMÓRIA: {alertas[0]['similaridade']}% similar a erro - "
                            f"score reduzido. {alertas[0].get('licao', '')}"
                        )
                        # Se score caiu abaixo de 4, não opera
                        if score < 4:
                            memoria_bloqueou = True
                            motivos_nao_operar.append(f"Score caiu para {score}/7 após penalidade de memória")
                            decisoes_operador.append(
                                f"[{hora}] ⚠ MEMÓRIA penalizou score para {score}/7 - entrada cancelada"
                            )
                            continue
                
                # Verificar regras aprendidas
                regras = mem_check.get("regras_ativas", []) if mem_check else []
                for regra in regras:
                    if regra.get("tipo") == "CUIDADO":
                        motivos_nao_operar.append(f"📋 REGRA: {regra.get('descricao', '')}")
                        score = max(score - 1, 0)
                        if score < 4:
                            decisoes_operador.append(
                                f"[{hora}] 📋 REGRA APRENDIDA bloqueou: {regra.get('descricao', '')}"
                            )
                            continue
            except Exception as mem_err:
                logger.error(f"Erro consultando memória: {mem_err}")
            
            # ================================================================
            # STOP E ALVO PRO (Elder + Murphy)
            # ================================================================
            stop_pts = round(atr_v * 1.5, 4) if ativo != 'WIN' else round(atr_v * 1.5)
            if ativo == "WIN":
                stop_pts = max(round(stop_pts / 5) * 5, 80)
                stop_pts = min(stop_pts, 350)
            else:
                stop_pts = max(round(stop_pts * 200) / 200, 0.015)
                stop_pts = min(stop_pts, 0.08)
            
            # R:R baseado na qualidade do setup (Bellafiore: A+ merece mais)
            if conf_label in ("A+", "A"):
                rr_ratio = 2.5  # Setup premium = alvo mais ambicioso
            elif conf_label == "B+":
                rr_ratio = 2.0
            else:
                rr_ratio = 1.8  # C+ = conservador
            
            alvo_pts = round(stop_pts * rr_ratio, 4)
            is_compra = tipo_sinal == "COMPRA"
            
            # Stop estrutural (Murphy: abaixo suporte / acima resistência)
            if is_compra and suporte and abs(c - suporte) < stop_pts:
                _structural_stop = round(c - suporte + atr_v * 0.3)
                if _structural_stop > stop_pts * 0.5 and _structural_stop < stop_pts * 2:
                    if ativo == "WIN":
                        stop_pts = max(round(_structural_stop / 5) * 5, 80)
                    else:
                        stop_pts = max(_structural_stop, 0.015)
                    alvo_pts = round(stop_pts * rr_ratio, 4)
            elif not is_compra and resistencia and abs(resistencia - c) < stop_pts:
                _structural_stop = round(resistencia - c + atr_v * 0.3)
                if _structural_stop > stop_pts * 0.5 and _structural_stop < stop_pts * 2:
                    if ativo == "WIN":
                        stop_pts = max(round(_structural_stop / 5) * 5, 80)
                    else:
                        stop_pts = max(_structural_stop, 0.015)
                    alvo_pts = round(stop_pts * rr_ratio, 4)
            
            # ================================================================
            # ENTRADA REALISTA (próxima vela + slippage)
            # ================================================================
            _signal_idx = day_indices.index(pos_idx)
            _next_idx = _signal_idx + 1
            if _next_idx >= len(day_indices):
                posicao_aberta = {"close_idx": _signal_idx + 3}
                continue
            
            _next_vela = dados.iloc[day_indices[_next_idx]]
            _entry_open = float(_next_vela['open'])
            _hora_entrada_real = dados.index[day_indices[_next_idx]].strftime("%H:%M")
            
            import random as _rnd
            if ativo == "WIN":
                _slippage = _rnd.randint(5, 15)
            else:
                _slippage = round(_rnd.uniform(0.0005, 0.0015), 4)
            
            if is_compra:
                preco_entrada_real = round(_entry_open + _slippage, 2)
            else:
                preco_entrada_real = round(_entry_open - _slippage, 2)
            
            _custo_pts = 5 if ativo == "WIN" else 1
            _preco_sinal = c
            preco_op = preco_entrada_real
            hora_op = _hora_entrada_real
            
            stop_price = round(preco_op - stop_pts, 2) if is_compra else round(preco_op + stop_pts, 2)
            alvo_price = round(preco_op + alvo_pts, 2) if is_compra else round(preco_op - alvo_pts, 2)
            
            # ================================================================
            # WALK FORWARD COM TRAILING STOP
            # ================================================================
            resultado = None
            preco_saida = 0
            hora_saida = ""
            velas_na_op = 0
            max_velas_op = 24  # 2h máximo
            _best_price = preco_op
            _trailing_active = False
            
            future_start = _next_idx + 1
            for fi in range(future_start, len(day_indices)):
                fv = dados.iloc[day_indices[fi]]
                fh = float(fv['high']); fl = float(fv['low'])
                fc = float(fv['close'])
                f_hora = dados.index[day_indices[fi]].strftime("%H:%M")
                velas_na_op += 1
                
                if velas_na_op >= max_velas_op:
                    preco_saida = fc; hora_saida = f_hora
                    pts_t = (preco_saida - preco_op) if is_compra else (preco_op - preco_saida)
                    resultado = "WIN" if pts_t > 0 else "LOSS"
                    break
                
                if is_compra:
                    if fh > _best_price: _best_price = fh
                    _lucro = _best_price - preco_op
                    if _lucro >= stop_pts and not _trailing_active:
                        _trailing_active = True
                        _novo_stop = preco_op + _lucro * 0.2
                        if _novo_stop > stop_price: stop_price = round(_novo_stop, 2)
                    elif _trailing_active and _lucro >= stop_pts * 1.5:
                        _novo_stop = preco_op + _lucro * 0.5
                        if _novo_stop > stop_price: stop_price = round(_novo_stop, 2)
                    if fh >= alvo_price:
                        resultado = "WIN"; preco_saida = alvo_price; hora_saida = f_hora; break
                    if fl <= stop_price:
                        preco_saida = stop_price; hora_saida = f_hora
                        pts_t = preco_saida - preco_op
                        resultado = "WIN" if pts_t > 0 else "LOSS"; break
                else:
                    if fl < _best_price: _best_price = fl
                    _lucro = preco_op - _best_price
                    if _lucro >= stop_pts and not _trailing_active:
                        _trailing_active = True
                        _novo_stop = preco_op - _lucro * 0.2
                        if _novo_stop < stop_price: stop_price = round(_novo_stop, 2)
                    elif _trailing_active and _lucro >= stop_pts * 1.5:
                        _novo_stop = preco_op - _lucro * 0.5
                        if _novo_stop < stop_price: stop_price = round(_novo_stop, 2)
                    if fl <= alvo_price:
                        resultado = "WIN"; preco_saida = alvo_price; hora_saida = f_hora; break
                    if fh >= stop_price:
                        preco_saida = stop_price; hora_saida = f_hora
                        pts_t = preco_op - preco_saida
                        resultado = "WIN" if pts_t > 0 else "LOSS"; break
            
            if resultado is None:
                last_v = dados.iloc[day_indices[-1]]
                preco_saida = float(last_v['close'])
                hora_saida = dados.index[day_indices[-1]].strftime("%H:%M")
                pts_f = (preco_saida - preco_op) if is_compra else (preco_op - preco_saida)
                resultado = "WIN" if pts_f > 0 else "LOSS"
            
            _raw_diff = (preco_saida - preco_op) if is_compra else (preco_op - preco_saida)
            if ativo == "WDO":
                pts_bruto = round(_raw_diff * 1000, 1)
            else:
                pts_bruto = round(_raw_diff, 1)
            pts = round(pts_bruto - _custo_pts, 4)
            rs = round(pts * valor_ponto, 2)
            
            # ---- ANÁLISE COMPLETA ----
            analise_completa = gerar_analise_completa(setup, vela_info, ativo)
            analise_completa += f"\nESTRATÉGIA: {setup_playbook_nome or 'Confluência Técnica'}\n"
            analise_completa += f"JANELA: {janela_nome} ({janela_qual})\n"
            analise_completa += f"RESULTADO: {resultado} | {round(abs(pts),1)}pts | R${round(abs(rs),2)}\n"
            analise_completa += f"Entrada: {preco_op} ({hora_op}) | Saída: {round(preco_saida,2)} ({hora_saida})\n"
            analise_completa += f"Duração: {velas_na_op} velas ({velas_na_op*5}min)\n"
            
            detalhes_perda = ""
            detalhes_vitoria = ""
            
            if resultado == "LOSS":
                # Análise PROFUNDA do loss - usar TODA a técnica
                detalhes_perda = ""
                problemas = []
                licoes = []
                
                # 1. Timing - entrou cedo demais?
                if velas_na_op <= 1:
                    problemas.append(f"STOP em apenas {velas_na_op} vela(s) = entrada prematura. Bellafiore: espere segunda chance/reteste")
                    licoes.append("Esperar pullback de confirmação antes de entrar")
                
                # 2. Contra tendência?
                if setup.get("contra_tendencia"):
                    problemas.append(f"CONTRA TENDÊNCIA macro ({tend_macro['tendencia']}) - Elder Tela 1: NUNCA opere contra o timeframe maior")
                    licoes.append("Axioma #11: Teimosia mata. Siga a tendência macro")
                
                # 3. Stop curto demais para o ATR?
                if stop_pts < atr_v * 1.0:
                    problemas.append(f"Stop ({stop_pts}pts) menor que 1x ATR ({round(atr_v)}pts) - qualquer volatilidade tira")
                    licoes.append("Stop mínimo = 1.5x ATR para dar respiração ao trade")
                
                # 4. Horário ruim?
                if janela_qual in ("RUIM", "PROIBIDO"):
                    problemas.append(f"Entrou em {janela_nome} ({janela_qual}) - volume baixo = movimentos erráticos")
                    licoes.append("Stormer: janela ruim = falsos rompimentos")
                
                # 5. RSI em zona neutra (sem extremo)?
                if 40 <= rsi_v <= 60:
                    problemas.append(f"RSI neutro ({rsi_v}) - sem sobrecompra/venda = sem pressão clara")
                    licoes.append("RSI entre 40-60 = mercado indeciso, melhor esperar extremo")
                
                # 6. MACD sem momento?
                if abs(macd_h) < 5:
                    problemas.append(f"MACD fraco ({macd_h}) - sem momentum na direção")
                
                # 7. Sem confirmação de Price Action?
                if not setup["confluencia"].get("price_action_confirma"):
                    problemas.append("Sem candle de confirmação (martelo/engolfo/pin bar)")
                    licoes.append("Murphy: sempre espere candle de reversão ANTES de entrar")
                
                # 8. Longe do S/R?
                if not setup["confluencia"].get("sr_relevante"):
                    problemas.append("Entrada longe de S/R - sem proteção natural de nível")
                    licoes.append("Melhor entrada é perto de suporte (compra) ou resistência (venda)")
                
                # 9. Contra VWAP?
                if vwap:
                    if (is_compra and c < vwap) or (not is_compra and c > vwap):
                        problemas.append(f"Contra VWAP ({round(vwap, 0)}) - institucional opera a favor do VWAP")
                
                # 10. Sequência de losses?
                if losses_consecutivos >= 2:
                    problemas.append(f"{losses_consecutivos} losses seguidos - Tendler: provável tilt, parar e respirar")
                    licoes.append("Mental Game: após 2+ losses, o emocional contamina a análise")
                
                if not problemas:
                    problemas.append(f"Setup {conf_label} ({score}/7) estava correto mas loss acontece - Douglas: distribuição aleatória")
                    licoes.append("Trading in the Zone: aceite o risco individual, confie no edge estatístico")
                
                detalhes_perda = f"STOP em {hora_saida} | -{round(abs(pts),1)}pts | "
                detalhes_perda += " | ".join(problemas[:4])
                if licoes:
                    detalhes_perda += " | LIÇÃO: " + licoes[0]
                analise_completa += f"\nANÁLISE DO LOSS:\n{detalhes_perda}\n"
            else:
                acertos = []
                if setup["confluencia"].get("tendencia_tf_maior"):
                    acertos.append(f"A FAVOR da tendência macro ({tend_macro['tendencia']}) - Elder Tela 1 confirmou")
                if setup["confluencia"].get("price_action_confirma"):
                    acertos.append("Candle de confirmação na entrada")
                if setup["confluencia"].get("sr_relevante"):
                    acertos.append("Entrada em nível de S/R = proteção natural")
                if vwap and ((is_compra and c >= vwap) or (not is_compra and c <= vwap)):
                    acertos.append(f"A favor do VWAP ({round(vwap,0)})")
                if janela_qual in ("PRIME", "BOA"):
                    acertos.append(f"Janela {janela_nome} ({janela_qual}) = volume institucional")
                if rsi_v < 35 and is_compra:
                    acertos.append(f"RSI sobrevendido ({rsi_v}) = reversão provável")
                elif rsi_v > 65 and not is_compra:
                    acertos.append(f"RSI sobrecomprado ({rsi_v}) = reversão provável")
                if not acertos:
                    acertos.append(f"Confluência {score}/7 alinhada na direção certa")
                detalhes_vitoria = f"ALVO em {hora_saida} | +{round(abs(pts),1)}pts | "
                detalhes_vitoria += " | ".join(acertos[:4])
                detalhes_vitoria += f" | Bellafiore: este é um trade do PlayBook - registre no diário"
                analise_completa += f"\nANÁLISE DO WIN:\n{detalhes_vitoria}\n"
            
            operacoes_recomendadas.append({
                "tipo": tipo_sinal,
                "hora_entrada": hora_op,
                "preco_entrada": round(preco_op, 2),
                "preco_sinal": round(_preco_sinal, 2),
                "slippage": _slippage,
                "custo_pts": _custo_pts,
                "stop_loss": stop_price,
                "take_profit": alvo_price,
                "stop_pts": round(stop_pts * 1000, 1) if ativo == "WDO" else stop_pts,
                "alvo_pts": round(alvo_pts * 1000, 1) if ativo == "WDO" else alvo_pts,
                "rr": f"1:{round(rr_ratio, 1)}",
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
                "confluencia_detalhes": setup["confluencia"],
                "qualidade_setup": conf_label,
                "estrategia": setup_playbook_nome or "Confluência Técnica",
                "estrategia_desc": setup_playbook_desc or "",
                "janela": janela_nome,
                "janela_qualidade": janela_qual,
                "raciocinio": raciocinio_operador,
            })
            
            # ===== CONTROLES PÓS-TRADE =====
            total_pts_dia += pts
            if resultado == "LOSS":
                losses_consecutivos += 1
                cooldown_velas = 4 if losses_consecutivos >= 2 else 3
                decisoes_operador.append(
                    f"[{hora_saida}] LOSS #{losses_consecutivos}. "
                    f"Tendler: {'PAUSA OBRIGATÓRIA - 3 losses' if losses_consecutivos >= MAX_LOSSES_CONSECUTIVOS else 'Cooldown ' + str(cooldown_velas) + ' velas'}. "
                    f"P&L dia: {round(total_pts_dia,1)}pts"
                )
            else:
                losses_consecutivos = 0
                cooldown_velas = 2
                decisoes_operador.append(
                    f"[{hora_saida}] WIN! P&L dia: {round(total_pts_dia,1)}pts. "
                    f"Axioma #2: Realizei lucro, não insisto."
                )
            
            if total_pts_dia <= LOSS_LIMIT_PTS:
                dia_bloqueado = True
                motivo_parada = f"Limite de perda diária atingido ({round(total_pts_dia,1)}pts). Tendler: elevar o piso, não forçar."
                decisoes_operador.append(f"[{hora_saida}] DIA BLOQUEADO: {motivo_parada}")
            
            if losses_consecutivos >= MAX_LOSSES_CONSECUTIVOS:
                motivo_parada = f"{MAX_LOSSES_CONSECUTIVOS} losses consecutivos. Tendler: identificar tilt antes de continuar."
                decisoes_operador.append(f"[{hora_saida}] PARADA: {motivo_parada}")
            
            # ================================================================
            # GRAVAR NA MEMÓRIA INTELIGENTE (cada trade individual)
            # ================================================================
            try:
                op_mem = {
                    "tipo": tipo_sinal,
                    "hora_entrada": hora_op,
                    "hora_saida": hora_saida,
                    "resultado": resultado,
                    "pts": pts,
                    "resultado_rs": rs,
                    "tendencia": tend,
                    "rsi": rsi_v,
                    "macd_hist": macd_h,
                    "score": score,
                    "conf_label": conf_label,
                    "motivos": motivos_operar[:5],
                    "detalhes_perda": detalhes_perda if resultado == "LOSS" else "",
                    "detalhes_vitoria": detalhes_vitoria if resultado == "WIN" else "",
                    "janela": janela_nome,
                    "janela_qualidade": janela_qual,
                    "setup_playbook": setup_playbook_nome,
                    "vwap": round(vwap, 0) if vwap else None,
                    "suporte": round(suporte, 0) if suporte else None,
                    "resistencia": round(resistencia, 0) if resistencia else None,
                }
                mem_resultado = registrar_trade_replay(ativo, op_mem)
                logger.info(
                    f"SimReal MEMÓRIA: {resultado} {pts}pts gravado | "
                    f"Total ops: {mem_resultado.get('total_operacoes')} | "
                    f"WR: {mem_resultado.get('win_rate_global')}% | "
                    f"Erros na memória: {mem_resultado.get('memoria_erros')}"
                )
                # Guardar alerta de memória na operação para o frontend
                if alerta_memoria and alerta_memoria.get("tem_alerta"):
                    operacoes_recomendadas[-1]["alerta_memoria"] = {
                        "alertas": alerta_memoria["alertas"][:2],
                        "total_erros": alerta_memoria.get("total_erros_memoria", 0),
                    }
                # Guardar licao da memória
                if resultado == "LOSS" and mem_resultado.get("licao"):
                    operacoes_recomendadas[-1]["licao_memoria"] = mem_resultado["licao"]
            except Exception as mem_save_err:
                logger.error(f"Erro gravando trade na memória: {mem_save_err}")
            
            posicao_aberta = {"close_idx": future_start + velas_na_op + cooldown_velas}
        
        # ================================================================
        # RESUMO DO DIA
        # ================================================================
        first_v = velas_analisadas[0] if velas_analisadas else {}
        last_v = velas_analisadas[-1] if velas_analisadas else {}
        abertura = first_v.get("open", 0)
        fechamento = last_v.get("close", 0)
        variacao = round((fechamento / abertura - 1) * 100, 2) if abertura else 0
        high_dia = max(v["high"] for v in velas_analisadas) if velas_analisadas else 0
        low_dia = min(v["low"] for v in velas_analisadas) if velas_analisadas else 0
        amplitude = round(high_dia - low_dia, 0)
        
        oportunidades = [v for v in velas_analisadas if v["score"] >= 4 and v["tipo_sinal"] is not None]
        
        total_ops = len(operacoes_recomendadas)
        wins = sum(1 for op in operacoes_recomendadas if op["resultado"] == "WIN")
        losses = total_ops - wins
        win_rate = round(wins / total_ops * 100) if total_ops > 0 else 0
        total_pts = sum(op["pts"] for op in operacoes_recomendadas)
        total_rs = sum(op["resultado_rs"] for op in operacoes_recomendadas)
        
        velas_operar = sum(1 for v in velas_analisadas if v["decisao"] == "OPERAR")
        velas_nao = sum(1 for v in velas_analisadas if v["decisao"] == "NAO OPERAR")
        
        # Factor de lucro
        ganhos = sum(op["pts"] for op in operacoes_recomendadas if op["resultado"] == "WIN")
        perdas = abs(sum(op["pts"] for op in operacoes_recomendadas if op["resultado"] == "LOSS"))
        fator_lucro = round(ganhos / perdas, 2) if perdas > 0 else (999 if ganhos > 0 else 0)
        
        # Melhor e pior trade
        melhor_trade = max(operacoes_recomendadas, key=lambda x: x["pts"]) if operacoes_recomendadas else None
        pior_trade = min(operacoes_recomendadas, key=lambda x: x["pts"]) if operacoes_recomendadas else None
        
        # Learning + Histórico completo
        if operacoes_recomendadas:
            try:
                registrar_sessao(
                    ativo, dia_analise.strftime("%d/%m/%Y"),
                    operacoes_recomendadas,
                    {"win_rate": win_rate, "total_pts": total_pts}
                )
                # Salvar histórico completo com todas as operações e análises
                registrar_historico_completo(
                    ativo=ativo,
                    data_sessao=dia_analise.strftime("%d/%m/%Y"),
                    modo=_modo if '_modo' in dir() else ("REAL" if modo_live else "REPLAY"),
                    operacoes=operacoes_recomendadas,
                    performance={
                        "total_operacoes": total_ops, "wins": wins, "losses": losses,
                        "win_rate": win_rate, "total_pts": total_pts, "total_rs": total_rs,
                        "fator_lucro": fator_lucro,
                    },
                    tend_macro=tend_macro,
                )
                logger.info(f"Histórico salvo: {total_ops} ops, WR {win_rate}%")
            except Exception as e:
                logger.error(f"Erro registrando aprendizado/histórico: {e}")
        
        aprendizado = obter_resumo_aprendizado()
        
        _agora = datetime.now(BRT_tz)
        _hora_atual = _agora.hour
        _dia_semana = _agora.weekday()
        _mercado_aberto = (_dia_semana < 5 and 9 <= _hora_atual < 18)
        _modo = "REAL" if modo_live else "REPLAY"
        
        return JSONResponse({
            "dia": dia_analise.strftime("%d/%m/%Y"),
            "ativo": ativo,
            "contrato": contrato_nome,
            "valor_ponto": valor_ponto,
            "modo": _modo,
            "mercado_aberto": _mercado_aberto,
            "aprendizado": aprendizado,
            "memoria_inteligente": {
                "total_erros_gravados": aprendizado.get("memoria_erros_total", 0),
                "regras_aprendidas": len(aprendizado.get("regras_aprendidas", [])),
                "win_rate_global": aprendizado.get("win_rate_global", 0),
                "total_operacoes_historico": aprendizado.get("total_operacoes", 0),
                "trades_bloqueados_memoria": sum(1 for d in decisoes_operador if "MEMÓRIA" in d and "BLOQUEOU" in d),
                "trades_penalizados_memoria": sum(1 for d in decisoes_operador if "MEMÓRIA penalizou" in d),
                "regras_detalhes": aprendizado.get("regras_aprendidas", [])[:5],
            },
            "max_entradas_config": max_entradas,
            "briefing_operador": briefing,
            "decisoes_operador": decisoes_operador,
            "tend_macro": {
                "tendencia": tend_macro["tendencia"],
                "forca": tend_macro["forca"],
                "descricao": tend_macro["descricao"],
                "estrutura": tend_macro["estrutura"],
            },
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
                "fator_lucro": fator_lucro,
                "melhor_trade": {"pts": round(melhor_trade["pts"], 1), "hora": melhor_trade["hora_entrada"], "estrategia": melhor_trade.get("estrategia", "")} if melhor_trade else None,
                "pior_trade": {"pts": round(pior_trade["pts"], 1), "hora": pior_trade["hora_entrada"], "estrategia": pior_trade.get("estrategia", "")} if pior_trade else None,
                "max_ops_dia": MAX_OPS_DIA,
                "losses_limit": MAX_LOSSES_CONSECUTIVOS,
                "dia_bloqueado": dia_bloqueado,
                "motivo_parada": motivo_parada,
            },
            "operacoes": operacoes_recomendadas,
            "oportunidades": oportunidades,
            "total_oportunidades": len(oportunidades),
            "velas": velas_analisadas,
            "chart_data": [{
                "hora": v["hora"],
                "open": v["open"], "high": v["high"], "low": v["low"], "close": v["close"],
                "rsi": v["rsi"], "ema9": v.get("ema9", 0), "ema21": v.get("ema21", 0),
                "atr": v.get("atr", 0), "vwap": v.get("vwap"),
                "suporte": v.get("suporte"), "resistencia": v.get("resistencia"),
                "fib_level": v.get("fib_level"),
                "score": v["score"], "decisao": v["decisao"],
                "tipo_sinal": v.get("tipo_sinal"),
                "conf_label": v.get("conf_label", ""),
                "motivos_operar": v.get("motivos_operar", [])[:3],
                "price_action": v.get("price_action", {}),
                "confluencia": v.get("confluencia", {}),
                "janela": v.get("janela", ""),
                "setup_playbook": v.get("setup_playbook"),
            } for v in velas_analisadas],
            "timestamp": datetime.now(BRT_tz).strftime("%H:%M:%S"),
        })
    except Exception as e:
        logger.error(f"Erro simulador-real: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"erro": str(e)}, status_code=500)



@app.get("/api/treinamento-ia")
async def treinamento_ia(ativo: str = Query("WIN")):
    """
    TREINAMENTO: IA opera com filtros ABERTOS para aprender.
    Diferente do Simulador Real (só A+/B+), aqui entra em C+ também.
    Objetivo: experimentar mais, errar mais, aprender mais.
    Registra tudo na memória para nunca repetir os mesmos erros.
    """
    try:
        from datetime import timezone, timedelta
        BRT_tz = timezone(timedelta(hours=-3))
        ativo = ativo.upper()
        ticker = "^BVSP" if ativo == "WIN" else "USDBRL=X"
        valor_ponto = 0.20 if ativo == "WIN" else 10.00
        
        from data_provider import obter_contrato_vigente as _ocv
        contrato_info = _ocv(ativo)
        contrato_nome = contrato_info.get('ticker_b3', ativo)
        
        from pro_trader_analysis import (
            calcular_tendencia_macro, detectar_setup_profissional,
            gerar_analise_completa
        )
        from learning_engine import consultar_memoria
        
        dados = yf.download(ticker, period="5d", interval="5m", progress=False)
        if dados.empty:
            return JSONResponse({"erro": "Sem dados do yfinance"})
        
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
        dados.columns = [c.lower() for c in dados.columns]
        dados.index = dados.index.tz_convert(BRT_tz)
        
        hoje = datetime.now(BRT_tz).date()
        dates = sorted(set(dados.index.date))
        
        # MODO LIVE: se mercado aberto E tem candles de hoje, analisar hoje ao vivo
        # MODO REPLAY: se mercado fechado, analisar dia anterior
        modo_live = False
        dia_analise = None
        dia_anterior = hoje  # default para modo live
        
        aviso_replay = None
        if mercado_aberto():
            today_indices = [i for i, d in enumerate(dados.index.date) if d == hoje and 9 <= dados.index[i].hour < 18]
            if len(today_indices) >= 2:
                dia_analise = hoje
                day_indices = today_indices
                modo_live = True
                logger.info(f"CT/SimReal LIVE: {len(day_indices)} candles de hoje {hoje}")
            else:
                aviso_replay = f"Mercado aberto mas dados de hoje ainda não disponíveis no yfinance. Mostrando replay do último pregão para treino."
        
        if not modo_live:
            dia_anterior = None
            for d in reversed(dates):
                if d < hoje:
                    test_mask = (dados.index.date == d) & (dados.index.hour >= 9) & (dados.index.hour < 18)
                    if dados[test_mask].shape[0] >= 5:
                        dia_anterior = d
                        break
            if not dia_anterior:
                return JSONResponse({"erro": "Nenhum dia util com dados encontrado nos ultimos dias"})
            dia_analise = dia_anterior
            day_indices = [i for i, d in enumerate(dados.index.date) if d == dia_analise and 9 <= dados.index[i].hour < 18]
            if not day_indices:
                return JSONResponse({"erro": "Sem velas do dia anterior"})
        
        from analysis_engine import calcular_rsi, calcular_macd, calcular_atr_series
        
        # ---- TREINAMENTO: FILTROS ABERTOS ----
        # Diferente do Simulador Real:
        # - Entra em C+ (3/7 confluências) pra cima
        # - Mais operações por dia (até 15)
        # - Permite contra-tendência em C+ com alerta
        # - Consulta memória de erros antes de cada entrada
        MAX_OPS_DIA = 15   # Mais que o Simulador Real (8)
        MAX_LOSSES_CONSECUTIVOS = 4  # Tolerante mas não descuidado
        losses_consecutivos = 0
        total_pts_dia = 0
        
        # Tendência macro
        macro_window = dados.iloc[:day_indices[-1]+1]
        tend_macro = calcular_tendencia_macro(macro_window, ativo)

        # ---- JANELAS DE OPERAÇÃO (CT) ----
        JANELAS_OTIMAS_CT = [
            (9, 15, 10, 30, "Abertura Prime", "PRIME"),
            (10, 30, 11, 30, "Manhã", "BOA"),
            (14, 0, 16, 0, "Tarde Prime", "PRIME"),
            (13, 30, 14, 0, "Pré-Tarde", "BOA"),
            (16, 0, 16, 30, "Fim de Tarde", "BOA"),
        ]
        JANELAS_RUINS_CT = [
            (9, 0, 9, 15, "Pré-Abertura", "RUIM"),
            (11, 30, 13, 30, "Almoço", "RUIM"),
            (12, 0, 13, 0, "Almoço Morto", "PROIBIDO"),
            (16, 30, 18, 0, "Pré-Fechamento", "RUIM"),
            (17, 0, 18, 0, "Leilão Fechamento", "PROIBIDO"),
        ]
        def classificar_janela_ct(hora_int, minuto):
            for h_ini, m_ini, h_fim, m_fim, nome, qual in JANELAS_RUINS_CT:
                t = hora_int * 60 + minuto
                if h_ini * 60 + m_ini <= t < h_fim * 60 + m_fim:
                    return nome, qual, qual != "PROIBIDO"
            for h_ini, m_ini, h_fim, m_fim, nome, qual in JANELAS_OTIMAS_CT:
                t = hora_int * 60 + minuto
                if h_ini * 60 + m_ini <= t < h_fim * 60 + m_fim:
                    return nome, qual, True
            return "Normal", "NORMAL", True

        # ---- NOTICIAS DE IMPACTO ----
        try:
            noticias_dia = obter_noticias_do_dia()
            logger.info(f"Noticias carregadas: {len(noticias_dia)} eventos")
        except Exception as _ne:
            logger.warning(f"Falha ao buscar noticias: {_ne}")
            noticias_dia = []

        
        velas_analisadas = []
        operacoes = []
        posicao_aberta = None
        
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
            
            rsi_v = 50; macd_h = 0; ema9 = 0; ema21 = 0; atr_v = 150
            tend = "LATERAL"
            
            if len(w) >= 20:
                try:
                    rsi_s = calcular_rsi(w)
                    rsi_v = round(float(rsi_s.iloc[-1]), 1)
                    ml, ms, mh = calcular_macd(w)
                    macd_h = round(float(mh.iloc[-1]), 1)
                    ema9 = round(float(w['close'].ewm(span=9, adjust=False).mean().iloc[-1]), 2)
                    ema21 = round(float(w['close'].ewm(span=21, adjust=False).mean().iloc[-1]), 2)
                    atr_s = calcular_atr_series(w)
                    atr_v = round(float(atr_s.iloc[-1]), 4) if len(atr_s) > 0 else (150 if ativo == 'WIN' else 0.01)
                    
                    if ema9 > ema21 and c > ema9: tend = "ALTA"
                    elif ema9 < ema21 and c < ema9: tend = "BAIXA"
                    elif ema9 > ema21: tend = "ALTA"
                    elif ema9 < ema21: tend = "BAIXA"
                except: pass
            
            # Análise PRO (mesma do simulador real)
            setup = detectar_setup_profissional(
                w=w, vela=vela, pos_idx=pos_idx, dados=dados,
                day_indices=day_indices, ativo=ativo,
                tend_macro=tend_macro, rsi_v=rsi_v, macd_h=macd_h,
                ema9=ema9, ema21=ema21, atr_v=atr_v,
                operacoes_anteriores=operacoes,
            )
            
            tipo_sinal = setup["direcao"]
            score = setup["total_confluencia"]
            conf_label = setup["qualidade"]
            
            # ---- TREINAMENTO: REGRAS MAIS ABERTAS ----
            # C+ (3/7) pra cima = ENTRA (no simulador real seria SKIP)
            # Contra-tendência: entra se tiver 3+ confluências (para aprender)
            operar_treino = (
                tipo_sinal is not None
                and score >= 3  # C+ mínimo (simulador real exige 4)
                and not setup["entrada_repetida"]
            )
            
            # Contra-tendência: PERMITE no treino, mas marca como aprendizado
            alerta_contra = ""
            if setup["contra_tendencia"] and score >= 3:
                operar_treino = True  # Entra mesmo contra tendência para APRENDER
                alerta_contra = f"⚠ CONTRA TENDÊNCIA - entrada de APRENDIZADO (Elder proibiria no real)"
            
            # Consultar memória de erros
            alerta_memoria = None
            if operar_treino:
                mem_check = consultar_memoria({
                    "tipo": tipo_sinal,
                    "hora_entrada": hora,
                    "tendencia": tend,
                    "rsi": rsi_v,
                    "macd_hist": macd_h,
                    "score": score,
                    "conf_label": conf_label,
                })
                if mem_check["tem_alerta"]:
                    alerta_memoria = mem_check
            

            # ---- IMPACTO DE NOTICIAS NA ENTRADA (TREINAMENTO) ----
            news_impact = avaliar_impacto_noticias(hora, ativo, tipo_sinal, noticias_dia)
            if news_impact["bloquear"] and operar_treino:
                # No treinamento, não bloqueia mas avisa fortemente
                motivos_nao_operar.append(f"ALERTA NOTÍCIA: {news_impact['motivo']}")
                # Só bloqueia se score < 5 (C+ em treino)
                if score < 5:
                    operar_treino = False
                    motivos_nao_operar.append(f"Score {score}/7 + notícia crítica = bloqueado")
            elif news_impact["modificador_score"] != 0:
                score = max(0, min(7, score + news_impact["modificador_score"]))
                if news_impact["modificador_score"] < 0:
                    motivos_nao_operar.append(f"Notícia: {news_impact['alerta']}")
                elif news_impact["modificador_score"] > 0:
                    motivos_operar.append(f"Notícia favorável: {news_impact['alerta']}")
            vies_noticias = news_impact.get("vies_noticias")
            
            decisao = "OPERAR" if operar_treino else "NAO OPERAR"
            
            # SMC (complementar)
            smc_data = {}
            try:
                _smc_score, _smc_motivos, smc_data = aplicar_smc_scoring(dados, pos_idx, tipo_sinal, tend)
                if _smc_motivos:
                    setup["motivos_operar"].extend(_smc_motivos)
            except: pass
            
            vela_info = {
                "idx": len(velas_analisadas),
                "hora": hora,
                "open": round(o, 2), "high": round(h, 2),
                "low": round(l, 2), "close": round(c, 2),
                "rsi": rsi_v, "macd_hist": macd_h,
                "ema9": ema9, "ema21": ema21, "atr": atr_v,
                "tendencia": tend, "score": score,
                "decisao": decisao,
                "tipo_sinal": tipo_sinal,
                "confianca": setup["confianca"],
                "conf_label": conf_label,
                "motivos_operar": setup["motivos_operar"],
                "motivos_nao_operar": setup["motivos_nao_operar"],
                "suporte": round(setup["suporte"], 0) if setup["suporte"] else None,
                "resistencia": round(setup["resistencia"], 0) if setup["resistencia"] else None,
                "vwap": round(setup["vwap"], 0) if setup["vwap"] else None,
                "fib_level": setup["fib_level"],
                "smc": smc_data,
                "price_action": setup["price_action"],
                "confluencia": setup["confluencia"],
                "alerta_contra": alerta_contra,
                "alerta_memoria": alerta_memoria,
            }
            velas_analisadas.append(vela_info)
            
            # Cooldown
            if posicao_aberta:
                current_day_idx = day_indices.index(pos_idx) if pos_idx in day_indices else 0
                if current_day_idx >= posicao_aberta.get("close_idx", 0):
                    posicao_aberta = None
            
            # ===== COOLDOWN APÓS LOSS (Tendler) =====
            current_vela_idx = day_indices.index(pos_idx) if pos_idx in day_indices else 0
            em_cooldown_ct = current_vela_idx < _ct_cool_idx if '_ct_cool_idx' in locals() else False
            
            pode_operar = (
                operar_treino and tipo_sinal
                and posicao_aberta is None
                and hora_int < 17
                and not (hora_int == 16 and minuto > 30)
                and len(operacoes) < MAX_OPS_DIA
                and losses_consecutivos < MAX_LOSSES_CONSECUTIVOS
                and not em_cooldown_ct
            )
            
            if pode_operar:
                stop_pts = round(atr_v * 1.5, 4) if ativo != 'WIN' else round(atr_v * 1.5)
                if ativo == "WIN":
                    stop_pts = max(round(stop_pts / 5) * 5, 80)
                    stop_pts = min(stop_pts, 350)
                else:
                    # WDO: stop mínimo 0.015 (equivale a ~5 candles de margem)
                    stop_pts = max(round(stop_pts * 200) / 200, 0.015)
                    stop_pts = min(stop_pts, 0.08)
                
                alvo_pts = round(stop_pts * 2.0, 4)
                is_compra = tipo_sinal == "COMPRA"
                
                # Entrada realista
                _signal_idx = day_indices.index(pos_idx)
                _next_idx = _signal_idx + 1
                if _next_idx >= len(day_indices):
                    posicao_aberta = {"close_idx": _signal_idx + 3}
                    continue
                
                _next_vela = dados.iloc[day_indices[_next_idx]]
                _entry_open = float(_next_vela['open'])
                _hora_real = dados.index[day_indices[_next_idx]].strftime("%H:%M")
                
                import random as _rnd
                _slippage = _rnd.randint(5, 15) if ativo == "WIN" else round(_rnd.uniform(0.0005, 0.0015), 4)
                
                preco_op = round(_entry_open + (_slippage if is_compra else -_slippage), 2)
                _custo_pts = 5 if ativo == "WIN" else 1
                _preco_sinal = c
                
                stop_price = round(preco_op - stop_pts, 2) if is_compra else round(preco_op + stop_pts, 2)
                alvo_price = round(preco_op + alvo_pts, 2) if is_compra else round(preco_op - alvo_pts, 2)
                
                # Walk forward
                resultado = None; preco_saida = 0; hora_saida = ""; velas_na_op = 0
                _best_price = preco_op; _trailing_active = False
                
                future_start = _next_idx + 1
                for fi in range(future_start, len(day_indices)):
                    fv = dados.iloc[day_indices[fi]]
                    fh = float(fv['high']); fl = float(fv['low']); fc = float(fv['close'])
                    f_hora = dados.index[day_indices[fi]].strftime("%H:%M")
                    velas_na_op += 1
                    
                    if velas_na_op >= 24:
                        preco_saida = fc; hora_saida = f_hora
                        resultado = "WIN" if ((preco_saida - preco_op) if is_compra else (preco_op - preco_saida)) > 0 else "LOSS"
                        break
                    
                    if is_compra:
                        if fh > _best_price: _best_price = fh
                        _lucro = _best_price - preco_op
                        if _lucro >= stop_pts and not _trailing_active:
                            _trailing_active = True
                            _ns = preco_op + _lucro * 0.2
                            if _ns > stop_price: stop_price = round(_ns, 2)
                        elif _trailing_active and _lucro >= stop_pts * 1.5:
                            _ns = preco_op + _lucro * 0.5
                            if _ns > stop_price: stop_price = round(_ns, 2)
                        if fh >= alvo_price:
                            resultado = "WIN"; preco_saida = alvo_price; hora_saida = f_hora; break
                        if fl <= stop_price:
                            preco_saida = stop_price; hora_saida = f_hora
                            resultado = "WIN" if (preco_saida - preco_op) > 0 else "LOSS"; break
                    else:
                        if fl < _best_price: _best_price = fl
                        _lucro = preco_op - _best_price
                        if _lucro >= stop_pts and not _trailing_active:
                            _trailing_active = True
                            _ns = preco_op - _lucro * 0.2
                            if _ns < stop_price: stop_price = round(_ns, 2)
                        elif _trailing_active and _lucro >= stop_pts * 1.5:
                            _ns = preco_op - _lucro * 0.5
                            if _ns < stop_price: stop_price = round(_ns, 2)
                        if fl <= alvo_price:
                            resultado = "WIN"; preco_saida = alvo_price; hora_saida = f_hora; break
                        if fh >= stop_price:
                            preco_saida = stop_price; hora_saida = f_hora
                            resultado = "WIN" if (preco_op - preco_saida) > 0 else "LOSS"; break
                
                if resultado is None:
                    last_v = dados.iloc[day_indices[-1]]
                    preco_saida = float(last_v['close']); hora_saida = dados.index[day_indices[-1]].strftime("%H:%M")
                    resultado = "WIN" if ((preco_saida - preco_op) if is_compra else (preco_op - preco_saida)) > 0 else "LOSS"
                
                _raw = (preco_saida - preco_op) if is_compra else (preco_op - preco_saida)
                pts_bruto = round(_raw * 1000, 1) if ativo == "WDO" else round(_raw, 1)
                pts = round(pts_bruto - _custo_pts, 4)
                rs = round(pts * valor_ponto, 2)
                
                # Análise completa
                analise = gerar_analise_completa(setup, vela_info, ativo)
                analise += f"\nRESULTADO: {resultado} | {round(abs(pts),1)}pts | R${round(abs(rs),2)}"
                analise += f"\nEntrada: {preco_op} ({_hora_real}) | Saída: {round(preco_saida,2)} ({hora_saida})"
                
                # Detalhes - classificar janela e obter vwap
                janela_nome, janela_qual, _ = classificar_janela_ct(hora_int, minuto)
                vwap = setup.get("vwap")
                
                detalhes_perda = ""
                detalhes_vitoria = ""
                licao_treino = ""
                
                if resultado == "LOSS":
                    problemas_ct = []
                    licoes_ct = []
                    if velas_na_op <= 1:
                        problemas_ct.append(f"STOP em {velas_na_op} vela = entrada prematura, sem confirmação")
                        licoes_ct.append("Bellafiore: espere reteste/segunda chance")
                    if setup.get("contra_tendencia"):
                        problemas_ct.append(f"CONTRA tendência macro ({tend_macro['tendencia']}) - Elder proíbe")
                        licoes_ct.append("Nunca opere contra Tela 1")
                    if stop_pts < atr_v * 1.0 if ativo == 'WIN' else stop_pts < atr_v * 0.8:
                        problemas_ct.append(f"Stop curto ({stop_pts}pts) vs ATR ({round(atr_v)}pts)")
                        licoes_ct.append("Stop mínimo = 1.5x ATR")
                    if janela_qual in ("RUIM", "PROIBIDO"):
                        problemas_ct.append(f"{janela_nome} ({janela_qual}) = volume baixo")
                    if 40 <= rsi_v <= 60:
                        problemas_ct.append(f"RSI neutro ({rsi_v}) - sem pressão")
                    if not setup["confluencia"].get("price_action_confirma"):
                        problemas_ct.append("Sem candle de confirmação")
                    if not setup["confluencia"].get("sr_relevante"):
                        problemas_ct.append("Longe de S/R")
                    if vwap and ((is_compra and c < vwap) or (not is_compra and c > vwap)):
                        problemas_ct.append(f"Contra VWAP ({round(vwap, 0)})")
                    if losses_consecutivos >= 2:
                        problemas_ct.append(f"{losses_consecutivos} losses seguidos - Tendler: parar")
                    if not problemas_ct:
                        problemas_ct.append(f"Setup {conf_label} correto - Douglas: loss individual é normal")
                        licoes_ct.append("Confie no edge estatístico sobre muitos trades")
                    
                    detalhes_perda = f"STOP em {hora_saida} | -{round(abs(pts),1)}pts | " + " | ".join(problemas_ct[:4])
                    if licoes_ct:
                        detalhes_perda += " | LIÇÃO: " + licoes_ct[0]
                    licao_treino = licoes_ct[0] if licoes_ct else "Loss faz parte do processo"
                    if alerta_memoria and alerta_memoria.get("tem_alerta"):
                        licao_treino += f" | MEMÓRIA: {alerta_memoria['alertas'][0]['licao']}"
                else:
                    acertos_ct = []
                    if setup["confluencia"].get("tendencia_tf_maior"):
                        acertos_ct.append(f"A FAVOR tendência ({tend_macro['tendencia']})")
                    if setup["confluencia"].get("price_action_confirma"):
                        acertos_ct.append("Candle confirmou")
                    if setup["confluencia"].get("sr_relevante"):
                        acertos_ct.append("Entrada em S/R")
                    if janela_qual in ("PRIME", "BOA"):
                        acertos_ct.append(f"{janela_nome} = volume bom")
                    if not acertos_ct:
                        acertos_ct.append(f"Confluência {score}/7 alinhada")
                    detalhes_vitoria = f"ALVO em {hora_saida} | +{round(abs(pts),1)}pts | " + " | ".join(acertos_ct[:4])
                    licao_treino = f"PlayBook: {conf_label} a favor da tendência = alta probabilidade"
                
                operacoes.append({
                    "tipo": tipo_sinal,
                    "hora_entrada": _hora_real,
                    "preco_entrada": round(preco_op, 2),
                    "preco_sinal": round(_preco_sinal, 2),
                    "slippage": _slippage,
                    "custo_pts": _custo_pts,
                    "stop_loss": stop_price,
                    "take_profit": alvo_price,
                    "stop_pts": round(stop_pts * 1000, 1) if ativo == "WDO" else stop_pts,
                    "alvo_pts": round(alvo_pts * 1000, 1) if ativo == "WDO" else alvo_pts,
                    "rr": f"1:{round(alvo_pts/stop_pts, 1)}",
                    "hora_saida": hora_saida,
                    "preco_saida": round(preco_saida, 2),
                    "resultado": resultado,
                    "pts": pts,
                    "resultado_rs": rs,
                    "velas_na_op": velas_na_op,
                    "score": score,
                    "confianca": setup["confianca"],
                    "conf_label": conf_label,
                    "motivos": setup["motivos_operar"],
                    "detalhes_perda": detalhes_perda,
                    "detalhes_vitoria": detalhes_vitoria,
                    "analise_completa": analise,
                    "licao_treino": licao_treino,
                    "alerta_contra": alerta_contra,
                    "alerta_memoria": alerta_memoria,
                    "suporte": round(setup["suporte"], 0) if setup["suporte"] else None,
                    "resistencia": round(setup["resistencia"], 0) if setup["resistencia"] else None,
                    "vwap": round(setup["vwap"], 0) if setup["vwap"] else None,
                    "fib_level": setup["fib_level"],
                    "rsi": rsi_v, "macd_hist": macd_h,
                    "ema9": ema9, "ema21": ema21, "atr": atr_v,
                    "tendencia": tend,
                    "confluencia_detalhes": setup["confluencia"],
                })
                
                total_pts_dia += pts
                if resultado == "LOSS":
                    losses_consecutivos += 1
                    _ct_cool_idx = current_vela_idx + 3  # Cooldown 3 velas (15min) após loss
                    # Operador de verdade: cooldown PROGRESSIVO
                    # 1º loss: 3 velas, 2º: 5 velas, 3º: 8 velas (respira, analisa)
                    cooldown = 3 + losses_consecutivos * 2
                else:
                    losses_consecutivos = 0
                    cooldown = 2  # Cooldown mínimo após win
                
                posicao_aberta = {"close_idx": future_start + velas_na_op + cooldown}
        
        # Registrar na memória
        if operacoes:
            try:
                wins = sum(1 for op in operacoes if op["resultado"] == "WIN")
                total_ops = len(operacoes)
                wr = round(wins / total_ops * 100) if total_ops > 0 else 0
                total_pts_ct = sum(op["pts"] for op in operacoes)
                total_rs_ct = sum(op.get("resultado_rs", 0) for op in operacoes)
                registrar_sessao(ativo, dia_analise.strftime("%d/%m/%Y"), operacoes, {"win_rate": wr, "total_pts": total_pts_ct})
                registrar_historico_completo(
                    ativo=ativo,
                    data_sessao=dia_analise.strftime("%d/%m/%Y"),
                    modo="CT",
                    operacoes=operacoes,
                    performance={
                        "total_operacoes": total_ops, "wins": wins, "losses": total_ops - wins,
                        "win_rate": wr, "total_pts": total_pts_ct, "total_rs": total_rs_ct,
                        "fator_lucro": round(sum(op["pts"] for op in operacoes if op["resultado"]=="WIN") / max(abs(sum(op["pts"] for op in operacoes if op["resultado"]=="LOSS")), 0.01), 2),
                    },
                    tend_macro=tend_macro,
                )
            except: pass
        
        # Resumo
        total_ops = len(operacoes)
        wins = sum(1 for op in operacoes if op["resultado"] == "WIN")
        losses = total_ops - wins
        win_rate = round(wins / total_ops * 100) if total_ops > 0 else 0
        total_pts = sum(op["pts"] for op in operacoes)
        total_rs = sum(op["resultado_rs"] for op in operacoes)
        
        # Comparação com Simulador Real
        ops_a_plus = [op for op in operacoes if op["score"] >= 5]
        ops_b_plus = [op for op in operacoes if op["score"] == 4]
        ops_c_plus = [op for op in operacoes if op["score"] == 3]
        
        wr_a = round(sum(1 for op in ops_a_plus if op["resultado"] == "WIN") / len(ops_a_plus) * 100) if ops_a_plus else 0
        wr_b = round(sum(1 for op in ops_b_plus if op["resultado"] == "WIN") / len(ops_b_plus) * 100) if ops_b_plus else 0
        wr_c = round(sum(1 for op in ops_c_plus if op["resultado"] == "WIN") / len(ops_c_plus) * 100) if ops_c_plus else 0
        
        aprendizados = []
        if wr_a >= 60:
            aprendizados.append(f"Setups A+/A ({len(ops_a_plus)} ops) = {wr_a}% WR - CONFIÁVEIS para Simulador Real")
        if wr_b >= 50:
            aprendizados.append(f"Setups B+ ({len(ops_b_plus)} ops) = {wr_b}% WR - OK para Simulador Real")
        if ops_c_plus:
            aprendizados.append(f"Setups C+ ({len(ops_c_plus)} ops) = {wr_c}% WR - {'ARRISCADO' if wr_c < 50 else 'surpreendeu'} no real")
        
        contra_ops = [op for op in operacoes if op.get("alerta_contra")]
        if contra_ops:
            wr_contra = round(sum(1 for op in contra_ops if op["resultado"] == "WIN") / len(contra_ops) * 100)
            aprendizados.append(f"Contra tendência ({len(contra_ops)} ops) = {wr_contra}% WR - {'CONFIRMA: evitar' if wr_contra < 40 else 'pode funcionar em exceções'}")
        
        return JSONResponse({
            "modo": "TREINAMENTO",
            "dia": dia_anterior.strftime("%d/%m/%Y"),
            "ativo": ativo,
            "contrato": contrato_nome,
            "valor_ponto": valor_ponto,
            "tend_macro": {
                "tendencia": tend_macro["tendencia"],
                "forca": tend_macro["forca"],
                "descricao": tend_macro["descricao"],
            },
            "performance": {
                "total_operacoes": total_ops,
                "wins": wins, "losses": losses,
                "win_rate": win_rate,
                "total_pts": round(total_pts, 1),
                "total_rs": round(total_rs, 2),
            },
            "comparacao_qualidade": {
                "a_plus": {"ops": len(ops_a_plus), "wr": wr_a},
                "b_plus": {"ops": len(ops_b_plus), "wr": wr_b},
                "c_plus": {"ops": len(ops_c_plus), "wr": wr_c},
            },
            "aprendizados": aprendizados,
            "operacoes": operacoes,
            "velas": [v for v in velas_analisadas if v.get('score', 0) >= 3],
            "total_velas": len(velas_analisadas),
        })
    except Exception as e:
        logger.error(f"Erro treinamento-ia: {e}")
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




@app.get("/api/historico-simulador")
async def get_historico_simulador(ativo: str = Query(None), limite: int = Query(20)):
    """Retorna histórico completo de sessões do Simulador Real e CT"""
    try:
        hist = obter_historico(ativo=ativo.upper() if ativo else None, limite=limite)
        # Resumo geral
        total_sessoes = len(hist)
        total_ops = sum(s["performance"]["total_ops"] for s in hist)
        total_wins = sum(s["performance"]["wins"] for s in hist)
        total_pts = round(sum(s["performance"]["total_pts"] for s in hist), 1)
        total_rs = round(sum(s["performance"].get("total_rs", 0) for s in hist), 2)
        wr_global = round(total_wins / total_ops * 100) if total_ops > 0 else 0
        
        # Sessões por modo
        sessoes_real = [s for s in hist if s.get("modo") in ("REAL", "REPLAY")]
        sessoes_ct = [s for s in hist if s.get("modo") == "CT"]
        
        return JSONResponse({
            "total_sessoes": total_sessoes,
            "resumo_global": {
                "total_ops": total_ops,
                "total_wins": total_wins,
                "win_rate": wr_global,
                "total_pts": total_pts,
                "total_rs": total_rs,
            },
            "sessoes_simulador": len(sessoes_real),
            "sessoes_ct": len(sessoes_ct),
            "sessoes": list(reversed(hist)),  # Mais recente primeiro
        })
    except Exception as e:
        logger.error(f"Erro historico: {e}")
        return JSONResponse({"erro": str(e), "sessoes": []}, status_code=500)




@app.get("/api/relatorio-dia")
async def relatorio_dia():
    """
    Relatório consolidado do dia - combina dados de todos os módulos
    (Simulador Real, Centro de Treinamento, Operador) para os 2 usuários.
    Gera métricas de assertividade, média, e análise completa.
    """
    try:
        from learning_engine import carregar_learning, obter_historico
        
        # Carregar learning data (global - compartilhado)
        learning = carregar_learning()
        historico = obter_historico(limite=50)
        
        # Data de hoje
        hoje_str = datetime.now(timezone(timedelta(hours=-3))).strftime("%d/%m/%Y")
        
        # Filtrar sessões de hoje
        sessoes_hoje = [s for s in historico if s.get("data") == hoje_str]
        
        # Se não tem de hoje, pegar o último dia com dados
        if not sessoes_hoje:
            datas_disponiveis = sorted(set(s.get("data", "") for s in historico if s.get("data")), 
                                       key=lambda d: datetime.strptime(d, "%d/%m/%Y") if d else datetime.min,
                                       reverse=True)
            if datas_disponiveis:
                ultimo_dia = datas_disponiveis[0]
                sessoes_hoje = [s for s in historico if s.get("data") == ultimo_dia]
                hoje_str = ultimo_dia
        
        # Separar por modo
        sessoes_simreal = [s for s in sessoes_hoje if s.get("modo") in ("REAL", "REPLAY")]
        sessoes_ct = [s for s in sessoes_hoje if s.get("modo") == "CT"]
        sessoes_operador = [s for s in sessoes_hoje if s.get("modo") == "OPERADOR"]
        
        # Consolidar operações
        todas_ops = []
        for s in sessoes_hoje:
            for op in s.get("operacoes", []):
                op_copy = dict(op)
                op_copy["modo"] = s.get("modo", "?")
                op_copy["sessao_id"] = s.get("id", "?")
                op_copy["ativo"] = s.get("ativo", "?")
                todas_ops.append(op_copy)
        
        total_ops = len(todas_ops)
        wins = sum(1 for op in todas_ops if op.get("resultado") == "WIN")
        losses = total_ops - wins
        win_rate = round(wins / total_ops * 100, 1) if total_ops > 0 else 0
        total_pts = round(sum(op.get("pts", 0) for op in todas_ops), 1)
        total_rs = round(sum(op.get("resultado_rs", 0) for op in todas_ops), 2)
        
        # Ganhos e perdas separados
        ganhos_pts = sum(op.get("pts", 0) for op in todas_ops if op.get("resultado") == "WIN")
        perdas_pts = abs(sum(op.get("pts", 0) for op in todas_ops if op.get("resultado") == "LOSS"))
        fator_lucro = round(ganhos_pts / perdas_pts, 2) if perdas_pts > 0 else (999 if ganhos_pts > 0 else 0)
        
        # Métricas por modo
        def calc_metricas_modo(sessoes):
            ops_m = []
            for s in sessoes:
                ops_m.extend(s.get("operacoes", []))
            t = len(ops_m)
            w = sum(1 for op in ops_m if op.get("resultado") == "WIN")
            pts = round(sum(op.get("pts", 0) for op in ops_m), 1)
            rs = round(sum(op.get("resultado_rs", 0) for op in ops_m), 2)
            return {
                "sessoes": len(sessoes),
                "total_ops": t,
                "wins": w,
                "losses": t - w,
                "win_rate": round(w / t * 100, 1) if t > 0 else 0,
                "total_pts": pts,
                "total_rs": rs,
            }
        
        # Métricas por ativo
        ativos_no_dia = list(set(s.get("ativo", "WIN") for s in sessoes_hoje))
        metricas_por_ativo = {}
        for at in ativos_no_dia:
            ops_at = [op for op in todas_ops if op.get("ativo") == at]
            t = len(ops_at)
            w = sum(1 for op in ops_at if op.get("resultado") == "WIN")
            pts = round(sum(op.get("pts", 0) for op in ops_at), 1)
            rs = round(sum(op.get("resultado_rs", 0) for op in ops_at), 2)
            metricas_por_ativo[at] = {
                "total_ops": t, "wins": w, "losses": t - w,
                "win_rate": round(w / t * 100, 1) if t > 0 else 0,
                "total_pts": pts, "total_rs": rs,
            }
        
        # Melhor e pior trade
        melhor = max(todas_ops, key=lambda x: x.get("pts", 0)) if todas_ops else None
        pior = min(todas_ops, key=lambda x: x.get("pts", 0)) if todas_ops else None
        
        # Análise por horário
        por_hora = {}
        for op in todas_ops:
            h = op.get("hora_entrada", "00:00")[:2]
            if h not in por_hora:
                por_hora[h] = {"total": 0, "wins": 0, "pts": 0}
            por_hora[h]["total"] += 1
            if op.get("resultado") == "WIN":
                por_hora[h]["wins"] += 1
            por_hora[h]["pts"] += op.get("pts", 0)
        for h in por_hora:
            por_hora[h]["win_rate"] = round(por_hora[h]["wins"] / por_hora[h]["total"] * 100) if por_hora[h]["total"] > 0 else 0
            por_hora[h]["pts"] = round(por_hora[h]["pts"], 1)
        
        # Análise por setup
        por_setup = {}
        for op in todas_ops:
            setup = op.get("conf_label", "?")
            if setup not in por_setup:
                por_setup[setup] = {"total": 0, "wins": 0, "pts": 0}
            por_setup[setup]["total"] += 1
            if op.get("resultado") == "WIN":
                por_setup[setup]["wins"] += 1
            por_setup[setup]["pts"] += op.get("pts", 0)
        for s in por_setup:
            por_setup[s]["win_rate"] = round(por_setup[s]["wins"] / por_setup[s]["total"] * 100) if por_setup[s]["total"] > 0 else 0
            por_setup[s]["pts"] = round(por_setup[s]["pts"], 1)
        
        # Erros mais comuns do dia
        erros_dia = []
        for op in todas_ops:
            if op.get("resultado") == "LOSS" and op.get("detalhes_perda"):
                erros_dia.append({
                    "hora": op.get("hora_entrada", "?"),
                    "tipo": op.get("tipo", "?"),
                    "pts": op.get("pts", 0),
                    "detalhes": op.get("detalhes_perda", "")[:200],
                    "modo": op.get("modo", "?"),
                })
        
        # Memória inteligente
        mem_info = {
            "total_erros_memoria": len(learning.get("memoria_erros", [])),
            "regras_ativas": len(learning.get("regras_aprendidas", [])),
            "win_rate_global": learning.get("win_rate_global", 0),
            "total_ops_global": learning.get("total_operacoes", 0),
            "regras": learning.get("regras_aprendidas", [])[:10],
        }
        
        # Histórico dos últimos 7 dias
        todas_datas = sorted(set(s.get("data", "") for s in historico if s.get("data")),
                            key=lambda d: datetime.strptime(d, "%d/%m/%Y") if d else datetime.min)
        historico_7d = []
        for dt in todas_datas[-7:]:
            sess_dt = [s for s in historico if s.get("data") == dt]
            ops_dt = []
            for s in sess_dt:
                ops_dt.extend(s.get("operacoes", []))
            t_dt = len(ops_dt)
            w_dt = sum(1 for op in ops_dt if op.get("resultado") == "WIN")
            pts_dt = round(sum(op.get("pts", 0) for op in ops_dt), 1)
            historico_7d.append({
                "data": dt,
                "total_ops": t_dt,
                "wins": w_dt,
                "win_rate": round(w_dt / t_dt * 100, 1) if t_dt > 0 else 0,
                "total_pts": pts_dt,
            })
        
        # Veredicto do dia
        if total_ops == 0:
            veredicto = "SEM OPERAÇÕES"
            veredicto_detalhe = "Nenhuma operação realizada hoje."
            nota = 0
        elif win_rate >= 70 and total_pts > 0:
            veredicto = "DIA EXCELENTE"
            veredicto_detalhe = f"Win rate de {win_rate}% com lucro de {total_pts}pts. Bellafiore: dia do PlayBook!"
            nota = 5
        elif win_rate >= 55 and total_pts > 0:
            veredicto = "DIA BOM"
            veredicto_detalhe = f"Acima de 55% de acerto e lucrativo. Consistência é o caminho."
            nota = 4
        elif total_pts > 0:
            veredicto = "DIA POSITIVO"
            veredicto_detalhe = f"Lucro no dia apesar de WR abaixo de 55%. R:R compensou."
            nota = 3
        elif total_pts == 0:
            veredicto = "DIA NEUTRO"
            veredicto_detalhe = "Zero a zero. Axiomas: preservou capital."
            nota = 2
        else:
            veredicto = "DIA DE PREJUÍZO"
            veredicto_detalhe = f"Perda de {abs(total_pts)}pts. Revisar erros e ajustar amanhã."
            nota = 1
        
        return {
            "data": hoje_str,
            "veredicto": veredicto,
            "veredicto_detalhe": veredicto_detalhe,
            "nota": nota,
            "consolidado": {
                "total_sessoes": len(sessoes_hoje),
                "total_operacoes": total_ops,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total_pts": total_pts,
                "total_rs": total_rs,
                "fator_lucro": fator_lucro,
                "ganhos_pts": round(ganhos_pts, 1),
                "perdas_pts": round(perdas_pts, 1),
            },
            "por_modo": {
                "simulador_real": calc_metricas_modo(sessoes_simreal),
                "centro_treinamento": calc_metricas_modo(sessoes_ct),
                "operador": calc_metricas_modo(sessoes_operador),
            },
            "por_ativo": metricas_por_ativo,
            "por_hora": por_hora,
            "por_setup": por_setup,
            "melhor_trade": {
                "tipo": melhor.get("tipo"), "hora": melhor.get("hora_entrada"),
                "pts": melhor.get("pts"), "setup": melhor.get("conf_label"),
                "motivos": melhor.get("motivos", [])[:3],
            } if melhor else None,
            "pior_trade": {
                "tipo": pior.get("tipo"), "hora": pior.get("hora_entrada"),
                "pts": pior.get("pts"), "setup": pior.get("conf_label"),
                "detalhes": pior.get("detalhes_perda", "")[:200],
            } if pior else None,
            "erros_dia": erros_dia[:10],
            "memoria": mem_info,
            "historico_7d": historico_7d,
            "operacoes": [{
                "tipo": op.get("tipo"), "hora_entrada": op.get("hora_entrada"),
                "hora_saida": op.get("hora_saida"), "resultado": op.get("resultado"),
                "pts": op.get("pts"), "resultado_rs": op.get("resultado_rs"),
                "score": op.get("score"), "conf_label": op.get("conf_label"),
                "modo": op.get("modo"), "ativo": op.get("ativo"),
                "motivos": op.get("motivos", [])[:3],
                "detalhes_perda": op.get("detalhes_perda", "")[:150],
                "detalhes_vitoria": op.get("detalhes_vitoria", "")[:150],
            } for op in todas_ops],
        }
    except Exception as e:
        logger.error(f"Erro no relatorio-dia: {e}")
        import traceback
        traceback.print_exc()
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
