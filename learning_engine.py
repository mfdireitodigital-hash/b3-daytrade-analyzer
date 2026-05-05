"""
LEARNING ENGINE v2 - Sistema de Aprendizado com Memória Persistente
A AI aprende com seus wins e losses, LEMBRA dos erros, e NUNCA repete.

Baseado em:
- Bellafiore (PlayBook): Revisar todo trade, melhorar continuamente
- Douglas (Trading in the Zone): Pensar em probabilidades
- Tendler (Mental Game): Identificar padrões de tilt e corrigi-los
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
BRT = timezone(timedelta(hours=-3))

LEARNING_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / "learning_data.json"

DEFAULT_LEARNING = {
    "versao": 2,
    "total_sessoes": 0,
    "total_operacoes": 0,
    "total_wins": 0,
    "total_losses": 0,
    "win_rate_global": 0,
    "total_pts": 0,
    "total_rs": 0,
    "pesos": {
        "horario": 1.0, "rsi": 1.0, "macd": 1.0, "ema": 1.0,
        "atr": 1.0, "vwap": 1.0, "fibonacci": 1.0, "candlestick": 1.0,
        "suporte_resistencia": 1.0, "tendencia": 1.0,
        "price_action": 1.0, "smc": 1.0, "volume": 1.0,
    },
    "sessoes": [],
    "livros_aplicados": [],
    # V2: Memória de erros
    "memoria_erros": [],       # Cada erro com fingerprint + contexto + lição
    "regras_aprendidas": [],   # Regras auto-geradas (CUIDADO / FAVORÁVEL)
    "situacoes_perigosas": {}, # Padrão -> {losses, total, taxa_loss}
    "situacoes_seguras": {},   # Padrão -> {wins, total, taxa_win}
    "evolucao": [],            # Registro de evolução ao longo do tempo
}


def carregar_learning() -> dict:
    try:
        if LEARNING_FILE.exists():
            with open(LEARNING_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Migrar v1 para v2
            if data.get("versao", 1) < 2:
                for key in ["memoria_erros", "regras_aprendidas", "evolucao"]:
                    if key not in data:
                        data[key] = DEFAULT_LEARNING[key]
                for key in ["situacoes_perigosas", "situacoes_seguras"]:
                    if key not in data:
                        data[key] = DEFAULT_LEARNING[key]
                data["versao"] = 2
                _salvar(data)
            return data
    except Exception as e:
        logger.error(f"Erro carregando learning: {e}")
    return DEFAULT_LEARNING.copy()


def _salvar(data: dict):
    try:
        with open(LEARNING_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Erro salvando learning: {e}")


def _extrair_fingerprint(op: dict) -> dict:
    """Extrai 'impressão digital' de uma operação para comparação futura."""
    hora = op.get("hora_entrada", "00:00")
    hora_h = int(hora.split(":")[0]) if ":" in str(hora) else 0
    
    # Faixa horária
    if hora_h < 10: faixa = "abertura"
    elif hora_h < 12: faixa = "manha"
    elif hora_h < 14: faixa = "almoco"
    elif hora_h < 16: faixa = "tarde"
    else: faixa = "fechamento"
    
    # RSI zona
    rsi = op.get("rsi", 50)
    if rsi < 30: rsi_zona = "sobrevendido"
    elif rsi > 70: rsi_zona = "sobrecomprado"
    elif rsi < 45: rsi_zona = "vendedor"
    elif rsi > 55: rsi_zona = "comprador"
    else: rsi_zona = "neutro"
    
    # MACD direção
    macd = op.get("macd_hist", 0)
    macd_dir = "positivo" if macd > 0 else "negativo" if macd < 0 else "zero"
    
    return {
        "tipo": op.get("tipo", ""),
        "hora_faixa": faixa,
        "tendencia": op.get("tendencia", "LATERAL"),
        "rsi_zona": rsi_zona,
        "macd_direcao": macd_dir,
        "score": op.get("score", 0),
        "conf_label": op.get("conf_label", ""),
        "motivos": op.get("motivos", [])[:5],
    }


def _comparar_situacoes(fp1: dict, fp2: dict) -> float:
    """Compara duas 'impressões digitais' e retorna similaridade 0-100%."""
    score = 0
    total = 0
    
    # Tipo (COMPRA/VENDA) - peso 2
    total += 2
    if fp1.get("tipo") == fp2.get("tipo"):
        score += 2
    
    # Faixa horária - peso 1
    total += 1
    if fp1.get("hora_faixa") == fp2.get("hora_faixa"):
        score += 1
    
    # Tendência - peso 2
    total += 2
    if fp1.get("tendencia") == fp2.get("tendencia"):
        score += 2
    
    # RSI zona - peso 1.5
    total += 1.5
    if fp1.get("rsi_zona") == fp2.get("rsi_zona"):
        score += 1.5
    
    # MACD direção - peso 1
    total += 1
    if fp1.get("macd_direcao") == fp2.get("macd_direcao"):
        score += 1
    
    # Score similar - peso 1
    total += 1
    s1 = fp1.get("score", 0)
    s2 = fp2.get("score", 0)
    if abs(s1 - s2) <= 1:
        score += 1
    elif abs(s1 - s2) <= 2:
        score += 0.5
    
    return round(score / total * 100, 1) if total > 0 else 0


def consultar_memoria(op_atual: dict) -> dict:
    """
    Consulta memória de erros para verificar se situação atual é similar a um erro passado.
    Retorna alerta se similaridade >= 70%.
    """
    data = carregar_learning()
    fp_atual = _extrair_fingerprint(op_atual)
    
    alertas = []
    for erro in data.get("memoria_erros", [])[-50:]:  # Últimos 50 erros
        fp_erro = erro.get("fingerprint", {})
        similaridade = _comparar_situacoes(fp_atual, fp_erro)
        
        if similaridade >= 70:
            alertas.append({
                "similaridade": similaridade,
                "erro_data": erro.get("data", ""),
                "licao": erro.get("licao", ""),
                "contexto": erro.get("contexto", ""),
                "fingerprint_erro": fp_erro,
            })
    
    # Verificar regras aprendidas
    regras_ativas = []
    for regra in data.get("regras_aprendidas", []):
        if regra.get("tipo") == "CUIDADO":
            # Verificar se a regra se aplica
            padrao = regra.get("padrao", "")
            chave_atual = f"{fp_atual.get('tipo')}_{fp_atual.get('hora_faixa')}_{fp_atual.get('tendencia')}"
            if padrao and padrao in chave_atual:
                regras_ativas.append(regra)
    
    return {
        "tem_alerta": len(alertas) > 0,
        "alertas": sorted(alertas, key=lambda x: -x["similaridade"])[:3],
        "regras_ativas": regras_ativas,
        "total_erros_memoria": len(data.get("memoria_erros", [])),
    }


def _gerar_licao(op: dict) -> str:
    """Gera lição específica a partir de uma operação perdedora."""
    tipo = op.get("tipo", "")
    tend = op.get("tendencia", "")
    rsi = op.get("rsi", 50)
    hora = op.get("hora_entrada", "")
    score = op.get("score", 0)
    conf = op.get("conf_label", "")
    detalhes = op.get("detalhes_perda", "")
    
    licoes = []
    
    # Contra tendência
    if (tipo == "COMPRA" and tend == "BAIXA") or (tipo == "VENDA" and tend == "ALTA"):
        licoes.append(f"{tipo} contra tendência {tend} - Elder proíbe (Triple Screen)")
    
    # RSI extremo
    if (tipo == "COMPRA" and rsi > 75) or (tipo == "VENDA" and rsi < 25):
        licoes.append(f"RSI extremo ({rsi}) na hora da entrada - mercado esticado")
    
    # Horário ruim
    hora_h = int(hora.split(":")[0]) if ":" in str(hora) else 0
    if hora_h in [12, 13]:
        licoes.append("Horário do almoço - baixa liquidez, evitar")
    elif hora_h >= 17:
        licoes.append("Fim do pregão - evitar novas entradas")
    
    # Setup fraco
    if score < 4:
        licoes.append(f"Confluência baixa ({score}/7 = {conf}) - Bellafiore: só A+/B+")
    
    if not licoes:
        licoes.append(f"Loss com setup {conf} - faz parte (Douglas: distribuição aleatória)")
    
    return "; ".join(licoes)


def _gerar_padrao_key(op: dict) -> str:
    """Gera chave de padrão para tracking estatístico."""
    fp = _extrair_fingerprint(op)
    return f"{fp['tipo']}_{fp['hora_faixa']}_{fp['tendencia']}_{fp['rsi_zona']}"


def registrar_sessao(ativo: str, data_sessao: str, operacoes: list, metricas: dict) -> dict:
    """Registra sessão completa com memória v2."""
    data = carregar_learning()
    
    data["total_sessoes"] += 1
    
    for op in operacoes:
        data["total_operacoes"] += 1
        resultado = op.get("resultado", "")
        pts = op.get("pts", 0)
        rs = op.get("resultado_rs", 0)
        
        if resultado == "WIN":
            data["total_wins"] += 1
        else:
            data["total_losses"] += 1
        
        data["total_pts"] += pts
        data["total_rs"] += rs
        
        # Fingerprint
        fp = _extrair_fingerprint(op)
        padrao_key = _gerar_padrao_key(op)
        
        if resultado == "LOSS":
            # Registrar na memória de erros
            erro = {
                "data": data_sessao,
                "ativo": ativo,
                "fingerprint": fp,
                "contexto": f"{op.get('tipo', '')} em {op.get('hora_entrada', '')} - {op.get('tendencia', '')} - Score {op.get('score', 0)} ({op.get('conf_label', '')})",
                "licao": _gerar_licao(op),
                "pts_perdidos": pts,
                "detalhes": op.get("detalhes_perda", "")[:200],
            }
            data["memoria_erros"].append(erro)
            
            # Atualizar situações perigosas
            if padrao_key not in data["situacoes_perigosas"]:
                data["situacoes_perigosas"][padrao_key] = {"losses": 0, "total": 0, "taxa_loss": 0}
            data["situacoes_perigosas"][padrao_key]["losses"] += 1
            data["situacoes_perigosas"][padrao_key]["total"] += 1
        else:
            # Atualizar situações seguras
            if padrao_key not in data["situacoes_seguras"]:
                data["situacoes_seguras"][padrao_key] = {"wins": 0, "total": 0, "taxa_win": 0}
            data["situacoes_seguras"][padrao_key]["wins"] += 1
            data["situacoes_seguras"][padrao_key]["total"] += 1
        
        # Atualizar contagem total em ambos dicts
        for d in [data["situacoes_perigosas"], data["situacoes_seguras"]]:
            if padrao_key in d:
                total = d[padrao_key]["total"]
                if "losses" in d[padrao_key]:
                    d[padrao_key]["taxa_loss"] = round(d[padrao_key]["losses"] / total * 100, 1) if total > 0 else 0
                if "wins" in d[padrao_key]:
                    d[padrao_key]["taxa_win"] = round(d[padrao_key]["wins"] / total * 100, 1) if total > 0 else 0
    
    # Gerar regras aprendidas
    data["regras_aprendidas"] = _gerar_regras(data)
    
    # Win rate global
    total = data["total_wins"] + data["total_losses"]
    data["win_rate_global"] = round(data["total_wins"] / total * 100, 1) if total > 0 else 0
    
    # Registrar sessão
    sessao = {
        "data": data_sessao,
        "ativo": ativo,
        "ops": len(operacoes),
        "wins": sum(1 for op in operacoes if op.get("resultado") == "WIN"),
        "losses": sum(1 for op in operacoes if op.get("resultado") != "WIN"),
        "pts": round(sum(op.get("pts", 0) for op in operacoes), 1),
        "win_rate": metricas.get("win_rate", 0),
        "timestamp": datetime.now(BRT).isoformat(),
    }
    data["sessoes"].append(sessao)
    
    # Evolução
    data["evolucao"].append({
        "data": data_sessao,
        "win_rate_sessao": metricas.get("win_rate", 0),
        "win_rate_global": data["win_rate_global"],
        "total_erros_memoria": len(data["memoria_erros"]),
        "regras_ativas": len(data["regras_aprendidas"]),
    })
    
    # Limpar memória antiga (manter últimas 100 sessões e 200 erros)
    if len(data["sessoes"]) > 100:
        data["sessoes"] = data["sessoes"][-100:]
    if len(data["memoria_erros"]) > 200:
        data["memoria_erros"] = data["memoria_erros"][-200:]
    if len(data["evolucao"]) > 100:
        data["evolucao"] = data["evolucao"][-100:]
    
    # Atualizar pesos baseado em resultados
    _atualizar_pesos(data, operacoes)
    
    _salvar(data)
    return data


def _gerar_regras(data: dict) -> list:
    """Gera regras automáticas baseadas em padrões perigosos/seguros."""
    regras = []
    
    # CUIDADO: padrões com 80%+ loss rate e 3+ amostras
    for padrao, stats in data.get("situacoes_perigosas", {}).items():
        if stats.get("total", 0) >= 3 and stats.get("taxa_loss", 0) >= 80:
            regras.append({
                "tipo": "CUIDADO",
                "padrao": padrao,
                "descricao": f"Padrão {padrao} tem {stats['taxa_loss']}% de loss ({stats['losses']}/{stats['total']})",
                "recomendacao": "EVITAR este tipo de entrada",
            })
    
    # FAVORÁVEL: padrões com 70%+ win rate e 3+ amostras
    for padrao, stats in data.get("situacoes_seguras", {}).items():
        if stats.get("total", 0) >= 3 and stats.get("taxa_win", 0) >= 70:
            regras.append({
                "tipo": "FAVORAVEL",
                "padrao": padrao,
                "descricao": f"Padrão {padrao} tem {stats['taxa_win']}% de win ({stats['wins']}/{stats['total']})",
                "recomendacao": "PRIORIZAR este tipo de entrada",
            })
    
    return regras


def _atualizar_pesos(data: dict, operacoes: list):
    """Ajusta pesos adaptativos baseado nos resultados."""
    for op in operacoes:
        motivos = op.get("motivos", [])
        resultado = op.get("resultado", "")
        ajuste = 0.02 if resultado == "WIN" else -0.01
        
        for motivo in motivos:
            motivo_lower = motivo.lower()
            for indicador in data["pesos"]:
                if indicador.lower() in motivo_lower:
                    data["pesos"][indicador] = max(0.5, min(2.0, data["pesos"][indicador] + ajuste))


def obter_pesos_atuais() -> dict:
    return carregar_learning().get("pesos", DEFAULT_LEARNING["pesos"])


def obter_score_minimo() -> int:
    """Score mínimo adaptativo baseado no aprendizado. Agora em escala de confluência (7)."""
    data = carregar_learning()
    wr = data.get("win_rate_global", 0)
    # Com o novo sistema de confluência (0-7), mínimo é 4
    if wr >= 65:
        return 3  # Se WR alto, pode flexibilizar
    elif wr >= 50:
        return 4  # Padrão
    else:
        return 4  # Manter 4 até melhorar


def obter_resumo_aprendizado() -> dict:
    data = carregar_learning()
    return {
        "versao": data.get("versao", 1),
        "total_sessoes": data.get("total_sessoes", 0),
        "total_operacoes": data.get("total_operacoes", 0),
        "win_rate_global": data.get("win_rate_global", 0),
        "total_pts": round(data.get("total_pts", 0), 1),
        "total_rs": round(data.get("total_rs", 0), 2),
        "pesos": data.get("pesos", {}),
        "ultimas_sessoes": data.get("sessoes", [])[-5:],
        "memoria_erros_total": len(data.get("memoria_erros", [])),
        "regras_aprendidas": data.get("regras_aprendidas", []),
        "evolucao": data.get("evolucao", [])[-10:],
    }


def registrar_livro(nome: str, conceitos: list):
    data = carregar_learning()
    if nome not in data.get("livros_aplicados", []):
        data.setdefault("livros_aplicados", []).append(nome)
        _salvar(data)


def registrar_trade_replay(ativo: str, op: dict) -> dict:
    """
    Registra um trade INDIVIDUAL do replay/CT na memória persistente.
    Diferente de registrar_sessao que recebe lista de ops.
    
    op deve ter: tipo, hora_entrada, resultado, pts, resultado_rs, 
                 tendencia, rsi, score, conf_label, motivos, detalhes_perda
    """
    data = carregar_learning()
    
    data["total_operacoes"] += 1
    resultado = op.get("resultado", "LOSS")
    pts = op.get("pts", 0)
    rs = op.get("resultado_rs", 0)
    
    if resultado == "WIN":
        data["total_wins"] += 1
    else:
        data["total_losses"] += 1
    
    data["total_pts"] += pts
    data["total_rs"] += rs
    
    # Fingerprint
    fp = _extrair_fingerprint(op)
    padrao_key = _gerar_padrao_key(op)
    
    if resultado == "LOSS":
        erro = {
            "data": datetime.now(BRT).strftime("%d/%m/%Y %H:%M"),
            "ativo": ativo,
            "fingerprint": fp,
            "contexto": f"{op.get('tipo', '')} em {op.get('hora_entrada', '')} - {op.get('tendencia', '')} - Score {op.get('score', 0)} ({op.get('conf_label', '')})",
            "licao": _gerar_licao(op),
            "pts_perdidos": pts,
            "detalhes": op.get("detalhes_perda", "")[:200],
            "origem": "replay_ct",
        }
        data["memoria_erros"].append(erro)
        
        if padrao_key not in data["situacoes_perigosas"]:
            data["situacoes_perigosas"][padrao_key] = {"losses": 0, "total": 0, "taxa_loss": 0}
        data["situacoes_perigosas"][padrao_key]["losses"] += 1
        data["situacoes_perigosas"][padrao_key]["total"] += 1
    else:
        if padrao_key not in data["situacoes_seguras"]:
            data["situacoes_seguras"][padrao_key] = {"wins": 0, "total": 0, "taxa_win": 0}
        data["situacoes_seguras"][padrao_key]["wins"] += 1
        data["situacoes_seguras"][padrao_key]["total"] += 1
    
    # Atualizar taxas
    for d in [data["situacoes_perigosas"], data["situacoes_seguras"]]:
        if padrao_key in d:
            total = d[padrao_key]["total"]
            if "losses" in d[padrao_key]:
                d[padrao_key]["taxa_loss"] = round(d[padrao_key]["losses"] / total * 100, 1) if total > 0 else 0
            if "wins" in d[padrao_key]:
                d[padrao_key]["taxa_win"] = round(d[padrao_key]["wins"] / total * 100, 1) if total > 0 else 0
    
    # Regras
    data["regras_aprendidas"] = _gerar_regras(data)
    
    # Win rate global
    total = data["total_wins"] + data["total_losses"]
    data["win_rate_global"] = round(data["total_wins"] / total * 100, 1) if total > 0 else 0
    
    # Limitar memória
    if len(data["memoria_erros"]) > 200:
        data["memoria_erros"] = data["memoria_erros"][-200:]
    
    _salvar(data)
    
    return {
        "gravado": True,
        "total_operacoes": data["total_operacoes"],
        "win_rate_global": data["win_rate_global"],
        "memoria_erros": len(data["memoria_erros"]),
        "regras_ativas": len(data["regras_aprendidas"]),
        "licao": erro["licao"] if resultado == "LOSS" else None,
    }


# ================================================================
# HISTÓRICO COMPLETO DE SESSÕES (Simulador Real + CT)
# ================================================================
HISTORICO_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / "historico_sessoes.json"

def _carregar_historico() -> list:
    try:
        if HISTORICO_FILE.exists():
            with open(HISTORICO_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return []

def _salvar_historico(hist: list):
    # Manter últimas 50 sessões
    if len(hist) > 50:
        hist = hist[-50:]
    with open(HISTORICO_FILE, "w") as f:
        json.dump(hist, f, ensure_ascii=False, indent=1)

def registrar_historico_completo(ativo: str, data_sessao: str, modo: str, operacoes: list, performance: dict, tend_macro: dict = None) -> dict:
    """Salva sessão completa com todas as operações e análises detalhadas."""
    hist = _carregar_historico()
    
    # Preparar operações para salvar (limpar campos pesados)
    ops_salvar = []
    for op in operacoes:
        ops_salvar.append({
            "tipo": op.get("tipo"),
            "hora_entrada": op.get("hora_entrada"),
            "hora_saida": op.get("hora_saida"),
            "preco_entrada": op.get("preco_entrada"),
            "preco_saida": op.get("preco_saida"),
            "stop_loss": op.get("stop_loss"),
            "take_profit": op.get("take_profit"),
            "stop_pts": op.get("stop_pts"),
            "alvo_pts": op.get("alvo_pts"),
            "rr": op.get("rr"),
            "resultado": op.get("resultado"),
            "pts": op.get("pts"),
            "resultado_rs": op.get("resultado_rs"),
            "velas_na_op": op.get("velas_na_op"),
            "score": op.get("score"),
            "conf_label": op.get("conf_label"),
            "motivos": op.get("motivos", [])[:5],
            "detalhes_perda": op.get("detalhes_perda", ""),
            "detalhes_vitoria": op.get("detalhes_vitoria", ""),
            "analise_completa": op.get("analise_completa", "")[:500],
            "licao_treino": op.get("licao_treino", ""),
            "alerta_contra": op.get("alerta_contra", ""),
            "tendencia": op.get("tendencia"),
            "rsi": op.get("rsi"),
            "vwap": op.get("vwap"),
            "suporte": op.get("suporte"),
            "resistencia": op.get("resistencia"),
            "confluencia_detalhes": op.get("confluencia_detalhes", {}),
        })
    
    sessao = {
        "id": hashlib.md5(f"{ativo}{data_sessao}{datetime.now(BRT).isoformat()}".encode()).hexdigest()[:10],
        "ativo": ativo,
        "data": data_sessao,
        "modo": modo,  # "REAL" ou "REPLAY" ou "CT"
        "timestamp": datetime.now(BRT).isoformat(),
        "tend_macro": {
            "tendencia": tend_macro.get("tendencia", "?") if tend_macro else "?",
            "forca": tend_macro.get("forca", 0) if tend_macro else 0,
        },
        "performance": {
            "total_ops": performance.get("total_operacoes", len(operacoes)),
            "wins": performance.get("wins", 0),
            "losses": performance.get("losses", 0),
            "win_rate": performance.get("win_rate", 0),
            "total_pts": round(performance.get("total_pts", 0), 1),
            "total_rs": round(performance.get("total_rs", 0), 2),
            "fator_lucro": performance.get("fator_lucro", 0),
        },
        "operacoes": ops_salvar,
    }
    
    hist.append(sessao)
    _salvar_historico(hist)
    return sessao

def obter_historico(ativo: str = None, limite: int = 20) -> list:
    """Retorna histórico de sessões, opcionalmente filtrado por ativo."""
    hist = _carregar_historico()
    if ativo:
        hist = [s for s in hist if s.get("ativo") == ativo]
    return hist[-limite:]

import hashlib
