"""
LEARNING ENGINE - Sistema de Aprendizado Automático
A AI aprende com seus wins e losses, ajusta pesos, evolui sozinha.
Salva histórico de operações e analisa padrões de sucesso/fracasso.
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
    "versao": 1,
    "total_sessoes": 0,
    "total_operacoes": 0,
    "total_wins": 0,
    "total_losses": 0,
    "win_rate_global": 0,
    "total_pts": 0,
    "total_rs": 0,
    # Pesos adaptativos - começam em 1.0, AI ajusta baseado em resultados
    "pesos": {
        "horario": 1.0,
        "rsi": 1.0,
        "ema": 1.0,
        "tendencia": 1.0,
        "macd": 1.0,
        "atr": 1.0,
        "suporte_resistencia": 1.0,
        "vwap": 1.0,
        "fibonacci": 1.0,
        "candlestick": 1.0,
    },
    # Score mínimo adaptativo - começa em 7, AI pode subir se estiver perdendo muito
    "score_minimo": 7,
    # Padrões aprendidos
    "padroes_vitoria": {},
    "padroes_derrota": {},
    # Melhores e piores horários
    "horarios_win_rate": {},
    # Histórico de sessões
    "sessoes": [],
    # Insights gerados pela AI
    "insights": [],
    # Livros/skills estudados
    "livros_estudados": [],
}


def carregar_learning():
    """Carrega dados de aprendizado do disco"""
    try:
        if LEARNING_FILE.exists():
            with open(LEARNING_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Merge with defaults for new fields
            for k, v in DEFAULT_LEARNING.items():
                if k not in data:
                    data[k] = v
            if "pesos" in data:
                for k, v in DEFAULT_LEARNING["pesos"].items():
                    if k not in data["pesos"]:
                        data["pesos"][k] = v
            return data
    except Exception as e:
        logger.error(f"Erro carregando learning: {e}")
    return DEFAULT_LEARNING.copy()


def salvar_learning(data):
    """Salva dados de aprendizado no disco"""
    try:
        with open(LEARNING_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erro salvando learning: {e}")


def registrar_sessao(ativo, dia, operacoes, resumo):
    """
    Registra uma sessão completa de trading (simulador real).
    Analisa cada operação, extrai padrões, ajusta pesos.
    """
    data = carregar_learning()
    
    wins = [op for op in operacoes if op.get("resultado") == "WIN"]
    losses = [op for op in operacoes if op.get("resultado") == "LOSS"]
    
    sessao = {
        "data": dia,
        "ativo": ativo,
        "timestamp": datetime.now(BRT).isoformat(),
        "total_ops": len(operacoes),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(operacoes) * 100) if operacoes else 0,
        "total_pts": sum(op.get("pts", 0) for op in operacoes),
        "total_rs": sum(op.get("resultado_rs", 0) for op in operacoes),
    }
    
    # Atualizar totais globais
    data["total_sessoes"] += 1
    data["total_operacoes"] += len(operacoes)
    data["total_wins"] += len(wins)
    data["total_losses"] += len(losses)
    data["total_pts"] += sessao["total_pts"]
    data["total_rs"] += sessao["total_rs"]
    if data["total_operacoes"] > 0:
        data["win_rate_global"] = round(data["total_wins"] / data["total_operacoes"] * 100, 1)
    
    # ---- ANÁLISE DE PADRÕES ----
    
    # 1. Quais indicadores estavam presentes nos WINs vs LOSSes?
    for op in operacoes:
        motivos = op.get("motivos", [])
        resultado = op.get("resultado", "")
        hora = op.get("hora_entrada", "00:00")
        hora_key = hora[:2]  # "09", "10", etc
        
        # Horário win rate
        if hora_key not in data["horarios_win_rate"]:
            data["horarios_win_rate"][hora_key] = {"wins": 0, "losses": 0, "total": 0}
        data["horarios_win_rate"][hora_key]["total"] += 1
        if resultado == "WIN":
            data["horarios_win_rate"][hora_key]["wins"] += 1
        else:
            data["horarios_win_rate"][hora_key]["losses"] += 1
        
        # Padrões por motivo
        for motivo in motivos:
            # Simplificar o motivo para key
            key = motivo.split("(")[0].strip().lower()[:50]
            
            target = data["padroes_vitoria"] if resultado == "WIN" else data["padroes_derrota"]
            if key not in target:
                target[key] = 0
            target[key] += 1
    
    # 2. AJUSTAR PESOS baseado em resultados
    _ajustar_pesos(data, operacoes)
    
    # 3. AJUSTAR SCORE MÍNIMO
    _ajustar_score_minimo(data, sessao)
    
    # 4. GERAR INSIGHTS
    _gerar_insights(data, sessao, operacoes)
    
    # Guardar sessão (últimas 50)
    data["sessoes"].append(sessao)
    if len(data["sessoes"]) > 50:
        data["sessoes"] = data["sessoes"][-50:]
    
    salvar_learning(data)
    return data


def _ajustar_pesos(data, operacoes):
    """Ajusta pesos dos indicadores baseado em quais indicadores estavam presentes em WINs vs LOSSes"""
    ajuste = 0.05  # Ajuste por sessão (conservador)
    
    indicador_map = {
        "horario": ["horario forte", "horario"],
        "rsi": ["rsi", "sobrevendido", "sobrecomprado"],
        "ema": ["ema9", "ema", "estrutura compradora", "estrutura vendedora"],
        "tendencia": ["tendencia", "triple screen"],
        "macd": ["macd", "momentum"],
        "atr": ["atr", "volatilidade"],
        "suporte_resistencia": ["suporte", "resistencia", "s/r murphy"],
        "vwap": ["vwap"],
        "fibonacci": ["fibonacci", "fib"],
        "candlestick": ["candlestick", "martelo", "engolfo", "estrela"],
    }
    
    for op in operacoes:
        motivos = " ".join(op.get("motivos", [])).lower()
        resultado = op.get("resultado", "")
        
        for indicador, keywords in indicador_map.items():
            presente = any(kw in motivos for kw in keywords)
            if presente:
                if resultado == "WIN":
                    # Indicador presente no WIN = aumenta peso
                    data["pesos"][indicador] = min(2.0, data["pesos"][indicador] + ajuste)
                elif resultado == "LOSS":
                    # Indicador presente no LOSS = diminui peso (mas não abaixo de 0.3)
                    data["pesos"][indicador] = max(0.3, data["pesos"][indicador] - ajuste * 0.5)


def _ajustar_score_minimo(data, sessao):
    """Se win rate está baixo, sobe o score mínimo. Se alto, pode baixar."""
    if data["total_operacoes"] < 10:
        return  # Precisa de dados suficientes
    
    wr = data["win_rate_global"]
    
    if wr < 50:
        # Perdendo muito - ser mais seletivo
        data["score_minimo"] = min(9, data["score_minimo"] + 1)
        data["insights"].append({
            "data": datetime.now(BRT).isoformat(),
            "tipo": "AJUSTE",
            "mensagem": f"Win rate {wr}% abaixo de 50%. Score minimo subiu para {data['score_minimo']}. Preciso ser MAIS SELETIVO."
        })
    elif wr > 75 and data["score_minimo"] > 6:
        # Ganhando muito - pode relaxar um pouco
        data["score_minimo"] = max(6, data["score_minimo"] - 1)
        data["insights"].append({
            "data": datetime.now(BRT).isoformat(),
            "tipo": "AJUSTE",
            "mensagem": f"Win rate {wr}% acima de 75%! Score minimo pode baixar para {data['score_minimo']}."
        })


def _gerar_insights(data, sessao, operacoes):
    """Gera insights automáticos sobre padrões detectados"""
    
    # Melhor horário
    if data["horarios_win_rate"]:
        melhor_hora = None
        melhor_wr = 0
        pior_hora = None
        pior_wr = 100
        for hora, stats in data["horarios_win_rate"].items():
            if stats["total"] >= 3:  # Mínimo 3 amostras
                wr = round(stats["wins"] / stats["total"] * 100)
                if wr > melhor_wr:
                    melhor_wr = wr
                    melhor_hora = hora
                if wr < pior_wr:
                    pior_wr = wr
                    pior_hora = hora
        
        if melhor_hora:
            data["insights"].append({
                "data": datetime.now(BRT).isoformat(),
                "tipo": "PADRAO",
                "mensagem": f"Melhor horario: {melhor_hora}h ({melhor_wr}% WR). Pior: {pior_hora}h ({pior_wr}% WR). Priorizar entradas no horario forte."
            })
    
    # Padrão de operações longas (muitas velas = baixo WR?)
    ops_longas = [op for op in operacoes if op.get("velas_na_op", 0) > 15]
    ops_curtas = [op for op in operacoes if op.get("velas_na_op", 0) <= 6]
    
    if len(ops_longas) >= 2:
        wr_longas = sum(1 for op in ops_longas if op["resultado"] == "WIN") / len(ops_longas) * 100
        if wr_longas < 40:
            data["insights"].append({
                "data": datetime.now(BRT).isoformat(),
                "tipo": "ALERTA",
                "mensagem": f"Operacoes longas (>15 velas) tem WR de {round(wr_longas)}%. Considere reduzir timeout ou ser mais agressivo no stop."
            })
    
    if len(ops_curtas) >= 2:
        wr_curtas = sum(1 for op in ops_curtas if op["resultado"] == "WIN") / len(ops_curtas) * 100
        if wr_curtas > 70:
            data["insights"].append({
                "data": datetime.now(BRT).isoformat(),
                "tipo": "POSITIVO",
                "mensagem": f"Operacoes rapidas (<=6 velas) tem WR de {round(wr_curtas)}%! Mercado ja estava no ponto - bons setups."
            })
    
    # Limitar insights (últimos 30)
    if len(data["insights"]) > 30:
        data["insights"] = data["insights"][-30:]


def obter_pesos_atuais():
    """Retorna os pesos adaptativos atuais para uso no scoring"""
    data = carregar_learning()
    return data.get("pesos", DEFAULT_LEARNING["pesos"])


def obter_score_minimo():
    """Retorna o score mínimo adaptativo"""
    data = carregar_learning()
    return data.get("score_minimo", 7)


def obter_resumo_aprendizado():
    """Retorna resumo do aprendizado para exibição"""
    data = carregar_learning()
    return {
        "total_sessoes": data["total_sessoes"],
        "total_operacoes": data["total_operacoes"],
        "win_rate_global": data["win_rate_global"],
        "total_pts": round(data["total_pts"], 1),
        "total_rs": round(data["total_rs"], 2),
        "score_minimo": data["score_minimo"],
        "pesos": data["pesos"],
        "horarios_win_rate": data["horarios_win_rate"],
        "insights": data.get("insights", [])[-10:],  # Últimos 10
        "livros_estudados": data.get("livros_estudados", []),
        "sessoes_recentes": data.get("sessoes", [])[-5:],  # Últimas 5
    }


def registrar_livro(titulo, autor, conceitos_chave):
    """Registra um livro estudado e seus conceitos incorporados"""
    data = carregar_learning()
    data["livros_estudados"].append({
        "titulo": titulo,
        "autor": autor,
        "conceitos": conceitos_chave,
        "data_estudo": datetime.now(BRT).isoformat(),
    })
    salvar_learning(data)
