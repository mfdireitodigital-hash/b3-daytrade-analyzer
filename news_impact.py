# =====================================================
# MODULO: Impacto de Noticias nas Entradas
# =====================================================
"""
Avalia como noticias economicas impactam decisoes de entrada.
- Eventos de ALTO IMPACTO (Payroll, FOMC, SELIC, CPI) bloqueiam entradas 30min antes/depois
- Surpresas (acima/abaixo) ajustam vies direcional
- Integra como modificador de score nos endpoints simulador-real e treinamento-ia
"""

import re
import json
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)
BRT = timezone(timedelta(hours=-3))

# Eventos de MAXIMO impacto - bloqueiam operacoes
EVENTOS_CRITICOS = [
    "payroll", "nonfarm", "non-farm", "fomc", "fed fund",
    "taxa de juros", "interest rate", "cpi", "ipc",
    "selic", "copom", "gdp", "pib"
]

# Eventos de ALTO impacto - cautela
EVENTOS_ALTO = [
    "pmi", "ism", "emprego", "employment", "unemployment",
    "vendas no varejo", "retail sales", "producao industrial",
    "industrial production", "pedidos de bens", "durable goods",
    "balanca comercial", "trade balance", "ipca", "inflacao"
]

def obter_noticias_do_dia():
    """Busca noticias de impacto do dia atual via investing.com"""
    try:
        import urllib.request, urllib.parse
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://br.investing.com/economic-calendar/",
            "Accept": "application/json, text/javascript, */*",
        }
        
        params = urllib.parse.urlencode({
            "country[]": ["25", "5"],
            "importance[]": "3",
            "timeZone": "12",
            "timeFilter": "timeRemain",
            "currentTab": "today",
        }, doseq=True)
        
        req = urllib.request.Request(
            "https://br.investing.com/economic-calendar/Service/getCalendarFilteredData",
            data=params.encode(),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode())
            html_data = raw.get("data", "")
        
        eventos = []
        rows = html_data.split('js-event-item')
        
        for row in rows[1:]:
            try:
                time_m = re.search(r'js-time"[^>]*>([^<]*)<', row)
                evt_time = time_m.group(1).strip() if time_m else ""
                
                cur_m = re.search(r'ceFlags[^>]*>[^<]*</span>\s*(\w{3})', row)
                currency = cur_m.group(1).strip() if cur_m else ""
                
                name_m = re.search(r'class="[^"]*event[^"]*"[^>]*>([^<]+)<', row)
                evt_name = name_m.group(1).strip() if name_m else ""
                evt_name = evt_name.replace("&amp;", "&").replace("&nbsp;", " ").replace("&#39;", "'")
                
                bold_vals = re.findall(r'<td[^>]*class="[^"]*bold[^"]*"[^>]*>\s*([^<]*?)\s*</td>', row)
                if not bold_vals:
                    bold_vals = re.findall(r'bold[^>]*>([^<]*)<', row)
                
                actual = bold_vals[0].strip().replace("&nbsp;", "") if len(bold_vals) >= 1 else ""
                forecast = bold_vals[1].strip().replace("&nbsp;", "") if len(bold_vals) >= 2 else ""
                
                if evt_time and evt_name:
                    evt_lower = evt_name.lower()
                    nivel = "NORMAL"
                    if any(kw in evt_lower for kw in EVENTOS_CRITICOS):
                        nivel = "CRITICO"
                    elif any(kw in evt_lower for kw in EVENTOS_ALTO):
                        nivel = "ALTO"
                    
                    surpresa = None
                    if actual and forecast:
                        try:
                            act_num = float(actual.replace("%","").replace(",",".").replace("K","000").replace("M","000000").strip())
                            for_num = float(forecast.replace("%","").replace(",",".").replace("K","000").replace("M","000000").strip())
                            if act_num > for_num: surpresa = "ACIMA"
                            elif act_num < for_num: surpresa = "ABAIXO"
                            else: surpresa = "NEUTRO"
                        except:
                            pass
                    
                    eventos.append({
                        "hora": evt_time,
                        "nome": evt_name,
                        "moeda": currency,
                        "nivel": nivel,
                        "surpresa": surpresa,
                        "actual": actual,
                        "forecast": forecast,
                        "ja_divulgado": bool(actual),
                    })
            except:
                continue
        
        return eventos
    except Exception as e:
        logger.warning(f"Erro ao buscar noticias: {e}")
        return []


def avaliar_impacto_noticias(hora_vela: str, ativo: str, direcao_setup: str, noticias: list = None):
    """
    Avalia o impacto das noticias na decisao de entrada.
    
    Args:
        hora_vela: "HH:MM" da vela sendo analisada
        ativo: "WIN" ou "WDO"
        direcao_setup: "COMPRA" ou "VENDA" ou None
        noticias: lista de eventos (se None, busca automaticamente)
    
    Returns:
        dict com:
        - modificador_score: int (-2 a +1) para somar ao score
        - bloquear: bool - se deve bloquear a entrada
        - motivo: str - explicacao
        - alerta: str ou None - alerta para exibir
        - vies_noticias: str ou None - "COMPRA"/"VENDA" se noticias indicam direcao
    """
    resultado = {
        "modificador_score": 0,
        "bloquear": False,
        "motivo": "",
        "alerta": None,
        "vies_noticias": None,
        "eventos_proximos": [],
    }
    
    if not noticias:
        return resultado
    
    try:
        h_vela, m_vela = map(int, hora_vela.split(":"))
        minutos_vela = h_vela * 60 + m_vela
    except:
        return resultado
    
    alertas = []
    modificador_total = 0
    bloquear = False
    vies = None
    
    for evt in noticias:
        try:
            evt_hora = evt.get("hora", "")
            if not evt_hora or ":" not in evt_hora:
                continue
            
            h_evt, m_evt = map(int, evt_hora.split(":"))
            minutos_evt = h_evt * 60 + m_evt
            diff_min = minutos_vela - minutos_evt  # positivo = vela depois do evento
            
            nivel = evt.get("nivel", "NORMAL")
            moeda = evt.get("moeda", "")
            nome = evt.get("nome", "")
            surpresa = evt.get("surpresa")
            ja_divulgado = evt.get("ja_divulgado", False)
            
            # Relevancia para o ativo
            relevante = False
            if moeda == "USD":
                relevante = True  # USD impacta WIN e WDO
            elif moeda == "BRL":
                relevante = True  # BRL impacta ambos
            
            if not relevante:
                continue
            
            # ---- EVENTO CRITICO (Payroll, FOMC, SELIC, CPI) ----
            if nivel == "CRITICO":
                if not ja_divulgado:
                    # Evento ainda nao saiu
                    if -30 <= diff_min <= 0:
                        # 30min ANTES: BLOQUEAR
                        bloquear = True
                        modificador_total -= 2
                        alertas.append(f"BLOQUEIO: {nome} em {abs(diff_min)}min - NÃO OPERAR")
                    elif -60 <= diff_min < -30:
                        # 60-30min antes: cautela
                        modificador_total -= 1
                        alertas.append(f"CAUTELA: {nome} em {abs(diff_min)}min")
                else:
                    # Evento ja saiu
                    if 0 <= diff_min <= 15:
                        # Primeiros 15min apos: volatilidade extrema
                        bloquear = True
                        modificador_total -= 2
                        alertas.append(f"BLOQUEIO: {nome} acabou de sair - aguardar estabilizar")
                    elif 15 < diff_min <= 30:
                        # 15-30min apos: ainda volátil mas pode operar com cuidado
                        modificador_total -= 1
                        alertas.append(f"CAUTELA: {nome} divulgado há {diff_min}min - volatilidade alta")
                    
                    # Ajustar vies baseado na surpresa
                    if surpresa and 5 <= diff_min <= 120:
                        if moeda == "USD":
                            if surpresa == "ACIMA":
                                # USD forte = WDO sobe, WIN cai
                                vies = "COMPRA" if ativo == "WDO" else "VENDA"
                                alertas.append(f"{nome}: dado ACIMA → dólar forte")
                            elif surpresa == "ABAIXO":
                                vies = "VENDA" if ativo == "WDO" else "COMPRA"
                                alertas.append(f"{nome}: dado ABAIXO → dólar fraco")
                        elif moeda == "BRL":
                            if surpresa == "ACIMA":
                                vies = "COMPRA" if ativo == "WIN" else "VENDA"
                                alertas.append(f"{nome}: dado ACIMA → real forte")
                            elif surpresa == "ABAIXO":
                                vies = "VENDA" if ativo == "WIN" else "COMPRA"
                                alertas.append(f"{nome}: dado ABAIXO → real fraco")
                        
                        # Bonus se setup alinhado com vies
                        if vies and direcao_setup == vies:
                            modificador_total += 1
                            alertas.append(f"BONUS: setup alinhado com notícia ({vies})")
                        elif vies and direcao_setup and direcao_setup != vies:
                            modificador_total -= 1
                            alertas.append(f"PENALIDADE: setup CONTRA a notícia")
            
            # ---- EVENTO ALTO (PMI, emprego, vendas) ----
            elif nivel == "ALTO":
                if not ja_divulgado and -20 <= diff_min <= 0:
                    modificador_total -= 1
                    alertas.append(f"CAUTELA: {nome} em {abs(diff_min)}min")
                elif ja_divulgado and 0 <= diff_min <= 10:
                    modificador_total -= 1
                    alertas.append(f"CAUTELA: {nome} acabou de sair")
                elif ja_divulgado and surpresa and 10 < diff_min <= 60:
                    if moeda == "USD":
                        if surpresa == "ACIMA":
                            vies = "COMPRA" if ativo == "WDO" else "VENDA"
                        elif surpresa == "ABAIXO":
                            vies = "VENDA" if ativo == "WDO" else "COMPRA"
            
            # Registrar eventos proximos (30min range)
            if abs(diff_min) <= 60 and nivel in ("CRITICO", "ALTO"):
                resultado["eventos_proximos"].append({
                    "nome": nome, "hora": evt_hora, "nivel": nivel,
                    "diff_min": diff_min, "surpresa": surpresa
                })
        except:
            continue
    
    resultado["modificador_score"] = max(-2, min(1, modificador_total))
    resultado["bloquear"] = bloquear
    resultado["motivo"] = " | ".join(alertas) if alertas else "Sem notícias de impacto próximas"
    resultado["alerta"] = alertas[0] if alertas else None
    resultado["vies_noticias"] = vies
    
    return resultado
