"""
TRADING BOOKS KNOWLEDGE BASE
Conceitos-chave extraídos dos maiores livros de trading do mundo.
Integrados ao Learning Engine e ao Scoring System.
"""

# ===================================================================
# LIVROS ADICIONAIS INCORPORADOS (além dos 7 já no trader-pro)
# ===================================================================

LIVROS_CONHECIMENTO = {
    # ---- AL BROOKS - Price Action ----
    "al_brooks_price_action": {
        "titulo": "Trading Price Action (Trilogy)",
        "autor": "Al Brooks",
        "conceitos": [
            "TODA barra (vela) é um sinal - não existe barra sem informação",
            "Price action puro: opere o que VÊ no gráfico, não o que indicadores dizem",
            "Barras de sinal: barra com corpo grande NA DIREÇÃO do trade = boa. Doji/sombras grandes = ruim",
            "Always In: identifique se o mercado está Always In Long, Short, ou em Range",
            "Second Entry: a 2ª entrada na mesma direção é mais confiável que a 1ª",
            "Measured Move: projeção do alvo = tamanho do impulso anterior",
            "Breakout Pullback: após rompimento, espere pullback para entrar. 1o pullback = mais forte",
            "Tight Channel: em canal apertado, NÃO opere contra. Espere rompimento",
            "Trading Range: 80% dos rompimentos de range FALHAM. Fade os extremos",
            "Final Flag: último pullback antes de reversão. Reconheça = lucro enorme",
            "Gap: abertura longe do fechamento anterior = sinal forte. Gap fechado no dia = reversão",
            "Spike and Channel: impulso + canal = tendência saudável. Opere pullbacks no canal",
        ],
        "regras_scoring": {
            "barra_sinal_forte": "+1 se corpo > 60% do range da vela e na direção do sinal",
            "second_entry": "+1 se é a 2ª vela de sinal na mesma direção (após pullback)",
            "always_in": "+1 se sempre-no-mercado (trend) confirma direção",
            "tight_channel_contra": "-2 se operando contra canal apertado",
        }
    },
    
    # ---- JESSE LIVERMORE ----
    "jesse_livermore": {
        "titulo": "How to Trade in Stocks / Reminiscences of a Stock Operator",
        "autor": "Jesse Livermore / Edwin Lefèvre",
        "conceitos": [
            "O mercado NUNCA está errado - opiniões SIM. Siga o mercado, não seu ego",
            "Pivotal Points: pontos de pivô (máx/mín anteriores) são onde o mercado decide",
            "Timing: estar CERTO sobre a direção mas ERRADO sobre o timing = PERDER dinheiro",
            "Pyramiding: adicione posição SO quando o mercado confirma (lucro acumulado)",
            "Nunca faça preço médio em posição PERDEDORA - isso é o caminho da ruína",
            "Regra da Paciência: O dinheiro grande NÃO está no trading, está no ESPERAR",
            "Mercado de ALTA não dura pra sempre. Mercado de BAIXA tb não. Ciclos",
            "A coisa mais cara do mundo para um especulador é a ESPERANÇA",
            "Opere somente quando TODAS as condições estão a seu favor. Fique fora o resto",
        ],
        "regras_scoring": {
            "pivotal_point": "+1 se preço está em máxima/mínima de referência (high/low 20 períodos)",
            "paciencia": "+0 mas NÃO opere com score < mínimo (Livermore: espere o setup perfeito)",
        }
    },
    
    # ---- LARRY WILLIAMS ----
    "larry_williams": {
        "titulo": "Long-Term Secrets to Short-Term Trading",
        "autor": "Larry Williams",
        "conceitos": [
            "O mercado se move em ciclos de EXPANSÃO e CONTRAÇÃO de range",
            "Dias de range grande tendem a ser seguidos por dias de range pequeno (e vice-versa)",
            "Setup da Volatilidade: ATR contraído = explosão iminente. Opere o breakout",
            "Inside Day: dia com máx<máx anterior e mín>mín anterior = acumulação. Breakout forte",
            "OOPS: abertura gap forte + reversão = sinal poderoso contra o gap",
            "Commitment of Traders (COT): institucionais (comerciais) geralmente estão CERTOS",
            "Dia da semana importa: terça/quarta têm mais follow-through que segunda/sexta",
            "O melhor indicador é PREÇO + VOLUME + TEMPO. Nada mais necessário",
            "Money Management é 90% do jogo. Position sizing > indicadores",
        ],
        "regras_scoring": {
            "volatilidade_contraida": "+1 se ATR atual < 70% ATR médio (expansão iminente)",
            "dia_semana": "+0.5 se terça ou quarta (melhor follow-through estatístico)",
        }
    },
    
    # ---- VAN THARP ----
    "van_tharp": {
        "titulo": "Trade Your Way to Financial Freedom / Super Trader",
        "autor": "Van K. Tharp",
        "conceitos": [
            "Você NÃO opera o mercado. Você opera suas CRENÇAS sobre o mercado",
            "Expectancy: E = (WR × Avg Win) - (LR × Avg Loss). E > 0 = sistema lucrativo",
            "Position Sizing é O fator mais importante: determina quanto você ganha/perde NO TOTAL",
            "R-Multiple: meça tudo em R (risco inicial). Win de 3R compensa 3 losses de 1R",
            "Sistema de trading tem 3 partes: Setup (filtro) + Entry (gatilho) + Exit (stop/alvo)",
            "Quality of trade = R-multiple. Média de R alto = trader excelente",
            "SQN (System Quality Number): mede qualidade do sistema. >2 = bom. >3 = excelente. >7 = Santo Graal",
            "10 tarefas do trading: pesquisa, estudo mental, plano negócios, plano trading, regras, monitorar",
            "Seu sistema deve refletir SUA personalidade. Não copie - adapte",
        ],
        "regras_scoring": {
            "rr_minimo": "+1 se R:R >= 1.5 (Van Tharp: R-multiple mínimo para expectancy positiva)",
            "expectancy_check": "Sistema deve ter WR*AvgWin > LR*AvgLoss (verificar no learning engine)",
        }
    },
    
    # ---- JOHN CARTER ----
    "john_carter": {
        "titulo": "Mastering the Trade",
        "autor": "John F. Carter",
        "conceitos": [
            "Squeeze: Bollinger dentro de Keltner = compressão EXTREMA. Breakout explosivo",
            "TTM Squeeze: quando o squeeze dispara, fique NO LADO do momentum",
            "TICK: NYSE TICK extremo (>+800 ou <-800) = reversão de curto prazo",
            "TRIN: <0.5 = muito comprador (possível topo). >2.0 = muito vendedor (possível fundo)",
            "VIX: Volatilidade alta = MEDO = oportunidade de compra (contrarian)",
            "Best trade setups: aberturas de mercado (15-30min), squeeze releases, pivôs de dia anterior",
            "Worst times: hora do almoço (11:30-13:30), última hora de sexta-feira",
            "Use MÚLTIPLOS timeframes: 5min para entrada, 15min para filtro, 60min para tendência",
            "Trading plan é um contrato consigo mesmo. Quebre o contrato = quebre a disciplina",
        ],
        "regras_scoring": {
            "squeeze_bollinger": "+1 se Bollinger Bands contraídas (BB width < média)",
            "horario_carter": "+1 se 09:15-10:30 ou 14:00-15:30 (melhor liquidez Carter)",
        }
    },
    
    # ---- STEVE NISON ----
    "steve_nison": {
        "titulo": "Japanese Candlestick Charting Techniques",
        "autor": "Steve Nison",
        "conceitos": [
            "Candlesticks são REAÇÕES EMOCIONAIS em forma de gráfico",
            "Martelo/Hanging Man: mesma forma, contexto muda tudo (fundo=martelo, topo=enforcado)",
            "Engulfing: vela que 'engole' a anterior = mudança de poder. Mais forte em S/R",
            "Morning Star/Evening Star: padrão de 3 velas = reversão forte com confirmação",
            "Doji em zona de S/R: indecisão EXATAMENTE onde importa = setup de reversão",
            "Harami: vela pequena dentro da anterior = perda de momentum. Espere confirmação",
            "Three White Soldiers / Three Black Crows: 3 velas consecutivas = tendência FORTE",
            "NUNCA use candle patterns ISOLADOS. Sempre com tendência + S/R + volume",
            "Quanto maior o timeframe do padrão, mais confiável ele é",
            "Sombras longas = rejeição de preço. Corpo grande = convicção direcional",
        ],
        "regras_scoring": {
            "candle_em_sr": "+2 se padrão de candle ocorre em zona de suporte/resistência",
            "sombra_rejeicao": "+1 se sombra > 2x corpo (rejeição forte de nível)",
        }
    },
    
    # ---- NASSIM TALEB ----
    "nassim_taleb": {
        "titulo": "Fooled by Randomness / The Black Swan / Antifragile",
        "autor": "Nassim Nicholas Taleb",
        "conceitos": [
            "A maioria dos lucros de trading vem de POUCOS trades. A maioria é ruído",
            "Survivorship bias: você vê os traders que deram certo, não os milhares que quebraram",
            "Antifragilidade: posicione-se para ganhar com a volatilidade, não para sofrer com ela",
            "Stop loss curto + alvo longo = você aceita muitos losses pequenos por poucos gains grandes",
            "NÃO confie em backtest que mostra 90% win rate. Provavelmente é curve fitting",
            "Distribuição fat-tail: eventos extremos acontecem MAIS do que modelos preveem",
            "Barbell strategy: 90% seguro + 10% risco alto = antifragil",
            "O que NÃO te mata te fortalece (se você aprende com a perda)",
        ],
        "regras_scoring": {
            "assimetria": "+1 se R:R >= 2.0 (Taleb: assimetria positiva protege contra cisne negro)",
        }
    },
    
    # ---- ADAM GRIMES ----
    "adam_grimes": {
        "titulo": "The Art and Science of Technical Analysis",
        "autor": "Adam Grimes",
        "conceitos": [
            "Mercados alternam entre TENDÊNCIA e RANGE. Identifique QUAL estado antes de operar",
            "Pullback em tendência: a MELHOR estratégia de trading com maior win rate histórico",
            "Breakout de range: funciona MENOS do que as pessoas pensam. Fades são mais confiáveis",
            "Confluência: cada fator adicional MULTIPLICA probabilidade (não soma)",
            "Estatística: precisamos de pelo menos 30 trades para qualquer conclusão válida",
            "Quantifique TUDO: se não pode medir, não pode melhorar",
            "Padrões gráficos funcionam porque refletem PSICOLOGIA de massa, não magia",
            "A edge está na EXECUÇÃO, não no conhecimento do setup",
        ],
        "regras_scoring": {
            "pullback_tendencia": "+1 se entrada é pullback (RSI voltando de extremo) em tendência definida",
        }
    },
}


def obter_todos_conceitos():
    """Retorna lista flat de todos os conceitos de todos os livros"""
    conceitos = []
    for livro_id, livro in LIVROS_CONHECIMENTO.items():
        for c in livro["conceitos"]:
            conceitos.append({
                "conceito": c,
                "livro": livro["titulo"],
                "autor": livro["autor"],
            })
    return conceitos


def obter_regras_scoring_extras():
    """Retorna regras de scoring adicionais baseadas nos livros"""
    regras = {}
    for livro_id, livro in LIVROS_CONHECIMENTO.items():
        for regra_id, descricao in livro.get("regras_scoring", {}).items():
            regras[regra_id] = {
                "descricao": descricao,
                "livro": livro["titulo"],
                "autor": livro["autor"],
            }
    return regras


def obter_livros_lista():
    """Retorna lista resumida de todos os livros para exibição"""
    return [
        {
            "id": lid,
            "titulo": l["titulo"],
            "autor": l["autor"],
            "conceitos_count": len(l["conceitos"]),
        }
        for lid, l in LIVROS_CONHECIMENTO.items()
    ]


def aplicar_scoring_avancado(vela_data, tipo_sinal, tend, rsi_v, atr_v, macd_h, c, o, h, l, suporte, resistencia, ema9, ema21, dados_window):
    """
    Aplica regras de scoring avançadas baseadas nos livros.
    Retorna (score_extra, motivos_extra).
    """
    score_extra = 0
    motivos = []
    
    body = abs(c - o)
    upper_shadow = h - max(c, o)
    lower_shadow = min(c, o) - l
    total_range = h - l if h > l else 1
    
    # AL BROOKS: Barra de sinal forte (corpo > 60% do range)
    if body > total_range * 0.6:
        is_bull = c > o
        if (is_bull and tipo_sinal == "COMPRA") or (not is_bull and tipo_sinal == "VENDA"):
            score_extra += 1
            motivos.append("Barra de sinal forte - corpo >60% (Al Brooks)")
    
    # AL BROOKS: Sombra de rejeição (Nison tb)
    if tipo_sinal == "COMPRA" and lower_shadow > body * 2:
        score_extra += 1
        motivos.append("Sombra inferior longa = rejeição de queda (Brooks/Nison)")
    elif tipo_sinal == "VENDA" and upper_shadow > body * 2:
        score_extra += 1
        motivos.append("Sombra superior longa = rejeição de alta (Brooks/Nison)")
    
    # NISON: Candle pattern em zona de S/R (vale +2 em vez de +1)
    if suporte and resistencia:
        if tipo_sinal == "COMPRA" and abs(c - suporte) < atr_v * 0.3:
            score_extra += 1
            motivos.append("Candle em zona de SUPORTE (Nison: padrão em S/R = alta confiança)")
        elif tipo_sinal == "VENDA" and abs(resistencia - c) < atr_v * 0.3:
            score_extra += 1
            motivos.append("Candle em zona de RESISTENCIA (Nison: padrão em S/R = alta confiança)")
    
    # GRIMES: Pullback em tendência (melhor setup estatístico)
    if tend == "ALTA" and tipo_sinal == "COMPRA" and 35 <= rsi_v <= 50:
        score_extra += 1
        motivos.append("Pullback de RSI em tendência ALTA (Grimes: melhor win rate histórico)")
    elif tend == "BAIXA" and tipo_sinal == "VENDA" and 50 <= rsi_v <= 65:
        score_extra += 1
        motivos.append("Pullback de RSI em tendência BAIXA (Grimes: melhor win rate histórico)")
    
    # LARRY WILLIAMS: Volatilidade contraída (ATR baixo = explosão)
    if dados_window is not None and len(dados_window) >= 20:
        try:
            from analysis_engine import calcular_atr_series
            atr_series = calcular_atr_series(dados_window)
            if len(atr_series) >= 10:
                atr_media = float(atr_series.iloc[-10:].mean())
                if atr_v < atr_media * 0.7:
                    score_extra += 1
                    motivos.append(f"ATR contraído ({atr_v} < média {round(atr_media)}) - expansão iminente (Larry Williams)")
        except:
            pass
    
    # TALEB: Assimetria positiva (R:R >= 2.0)
    # Já calculado no stop/alvo, mas motivamos aqui
    
    return score_extra, motivos
