# B3 Day Trade Analyzer - Guia de Deploy

## O que este projeto faz
Analisa Mini-Indice (WIN) e Mini-Dolar (WDO) da B3 em tempo real com Fibonacci, RSI, MACD, Volume, Detector de Violinada e Sinais de Entrada.

## Deploy no Railway
1. No Railway, clique em + New > Deploy from GitHub repo
2. Selecione este repositorio
3. Adicione variavel: DATA_SOURCE=yfinance
4. Deploy automatico!

## Endpoints
- GET / = Dashboard web
- GET /api/analise?ativo=WIN&timeframe=5m = Analise completa
- GET /api/painel?ativo=WIN = Painel multi-timeframe
- GET /api/sinais?ativo=WIN = Sinais de entrada
- GET /api/status = Status do sistema
- POST /api/forcar-atualizacao = Forca refresh

## AVISO
Ferramenta de apoio a decisao. NAO garante lucros. Use gerenciamento de risco.