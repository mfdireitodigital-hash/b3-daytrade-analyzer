#!/bin/bash
echo "========================================="
echo "  B3 Day Trade - Build APK"
echo "========================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERRO: Docker nao encontrado!"
    echo "Instale Docker Desktop: https://docker.com/products/docker-desktop"
    exit 1
fi

echo "[1/3] Construindo imagem Docker..."
docker build -t b3trade-builder .

echo "[2/3] Gerando APK..."
mkdir -p output
docker run --rm -v "$(pwd)/output:/output" b3trade-builder

echo "[3/3] Verificando..."
if [ -f output/B3DayTrade.apk ]; then
    echo ""
    echo "========================================="
    echo "  APK GERADO COM SUCESSO!"
    echo "  Arquivo: output/B3DayTrade.apk"
    echo "  Tamanho: $(du -h output/B3DayTrade.apk | cut -f1)"
    echo "========================================="
    echo ""
    echo "Para instalar no celular:"
    echo "  1. Envie o arquivo para o celular (WhatsApp/cabo)"
    echo "  2. Config > Seguranca > Permitir fontes desconhecidas"
    echo "  3. Abra o .apk e instale"
else
    echo "ERRO: Falha ao gerar APK"
    exit 1
fi
