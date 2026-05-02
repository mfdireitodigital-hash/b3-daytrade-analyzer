# Como Gerar o APK

## Opção 1: Android Studio (Mais Fácil)
1. Abra Android Studio
2. File > Open > selecione a pasta `android-app/`
3. Aguarde o Gradle sincronizar
4. Build > Build Bundle(s) / APK(s) > Build APK(s)
5. O APK estará em: `app/build/outputs/apk/debug/app-debug.apk`

## Opção 2: Linha de Comando
```bash
cd android-app
./gradlew assembleDebug
```
O APK estará em: `app/build/outputs/apk/debug/app-debug.apk`

## Opção 3: PWA (Sem compilar nada!)
1. No celular Android, abra Chrome
2. Acesse: https://web-production-caf2f.up.railway.app/
3. Toque nos 3 pontinhos (menu) > "Instalar app" ou "Adicionar à tela inicial"
4. Pronto! O app aparece como ícone na home, abre fullscreen, sem barra do Chrome

## Instalar o APK no celular
1. Transfira o APK para o celular (WhatsApp, email, cabo USB)
2. Configurações > Segurança > Permitir fontes desconhecidas
3. Abra o arquivo .apk e instale
