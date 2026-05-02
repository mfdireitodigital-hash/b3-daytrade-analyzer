FROM openjdk:17-slim

# Install Android SDK
RUN apt-get update && apt-get install -y wget unzip && \
    mkdir -p /opt/android-sdk/cmdline-tools && \
    wget -q "https://dl.google.com/android/repository/commandlinetools-linux-10406996_latest.zip" -O /tmp/tools.zip && \
    unzip -q /tmp/tools.zip -d /opt/android-sdk/cmdline-tools && \
    mv /opt/android-sdk/cmdline-tools/cmdline-tools /opt/android-sdk/cmdline-tools/latest && \
    rm /tmp/tools.zip

ENV ANDROID_HOME=/opt/android-sdk
ENV PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools

# Accept licenses and install required SDK packages
RUN yes | sdkmanager --licenses > /dev/null 2>&1 && \
    sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"

# Copy project
WORKDIR /app
COPY . .

# Build APK
RUN chmod +x gradlew || true
RUN gradle wrapper --gradle-version=8.2 || true

CMD ["sh", "-c", "gradle assembleDebug && cp app/build/outputs/apk/debug/app-debug.apk /output/B3DayTrade.apk && echo 'APK pronto em /output/B3DayTrade.apk'"]
