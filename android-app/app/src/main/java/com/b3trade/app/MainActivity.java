package com.b3trade.app;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Vibrator;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.webkit.CookieManager;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

public class MainActivity extends AppCompatActivity {

    private static final String APP_URL = "https://web-production-caf2f.up.railway.app/";
    private WebView webView;
    private SwipeRefreshLayout swipeRefresh;
    private FrameLayout errorLayout;
    private FrameLayout loadingLayout;
    private boolean isLoaded = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Fullscreen immersive
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        );
        getWindow().setStatusBarColor(Color.parseColor("#0a0f1a"));
        getWindow().setNavigationBarColor(Color.parseColor("#0a0f1a"));
        getWindow().getDecorView().setSystemUiVisibility(
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE |
            View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN |
            View.SYSTEM_UI_FLAG_FULLSCREEN |
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        );

        setContentView(R.layout.activity_main);

        webView = findViewById(R.id.webView);
        swipeRefresh = findViewById(R.id.swipeRefresh);
        errorLayout = findViewById(R.id.errorLayout);
        loadingLayout = findViewById(R.id.loadingLayout);

        setupWebView();
        setupSwipeRefresh();

        if (isNetworkAvailable()) {
            loadApp();
        } else {
            showError("Sem conexão com a internet.\nConecte-se para operar.");
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private void setupWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);
        settings.setDatabaseEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(true);
        settings.setSupportZoom(false);
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);
        settings.setUserAgentString(settings.getUserAgentString() + " B3TradeApp/1.0");

        // Enable cookies
        CookieManager.getInstance().setAcceptCookie(true);
        CookieManager.getInstance().setAcceptThirdPartyCookies(webView, true);

        // Hardware acceleration
        webView.setLayerType(View.LAYER_TYPE_HARDWARE, null);
        webView.setBackgroundColor(Color.parseColor("#0a0f1a"));

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageStarted(WebView view, String url, Bitmap favicon) {
                if (!isLoaded) {
                    loadingLayout.setVisibility(View.VISIBLE);
                }
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                isLoaded = true;
                loadingLayout.setVisibility(View.GONE);
                errorLayout.setVisibility(View.GONE);
                webView.setVisibility(View.VISIBLE);
                swipeRefresh.setRefreshing(false);

                // Inject mobile optimizations
                view.evaluateJavascript(
                    "document.body.style.overscrollBehavior='none';" +
                    "document.documentElement.style.overscrollBehavior='none';",
                    null
                );
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                if (request.isForMainFrame()) {
                    showError("Erro ao conectar ao servidor.\nVerifique sua conexão.");
                }
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                String url = request.getUrl().toString();
                if (url.startsWith(APP_URL) || url.startsWith("https://web-production")) {
                    return false; // Load in WebView
                }
                // External links open in browser
                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
                startActivity(intent);
                return true;
            }
        });

        webView.setWebChromeClient(new WebChromeClient());
    }

    private void setupSwipeRefresh() {
        swipeRefresh.setColorSchemeColors(
            Color.parseColor("#3b82f6"),
            Color.parseColor("#22c55e"),
            Color.parseColor("#eab308")
        );
        swipeRefresh.setProgressBackgroundColorSchemeColor(Color.parseColor("#1a1f2e"));
        swipeRefresh.setOnRefreshListener(() -> {
            if (isNetworkAvailable()) {
                webView.reload();
            } else {
                swipeRefresh.setRefreshing(false);
                showError("Sem conexão com a internet.");
            }
        });
    }

    private void loadApp() {
        errorLayout.setVisibility(View.GONE);
        loadingLayout.setVisibility(View.VISIBLE);
        webView.loadUrl(APP_URL);
    }

    private void showError(String message) {
        loadingLayout.setVisibility(View.GONE);
        webView.setVisibility(View.GONE);
        errorLayout.setVisibility(View.VISIBLE);
        TextView errorText = findViewById(R.id.errorText);
        if (errorText != null) errorText.setText(message);
    }

    private boolean isNetworkAvailable() {
        ConnectivityManager cm = (ConnectivityManager) getSystemService(CONNECTIVITY_SERVICE);
        NetworkInfo info = cm.getActiveNetworkInfo();
        return info != null && info.isConnected();
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        webView.onResume();
        // Auto-refresh when coming back
        if (isLoaded) {
            webView.evaluateJavascript(
                "if(typeof forcarAtualizacao==='function') forcarAtualizacao();", null
            );
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        webView.onPause();
    }

    @Override
    protected void onDestroy() {
        webView.destroy();
        super.onDestroy();
    }
}
