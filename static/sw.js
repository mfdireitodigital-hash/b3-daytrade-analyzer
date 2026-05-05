// B3 Day Trade Analyzer - Service Worker v1.0
const CACHE_NAME = 'b3-trade-v14';
const STATIC_ASSETS = [
  '/',
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

// Install - cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// Activate - clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => 
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch - Network first for API, Cache first for static
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  
  // API calls: network only (real-time data)
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(event.request).catch(() => 
        new Response(JSON.stringify({erro: 'Sem conexão'}), {
          headers: {'Content-Type': 'application/json'}
        })
      )
    );
    return;
  }
  
  // Static assets: cache first, then network
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(event.request).then(cached => cached || fetch(event.request))
    );
    return;
  }
  
  // HTML pages: network first, cache fallback
  event.respondWith(
    fetch(event.request).then(response => {
      const clone = response.clone();
      caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
      return response;
    }).catch(() => caches.match(event.request))
  );
});
