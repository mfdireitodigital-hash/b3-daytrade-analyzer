// B3 Day Trade Analyzer - Service Worker v21
const CACHE_NAME = 'b3-trade-v21';
const STATIC_ASSETS = [
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

// Install - cache only icons/manifest (NOT HTML)
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// Activate - clean ALL old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => 
      Promise.all(keys.map(k => caches.delete(k)))
    ).then(() => caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS)))
  );
  self.clients.claim();
});

// Fetch
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  
  // API calls: network only
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
  
  // Static assets (icons, manifest): cache first
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(event.request).then(cached => cached || fetch(event.request))
    );
    return;
  }
  
  // HTML pages: ALWAYS network, NEVER cache
  event.respondWith(fetch(event.request));
});
