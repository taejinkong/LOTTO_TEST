// 오프라인 캐시 (localhost/https 보안 컨텍스트에서만 등록됨)
// 전략: stale-while-revalidate — 캐시를 즉시 응답하고 백그라운드에서 최신본으로 갱신.
//        다음 방문 때 새 데이터가 반영된다(주간 회차 갱신에 적합).
const CACHE = "lotto-app-v20";
const ASSETS = [
  "./", "./index.html", "./styles.css", "./app.js", "./data.js",
  "./manifest.webmanifest", "./icon-192.png", "./icon-512.png",
];

self.addEventListener("install", (e) => {
  /* addAll 은 브라우저 HTTP 캐시를 거쳐서 가져온다. 그래서 CACHE 버전을 올려도
     새 캐시에 '옛 파일'이 그대로 심기는 일이 실제로 있었다(v18 배포 때 app.js 가
     구버전으로 캐시돼 대시보드 기본값 변경이 반영되지 않았다).
     cache: "reload" 로 HTTP 캐시를 건너뛰고 항상 서버에서 새로 받는다. */
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => Promise.all(ASSETS.map((url) =>
        fetch(new Request(url, { cache: "reload" }))
          .then((res) => (res.ok ? c.put(url, res) : null))
          .catch(() => null)
      )))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith(
    caches.open(CACHE).then((cache) =>
      cache.match(req).then((cached) => {
        const network = fetch(req)
          .then((res) => {
            if (res && res.ok) cache.put(req, res.clone());
            return res;
          })
          .catch(() => cached);
        return cached || network;
      })
    )
  );
});
