// Cloudflare Worker that relays the app's requests to the Opta feed with the Scoresway
// headers the feed requires, so they come from Cloudflare instead of the app's host.
//
// App setup: set OPTA_BASE_URL to https://<worker>.<account>.workers.dev/soccerdata
// Optional: add a RELAY_KEY secret to this Worker and set the same value as
// OPTA_RELAY_KEY in the app, so only your app can use the relay.

const UPSTREAM = "https://api.performfeeds.com";

const FEED_HEADERS = {
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0",
  "Accept": "*/*",
  "Accept-Language": "en-US,en;q=0.5",
  "Referer": "https://www.scoresway.com/",
  "Origin": "https://www.scoresway.com",
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/") {
      return new Response("Opta relay is running");
    }
    if (request.method !== "GET") {
      return new Response("Method not allowed", { status: 405 });
    }
    if (env.RELAY_KEY && request.headers.get("X-Relay-Key") !== env.RELAY_KEY) {
      return new Response("Forbidden: missing or wrong X-Relay-Key", { status: 403 });
    }
    // Only relay the soccer data feed, not arbitrary URLs
    if (!url.pathname.startsWith("/soccerdata/")) {
      return new Response("Not found", { status: 404 });
    }

    const upstream = await fetch(UPSTREAM + url.pathname + url.search, { headers: FEED_HEADERS });

    const headers = new Headers({
      "Content-Type": upstream.headers.get("Content-Type") || "application/javascript",
    });
    const retryAfter = upstream.headers.get("Retry-After");
    if (retryAfter) {
      headers.set("Retry-After", retryAfter);
    }
    return new Response(upstream.body, { status: upstream.status, headers });
  },
};
