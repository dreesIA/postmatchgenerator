# postmatchgenerator

https://postmatchgenerator.streamlit.app 

Streamlit app for post-match and player analysis from Opta event data (via Scoresway).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Choose a season, team, and match in the sidebar.

## Bundled data

The Opta feed rate limits some hosting servers (error `HTTP 429`), so the app ships with
feed responses in [snapshots/](snapshots/). The app serves those before calling the feed, so
bundled matches, teams and players load on any host, with no network access needed.

Bundled now (these are the seasons in the app's match picker):

- USL Championship 2026: every match played through October 14, 2026, plus all squads
- USL Championship 2025: the full season, plus all squads

Anything not bundled is fetched live from the feed.

### Adding or refreshing data

Run on a computer that can reach the feed, then commit `snapshots/`:

```bash
# Every played match in a season (add newly played matches by re-running)
python build_snapshots.py --season cbkjk31sv9wm38ljg3vxhtus4

# A single match, or a team's or player's season, from Scoresway URLs
python build_snapshots.py "<Scoresway URL>" ["<another URL>" ...]
```

Matches already bundled are skipped. Match lists and squads are always refreshed.
The season ID is the part of a Scoresway URL after the league name, e.g.
`/soccer/usl-championship-2026/cbkjk31sv9wm38ljg3vxhtus4/`.

## Live data on a rate-limited host: Cloudflare Worker relay

The app's live requests can go through a free Cloudflare Worker
([cloudflare_worker/worker.js](cloudflare_worker/worker.js)), so they come from Cloudflare
instead of the app's host. Cloudflare's servers may be rate limited too.

1. Create a free account at [dash.cloudflare.com](https://dash.cloudflare.com).
2. Go to **Workers & Pages → Create → Create Worker**, name it `opta-relay`, and deploy the starter code.
3. Click **Edit code**, replace everything with the contents of `cloudflare_worker/worker.js`, and deploy.
4. Under the Worker's **Settings → Variables and Secrets**, add a **Secret** named `RELAY_KEY`
   with a random value, e.g. from `python3 -c "import secrets; print(secrets.token_urlsafe(24))"`.
   This stops other people from using your relay.
5. Open `https://opta-relay.<your-subdomain>.workers.dev`. It should say "Opta relay is running".
6. In the hosted app's **Settings → Secrets**, add the following, then reboot the app:

   ```toml
   OPTA_BASE_URL = "https://opta-relay.<your-subdomain>.workers.dev/soccerdata"
   OPTA_RELAY_KEY = "<the same value as RELAY_KEY>"
   ```

To stop using the relay, remove `OPTA_BASE_URL`. With the Wrangler CLI you can instead run
`npx wrangler deploy` and `npx wrangler secret put RELAY_KEY` in `cloudflare_worker/`.
