# postmatchgenerator

Streamlit app for post-match and player analysis from Opta event data (via Scoresway).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Paste a Scoresway match or player URL in the sidebar.

## When the hosted app is rate limited

The Opta feed rate limits some hosting servers (error `HTTP 429`). There are two workarounds,
and you can set up both.

### Option A: Cloudflare Worker relay

The app's requests go through a free Cloudflare Worker ([cloudflare_worker/worker.js](cloudflare_worker/worker.js)),
so they come from Cloudflare instead of the app's host.

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

### Option B: Upload a match file

On a computer that can reach the feed (e.g. your own), download the match:

```bash
python download_match.py "<Scoresway match URL>" ["<another URL>" ...]
```

This saves a `.json` file per match in `match_files/`. In the app's sidebar, use
**Or upload a match file**. The app then doesn't contact the feed for that match.
