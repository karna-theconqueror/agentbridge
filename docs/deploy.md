# Deploy AgentBridge to a Public URL

Get the dashboard on a public link you can paste into a resume or send to a
recruiter. Two easy options — both free-tier friendly.

The service is a single `uvicorn` process serving the API + dashboard, so any
Python host works.

---

## Option A — Render (recommended, simplest)

1. Push the repo to GitHub (done: `karna-theconqueror/agentbridge`).
2. Go to **https://dashboard.render.com** → **New +** → **Web Service**.
3. Connect the `agentbridge` repo.
4. Fill in:
   - **Name:** `agentbridge`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**. Wait ~2 min for the build.
6. Your live URL: `https://agentbridge-xxxx.onrender.com`
   - Dashboard: `https://agentbridge-xxxx.onrender.com/`
   - Health: `https://agentbridge-xxxx.onrender.com/health`

> Render free services **spin down after inactivity** — the first request after
> idle takes ~30s to wake. Fine for a demo; upgrade to a paid instance for
> always-on.

---

## Option B — Railway

1. Go to **https://railway.app** → **New Project** → **Deploy from GitHub repo**.
2. Pick `agentbridge`. Railway auto-detects Python.
3. If it doesn't set the start command, add a **Root Directory** of `.` and set:
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. **Settings → Networking → Generate Domain** → get your public URL.

Railway's free trial gives $5 credit; a small uvicorn service stays well within it.

---

## Required environment variable

The live agent needs the model key **as a platform secret** — never in the repo:

| Name | Required | Notes |
|---|---|---|
| `GEMINI_API_KEY` | yes (for real answers) | Platform → *Environment/Secrets*. Without it the app still runs on the offline MockLLM. |
| `GEMINI_MODEL` | no | Pins one model. **Leave unset** to get automatic model failover when one model's quota runs out. |
| `RATE_LIMIT_PER_MIN` | no (default `10`) | Per-client cap on `/invoke`. Set `0` to disable. |

> **Set the rate limit.** A public URL gets discovered by scanners within hours,
> and every un-throttled `/invoke` spends your free-tier quota. `/invoke` answers
> `429` with a `Retry-After` header once a client is over budget; `/health`,
> `/metrics` and the dashboard stay open.

Also add a `PORT` note if your host binds differently — the Dockerfile already
honours `${PORT}` (default `8000`).

---

## One thing to know: metrics are per-instance

The TraceLogger keeps aggregates **in memory** and appends to a local JSONL file.
On a hosted instance this means:

- Metrics reset when the service restarts / redeploys.
- `/metrics` reflects only that instance's traffic.

That's fine for a live demo. For persistent metrics you'd swap the JSONL logger
for a real backend (see `docs/interview-prep.md`, "How would you take this to real
production").

---

## Quick verification after deploy

```bash
curl https://YOUR-URL/health            # → {"status":"ok"}
curl -X POST https://YOUR-URL/invoke \
     -H "Content-Type: application/json" \
     -d '{"prompt":"What is 125 * 8?","hitl":true}'
# → {"answer":"The result is 1000.", ...}
```

Then open `https://YOUR-URL/` in a browser to show the dashboard.
