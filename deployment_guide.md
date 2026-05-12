# Deployment Guide — Vercel (Frontend) + GCP Cloud Run GUI (Backend)

> **Read this whole guide once before starting.** It's written so a newbie can
> follow every step. After deployment your app will be live on the public
> internet with HTTPS, and the trained ML models will drive the signals.
>
> **Prerequisite:** You have already followed `cloud_training_guide.md` and
> your `backend/data/models/` folder contains 11 trained model files with
> `training_report.json`.

---

## What you need before starting

### 1. Accounts (all free to start)

| Account | Purpose | Free Tier |
|---------|---------|-----------|
| [GitHub](https://github.com) | Source code hosting | Free unlimited public repos |
| [Vercel](https://vercel.com/signup) | Frontend hosting | Hobby tier free; sign in with GitHub |
| [Google Cloud](https://cloud.google.com) | Backend hosting | $300 free credit for new accounts; Cloud Run free tier covers personal use |

### 2. Local tools

| Tool | Purpose | Install |
|------|---------|---------|
| Git | Push code to GitHub | <https://git-scm.com/download/win> |
| Docker Desktop | Build images locally | <https://www.docker.com/products/docker-desktop> |
| curl or PowerShell *(already on Windows)* | Test endpoints | Built-in |

> **Note:** No `gcloud CLI` needed — we deploy entirely via the GCP Console (browser GUI).

### 3. Verify trained models are ready

```powershell
Get-ChildItem F:\Coding\StockMarketPredectior\backend\data\models\ |
  Where-Object Name -ne ".gitkeep" |
  Measure-Object | Select-Object -ExpandProperty Count
```

Output should be **11**. If less, go back to `cloud_training_guide.md`.

---

## Architecture overview

```
┌──────────────┐      ┌───────────────────────┐      ┌────────────────────┐
│ Users        │─────▶│ Vercel (Frontend)     │─────▶│ GCP Cloud Run      │
│ Browser      │◀─────│ Next.js SSR/SSG (free)│◀─────│ FastAPI + ML       │
│              │      │                       │      │ (~$0-5/mo)         │
└──────────────┘      └───────────────────────┘      └────────────────────┘
                                                              │
                                                       ┌──────┴──────┐
                                                       │ AlphaVantage│
                                                       │ YahooFinance│
                                                       │ FRED        │
                                                       └─────────────┘
```

| Component | Platform | Tier | Cost |
|-----------|----------|------|------|
| Frontend | Vercel | Free (Hobby) | $0/mo |
| Backend | GCP Cloud Run | Free tier | $0–5/mo |
| Domain *(optional)* | Any registrar | — | ~$12/yr |

The trained model files are **baked into the Docker image** so the backend
loads them at startup — no external storage, no extra latency.

---

# Part 1 — Push your code to GitHub

Skip this if your code is already on GitHub.

### 1.1 — Initialize a repo

```powershell
cd F:\Coding\StockMarketPredectior
git init
git add .
git commit -m "Initial commit with trained models"
```

### 1.2 — Create a GitHub repo

1. Go to <https://github.com/new>
2. Repository name: `StockMarketPredectior` (or your choice)
3. Choose **Private** (recommended) or Public
4. Click **Create repository** — do NOT add README/license (you have one)

Push:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/StockMarketPredectior.git
git branch -M main
git push -u origin main
```

---

# Part 2 — Deploy the frontend on Vercel

### 2.1 — Import the project

1. Go to <https://vercel.com/new>
2. Click **Import** next to your GitHub repo
3. Configure:

| Setting | Value |
|---------|-------|
| **Framework Preset** | Next.js |
| **Root Directory** | `frontend` |
| **Build Command** | *leave default* (`npm run build`) |
| **Output Directory** | *leave default* (`.next`) |

### 2.2 — Add environment variable (placeholder for now)

In the **Environment Variables** section:

| Key | Value | Notes |
|-----|-------|-------|
| `NEXT_PUBLIC_API_URL` | `https://placeholder.run.app` | We'll update this after Cloud Run is live |

### 2.3 — Deploy

Click **Deploy**. After ~1 minute, you'll get a URL like `https://stockmarketpredectior.vercel.app`.

**Save this URL** — you'll need it when configuring Cloud Run CORS.

---

# Part 3 — Deploy the backend on GCP Cloud Run (GUI)

This entire section uses the **Google Cloud Console** (browser GUI) — no CLI needed.

### 3.1 — Create a GCP project

1. Go to <https://console.cloud.google.com>
2. Click the project dropdown (top bar, left of the search box)
3. Click **NEW PROJECT**
4. Project name: `StockML Backend`
5. Click **Create**
6. After creation, select this project from the dropdown

> If prompted, link a billing account at Billing → My projects. The $300
> free credit means you won't actually be charged.

### 3.2 — Enable required APIs

1. Go to **APIs & Services → Library** (<https://console.cloud.google.com/apis/library>)
2. Search and enable these (click each → click **ENABLE**):
   - **Cloud Run Admin API**
   - **Cloud Build API**
   - **Artifact Registry API**

### 3.3 — Build the Docker image locally

Open **Docker Desktop** first (make sure it's running), then in PowerShell:

```powershell
cd F:\Coding\StockMarketPredectior\backend

# Build the image (takes 3-5 minutes the first time)
docker build -t stockml-backend .
```

Verify it built:

```powershell
docker images | Select-String stockml
```

You should see `stockml-backend` with a size around 3-4 GB.

### 3.4 — Push the image to Artifact Registry

#### A. Create a repository in Artifact Registry (one-time setup)

1. Go to <https://console.cloud.google.com/artifacts>
2. Click **+ CREATE REPOSITORY**
3. Fill in:

| Field | Value |
|-------|-------|
| **Name** | `stockml-repo` |
| **Format** | Docker |
| **Mode** | Standard |
| **Location type** | Region |
| **Region** | `asia-south1` (Mumbai) or `us-central1` |

4. Click **CREATE**

#### B. Tag and push your image

```powershell
# Replace REGION and PROJECT_ID with your values.
# Example: asia-south1-docker.pkg.dev/stockml-backend-12345/stockml-repo/api:v1

$REGION = "asia-south1"                    # or us-central1
$PROJECT_ID = "your-project-id"            # find at: console.cloud.google.com → project dropdown
$REPO = "$REGION-docker.pkg.dev/$PROJECT_ID/stockml-repo"

# Authenticate Docker with GCP (one-time)
# Go to https://console.cloud.google.com/artifacts → click SET UP INSTRUCTIONS
# Or run: (you DO need gcloud CLI for this one step, or use a JSON key)
gcloud auth configure-docker $REGION-docker.pkg.dev

# Tag the image
docker tag stockml-backend "$REPO/api:v1"

# Push (takes 2-5 min depending on upload speed)
docker push "$REPO/api:v1"
```

> **Alternative — Cloud Build (no local Docker needed):**
>
> If you don't have Docker Desktop, you can use Cloud Build.
> Go to <https://console.cloud.google.com/cloud-build/builds> → click
> **CREATE TRIGGER** or use this one CLI command:
>
> ```powershell
> cd F:\Coding\StockMarketPredectior\backend
> gcloud builds submit --tag $REPO/api:v1 .
> ```
>
> This uploads your code and builds the image entirely in the cloud.

### 3.5 — Deploy to Cloud Run (GUI — the main event!)

1. Go to <https://console.cloud.google.com/run>
2. Click **CREATE SERVICE**
3. Fill in the form:

#### Container image

- Click **SELECT** → browse to your Artifact Registry image
- Navigate: `stockml-repo` → `api` → select `v1`

#### Service name & region

| Field | Value |
|-------|-------|
| **Service name** | `stockml-api` |
| **Region** | Same region as your Artifact Registry (e.g. `asia-south1`) |

#### Authentication

- Select: **Allow unauthenticated invocations** ✅
  *(so your frontend can reach it without auth)*

#### CPU allocation and pricing

- Select: **CPU is only allocated during request processing**
  *(cheapest — scales to zero)*

#### Expand "Container, Networking, Security" section

Click the accordion to expand it.

##### Container tab:

| Setting | Value |
|---------|-------|
| **Container port** | `8080` |
| **Memory** | `2 GiB` |
| **CPU** | `2` |
| **Request timeout** | `300` seconds |
| **Maximum concurrent requests** | `80` |

##### Environment Variables — add these:

| Key | Value |
|-----|-------|
| `ALPHA_VANTAGE_API_KEY` | `XH582O310WG11HT2` (your key) |
| `FRED_API_KEY` | (your FRED key, or leave empty) |
| `FRONTEND_URL` | `https://stockmarketpredectior.vercel.app` (your Vercel URL) |
| `DEVICE` | `cpu` |
| `LOG_LEVEL` | `INFO` |

##### Autoscaling tab:

| Setting | Value |
|---------|-------|
| **Minimum instances** | `0` (free when idle) |
| **Maximum instances** | `3` |

4. Click **CREATE** 🚀

Wait 1-2 minutes. Cloud Run will show a green checkmark and give you a URL like:

```
https://stockml-api-xxxxx-el.a.run.app
```

**Copy this URL!**

### 3.6 — Verify the backend is live

Open these URLs in your browser (or use PowerShell):

```powershell
$BACKEND_URL = "https://stockml-api-xxxxx-el.a.run.app"

# Health check
Invoke-RestMethod "$BACKEND_URL/health"

# Test a signal
Invoke-RestMethod "$BACKEND_URL/api/signals/generate/AAPL"
```

Expected `/health` response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "loaded_models": ["lstm","transformer","cnn","xgboost","lightgbm","catboost"],
  "gates_passed": true
}
```

> If `models_loaded: false`, the trained files weren't in the Docker image.
> Re-check that `backend/data/models/` has 11 files and rebuild.

### 3.7 — Connect Vercel frontend to Cloud Run backend

1. Go to your Vercel project → **Settings** → **Environment Variables**
2. Find `NEXT_PUBLIC_API_URL`
3. Click Edit → Replace with your real Cloud Run URL:
   `https://stockml-api-xxxxx-el.a.run.app`
4. Save
5. Go to **Deployments** → three-dot menu on latest → **Redeploy**

After ~1 minute, your site is live with ML-powered signals! 🎉

---

# Part 4 — Use Secret Manager for API keys (production best practice)

Instead of plain-text env vars, use Secret Manager:

### 4.1 — Create secrets via GUI

1. Go to <https://console.cloud.google.com/security/secret-manager>
2. Click **CREATE SECRET**
3. Name: `av-api-key`, paste your Alpha Vantage key → **CREATE**
4. Repeat for `fred-api-key`

### 4.2 — Link secrets to Cloud Run

1. Go to <https://console.cloud.google.com/run>
2. Click your `stockml-api` service
3. Click **EDIT & DEPLOY NEW REVISION**
4. Scroll to **Container** → **Variables & Secrets** tab
5. For `ALPHA_VANTAGE_API_KEY`:
   - Change from "Value" to "Reference a Secret"
   - Select `av-api-key` → version `latest`
6. Repeat for `FRED_API_KEY`
7. Click **DEPLOY**

> If you get a permissions error, go to IAM → grant the Cloud Run service
> account the `Secret Manager Secret Accessor` role.

---

# Part 5 — Updating deployments

### 5.1 — Update backend code (no new training)

1. Rebuild locally: `docker build -t stockml-backend .`
2. Tag: `docker tag stockml-backend "$REPO/api:v2"`
3. Push: `docker push "$REPO/api:v2"`
4. Go to Cloud Run console → your service → **EDIT & DEPLOY NEW REVISION**
5. Change the image tag to `v2` → **DEPLOY**

### 5.2 — Update trained models (most common refresh)

1. Train new models via `cloud_training_guide.md`
2. Replace files in `backend/data/models/`
3. Rebuild & push image (v3, v4, etc.)
4. Deploy new revision via Cloud Run GUI

### 5.3 — Update frontend code

Just push to GitHub. Vercel auto-deploys:

```powershell
git add frontend/
git commit -m "Update frontend"
git push origin main
```

---

# Part 6 — Monitoring

### 6.1 — Cloud Run console

<https://console.cloud.google.com/run> → click your service → tabs:
- **Metrics**: request count, latency, error rate, CPU/memory
- **Logs**: application logs (loguru output)
- **Revisions**: rollback to a previous deployment if needed

### 6.2 — Health endpoint as uptime probe

Any uptime monitor (e.g. UptimeRobot, GCP Uptime Checks) can hit:
```
https://stockml-api-xxxxx-el.a.run.app/health
```
Expect HTTP 200 with `models_loaded: true`.

### 6.3 — Live signal source check

Visit your Vercel site, search a ticker, and look at the badge:
- **"ML Ensemble"** (blue, brain icon) = trained models active ✅
- **"Statistical Fallback"** (yellow, sigma icon) = models missing; check `/health`

---

# Part 7 — Cost estimation

### Vercel (frontend)

| Feature | Hobby (Free) | Pro ($20/mo) |
|---------|-------------|--------------|
| Deployments | Unlimited | Unlimited |
| Bandwidth | 100 GB/mo | 1 TB/mo |
| Custom domains | yes | yes |

For personal use: **free tier is fine.**

### GCP Cloud Run (backend)

| Resource | Free tier (per month) | Your expected usage |
|----------|----------------------|---------------------|
| CPU | 180,000 vCPU-seconds | ~5,000 |
| Memory | 360,000 GiB-seconds | ~10,000 |
| Requests | 2 million | ~1,000 |
| Networking | 1 GB egress | ~500 MB |

**Estimated cost: $0–5/month** for personal use. Cloud Run scales to zero
when idle (`min instances = 0`), so you pay nothing when no one is using
the app.

### External APIs

| API | Free tier | Cost if exceeded |
|-----|-----------|-----------------:|
| Alpha Vantage | 25 req/day | $49.99/mo for 120/min |
| FRED | 120 req/min | Free (government service) |
| Yahoo Finance | Unlimited (scraped) | Free |

The app uses Yahoo Finance as automatic fallback, so Alpha Vantage rate
limits never break the app.

---

# Part 8 — Security checklist

```
☐ API keys live in GCP Secret Manager, not in env vars
☐ CORS whitelist (FRONTEND_URL) only includes your Vercel domain
☐ /docs and /redoc are accessible (or disabled — comment out in main.py)
☐ Vercel custom domain has HTTPS (automatic)
☐ Cloud Run service has HTTPS (automatic)
☐ Source repo is private OR contains no secrets
☐ training_report.json is fine to ship (no secrets)
☐ .env is in .gitignore (already done)
```

---

# Part 9 — Troubleshooting

| Symptom | Fix |
|---------|-----|
| `curl /health` returns `models_loaded: false` | Trained files not in image. Re-build Docker; check `backend/data/models/` has 11 files |
| Frontend shows "Statistical Fallback" everywhere | Same as above — backend has no models |
| CORS error in browser console | `FRONTEND_URL` env var on Cloud Run doesn't match your Vercel domain |
| Cloud Run cold start ~10s | Normal. Set min instances = 1 to keep warm (costs $5–10/mo) |
| `502 Bad Gateway` | Container crashed. Check Logs tab in Cloud Run console |
| Alpha Vantage rate limit (25/day) | Expected. Yahoo Finance fallback handles it |
| Vercel build fails | Check `NEXT_PUBLIC_API_URL` is set in Vercel env vars |
| Docker build fails locally | Make sure Docker Desktop is running |
| Image push fails with auth error | Run `gcloud auth configure-docker REGION-docker.pkg.dev` |
| Latency > 2 seconds | First request after cold start. Increase min instances or accept it |

---

# Final checklist before announcing your app

```
☐ Trained models in production (curl /health → models_loaded: true)
☐ Live signal returns "ml_ensemble" source (curl /api/signals/generate/AAPL)
☐ Frontend loads without console errors
☐ Search bar works (try "RELIANCE.NS")
☐ Signal panel shows ML Ensemble badge
☐ Backtest tab runs and shows equity curve
☐ Per-model probabilities visible in signal panel
☐ HTTPS green padlock on both URLs
☐ Keys are in Secret Manager (Part 4 done)
☐ You know how to retrain monthly (Part 5.2)
```

---

# What's next

- **Retraining cadence**: every 4 weeks (see `cloud_training_guide.md`)
- **Custom domain**: Vercel → Settings → Domains → Add → follow DNS instructions
- **Cost alerts**: GCP Console → Billing → Budgets & alerts → set $10/mo alert
- **Alert system**: see `alert_system_plan.md` for automated Telegram notifications
