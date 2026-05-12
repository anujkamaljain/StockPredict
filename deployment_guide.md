# Deployment Guide — Vercel (Frontend) + GCP Cloud Run (Backend)

> **Read this whole guide once before starting.** It's written so a newbie can
> follow every step. After deployment your app will be live on the public
> internet with HTTPS, and the trained ML models will drive the signals.
>
> **Prerequisite:** You have already followed `cloud_training_guide.md` and
> your `backend/data/models/` folder contains 11 trained model files with
> `training_report.json` showing `acceptance_gates.passed: true`.

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
| gcloud CLI | Deploy to Cloud Run | <https://cloud.google.com/sdk/docs/install> |
| Docker Desktop *(optional)* | Build images locally; you can skip and use Cloud Build instead | <https://www.docker.com/products/docker-desktop> |
| curl or PowerShell *(already on Windows)* | Test endpoints | Built-in |

### 3. Verify trained models are ready

```powershell
Get-ChildItem F:\Coding\StockMarketPredectior\backend\data\models\ |
  Where-Object Name -ne ".gitkeep" |
  Measure-Object | Select-Object -ExpandProperty Count
```

Output should be **11**. If less, go back to `cloud_training_guide.md`.

Also verify gates passed:

```powershell
Get-Content F:\Coding\StockMarketPredectior\backend\data\models\training_report.json |
  ConvertFrom-Json |
  Select-Object -ExpandProperty acceptance_gates |
  Select-Object passed
```

Output must be `passed : True`. **Do not deploy a failed model.**

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
loads them at startup and uses them on every request — no external storage,
no extra latency.

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

> The trained model files in `backend/data/models/` are **not** ignored by
> `.gitignore` if you intentionally want them in git. **Better practice:**
> the included `.gitignore` excludes them — copy them out of band (via the
> Docker build context, see below) and keep the repo lean. The repo's
> `.gitignore` already does this for you.

### 1.2 — Create a GitHub repo

1. Go to <https://github.com/new>
2. Repository name: `StockMarketPredectior` (or your choice)
3. Choose **Private** (recommended) or Public
4. Click **Create repository** — do NOT add README/license (you have one)

GitHub shows commands to push existing repo. Copy them or use this template:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/StockMarketPredectior.git
git branch -M main
git push -u origin main
```

You will be prompted for GitHub username + a personal access token (not your password). Create one at <https://github.com/settings/tokens> if you don't have it.

---

# Part 2 — Deploy the frontend on Vercel

### 2.1 — Import the project

1. Go to <https://vercel.com/new>
2. Click **Import** next to your GitHub repo (you may need to grant Vercel access to your repos first)
3. Configure:

| Setting | Value |
|---------|-------|
| **Framework Preset** | Next.js |
| **Root Directory** | `frontend` |
| **Build Command** | *leave default* (`npm run build`) |
| **Output Directory** | *leave default* (`.next`) |
| **Install Command** | *leave default* (`npm install`) |

### 2.2 — Add environment variable (placeholder for now)

In the **Environment Variables** section:

| Key | Value | Notes |
|-----|-------|-------|
| `NEXT_PUBLIC_API_URL` | `https://placeholder.run.app` | We'll update this in Part 3.7 after Cloud Run is live |

### 2.3 — Deploy

Click **Deploy**. Vercel builds and deploys automatically. After ~1 minute, you'll get a URL like `https://stockmarketpredectior.vercel.app`.

**Save this URL** — you'll need it in Part 3.6.

The site will be live but will throw network errors (it can't reach the placeholder backend). That's expected — we fix it next.

---

# Part 3 — Deploy the backend on GCP Cloud Run

### 3.1 — Authenticate gcloud

After installing the gcloud CLI:

```powershell
gcloud auth login
gcloud auth configure-docker     # only needed if you build images locally
```

A browser will open — sign in with your Google account.

### 3.2 — Create a GCP project

```powershell
$PROJECT_ID = "stockml-backend"     # change if taken
gcloud projects create $PROJECT_ID --name="StockML Backend"
gcloud config set project $PROJECT_ID
```

If you get "billing required", link a billing account at <https://console.cloud.google.com/billing> (the $300 free credit means you won't actually be charged for personal use).

### 3.3 — Enable required APIs

```powershell
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

This takes ~30 seconds.

### 3.4 — Create the Dockerfile and .dockerignore

> If these files already exist in your `backend/` folder from earlier
> versions of this guide, **replace them** with the contents below — they
> have been updated for the trained-model workflow.

Create `backend/Dockerfile`:

```dockerfile
# ============================================================
# Stock Market ML Backend — Production Dockerfile
# Trained models are baked into the image so the API runs offline.
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (production uses backend/requirements.txt)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY app/ ./app/

# Bake trained models into the image so Cloud Run runs offline.
# (These must already exist locally — see cloud_training_guide.md.)
COPY data/models/ ./data/models/

# Ensure runtime data subdirs exist (caches, logs)
RUN mkdir -p data/cache data/raw data/processed data/logs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fs http://localhost:8080/health || exit 1

# 2 workers fits comfortably in 2 GiB; AMP is irrelevant on CPU.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 2 --timeout-keep-alive 30"]
```

Create `backend/.dockerignore`:

```
venv/
.venv/
__pycache__/
*.pyc
*.pyo
.env
.env.*
.gitignore
data/cache/**
data/raw/**
data/processed/**
data/logs/**
data/*.db
.git
*.parquet
trained_models/
trained_models_*/
catboost_info/
requirements_dgx.txt
train_cloud.py
```

> The `.dockerignore` deliberately excludes `train_cloud.py` and
> `requirements_dgx.txt` — they're training-only artifacts not needed at
> serving time. `data/models/` is **not** excluded — those files must be in
> the image.

### 3.5 — Build and push the image

You have two options.

**Option A — Cloud Build (no local Docker needed, recommended):**

```powershell
cd F:\Coding\StockMarketPredectior\backend
gcloud builds submit --tag gcr.io/$PROJECT_ID/api:v1 .
```

This uploads your `backend/` folder to GCP, builds the image in the cloud, and pushes it to GCR. Takes 3–5 minutes.

**Option B — Local Docker:**

```powershell
cd F:\Coding\StockMarketPredectior\backend
docker build -t gcr.io/$PROJECT_ID/api:v1 .
docker push gcr.io/$PROJECT_ID/api:v1
```

### 3.6 — Deploy to Cloud Run

> Replace `YOUR_ALPHA_VANTAGE_KEY` and `YOUR_FRED_KEY` with your actual keys.
> Replace `https://your-app.vercel.app` with the Vercel URL from Part 2.3.

```powershell
$AV_KEY = "YOUR_ALPHA_VANTAGE_KEY"
$FRED_KEY = "YOUR_FRED_KEY"
$FRONTEND_URL = "https://stockmarketpredectior.vercel.app"

gcloud run deploy stockml-api `
    --image gcr.io/$PROJECT_ID/api:v1 `
    --region us-central1 `
    --platform managed `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --min-instances 0 `
    --max-instances 3 `
    --timeout 300 `
    --concurrency 80 `
    --set-env-vars "ALPHA_VANTAGE_API_KEY=$AV_KEY,FRED_API_KEY=$FRED_KEY,FRONTEND_URL=$FRONTEND_URL,DEVICE=cpu,LOG_LEVEL=INFO"
```

After ~30 seconds, gcloud prints a URL like:

```
Service URL: https://stockml-api-xxxxx-uc.a.run.app
```

**Save this URL.** This is your production backend.

### 3.7 — Verify the backend is live

```powershell
$BACKEND_URL = "https://stockml-api-xxxxx-uc.a.run.app"

# Health check — should show models_loaded: true and gates_passed: true
curl "$BACKEND_URL/health"

# Sanity check — should return a real signal with signal_source: "ml_ensemble"
curl "$BACKEND_URL/api/signals/generate/AAPL"

# Model info — confirms which trained models loaded
curl "$BACKEND_URL/api/signals/model-info"
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

> If `models_loaded: false`, the trained files weren't included in the
> image. Re-check Part 3.5 (the Cloud Build output should mention
> `COPY data/models/` — look for files like `lstm_best.pt` in the build log).

### 3.8 — Connect Vercel to the new backend

Back in Vercel:
1. Open your project → **Settings** → **Environment Variables**
2. Find `NEXT_PUBLIC_API_URL`
3. Click the three-dot menu → **Edit**
4. Replace the placeholder with `https://stockml-api-xxxxx-uc.a.run.app` (your real Cloud Run URL)
5. Save
6. Go to **Deployments** tab → click the three-dot menu on the latest deployment → **Redeploy**

After redeploy (~1 minute), open your Vercel site. The frontend now talks to the live backend. Search for AAPL → you should see a real signal with the **"ML Ensemble"** badge.

---

# Part 4 — Use Secret Manager for API keys (production best practice)

Setting API keys as env vars in `gcloud run deploy` is fine for prototypes,
but for anything public-facing, use Secret Manager.

### 4.1 — Create secrets

```powershell
echo "YOUR_ALPHA_VANTAGE_KEY" | gcloud secrets create av-api-key --data-file=-
echo "YOUR_FRED_KEY" | gcloud secrets create fred-api-key --data-file=-
```

### 4.2 — Grant the Cloud Run service account access

```powershell
$PROJECT_NUMBER = gcloud projects describe $PROJECT_ID --format="value(projectNumber)"
$SA = "$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding av-api-key `
    --member="serviceAccount:$SA" `
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding fred-api-key `
    --member="serviceAccount:$SA" `
    --role="roles/secretmanager.secretAccessor"
```

### 4.3 — Redeploy referencing secrets

```powershell
gcloud run deploy stockml-api `
    --image gcr.io/$PROJECT_ID/api:v1 `
    --region us-central1 `
    --set-secrets "ALPHA_VANTAGE_API_KEY=av-api-key:latest,FRED_API_KEY=fred-api-key:latest" `
    --set-env-vars "FRONTEND_URL=$FRONTEND_URL,DEVICE=cpu,LOG_LEVEL=INFO"
```

This removes the plain-text keys from the service definition.

---

# Part 5 — Updating deployments

### 5.1 — Update backend code (no new training)

```powershell
cd F:\Coding\StockMarketPredectior\backend
gcloud builds submit --tag gcr.io/$PROJECT_ID/api:v2 .
gcloud run deploy stockml-api `
    --image gcr.io/$PROJECT_ID/api:v2 `
    --region us-central1
```

### 5.2 — Update trained models (most common refresh)

This is the cycle you'll repeat every month or after major market events:

1. Open `cloud_training_guide.md`, follow it end-to-end to produce a new `trained_models/`
2. Verify `acceptance_gates.passed: true` in `training_report.json`
3. Replace `backend/data/models/` contents with the new files
4. Build and deploy a fresh image:
   ```powershell
   cd F:\Coding\StockMarketPredectior\backend
   gcloud builds submit --tag gcr.io/$PROJECT_ID/api:v3 .
   gcloud run deploy stockml-api `
       --image gcr.io/$PROJECT_ID/api:v3 `
       --region us-central1
   ```
5. Verify with `curl $BACKEND_URL/health` — `gates_passed` should reflect the new training.

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
https://stockml-api-xxxxx-uc.a.run.app/health
```
Expect HTTP 200 with `models_loaded: true`.

### 6.3 — Live signal source check

Visit your Vercel site, search a ticker, and look at the badge above the
signal:
- **"ML Ensemble"** (blue, brain icon) = trained models active
- **"Statistical Fallback"** (yellow, sigma icon) = models missing or broken; check `/health`

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
when idle (`--min-instances 0`), so you pay nothing when no one is using
the app.

### External APIs

| API | Free tier | Cost if exceeded |
|-----|-----------|-----------------|
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

Optional hardening (rate limiting via `slowapi`):

```python
# In backend/app/main.py, after creating `app`:
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Then on hot endpoints:
@router.get("/generate/{ticker}")
@limiter.limit("30/minute")
async def generate_signal(...): ...
```

---

# Part 9 — Troubleshooting

| Symptom | Fix |
|---------|-----|
| `curl /health` returns `models_loaded: false` | Trained files not in image. Re-run Part 3.5; check Cloud Build log shows `data/models/` files |
| Frontend shows "Statistical Fallback" everywhere | Same as above — backend has no models |
| `gcloud run deploy` fails with permission errors | Run `gcloud auth login` again; verify project is selected with `gcloud config get-value project` |
| Build fails: `No such file or directory: data/models/` | `data/models/` is empty. Train first (see `cloud_training_guide.md`) |
| CORS error in browser console | `FRONTEND_URL` env var on Cloud Run doesn't match your actual Vercel domain. Redeploy with the right URL |
| Cloud Run cold start ~10s | Normal. Use `--min-instances 1` to keep warm (costs $5–10/mo) |
| `502 Bad Gateway` | Container crashed on startup. Check Logs tab in Cloud Run console |
| Alpha Vantage rate limit (25/day) | Expected. Yahoo Finance fallback handles it transparently |
| Vercel build fails: `NEXT_PUBLIC_API_URL is undefined` | You forgot Part 2.2. Add the env var and redeploy |
| Latency > 2 seconds | First request after cold start is slow. Increase `--min-instances` or accept the cold start |

---

# Final checklist before announcing your app

```
☐ Trained models in production (curl /health → models_loaded: true)
☐ Acceptance gates passed (curl /health → gates_passed: true)
☐ Live signal returns "ml_ensemble" source (curl /api/signals/generate/AAPL)
☐ Frontend loads without console errors
☐ Search bar works (try "apple")
☐ Signal panel shows ML Ensemble badge
☐ Backtest tab runs and shows powered-by-ML badge
☐ Equity curve tooltip shows the correct date and value at hover position
☐ Per-model probabilities visible in signal panel
☐ HTTPS green padlock on both Vercel and Cloud Run URLs
☐ Keys are in Secret Manager (Part 4 done)
☐ You have monitoring set up (Part 6)
☐ You know how to retrain monthly (Part 5.2)
```

---

# What's next

- **Retraining cadence**: every 4 weeks (see `cloud_training_guide.md` retraining schedule)
- **Custom domain**: Vercel → Settings → Domains → Add → follow DNS instructions
- **Cost alerts**: GCP Console → Billing → Budgets & alerts → set $10/mo alert
- **Backups**: keep the previous `trained_models/` folder until the new one is verified live for 1 week
