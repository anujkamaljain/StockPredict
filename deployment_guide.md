# 🚀 Deployment Guide — Vercel + GCP Cloud Run

> Deploy the frontend on **Vercel** (free) and the backend on **GCP Cloud Run** (serverless, pay-per-use).

---

## 📐 Architecture Overview

```
┌──────────────┐         ┌──────────────────────┐         ┌──────────────┐
│   Users       │────────▶│   Vercel (Frontend)   │────────▶│  GCP Cloud   │
│   Browser     │◀────────│   Next.js SSR/SSG     │◀────────│  Run (API)   │
└──────────────┘         └──────────────────────┘         └──────────────┘
                                                                  │
                                                           ┌──────┴──────┐
                                                           │ Alpha Vantage│
                                                           │ Yahoo Finance│
                                                           │ FRED API     │
                                                           └─────────────┘
```

| Component | Platform | Tier | Cost |
|-----------|----------|------|------|
| Frontend | Vercel | Free (Hobby) | $0/mo |
| Backend | GCP Cloud Run | Free tier | $0-5/mo |
| Domain | Any registrar | Optional | ~$12/yr |

---

## 🎨 Part 1: Deploy Frontend on Vercel

### Prerequisites

- [Vercel account](https://vercel.com/signup) (sign up with GitHub)
- Project pushed to GitHub

### Step 1: Push Frontend to GitHub

If not already pushed:

```powershell
cd F:\Coding\StockMarketPredectior
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Step 2: Import Project on Vercel

1. Go to → [https://vercel.com/new](https://vercel.com/new)
2. Click **"Import Git Repository"**
3. Select your `StockMarketPredectior` repository
4. Configure:

| Setting | Value |
|---------|-------|
| **Framework Preset** | Next.js |
| **Root Directory** | `frontend` |
| **Build Command** | `npm run build` |
| **Output Directory** | `.next` |
| **Install Command** | `npm install` |

### Step 3: Set Environment Variables

In Vercel project settings → **Environment Variables**:

| Key | Value | Notes |
|-----|-------|-------|
| `NEXT_PUBLIC_API_URL` | `https://your-backend-url.run.app` | Set after Cloud Run deploy |

> [!IMPORTANT]
> You'll set the actual backend URL after deploying Cloud Run in Part 2.
> For now, set a placeholder and update it later.

### Step 4: Deploy

Click **"Deploy"** — Vercel will build and deploy automatically.

Your frontend will be live at: `https://your-project.vercel.app`

### Step 5: Custom Domain (Optional)

1. Go to your project **Settings** → **Domains**
2. Add your domain (e.g., `stockml.yourdomain.com`)
3. Update DNS records as instructed by Vercel

---

## ⚙️ Part 2: Deploy Backend on GCP Cloud Run

### Prerequisites

- [Google Cloud account](https://cloud.google.com/) (free $300 credit for new users)
- [Google Cloud CLI (gcloud)](https://cloud.google.com/sdk/docs/install) installed
- Docker installed (for building the container image)

### Step 1: Create a GCP Project

```bash
# Login
gcloud auth login

# Create project (or use existing)
gcloud projects create stockml-backend --name="StockML Backend"
gcloud config set project stockml-backend

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### Step 2: Create Dockerfile

Create `backend/Dockerfile`:

```dockerfile
# ============================================================
# Stock Market ML Backend — Production Dockerfile
# ============================================================
FROM python:3.11-slim

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY data/models/ ./data/models/ 2>/dev/null || true

# Create data directories
RUN mkdir -p data/cache data/raw data/processed data/models data/logs

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start server
CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 2 \
    --timeout-keep-alive 30
```

### Step 3: Create .dockerignore

Create `backend/.dockerignore`:

```
venv/
__pycache__/
*.pyc
.env
data/cache/*
data/raw/*
data/logs/*
.git
*.parquet
```

### Step 4: Update CORS for Production

Update `backend/app/main.py` to accept your Vercel domain:

```python
# CORS for frontend
import os

FRONTEND_URLS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    os.getenv("FRONTEND_URL", ""),          # Production Vercel URL
]
# Filter out empty strings
FRONTEND_URLS = [u for u in FRONTEND_URLS if u]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Step 5: Build & Push Docker Image

```bash
cd F:\Coding\StockMarketPredectior\backend

# Build the image
docker build -t stockml-backend .

# Tag for Google Artifact Registry
docker tag stockml-backend gcr.io/stockml-backend/api:latest

# Push to GCR
docker push gcr.io/stockml-backend/api:latest
```

**Or use Cloud Build (no local Docker needed):**

```bash
gcloud builds submit --tag gcr.io/stockml-backend/api:latest .
```

### Step 6: Deploy to Cloud Run

```bash
gcloud run deploy stockml-api \
    --image gcr.io/stockml-backend/api:latest \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 3 \
    --timeout 300 \
    --concurrency 80 \
    --set-env-vars "ALPHA_VANTAGE_API_KEY=YOUR_KEY_HERE" \
    --set-env-vars "FRED_API_KEY=YOUR_KEY_HERE" \
    --set-env-vars "FRONTEND_URL=https://your-project.vercel.app" \
    --set-env-vars "DEVICE=cpu" \
    --set-env-vars "API_HOST=0.0.0.0" \
    --set-env-vars "API_PORT=8080"
```

After deployment, you'll get a URL like:
```
https://stockml-api-xxxxx-uc.a.run.app
```

> [!TIP]
> **Use GCP Secret Manager for API keys** instead of plain env vars:
> ```bash
> # Create secret
> echo -n "YOUR_ALPHA_VANTAGE_KEY" | gcloud secrets create alpha-vantage-key --data-file=-
>
> # Use in Cloud Run
> gcloud run deploy stockml-api \
>     --set-secrets "ALPHA_VANTAGE_API_KEY=alpha-vantage-key:latest"
> ```

### Step 7: Update Vercel Environment Variable

Now that you have the Cloud Run URL:

1. Go to Vercel → Project Settings → Environment Variables
2. Update `NEXT_PUBLIC_API_URL` to `https://stockml-api-xxxxx-uc.a.run.app`
3. Redeploy the frontend

### Step 8: Verify Deployment

```bash
# Health check
curl https://stockml-api-xxxxx-uc.a.run.app/health

# Test data fetch
curl https://stockml-api-xxxxx-uc.a.run.app/api/data/fetch/AAPL

# Test search
curl "https://stockml-api-xxxxx-uc.a.run.app/api/data/search?q=apple"

# Test signal
curl https://stockml-api-xxxxx-uc.a.run.app/api/signals/generate/AAPL
```

---

## 🔁 Part 3: CI/CD with GitHub Actions

### Auto-deploy Backend on Push

Create `.github/workflows/deploy-backend.yml`:

```yaml
name: Deploy Backend to Cloud Run

on:
  push:
    branches: [main]
    paths:
      - 'backend/**'

env:
  PROJECT_ID: stockml-backend
  SERVICE: stockml-api
  REGION: us-central1

jobs:
  deploy:
    runs-on: ubuntu-latest

    permissions:
      contents: read
      id-token: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Setup Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Build & Push
        run: |
          cd backend
          gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE:${{ github.sha }}

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy $SERVICE \
            --image gcr.io/$PROJECT_ID/$SERVICE:${{ github.sha }} \
            --region $REGION \
            --platform managed \
            --allow-unauthenticated
```

### Auto-deploy Frontend on Push

Vercel handles this automatically when connected to GitHub.
Every push to `main` triggers a new deployment.

### GitHub Secrets to Configure

| Secret | Value | Where to Get |
|--------|-------|-------------|
| `GCP_SA_KEY` | Service account JSON key | GCP Console → IAM → Service Accounts |

---

## 💰 Part 4: Cost Estimation

### Vercel (Frontend)

| Feature | Hobby (Free) | Pro ($20/mo) |
|---------|-------------|--------------|
| Deployments | Unlimited | Unlimited |
| Bandwidth | 100 GB/mo | 1 TB/mo |
| Serverless functions | 100 GB-hrs | 1000 GB-hrs |
| Custom domains | ✅ | ✅ |

**For personal use**: Free tier is more than sufficient.

### GCP Cloud Run (Backend)

| Resource | Free Tier (per month) | Your Expected Usage |
|----------|----------------------|---------------------|
| CPU | 180,000 vCPU-seconds | ~5,000 (well within) |
| Memory | 360,000 GiB-seconds | ~10,000 (well within) |
| Requests | 2 million | ~1,000 (well within) |
| Networking | 1 GB egress | ~500 MB (well within) |

**Estimated cost**: **$0 — $5/month** for personal use.

> [!NOTE]
> Cloud Run scales to zero when idle — you only pay when requests come in.
> With `--min-instances 0`, there's no cost when you're not using it.

### External APIs

| API | Free Tier | Cost if Exceeded |
|-----|-----------|-----------------|
| Alpha Vantage | 25 req/day | $49.99/mo for 120/min |
| FRED | 120 req/min | Free (gov service) |
| Yahoo Finance | Unlimited (scraped) | Free |

---

## 🛡️ Part 5: Security Hardening

### 1. API Key Security

**Never hardcode API keys**. Use environment variables or GCP Secret Manager:

```bash
# Create secrets
echo -n "YOUR_AV_KEY" | gcloud secrets create av-api-key --data-file=-
echo -n "YOUR_FRED_KEY" | gcloud secrets create fred-api-key --data-file=-

# Grant Cloud Run access
gcloud secrets add-iam-policy-binding av-api-key \
    --member="serviceAccount:YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### 2. CORS Configuration

Only allow your Vercel domain:

```python
allow_origins=[
    "https://your-project.vercel.app",
    "https://stockml.yourdomain.com",  # custom domain
]
```

### 3. Rate Limiting (Optional)

Add rate limiting middleware to protect your API:

```python
# pip install slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/signals/generate/{ticker}")
@limiter.limit("30/minute")
async def generate_signal(request: Request, ticker: str):
    ...
```

### 4. HTTPS

- **Vercel**: HTTPS enabled by default ✅
- **Cloud Run**: HTTPS enabled by default ✅

---

## 📊 Part 6: Monitoring & Logging

### GCP Cloud Run Monitoring

1. Go to → [GCP Console](https://console.cloud.google.com/run) → Cloud Run → your service
2. **Metrics tab**: Request count, latency, error rate, CPU/memory
3. **Logs tab**: Application logs (loguru output)

### Set Up Alerts

```bash
# Alert if error rate > 5%
gcloud alpha monitoring policies create \
    --display-name="High Error Rate" \
    --condition-display-name="Error Rate > 5%" \
    --condition-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count" AND metric.labels.response_code_class="5xx"'
```

### Uptime Checks

```bash
# Check if backend is healthy every 5 minutes
gcloud monitoring uptime-check-configs create \
    --display-name="StockML API Health" \
    --resource-type=uptime-url \
    --hostname=stockml-api-xxxxx-uc.a.run.app \
    --path=/health \
    --check-interval=300s
```

---

## 🔄 Part 7: Updating Deployments

### Update Backend

```bash
cd backend

# Build new image
gcloud builds submit --tag gcr.io/stockml-backend/api:v2

# Deploy new version
gcloud run deploy stockml-api \
    --image gcr.io/stockml-backend/api:v2 \
    --region us-central1
```

### Update Frontend

Just push to GitHub — Vercel auto-deploys:

```bash
git add frontend/
git commit -m "Update frontend"
git push origin main
```

### Update Trained Models

1. Retrain on Colab (see Cloud Training Guide)
2. Download new model files
3. Rebuild Docker image with new models:
   ```bash
   # Copy models to backend/data/models/
   # Rebuild & deploy
   gcloud builds submit --tag gcr.io/stockml-backend/api:v3
   gcloud run deploy stockml-api --image gcr.io/stockml-backend/api:v3 --region us-central1
   ```

---

## 🌐 Part 8: Alternative Backend Hosts

If you prefer simpler alternatives to GCP Cloud Run:

### Railway.app (Simplest)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login & deploy
cd backend
railway login
railway init
railway up
```

- **Free tier**: $5/mo credit, 500 hours
- **Pros**: Zero config, GitHub integration
- **Cons**: Less control than Cloud Run

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

cd backend
fly launch
fly deploy
```

- **Free tier**: 3 shared VMs, 160 GB bandwidth
- **Pros**: Global edge deployment
- **Cons**: More complex networking

### Render.com

1. Connect GitHub repo
2. Select `backend/` as root
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

- **Free tier**: 750 hours/mo
- **Pros**: Dead simple, free tier
- **Cons**: Spins down after 15 min inactivity (cold starts)

---

## ✅ Deployment Checklist

```
Pre-deployment:
  □ All API keys configured in .env
  □ Trained models in backend/data/models/
  □ Frontend builds successfully (npm run build)
  □ Backend runs locally without errors
  □ All tests pass

Backend (Cloud Run):
  □ Dockerfile created and tested locally
  □ .dockerignore configured
  □ CORS updated for production domain
  □ API keys set via env vars or Secret Manager
  □ Image built and pushed to GCR
  □ Cloud Run service deployed
  □ Health check returns 200
  □ API endpoints respond correctly

Frontend (Vercel):
  □ GitHub repo connected to Vercel
  □ Root directory set to 'frontend'
  □ NEXT_PUBLIC_API_URL set to Cloud Run URL
  □ Build succeeds on Vercel
  □ Site loads and search works
  □ Signals generate correctly

Post-deployment:
  □ Custom domain configured (optional)
  □ HTTPS verified on both frontend and backend
  □ Monitoring alerts set up
  □ CI/CD pipeline configured
  □ README updated with production URLs
```
