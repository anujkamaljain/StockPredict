# ☁️ Cloud Training Guide — Train on GPU, Run Locally

> You don't have a local GPU — this guide shows you how to train all models on
> **Google Colab (free T4 GPU)** and use the trained weights on your local CPU machine.

---

## 🎯 The Workflow

```
┌─────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  Your Local PC   │────▶│  Google Colab (GPU) │────▶│  Your Local PC   │
│  Upload backend/ │     │  Train all 6 models │     │  Download models │
│                  │     │  ~45 min on T4 GPU  │     │  Place in data/  │
│                  │     │                      │     │  Restart FastAPI │
└─────────────────┘     └────────────────────┘     └──────────────────┘
```

**Total output**: ~36 MB of model files → easy to download.

---

## 📦 What Gets Trained

| Model | File | Size | Description |
|-------|------|------|-------------|
| LSTM | `lstm_best.pt` | ~5 MB | Bidirectional LSTM with attention |
| Transformer | `transformer_best.pt` | ~15 MB | Temporal Fusion Transformer |
| CNN | `cnn_best.pt` | ~8 MB | Multi-scale 1D CNN |
| XGBoost | `xgboost_model.json` | ~2 MB | Gradient boosted trees |
| LightGBM | `lightgbm_model.pkl` | ~1 MB | Fast gradient boosting |
| CatBoost | `catboost_model.cbm` | ~3 MB | Categorical boosting |
| Ensemble | `ensemble_model.pkl` | ~1 MB | Meta-learner (stacking) |
| Scaler | `scaler.pkl` | ~1 MB | Feature normalizer |
| Feature list | `feature_names.json` | ~5 KB | Feature column names |
| Report | `training_report.json` | ~2 KB | Training metrics |

---

## 🚀 Method 1: Google Colab (Recommended — FREE)

### Step 1: Open Google Colab

1. Go to → [https://colab.research.google.com](https://colab.research.google.com)
2. Click **"New Notebook"**
3. **Enable GPU**: `Runtime` → `Change runtime type` → **T4 GPU** → Save

> [!TIP]
> Google Colab gives you **~12 hours** of free GPU time per session.
> Training all 6 models on 5 tickers takes **~30-45 minutes** on T4.

### Step 2: Upload Your Backend Code

**Option A — Upload from local (simplest):**

In the first cell, run:

```python
# Mount Google Drive (optional but recommended for persistence)
from google.colab import drive
drive.mount('/content/drive')
```

Then upload your `backend/` folder:
1. Click the **📁 folder icon** on the left sidebar
2. Right-click → **Upload folder**
3. Select your `backend/` folder

**Option B — Clone from GitHub (if you have a repo):**

```python
!git clone https://github.com/YOUR_USERNAME/StockMarketPredectior.git
%cd StockMarketPredectior/backend
```

**Option C — Upload as ZIP:**

```python
from google.colab import files

# Upload backend.zip from your local machine
uploaded = files.upload()

# Unzip
!unzip backend.zip -d /content/
%cd /content/backend
```

### Step 3: Install Dependencies

```python
# Install all dependencies
!pip install -r requirements.txt

# Verify GPU is available
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

Expected output:
```
GPU available: True
GPU: Tesla T4
Memory: 15.1 GB
```

### Step 4: Set Your Alpha Vantage API Key

```python
import os
os.environ["ALPHA_VANTAGE_API_KEY"] = "YOUR_ALPHA_VANTAGE_KEY_HERE"
```

> [!IMPORTANT]
> Alpha Vantage free tier = 25 requests/day. If training on 5+ tickers, the system
> will automatically fall back to Yahoo Finance after hitting the limit.
> **Tip**: Fetch data locally first, cache it, then upload the cache to Colab.

### Step 5: Run Training

```python
# Basic training — 5 core US stocks
!python train_cloud.py \
    --tickers AAPL,MSFT,GOOGL,AMZN,NVDA \
    --epochs 100 \
    --batch_size 64 \
    --seq_length 60 \
    --start 2012-01-01 \
    --alpha_vantage_key YOUR_ALPHA_VANTAGE_KEY_HERE
```

**Expanded training** (more tickers for better generalization):

```python
!python train_cloud.py \
    --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,JNJ,V,PG,UNH,HD,MA,DIS \
    --epochs 150 \
    --batch_size 64 \
    --seq_length 60 \
    --start 2010-01-01 \
    --patience 20 \
    --alpha_vantage_key YOUR_ALPHA_VANTAGE_KEY_HERE
```

**Indian stocks:**

```python
!python train_cloud.py \
    --tickers RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS \
    --epochs 100 \
    --start 2015-01-01 \
    --alpha_vantage_key YOUR_ALPHA_VANTAGE_KEY_HERE
```

> [!NOTE]
> **Indian stocks with Alpha Vantage**: Use BSE symbols (e.g., `RELIANCE.BSE`).
> For NSE, Yahoo Finance fallback (`RELIANCE.NS`) handles these well.

### Step 6: Monitor Training

You'll see output like:

```
🎯 Training on: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

📊 STEP 1: Fetching data...
   🥇 Primary: Alpha Vantage | 🥈 Fallback: Yahoo Finance
   ✅ Alpha Vantage API key configured
  Fetching AAPL... [AV: 3200 rows] ✅ 3140 samples, 107 features
  Fetching MSFT... [AV: 3200 rows] ✅ 3140 samples, 107 features
  ...

📦 Combined dataset: 15700 samples, 107 features
   Class balance: 52.3% positive

============================================================
  Training lstm
============================================================
  Epoch  10/100: train=0.6823  val=0.6891
  Epoch  20/100: train=0.6654  val=0.6745
  ...
  ✅ Best epoch: 67, val_loss: 0.6412

============================================================
  Training transformer
============================================================
  ...

  ✅ TRAINING COMPLETE
  Models saved to: /content/backend/trained_models
```

### Step 7: Download Trained Models

**Option A — Direct download:**

```python
# Zip the trained models
!zip -r /content/trained_models.zip /content/backend/trained_models/

# Download to your local machine
from google.colab import files
files.download('/content/trained_models.zip')
```

**Option B — Save to Google Drive (persistent):**

```python
!cp -r /content/backend/trained_models /content/drive/MyDrive/StockML_Models/
```

### Step 8: Deploy Models Locally

1. **Unzip** `trained_models.zip` on your local machine
2. **Copy** all files to:
   ```
   F:\Coding\StockMarketPredectior\backend\data\models\
   ```
3. **Verify** the files are there:
   ```powershell
   ls F:\Coding\StockMarketPredectior\backend\data\models\
   ```
   Expected:
   ```
   lstm_best.pt
   transformer_best.pt
   cnn_best.pt
   xgboost_model.json
   lightgbm_model.pkl
   catboost_model.cbm
   ensemble_model.pkl
   scaler.pkl
   feature_names.json
   training_report.json
   ```
4. **Restart** your local FastAPI server:
   ```powershell
   cd F:\Coding\StockMarketPredectior\backend
   .\venv\Scripts\Activate.ps1
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
5. **Test** — signals now use trained models:
   ```powershell
   curl http://localhost:8000/api/signals/generate/AAPL
   ```

---

## 🔄 Method 2: Kaggle Notebooks (Alternative FREE)

Kaggle offers **30 hours/week** of free GPU (P100 or T4).

1. Go to → [https://www.kaggle.com](https://www.kaggle.com)
2. **New Notebook** → Enable GPU accelerator
3. Upload your `backend/` folder as a dataset
4. Run the same training commands

> [!TIP]
> Kaggle keeps your notebook outputs for 30 days — useful for versioning models.

---

## ☁️ Method 3: GCP Vertex AI (Paid, Production-grade)

For production-level training with more control:

### Setup

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create a custom training job
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=stockml-training \
    --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/YOUR_PROJECT/stockml-train:latest
```

### Dockerfile for GCP

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "train_cloud.py"]
```

### Push & Run

```bash
docker build -t gcr.io/YOUR_PROJECT/stockml-train:latest .
docker push gcr.io/YOUR_PROJECT/stockml-train:latest
```

---

## 💡 Pro Tips

### Maximize Free Colab Time

1. **Don't leave Colab idle** — sessions timeout after ~30 min of inactivity
2. **Save checkpoints to Drive** — in case the session dies
3. **Use `compact` output size** for Alpha Vantage during testing (100 days vs 20+ years)
4. **Pre-cache data locally** → upload cache to Colab to avoid API calls

### Handle Alpha Vantage Rate Limits

```python
# If you have lots of tickers, pre-fetch data:
# Run this LOCALLY first (where rate limits don't matter as much):
python -c "
from app.data.providers.alpha_vantage import AlphaVantageProvider
av = AlphaVantageProvider()
for t in ['AAPL','MSFT','GOOGL','AMZN','NVDA']:
    av.fetch_daily(t, use_cache=True)
    print(f'Cached {t}')
"
# Then upload the cache/ folder to Colab
```

### Retraining Schedule

| Frequency | When | Why |
|-----------|------|-----|
| **Monthly** | First weekend of each month | Market regimes shift |
| **After major events** | Fed meetings, earnings season | Structural breaks |
| **When performance drops** | Sharpe < 0.5 for 2+ weeks | Model decay |

### Model Versioning

Keep track of your trained models:

```
data/models/
├── v1_2026-05-10/          # First training
│   ├── lstm_best.pt
│   ├── training_report.json
│   └── ...
├── v2_2026-06-01/          # Monthly retrain
│   └── ...
└── current/                # Symlink to active version
    └── ...
```

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `--batch_size` to 32 or 16 |
| `No data fetched` | Check internet; AV rate limit → Yahoo fallback should work |
| Session disconnected | Save to Google Drive; resume with cached data |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |
| Models not loading locally | Check file paths — must be in `backend/data/models/` |
| Colab keeps disconnecting | Open browser console, run: `setInterval(() => { document.querySelector("colab-connect-button").click() }, 60000)` |
| Training too slow on CPU | **You MUST use GPU runtime** — `Runtime` → `Change runtime type` → T4 GPU |

---

## 📊 Expected Training Times

| Configuration | T4 GPU | CPU (don't do this) |
|--------------|--------|---------------------|
| 5 tickers, 100 epochs | ~30 min | ~6 hours |
| 15 tickers, 150 epochs | ~90 min | ~18 hours |
| 30 tickers, 200 epochs | ~3 hours | ~36 hours |

> [!CAUTION]
> **Never train on CPU in Colab** — it wastes your session time.
> Always enable GPU before running `train_cloud.py`.
