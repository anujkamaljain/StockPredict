# Cloud Training Guide — Train on GPU, Run Locally

> **Read this whole guide once before starting.** It's written for a newbie:
> every step is copy-paste, with screenshots-describable hints and a
> troubleshooting section at the bottom.
>
> You don't have a local GPU. You will train on **Google Colab (free T4 GPU)**
> *or* on your **college shared DGX node**, then download the trained files
> and drop them into your local project for inference.

---

## What you need before starting

### 1. An Alpha Vantage API key (free, ~30 seconds)

- Go to <https://www.alphavantage.co/support/#api-key>
- Enter any name and email → **Get Free API Key** button
- Copy the 16-character key (looks like `XH582O310WG11HT2`)
- This gives you 25 requests/day. If you hit the limit, the system auto-falls back to Yahoo Finance — so you can still train, just slower for the first run.

### 2. A way to upload `backend/` to the cloud

Either:
- **Push to GitHub** (recommended — easiest re-use): `git init && git add . && git commit -m "init" && git push`
- **Zip it locally**: right-click `backend/` folder → *Send to* → *Compressed (zipped) folder* → upload the ZIP

### 3. Pick your training environment

| You have… | Use… | Time | Cost |
|---|---|---|---|
| Just a laptop | **Path A — Google Colab** (free T4 GPU) | 30–90 min | $0 |
| SSH access to a college DGX | **Path B — DGX node** (V100 / A100) | 10–50 min | $0 |

Both paths produce the same trained files. Pick one.

---

## The complete workflow at a glance

```
┌──────────────┐   ┌────────────────────────┐   ┌────────────────────┐
│ Your laptop  │──▶│ Colab T4  OR  DGX node │──▶│ Your laptop        │
│ Upload code  │   │ Run train_cloud.py     │   │ Drop trained files │
│              │   │ Verify gates pass      │   │ in data/models/    │
└──────────────┘   └────────────────────────┘   └────────────────────┘
       │                                                  │
       └──── Then proceed to deployment_guide.md ─────────┘
```

You only have to do **two things**:
1. Read this guide → train your models (this file).
2. Read `deployment_guide.md` → deploy.

---

## What the training script produces

Every successful training writes these 11 files into `trained_models/`. You will
copy them later into `backend/data/models/`.

| File | Size | Purpose |
|------|------|---------|
| `lstm_best.pt` | ~3 MB | Bidirectional LSTM with attention |
| `transformer_best.pt` | ~4 MB | Temporal Fusion Transformer |
| `cnn_best.pt` | ~4 MB | Multi-scale 1D CNN |
| `xgboost_model.json` | ~2 MB | XGBoost classifier |
| `lightgbm_model.pkl` | ~1 MB | LightGBM classifier |
| `catboost_model.cbm` | ~3 MB | CatBoost classifier |
| `ensemble_model.pkl` | ~10 KB | Stacking meta-learner + isotonic calibration |
| `scaler.pkl` | ~2 KB | RobustScaler (fit on TRAIN only) |
| `feature_names.json` | ~1 KB | Selected feature names |
| `selected_feature_indices.npy` | ~500 B | Indices of selected features |
| `training_report.json` | ~10 KB | All metrics, gate results, training histories |

> **The training_report.json is your proof that the model is good.** Check
> `acceptance_gates.passed == true` before deploying. If it is `false`, the
> training script also exits with non-zero status — don't skip this check.

---

## Robustness features built into `train_cloud.py`

| Feature | Why it matters |
|---|---|
| Per-ticker sequence construction | Prevents cross-ticker contamination in 60-day windows |
| Aligned tree + deep predictions | Prevents off-by-`seq_length` ensemble bugs |
| Reproducibility seed (`--seed`) | Same input = same model every time |
| Feature selection on TRAIN only | No test leakage |
| Label smoothing | Better-calibrated probabilities |
| Purged gap between train/val/test | Prevents rolling-window leakage at split boundaries |
| Early stopping on validation **AUC** | Robust model selection |
| LR warmup + cosine annealing | Stable optimization |
| Gradient clipping + AdamW + weight decay | No exploding gradients |
| Mixed precision (`--use_amp`) | Faster training, lower VRAM on GPU |
| Cross-validated isotonic calibration | Well-calibrated final probabilities |
| Threshold optimization on validation | Picks optimal trading thresholds |
| Permutation-importance sanity check | Confirms model uses real signal |
| **Mandatory acceptance gates** | Training **fails loudly** on bad models |

### Acceptance gates the script enforces

Training will exit with **non-zero status** (and `acceptance_gates.passed = false`) unless:

- Test AUC ≥ **0.52** (better than coin flip)
- Test accuracy ≥ **0.51**
- Test Brier ≤ **0.26** (better than uniform)
- Train–Test AUC gap ≤ **0.10** (no severe overfitting)
- Each individual model has test AUC ≥ **0.50**
- Ensemble AUC ≥ best individual − 0.005 (the ensemble actually adds value)
- Permutation-importance AUC drop ≥ **0.005** (model uses real signal, not noise)

Override only with `--force_accept` (**not recommended**).

---

## Path A — Google Colab (free T4 GPU)

### A.1 — Open Colab and enable GPU

1. Go to <https://colab.research.google.com>
2. Click **New Notebook**
3. Top menu: **Runtime** → **Change runtime type** → set **Hardware accelerator: T4 GPU** → **Save**
4. Free tier gives you ~12 hours of GPU/session. A 10-ticker 150-epoch run takes ~1 hour, well within budget.

### A.2 — Upload the `backend/` folder

Pick whichever upload method is easiest:

**Option A — Drag-and-drop (simplest):**
1. Click the **folder icon** on the left sidebar
2. Right-click in the file explorer → *Upload folder*
3. Select your local `backend/` folder

**Option B — Clone from GitHub:**

```python
!git clone https://github.com/YOUR_USERNAME/StockMarketPredectior.git
%cd StockMarketPredectior/backend
```

**Option C — Upload a ZIP:**

```python
from google.colab import files
uploaded = files.upload()      # pick your backend.zip
!unzip -q backend.zip -d /content/
%cd /content/backend
```

### A.3 — Install dependencies (modern stack)

```python
!pip install -q -r requirements.txt
```

Verify the GPU is visible:

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

Expected:
```
GPU available: True
GPU: Tesla T4
Memory: 15.1 GB
```

If `GPU available: False`, you forgot step A.1 — go back and enable T4 GPU.

### A.4 — Run training (copy-paste this entire block)

> Replace `YOUR_ALPHA_VANTAGE_KEY_HERE` with the key you got at the top of this guide.

```python
!python train_cloud.py \
    --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,META,JPM,JNJ,V,PG \
    --start 2010-01-01 \
    --epochs 150 \
    --patience 25 \
    --seq_length 60 \
    --max_features 80 \
    --batch_size 64 \
    --lr 0.0005 \
    --label_smoothing 0.05 \
    --gap 5 \
    --seed 42 \
    --use_amp \
    --device auto \
    --alpha_vantage_key YOUR_ALPHA_VANTAGE_KEY_HERE
```

You will see output like:

```
======================================================================
 ROBUST CLOUD TRAINING — STOCK MARKET PREDICTOR
======================================================================
Tickers       : ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'JNJ', 'V', 'PG']
...
Device        : cuda
Mixed prec.   : True
======================================================================

--- STEP 1: Fetching data ---
  Fetching AAPL... [Yahoo: 3650 rows] OK - 3450 samples
  Fetching MSFT... [Yahoo: 3650 rows] OK - 3450 samples
  ...
============================================================
  Training lstm
============================================================
  Epoch   5/150: train=0.6823  val=0.6712  val_AUC=0.5621  lr=0.000500  [OK] (best)
  ...
  Best epoch: 67, val_loss: 0.6412, val_AUC: 0.6021
```

### A.5 — Verify gates passed

The last block of output will look like one of these:

**Success:**
```
======================================================================
  ALL ACCEPTANCE GATES PASSED. Model is ready.
======================================================================
```

**Failure (and what to do):**
```
======================================================================
  ACCEPTANCE GATES FAILED:
======================================================================
  1. Test AUC 0.5048 < min 0.5200 — model is no better than random.
  2. Train-Test AUC gap 0.1543 > max 0.1000 — overfitting.

  Recommendations:
    - Add more tickers / longer history (--tickers / --start)
    - Increase --epochs and --patience
    - Try a different --seq_length (30, 90, 120)
    - Adjust --max_features (try 40 or 120)
```

> **If gates failed: try the recommendations and re-run. Do not deploy.**

### A.6 — Download `trained_models/` to your laptop

```python
!zip -qr /content/trained_models.zip /content/backend/trained_models/
from google.colab import files
files.download('/content/trained_models.zip')
```

A "trained_models.zip" will be downloaded to your computer's Downloads folder.

(Optional, if your Colab session might die before downloading) Save to Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/backend/trained_models /content/drive/MyDrive/StockML_Models/
```

### A.7 — Now jump to "Deploy Trained Models Locally" below

---

## Path B — College Shared DGX node (older Python / CUDA OK)

DGX nodes typically run an older but stable software stack. Use
`backend/requirements_dgx.txt` — it pins versions that have prebuilt wheels
(no compiler, no CMake required).

### B.1 — Connect to the DGX

```bash
ssh your-user@dgx.your-college.edu
```

If the DGX uses Slurm, request an interactive GPU node first:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 --pty bash
```

### B.2 — Copy your `backend/` folder up

Pick whichever fits your access:

**Option A — `scp` from your laptop:**
```bash
scp -r F:\Coding\StockMarketPredectior\backend your-user@dgx.your-college.edu:~/stockml/
```

**Option B — Clone from GitHub on the DGX:**
```bash
git clone https://github.com/YOUR_USERNAME/StockMarketPredectior.git
cd StockMarketPredectior/backend
```

### B.3 — Create a venv and install pinned deps

```bash
cd ~/stockml/backend          # or wherever you put it
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# Use the DGX-friendly pinned requirements (prebuilt wheels only)
pip install -r requirements_dgx.txt
```

If `torch.cuda.is_available()` is `False`, install the matching CUDA build for the DGX. Check the CUDA version with `nvidia-smi` first (e.g. `12.1`):

```bash
pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cu121
```

### B.4 — Verify GPU is visible

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

You should see your GPU name (V100 / A100 / H100).

### B.5 — Run training (interactive)

DGX GPUs are powerful — use the full multi-market universe (US + India, all caps + ETFs).

**If using Jupyter notebook** (copy-paste this single line):

```python
!python train_cloud.py --tickers "AAPL,MSFT,GOOGL,AMZN,NVDA,META,JPM,JNJ,V,PG,UNH,HD,MA,DIS,TSLA,BRK-B,XOM,WMT,KO,PFE,SQ,SNAP,NET,ENPH,DKNG,CROX,FIVE,DECK,ETSY,ROKU,SOFI,AFRM,UPST,CELH,RBLX,PUBM,SFIX,IONQ,SPY,QQQ,IWM,DIA,ARKK,XLF,XLE,GLD,VTI,EEM,RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS,BHARTIARTL.NS,ITC.NS,SBIN.NS,LT.NS,KOTAKBANK.NS,HINDUNILVR.NS,BAJFINANCE.NS,MARUTI.NS,TATAMOTORS.NS,WIPRO.NS,MPHASIS.NS,COFORGE.NS,TRENT.NS,AUROPHARMA.NS,JUBLFOOD.NS,FEDERALBNK.NS,VOLTAS.NS,TATAELXSI.NS,PERSISTENT.NS,CDSL.NS,DEEPAKNTR.NS,KPITTECH.NS,ATUL.NS,ROUTE.NS,RATNAMANI.NS,NIFTYBEES.NS,BANKBEES.NS,GOLDBEES.NS,JUNIORBEES.NS" --start 2010-01-01 --epochs 200 --patience 35 --seq_length 60 --max_features 120 --batch_size 128 --lr 0.0005 --label_smoothing 0.05 --gap 7 --seed 42 --use_amp --device auto --output_dir trained_models
```

**If using a bash terminal** (SSH into DGX):

```bash
python train_cloud.py \
    --tickers "AAPL,MSFT,GOOGL,AMZN,NVDA,META,JPM,JNJ,V,PG,UNH,HD,MA,DIS,TSLA,BRK-B,XOM,WMT,KO,PFE,SQ,SNAP,NET,ENPH,DKNG,CROX,FIVE,DECK,ETSY,ROKU,SOFI,AFRM,UPST,CELH,RBLX,PUBM,SFIX,IONQ,SPY,QQQ,IWM,DIA,ARKK,XLF,XLE,GLD,VTI,EEM,RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS,BHARTIARTL.NS,ITC.NS,SBIN.NS,LT.NS,KOTAKBANK.NS,HINDUNILVR.NS,BAJFINANCE.NS,MARUTI.NS,TATAMOTORS.NS,WIPRO.NS,MPHASIS.NS,COFORGE.NS,TRENT.NS,AUROPHARMA.NS,JUBLFOOD.NS,FEDERALBNK.NS,VOLTAS.NS,TATAELXSI.NS,PERSISTENT.NS,CDSL.NS,DEEPAKNTR.NS,KPITTECH.NS,ATUL.NS,ROUTE.NS,RATNAMANI.NS,NIFTYBEES.NS,BANKBEES.NS,GOLDBEES.NS,JUNIORBEES.NS" \
    --start 2010-01-01 \
    --epochs 200 \
    --patience 35 \
    --seq_length 60 \
    --max_features 120 \
    --batch_size 128 \
    --lr 0.0005 \
    --label_smoothing 0.05 \
    --gap 7 \
    --seed 42 \
    --use_amp \
    --device auto \
    --output_dir trained_models
```

> **Ticker breakdown (82 total):**
>
> | Market | Segment | Count | Examples |
> |--------|---------|-------|----------|
> | US | Large cap | 20 | AAPL, MSFT, NVDA, JPM, TSLA… |
> | US | Mid cap | 10 | SQ, NET, ENPH, DKNG, CROX… |
> | US | Small cap | 8 | SOFI, AFRM, UPST, CELH, IONQ… |
> | US | ETFs | 10 | SPY, QQQ, IWM, DIA, ARKK, GLD… |
> | India | Large cap | 15 | RELIANCE, TCS, INFY, HDFC, ICICI… |
> | India | Mid cap | 7 | MPHASIS, COFORGE, TRENT, JUBLFOOD… |
> | India | Small cap | 8 | TATAELXSI, PERSISTENT, CDSL, KPITTECH… |
> | India | ETFs | 4 | NIFTYBEES, BANKBEES, GOLDBEES, JUNIORBEES |
>
> **Tuning for reliability:**
> - `patience 35` — more data needs more convergence time; avoids underfitting
> - `max_features 120` — 82 tickers × 3000+ rows = ~250K samples; can support more features without overfitting
> - `gap 7` — wider purged gap; extra protection against temporal leakage with diverse tickers
> - Acceptance gates still enforce: AUC ≥ 0.52, train-test gap ≤ 0.10, permutation check
>
> **Estimated time on DGX:** ~3–4 hours. No Alpha Vantage key needed (Yahoo handles all).

### B.6 — Or submit as a Slurm batch job

Create `train.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=stockml-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=train_%j.log

cd $HOME/stockml/backend
source venv/bin/activate

python train_cloud.py \
    --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,META,JPM,JNJ,V,PG \
    --start 2010-01-01 \
    --epochs 150 \
    --patience 25 \
    --use_amp \
    --alpha_vantage_key YOUR_ALPHA_VANTAGE_KEY_HERE
```

Submit and monitor:

```bash
sbatch train.sbatch
squeue -u $USER
tail -f train_*.log         # watch live
```

### B.7 — Verify gates passed

Same as Path A.5: look for `ALL ACCEPTANCE GATES PASSED`. If not, fix and re-run.

### B.8 — Copy `trained_models/` back to your laptop

From your **laptop's** terminal (not the DGX):

```bash
scp -r your-user@dgx.your-college.edu:~/stockml/backend/trained_models .
```

---

## Deploy Trained Models Locally (both paths)

This is the final step before you proceed to `deployment_guide.md`.

### Step 1 — Move the files into the project

Open PowerShell on your laptop:

```powershell
# If the zip downloaded from Colab:
Expand-Archive -Path "$env:USERPROFILE\Downloads\trained_models.zip" -DestinationPath F:\Coding\StockMarketPredectior\backend\data\models -Force

# OR if you scp'd a folder from DGX:
Move-Item .\trained_models\* F:\Coding\StockMarketPredectior\backend\data\models\
```

### Step 2 — Verify all 11 files are present

```powershell
Get-ChildItem F:\Coding\StockMarketPredectior\backend\data\models\ | Select-Object Name
```

You should see exactly these:
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
selected_feature_indices.npy
training_report.json
```

(`.gitkeep` is fine if it's also there.)

### Step 3 — Confirm gates passed

```powershell
Get-Content F:\Coding\StockMarketPredectior\backend\data\models\training_report.json |
  ConvertFrom-Json |
  Select-Object -ExpandProperty acceptance_gates
```

Output should include `passed : True`. If `False`, retrain — do **not** deploy a failed model.

### Step 4 — Test locally (optional, recommended)

Start the backend:

```powershell
cd F:\Coding\StockMarketPredectior\backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The startup log should show:
```
Trained models loaded: [LSTM, TFT, CNN, XGB, LGB, CAT] (n_features=80, seq_length=60, gates_passed=True)
```

In another terminal, check the health endpoint:

```powershell
curl http://localhost:8000/health
```

Expected:
```json
{"status":"healthy","models_loaded":true,"loaded_models":["lstm","transformer","cnn","xgboost","lightgbm","catboost"],"gates_passed":true}
```

Test a real signal:

```powershell
curl http://localhost:8000/api/signals/generate/AAPL
```

The response's `meta.signal_source` should be `"ml_ensemble"` (not `"statistical_fallback"`) and `meta.individual_predictions` should show all six base model probabilities.

### Step 5 — Proceed to deployment

Open `deployment_guide.md` and follow it end-to-end. You're done with training.

---

## Hot-reloading models on a running server

If your FastAPI server is already running and you just dropped a fresh
`trained_models/` payload, you can reload without restart:

```powershell
curl -X POST http://localhost:8000/api/signals/reload-models
curl http://localhost:8000/api/signals/model-info     # verify
```

---

## Tuning knobs (quick reference)

| Flag | What it controls | When to change |
|------|------------------|----------------|
| `--tickers` | Universe of stocks | 10+ recommended; more = more robust |
| `--start` | History start date | `2008-01-01` for max data; `2015-01-01` if Alpha Vantage free tier is your only source |
| `--epochs` | Max training epochs | 150–200; early stopping picks the optimum |
| `--patience` | Early-stopping patience | 20–30; higher = let training run longer |
| `--seq_length` | LSTM/Transformer/CNN lookback | 30 / 60 / 90; longer = more memory |
| `--max_features` | Top-N features kept after selection | 60–100. Smaller = less overfitting on small data |
| `--batch_size` | Mini-batch size | 64 on T4, 128 on V100/A100 |
| `--lr` | Learning rate | 0.0005 is the sweet spot |
| `--label_smoothing` | Soft-label factor | 0.05 default; 0.0 disables it |
| `--gap` | Purged gap days between splits | 5 default; raise to 10 if rolling windows are wider |
| `--seed` | RNG seed | Keep at 42 unless A/B testing |
| `--use_amp` | Mixed-precision on CUDA | Always include for GPU training |
| `--device` | `auto` / `cuda` / `cpu` | `auto` for both Colab and DGX |
| `--force_accept` | Skip acceptance gates | **DO NOT USE** for production models |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Lower `--batch_size` to 32; lower `--seq_length` to 30 |
| `No GPU detected, using CPU` on Colab | You forgot Runtime → Change runtime type → T4 GPU |
| `No data fetched` for some tickers | Yahoo Finance fallback should kick in; if Yahoo also fails, the ticker is likely delisted/wrong |
| Gates fail with low AUC | Add more tickers; extend `--start`; raise `--epochs` |
| Gates fail with high train-test gap | More regularization: smaller `--max_features`, more `--label_smoothing` |
| Gates fail with low permutation drop | Features may be noisy; revisit feature engineering |
| `xgboost` / `lightgbm` build error on DGX | You used `requirements.txt` — switch to `requirements_dgx.txt` |
| Colab session disconnected | Save to Drive frequently; restart and re-run |
| Training too slow | Verify `nvidia-smi` shows the GPU is busy; check `--device auto` picked CUDA |
| `Alpha Vantage rate limit hit` | Free tier = 25 req/day. The script auto-falls back to Yahoo Finance |
| Local server still shows `statistical_fallback` after copying files | Run `POST /api/signals/reload-models` or restart the server |
| `models_loaded: false` in `/health` | Files not in `backend/data/models/`; verify with `Get-ChildItem` |

---

## Retraining schedule

Production models drift. Retrain on this cadence:

| Trigger | Why |
|---------|-----|
| **Monthly** (first weekend) | Market regimes drift, model staleness |
| **After major regime shifts** | Fed pivots, earnings season, crisis events |
| **Live performance drops** | Backtest Sharpe < 0.5 for 2+ weeks, hit rate < 0.5 |
| **Adding new tickers** | Need to retrain to learn the new universe |

Always keep your previous `trained_models/` until the new one is verified live.

---

## Expected training times

| Config | T4 (Colab) | V100/A100 (DGX) |
|--------|-----------|------------------|
| 5 tickers, 100 epochs | ~25 min | ~10 min |
| 10 tickers, 150 epochs | ~60 min | ~25 min |
| 15 tickers, 200 epochs | ~2 h | ~50 min |

> Never train on CPU. It wastes your Colab session for nothing.

---

## After training, your only remaining task

Read [`deployment_guide.md`](./deployment_guide.md) end-to-end and follow it.
The trained files you just produced will be baked into the production
Docker image automatically.
