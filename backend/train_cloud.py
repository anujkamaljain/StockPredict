"""
=============================================================================
CLOUD TRAINING SCRIPT — Run this on Google Colab / Kaggle / AWS / GCP
=============================================================================

WORKFLOW:
1. Upload this entire 'backend/' folder to your cloud environment
2. Install dependencies: pip install -r requirements.txt
3. Run: python train_cloud.py --tickers AAPL,MSFT,GOOGL --epochs 100
4. Download the 'trained_models/' folder
5. Place it at: backend/data/models/ on your local machine
6. Restart your local FastAPI server — it will auto-load trained models

The trained model files are:
  - lstm_best.pt          (~5 MB)
  - transformer_best.pt   (~15 MB)
  - cnn_best.pt           (~8 MB)
  - xgboost_model.json    (~2 MB)
  - lightgbm_model.pkl    (~1 MB)
  - catboost_model.cbm    (~3 MB)
  - ensemble_model.pkl    (~1 MB)
  - scaler.pkl            (~1 MB)
  - feature_names.json    (~5 KB)
  - training_report.json  (~2 KB)

Total: ~36 MB — easy to download.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from datetime import datetime

# Add parent to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.data.providers.yahoo import YahooFinanceProvider
from app.data.providers.alpha_vantage import AlphaVantageProvider
from app.data.validation import DataValidator
from app.features.engineering import FeatureEngineer
from app.models.lstm import LSTMModel
from app.models.transformer import TemporalFusionTransformer
from app.models.cnn import MultiScaleCNN
from app.models.ensemble import EnsembleModel

# Try importing tree models (may need separate install)
try:
    from app.models.tree_models import TreeModels
    HAS_TREES = True
except ImportError:
    HAS_TREES = False
    print("⚠️  XGBoost/LightGBM/CatBoost not installed. Skipping tree models.")
    print("   Install with: pip install xgboost lightgbm catboost")


def parse_args():
    parser = argparse.ArgumentParser(description="Train stock ML models on cloud GPU")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOGL,AMZN,NVDA",
                        help="Comma-separated ticker list")
    parser.add_argument("--start", type=str, default="2012-01-01",
                        help="Training data start date")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=60,
                        help="Lookback window for sequence models")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--output_dir", type=str, default="trained_models")
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda, cpu, or auto")
    parser.add_argument("--alpha_vantage_key", type=str, default="",
                        help="Alpha Vantage API key (primary data source)")
    return parser.parse_args()


def detect_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
            return dev
        print("⚠️  No GPU found, using CPU (training will be slow)")
        return torch.device("cpu")
    return torch.device(preference)


def fetch_and_prepare_data(tickers: list, start: str, seq_length: int, av_key: str = ""):
    """
    Fetch data for all tickers and build combined feature matrix.

    Priority:
    🥇 Alpha Vantage (primary — official API)
    🥈 Yahoo Finance (fallback — missing data recovery)
    """
    print("\n📊 STEP 1: Fetching data...")
    print(f"   🥇 Primary: Alpha Vantage | 🥈 Fallback: Yahoo Finance")

    # Initialize providers
    av_api_key = av_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
    alpha_vantage = AlphaVantageProvider(api_key=av_api_key)
    yahoo = YahooFinanceProvider()
    validator = DataValidator()
    engineer = FeatureEngineer()

    if alpha_vantage.is_configured:
        print(f"   ✅ Alpha Vantage API key configured")
    else:
        print(f"   ⚠️  No Alpha Vantage key — using Yahoo Finance only")

    all_X, all_targets = [], []
    feature_names = None

    for ticker in tickers:
        print(f"  Fetching {ticker}...", end=" ")
        df = pd.DataFrame()

        # 🥇 Try Alpha Vantage first
        if alpha_vantage.is_configured and alpha_vantage.requests_remaining > 0:
            try:
                df = alpha_vantage.fetch_daily(ticker, use_cache=False)
                if not df.empty:
                    # Filter by start date
                    df = df[df.index >= pd.Timestamp(start)]
                    print(f"[AV: {len(df)} rows]", end=" ")
            except Exception as e:
                print(f"[AV error: {e}]", end=" ")

        # 🥈 Fallback to Yahoo
        if df.empty:
            df = yahoo.fetch_ohlcv(ticker, start=start, use_cache=False)
            if not df.empty:
                print(f"[Yahoo: {len(df)} rows]", end=" ")

        if df.empty:
            print("❌ No data from any source")
            continue

        df = validator.clean(df, ticker)
        featured = engineer.compute_features(df)
        featured = featured.dropna()

        if len(featured) < 300:
            print(f"❌ Too few rows ({len(featured)})")
            continue

        X, y, names = engineer.prepare_for_training(
            featured, target_col="target_direction", fit_scaler=(feature_names is None)
        )

        if feature_names is None:
            feature_names = names

        all_X.append(X)
        all_targets.append(y)
        print(f"✅ {len(X)} samples, {len(names)} features")

    if not all_X:
        raise RuntimeError("No valid data fetched. Check tickers and internet.")

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_targets, axis=0)

    print(f"\n📦 Combined dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
    print(f"   Class balance: {y_combined.mean():.2%} positive")

    return X_combined, y_combined, feature_names, engineer.scaler


def time_series_split(X, y, train_pct=0.7, val_pct=0.15):
    """Split data chronologically (no shuffling — critical for time series)."""
    n = len(X)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:],
    )


def make_sequences(X, y, seq_length):
    """Create sequence windows for LSTM/Transformer/CNN."""
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length - 1])
    return np.array(Xs), np.array(ys)


def train_deep_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                     device, name, output_dir, epochs=100, batch_size=64,
                     lr=0.001, patience=15):
    """Train a single deep learning model."""
    print(f"\n{'='*60}")
    print(f"  Training {name}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

    X_t = torch.FloatTensor(X_train_seq).to(device)
    y_t = torch.FloatTensor(y_train_seq).to(device)
    X_v = torch.FloatTensor(X_val_seq).to(device)
    y_v = torch.FloatTensor(y_val_seq).to(device)

    best_val_loss = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for i in range(0, len(X_t), batch_size):
            bx = X_t[i:i + batch_size]
            by = y_t[i:i + batch_size]
            if len(bx) < 2:
                continue

            optimizer.zero_grad()
            out = model(bx)
            pred = out["direction_prob"]
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(X_v)
            val_loss = criterion(val_out["direction_prob"], y_v).item()

        t_loss = np.mean(train_losses)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: train={t_loss:.4f}  val={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / f"{name}_best.pt")
        elif epoch - best_epoch >= patience:
            print(f"  ⏹  Early stop at epoch {epoch+1}")
            break

    # Reload best
    model.load_state_dict(torch.load(output_dir / f"{name}_best.pt", weights_only=True))
    print(f"  ✅ Best epoch: {best_epoch+1}, val_loss: {best_val_loss:.4f}")

    # Get predictions for ensemble
    model.eval()
    with torch.no_grad():
        train_pred = model(X_t)["direction_prob"].cpu().numpy()
        val_pred = model(X_v)["direction_prob"].cpu().numpy()

    return model, train_pred, val_pred, history


def main():
    args = parse_args()
    device = detect_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    print(f"🎯 Training on: {tickers}")

    # ---- STEP 1: Fetch & prepare data ----
    X, y, feature_names, scaler = fetch_and_prepare_data(
        tickers, args.start, args.seq_length, av_key=args.alpha_vantage_key
    )

    # Save scaler and feature names
    joblib.dump(scaler, output_dir / "scaler.pkl")
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print(f"💾 Saved scaler and {len(feature_names)} feature names")

    # ---- STEP 2: Split data ----
    X_train, y_train, X_val, y_val, X_test, y_test = time_series_split(X, y)
    print(f"\n📊 Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ---- STEP 3: Create sequences for deep models ----
    seq = args.seq_length
    X_train_seq, y_train_seq = make_sequences(X_train, y_train, seq)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val, seq)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test, seq)
    print(f"🔗 Sequences: train={len(X_train_seq)}, val={len(X_val_seq)}, test={len(X_test_seq)}")

    n_features = X.shape[1]
    ensemble_test_preds = {}
    results = {}

    # ---- STEP 4: Train LSTM ----
    lstm = LSTMModel(input_size=n_features, hidden_size=128, num_layers=2, dropout=0.3)
    lstm, lstm_train_p, lstm_val_p, lstm_hist = train_deep_model(
        lstm, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        device, "lstm", output_dir, args.epochs, args.batch_size, args.lr, args.patience,
    )
    lstm.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test_seq).to(device)
        ensemble_test_preds["lstm"] = lstm(X_test_t)["direction_prob"].cpu().numpy()
    results["lstm"] = {"val_loss": min(lstm_hist["val_loss"])}

    # ---- STEP 5: Train Transformer ----
    tft = TemporalFusionTransformer(
        input_size=n_features, d_model=128, nhead=8,
        num_encoder_layers=4, dim_feedforward=256, dropout=0.1, seq_length=seq,
    )
    tft, tft_train_p, tft_val_p, tft_hist = train_deep_model(
        tft, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        device, "transformer", output_dir, args.epochs, args.batch_size,
        args.lr * 0.5, args.patience,
    )
    tft.eval()
    with torch.no_grad():
        ensemble_test_preds["transformer"] = tft(X_test_t)["direction_prob"].cpu().numpy()
    results["transformer"] = {"val_loss": min(tft_hist["val_loss"])}

    # ---- STEP 6: Train CNN ----
    cnn = MultiScaleCNN(input_size=n_features, dropout=0.3)
    cnn, cnn_train_p, cnn_val_p, cnn_hist = train_deep_model(
        cnn, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        device, "cnn", output_dir, args.epochs, args.batch_size, args.lr, args.patience,
    )
    cnn.eval()
    with torch.no_grad():
        ensemble_test_preds["cnn"] = cnn(X_test_t)["direction_prob"].cpu().numpy()
    results["cnn"] = {"val_loss": min(cnn_hist["val_loss"])}

    # ---- STEP 7: Train tree models ----
    if HAS_TREES:
        print(f"\n{'='*60}")
        print("  Training Tree Models (XGBoost, LightGBM, CatBoost)")
        print(f"{'='*60}")

        trees = TreeModels(model_dir=output_dir)
        try:
            xgb_m = trees.train_xgboost(X_train, y_train, X_val, y_val, feature_names)
            results["xgboost"] = xgb_m
        except Exception as e:
            print(f"  ⚠️  XGBoost failed: {e}")

        try:
            lgb_m = trees.train_lightgbm(X_train, y_train, X_val, y_val, feature_names)
            results["lightgbm"] = lgb_m
        except Exception as e:
            print(f"  ⚠️  LightGBM failed: {e}")

        try:
            cb_m = trees.train_catboost(X_train, y_train, X_val, y_val, feature_names)
            results["catboost"] = cb_m
        except Exception as e:
            print(f"  ⚠️  CatBoost failed: {e}")

        tree_preds = trees.predict(X_test)
        ensemble_test_preds.update(tree_preds)

    # ---- STEP 8: Build ensemble ----
    if len(ensemble_test_preds) >= 2:
        print(f"\n{'='*60}")
        print("  Building Ensemble")
        print(f"{'='*60}")

        min_len = min(len(v) for v in ensemble_test_preds.values())
        aligned_preds = {k: v[:min_len] for k, v in ensemble_test_preds.items()}
        y_ens = y_test_seq[:min_len] if min_len <= len(y_test_seq) else y_test[:min_len]

        ensemble = EnsembleModel(model_dir=output_dir)
        ens_metrics = ensemble.fit(aligned_preds, y_ens)
        results["ensemble"] = ens_metrics
        print(f"  ✅ Ensemble AUC: {ens_metrics['ensemble_auc']:.4f}")

    # ---- STEP 9: Save training report ----
    report = {
        "trained_at": datetime.now().isoformat(),
        "tickers": tickers,
        "data_start": args.start,
        "total_samples": len(X),
        "n_features": n_features,
        "seq_length": seq,
        "epochs": args.epochs,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "results": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                        for kk, vv in v.items()} for k, v in results.items()},
        "models_saved": [f.name for f in output_dir.iterdir() if f.is_file()],
    }
    with open(output_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- DONE ----
    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Models saved to: {output_dir.absolute()}")
    print(f"  Files:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:30s} ({size_kb:8.1f} KB)")

    print(f"\n📥 NEXT STEPS:")
    print(f"  1. Download the '{output_dir}/' folder")
    print(f"  2. Copy it to: backend/data/models/ on your local machine")
    print(f"  3. Restart your FastAPI server")
    print(f"  4. Signals will now use trained ML models! 🚀")


if __name__ == "__main__":
    main()
