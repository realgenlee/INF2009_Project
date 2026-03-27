import glob
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs("models", exist_ok=True)

# ==============================================================================
# Configuration
# ==============================================================================
ROLL_WINDOW = 100
NORMAL_LABELS = {"good_contact", "cold_temp"}
RANDOM_STATE = 42
STARTUP_TRIM_ROWS = 150

# Allowed data directories
TRAIN_DIR = "training_data_stage2_train"
VAL_DIR   = "training_data_stage2_val"

# Stage 2 feature set: use only derived / physiological context features
# Avoid absolute ir_raw and absolute t_room to reduce confounding.
FEATURE_COLS = [
    "ac_dc_ratio",
    "peak_to_peak",
    "p2p_cv",
    "ac_dc_ratio_std",
    "peak_to_peak_std",
    "ir_std_5s",
    "ir_slope",
    "ir_drop",
    "low_variation",
    "hr",
    "hr_std_5s",
    "hr_valid",
    "spo2_est",
    "dt_room",
    "t_room_std_5s",
]

# Target false positive rate on validation-normal set.
# Example: 0.05 means about 5% of truly normal validation rows may be flagged.
TARGET_NORMAL_FPR = 0.05

print(f"[CONFIG] {len(FEATURE_COLS)} Stage 2 features")
print(f"[CONFIG] Normal labels: {sorted(NORMAL_LABELS)}")
print(f"[CONFIG] Target validation false positive rate: {TARGET_NORMAL_FPR:.2f}")


# ==============================================================================
# Utilities
# ==============================================================================
def _coerce_numeric(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def per_file_engineering(g: pd.DataFrame) -> pd.DataFrame:
    """
    Derive rolling features if missing.
    """
    g = g.copy()

    if "ir_raw" not in g.columns:
        g["ir_raw"] = np.nan
    if "t_room" not in g.columns:
        g["t_room"] = np.nan
    if "hr" not in g.columns:
        g["hr"] = np.nan
    if "spo2_est" not in g.columns:
        g["spo2_est"] = np.nan
    if "ac_dc_ratio" not in g.columns:
        g["ac_dc_ratio"] = np.nan
    if "peak_to_peak" not in g.columns:
        g["peak_to_peak"] = np.nan

    def fill_if_missing(col, fallback_fn):
        g[col] = pd.to_numeric(g.get(col), errors="coerce")
        if g[col].isna().all():
            g[col] = fallback_fn()

    fill_if_missing(
        "hr_std_5s",
        lambda: g["hr"].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
    )
    fill_if_missing(
        "ir_std_5s",
        lambda: g["ir_raw"].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
    )
    fill_if_missing(
        "ir_slope",
        lambda: g["ir_raw"].diff().fillna(0)
    )
    fill_if_missing(
        "ac_dc_ratio_std",
        lambda: g["ac_dc_ratio"].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
    )
    fill_if_missing(
        "peak_to_peak_std",
        lambda: g["peak_to_peak"].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
    )
    fill_if_missing(
        "t_room_std_5s",
        lambda: g["t_room"].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
    )
    fill_if_missing(
        "dt_room",
        lambda: g["t_room"].diff().fillna(0)
    )
    fill_if_missing(
        "hr_valid",
        lambda: g["hr"].notna().astype(float)
    )
    fill_if_missing(
        "ir_drop",
        lambda: (g["ir_raw"].rolling(ROLL_WINDOW, min_periods=1).mean() < 5000).astype(float)
    )
    fill_if_missing(
        "low_variation",
        lambda: (g["peak_to_peak"].rolling(ROLL_WINDOW, min_periods=1).mean() < 1000).astype(float)
    )
    fill_if_missing(
        "p2p_cv",
        lambda: (
            g["peak_to_peak"].rolling(ROLL_WINDOW, min_periods=1).mean() /
            (g["ir_raw"].rolling(ROLL_WINDOW, min_periods=1).mean() + 1e-6)
        )
    )

    return g

def load_and_engineer(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    sort_cols = ["source_file"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    base_cols = ["timestamp", "ir_raw", "ac_dc_ratio", "peak_to_peak", "t_room", "hr", "spo2_est"]
    df = _coerce_numeric(df, [c for c in base_cols if c in df.columns])

    # Startup trim per file
    before = len(df)
    trimmed = []
    for _, g in df.groupby("source_file", sort=False):
        if len(g) > STARTUP_TRIM_ROWS:
            trimmed.append(g.iloc[STARTUP_TRIM_ROWS:])
        else:
            trimmed.append(g)
    df = pd.concat(trimmed, ignore_index=True)
    print(f"[LOAD] {data_dir}: startup trim {before} -> {len(df)} rows")

    engineered = []
    for _, g in df.groupby("source_file", sort=False):
        engineered.append(per_file_engineering(g))
    df = pd.concat(engineered, ignore_index=True)

    df = _coerce_numeric(df, FEATURE_COLS)

    if "label" not in df.columns:
        raise ValueError(f"'label' column missing in {data_dir}")

    df["is_normal"] = df["label"].isin(NORMAL_LABELS).astype(int)

    print(f"\n[LOAD] {data_dir} label counts:")
    print(df["label"].value_counts().to_string())
    print(f"\n[LOAD] {data_dir} normal rows: {int((df['is_normal'] == 1).sum())}")
    print(f"[LOAD] {data_dir} non-normal rows: {int((df['is_normal'] == 0).sum())}")

    return df


def feature_importance_report(X, pseudo_y, feat_cols):
    """
    Optional interpretability helper:
    uses a small RandomForest on pseudo-labels if available.
    Here pseudo_y can be any binary grouping for rough importance inspection.
    """
    print("\n[FEAT IMPORTANCE] Quick feature ranking...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, pseudo_y)
    ranked = sorted(zip(feat_cols, rf.feature_importances_), key=lambda x: -x[1])
    for name, imp in ranked:
        bar = "¦" * int(imp * 80)
        print(f"  {name:<22} {imp:.4f}  {bar}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    # --------------------------------------------------------------------------
    # Load train and validation sets
    # --------------------------------------------------------------------------
    df_train = load_and_engineer(TRAIN_DIR)
    df_val   = load_and_engineer(VAL_DIR)

    # Keep ONLY normal rows for Stage 2 training
    train_norm = df_train[df_train["is_normal"] == 1].copy()
    val_norm   = df_val[df_val["is_normal"] == 1].copy()

    if train_norm.empty:
        raise ValueError("No normal rows found in training set.")
    if val_norm.empty:
        raise ValueError("No normal rows found in validation set.")

    print(f"\n[TRAIN] Normal training rows: {len(train_norm)}")
    print(f"[VAL]   Normal validation rows: {len(val_norm)}")

    X_train = train_norm[FEATURE_COLS].values.astype(np.float32)
    X_val   = val_norm[FEATURE_COLS].values.astype(np.float32)

    # --------------------------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_val_i   = imputer.transform(X_val)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)
    X_val_s   = scaler.transform(X_val_i)

    # --------------------------------------------------------------------------
    # Optional feature ranking:
    # compare good_contact vs cold_temp to see which features separate contexts
    # --------------------------------------------------------------------------
    pseudo_y = (train_norm["label"] == "cold_temp").astype(int).values
    feature_importance_report(X_train_i, pseudo_y, FEATURE_COLS)

    # --------------------------------------------------------------------------
    # Train anomaly detector
    # --------------------------------------------------------------------------
    print("\n[TRAIN] Training Isolation Forest on normal data only...")
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_s)

    # --------------------------------------------------------------------------
    # Validation on normal-only set:
    # use validation scores to set a practical threshold
    # Lower decision_function score = more anomalous
    # --------------------------------------------------------------------------
    train_scores = model.decision_function(X_train_s)
    val_scores   = model.decision_function(X_val_s)

    # Threshold tuned on validation normal data
    # e.g. bottom 5% of validation-normal scores are considered anomalies
    threshold = float(np.percentile(val_scores, 100 * TARGET_NORMAL_FPR))

    val_pred_anomaly = (val_scores < threshold).astype(int)
    n_flagged = int(val_pred_anomaly.sum())
    normal_fpr = float(n_flagged / len(val_pred_anomaly))

    print("\n[EVAL] Normal-only validation")
    print(f"[EVAL] Score threshold: {threshold:.6f}")
    print(f"[EVAL] Flagged normal rows: {n_flagged}/{len(val_pred_anomaly)}")
    print(f"[EVAL] Estimated false positive rate on normal validation set: {normal_fpr:.4f}")

    # Also show good_contact vs cold_temp separately
    val_breakdown = val_norm.copy()
    val_breakdown["anomaly_score"] = val_scores
    val_breakdown["pred_anomaly"] = val_pred_anomaly

    print("\n[EVAL] Validation anomaly rate by label:")
    for label_name, g in val_breakdown.groupby("label"):
        rate = float(g["pred_anomaly"].mean())
        print(f"  {label_name:<15} {rate:.4f} ({int(g['pred_anomaly'].sum())}/{len(g)})")


    # --------------------------------------------------------------------------
    # Save model bundle
    # --------------------------------------------------------------------------
    bundle = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "roll_window": ROLL_WINDOW,
        "normal_labels": sorted(NORMAL_LABELS),
        "score_threshold": threshold,
        "target_normal_fpr": TARGET_NORMAL_FPR,
        "model_type": "IsolationForest Stage2 Normal-Deviation Detector",
    }
    joblib.dump(bundle, "models/stage2_anomaly_detector.pkl", compress=3)

    metrics = {
        "train_dir": TRAIN_DIR,
        "val_dir": VAL_DIR,
        "n_train_normal_rows": int(len(train_norm)),
        "n_val_normal_rows": int(len(val_norm)),
        "score_threshold": threshold,
        "target_normal_fpr": TARGET_NORMAL_FPR,
        "estimated_val_normal_fpr": normal_fpr,
        "feature_cols": FEATURE_COLS,
        "startup_trim_rows": STARTUP_TRIM_ROWS,
        "normal_labels": sorted(NORMAL_LABELS),
        "validation_anomaly_rate_by_label": {
            label: float(g["pred_anomaly"].mean())
            for label, g in val_breakdown.groupby("label")
        },
    }
    with open("models/stage2_anomaly_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[EXPORT] Saved:")
    print("  models/stage2_anomaly_detector.pkl")
    print("  models/stage2_anomaly_metrics.json")
    print("[TRAIN] Done.")


if __name__ == "__main__":
    main()
