import json
import os
import glob
import warnings
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs("models", exist_ok=True)

# ==============================================================================
# Config
# ==============================================================================
PPG_WINDOW         = 250
LABEL_COL          = "label"
STARTUP_TRIM_ROWS  = 150   # drop first ~3s per file (sensor stabilisation)

# FIX: Simplified label normalisation — removed legacy variants that no longer
# appear in data_logger.py's --label choices. Only canonical mappings remain
# to avoid confusion between what data_logger.py produces and what the
# classifier expects.
LABEL_NORMALIZATION = {
    "good_contact_normal_room": "good_contact",
    "good_contact_cold_room":   "good_contact",
    "good_contact_normal":      "good_contact",
    "good_contact_cold":        "good_contact",
    "good_contact_cold_sim":    "good_contact",  # synthetic augmentation rows
}

PREFERRED_FEATURE_COLS = [
    "ir_raw",
    "ac_dc_ratio",
    "peak_to_peak",
    "t_room",
    "hr",
    "hr_std_5s",
]


# ==============================================================================
# Data loading
# ==============================================================================
def load_dataset(data_dir: str = "training_data") -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    files += glob.glob(os.path.join(data_dir, "*", "*.csv"))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"No CSV files in '{data_dir}'. Run data_logger.py first."
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)

        if LABEL_COL not in df.columns:
            parent = os.path.basename(os.path.dirname(f))
            df[LABEL_COL] = parent if parent else "unknown"

        df[LABEL_COL] = (
            df[LABEL_COL].astype(str).str.strip().replace(LABEL_NORMALIZATION)
        )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    required = [LABEL_COL, "ac_dc_ratio", "peak_to_peak"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"[TRAIN] Loaded {len(df)} rows from {len(files)} files.")
    print("[TRAIN] Label counts:")
    print(df[LABEL_COL].value_counts(dropna=False).to_string())
    return df


# ==============================================================================
# Window feature extraction
# ==============================================================================
def _safe_stats(arr: np.ndarray, prefix: str) -> dict:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return {f"{prefix}_{s}": 0.0 for s in
                ["mean","std","min","max","median","p25","p75","range","first","last","slope"]}
    first = float(arr[0])
    last  = float(arr[-1])
    return {
        f"{prefix}_mean":   float(np.mean(arr)),
        f"{prefix}_std":    float(np.std(arr)),
        f"{prefix}_min":    float(np.min(arr)),
        f"{prefix}_max":    float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p25":    float(np.percentile(arr, 25)),
        f"{prefix}_p75":    float(np.percentile(arr, 75)),
        f"{prefix}_range":  float(np.max(arr) - np.min(arr)),
        f"{prefix}_first":  first,
        f"{prefix}_last":   last,
        f"{prefix}_slope":  float((last - first) / max(len(arr) - 1, 1)),
    }


def build_window_table(
    df: pd.DataFrame,
    window_size: int = PPG_WINDOW,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Slide a window of 'window_size' rows with 50% overlap over each
    CSV file's data. Each window becomes one training example.

    With more data, this produces more windows and a more reliable model.
    Current: ~122 windows from 4 sessions.
    Target:  ~300+ windows after data collection above.
    """
    step = window_size // 2
    rows = []
    labels = []

    if "source_file" not in df.columns:
        df = df.copy()
        df["source_file"] = "all"

    base_feature_cols = [c for c in PREFERRED_FEATURE_COLS if c in df.columns]
    if not {"ac_dc_ratio", "peak_to_peak"}.issubset(set(base_feature_cols)):
        raise ValueError("CSV must include 'ac_dc_ratio' and 'peak_to_peak'.")
    if "ir_raw" not in base_feature_cols and "ir_raw" in df.columns:
        base_feature_cols.insert(0, "ir_raw")

    print(f"[TRAIN] Feature channels: {base_feature_cols}")

    for source_file, g in df.groupby("source_file"):
        g = g.reset_index(drop=True)

        # Startup trim: remove first N rows (sensor not yet stable)
        if len(g) > STARTUP_TRIM_ROWS:
            g = g.iloc[STARTUP_TRIM_ROWS:].reset_index(drop=True)
        else:
            print(f"[WARN] {source_file}: only {len(g)} rows, skipping trim.")

        for col in base_feature_cols:
            if col not in g.columns:
                g[col] = np.nan

        for start in range(0, len(g) - window_size + 1, step):
            end = start + window_size
            w   = g.iloc[start:end]

            window_labels = w[LABEL_COL].astype(str).values
            unique, counts = np.unique(window_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]

            feat = {
                "source_file":  source_file,
                "window_start": int(start),
                "window_end":   int(end),
            }

            for col in base_feature_cols:
                series = pd.to_numeric(w[col], errors="coerce").fillna(0).values
                feat.update(_safe_stats(series, col))

            # FIX: Read ac_dc_ratio and peak_to_peak directly from the raw CSV
            # columns rather than computing a single scalar and broadcasting it
            # with np.full_like(). The old approach made all window stats (std,
            # min, max, etc.) identical since the scalar was constant across the
            # window, rendering those features useless. Using the actual
            # per-sample values lets the model see real within-window variation.
            acdc = pd.to_numeric(w["ac_dc_ratio"], errors="coerce").fillna(0).values
            p2p  = pd.to_numeric(w["peak_to_peak"], errors="coerce").fillna(0).values
            ir   = (pd.to_numeric(w["ir_raw"], errors="coerce").fillna(0).values
                    if "ir_raw" in w.columns else np.zeros(len(w), dtype=float))

            feat["nonzero_ir_fraction"] = float(np.mean(ir > 0))
            feat["acdc_over_p2p_mean"]  = float(np.mean(acdc / (p2p + 1e-6)))
            feat["p2p_over_ir_mean"]    = float(np.mean(p2p / (np.abs(ir) + 1e-6)))

            if "hr" in w.columns:
                hr = pd.to_numeric(w["hr"], errors="coerce").fillna(0).values
                feat["nonzero_hr_fraction"] = float(np.mean(hr > 0))
                feat["hr_over_ir_mean"]     = float(np.mean(hr / (np.abs(ir) + 1e-6)))

            rows.append(feat)
            labels.append(majority_label)

    if not rows:
        raise ValueError(
            f"No windows built. Need at least {window_size} rows per file."
        )

    X_df = pd.DataFrame(rows)
    y    = np.array(labels)

    print(f"\n[TRAIN] Built {len(X_df)} windows (step={step}, "
          f"window={window_size}).")
    print("[TRAIN] Window label counts:")
    print(pd.Series(y).value_counts().to_string())
    return X_df, y


# ==============================================================================
# Main
# ==============================================================================
def main():
    df   = load_dataset()
    X_df, y = build_window_table(df)

    feature_cols = [c for c in X_df.columns
                    if c not in {"source_file", "window_start", "window_end"}]
    X = X_df[feature_cols].values

    label_names = sorted(pd.Series(y).astype(str).unique().tolist())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    id_to_label = {idx: name for name, idx in label_to_id.items()}
    y_enc       = np.array([label_to_id[v] for v in y], dtype=np.int32)

    with open("models/contact_label_map.json", "w") as f:
        json.dump(id_to_label, f, indent=2)
    print(f"\n[TRAIN] Classes: {id_to_label}")

    # FIX: Warn when any class has too few windows for reliable cross-validation.
    # With 5 folds and a minority class of <10 windows, some folds may contain
    # zero examples of that class, producing misleadingly high CV scores.
    min_class_count = pd.Series(y).value_counts().min()
    min_class_name  = pd.Series(y).value_counts().idxmin()
    if min_class_count < 10:
        print(f"\n[WARN] Smallest class '{min_class_name}' has only "
              f"{min_class_count} windows.")
        print("[WARN] CV scores may be unreliable. Collect more data for this class.")
    elif min_class_count < 20:
        print(f"\n[NOTE] Smallest class '{min_class_name}' has {min_class_count} windows "
              f"— aim for 20+ per class for robust CV.")

    # ── 5-fold cross-validation ──────────────────────────────────────────────
    cv_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_split=4, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])
    print("\n[CV] Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        cv_model, X, y_enc,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    print(f"[CV] Fold accuracies: {np.round(cv_scores, 4)}")
    print(f"[CV] Mean: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")
    if cv_scores.mean() >= 0.90:
        print("[CV] ✓ Model genuinely generalises well.")
    elif cv_scores.mean() >= 0.75:
        print("[CV] ⚠ Moderate generalisation — collect more data per class.")
    else:
        print("[CV] ✗ Poor generalisation — more diverse data needed.")

    # ── Final train/val split ────────────────────────────────────────────────
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_df[feature_cols], y_enc,
            test_size=0.2, random_state=42, stratify=y_enc,
        )
    except ValueError:
        print("[WARN] Stratified split failed; using random split.")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_df[feature_cols], y_enc,
            test_size=0.2, random_state=42,
        )

    final_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_split=4, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])
    print("\n[TRAIN] Training final model on 80% split...")
    final_model.fit(X_tr, y_tr)

    y_pred  = final_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    cm      = confusion_matrix(y_val, y_pred).tolist()

    print(f"\n[EVAL] Validation accuracy (single split): {val_acc:.4f}")
    print("[EVAL] Classification report:")
    print(classification_report(
        y_val, y_pred,
        target_names=[id_to_label[i] for i in sorted(id_to_label)],
        zero_division=0,
    ))
    print("[EVAL] Confusion matrix:")
    print(np.array(cm))

    # Save bundle (Pipeline includes imputer — no separate imputer needed)
    bundle = {
        "model":               final_model,
        "feature_cols":        feature_cols,
        "window_size":         PPG_WINDOW,
        "label_map":           id_to_label,
        "label_normalization": LABEL_NORMALIZATION,
    }
    joblib.dump(bundle, "models/contact_model.pkl")

    metrics = {
        "validation_accuracy":  float(val_acc),
        "cv_mean_accuracy":     float(cv_scores.mean()),
        "cv_std_accuracy":      float(cv_scores.std()),
        "cv_fold_accuracies":   cv_scores.tolist(),
        "n_windows":            int(len(X_df)),
        "n_train":              int(len(X_tr)),
        "n_val":                int(len(X_val)),
        "feature_count":        int(len(feature_cols)),
        "labels":               id_to_label,
        "startup_trim_rows":    STARTUP_TRIM_ROWS,
        "confusion_matrix":     cm,
        "used_channels":        [c for c in PREFERRED_FEATURE_COLS
                                 if c in df.columns],
        "min_class_windows":    int(min_class_count),
        "min_class_name":       str(min_class_name),
    }
    with open("models/contact_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[EXPORT] Saved:")
    print("  models/contact_model.pkl")
    print("  models/contact_label_map.json")
    print("  models/contact_metrics.json")
    print(f"\n[SUMMARY] CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"[SUMMARY] Single-split: {val_acc:.4f}")
    print(f"[SUMMARY] Total windows: {len(X_df)} "
          f"(target: 300+ for strong generalisation)")
    print("[TRAIN] Done.")


if __name__ == "__main__":
    main()