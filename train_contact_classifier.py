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
from sklearn.model_selection import GroupKFold

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
    "red_raw",
    "ac_dc_ratio",
    "peak_to_peak",
    "t_room",
    "hr",
    "hr_std_5s",
    "spo2_est",
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
    step = window_size
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
            red  = (pd.to_numeric(w["red_raw"], errors="coerce").fillna(0).values
                    if "red_raw" in w.columns else np.zeros(len(w), dtype=float))

            feat["nonzero_ir_fraction"] = float(np.mean(ir > 0))
            feat["acdc_over_p2p_mean"]  = float(np.mean(acdc / (p2p + 1e-6)))
            feat["p2p_over_ir_mean"]    = float(np.mean(p2p / (np.abs(ir) + 1e-6)))

            feat["red_ir_ratio_mean"] = float(np.mean(red / (ir + 1e-6)))
            feat["red_ir_ratio_std"]  = float(np.std(red / (ir + 1e-6)))
            feat["red_nonzero_fraction"] = float(np.mean(red > 0))

            if "hr" in w.columns:
                hr = pd.to_numeric(w["hr"], errors="coerce").fillna(0).values
                feat["nonzero_hr_fraction"] = float(np.mean(hr > 0))
                feat["hr_over_ir_mean"]     = float(np.mean(hr / (np.abs(ir) + 1e-6)))

            if "spo2_est" in w.columns:
                spo2 = pd.to_numeric(w["spo2_est"], errors="coerce").fillna(0).values
                feat["nonzero_spo2_fraction"]  = float(np.mean(spo2 > 0))
                feat["valid_spo2_fraction"]    = float(np.mean((spo2 >= 70) & (spo2 <= 100)))
                feat["spo2_below_90_fraction"] = float(np.mean(spo2 < 90))

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
# Cross-file Validation
# ==============================================================================
def run_grouped_cross_file_validation(X_df: pd.DataFrame, y: np.ndarray, n_splits: int = 5):
    feature_cols = [c for c in X_df.columns if c not in {"source_file", "window_start", "window_end"}]
    X = X_df[feature_cols].values
    groups = X_df["source_file"].values

    label_names = sorted(pd.Series(y).astype(str).unique().tolist())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    id_to_label = {idx: name for name, idx in label_to_id.items()}
    y_enc = np.array([label_to_id[v] for v in y], dtype=np.int32)

    gkf = GroupKFold(n_splits=n_splits)

    fold_accuracies = []
    all_true = []
    all_pred = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_enc, groups=groups), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]

        train_files = sorted(set(groups[train_idx]))
        val_files = sorted(set(groups[val_idx]))

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )),
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)

        all_true.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())

        print(f"\n[FOLD {fold}] Accuracy: {acc:.4f}")
        print(f"[FOLD {fold}] Train files ({len(train_files)}): {train_files}")
        print(f"[FOLD {fold}] Val files ({len(val_files)}): {val_files}")

    print("\n[GROUPED CV] Fold accuracies:")
    for i, acc in enumerate(fold_accuracies, start=1):
        print(f"  Fold {i}: {acc:.4f}")

    print(f"\n[GROUPED CV] Mean accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"[GROUPED CV] Std accuracy:  {np.std(fold_accuracies):.4f}")

    print("\n[GROUPED CV] Overall classification report:")
    print(classification_report(
        all_true,
        all_pred,
        target_names=[id_to_label[i] for i in sorted(id_to_label)],
        zero_division=0,
    ))

    print("[GROUPED CV] Overall confusion matrix:")
    print(confusion_matrix(all_true, all_pred))

    return fold_accuracies, feature_cols
    
# ==============================================================================
# Main
# ==============================================================================
def main():
    # --------------------------------------------------------------------------
    # Load ALL Stage 1 data together for grouped cross-file validation
    # --------------------------------------------------------------------------
    df_all = load_dataset("training_data_stage1_all")

    # Build window-level dataset
    X_df, y = build_window_table(df_all)

    print("\n[TRAIN] Running grouped cross-file validation...")
    fold_accuracies, feature_cols = run_grouped_cross_file_validation(
        X_df, y, n_splits=5
    )

    # --------------------------------------------------------------------------
    # Train final model on ALL available Stage 1 data after CV
    # --------------------------------------------------------------------------
    feature_cols = [c for c in X_df.columns
                    if c not in {"source_file", "window_start", "window_end"}]

    X_all = X_df[feature_cols].values

    # Label encoding based on all labels
    label_names = sorted(pd.Series(y).astype(str).unique().tolist())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    id_to_label = {idx: name for name, idx in label_to_id.items()}
    y_all_enc = np.array([label_to_id[v] for v in y], dtype=np.int32)

    with open("models/contact_label_map.json", "w") as f:
        json.dump(id_to_label, f, indent=2)

    print(f"\n[TRAIN] Classes: {id_to_label}")

    final_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])

    print("\n[TRAIN] Training final model on ALL files...")
    final_model.fit(X_all, y_all_enc)

    clf = final_model.named_steps["clf"]
    importances = clf.feature_importances_

    pairs = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop feature importances:")
    for name, score in pairs[:15]:
        print(f"{name}: {score:.4f}")

    bundle = {
        "model": final_model,
        "feature_cols": feature_cols,
        "window_size": PPG_WINDOW,
        "label_map": id_to_label,
        "label_normalization": LABEL_NORMALIZATION,
    }
    joblib.dump(bundle, "models/contact_model.pkl")

    metrics = {
        "grouped_cv_fold_accuracies": [float(a) for a in fold_accuracies],
        "grouped_cv_mean_accuracy": float(np.mean(fold_accuracies)),
        "grouped_cv_std_accuracy": float(np.std(fold_accuracies)),
        "n_total_windows": int(len(X_df)),
        "feature_count": int(len(feature_cols)),
        "labels": id_to_label,
        "startup_trim_rows": STARTUP_TRIM_ROWS,
        "used_channels": [c for c in PREFERRED_FEATURE_COLS if c in df_all.columns],
    }
    with open("models/contact_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[EXPORT] Saved:")
    print("  models/contact_model.pkl")
    print("  models/contact_label_map.json")
    print("  models/contact_metrics.json")
    print(f"\n[SUMMARY] Total windows: {len(X_df)}")
    print(f"[SUMMARY] Grouped CV mean accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"[SUMMARY] Grouped CV std accuracy: {np.std(fold_accuracies):.4f}")
    print("[TRAIN] Done.")


if __name__ == "__main__":
    main()
