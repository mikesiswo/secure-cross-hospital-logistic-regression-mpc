#!/usr/bin/env python3

"""
Preprocessing pipeline for MPC-based cross-hospital anomaly detection.

The script prepares the UCI Myocardial Infarction Complications dataset for
secure logistic regression in MP-SPDZ. It selects clinical and procedure-related
features, constructs distance-based anomaly labels, horizontally splits the data
between two parties, writes MP-SPDZ input files, and trains a plaintext baseline.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "dataset_mpc/myocardial_infarction.csv"
RANDOM_SEED = 42
ROWS_PER_PARTY = 800
ANOMALY_PERCENTILE = 0.20

OUT_DIR = Path("Player-Data")
OUT_P0 = OUT_DIR / "Input-P0-0"
OUT_P1 = OUT_DIR / "Input-P1-0"

covariates = [
    "AGE", "SEX", "INF_ANAM", "STENOK_AN", "FK_STENOK",
    "IBS_POST", "DLIT_AG", "R_AB_1_n", "R_AB_2_n"
]

emergency_interventions = [
    "NITR_S", "GEPAR_S_n", "ASP_S_n", "B_BLOK_S_n",
    "ANT_CA_S_n", "LID_S_n", "TIKL_S_n", "TRENT_S_n"
]

time_indexed = [
    "NA_R_1_n", "NA_R_2_n", "NA_R_3_n",
    "NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n"
]

procedure_like = [
    "fibr_ter_01", "fibr_ter_02", "fibr_ter_03",
    "fibr_ter_05", "fibr_ter_06", "fibr_ter_07", "fibr_ter_08"
]

high_missing = [
    "R_AB_3_n", "NA_KB", "NOT_NA_KB", "LID_KB"
]

procedure_features = (
    emergency_interventions + time_indexed + procedure_like + high_missing
)

all_features = covariates + procedure_features
TARGET_COLUMN = "ANOMALY_FLAG"

print(
    f"Features: {len(covariates)} covariates + "
    f"{len(procedure_features)} procedure = {len(all_features)} total"
)

df_full = pd.read_csv(INPUT_FILE)
print(f"Original shape: {df_full.shape}")

df_full = df_full[all_features].copy()
df_full = df_full.apply(pd.to_numeric, errors="coerce")

for col in df_full.columns:
    df_full[col] = df_full[col].fillna(df_full[col].mean())

df_full = np.floor(df_full + 0.5).astype(int)

required_rows = 2 * ROWS_PER_PARTY
df_subset = df_full.sample(
    n=required_rows,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# Anomaly labels are based only on procedure-related features.
mu_proc = df_subset[procedure_features].mean()
dist_scores = ((df_subset[procedure_features] - mu_proc) ** 2).sum(axis=1)
dist_scores = dist_scores / dist_scores.mean()

threshold = dist_scores.quantile(1 - ANOMALY_PERCENTILE)
df_subset[TARGET_COLUMN] = (dist_scores >= threshold).astype(int)

n_pos = df_subset[TARGET_COLUMN].sum()
n_tot = len(df_subset)

print(
    f"\nAnomaly labeling "
    f"(procedure-feature distance, top {ANOMALY_PERCENTILE * 100:.0f}%):"
)
print(f"  Threshold : {threshold:.4f}")
print(f"  Positives : {n_pos} / {n_tot} ({n_pos / n_tot * 100:.1f}%)")
print(f"  Negatives : {n_tot - n_pos} / {n_tot} ({(n_tot - n_pos) / n_tot * 100:.1f}%)")

df_p0 = df_subset.iloc[:ROWS_PER_PARTY].copy()
df_p1 = df_subset.iloc[ROWS_PER_PARTY:].copy()

print(
    f"\nParty 0 (Hospital A): {len(df_p0)} samples, "
    f"{df_p0[TARGET_COLUMN].sum()} positives"
)
print(
    f"Party 1 (Hospital B): {len(df_p1)} samples, "
    f"{df_p1[TARGET_COLUMN].sum()} positives"
)

# Standardization improves numerical stability under MP-SPDZ fixed-point arithmetic.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_subset[all_features].to_numpy(dtype=float))

X0 = X_scaled[:ROWS_PER_PARTY]
X1 = X_scaled[ROWS_PER_PARTY:]
y0 = df_p0[TARGET_COLUMN].to_numpy(dtype=int)
y1 = df_p1[TARGET_COLUMN].to_numpy(dtype=int)

OUT_DIR.mkdir(exist_ok=True)


def write_inputs(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """Write features followed by the label for each sample."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write(f"{X[i, j]:.6f}\n")
            f.write(f"{int(y[i])}\n")


write_inputs(OUT_P0, X0, y0)
write_inputs(OUT_P1, X1, y1)

print(f"\nWrote: {OUT_P0}  ({ROWS_PER_PARTY} samples)")
print(f"Wrote: {OUT_P1}  ({ROWS_PER_PARTY} samples)")

print("\nMPC constants:")
print(f"  n_features   = {len(all_features)}")
print(f"  n_samples_p0 = {ROWS_PER_PARTY}")
print(f"  n_samples_p1 = {ROWS_PER_PARTY}")

X_plain = np.vstack([X0, X1])
y_plain = np.concatenate([y0, y1])

clf = LogisticRegression(
    max_iter=1000,
    fit_intercept=True,
    class_weight=None,
    solver="lbfgs",
    random_state=RANDOM_SEED
)

t0 = time.perf_counter()
clf.fit(X_plain, y_plain)
train_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
y_pred = clf.predict(X_plain)
infer_ms = (time.perf_counter() - t0) * 1000

y_prob = clf.predict_proba(X_plain)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_plain, y_pred).ravel()

print("\n" + "=" * 60)
print("PLAINTEXT BASELINE")
print("=" * 60)
print(
    classification_report(
        y_plain,
        y_pred,
        target_names=["Compliant (0)", "Noncompliant (1)"]
    )
)
print(f"AUC-ROC : {roc_auc_score(y_plain, y_prob):.4f}")

print("\nConfusion Matrix:")
print(f"  TP = {tp}")
print(f"  TN = {tn}")
print(f"  FP = {fp}")
print(f"  FN = {fn}")

raw_mb = (X_plain.nbytes + y_plain.nbytes) / 1e6

print("\n" + "=" * 60)
print("OVERHEAD REFERENCE")
print("=" * 60)
print(f"  Plaintext train time : {train_ms:.2f} ms")
print(f"  Plaintext infer time : {infer_ms:.2f} ms")
print(f"  Raw data size        : {raw_mb:.4f} MB")
print(f"  overhead_factor      = mpc_seconds / {train_ms / 1000:.4f}")
print(f"  comm_ratio           = MPC_data_sent_MB / {raw_mb:.4f}")

print("\n" + "=" * 60)
print("MODEL WEIGHTS")
print("=" * 60)
print(f"  Bias : {clf.intercept_[0]:.6f}")

for j, feat in enumerate(all_features):
    print(f"  w[{j:02d}] {feat:<20s}: {clf.coef_[0][j]:.6f}")

