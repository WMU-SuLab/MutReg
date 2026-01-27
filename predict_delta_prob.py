#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------
# Reproducibility
# -------------------------
torch.manual_seed(12)
np.random.seed(12)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------
# Model definition (MUST match training)
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=(128, 64), output_dim=2,
                 activation=nn.ReLU, dropout_p=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Utils
# -------------------------
def load_labels_from_bed(bed_path: str) -> np.ndarray:
    """Assume label is the LAST column of bed/tsv."""
    df = pd.read_csv(bed_path, sep="\t", header=None)
    y = df.iloc[:, -1].astype(int).to_numpy()
    return y


@torch.no_grad()
def predict_prob_class1(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    for (xb,) in dl:
        xb = xb.to(device)
        logits = model(xb)
        p1 = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p1.detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.array([])


def build_key_df(bed_path: str) -> pd.DataFrame:
    """
    Read bed file (no header) and return the whole dataframe.
    We'll build matching keys for pre/post comparison.
    """
    df = pd.read_csv(bed_path, sep="\t", header=None)
    return df


def make_mutation_key(df: pd.DataFrame) -> pd.Series:
    """
    Create a unique key per mutation row to align pre/post rows.
    Based on your example bed columns:
      0 chr
      1 pos
      2 pos
      3 ref
      4 alt
      5 sampleID (e.g. DO52565)
      6 elem_chr
      7 elem_start
      8 elem_end
      9 elem_id (e.g. silencer543)
      10 sequence
      11 label (last col)

    We'll use (0,1,3,4,5,6,7,8,9) as key.
    If your bed schema differs, adjust the indices here.
    """
    key_cols = [0, 1, 3, 4, 5, 6, 7, 8, 9]
    for c in key_cols:
        if c >= df.shape[1]:
            raise ValueError(f"Bed file has only {df.shape[1]} cols, but key needs col {c}. Please adjust key cols.")
    return df[key_cols].astype(str).agg("|".join, axis=1)


def element_group_cols(df: pd.DataFrame):
    """
    Define what counts as the same 'element'.
    Default: use element id in col 9 plus its coordinates (6,7,8).
    """
    needed = [6, 7, 8, 9]
    for c in needed:
        if c >= df.shape[1]:
            raise ValueError(f"Bed file has only {df.shape[1]} cols, but element grouping needs col {c}.")
    return needed


def load_or_fit_scaler(pre_X, post_X, scaler_path=None):
    """
    Use training scaler if provided; else fit scaler on concatenated (pre+post)
    to keep them on the same scale.
    """
    if scaler_path and os.path.isfile(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler: {scaler_path}")
        pre_Xs = scaler.transform(pre_X)
        post_Xs = scaler.transform(post_X)
        return pre_Xs, post_Xs, scaler

    print("No scaler provided. Fitting scaler on concatenated pre+post features (recommended vs fitting separately).")
    scaler = StandardScaler()
    all_X = np.vstack([pre_X, post_X])
    scaler.fit(all_X)
    pre_Xs = scaler.transform(pre_X)
    post_Xs = scaler.transform(post_X)
    return pre_Xs, post_Xs, scaler


# -------------------------
# Main pipeline
# -------------------------
def run_delta_pipeline(
    model_path: str,
    pre_bed: str,
    post_bed: str,
    pre_feat: str,
    post_feat: str,
    output_dir: str,
    hidden_layers=(128, 64),
    dropout_p=0.3,
    batch_size=256,
    scaler_path=None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    pre_df = build_key_df(pre_bed)
    post_df = build_key_df(post_bed)

    X_pre = np.load(pre_feat)
    X_post = np.load(post_feat)

    if X_pre.shape[1] != X_post.shape[1]:
        raise ValueError(f"Feature dims mismatch: pre={X_pre.shape}, post={X_post.shape}")

    # Scale consistently
    X_pre, X_post, _scaler = load_or_fit_scaler(X_pre, X_post, scaler_path=scaler_path)

    # Build model and load weights
    model = MLP(input_dim=X_pre.shape[1], hidden_layers=hidden_layers, dropout_p=dropout_p).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Predict
    prob_pre = predict_prob_class1(model, X_pre, batch_size=batch_size)
    prob_post = predict_prob_class1(model, X_post, batch_size=batch_size)

    if len(prob_pre) != len(pre_df) or len(prob_post) != len(post_df):
        raise ValueError(
            f"Row count mismatch:\n"
            f"  pre_df rows={len(pre_df)}, prob_pre={len(prob_pre)}\n"
            f"  post_df rows={len(post_df)}, prob_post={len(prob_post)}\n"
            f"Check that bed rows align with feature rows."
        )

    # Add probabilities
    pre_df = pre_df.copy()
    post_df = post_df.copy()
    pre_df["prob_pre"] = prob_pre
    post_df["prob_post"] = prob_post

    # Align pre/post by mutation key
    pre_df["_key"] = make_mutation_key(pre_df)
    post_df["_key"] = make_mutation_key(post_df)

    merged = pd.merge(
        pre_df,
        post_df[["_key", "prob_post"]],
        on="_key",
        how="inner",
        validate="one_to_one"
    )

    if merged.empty:
        raise ValueError("No matched mutations between pre and post. Check bed schemas / key columns / row consistency.")

    # Compute abs delta
    merged["abs_delta_prob1"] = (merged["prob_post"] - merged["prob_pre"]).abs()

    # Per-element mean
    grp_cols_idx = element_group_cols(merged)  # [6,7,8,9] by default
    # Give friendly names (optional)
    col_map = {0: "chr", 1: "pos", 3: "ref", 4: "alt", 5: "sample",
               6: "elem_chr", 7: "elem_start", 8: "elem_end", 9: "elem_id"}
    for k, v in col_map.items():
        if k in merged.columns:
            merged.rename(columns={k: v}, inplace=True)

    # After rename, group column names become:
    grp_cols = [col_map.get(i, i) for i in grp_cols_idx]

    element_mean = (
        merged.groupby(grp_cols, as_index=False)["abs_delta_prob1"]
        .mean()
        .rename(columns={"abs_delta_prob1": "mean_abs_delta_prob1"})
        .sort_values("mean_abs_delta_prob1", ascending=False)
    )

    # Save outputs
    per_mutation_out = os.path.join(output_dir, "mutation_abs_delta.tsv")
    per_element_out = os.path.join(output_dir, "element_mean_abs_delta.tsv")

    # Keep some useful columns in per-mutation output
    keep_cols = []
    for c in ["chr", "pos", "ref", "alt", "sample", "elem_chr", "elem_start", "elem_end", "elem_id",
              "prob_pre", "prob_post", "abs_delta_prob1"]:
        if c in merged.columns:
            keep_cols.append(c)
    merged[keep_cols].to_csv(per_mutation_out, sep="\t", index=False)
    element_mean.to_csv(per_element_out, sep="\t", index=False)

    print(f"✅ Done.")
    print(f"Per-mutation abs delta saved to: {per_mutation_out}")
    print(f"Per-element mean abs delta saved to: {per_element_out}")
    print(f"Matched mutations: {len(merged)}; Elements: {len(element_mean)}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Predict pre/post mutation probabilities, compute abs delta, then mean per element."
    )
    ap.add_argument("--model_path", required=True, help="Trained model .pth (state_dict)")
    ap.add_argument("--mutation1", required=True, help="Pre-mutation bed/tsv (label in last col)")
    ap.add_argument("--mutation2", required=True, help="Post-mutation bed/tsv (label in last col)")
    ap.add_argument("--sequence_signature1", required=True, help="Pre-mutation features .npy")
    ap.add_argument("--sequence_signature2", required=True, help="Post-mutation features .npy")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--dropout_p", type=float, default=0.3)
    ap.add_argument("--hidden_layers", type=str, default="128,64", help="e.g. '128,64'")
    ap.add_argument("--scaler", default=None, help="Optional training StandardScaler.pkl (joblib)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hidden_layers = tuple(int(x.strip()) for x in args.hidden_layers.split(",") if x.strip())

    run_delta_pipeline(
        model_path=args.model_path,
        mutation1=args.mutation1,
        mutation2=args.mutation2,
        sequence_signature1=args.sequence_signature1,
        sequence_signature2=args.sequence_signature2,
        output=args.output,
        hidden_layers=hidden_layers,
        dropout_p=args.dropout_p,
        batch_size=args.batch_size,
        scaler_path=args.scaler,
    )
