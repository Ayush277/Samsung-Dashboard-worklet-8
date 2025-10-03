#!/usr/bin/env python3
"""
Generate static charts/images for README from available CSV datasets.
Outputs are saved under docs/assets/ and referenced by README.
"""
import os
import sys
import math
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 5)


def savefig(path: Path, fig=None, dpi=130):
    (fig or plt).savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved: {path.relative_to(ROOT)}")


def gen_case2_charts():
    base = ROOT / "Case2-Nosalesuplift(pipeline) 2"
    csv = base / "train2.csv"
    if not csv.exists():
        print("[Case2] train2.csv not found, skipping charts")
        return
    df = pd.read_csv(csv, parse_dates=["date"]) if "date" in pd.read_csv(csv, nrows=1).columns else pd.read_csv(csv)
    if "date" not in df.columns or "sales" not in df.columns:
        print("[Case2] train2.csv missing date/sales, skipping charts")
        return
    # Daily mean sales
    daily = df.groupby("date", as_index=False)["sales"].mean()
    fig, ax = plt.subplots()
    ax.plot(daily["date"], daily["sales"], color="#1f77b4")
    ax.set_title("Case2 — Overall Daily Mean Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    savefig(ASSETS / "case2_daily_mean.png", fig)

    # Boxplot by month
    df["month"] = df["date"].dt.month
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="month", y="sales", ax=ax, palette="magma")
    ax.set_title("Case2 — Sales by Month")
    savefig(ASSETS / "case2_month_box.png", fig)


def gen_rossmann_charts():
    # Prefer pipeline/train.csv
    base = ROOT / "Sales Uplift 2" / "pipeline"
    csv = base / "train.csv"
    if not csv.exists():
        # fallback dataset
        alt = ROOT / "Sales Uplift 2" / "Dataset" / "train.csv"
        csv = alt if alt.exists() else csv
    if not csv.exists():
        print("[Rossmann] train.csv not found, skipping charts")
        return
    df = pd.read_csv(csv, parse_dates=["Date"]) if "Date" in pd.read_csv(csv, nrows=1).columns else pd.read_csv(csv)
    if "Date" not in df.columns or "Sales" not in df.columns:
        print("[Rossmann] train.csv missing Date/Sales, skipping charts")
        return
    # Daily mean sales
    daily = df.groupby("Date", as_index=False)["Sales"].mean()
    fig, ax = plt.subplots()
    ax.plot(daily["Date"], daily["Sales"], color="#2ca02c")
    ax.set_title("Rossmann — Daily Mean Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    savefig(ASSETS / "rossmann_daily_mean.png", fig)

    # Heatmap: dayofweek x month average sales
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    # DayOfWeek may exist; otherwise derive
    if "DayOfWeek" in df.columns:
        df["DOW"] = df["DayOfWeek"]
    else:
        df["DOW"] = pd.to_datetime(df["Date"]).dt.dayofweek + 1  # 1..7
    piv = df.pivot_table(index="DOW", columns="Month", values="Sales", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(piv, cmap="YlGnBu", ax=ax)
    ax.set_title("Rossmann — Avg Sales Heatmap (DOW × Month)")
    savefig(ASSETS / "rossmann_heatmap.png", fig)


def gen_loan_charts():
    base = ROOT / "loan_app"
    csv = base / "approach_train.csv"
    if not csv.exists():
        print("[Loan] approach_train.csv not found, skipping charts")
        return
    df = pd.read_csv(csv)
    # Try common label names
    label = None
    for c in ["default", "label", "target", "y", "is_default", "loan_status"]:
        if c in df.columns:
            label = c; break
    # Class distribution (if label exists)
    if label is not None:
        vc = df[label].value_counts().sort_index()
        fig, ax = plt.subplots()
        vc.plot(kind="bar", color=["#4CAF50", "#E74C3C"], ax=ax)
        ax.set_title(f"Loan — Class Distribution ({label})")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        savefig(ASSETS / "loan_class_dist.png", fig)
    else:
        print("[Loan] no label column found, skipping class dist")

    # Correlation heatmap for numeric columns
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Loan — Numeric Feature Correlation")
        savefig(ASSETS / "loan_corr.png", fig)
    else:
        print("[Loan] insufficient numeric columns for correlation plot")


def main():
    gen_case2_charts()
    gen_rossmann_charts()
    gen_loan_charts()
    print("Done.")

if __name__ == "__main__":
    main()
