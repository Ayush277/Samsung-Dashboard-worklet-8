"""Convert the trained .pkl artifacts into ONNX and prove the conversion is faithful.

Why this exists
---------------
The trained models live in three separate projects and depend on scikit-learn,
CatBoost, LightGBM and XGBoost. Installed together those libraries weigh ~330 MB,
and Vercel's serverless bundle limit is 250 MB — which is why the ML was never
actually deployed. ONNX Runtime executes all four model families from a single
~16 MB wheel, so the deployed app can run the real models instead of mocking them.

A conversion is only worth anything if the exported graph computes the same
function as the model that was trained. Every model here is therefore replayed
against its original implementation on real rows from the project's own data, and
the run FAILS if any output drifts beyond TOLERANCE. A silently-wrong export would
be worse than no deployment at all: the site would look live and serve numbers
that no one ever validated.

This runs in CI (see .github/workflows/build-models.yml), never on a laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

# Exports must match the original to this many decimal places. Regressors here
# predict sales in the hundreds, so 1e-3 is far tighter than it looks: it is
# fractions of a cent on a unit-sales prediction.
TOLERANCE = 1e-3
OPSET = 13

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOAN = os.path.join(REPO, "Loan delinquency risk")
CAMPAIGN = os.path.join(REPO, "Campaign performance (marketing)")
SELLOUT = os.path.join(REPO, "Sell-out performance forecasting (sales uplift)", "pipeline")
OUT_DIR = os.path.join(REPO, "prism", "models")

failures: list[str] = []
manifest: dict[str, dict] = {}


def _load(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _onnx_run(path: str, X: np.ndarray, output_index: int = 0) -> np.ndarray:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.asarray(sess.run(None, {name: X.astype(np.float32)})[output_index])


def verify(key: str, path: str, expected: np.ndarray, X: np.ndarray,
           output_index: int = 0) -> None:
    """Replay the exported graph against the original model's own output.

    Both sides are flattened, so a classifier's full (n, 2) probability matrix is
    compared element-wise against predict_proba — every class, not just one
    column.
    """
    got = _onnx_run(path, X, output_index).ravel()
    exp = np.asarray(expected).ravel()
    if got.shape != exp.shape:
        failures.append(f"{key}: shape {got.shape} != original {exp.shape}")
        print(f"  FAIL {key:26} shape {got.shape} vs {exp.shape}")
        return

    diff = float(np.abs(got - exp).max())
    size_mb = os.path.getsize(path) / 1e6
    ok = diff <= TOLERANCE
    if not ok:
        failures.append(f"{key}: max drift {diff:.3e} exceeds {TOLERANCE:.0e}")
    print(f"  {'OK  ' if ok else 'FAIL'} {key:26} drift {diff:9.2e}   {size_mb:6.2f} MB   "
          f"({len(exp)} rows)")
    manifest[key] = {
        "file": os.path.relpath(path, REPO),
        "max_drift_vs_original": float(diff),
        "size_mb": round(size_mb, 3),
        "verified_rows": int(len(exp)),
    }


def export_sklearn(model, X_sample: np.ndarray, dest: str, options: dict | None = None):
    onx = to_onnx(model, X_sample[:1].astype(np.float32), target_opset=OPSET,
                  options=options)
    with open(dest, "wb") as fh:
        fh.write(onx.SerializeToString())
    return dest


# --------------------------------------------------------------------------
# Loan delinquency — StandardScaler(30) + RandomForestClassifier
# --------------------------------------------------------------------------
def convert_loan() -> None:
    print("\nLOAN delinquency risk")
    import joblib

    # NB: the artifact is named tabpfn.pkl for historical reasons but is a
    # RandomForestClassifier — see docs/MODEL_CARDS.md. Named honestly on export.
    model = joblib.load(os.path.join(LOAN, "models", "tabpfn.pkl"))
    scaler = joblib.load(os.path.join(LOAN, "models", "scaler.pkl"))
    columns = joblib.load(os.path.join(LOAN, "models", "dummy_columns.pkl"))
    print(f"  source: {type(model).__name__} "
          f"({getattr(model, 'n_estimators', '?')} trees, "
          f"depth {getattr(model, 'max_depth', '?')}), {len(columns)} features")

    # Real rows, not synthetic noise: replay the actual training distribution.
    df = pd.read_csv(os.path.join(LOAN, "approach_train.csv"), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    numeric = [c for c in columns if c in df.columns]
    frame = pd.DataFrame(0.0, index=range(min(500, len(df))), columns=columns)
    for c in numeric:
        frame[c] = pd.to_numeric(df[c].head(len(frame)), errors="coerce").fillna(0.0).values
    X = frame.values.astype(np.float64)

    Xs = scaler.transform(X)
    export_sklearn(scaler, X, os.path.join(OUT_DIR, "loan_scaler.onnx"))
    verify("loan_scaler", os.path.join(OUT_DIR, "loan_scaler.onnx"), Xs, X)

    # zipmap=False keeps probabilities as a plain tensor rather than a list of
    # dicts, which ONNX Runtime can return without any Python-side conversion.
    export_sklearn(model, Xs, os.path.join(OUT_DIR, "loan_random_forest.onnx"),
                   options={id(model): {"zipmap": False}})
    # Compare the whole (n, 2) probability matrix, not one column: the service
    # reads P(mx=0) because the target is inverted, so both columns must match.
    verify("loan_random_forest", os.path.join(OUT_DIR, "loan_random_forest.onnx"),
           model.predict_proba(Xs), Xs, output_index=1)

    with open(os.path.join(OUT_DIR, "loan_meta.json"), "w") as fh:
        json.dump({
            "columns": list(columns),
            "medians": json.load(open(os.path.join(LOAN, "models", "medians.json"))),
            "model_type": type(model).__name__,
            "n_estimators": int(getattr(model, "n_estimators", 0)),
            "max_depth": int(getattr(model, "max_depth", 0) or 0),
            "feature_importances": {
                c: float(v) for c, v in
                sorted(zip(columns, model.feature_importances_), key=lambda t: -t[1])
            },
        }, fh, indent=2)


# --------------------------------------------------------------------------
# Campaign — StandardScaler(7) + {Ridge, LightGBM, CatBoost}
# --------------------------------------------------------------------------
def convert_campaign() -> None:
    print("\nCAMPAIGN performance (store/item demand)")
    scaler = _load(os.path.join(CAMPAIGN, "scaler.pkl"))
    feats = list(scaler.feature_names_in_)
    print(f"  features: {feats}")

    test = pd.read_csv(os.path.join(CAMPAIGN, "test.csv"))
    test["date"] = pd.to_datetime(test["date"])
    X = pd.DataFrame({
        "store": test["store"], "item": test["item"],
        "month": test["date"].dt.month, "day": test["date"].dt.day,
        "dayofweek": test["date"].dt.dayofweek,
        "dayofyear": test["date"].dt.dayofyear,
        "weekofyear": test["date"].dt.isocalendar().week.astype(int),
    })[feats].head(500).values.astype(np.float64)
    Xs = scaler.transform(X)

    export_sklearn(scaler, X, os.path.join(OUT_DIR, "campaign_scaler.onnx"))
    verify("campaign_scaler", os.path.join(OUT_DIR, "campaign_scaler.onnx"), Xs, X)

    ridge = _load(os.path.join(CAMPAIGN, "ridge_model.pkl"))
    export_sklearn(ridge, Xs, os.path.join(OUT_DIR, "campaign_ridge.onnx"))
    verify("campaign_ridge", os.path.join(OUT_DIR, "campaign_ridge.onnx"),
           ridge.predict(Xs), Xs)

    from onnxmltools.convert import convert_lightgbm
    lgbm = _load(os.path.join(CAMPAIGN, "lgbm_model.pkl"))
    onx = convert_lightgbm(
        lgbm, initial_types=[("X", FloatTensorType([None, len(feats)]))],
        target_opset=OPSET)
    dest = os.path.join(OUT_DIR, "campaign_lightgbm.onnx")
    with open(dest, "wb") as fh:
        fh.write(onx.SerializeToString())
    verify("campaign_lightgbm", dest, lgbm.predict(Xs), Xs)

    # CatBoost exports ONNX itself; going through onnxmltools loses its
    # internal float handling and drifts.
    cat = _load(os.path.join(CAMPAIGN, "catboost_model.pkl"))
    dest = os.path.join(OUT_DIR, "campaign_catboost.onnx")
    cat.save_model(dest, format="onnx")
    verify("campaign_catboost", dest, cat.predict(Xs), Xs)

    with open(os.path.join(OUT_DIR, "campaign_meta.json"), "w") as fh:
        json.dump({"features": feats,
                   "models": ["ridge", "lightgbm", "catboost"]}, fh, indent=2)

    build_campaign_baselines()


def build_campaign_baselines() -> None:
    """Precompute real per-store/item sales baselines from 913k rows of history.

    Uplift only means anything against a baseline. The original app invented one
    with `store_id % 10` and `item_id % 20` — arbitrary arithmetic on an ID that
    encodes nothing about the store. train2.csv holds five years of actual sales
    per store/item, so the honest baseline is simply what that pair really sold.

    The CSV is 17 MB and .vercelignore excludes *.csv, so the aggregate is
    computed here in CI and shipped as a ~100 KB JSON instead.
    """
    src = os.path.join(CAMPAIGN, "train2.csv")
    if not os.path.exists(src):
        print("  WARN train2.csv missing — skipping baselines (uplift will be unavailable)")
        return

    hist = pd.read_csv(src, parse_dates=["date"])
    print(f"  history: {len(hist):,} rows, {hist['date'].min().date()} → "
          f"{hist['date'].max().date()}")

    pair = hist.groupby(["store", "item"])["sales"].agg(["mean", "std", "count"])
    monthly = hist.assign(month=hist["date"].dt.month) \
                  .groupby(["store", "item", "month"])["sales"].mean()
    dow = hist.assign(dow=hist["date"].dt.dayofweek) \
              .groupby(["store", "item", "dow"])["sales"].mean()

    baselines = {
        "generated_from": "train2.csv",
        "rows": int(len(hist)),
        "date_range": [str(hist["date"].min().date()), str(hist["date"].max().date())],
        "overall_mean": float(hist["sales"].mean()),
        "pair": {f"{s}|{i}": {"mean": round(float(r["mean"]), 3),
                              "std": round(float(r["std"]), 3),
                              "n": int(r["count"])}
                 for (s, i), r in pair.iterrows()},
        "pair_month": {f"{s}|{i}|{m}": round(float(v), 3)
                       for (s, i, m), v in monthly.items()},
        "pair_dow": {f"{s}|{i}|{d}": round(float(v), 3)
                     for (s, i, d), v in dow.items()},
    }
    dest = os.path.join(OUT_DIR, "campaign_baselines.json")
    with open(dest, "w") as fh:
        json.dump(baselines, fh, separators=(",", ":"))
    print(f"  OK   campaign_baselines        {len(baselines['pair'])} store/item pairs, "
          f"{os.path.getsize(dest)/1e6:.2f} MB")


def _patch_onnxmltools_base_score() -> None:
    """Teach onnxmltools to read XGBoost >= 2.1's array-encoded base_score.

    XGBoost serialises base_score as a JSON array ("[6.752716E1]") to support
    multi-output targets. onnxmltools calls float() straight on that string:

        ValueError: could not convert string to float: '[6.752716E1]'

    Rewriting the booster's own config does not help — base_score lives in
    `learner_model_param`, which is derived from the trained model, so
    `load_config` silently discards the change (the first attempt printed a
    successful rewrite and converted nothing). The read has to be fixed instead.

    Patching a third-party function is not free, so this keeps the original's
    behaviour exactly and only unwraps a one-element array. A genuine
    multi-output score raises rather than being silently flattened, and the
    parity check downstream is what proves the intercept survived.

    Preferable to pinning an older XGBoost: 3.2.0 is the version these pickles
    were verified to load under, and downgrading risks not loading them at all.
    """
    import json as _json

    from onnxmltools.convert.xgboost import common as _common
    from onnxmltools.convert.xgboost.operator_converters import XGBoost as _ops

    def get_xgb_params(xgb_node):
        params = (xgb_node.get_xgb_params() if hasattr(xgb_node, "kwargs")
                  else xgb_node.__dict__)
        booster = xgb_node.get_booster() if hasattr(xgb_node, "get_booster") else xgb_node
        config = _json.loads(booster.save_config())
        raw = config["learner"]["learner_model_param"]["base_score"]
        if isinstance(raw, str) and raw.strip().startswith("["):
            values = _json.loads(raw)
            if len(values) != 1:
                raise ValueError(f"multi-output base_score {values!r} cannot be "
                                 f"flattened to a scalar")
            raw = values[0]
        params = {k: v for k, v in params.items() if v is not None}
        params["base_score"] = float(raw)
        return params

    # XGBoost.py does `from ..common import get_xgb_params`, so it holds its own
    # reference — patching the source module alone would not take effect.
    _common.get_xgb_params = get_xgb_params
    _ops.get_xgb_params = get_xgb_params
    print("  patched onnxmltools to read array-encoded base_score")


# --------------------------------------------------------------------------
# Sell-out — StandardScaler(17) + XGBRegressor (unwrapped from RandomizedSearchCV)
# --------------------------------------------------------------------------
def convert_sellout() -> None:
    print("\nSELL-OUT forecasting (Rossmann)")
    search = _load(os.path.join(SELLOUT, "xgb_model.pkl"))
    # The pickle is the fitted search object, not the estimator.
    model = getattr(search, "best_estimator_", search)
    scaler = _load(os.path.join(SELLOUT, "scaler.pkl"))
    n = int(scaler.n_features_in_)
    print(f"  source: {type(model).__name__}, {n} features")
    if hasattr(search, "best_params_"):
        print(f"  best_params: {search.best_params_}")

    sample_csv = os.path.join(SELLOUT, "sample_sales_data.csv")
    X = None
    if os.path.exists(sample_csv):
        raw = pd.read_csv(sample_csv)
        num = raw.select_dtypes(include=[np.number])
        if num.shape[1] >= n:
            X = num.iloc[:, :n].head(300).values.astype(np.float64)
    if X is None or len(X) == 0:
        # Fall back to the scaler's own fitted statistics, which describe the
        # real training distribution, rather than arbitrary noise.
        centre = np.asarray(scaler.mean_, dtype=np.float64)
        spread = np.sqrt(np.asarray(scaler.var_, dtype=np.float64))
        X = (np.random.default_rng(0).normal(0, 1, (300, n)) * spread + centre)
        print("  (verifying against scaler's fitted distribution)")

    Xs = scaler.transform(X)
    export_sklearn(scaler, X, os.path.join(OUT_DIR, "sellout_scaler.onnx"))
    verify("sellout_scaler", os.path.join(OUT_DIR, "sellout_scaler.onnx"), Xs, X)

    _patch_onnxmltools_base_score()
    from onnxmltools.convert import convert_xgboost
    onx = convert_xgboost(model, initial_types=[("X", FloatTensorType([None, n]))],
                          target_opset=OPSET)
    dest = os.path.join(OUT_DIR, "sellout_xgboost.onnx")
    with open(dest, "wb") as fh:
        fh.write(onx.SerializeToString())
    # Parity here is the only thing that makes the base_score rewrite safe: if the
    # patch corrupted the intercept, predictions shift and the build fails.
    verify("sellout_xgboost", dest, model.predict(Xs.astype(np.float32)), Xs)

    try:
        from config import Config  # noqa
        features = list(Config.EXPECTED_FEATURES)
    except Exception:
        features = [
            "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday",
            "CompetitionDistance", "CompetitionOpenNumMonths", "Promo2NumWeeks",
            "WeekOfYear", "PromoInterval_0", "PromoInterval_Feb,May,Aug,Nov",
            "PromoInterval_Mar,Jun,Sept,Dec", "StoreType_a", "StoreType_b",
            "StoreType_d", "Assortment_a", "Assortment_c",
        ]
    with open(os.path.join(OUT_DIR, "sellout_meta.json"), "w") as fh:
        json.dump({"features": features,
                   "best_params": {k: str(v) for k, v in
                                   getattr(search, "best_params_", {}).items()}},
                  fh, indent=2)


@dataclass
class Step:
    name: str
    fn: Callable[[], None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["loan", "campaign", "sellout"])
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    steps = [Step("loan", convert_loan), Step("campaign", convert_campaign),
             Step("sellout", convert_sellout)]
    if args.only:
        steps = [s for s in steps if s.name == args.only]

    print("=" * 68)
    print(f"ONNX conversion — every export replayed against its original "
          f"(tolerance {TOLERANCE:.0e})")
    print("=" * 68)

    for step in steps:
        try:
            step.fn()
        except Exception as exc:  # keep going so CI reports every problem at once
            import traceback
            traceback.print_exc()
            failures.append(f"{step.name}: {type(exc).__name__}: {exc}")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    total = sum(os.path.getsize(os.path.join(OUT_DIR, f))
                for f in os.listdir(OUT_DIR)) / 1e6
    print("\n" + "=" * 68)
    print(f"exported {len(manifest)} graphs, {total:.2f} MB total")
    if failures:
        print(f"\n{len(failures)} FAILURE(S) — not publishing these artifacts:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all exports match their originals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
