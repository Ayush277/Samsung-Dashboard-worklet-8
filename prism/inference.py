"""ONNX Runtime inference for all three models.

Sessions are created lazily and cached per process. On Vercel that means the
first request after a cold start pays the graph-load cost (tens of ms for these
models) and every later request on the same warm instance reuses the session.

Every number returned by this module comes out of a model that was trained on
the project's data. Nothing here fabricates, simulates or randomises a
prediction: if a model cannot be loaded, the caller gets an error, not a
plausible-looking number.
"""

from __future__ import annotations

import json
import os
import threading
import time
from functools import lru_cache

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # surfaced by /api/health rather than crashing at import
    ort = None

from . import features as F

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_lock = threading.Lock()


class ModelUnavailable(RuntimeError):
    """Raised when an artifact is missing or unloadable.

    Deliberately fatal to the request. The predecessor to this app reported
    every model as healthy regardless of state; a caller must be able to tell
    a real prediction from a broken one.
    """


def model_dir_status() -> dict:
    present = sorted(f for f in os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []
    return {
        "onnxruntime": getattr(ort, "__version__", None),
        "model_dir": MODEL_DIR,
        "artifacts": present,
    }


@lru_cache(maxsize=None)
def _session(filename: str):
    if ort is None:
        raise ModelUnavailable("onnxruntime is not installed")
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise ModelUnavailable(
            f"{filename} not found. Models are built by the 'Build ONNX models' "
            f"GitHub Actions workflow; run it to publish artifacts."
        )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1  # serverless gives us one useful core
    with _lock:
        return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=None)
def _meta(filename: str) -> dict:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise ModelUnavailable(f"{filename} not found")
    with open(path) as fh:
        return json.load(fh)


def _run(filename: str, X: np.ndarray, output_index: int = 0) -> np.ndarray:
    sess = _session(filename)
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: np.ascontiguousarray(X, dtype=np.float32)})
    return np.asarray(out[output_index])


# ---------------------------------------------------------------------------
# Loan delinquency
# ---------------------------------------------------------------------------
def loan_meta() -> dict:
    return _meta("loan_meta.json")


def _clamp(name: str, value: float) -> float:
    lo, hi = F.LOAN_CLAMPS.get(name, (-float("inf"), float("inf")))
    return max(lo, min(hi, value))


def build_loan_vector(payload: dict) -> tuple[np.ndarray, dict]:
    """Encode a raw form payload into the model's 30-column feature vector.

    Categoricals are encoded explicitly against the fitted dummy columns. The
    original app called pd.get_dummies(drop_first=True) on a single row, which
    always drops the only category present — so every categorical silently
    collapsed to its reference level and never reached the model.
    """
    meta = loan_meta()
    columns: list[str] = meta["columns"]
    medians: dict = meta["medians"]

    row: dict[str, float] = {}
    resolved: dict[str, object] = {}

    for name in F.LOAN_NUMERIC:
        raw = payload.get(name)
        if raw in (None, "", "None"):
            value = float(medians.get(name, 0.0))
        else:
            try:
                value = _clamp(name, float(raw))
            except (TypeError, ValueError):
                value = float(medians.get(name, 0.0))
        if name in columns:
            row[name] = value
        resolved[name] = value

    for name, allowed in F.LOAN_CATEGORICAL.items():
        chosen = payload.get(name) or allowed[0]
        if chosen not in allowed:
            chosen = allowed[0]
        resolved[name] = chosen
        # allowed[0] is the reference level: all its dummies stay 0.
        for level in allowed[1:]:
            column = f"{name}_{level}"
            if column in columns:
                row[column] = 1.0 if chosen == level else 0.0

    vector = np.array([[float(row.get(c, 0.0)) for c in columns]], dtype=np.float32)
    return vector, resolved


def predict_loan(payload: dict) -> dict:
    started = time.perf_counter()
    X, resolved = build_loan_vector(payload)
    Xs = _run("loan_scaler.onnx", X)
    proba = _run("loan_random_forest.onnx", Xs, output_index=1)

    # Training target mx is inverted: mx=1 marks a GOOD loan, so the
    # probability of delinquency is P(mx=0) — column 0, not column 1.
    p_default = float(np.asarray(proba).reshape(1, -1)[0, 0])
    elapsed_ms = (time.perf_counter() - started) * 1000

    meta = loan_meta()
    importances: dict = meta.get("feature_importances", {})
    drivers = []
    for name in F.LOAN_NUMERIC:
        if name not in importances or name not in meta["medians"]:
            continue
        median = float(meta["medians"][name])
        value = float(resolved.get(name, median))
        delta = value - median
        if abs(delta) < 1e-9:
            continue
        protective = name in ("borrower_credit_score", "co-borrower_credit_score",
                              "Annual_Income", "total_on_time_payments")
        raises_risk = (delta < 0) if protective else (delta > 0)
        drivers.append({
            "feature": name,
            "value": value,
            "median": median,
            "importance": float(importances[name]),
            "direction": "raises risk" if raises_risk else "lowers risk",
        })
    drivers.sort(key=lambda d: -d["importance"])

    return {
        "probability_of_default": round(p_default, 6),
        "risk_band": F.loan_risk_band(p_default),
        "prediction": "Delinquent" if p_default >= 0.5 else "Not delinquent",
        "drivers": drivers[:6],
        "inference_ms": round(elapsed_ms, 2),
        "model": {
            "type": meta.get("model_type"),
            "n_estimators": meta.get("n_estimators"),
            "max_depth": meta.get("max_depth"),
        },
        "resolved_inputs": resolved,
    }


# ---------------------------------------------------------------------------
# Campaign — store/item demand
# ---------------------------------------------------------------------------
_CAMPAIGN_FILES = {
    "catboost": "campaign_catboost.onnx",
    "lightgbm": "campaign_lightgbm.onnx",
    "ridge": "campaign_ridge.onnx",
}


def campaign_vector(store: int, item: int, month: int, day: int, year: int = 2024):
    import datetime as dt

    try:
        date = dt.date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"invalid date: {year}-{month:02d}-{day:02d}") from exc
    return np.array([[
        store, item, month, day, date.weekday(), date.timetuple().tm_yday,
        date.isocalendar()[1],
    ]], dtype=np.float32), date


def predict_campaign(store: int, item: int, month: int, day: int,
                     model: str = "catboost", year: int = 2024) -> dict:
    if model not in _CAMPAIGN_FILES:
        raise ValueError(f"unknown model {model!r}; expected one of "
                         f"{sorted(_CAMPAIGN_FILES)}")
    started = time.perf_counter()
    X, date = campaign_vector(store, item, month, day, year)
    Xs = _run("campaign_scaler.onnx", X)
    raw = _run(_CAMPAIGN_FILES[model], Xs)
    sales = float(np.asarray(raw).ravel()[0])
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "predicted_sales": round(sales, 2),
        "model": model,
        "date": date.isoformat(),
        "inference_ms": round(elapsed_ms, 2),
    }


def predict_campaign_all(store: int, item: int, month: int, day: int,
                         year: int = 2024) -> dict:
    """Run all three models on one input so they can be compared side by side."""
    out = {}
    for name in _CAMPAIGN_FILES:
        try:
            out[name] = predict_campaign(store, item, month, day, name, year)
        except ModelUnavailable as exc:
            out[name] = {"error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# Sell-out — Rossmann
# ---------------------------------------------------------------------------
def sellout_meta() -> dict:
    return _meta("sellout_meta.json")


def build_sellout_vector(payload: dict) -> tuple[np.ndarray, dict]:
    columns: list[str] = sellout_meta()["features"]

    def num(key: str, default: float) -> float:
        try:
            return float(payload.get(key, default))
        except (TypeError, ValueError):
            return default

    values = {
        "DayOfWeek": num("DayOfWeek", 3),
        "Open": num("Open", 1),
        "Promo": num("Promo", 1),
        "StateHoliday": num("StateHoliday", 0),
        "SchoolHoliday": num("SchoolHoliday", 0),
        "CompetitionDistance": num("CompetitionDistance", 1270),
        "CompetitionOpenNumMonths": num("CompetitionOpenNumMonths", 24),
        "Promo2NumWeeks": num("Promo2NumWeeks", 0),
        "WeekOfYear": num("WeekOfYear", 28),
    }
    # One-hot columns exist only for the levels kept at training time; anything
    # else is the reference level and is correctly encoded as all-zeros.
    store_type = str(payload.get("StoreType", "a"))
    assortment = str(payload.get("Assortment", "a"))
    interval = str(payload.get("PromoInterval", "0"))
    for col in columns:
        if col.startswith("StoreType_"):
            values[col] = 1.0 if col == f"StoreType_{store_type}" else 0.0
        elif col.startswith("Assortment_"):
            values[col] = 1.0 if col == f"Assortment_{assortment}" else 0.0
        elif col.startswith("PromoInterval_"):
            values[col] = 1.0 if col == f"PromoInterval_{interval}" else 0.0

    vector = np.array([[float(values.get(c, 0.0)) for c in columns]], dtype=np.float32)
    resolved = dict(values, StoreType=store_type, Assortment=assortment,
                    PromoInterval=interval)
    return vector, resolved


def predict_sellout(payload: dict) -> dict:
    started = time.perf_counter()
    X, resolved = build_sellout_vector(payload)
    Xs = _run("sellout_scaler.onnx", X)
    raw = _run("sellout_xgboost.onnx", Xs)
    sales = float(np.asarray(raw).ravel()[0])
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "predicted_sales": round(sales, 2),
        "inference_ms": round(elapsed_ms, 2),
        "resolved_inputs": resolved,
        "model": {"type": "XGBRegressor",
                  "best_params": sellout_meta().get("best_params", {})},
    }
