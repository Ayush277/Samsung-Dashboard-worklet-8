"""The campaign decision layer: what to do with a sales prediction.

Read this before trusting a number out of it.
=============================================
This module is deliberately *not* machine learning, and the split matters:

  * predicted_sales     — a real model output (CatBoost / LightGBM / Ridge).
  * baseline / uplift   — real, measured. The baseline is what a store/item pair
                          actually sold across 913k rows of history, so uplift is
                          a genuine comparison against observed sales.
  * channel / cost      — a transparent business rule with configurable inputs.
                          Not learned from anything.

The version this replaces computed a "store performance factor" as
``0.8 + (store_id % 7) * 0.1`` and an "item appeal" as ``0.9 + (item_id % 15)
* 0.033``, with comments saying "simulate store quality impact". Those are
arbitrary arithmetic on an ID — store 8 scored higher than store 7 purely
because 8 % 7 == 1. They were presented in the UI as AI output. They are gone.

Anything below that is an assumption is labelled ``basis: assumption`` in the
payload and rendered as such in the UI, so no one mistakes a planning default
for a measurement.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Cost per contact, in currency units. These are planning inputs, not measured
# values, and no dataset in this project contains channel spend.
CHANNEL_COSTS: dict[str, float] = {
    "premium_digital": 25.0,
    "retention_program": 20.0,
    "personalized_offers": 15.0,
    "display_ads": 10.0,
    "social_media": 8.0,
    "targeted_email": 5.0,
    "standard_email": 2.0,
    "newsletter": 1.5,
    "broad_awareness": 12.0,
}

CHANNEL_RATIONALE: dict[str, str] = {
    "premium_digital": "Large measured uplift justifies the highest cost per contact.",
    "retention_program": "Sales hold near baseline — spend defends the existing run rate.",
    "personalized_offers": "Strong uplift with moderate volume rewards tailored offers.",
    "display_ads": "Middling uplift; broad reach at moderate cost.",
    "social_media": "Moderate uplift with high volume suits social reach.",
    "targeted_email": "Uplift concentrated in a small volume — target it cheaply.",
    "standard_email": "Modest uplift does not justify premium spend.",
    "newsletter": "Low uplift; keep contact cost minimal.",
    "broad_awareness": "Sales below baseline — awareness rather than conversion spend.",
}

UPLIFT_BANDS = ((0.15, "high"), (0.0, "moderate"))
VOLUME_BANDS = ((60.0, "high"), (25.0, "moderate"))

# (uplift band, volume band) -> channel
CHANNEL_MATRIX: dict[tuple[str, str], str] = {
    ("high", "high"): "premium_digital",
    ("high", "moderate"): "personalized_offers",
    ("high", "low"): "targeted_email",
    ("moderate", "high"): "social_media",
    ("moderate", "moderate"): "display_ads",
    ("moderate", "low"): "standard_email",
    ("low", "high"): "retention_program",
    ("low", "moderate"): "newsletter",
    ("low", "low"): "broad_awareness",
}


@lru_cache(maxsize=1)
def _baselines() -> dict | None:
    path = os.path.join(MODEL_DIR, "campaign_baselines.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _band(value: float, bands: tuple) -> str:
    for threshold, label in bands:
        if value >= threshold:
            return label
    return "low"


def baseline_for(store: int, item: int, month: int | None = None,
                 dow: int | None = None) -> dict | None:
    """The sales this store/item pair actually averaged, from real history.

    Prefers the most specific match available: month-of-year, then day-of-week,
    then the pair's overall mean.
    """
    data = _baselines()
    if not data:
        return None
    pair = data["pair"].get(f"{store}|{item}")
    if not pair:
        return None

    value, basis = pair["mean"], "historical mean for this store/item"
    if month is not None:
        seasonal = data["pair_month"].get(f"{store}|{item}|{month}")
        if seasonal is not None:
            value, basis = seasonal, f"historical mean for this store/item in month {month}"
    elif dow is not None:
        weekly = data["pair_dow"].get(f"{store}|{item}|{dow}")
        if weekly is not None:
            value, basis = weekly, "historical mean for this store/item on this weekday"

    return {
        "value": round(float(value), 2),
        "basis": basis,
        "pair_mean": pair["mean"],
        "pair_std": pair["std"],
        "observations": pair["n"],
        "date_range": data["date_range"],
        "measured": True,
    }


def evaluate(predicted_sales: float, store: int, item: int,
             month: int | None = None, dow: int | None = None) -> dict:
    """Turn a sales prediction into a campaign recommendation.

    Returns uplift measured against real history plus a rule-based channel
    choice. Every field carries a ``basis`` so the UI can show which numbers are
    measured and which are assumptions.
    """
    baseline = baseline_for(store, item, month, dow)
    if baseline is None:
        return {
            "available": False,
            "reason": "No historical baseline for this store/item pair. Uplift is "
                      "only meaningful against observed sales, so it is not reported.",
        }

    base = baseline["value"]
    uplift_abs = predicted_sales - base
    uplift_pct = (uplift_abs / base) if base > 0 else 0.0

    uplift_band = _band(uplift_pct, UPLIFT_BANDS)
    volume_band = _band(predicted_sales, VOLUME_BANDS)
    channel = CHANNEL_MATRIX[(uplift_band, volume_band)]
    cost = CHANNEL_COSTS[channel]

    # Expected incremental units per unit of channel spend. Real in the
    # numerator (model + measured baseline), assumed in the denominator (cost).
    efficiency = (uplift_abs / cost) if cost > 0 else 0.0

    # How unusual is this prediction against the pair's own variability?
    std = baseline["pair_std"] or 0.0
    z = (uplift_abs / std) if std > 0 else 0.0

    if uplift_pct >= 0.15:
        action = "Launch"
    elif uplift_pct > 0:
        action = "Test on a subset"
    else:
        action = "Hold"

    return {
        "available": True,
        "baseline": baseline,
        "uplift_absolute": round(uplift_abs, 2),
        "uplift_pct": round(uplift_pct, 4),
        "uplift_band": uplift_band,
        "uplift_sigma": round(z, 2),
        "volume_band": volume_band,
        "recommended_channel": channel,
        "channel_rationale": CHANNEL_RATIONALE[channel],
        "channel_cost_per_contact": cost,
        "units_per_cost_unit": round(efficiency, 3),
        "recommended_action": action,
        "bases": {
            "predicted_sales": "model",
            "baseline": "measured from history",
            "uplift": "measured (model vs history)",
            "recommended_channel": "rule",
            "channel_cost_per_contact": "assumption",
            "units_per_cost_unit": "rule (uses assumed cost)",
            "recommended_action": "rule",
        },
        "rules": {
            "uplift_bands": {"high": ">= +15%", "moderate": "0% to +15%", "low": "< 0%"},
            "volume_bands": {"high": ">= 60 units", "moderate": "25-60 units",
                             "low": "< 25 units"},
            "matrix": {f"{u}/{v}": c for (u, v), c in CHANNEL_MATRIX.items()},
        },
    }
