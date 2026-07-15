"""HTTP surface: pages, JSON prediction APIs, batch scoring, health."""

from __future__ import annotations

import csv
import io
import json
import os
import time

from flask import Blueprint, Response, jsonify, render_template, request

from . import __version__, campaign_rules
from . import features as F
from . import inference as I

bp = Blueprint("prism", __name__)

USE_CASES = (
    {
        "slug": "loan",
        "name": "Loan Delinquency Risk",
        "tagline": "Probability of delinquency for a loan application",
        "model": "RandomForest · 160 trees",
        "task": "Binary classification",
        "metric": "AUC 0.72",
        "endpoint": "/api/loan/predict",
    },
    {
        "slug": "campaign",
        "name": "Campaign Performance",
        "tagline": "Store/item demand with measured uplift vs history",
        "model": "CatBoost · LightGBM · Ridge",
        "task": "Regression",
        "metric": "3 models, compare live",
        "endpoint": "/api/campaign/predict",
    },
    {
        "slug": "sellout",
        "name": "Sell-out Forecasting",
        "tagline": "Daily store sales from promotion and competition context",
        "model": "XGBoost",
        "task": "Regression",
        "metric": "17 features",
        "endpoint": "/api/sellout/predict",
    },
)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@bp.get("/")
def index():
    return render_template("index.html", use_cases=USE_CASES, version=__version__)


@bp.get("/loan")
def loan_page():
    return render_template("loan.html", fields=F.LOAN_FIELDS, use_cases=USE_CASES)


@bp.get("/campaign")
def campaign_page():
    return render_template("campaign.html", fields=F.CAMPAIGN_FIELDS,
                           models=F.CAMPAIGN_MODELS, use_cases=USE_CASES)


@bp.get("/sellout")
def sellout_page():
    return render_template("sellout.html", fields=F.SELLOUT_FIELDS, use_cases=USE_CASES)


@bp.get("/docs")
def docs_page():
    return render_template("docs.html", use_cases=USE_CASES, version=__version__)


@bp.get("/architecture")
def architecture_page():
    return render_template("architecture.html", use_cases=USE_CASES)


# ---------------------------------------------------------------------------
# Health — measured, never asserted
# ---------------------------------------------------------------------------
@bp.get("/api/health")
def health():
    """Load every model and run one real inference through it.

    The predecessor returned running:True unconditionally. This endpoint is slow
    on a cold start by design: a health check that does not exercise the thing it
    reports on is decoration.
    """
    started = time.perf_counter()
    checks: dict[str, dict] = {}

    probes = {
        "loan": lambda: I.predict_loan({}),
        "campaign": lambda: I.predict_campaign(1, 1, 7, 15, "catboost"),
        "sellout": lambda: I.predict_sellout({}),
    }
    for name, probe in probes.items():
        t0 = time.perf_counter()
        try:
            probe()
            checks[name] = {"ok": True,
                            "inference_ms": round((time.perf_counter() - t0) * 1000, 2)}
        except Exception as exc:
            checks[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    baselines = campaign_rules._baselines()
    ok = all(c["ok"] for c in checks.values())
    return jsonify({
        "ok": ok,
        "version": __version__,
        "checks": checks,
        "runtime": I.model_dir_status(),
        "campaign_baselines": {
            "loaded": baselines is not None,
            "pairs": len(baselines["pair"]) if baselines else 0,
            "rows_of_history": baselines.get("rows") if baselines else 0,
        },
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
    }), (200 if ok else 503)


@bp.get("/api/models")
def models_manifest():
    """Conversion provenance: drift of each ONNX graph vs the model it came from."""
    path = os.path.join(I.MODEL_DIR, "manifest.json")
    if not os.path.exists(path):
        return jsonify({"error": "manifest not found; run the Build ONNX models workflow"}), 404
    with open(path) as fh:
        return jsonify(json.load(fh))


# ---------------------------------------------------------------------------
# Prediction APIs
# ---------------------------------------------------------------------------
def _payload() -> dict:
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict()


@bp.post("/api/loan/predict")
def api_loan():
    try:
        return jsonify(I.predict_loan(_payload()))
    except I.ModelUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400


@bp.post("/api/campaign/predict")
def api_campaign():
    data = _payload()
    try:
        store = int(data.get("store", 1))
        item = int(data.get("item", 1))
        month = int(data.get("month", 7))
        day = int(data.get("day", 15))
        model = str(data.get("model", "catboost"))
        compare = str(data.get("compare", "")).lower() in ("1", "true", "on", "yes")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"invalid input: {exc}"}), 400

    try:
        result = I.predict_campaign(store, item, month, day, model)
        import datetime as dt
        dow = dt.date(2024, month, day).weekday()
        result["campaign"] = campaign_rules.evaluate(
            result["predicted_sales"], store, item, month=month, dow=dow)
        if compare:
            result["all_models"] = I.predict_campaign_all(store, item, month, day)
        return jsonify(result)
    except I.ModelUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.post("/api/sellout/predict")
def api_sellout():
    try:
        return jsonify(I.predict_sellout(_payload()))
    except I.ModelUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------
BATCH_ROW_LIMIT = 2000


@bp.post("/api/<use_case>/batch")
def api_batch(use_case: str):
    """Score an uploaded CSV row by row and stream back a CSV of predictions."""
    if use_case not in ("loan", "campaign", "sellout"):
        return jsonify({"error": f"unknown use case {use_case!r}"}), 404
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"error": "no file uploaded (field name: 'file')"}), 400

    try:
        text = upload.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        return jsonify({"error": "file must be UTF-8 encoded CSV"}), 400

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return jsonify({"error": "CSV contained no data rows"}), 400
    if len(rows) > BATCH_ROW_LIMIT:
        return jsonify({"error": f"{len(rows)} rows exceeds the {BATCH_ROW_LIMIT}-row "
                                 f"limit for a serverless request"}), 413

    out = io.StringIO()
    writer = None
    errors = 0
    started = time.perf_counter()

    for row in rows:
        try:
            if use_case == "loan":
                r = I.predict_loan(row)
                record = dict(row, probability_of_default=r["probability_of_default"],
                              risk_band=r["risk_band"], prediction=r["prediction"])
            elif use_case == "campaign":
                store, item = int(row["store"]), int(row["item"])
                if "date" in row and row["date"]:
                    import datetime as dt
                    d = dt.date.fromisoformat(row["date"].strip()[:10])
                    month, day = d.month, d.day
                else:
                    month, day = int(row.get("month", 7)), int(row.get("day", 15))
                r = I.predict_campaign(store, item, month, day,
                                       str(row.get("model", "catboost")))
                ev = campaign_rules.evaluate(r["predicted_sales"], store, item, month=month)
                record = dict(row, predicted_sales=r["predicted_sales"],
                              baseline=ev.get("baseline", {}).get("value") if ev.get("available") else "",
                              uplift_pct=ev.get("uplift_pct", "") if ev.get("available") else "",
                              recommended_channel=ev.get("recommended_channel", "") if ev.get("available") else "")
            else:
                r = I.predict_sellout(row)
                record = dict(row, predicted_sales=r["predicted_sales"])
        except I.ModelUnavailable as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:
            errors += 1
            record = dict(row, error=f"{type(exc).__name__}: {exc}")

        if writer is None:
            writer = csv.DictWriter(out, fieldnames=list(record.keys()),
                                    extrasaction="ignore")
            writer.writeheader()
        writer.writerow(record)

    elapsed = (time.perf_counter() - started) * 1000
    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{use_case}_predictions.csv"',
            "X-Rows-Scored": str(len(rows)),
            "X-Rows-Failed": str(errors),
            "X-Elapsed-Ms": f"{elapsed:.1f}",
        },
    )


@bp.get("/api/<use_case>/sample.csv")
def api_sample(use_case: str):
    samples = {
        "loan": ("borrower_credit_score,current_dpd,total_on_time_payments,"
                 "total_late_payments,avg_payment_delay,interest_rate,"
                 "unpaid_principal_bal,loan_to_value,debt_to_income_ratio,Age\n"
                 "782,11,9,4,12.4,3.875,183000,72,31,40\n"
                 "610,45,3,14,28.0,6.5,240000,91,48,29\n"
                 "805,0,24,0,2.0,3.1,120000,55,18,52\n"),
        "campaign": "store,item,date,model\n1,1,2024-07-15,catboost\n"
                    "3,12,2024-12-20,lightgbm\n7,40,2024-03-02,ridge\n",
        "sellout": ("DayOfWeek,Open,Promo,StateHoliday,SchoolHoliday,"
                    "CompetitionDistance,CompetitionOpenNumMonths,Promo2NumWeeks,"
                    "WeekOfYear,StoreType,Assortment,PromoInterval\n"
                    "3,1,1,0,0,1270,24,0,28,a,a,0\n"
                    "6,1,0,0,1,570,60,12,50,d,c,\"Feb,May,Aug,Nov\"\n"),
    }
    if use_case not in samples:
        return jsonify({"error": "unknown use case"}), 404
    return Response(samples[use_case], mimetype="text/csv",
                    headers={"Content-Disposition":
                             f'attachment; filename="{use_case}_sample.csv"'})
