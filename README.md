# PRISM Worklet 8 — Financing, Campaign and Sales Intelligence

Three trained models, served in real time from one Vercel deployment:

| Use case | Model | Task | Held-out result |
|---|---|---|---|
| Loan delinquency risk | RandomForest (160 trees, depth 12) | Binary classification | **AUC 0.7209** vs 0.500 baseline |
| Campaign performance | CatBoost · LightGBM · Ridge | Regression (units sold) | 3 models, compared live |
| Sell-out forecasting | XGBoost (RandomizedSearchCV) | Regression (daily sales) | Not recorded — see below |

**Live:** https://samsung-dashboard-worklet-8.vercel.app
**Docs:** `/docs` · **Architecture:** `/architecture` · **Health:** `/api/health`

---

## What changed in v2, and why

The previous deployment was a launcher page. It reported `running: true` for all
three apps unconditionally and linked to `/loan/`, `/campaign/` and `/sales/`,
which returned 404. Its `requirements.txt` contained Flask and Werkzeug and no ML
library at all. **No model had ever been deployed.**

That was not carelessness — it was a dependency budget. scikit-learn, CatBoost,
LightGBM, XGBoost, SciPy and pandas total **~330 MB installed**, and a Vercel
Python function gets **250 MB**. The models could not fit next to the libraries
that trained them.

**The fix:** a trained tree ensemble is a set of split thresholds. Evaluating it
does not require the library that grew it. Every model is exported to ONNX and
served by ONNX Runtime — **~95 MB with numpy**, comfortably inside the budget, and
covering all four model families from one wheel.

Conversion runs in [GitHub Actions](.github/workflows/build-models.yml), never on
a developer machine, and **every export is replayed against the original model**.
The build fails on drift beyond `1e-3`, and the commit step only runs on success,
so an unverified graph cannot reach production. Measured drift for every deployed
graph is published at [`/api/models`](https://samsung-dashboard-worklet-8.vercel.app/api/models).

### Corrections to earlier documentation

Three claims in the previous README and app code were not true. They are recorded
here rather than quietly deleted.

1. **"TabPFN foundation model integration"** — the artifact `models/tabpfn.pkl`
   loads as `sklearn.ensemble.RandomForestClassifier`. The code that writes it
   trains a RandomForest and saves it under that name; the `tabpfn` package is
   imported nowhere that runs. This repository's title says "Harnessing TabFM",
   and no tabular foundation model is used. See [On TabFM](#on-tabfm).

2. **Campaign "conversion probability" and uplift were invented.** They were
   computed from `store_id % 10`, `item_id % 20` and `0.8 + (store_id % 7) * 0.1`,
   with source comments reading *"simulate store quality impact"*. Store 8 scored
   above store 7 because `8 % 7 == 1`. These were shown to users as AI output.
   Uplift is now measured against the real historical mean for that store and item
   across 913,001 rows. Conversion probability is **removed** — no dataset here
   records impressions, clicks or conversions.

3. **Categorical inputs never reached the loan model.** The app called
   `pd.get_dummies(..., drop_first=True)` on a single row at inference time. One
   row has one category, `drop_first` removes it, and the reindex filled every
   dummy with zero — so all six categorical fields silently collapsed to their
   reference level. Encoding is now explicit against the fitted dummy columns.

---

## Quick start

```bash
git clone https://github.com/Ayush277/Samsung-Dashboard-worklet-8.git
cd Samsung-Dashboard-worklet-8
pip install -r requirements.txt     # Flask, numpy, onnxruntime — no training libs
python -m flask --app api.index run --port 5050
```

Open http://127.0.0.1:5050.

The ONNX graphs live in `prism/models/` and are built by CI. Building them
yourself needs the training-library stack (~330 MB), which is exactly what CI is
for:

```bash
gh workflow run "Build ONNX models"     # runs on GitHub's machines
```

---

## Layout

```
api/index.py                  Vercel entrypoint (WSGI)
prism/
  __init__.py                 app factory
  routes.py                   pages, prediction APIs, batch, health
  inference.py                ONNX Runtime sessions — all real predictions
  features.py                 feature schemas (order matters: ONNX takes bare tensors)
  campaign_rules.py           decision layer — explicitly NOT ML, and labelled so
  models/                     .onnx graphs + metadata (built by CI)
  templates/  static/         UI
tools/convert_models.py       .pkl → .onnx + parity verification (CI only)
.github/workflows/            model build pipeline

Loan delinquency risk/                           source project + trained .pkl
Campaign performance (marketing)/                source project + trained .pkl
Sell-out performance forecasting (sales uplift)/ source project + trained .pkl
```

The three source projects stay in the repo as the provenance of the artifacts.
They are excluded from the Vercel bundle via `.vercelignore` — only the ONNX
graphs ship.

---

## Every number is labelled

The API tags each field with its basis, and the UI renders the tag:

| Tag | Meaning |
|---|---|
| **model** | Output of a trained model |
| **measured** | Counted from real data (e.g. 913k rows of sales history) |
| **rule** | Deterministic business logic |
| **assumption** | A planning input nobody measured (e.g. channel cost) |

Channel selection is a rule. Cost per contact is an assumption. Predicted sales is
a model output. Uplift is measured. All four are legitimate — conflating them is
what made the previous version misleading.

---

## Loan model — the honest summary

**AUC 0.7209** on a stratified 25% held-out split of 116,058 loans, against a
majority-class baseline of 0.500. Real signal.

Accuracy is **0.664**, *below* the 0.690 you get by predicting "good loan" for
everyone. That is deliberate: `class_weight='balanced'` makes the model pay for
missing the minority class, trading raw accuracy for ranking ability. For triage,
ranking is what matters. Accuracy alone on a 69/31 split just reports the baseline.

Where the signal lives:

| Feature | Importance |
|---|---|
| total_on_time_payments | 0.364 |
| total_late_payments | 0.262 |
| current_dpd | 0.136 |
| avg_payment_delay | 0.105 |
| Annual Income | 0.016 |
| borrower_credit_score | **0.015** |

Payment history is **87%** of total importance. Credit score is 1.5%, and
correlates `+0.0018` with the target — effectively zero. On a real lending book
credit score is *the* canonical delinquency predictor, so a near-zero correlation
strongly suggests the demographic columns of this dataset are synthetic. The
pipeline is real; the data is not a lending book. Treat the model as a working
demonstration, not a credit policy.

### The inverted target

In `approach_train.csv`, **`mx = 1` means a GOOD loan**. Reading column 1 of
`predict_proba` as "risk" — the obvious move — returns the probability the loan is
*fine*. The service reads column 0. This is asserted in code and documented
because it is the easiest way to ship a model that is confidently backwards.

## On TabFM

TabPFN was evaluated for this deployment and ruled out on a hard constraint: it
requires PyTorch (~2 GB installed) against a 250 MB serverless budget — roughly 8×
over before the weights are counted. It cannot serve real-time predictions on this
URL under any arrangement.

If the worklet brief requires a tabular foundation model, the honest options are to
run TabPFN offline and publish the comparison as a result, or to move inference to
a host with room for PyTorch. What is not an option is leaving a RandomForest named
`tabpfn.pkl` and calling it a foundation model. The RandomForest needs no cover
story: AUC 0.72 on 116k rows, 4.4 MB, single-digit millisecond inference.

---

## API

```bash
BASE=https://samsung-dashboard-worklet-8.vercel.app

# Loan — omitted fields fall back to their training median
curl -X POST $BASE/api/loan/predict -H 'Content-Type: application/json' \
  -d '{"borrower_credit_score":610,"current_dpd":45,"total_late_payments":14}'

# Campaign — compare=1 scores all three models on the same input
curl -X POST $BASE/api/campaign/predict -d 'store=3&item=12&month=12&day=20&compare=1'

# Sell-out
curl -X POST $BASE/api/sellout/predict -d 'Open=1&Promo=1&DayOfWeek=6&StoreType=d'

# Batch: CSV in, CSV out (≤2000 rows)
curl -X POST $BASE/api/loan/batch -F 'file=@applications.csv' -o scored.csv

# Health — loads every model and runs a real inference through each
curl $BASE/api/health

# Manifest — measured drift of each deployed graph vs its original
curl $BASE/api/models
```

---

## Limitations

- The loan dataset's demographic columns look synthetic (credit score correlates
  +0.002 with the target).
- **No sell-out error metric is published** — none was recorded in the artifact,
  and inventing one is not an option. Recovering it requires rerunning training.
- Channel costs are assumptions, labelled as such.
- Campaign uplift requires a store/item pair present in the 913k-row history;
  unknown pairs get no baseline and the API says so rather than guessing.
- Cold starts pay ONNX session construction on the first request to an idle
  instance.
- **None of these models has been validated for real lending, pricing or
  inventory decisions.**
