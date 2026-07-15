"""Feature schemas for the three models.

These mirror the columns each model was trained on. The order in ``COLUMNS`` is
load-bearing: ONNX takes a bare float tensor, so a reordered column silently
feeds the wrong number into the wrong split. Order comes from the fitted
artifacts themselves (dummy_columns.pkl, scaler.feature_names_in_), never from
a hand-written list.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    kind: str  # "number" | "select"
    default: float | str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    hint: str = ""
    unit: str = ""


# ---------------------------------------------------------------------------
# Loan delinquency
# ---------------------------------------------------------------------------

# Clamps replicated from the original app so a typo can't push a value into a
# region the model never saw during training.
LOAN_CLAMPS: dict[str, tuple[float, float]] = {
    "borrower_credit_score": (300, 850),
    "co-borrower_credit_score": (300, 850),
    "interest_rate": (0, 50),
    "Age": (18, 100),
    "loan_to_value": (0, 100),
    "insurance_percent": (0, 100),
    "total_on_time_payments": (0, float("inf")),
    "total_late_payments": (0, float("inf")),
    "current_dpd": (0, float("inf")),
}

LOAN_NUMERIC = [
    "interest_rate", "unpaid_principal_bal", "Loan_term", "loan_to_value",
    "number_of_borrowers", "debt_to_income_ratio", "borrower_credit_score",
    "insurance_percent", "co-borrower_credit_score", "Age", "NumberOfDependents",
    "Annual_Income", "total_on_time_payments", "total_late_payments",
    "avg_payment_delay", "current_dpd",
]

# Reference category first — it is the one dropped by drop_first during
# training, so it is encoded as all-zeros rather than getting its own column.
LOAN_CATEGORICAL: dict[str, tuple[str, ...]] = {
    "source": ("X", "Y", "Z"),
    "loan_purpose": ("A23", "B12", "C86"),
    "EducationLevel": ("Bachelor's", "Doctorate", "High School", "Master's", "PhD"),
    "MaritalStatus": ("Divorced", "Married", "Single"),
    "Gender": ("Female", "Male", "Other"),
    "EmploymentStatus": ("Employed", "Self-Employed", "Unemployed"),
}

LOAN_FIELDS: tuple[Field, ...] = (
    Field("borrower_credit_score", "Borrower credit score", "number", 782, 300, 850, 1,
          hint="FICO range 300–850"),
    Field("co-borrower_credit_score", "Co-borrower credit score", "number", 740, 300, 850, 1),
    Field("current_dpd", "Current days past due", "number", 11, 0, 365, 1,
          hint="Strongest single signal in the model", unit="days"),
    Field("total_on_time_payments", "On-time payments", "number", 9, 0, 500, 1,
          hint="Highest-importance feature (0.36)"),
    Field("total_late_payments", "Late payments", "number", 4, 0, 500, 1,
          hint="Second-highest importance (0.26)"),
    Field("avg_payment_delay", "Average payment delay", "number", 12.4, 0, 365, 0.1,
          unit="days"),
    Field("interest_rate", "Interest rate", "number", 3.875, 0, 50, 0.001, unit="%"),
    Field("unpaid_principal_bal", "Unpaid principal balance", "number", 183000, 0, 1e7, 1000,
          unit="$"),
    Field("Loan_term", "Loan term", "number", 18, 1, 480, 1, unit="months"),
    Field("loan_to_value", "Loan-to-value", "number", 72, 0, 100, 0.1, unit="%"),
    Field("debt_to_income_ratio", "Debt-to-income", "number", 31, 0, 100, 0.1, unit="%"),
    Field("Annual_Income", "Annual income", "number", 6256.41, 0, 1e7, 100, unit="$"),
    Field("number_of_borrowers", "Number of borrowers", "number", 2, 1, 4, 1),
    Field("insurance_percent", "Insurance percent", "number", 0, 0, 100, 0.1, unit="%"),
    Field("Age", "Age", "number", 40, 18, 100, 1, unit="years"),
    Field("NumberOfDependents", "Dependents", "number", 1, 0, 12, 1),
    Field("source", "Source", "select", "X", choices=("X", "Y", "Z")),
    Field("loan_purpose", "Loan purpose", "select", "A23", choices=("A23", "B12", "C86")),
    Field("EducationLevel", "Education", "select", "Bachelor's",
          choices=("Bachelor's", "Doctorate", "High School", "Master's", "PhD")),
    Field("MaritalStatus", "Marital status", "select", "Married",
          choices=("Divorced", "Married", "Single")),
    Field("Gender", "Gender", "select", "Female", choices=("Female", "Male", "Other")),
    Field("EmploymentStatus", "Employment", "select", "Employed",
          choices=("Employed", "Self-Employed", "Unemployed")),
)

# P(default) thresholds, from the original app.
LOAN_RISK_BANDS = ((0.25, "Low"), (0.50, "Moderate"), (0.75, "High"))
LOAN_RISK_CRITICAL = "Critical"


def loan_risk_band(p: float) -> str:
    for threshold, label in LOAN_RISK_BANDS:
        if p < threshold:
            return label
    return LOAN_RISK_CRITICAL


# ---------------------------------------------------------------------------
# Campaign — store/item demand
# ---------------------------------------------------------------------------
CAMPAIGN_FIELDS: tuple[Field, ...] = (
    Field("store", "Store", "number", 1, 1, 10, 1),
    Field("item", "Item", "number", 1, 1, 50, 1),
    Field("month", "Month", "number", 7, 1, 12, 1),
    Field("day", "Day", "number", 15, 1, 31, 1),
)
CAMPAIGN_MODELS = ("catboost", "lightgbm", "ridge")


# ---------------------------------------------------------------------------
# Sell-out — Rossmann
# ---------------------------------------------------------------------------
SELLOUT_FIELDS: tuple[Field, ...] = (
    Field("DayOfWeek", "Day of week", "number", 3, 1, 7, 1),
    Field("Open", "Store open", "select", "1", choices=("1", "0")),
    Field("Promo", "Promotion running", "select", "1", choices=("1", "0")),
    Field("StateHoliday", "State holiday", "select", "0", choices=("0", "1")),
    Field("SchoolHoliday", "School holiday", "select", "0", choices=("0", "1")),
    Field("CompetitionDistance", "Competition distance", "number", 1270, 0, 100000, 10,
          unit="m"),
    Field("CompetitionOpenNumMonths", "Competition open", "number", 24, 0, 600, 1,
          unit="months"),
    Field("Promo2NumWeeks", "Promo2 duration", "number", 0, 0, 600, 1, unit="weeks"),
    Field("WeekOfYear", "Week of year", "number", 28, 1, 53, 1),
    Field("StoreType", "Store type", "select", "a", choices=("a", "b", "c", "d")),
    Field("Assortment", "Assortment", "select", "a", choices=("a", "b", "c")),
    Field("PromoInterval", "Promo interval", "select", "0",
          choices=("0", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec")),
)
