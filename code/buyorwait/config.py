from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_IMAGES_DIR = DATASET_DIR / "media" / "images"
CACHE_DIR = REPO_ROOT / "code" / "cache"
EVALUATION_DIR = REPO_ROOT / "evaluation"
OUTPUT_CSV = REPO_ROOT / "output.csv"

REQUESTS_CSV = DATASET_DIR / "requests.csv"
PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"
EVENTS_CSV = DATASET_DIR / "financial_events.csv"
EXCHANGE_RATES_CSV = DATASET_DIR / "exchange_rates.csv"
PAYMENT_OPTIONS_CSV = DATASET_DIR / "request_payment_options.csv"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"

FORECAST_HORIZON_DAYS = 90

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

AFFORDABILITY_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

PAYMENT_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}

# One tier below each status, used for the low-confidence / unresolvable-debit downgrade.
AFFORDABILITY_DOWNGRADE = {
    "affordable_now": "affordable_with_plan",
    "affordable_with_plan": "affordable_later",
    "affordable_later": "not_affordable",
    "not_affordable": "not_affordable",
}

LLM_PROVIDER = "anthropic"
VISION_MODEL = "claude-sonnet-5"
TEXT_MODEL = "claude-sonnet-5"
