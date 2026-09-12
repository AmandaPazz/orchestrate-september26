from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Loads ANTHROPIC_API_KEY / GEMINI_API_KEY from a .env file at the repo root into the
# process environment, if one exists -- so a key set once doesn't need re-exporting in
# every new shell. Never overrides a variable already set in the real environment
# (override=False), and does nothing (silently) if no .env file is present -- manual
# `export`/`$env:` still works exactly as before, this is purely additive.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=False)
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

# Provider order: Anthropic first if ANTHROPIC_API_KEY is set (env-driven, checked at call
# time in llm_client.py, never hardcoded here), Gemini as fallback or primary when only
# GEMINI_API_KEY is set. Both providers' text and vision calls use the same model per
# provider for simplicity.
ANTHROPIC_TEXT_MODEL = "claude-sonnet-5"
ANTHROPIC_VISION_MODEL = "claude-sonnet-5"
GEMINI_TEXT_MODEL = "gemini-3.6-flash"
GEMINI_VISION_MODEL = "gemini-3.6-flash"
