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
# Matches the organizer's own starter template: code/evaluation/usage_report.md is an
# empty placeholder shipped in the original repo (git history: commit c280fff "feat: add
# code folder"), not repo-root evaluation/ -- confirmed by checking git log, not assumed.
EVALUATION_DIR = REPO_ROOT / "code" / "evaluation"
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

# Gemini free-tier daily request quotas are tracked PER MODEL (confirmed directly from a
# real 429's quotaId: "GenerateRequestsPerDayPerProjectPerModel-FreeTier", model in the
# quotaDimensions) -- so a chain of distinct models each carries its own separate
# allowance. Tried in order; llm_client.py's circuit breaker marks a model exhausted the
# moment it sees a quota-exhaustion error and skips straight past it on every later call
# in the same process, rather than re-attempting (and waiting out) a doomed request.
# gemini-3.5-flash-lite is prioritized 2nd: confirmed via the user's real AI Studio
# dashboard (not a guess) at ~484/500 RPD still available today, vs. gemini-3.6-flash
# already confirmed exhausted (23/20 RPD) and gemini-2.5-flash-lite/gemini-flash-lite-latest
# of unknown/partially-used status.
GEMINI_TEXT_MODEL_CHAIN = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-2.5-flash"]
GEMINI_VISION_MODEL_CHAIN = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-2.5-flash"]

# Confirmed real cap for gemini-3.5-flash-lite (and used as the safe default for the whole
# Gemini chain, since per-model RPM isn't individually confirmed for the others): 15
# requests per minute. This is a pacing constraint, not a daily blocker -- llm_client.py
# throttles calls to stay under it rather than hitting avoidable 429s.
GEMINI_RPM_LIMIT = 15
