"""Load and type dataset/*.csv into plain dataclasses, joined by user_id / request_id."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from . import config


def _parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    return date.fromisoformat(s)


def _parse_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if s == "":
        return None
    return float(s)


def _parse_bool(s: str) -> bool:
    return (s or "").strip().lower() == "true"


def _parse_pipe_list(s: str) -> tuple[str, ...]:
    s = (s or "").strip()
    if not s:
        return ()
    return tuple(part.strip() for part in s.split("|") if part.strip())


def _parse_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if s == "":
        return None
    return int(float(s))


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[str, ...]
    max_installment_months: Optional[int]


@dataclass
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str  # "credit" | "debit"
    amount: Optional[float]  # None until resolved (blank in source, may be filled by vision extraction)
    currency: str
    event_date: Optional[date]
    settlement_date: Optional[date]
    status: str  # settled|pending|scheduled|cancelled|failed|unrealized
    linked_event_id: Optional[str]
    flexibility: str  # fixed|stoppable|reducible|reducible_or_stoppable
    minimum_allowed_amount: Optional[float]
    # Populated during resolution (events.py); not from the raw CSV.
    home_currency_amount: Optional[float] = field(default=None)
    amount_confidence: str = field(default="high")  # "high" unless filled by a low-confidence LLM extraction
    excluded_reason: Optional[str] = field(default=None)  # set when dropped from the numeric forecast


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str  # full_payment|installments (as observed in the data)
    payment_amount: float
    number_of_payments: int
    first_payment_date: Optional[date]
    payment_frequency_days: Optional[int]
    financing_fee: float
    total_payable_amount: float


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: str
    source_type: str
    message_text: str


@dataclass(frozen=True)
class Image:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]


@dataclass(frozen=True)
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: float


def load_profiles() -> dict[str, Profile]:
    profiles: dict[str, Profile] = {}
    with open(config.PROFILES_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            profiles[row["user_id"]] = Profile(
                user_id=row["user_id"],
                home_currency=row["home_currency"].strip(),
                current_available_balance=_parse_float(row["current_available_balance"]) or 0.0,
                minimum_balance_to_keep=_parse_float(row["minimum_balance_to_keep"]) or 0.0,
                financial_priorities=_parse_pipe_list(row["financial_priorities"]),
                expense_categories_to_protect=_parse_pipe_list(row["expense_categories_to_protect"]),
                expense_categories_user_is_willing_to_reduce=_parse_pipe_list(
                    row["expense_categories_user_is_willing_to_reduce"]
                ),
                expense_categories_user_is_willing_to_stop=_parse_pipe_list(
                    row["expense_categories_user_is_willing_to_stop"]
                ),
                payment_methods_user_will_consider=_parse_pipe_list(
                    row["payment_methods_user_will_consider"]
                ),
                max_installment_months=_parse_int(row["max_installment_months"]),
            )
    return profiles


def load_events() -> dict[str, list[Event]]:
    """Returns events grouped by user_id, in file order (not yet resolved/deduped)."""
    by_user: dict[str, list[Event]] = {}
    with open(config.EVENTS_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ev = Event(
                event_id=row["event_id"],
                user_id=row["user_id"],
                event_type=row["event_type"],
                description=row["description"],
                category=row["category"],
                direction=row["direction"],
                amount=_parse_float(row["amount"]),
                currency=row["currency"].strip(),
                event_date=_parse_date(row["event_date"]),
                settlement_date=_parse_date(row["settlement_date"]),
                status=row["status"].strip(),
                linked_event_id=row["linked_event_id"].strip() or None,
                flexibility=row["flexibility"].strip(),
                minimum_allowed_amount=_parse_float(row["minimum_allowed_amount"]),
            )
            by_user.setdefault(ev.user_id, []).append(ev)
    return by_user


def load_requests() -> list[Request]:
    requests: list[Request] = []
    with open(config.REQUESTS_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            requests.append(
                Request(
                    request_id=row["request_id"],
                    user_id=row["user_id"],
                    request_date=_parse_date(row["request_date"]),
                    request_type=row["request_type"],
                    requested_amount=_parse_float(row["requested_amount"]) or 0.0,
                    desired_completion_date=_parse_date(row["desired_completion_date"]),
                    allows_partial_payment=_parse_bool(row["allows_partial_payment"]),
                    request_text=row["request_text"],
                )
            )
    return requests


def load_payment_options() -> dict[str, list[PaymentOption]]:
    by_request: dict[str, list[PaymentOption]] = {}
    with open(config.PAYMENT_OPTIONS_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            opt = PaymentOption(
                payment_option_id=row["payment_option_id"],
                request_id=row["request_id"],
                payment_method=row["payment_method"].strip(),
                payment_amount=_parse_float(row["payment_amount"]) or 0.0,
                number_of_payments=_parse_int(row["number_of_payments"]) or 1,
                first_payment_date=_parse_date(row["first_payment_date"]),
                payment_frequency_days=_parse_int(row["payment_frequency_days"]),
                financing_fee=_parse_float(row["financing_fee"]) or 0.0,
                total_payable_amount=_parse_float(row["total_payable_amount"]) or 0.0,
            )
            by_request.setdefault(opt.request_id, []).append(opt)
    return by_request


def load_messages() -> dict[str, list[Message]]:
    """Returns messages grouped by user_id (includes user-level messages with blank request_id)."""
    by_user: dict[str, list[Message]] = {}
    with open(config.MESSAGES_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            msg = Message(
                message_id=row["message_id"],
                user_id=row["user_id"],
                request_id=row["request_id"].strip() or None,
                related_event_id=row["related_event_id"].strip() or None,
                sent_at=row["sent_at"],
                source_type=row["source_type"],
                message_text=row["message_text"],
            )
            by_user.setdefault(msg.user_id, []).append(msg)
    return by_user


def load_images() -> dict[str, Image]:
    """Returns images keyed by related_event_id (only rows that actually link to an event)."""
    by_event: dict[str, Image] = {}
    with open(config.IMAGES_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            img = Image(
                image_id=row["image_id"],
                user_id=row["user_id"],
                request_id=row["request_id"].strip() or None,
                related_event_id=row["related_event_id"].strip() or None,
            )
            if img.related_event_id:
                by_event[img.related_event_id] = img
    return by_event


def load_exchange_rates() -> dict[tuple[date, str, str], float]:
    rates: dict[tuple[date, str, str], float] = {}
    with open(config.EXCHANGE_RATES_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (_parse_date(row["rate_date"]), row["from_currency"].strip(), row["to_currency"].strip())
            rates[key] = _parse_float(row["rate"]) or 0.0
    return rates
