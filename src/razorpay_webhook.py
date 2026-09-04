"""Ingest a real Razorpay `payment.failed` webhook.

Everything else in this project runs on a simulator. This module is the seam
where it meets Razorpay's actual rails: it takes the JSON Razorpay POSTs to a
merchant endpoint and turns it into the observation the agent consumes.

Two details matter, and both are places a naive integration goes wrong.

1. `error_code` IS NOT THE ERROR.
   Razorpay's error object separates a broad category from the specific cause:

       "error_code":   "BAD_REQUEST_ERROR"      <- category, nearly useless
       "error_reason": "invalid_otp"            <- the actual cause

   An integration that keys off `error_code` sees `BAD_REQUEST_ERROR` for a
   wrong OTP, an expired card and an empty account alike. The taxonomy in this
   project is built on `error_reason`, so that is what we read, falling back to
   `error_code` only when no reason is supplied.
   Reference: https://razorpay.com/docs/errors/

2. AMOUNTS ARE IN PAISE.
   Razorpay sends 100000 for INR 1,000. Reading it as rupees inflates every
   expected value by 100x and would make the agent chase everything.

Signature verification is included because a recovery agent acting on an
unverified webhook is a way to be driven by anyone who can find the URL.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

from .taxonomy import classify, is_known, is_opaque, normalise_code

SUPPORTED_EVENT = "payment.failed"

# Razorpay's broad categories. Seeing one of these in `error_reason` means the
# payload gave us nothing specific to work with.
BROAD_CODES = frozenset({
    "bad_request_error", "gateway_error", "server_error",
})


class WebhookError(ValueError):
    """The payload is not something we can act on."""


def verify_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """Confirm the payload really came from Razorpay.

    HMAC-SHA256 over the exact raw request body, compared in constant time.
    Must run against the bytes as received -- re-serialising parsed JSON changes
    key order and whitespace, and the digest no longer matches.
    """
    if not signature or not secret:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body,
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _payment_entity(payload: dict) -> dict:
    """Pull the payment entity out, tolerating both shapes seen in the wild."""
    if not isinstance(payload, dict):
        raise WebhookError("payload is not a JSON object")

    event = payload.get("event")
    if event and event != SUPPORTED_EVENT:
        raise WebhookError(f"unsupported event: {event}")

    node = payload.get("payload", {})
    if isinstance(node, dict):
        pay = node.get("payment", {})
        if isinstance(pay, dict) and isinstance(pay.get("entity"), dict):
            return pay["entity"]

    # Some integrations forward the payment entity on its own.
    if payload.get("entity") == "payment":
        return payload

    raise WebhookError("no payment entity found in payload")


def _reason(entity: dict) -> str:
    """The specific failure reason, however this payload happens to carry it.

    Prefers `error_reason`. Falls back to a nested error object, then to
    `error_code` -- and if all we are left with is a broad category, says so
    rather than pretending it is a real code.
    """
    err = entity.get("error") if isinstance(entity.get("error"), dict) else {}

    for value in (entity.get("error_reason"), err.get("reason"),
                  entity.get("error_code"), err.get("code")):
        code = normalise_code(value)
        if code and code not in BROAD_CODES:
            return code

    return "unspecified_bank_decline"


def parse_payment_failed(payload: dict) -> dict[str, Any]:
    """Turn a `payment.failed` webhook into fields the agent understands.

    Returns the observation plus the context a caller needs for logging: the
    Razorpay payment id, the raw error fields, and whether the reason is one we
    recognise.
    """
    entity = _payment_entity(payload)

    raw_amount = entity.get("amount")
    if raw_amount is None:
        raise WebhookError("payment entity has no amount")
    try:
        amount_paise = int(raw_amount)
    except (TypeError, ValueError):
        raise WebhookError(f"amount is not an integer: {raw_amount!r}")
    if amount_paise <= 0:
        raise WebhookError(f"amount must be positive, got {amount_paise}")

    currency = (entity.get("currency") or "INR").upper()
    if currency != "INR":
        raise WebhookError(f"only INR is handled, got {currency}")

    method = normalise_code(entity.get("method")) or "card"
    if method in ("emi", "cardless_emi", "paylater"):
        method = "card"          # these settle over card rails
    if method not in ("card", "upi", "netbanking", "wallet"):
        method = "card"

    reason = _reason(entity)

    return {
        "payment_id": entity.get("id"),
        "order_id": entity.get("order_id"),
        "amount": amount_paise / 100.0,          # paise -> rupees
        "amount_paise": amount_paise,
        "currency": currency,
        "method": method,
        "error_code": reason,                    # what the taxonomy keys on
        "error_source": entity.get("error_source"),
        "error_step": entity.get("error_step"),
        "error_description": entity.get("error_description"),
        "raw_error_code": entity.get("error_code"),
        "created_at": entity.get("created_at"),
        "recognised": bool(is_known(reason)),
        "recovery_class": classify(reason).value,
        "opaque": bool(is_opaque(reason)),
    }
