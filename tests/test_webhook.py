"""Tests for the Razorpay webhook adapter.

These run against payloads shaped like the ones Razorpay actually POSTs, and
cover the two mistakes that would quietly break a real integration: reading the
broad `error_code` instead of the specific `error_reason`, and reading paise as
rupees.

Run:  python -m tests.test_webhook
"""

from __future__ import annotations

import hashlib
import hmac
import json

from src.razorpay_webhook import (WebhookError, parse_payment_failed,
                                  verify_signature)
from src.taxonomy import RecoveryClass

FAILURES: list[str] = []
PASSES = 0


def check(name, cond, detail=""):
    global PASSES
    if cond:
        PASSES += 1
        print(f"[PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        print(f"[FAIL] {name}" + (f" -- {detail}" if detail else ""))
        FAILURES.append(name)


def webhook(**over) -> dict:
    """A payment.failed payload shaped like Razorpay's."""
    entity = {
        "id": "pay_29QQoUBi66xm2f",
        "entity": "payment",
        "amount": 100000,                  # paise -> INR 1,000
        "currency": "INR",
        "status": "failed",
        "order_id": "order_9A33XWu170gUtm",
        "method": "card",
        "error_code": "BAD_REQUEST_ERROR",
        "error_description": "Payment failed because of insufficient funds",
        "error_source": "bank",
        "error_step": "payment_authorization",
        "error_reason": "insufficient_funds",
        "created_at": 1757000000,
    }
    entity.update(over)
    return {"entity": "event", "account_id": "acc_BFQ7uQEaa7j2z7",
            "event": "payment.failed", "contains": ["payment"],
            "payload": {"payment": {"entity": entity}},
            "created_at": 1757000000}


# --------------------------------------------------------------------------
def test_amount_is_converted_from_paise():
    """Razorpay sends paise. Reading it as rupees inflates every value 100x."""
    p = parse_payment_failed(webhook(amount=100000))
    check("100000 paise is read as INR 1,000", p["amount"] == 1000.0,
          f"got {p['amount']}")
    check("the original paise figure is preserved for logging",
          p["amount_paise"] == 100000)

    p2 = parse_payment_failed(webhook(amount=1))
    check("1 paise does not round to zero", p2["amount"] == 0.01)


def test_specific_reason_beats_broad_code():
    """The whole point: `error_code` is a category, `error_reason` is the cause.

    An integration keying off error_code sees BAD_REQUEST_ERROR for a wrong
    OTP, a dead card and an empty account alike.
    """
    p = parse_payment_failed(webhook(
        error_code="BAD_REQUEST_ERROR", error_reason="insufficient_funds"))
    check("reads error_reason, not the broad error_code",
          p["error_code"] == "insufficient_funds",
          f"got {p['error_code']}")
    check("and classifies it correctly",
          p["recovery_class"] == RecoveryClass.FUNDS.value)
    check("the broad category is still kept for the audit trail",
          p["raw_error_code"] == "BAD_REQUEST_ERROR")

    risky = parse_payment_failed(webhook(
        error_code="BAD_REQUEST_ERROR", error_reason="payment_risk_check_failed"))
    check("a fraud decline is recognised through the same path",
          risky["recovery_class"] == RecoveryClass.FRAUD.value)


def test_broad_code_alone_is_not_treated_as_a_reason():
    """If only a category arrives, say so rather than inventing a cause."""
    p = parse_payment_failed(webhook(error_code="GATEWAY_ERROR",
                                     error_reason=None))
    check("a bare category does not masquerade as a specific reason",
          p["error_code"] == "unspecified_bank_decline",
          f"got {p['error_code']}")
    check("and an unrecognised reason fails safe to HARD_DECLINE",
          p["recovery_class"] == RecoveryClass.HARD_DECLINE.value)
    check("it is reported as not recognised", p["recognised"] is False)


def test_nested_error_object_shape():
    """Some payloads carry a nested error object instead of flat fields."""
    entity = {"id": "pay_x", "entity": "payment", "amount": 250000,
              "currency": "INR", "method": "upi",
              "error": {"code": "BAD_REQUEST_ERROR", "reason": "invalid_vpa",
                        "source": "customer", "step": "payment_initiation"}}
    p = parse_payment_failed({"event": "payment.failed",
                              "payload": {"payment": {"entity": entity}}})
    check("nested error objects are handled", p["error_code"] == "invalid_vpa")
    check("and the amount still converts", p["amount"] == 2500.0)


def test_opaque_codes_are_flagged():
    p = parse_payment_failed(webhook(error_reason="payment_failed"))
    check("an opaque reason is flagged as such", p["opaque"] is True,
          f"class={p['recovery_class']}")
    q = parse_payment_failed(webhook(error_reason="card_expired"))
    check("a specific reason is not flagged opaque", q["opaque"] is False)


def test_method_normalisation():
    for sent, want in [("card", "card"), ("UPI", "upi"), ("netbanking", "netbanking"),
                       ("wallet", "wallet"), ("emi", "card"),
                       ("cardless_emi", "card"), ("something_new", "card")]:
        p = parse_payment_failed(webhook(method=sent))
        if p["method"] != want:
            check(f"method {sent!r} maps to {want!r}", False, f"got {p['method']}")
            return
    check("payment methods normalise, EMI settles as card", True)


def test_bad_payloads_are_rejected_not_guessed():
    cases = {
        "wrong event": {"event": "payment.captured", "payload": {}},
        "no payment entity": {"event": "payment.failed", "payload": {}},
        "not an object": "just a string",
        "missing amount": {"event": "payment.failed", "payload": {"payment":
            {"entity": {"entity": "payment", "currency": "INR"}}}},
        "negative amount": webhook(amount=-500),
        "zero amount": webhook(amount=0),
        "non-numeric amount": webhook(amount="lots"),
        "foreign currency": webhook(currency="USD"),
    }
    bad = []
    for label, payload in cases.items():
        try:
            parse_payment_failed(payload)
            bad.append(label)
        except WebhookError:
            pass
        except Exception as exc:
            bad.append(f"{label}({type(exc).__name__})")
    check("malformed payloads raise WebhookError rather than being guessed at",
          not bad, "; ".join(bad))


def test_signature_verification():
    secret = "whsec_test_1234"
    body = json.dumps(webhook()).encode("utf-8")
    good = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    check("a correct signature verifies", verify_signature(body, good, secret))
    check("a wrong signature is rejected",
          not verify_signature(body, "0" * 64, secret))
    check("a tampered body is rejected",
          not verify_signature(body + b" ", good, secret))
    check("an empty signature is rejected", not verify_signature(body, "", secret))
    check("a missing secret is rejected", not verify_signature(body, good, ""))


def main() -> int:
    for fn in (test_amount_is_converted_from_paise,
               test_specific_reason_beats_broad_code,
               test_broad_code_alone_is_not_treated_as_a_reason,
               test_nested_error_object_shape,
               test_opaque_codes_are_flagged,
               test_method_normalisation,
               test_bad_payloads_are_rejected_not_guessed,
               test_signature_verification):
        print(f"\n--- {fn.__name__} ---")
        fn()

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print(f"all {PASSES} webhook checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
