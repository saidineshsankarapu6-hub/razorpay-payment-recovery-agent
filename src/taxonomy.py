"""Failure taxonomy for Razorpay payment errors.

Every error code here is a real string from Razorpay's public docs:
  https://razorpay.com/docs/errors/payments/cards/
  https://razorpay.com/docs/errors/payments/upi/

The mapping from code -> RecoveryClass is our inference layer: it collapses
27 raw codes into 7 recovery intents that imply different actions.

The critical design point is OPAQUE_CODES. Razorpay documents several codes
that mean only "the issuer refused, reason undisclosed". A rules engine must
pick one fixed action for that whole bucket. Our agent infers the likely
underlying cause instead -- that is the thesis of this project.
"""

from enum import Enum


class RecoveryClass(str, Enum):
    SOFT_TRANSIENT = "SOFT_TRANSIENT"  # infra/timing failed, instrument fine
    FUNDS = "FUNDS"                    # instrument fine, money absent
    AUTH_FAILED = "AUTH_FAILED"        # customer failed authentication
    METHOD_BROKEN = "METHOD_BROKEN"    # instrument unusable as-is
    HARD_DECLINE = "HARD_DECLINE"      # issuer said no, reason opaque
    FRAUD = "FRAUD"                    # risk decline -- never retry
    ABANDONED = "ABANDONED"            # customer walked away


class Method(str, Enum):
    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"


# --- Card codes (razorpay.com/docs/errors/payments/cards/) -------------------
CARD_CODES = {
    "gateway_technical_error":           RecoveryClass.SOFT_TRANSIENT,
    "bank_technical_error":              RecoveryClass.SOFT_TRANSIENT,
    "payment_timed_out":                 RecoveryClass.SOFT_TRANSIENT,
    "insufficient_funds":                RecoveryClass.FUNDS,
    "authentication_failed":             RecoveryClass.AUTH_FAILED,
    "incorrect_cvv":                     RecoveryClass.AUTH_FAILED,
    "payment_cancelled":                 RecoveryClass.ABANDONED,
    "card_expired":                      RecoveryClass.METHOD_BROKEN,
    "card_not_enrolled":                 RecoveryClass.METHOD_BROKEN,
    "card_disabled_for_online_payments": RecoveryClass.METHOD_BROKEN,
    "debit_instrument_inactive":         RecoveryClass.METHOD_BROKEN,
    "debit_instrument_blocked":          RecoveryClass.METHOD_BROKEN,
    "transaction_limit_exceeded":        RecoveryClass.METHOD_BROKEN,
    "card_declined":                     RecoveryClass.HARD_DECLINE,
    "payment_failed":                    RecoveryClass.HARD_DECLINE,
    "payment_risk_check_failed":         RecoveryClass.FRAUD,
}

# --- UPI codes (razorpay.com/docs/errors/payments/upi/) ---------------------
UPI_CODES = {
    "bank_technical_error":            RecoveryClass.SOFT_TRANSIENT,
    "gateway_technical_error":         RecoveryClass.SOFT_TRANSIENT,
    "payment_timed_out":               RecoveryClass.SOFT_TRANSIENT,
    "payment_collect_request_expired": RecoveryClass.ABANDONED,
    "payment_cancelled":               RecoveryClass.ABANDONED,
    "insufficient_funds":              RecoveryClass.FUNDS,
    "payment_declined":                RecoveryClass.HARD_DECLINE,
    "credit_failed":                   RecoveryClass.HARD_DECLINE,
    "invalid_vpa":                     RecoveryClass.METHOD_BROKEN,
    "vpa_resolution_failed":           RecoveryClass.METHOD_BROKEN,
    "customer_bank_account_mismatch":  RecoveryClass.METHOD_BROKEN,
}

ALL_CODES = {**CARD_CODES, **UPI_CODES}

# Codes that disclose nothing about the underlying cause. These are the codes
# the agent must reason about rather than look up.
OPAQUE_CODES = frozenset({
    "card_declined",
    "payment_failed",
    "payment_declined",
    "credit_failed",
})


def normalise_code(error_code) -> str:
    """Canonicalise an error code before any lookup.

    The code arrives from an issuer via a gateway. A processor does not control
    its exact bytes, and case or padding differences are ordinary in practice.
    Without this, `PAYMENT_RISK_CHECK_FAILED` and ` payment_risk_check_failed`
    both miss the FRAUD mapping and defeat the single most important guardrail
    in the system -- a whitespace character would be enough.

    Non-string inputs are coerced rather than raised on: an unhashable value
    reaching a dict lookup would otherwise take down recovery for every payment
    behind it, and an unrecognisable code is exactly what the HARD_DECLINE
    fallback exists for.
    """
    if isinstance(error_code, bytes):
        error_code = error_code.decode("utf-8", "replace")
    elif not isinstance(error_code, str):
        error_code = "" if error_code is None else str(error_code)
    return error_code.strip().lower()


def is_known(error_code) -> bool:
    """True if this code appears in Razorpay's documented set.

    Unknown codes are not safe codes. Issuers add them, and a new RISK code
    would be invisible to `never_retry` until someone triages it into the
    taxonomy -- so callers should hold unknown codes to a higher bar rather
    than treating them like any other hard decline.
    """
    return normalise_code(error_code) in ALL_CODES


def classify(error_code) -> RecoveryClass:
    """Map a Razorpay error code to its recovery class.

    Unknown codes are treated as HARD_DECLINE: unrecognised does not mean safe,
    and HARD_DECLINE routes to inference rather than to a blind retry.
    """
    return ALL_CODES.get(normalise_code(error_code), RecoveryClass.HARD_DECLINE)


def is_opaque(error_code) -> bool:
    """True if the code hides its cause and requires inference."""
    return normalise_code(error_code) in OPAQUE_CODES


def never_retry(error_code) -> bool:
    """Hard guardrail. Retrying a risk decline damages merchant standing with
    the issuer, so this is enforced independently of any model output."""
    return classify(error_code) is RecoveryClass.FRAUD
