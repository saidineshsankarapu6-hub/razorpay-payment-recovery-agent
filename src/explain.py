"""Turn the agent's decision into an explanation a merchant can read.

The agent decides by expected value over 18 candidate actions. That is the
right way to decide and the wrong way to explain: nobody operating a business
wants a probability table, they want to know what happened to their money and
what is being done about it.

Everything here is derived from values the agent actually used -- the pushback
score, the issuer stress feature, the estimated payday, the chosen action. No
cause is invented. When the evidence does not clearly point anywhere, the
explanation says so rather than dressing up a guess.
"""

from __future__ import annotations

from .features import CYCLE_DAYS, day_of_month
from .generator import Action
from .taxonomy import RecoveryClass, classify, is_opaque

# Thresholds for narrating a signal. Deliberately conservative: a confident
# wrong explanation is worse than an honest vague one.
PUSHBACK_STRONG = 0.30
STRESS_STRONG = 0.30
PAYDAY_TRUSTED = 0.20


# Plain-English meaning for every documented code, so nobody has to know what
# `debit_instrument_inactive` is. Wording follows Razorpay's own docs.
CODE_MEANING = {
    "gateway_technical_error": "A technical fault while processing",
    "bank_technical_error": "The customer's bank was down",
    "payment_timed_out": "The customer ran out of time to pay",
    "insufficient_funds": "Not enough money in the account",
    "authentication_failed": "The customer got the OTP wrong, or closed the page",
    "incorrect_cvv": "The customer typed the wrong CVV",
    "payment_cancelled": "The customer cancelled or backed out",
    "card_expired": "The card has expired",
    "card_not_enrolled": "The card is not switched on for online payments",
    "card_disabled_for_online_payments": "Online payments are turned off on this card",
    "debit_instrument_inactive": "The card is not activated for online use",
    "debit_instrument_blocked": "The card is blocked",
    "transaction_limit_exceeded": "The customer hit their daily limit",
    "card_declined": "The bank said no, without saying why",
    "payment_failed": "The bank said no, without saying why",
    "payment_risk_check_failed": "The bank's fraud system blocked it",
    "payment_collect_request_expired": "The UPI request expired before they paid",
    "payment_declined": "The bank said no, without saying why",
    "credit_failed": "The bank said no, without saying why",
    "invalid_vpa": "That UPI ID is not usable",
    "vpa_resolution_failed": "Their UPI ID could not be reached",
    "customer_bank_account_mismatch": "They picked a different bank account",
}

# The simulator's ground truth, in words. Shown to the viewer for verification,
# never to any policy.
CAUSE_PLAIN = {
    "NO_FUNDS": "The customer genuinely had no money",
    "ISSUER_DOWN": "Their bank was down",
    "GATEWAY_ISSUE": "A temporary technical fault",
    "AUTH_FAIL": "The customer failed authentication",
    "INSTRUMENT_STATE": "Their card or UPI ID was unusable",
    "LIMIT_EXCEEDED": "They hit their daily limit",
    "RISK_BLOCK": "The bank's fraud system blocked it",
    "CUSTOMER_ABANDON": "The customer walked away",
}


def code_meaning(code: str) -> str:
    from .taxonomy import normalise_code
    return CODE_MEANING.get(normalise_code(code), "An unrecognised bank response")


def cause_plain(cause: str) -> str:
    return CAUSE_PLAIN.get(cause, cause.replace("_", " ").lower())


def _money(x: float) -> str:
    return f"₹{x:,.0f}"


def diagnose(rec, decision, p_recover, p_pushback, prof, issuer_stress) -> dict:
    """A plain-English read on why this payment failed and what we are doing.

    Returns: headline read, the evidence behind it, the action in plain words,
    and a confidence band.
    """
    code = rec["error_code"]
    cls = classify(code)
    opaque = is_opaque(code)

    fail_hour = float(rec["hour"])
    payday = int(prof["payday_hat"])
    payday_conf = float(prof["payday_confidence"])

    if decision is None:
        action_hours = None
        action_day = None
    else:
        action_hours = decision.at_hour - fail_hour
        action_day = int(day_of_month(
            __import__("numpy").array([decision.at_hour]))[0])

    days_to_payday = None
    if action_day is not None:
        days_to_payday = (payday - int(day_of_month(
            __import__("numpy").array([fail_hour]))[0])) % CYCLE_DAYS

    # ---- the read -------------------------------------------------------
    if decision is None and cls is RecoveryClass.FRAUD:
        return {
            "read": "The bank's fraud system blocked this.",
            "evidence": [
                f"The bank returned {code}, which states this outright.",
            ],
            "action": "Not touching this one. No retry, no message.",
            "rationale": "Retrying a fraud block never recovers the money and "
                         "counts against the merchant with the bank.",
            "confidence": "certain",
            "tone": "blocked",
        }

    if p_pushback >= PUSHBACK_STRONG:
        return {
            "read": "This looks like a fraud block, not a payment problem.",
            "evidence": [
                f"The bank only said {code} — it did not tell us why."
                if opaque else f"The bank returned {code}.",
                f"Payments that look like this draw bank pushback "
                f"{p_pushback:.0%} of the time.",
            ],
            "action": "Not sending this back to the bank. Messaging the "
                      "customer instead." if decision is not None
                      else "Not pursuing this one.",
            "rationale": "Pushing a blocked payment back to the bank earns "
                         "nothing and damages the merchant's standing.",
            "confidence": "high" if p_pushback > 0.5 else "moderate",
            "tone": "blocked",
        }

    if issuer_stress >= STRESS_STRONG:
        return {
            "read": "The customer's bank is having problems right now.",
            "evidence": [
                f"{issuer_stress:.0%} of payments through this bank failed in "
                f"the last two hours.",
                "Nothing appears wrong with the customer's card.",
            ],
            "action": (f"Giving the bank time to recover, then trying again "
                       f"{_hours(action_hours)}."
                       if decision is not None else "Holding off for now."),
            "rationale": "Retrying into an outage just burns attempts. The "
                         "payment will most likely go through once it clears.",
            "confidence": "high",
            "tone": "wait",
        }

    # A funds read is only permissible where the code is actually consistent
    # with a funds problem: an explicit insufficient-funds code, or an opaque
    # one that could be hiding it. Without this gate the timing heuristic alone
    # would explain a bank outage as "the customer is short of money" -- which
    # it did, and which is worse than saying nothing.
    funds_plausible = cls is RecoveryClass.FUNDS or (
        opaque and cls is RecoveryClass.HARD_DECLINE)

    if (funds_plausible and decision is not None
            and decision.action is Action.RETRY
            and action_hours is not None and action_hours >= 48
            and payday_conf >= PAYDAY_TRUSTED and days_to_payday is not None
            and days_to_payday <= 12):
        return {
            "read": "The customer is most likely short of money.",
            "evidence": [
                f"The bank said {code}, which does not say why."
                if opaque else f"The bank said {code}.",
                f"This customer's payments succeed far more often just after "
                f"the {_ordinal(payday)} — that looks like their payday.",
                f"Their next one is about {days_to_payday} day"
                f"{'s' if days_to_payday != 1 else ''} away.",
            ],
            "action": f"Waiting until around their payday, then retrying.",
            "rationale": "Retrying today would almost certainly fail again. "
                         "Waiting for money to land is what actually recovers "
                         "this kind of failure.",
            "confidence": "moderate" if payday_conf < 0.35 else "high",
            "tone": "wait",
        }

    if decision is not None and decision.action is Action.SWITCH_METHOD:
        return {
            "read": "The customer's payment method looks unusable.",
            "evidence": [
                f"The bank returned {code}.",
                "Retrying the same card is unlikely to change anything.",
            ],
            "action": "Offering them a different way to pay.",
            "rationale": "A broken or blocked instrument does not fix itself. "
                         "Switching rails is the only thing that can work.",
            "confidence": "moderate",
            "tone": "act",
        }

    if decision is not None and decision.action is Action.NUDGE:
        return {
            "read": "This one needs the customer to do something.",
            "evidence": [
                f"The bank returned {code}.",
                "A silent retry does not fix a customer who abandoned "
                "checkout or failed their OTP.",
            ],
            "action": f"Sending them a reminder with a payment link "
                      f"{_hours(action_hours)}.",
            "rationale": "Nothing here can be fixed on our side, and a "
                         "message costs a fraction of a retry.",
            "confidence": "moderate",
            "tone": "act",
        }

    if decision is not None and decision.action is Action.RETRY:
        soon = action_hours is not None and action_hours <= 12
        if cls is RecoveryClass.SOFT_TRANSIENT:
            read = ("A technical glitch on the bank or gateway side."
                    if soon else
                    "A technical failure on the bank side, not a problem with "
                    "the customer.")
        elif cls is RecoveryClass.FUNDS:
            read = "The customer did not have the money at that moment."
        else:
            read = ("This looks like a temporary glitch." if soon
                    else "This looks recoverable, but not yet.")
        return {
            "read": read,
            "evidence": [
                f"The bank returned {code}.",
                f"Payments like this one recover {p_recover:.0%} of the time "
                f"when retried at this point.",
            ],
            "action": f"Retrying {_hours(action_hours)}.",
            "rationale": ("Short-lived failures usually clear on their own."
                          if soon else
                          "Trying again immediately would most likely fail. "
                          "This is the moment with the best odds."),
            "confidence": "high" if p_recover > 0.6 else "moderate",
            "tone": "retry",
        }

    return {
        "read": "Not worth chasing.",
        "evidence": [
            f"The bank returned {code}.",
            f"Best odds of recovering this are only {p_recover:.0%}.",
        ],
        "action": "Stopping here.",
        "rationale": "Every attempt costs money. Chasing this one would cost "
                     "more than it is likely to bring back.",
        "confidence": "moderate",
        "tone": "blocked",
    }


def _hours(h) -> str:
    if h is None:
        return "shortly"
    if h < 1:
        return "in about half an hour"
    if h < 24:
        return f"in {int(round(h))} hour{'s' if round(h) != 1 else ''}"
    days = h / 24
    if days < 1.5:
        return "tomorrow"
    return f"in {int(round(days))} days"


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def outcome_line(recovered: bool, amount: float, cost: float) -> str:
    if recovered:
        return f"Recovered {_money(amount)} for {_money(cost)}."
    return f"Not recovered. Spent {_money(cost)} trying."
