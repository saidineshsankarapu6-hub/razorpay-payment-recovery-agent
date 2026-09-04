"""Feature engineering from observable history only.

The two signals that matter most for recovery -- when a customer will have
money, and whether their bank is currently broken -- are both latent in the
simulator. The agent cannot read either. This module reconstructs them from
what a payment processor genuinely observes.

Customer payday
---------------
Never read from `_salary_day`. Estimated from the shape of a customer's own
transaction outcomes: success probability rises just after payday and decays
across the month, so the payday is the offset whose implied liquidity curve
best explains which of that customer's payments succeeded. Estimated from the
TRAINING WINDOW ONLY, and its accuracy is measured rather than assumed
(see scripts/train_agent.py).

Issuer stress
-------------
Never read from `_in_downtime`. Computed as the failure rate on that issuer in
the two hours BEFORE the payment -- strictly backward-looking, so it uses
information a processor would actually hold at decision time. This is a real
signal, not a proxy for a hidden flag: when a bank is down, its other customers
are failing too, and that is visible.

Everything here is computed from columns in OBSERVABLE_FIELDS plus prior
transaction outcomes. No underscore-prefixed column is ever read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .generator import Action
from .taxonomy import (ALL_CODES, OPAQUE_CODES, RecoveryClass, classify,
                       normalise_code)

# Decay constant of the assumed liquidity curve. This mirrors the generator's
# salary-cycle shape. Treating it as a known functional form is a modelling
# choice, not leakage: it is a single global constant, it encodes the ordinary
# domain fact that people are flush after payday and thin before it, and it
# reveals nothing about any individual customer's payday -- which is exactly
# what has to be estimated.
LIQUIDITY_TAU = 11.0
CYCLE_DAYS = 30
ISSUER_STRESS_WINDOW_H = 2.0


def day_of_month(hour: np.ndarray) -> np.ndarray:
    return (hour // 24) % CYCLE_DAYS + 1


def liquidity_curve(day: np.ndarray, payday: np.ndarray) -> np.ndarray:
    """Implied liquidity for a given day-of-month under an assumed payday."""
    return np.exp(-((day - payday) % CYCLE_DAYS) / LIQUIDITY_TAU)


# ---------------------------------------------------------------------------
def estimate_paydays(train_txns: pd.DataFrame) -> pd.DataFrame:
    """Estimate each customer's payday from their observed success pattern.

    For every candidate payday 1..30 we score how well the implied liquidity
    curve separates that customer's successes from their failures, and take the
    best-scoring candidate. Customers with too little history fall back to the
    population mode, which is the honest behaviour for a cold-start customer.

    Returns one row per customer: payday_hat, confidence, n_txns, success_rate.
    """
    df = train_txns[["customer_id", "hour", "success"]].copy()
    df["dom"] = day_of_month(df.hour.values)

    customers = np.sort(df.customer_id.unique())
    cand = np.arange(1, CYCLE_DAYS + 1)

    # curve[d-1, p-1] = implied liquidity on day d if payday were p
    curve = liquidity_curve(cand[:, None].astype(float), cand[None, :].astype(float))

    payday_hat = np.zeros(len(customers), dtype=int)
    confidence = np.zeros(len(customers))
    n_txns = np.zeros(len(customers), dtype=int)
    succ_rate = np.zeros(len(customers))

    grouped = df.groupby("customer_id", sort=True)
    for i, (cid, g) in enumerate(grouped):
        dom = g.dom.values.astype(int)
        s = g.success.values.astype(float)
        n_txns[i] = len(g)
        succ_rate[i] = s.mean()

        if len(g) < 12 or s.std() == 0:
            payday_hat[i] = -1          # insufficient history; filled in below
            continue

        # Score each candidate payday by the point-biserial correlation between
        # implied liquidity and observed success.
        liq = curve[dom - 1, :]                     # (n_txns, 30)
        liq_c = liq - liq.mean(axis=0, keepdims=True)
        s_c = s - s.mean()
        denom = np.sqrt((liq_c ** 2).sum(axis=0) * (s_c ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(denom > 0, (liq_c * s_c[:, None]).sum(axis=0) / denom, 0.0)

        best = int(np.argmax(corr))
        payday_hat[i] = cand[best]
        confidence[i] = float(corr[best])

    out = pd.DataFrame({
        "customer_id": customers,
        "payday_hat": payday_hat,
        "payday_confidence": confidence,
        "cust_n_txns": n_txns,
        "cust_success_rate": succ_rate,
    })

    # Cold-start fallback: the population's most common estimate.
    known = out.payday_hat[out.payday_hat > 0]
    fallback = int(known.mode().iloc[0]) if len(known) else 1
    out.loc[out.payday_hat < 0, "payday_hat"] = fallback
    return out


def build_customer_profiles(train_txns: pd.DataFrame) -> pd.DataFrame:
    """Per-customer features, all from training-window observations."""
    prof = estimate_paydays(train_txns)

    agg = train_txns.groupby("customer_id").agg(
        cust_mean_amount=("amount", "mean"),
        cust_n_methods=("method", "nunique"),     # observable proxy for "has an alternative"
    ).reset_index()

    return prof.merge(agg, on="customer_id", how="left")


def build_issuer_stress(all_txns: pd.DataFrame) -> pd.Series:
    """Failure rate on each payment's issuer in the 2h BEFORE it.

    Strictly backward-looking, so this is information a processor holds at
    decision time. Indexed by txn_id.
    """
    df = all_txns[["txn_id", "issuer_id", "hour", "success"]].sort_values(
        ["issuer_id", "hour"]).reset_index(drop=True)

    stress = np.zeros(len(df))
    for _, g in df.groupby("issuer_id", sort=False):
        h = g.hour.values
        fails = (~g.success.values).astype(float)
        cum_fail = np.concatenate([[0.0], np.cumsum(fails)])
        idx = np.arange(len(g))
        lo = np.searchsorted(h, h - ISSUER_STRESS_WINDOW_H, side="left")
        n = idx - lo
        f = cum_fail[idx] - cum_fail[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            stress[g.index.values] = np.where(n > 0, f / np.maximum(n, 1), 0.0)

    return pd.Series(stress, index=df.txn_id.values, name="issuer_stress")


# ---------------------------------------------------------------------------
@dataclass
class FeatureContext:
    """Everything needed to featurise a candidate action. Built once."""
    profiles: dict          # customer_id -> profile dict
    issuer_stress: dict     # txn_id -> float
    fallback_profile: dict


def make_context(train_txns: pd.DataFrame, all_txns: pd.DataFrame) -> FeatureContext:
    prof = build_customer_profiles(train_txns)
    stress = build_issuer_stress(all_txns)

    fallback = {
        "payday_hat": int(prof.payday_hat.mode().iloc[0]),
        "payday_confidence": 0.0,
        "cust_n_txns": 0,
        "cust_success_rate": float(prof.cust_success_rate.mean()),
        "cust_mean_amount": float(prof.cust_mean_amount.mean()),
        "cust_n_methods": 1,
    }
    cols = list(fallback.keys())
    return FeatureContext(
        profiles=prof.set_index("customer_id")[cols].to_dict("index"),
        issuer_stress=stress.to_dict(),
        fallback_profile=fallback,
    )


FEATURE_COLUMNS = [
    # categoricals first, so the model's categorical mask is a simple prefix
    "method", "category", "error_code", "recovery_class", "action",
    # numerics
    "amount", "log_amount", "amount_vs_cust_mean",
    "fail_hour_of_day", "fail_day_of_month", "is_opaque",
    "cust_n_txns", "cust_success_rate", "cust_n_methods",
    "payday_hat", "payday_confidence",
    "issuer_stress",
    "delay_h", "log_delay",
    "action_day_of_month", "est_liquidity_at_action", "days_to_est_payday",
    "attempt_index", "n_prior_retries", "n_prior_nudges", "n_prior_switches",
]

N_CATEGORICAL = 5
CATEGORICAL_MASK = np.array(
    [i < N_CATEGORICAL for i in range(len(FEATURE_COLUMNS))], dtype=bool)

# Fixed vocabularies. Encoding categoricals as integer codes lets the whole
# feature path be numpy: the agent scores ~18 candidate actions per decision,
# hundreds of thousands of times, and building a pandas frame per call
# dominated the runtime by an order of magnitude.
_METHOD_VOCAB = ("card", "upi", "netbanking", "wallet")
_CATEGORY_VOCAB = ("ecommerce", "saas_subscription", "utility", "travel", "edtech")
_ERROR_VOCAB = tuple(sorted(ALL_CODES))
_CLASS_VOCAB = tuple(c.value for c in RecoveryClass)
_ACTION_VOCAB = tuple(a.value for a in Action)


def _index(vocab: tuple) -> dict:
    """Map value -> code, with one extra code reserved for unseen values."""
    return {v: i for i, v in enumerate(vocab)}


_METHOD_IX, _CATEGORY_IX = _index(_METHOD_VOCAB), _index(_CATEGORY_VOCAB)
_ERROR_IX, _CLASS_IX = _index(_ERROR_VOCAB), _index(_CLASS_VOCAB)
_ACTION_IX = _index(_ACTION_VOCAB)

_UNKNOWN = {
    "method": len(_METHOD_VOCAB), "category": len(_CATEGORY_VOCAB),
    "error_code": len(_ERROR_VOCAB), "recovery_class": len(_CLASS_VOCAB),
    "action": len(_ACTION_VOCAB),
}


def _static_features(rec: dict, ctx: FeatureContext) -> dict:
    """The part of the feature vector that does not depend on the candidate."""
    prof = ctx.profiles.get(rec["customer_id"], ctx.fallback_profile)
    amount = float(rec["amount"])
    fail_hour = float(rec["hour"])
    code = normalise_code(rec["error_code"])
    return {
        "method": _METHOD_IX.get(rec["method"], _UNKNOWN["method"]),
        "category": _CATEGORY_IX.get(rec["category"], _UNKNOWN["category"]),
        "error_code": _ERROR_IX.get(code, _UNKNOWN["error_code"]),
        "recovery_class": _CLASS_IX.get(classify(code).value, _UNKNOWN["recovery_class"]),
        "amount": amount,
        "log_amount": np.log1p(amount),
        "amount_vs_cust_mean": amount / max(prof["cust_mean_amount"], 1.0),
        "fail_hour_of_day": fail_hour % 24,
        "fail_day_of_month": float((fail_hour // 24) % CYCLE_DAYS + 1),
        "is_opaque": float(code in OPAQUE_CODES),
        "cust_n_txns": float(prof["cust_n_txns"]),
        "cust_success_rate": float(prof["cust_success_rate"]),
        "cust_n_methods": float(prof["cust_n_methods"]),
        "payday_hat": float(prof["payday_hat"]),
        "payday_confidence": float(prof["payday_confidence"]),
        "issuer_stress": float(ctx.issuer_stress.get(rec["txn_id"], 0.0)),
        "_fail_hour": fail_hour,
    }


def _history_counts(history) -> tuple[float, float, float, float]:
    n_retry = n_nudge = n_switch = 0
    for a in history:
        if a.action is Action.RETRY:
            n_retry += 1
        elif a.action is Action.NUDGE:
            n_nudge += 1
        elif a.action is Action.SWITCH_METHOD:
            n_switch += 1
    return float(len(history)), float(n_retry), float(n_nudge), float(n_switch)


def featurise_candidates(
    rec: dict,
    history,
    ctx: FeatureContext,
    actions: list[Action],
    delays: np.ndarray,
) -> np.ndarray:
    """Fast path: one payment, many candidate actions.

    Static features are computed once and broadcast; only the action-dependent
    columns vary across rows. This is the agent's inner loop.
    """
    st = _static_features(rec, ctx)
    n = len(actions)
    d = np.asarray(delays, dtype=float)

    fail_hour = st["_fail_hour"]
    payday = st["payday_hat"]
    act_dom = ((fail_hour + d) // 24) % CYCLE_DAYS + 1

    attempt_idx, n_retry, n_nudge, n_switch = _history_counts(history)

    X = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float64)
    for j, col in enumerate(FEATURE_COLUMNS):
        if col == "action":
            X[:, j] = [_ACTION_IX.get(a.value, _UNKNOWN["action"]) for a in actions]
        elif col == "delay_h":
            X[:, j] = d
        elif col == "log_delay":
            X[:, j] = np.log1p(d)
        elif col == "action_day_of_month":
            X[:, j] = act_dom
        elif col == "est_liquidity_at_action":
            X[:, j] = liquidity_curve(act_dom, np.full(n, payday))
        elif col == "days_to_est_payday":
            X[:, j] = (payday - act_dom) % CYCLE_DAYS
        elif col == "attempt_index":
            X[:, j] = attempt_idx
        elif col == "n_prior_retries":
            X[:, j] = n_retry
        elif col == "n_prior_nudges":
            X[:, j] = n_nudge
        elif col == "n_prior_switches":
            X[:, j] = n_switch
        else:
            X[:, j] = st[col]
    return X


def featurise(
    records: list[dict],
    actions: list[Action],
    delays: list[float],
    histories: list[list],
    ctx: FeatureContext,
) -> np.ndarray:
    """Batch path, used to build the training matrix."""
    n = len(records)
    X = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float64)
    col_ix = {c: i for i, c in enumerate(FEATURE_COLUMNS)}

    for i, (rec, action, delay, hist) in enumerate(
            zip(records, actions, delays, histories)):
        st = _static_features(rec, ctx)
        payday = st["payday_hat"]
        act_dom = float((st["_fail_hour"] + delay) // 24 % CYCLE_DAYS + 1)
        attempt_idx, n_retry, n_nudge, n_switch = _history_counts(hist)

        for col, j in col_ix.items():
            if col == "action":
                X[i, j] = _ACTION_IX.get(action.value, _UNKNOWN["action"])
            elif col == "delay_h":
                X[i, j] = delay
            elif col == "log_delay":
                X[i, j] = np.log1p(delay)
            elif col == "action_day_of_month":
                X[i, j] = act_dom
            elif col == "est_liquidity_at_action":
                X[i, j] = float(liquidity_curve(
                    np.array([act_dom]), np.array([payday]))[0])
            elif col == "days_to_est_payday":
                X[i, j] = (payday - act_dom) % CYCLE_DAYS
            elif col == "attempt_index":
                X[i, j] = attempt_idx
            elif col == "n_prior_retries":
                X[i, j] = n_retry
            elif col == "n_prior_nudges":
                X[i, j] = n_nudge
            elif col == "n_prior_switches":
                X[i, j] = n_switch
            else:
                X[i, j] = st[col]
    return X
