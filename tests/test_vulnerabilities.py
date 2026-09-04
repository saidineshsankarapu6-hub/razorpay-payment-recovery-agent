"""Vulnerability and robustness tests.

The integrity suite (test_integrity.py) checks that the evaluation is honest.
This suite is adversarial: it tries to BREAK the system. Every test here is an
attack or a hostile input, and a passing test means the attack failed.

Three classes of attack:

  A. GUARDRAIL BYPASS -- can a crafted error code, amount, or history get the
     agent to do something it is supposed to refuse? The fraud suppression is
     the crown jewel: retrying a risk decline damages a merchant's standing
     with its issuers, and that damage is not recoverable by any later fix.

  B. HOSTILE INPUT -- malformed, missing, extreme or adversarial field values.
     A payment processor does not control what an issuer returns, so anything
     reaching the taxonomy is untrusted by definition.

  C. DEPLOYMENT SURFACE -- what an operator can get wrong, and what an attacker
     could reach if part of the system were compromised.

Run:  python -m tests.test_vulnerabilities
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent import CANDIDATES, MIN_GAP_H, RecoveryAgent
from src.evaluate import MAX_ATTEMPTS, MIN_ATTEMPT_GAP_H, evaluate, load, observe
from src.generator import Action, SimConfig
from src.policies import Attempt, Decision, Policy, RuleBased
from src.taxonomy import OPAQUE_CODES, RecoveryClass, classify, never_retry

ROOT = Path(__file__).resolve().parents[1]

FINDINGS: list[tuple[str, str]] = []
PASSES = 0


def secure(name: str, condition: bool, detail: str = "", severity: str = "HIGH") -> None:
    """condition True == the attack failed == we are safe."""
    global PASSES
    if condition:
        PASSES += 1
        print(f"[SAFE] {name}" + (f" -- {detail}" if detail else ""))
    else:
        print(f"[VULN/{severity}] {name}" + (f" -- {detail}" if detail else ""))
        FINDINGS.append((severity, name))


def _agent():
    with open(ROOT / "models" / "agent.pkl", "rb") as fh:
        b = pickle.load(fh)
    cfg = SimConfig()
    return RecoveryAgent(b["model"], b["ctx"], cfg,
                         p_floor=b["p_floor"], risk_model=b.get("risk_model")), cfg


def _obs(**over):
    base = {
        "txn_id": 10_000_000, "hour": 1500.0, "customer_id": 7,
        "merchant_id": 3, "issuer_id": 2, "category": "ecommerce",
        "method": "card", "amount": 1000.0,
        "error_code": "card_declined",
    }
    base.update(over)
    return base


# =========================================================================
# A. GUARDRAIL BYPASS
# =========================================================================
def test_fraud_suppression_cannot_be_bypassed_by_string_tricks() -> None:
    """The risk-decline block is a dict lookup on an issuer-supplied string.

    A processor does not control the exact bytes an issuer or an upstream
    integration puts in that field. If casing or padding defeats the check, the
    single most important guardrail in the system is defeated by a whitespace.
    """
    variants = [
        "PAYMENT_RISK_CHECK_FAILED",
        "Payment_Risk_Check_Failed",
        " payment_risk_check_failed",
        "payment_risk_check_failed ",
        "payment_risk_check_failed\n",
        "payment_risk_check_failed\t",
    ]
    bypassed = [v for v in variants if not never_retry(v)]
    secure("fraud suppression survives case/whitespace variants of the code",
           not bypassed,
           f"{len(bypassed)}/{len(variants)} bypassed: {bypassed[:3]}")


def test_unknown_code_does_not_fail_open_for_the_agent() -> None:
    """An unrecognised code must not become a licence to hammer the issuer.

    Issuers add codes. `classify()` deliberately routes unknown codes to
    HARD_DECLINE so they reach inference rather than a blind retry -- but that
    means a NEW fraud code is, by construction, not covered by guardrail 1.
    The second guardrail (the learned pushback head plus the probability floor)
    has to carry it.
    """
    agent, cfg = _agent()
    d = agent.decide(_obs(error_code="risk_engine_declined_2027",
                          amount=50_000.0), [])
    unknown_is_retried_at_issuer = (
        d is not None and d.action in (Action.RETRY, Action.SWITCH_METHOD))
    secure("a large payment with an unknown code is not blindly sent to the issuer",
           not unknown_is_retried_at_issuer,
           f"action={d.action.value if d else None}", severity="MEDIUM")


def test_large_amount_cannot_buy_past_the_probability_floor() -> None:
    """EV scales with amount; the floor must not.

    If the guardrail were expressed purely in rupees, a big enough payment
    would always clear it and the agent would chase hopeless high-value traffic
    forever. The floor is deliberately a probability, not a value.
    """
    agent, cfg = _agent()
    escalating = []
    for amt in (1e3, 1e5, 1e7, 1e9):
        d = agent.decide(_obs(error_code="payment_risk_check_failed",
                              amount=amt), [])
        escalating.append(d)
    secure("no amount, however large, defeats the explicit fraud block",
           all(d is None for d in escalating),
           f"actions={[d.action.value if d else None for d in escalating]}")


def test_attempt_budget_cannot_be_exceeded_via_history_forgery() -> None:
    """A policy that under-reports its own history must not get extra attempts.

    The cap lives in the harness and counts real attempts, so a policy lying
    about `history` gains nothing.
    """
    class Amnesiac(Policy):
        name = "amnesiac"

        def decide(self, obs, history):
            # Always claims this is the first attempt.
            return Decision(Action.RETRY,
                            obs["hour"] + 1.0 + 2.0 * len(history), "amnesiac")

    records, sim, cfg = load()
    res, _ = evaluate(Amnesiac(), records[:400], sim, cfg)
    secure("attempt cap holds against a policy that ignores its history",
           res.attempts <= MAX_ATTEMPTS * 400,
           f"{res.attempts} attempts over 400 payments (cap {MAX_ATTEMPTS})")


def test_horizon_cannot_be_extended_by_negative_or_nan_times() -> None:
    """Time arithmetic must not be steerable into the past or into NaN.

    A NaN comparison is False, so `at_hour > deadline` would not fire and a
    NaN-scheduled attempt could slip through every bound check.
    """
    class TimeTraveller(Policy):
        name = "time_traveller"

        def decide(self, obs, history):
            return Decision(Action.RETRY, obs["hour"] - 1000.0, "past")

    class NaNScheduler(Policy):
        name = "nan_scheduler"

        def decide(self, obs, history):
            return Decision(Action.RETRY, float("nan"), "nan")

    records, sim, cfg = load()
    past, _ = evaluate(TimeTraveller(), records[:300], sim, cfg)
    secure("attempts scheduled before the failure are rejected",
           past.attempts == 0, f"attempts={past.attempts}")

    nan, _ = evaluate(NaNScheduler(), records[:300], sim, cfg)
    secure("NaN-scheduled attempts are rejected",
           nan.attempts == 0,
           f"attempts={nan.attempts} cost={nan.cost}")


def test_agent_never_proposes_a_non_advancing_time() -> None:
    """The exploit that was closed must stay closed, from the agent's side.

    The harness rejects non-advancing attempts, but an agent that keeps
    proposing them would silently lose its remaining attempts rather than plan.
    """
    agent, cfg = _agent()
    obs = _obs(amount=5000.0)
    history: list[Attempt] = []
    proposed = []
    for _ in range(MAX_ATTEMPTS):
        d = agent.decide(obs, history)
        if d is None:
            break
        proposed.append(d.at_hour)
        history.append(Attempt(d.action, d.at_hour, d.reason, False))

    gaps_ok = all(b - a >= MIN_GAP_H for a, b in zip(proposed, proposed[1:]))
    secure("the agent's own proposals always advance in time",
           gaps_ok and len(proposed) == len(set(proposed)),
           f"offsets={[round(t - obs['hour'], 1) for t in proposed]}")


# =========================================================================
# B. HOSTILE INPUT
# =========================================================================
def test_agent_survives_malformed_amounts() -> None:
    """Amount is the multiplier on every expected value.

    NaN propagates through EV and makes argmax meaningless; a negative or
    absurd amount should not crash the agent or produce a nonsense action.
    """
    agent, cfg = _agent()
    cases = {
        "zero": 0.0,
        "negative": -5000.0,
        "nan": float("nan"),
        "inf": float("inf"),
        "absurd": 1e15,
    }
    crashed, bad = [], []
    for label, amt in cases.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = agent.decide(_obs(amount=amt), [])
            if d is not None and not np.isfinite(d.at_hour):
                bad.append(label)
        except Exception as exc:
            crashed.append(f"{label}:{type(exc).__name__}")

    secure("malformed amounts do not crash the agent", not crashed,
           "; ".join(crashed))
    secure("malformed amounts never yield a non-finite schedule", not bad,
           "; ".join(bad))


def test_agent_survives_unknown_categorical_values() -> None:
    """Categoricals are encoded against fixed vocabularies.

    A new payment method, a new merchant category or a brand-new error code
    must map to the reserved unknown slot rather than raising -- an exception
    here would take down recovery for every payment behind it.
    """
    agent, cfg = _agent()
    cases = {
        "method": _obs(method="crypto_wallet"),
        "category": _obs(category="defence_procurement"),
        "error_code": _obs(error_code="totally_new_code_9999"),
        "all three": _obs(method="x", category="y", error_code="z"),
    }
    crashed = []
    for label, obs in cases.items():
        try:
            agent.decide(obs, [])
        except Exception as exc:
            crashed.append(f"{label}:{type(exc).__name__}: {exc}")
    secure("unknown categorical values are handled, not raised", not crashed,
           " | ".join(crashed)[:200])


def test_agent_survives_unknown_customer() -> None:
    """Cold start: a customer with no history at all.

    The profile lookup must fall back rather than KeyError, and the payday
    estimate must carry zero confidence rather than a confident guess.
    """
    agent, cfg = _agent()
    try:
        d = agent.decide(_obs(customer_id=999_999_999), [])
        ok = True
        detail = f"action={d.action.value if d else 'stop'}"
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    secure("an unseen customer does not break the agent", ok, detail)

    fb = agent.ctx.fallback_profile
    secure("cold-start payday estimate carries zero confidence",
           fb["payday_confidence"] == 0.0,
           f"confidence={fb['payday_confidence']}")


def test_taxonomy_rejects_non_string_codes() -> None:
    """None / numeric / bytes codes must not crash classification."""
    crashed = []
    for value in (None, 12345, b"card_declined", ["card_declined"], {}):
        try:
            classify(value)  # type: ignore[arg-type]
            never_retry(value)  # type: ignore[arg-type]
        except Exception as exc:
            crashed.append(f"{type(value).__name__}:{type(exc).__name__}")
    secure("non-string error codes are classified without raising",
           not crashed, "; ".join(crashed), severity="MEDIUM")


def test_observation_whitelist_drops_injected_fields() -> None:
    """A record carrying extra attacker-controlled keys must not leak them."""
    records, _, _ = load()
    poisoned = dict(records[0])
    poisoned["_cause"] = "RISK_BLOCK"
    poisoned["__class__"] = "evil"
    poisoned["oracle_answer"] = True

    obs = observe(poisoned)
    leaked = [k for k in obs if k.startswith("_") or k == "oracle_answer"]
    secure("injected fields never reach a policy", not leaked, f"leaked={leaked}")


# =========================================================================
# C. DEPLOYMENT SURFACE
# =========================================================================
def test_model_artifact_is_an_untrusted_code_path() -> None:
    """models/agent.pkl is loaded with pickle, which executes arbitrary code.

    This is a REAL and unavoidable property of pickle, not a hypothetical. The
    test does not pretend otherwise -- it asserts that the risk is documented,
    so an operator cannot be surprised by it. Anyone who can write that file
    can run code as whoever runs the evaluation.
    """
    doc = (ROOT / "docs" / "security.md")
    secure("the pickle deserialisation risk is documented for operators",
           doc.exists() and "pickle" in doc.read_text(encoding="utf-8").lower(),
           "docs/security.md missing or silent on pickle", severity="MEDIUM")


def test_guardrail_one_holds_without_the_pushback_model() -> None:
    """Defence in depth: if the learned head is missing, the hard rule remains.

    A model file that fails to load, or a rollback to an older bundle, must not
    silently remove fraud suppression.
    """
    with open(ROOT / "models" / "agent.pkl", "rb") as fh:
        b = pickle.load(fh)
    cfg = SimConfig()
    crippled = RecoveryAgent(b["model"], b["ctx"], cfg,
                             p_floor=b["p_floor"], risk_model=None)
    d = crippled.decide(_obs(error_code="payment_risk_check_failed",
                             amount=100_000.0), [])
    secure("explicit fraud block survives loss of the pushback model",
           d is None, f"action={d.action.value if d else None}")


def test_degraded_model_cannot_break_hard_guardrails() -> None:
    """A model returning garbage must still not defeat the coded rules."""

    class ConstantModel:
        classes_ = np.array([False, True])

        def predict_proba(self, X):
            # Maximally confident that everything recovers.
            return np.tile([0.0, 1.0], (X.shape[0], 1))

    with open(ROOT / "models" / "agent.pkl", "rb") as fh:
        b = pickle.load(fh)
    cfg = SimConfig()
    rogue = RecoveryAgent(ConstantModel(), b["ctx"], cfg,
                          p_floor=b["p_floor"], risk_model=ConstantModel())
    d = rogue.decide(_obs(error_code="payment_risk_check_failed",
                          amount=100_000.0), [])
    secure("a maximally over-confident model cannot retry an explicit risk decline",
           d is None, f"action={d.action.value if d else None}")


def test_costs_are_always_charged_before_the_outcome_is_known() -> None:
    """Cost accounting must not be conditional on success.

    If cost were only charged on failure (or only on success), a policy could
    look profitable by construction. Verified by driving a policy that always
    acts and confirming cost scales with attempts, not recoveries.
    """
    class AlwaysNudge(Policy):
        name = "always_nudge"

        def decide(self, obs, history):
            if len(history) >= 2:
                return None
            return Decision(Action.NUDGE,
                            obs["hour"] + 1.0 + 24.0 * len(history), "nudge")

    records, sim, cfg = load()
    res, _ = evaluate(AlwaysNudge(), records[:600], sim, cfg)
    expected = res.attempts * cfg.nudge_cost_inr
    secure("every attempt is charged, recovered or not",
           abs(res.cost - expected) < 1e-6,
           f"cost={res.cost:.2f} expected={expected:.2f} attempts={res.attempts}")


def test_no_policy_can_recover_more_than_the_value_at_risk() -> None:
    """A sanity ceiling: recovered revenue cannot exceed what was lost."""
    agent, cfg = _agent()
    records, sim, _ = load()
    ev = [r for r in records if r["hour"] >= cfg.train_hours][:1500]
    res, _ = evaluate(agent, ev, sim, cfg)
    secure("recovered revenue never exceeds value at risk",
           res.revenue_recovered <= res.value_at_risk + 1e-6,
           f"recovered={res.revenue_recovered:,.0f} at_risk={res.value_at_risk:,.0f}")
    secure("recovered count never exceeds payment count",
           res.recovered_n <= res.n_payments,
           f"{res.recovered_n}/{res.n_payments}")


def main() -> int:
    tests = [
        # A. guardrail bypass
        test_fraud_suppression_cannot_be_bypassed_by_string_tricks,
        test_unknown_code_does_not_fail_open_for_the_agent,
        test_large_amount_cannot_buy_past_the_probability_floor,
        test_attempt_budget_cannot_be_exceeded_via_history_forgery,
        test_horizon_cannot_be_extended_by_negative_or_nan_times,
        test_agent_never_proposes_a_non_advancing_time,
        # B. hostile input
        test_agent_survives_malformed_amounts,
        test_agent_survives_unknown_categorical_values,
        test_agent_survives_unknown_customer,
        test_taxonomy_rejects_non_string_codes,
        test_observation_whitelist_drops_injected_fields,
        # C. deployment surface
        test_model_artifact_is_an_untrusted_code_path,
        test_guardrail_one_holds_without_the_pushback_model,
        test_degraded_model_cannot_break_hard_guardrails,
        test_costs_are_always_charged_before_the_outcome_is_known,
        test_no_policy_can_recover_more_than_the_value_at_risk,
    ]
    for fn in tests:
        print(f"\n--- {fn.__name__} ---")
        fn()

    print("\n" + "=" * 70)
    print(f"{PASSES} attacks repelled, {len(FINDINGS)} vulnerabilities found")
    if FINDINGS:
        for sev, name in FINDINGS:
            print(f"  [{sev}] {name}")
        return 1
    print("no vulnerabilities found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
