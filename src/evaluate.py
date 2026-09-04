"""Evaluation harness: score any recovery policy against the counterfactual oracle.

What makes this comparison fair
-------------------------------
1. IDENTICAL DATA. Every policy is scored on the same failed payments.

2. IDENTICAL RANDOMNESS. The oracle is stochastic, so a lucky stream could
   manufacture a lift. Each payment gets its own RNG seeded from its txn_id,
   so every policy starts that payment from the same random state rather than
   sharing one global stream whose position depends on how many attempts the
   previous policies happened to make.

3. NO PEEKING. The observation handed to a policy is built from the
   OBSERVABLE_FIELDS whitelist, so the latent cause, liquidity and salary day
   used by the simulator are unreachable from policy code.

4. FULL COSTING. Recovered revenue is reported net of what it cost to chase:
   per-attempt gateway cost, nudge cost, and the goodwill penalty for putting
   a risk-declined payment back in front of an issuer.

The headline metric is NET VALUE, not recovery rate. A policy that recovers
more by retrying everything forever is not better, and reporting only the
recovery rate would hide that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .generator import Action, Cause, PaymentSimulator, SimConfig
from .policies import MAX_ATTEMPTS, OBSERVABLE_FIELDS, Attempt, Policy
from .taxonomy import classify

DATA = Path(__file__).resolve().parents[1] / "data"
# Successive attempts on one payment must be at least this far apart.
MIN_ATTEMPT_GAP_H = 0.5


@dataclass
class Result:
    policy: str
    n_payments: int = 0
    value_at_risk: float = 0.0

    recovered_n: int = 0
    revenue_recovered: float = 0.0

    attempts: int = 0
    issuer_attempts: int = 0        # attempts that hit the issuer/rails
    nudges: int = 0
    cost: float = 0.0

    fraud_attempts: int = 0         # issuer-facing attempts on a risk decline
    fraud_penalty: float = 0.0

    hours_to_recovery: list = field(default_factory=list)
    per_class: dict = field(default_factory=dict)

    @property
    def recovery_rate(self) -> float:
        return self.recovered_n / self.n_payments if self.n_payments else 0.0

    @property
    def value_recovery_rate(self) -> float:
        return self.revenue_recovered / self.value_at_risk if self.value_at_risk else 0.0

    @property
    def net_value(self) -> float:
        return self.revenue_recovered - self.cost

    @property
    def attempts_per_payment(self) -> float:
        return self.attempts / self.n_payments if self.n_payments else 0.0

    @property
    def median_hours_to_recovery(self) -> float:
        return float(np.median(self.hours_to_recovery)) if self.hours_to_recovery else float("nan")

    def row(self) -> dict:
        return {
            "policy": self.policy,
            "recovery_rate": round(self.recovery_rate, 4),
            "value_rate": round(self.value_recovery_rate, 4),
            "revenue_recovered": round(self.revenue_recovered, 0),
            "cost": round(self.cost, 0),
            "net_value": round(self.net_value, 0),
            "attempts_pp": round(self.attempts_per_payment, 2),
            "fraud_attempts": self.fraud_attempts,
            "median_h_to_recover": round(self.median_hours_to_recovery, 1),
        }


def observe(record: dict) -> dict:
    """Build the policy-visible view. Latent fields are unreachable from here."""
    return {k: record[k] for k in OBSERVABLE_FIELDS}


def evaluate(
    policy: Policy,
    records: list[dict],
    sim: PaymentSimulator,
    cfg: SimConfig,
    seed: int = 20260826,
    collect_audit: int = 0,
) -> tuple[Result, pd.DataFrame]:
    """Run one policy over every failed payment and score it."""
    res = Result(policy=policy.name)
    audit_rows: list[dict] = []
    deadline_offset = cfg.horizon_hours

    for rec in records:
        res.n_payments += 1
        res.value_at_risk += rec["amount"]

        cls = classify(rec["error_code"]).value
        bucket = res.per_class.setdefault(cls, {"n": 0, "recovered": 0, "attempts": 0})
        bucket["n"] += 1

        # Per-payment RNG: every policy faces this payment from the same state.
        rng = np.random.default_rng(seed * 1_000_003 + int(rec["txn_id"]))

        policy.reset()
        obs = observe(rec)
        history: list[Attempt] = []
        deadline = rec["hour"] + deadline_offset
        is_fraud = rec["_cause"] == Cause.RISK_BLOCK.value

        last_hour = rec["hour"]
        while len(history) < MAX_ATTEMPTS:
            decision = policy.decide(obs, history)
            if decision is None:
                break

            # Reject non-finite times BEFORE any comparison. Every bound check
            # below is a `<` or `>`, and every comparison against NaN is False,
            # so a NaN-scheduled attempt would pass the deadline check, pass
            # the forward-time check, and execute. Measured: 947 attempts and
            # INR 14,367 of cost slipped through this way.
            if not np.isfinite(decision.at_hour):
                break

            if decision.at_hour > deadline:
                break

            # Attempts must move forward in time. Without this a policy can
            # schedule the same action at the same instant repeatedly and
            # collect several independent draws from the oracle for one moment
            # -- free retries that no real processor gets. The agent found and
            # exploited exactly this, so the rule is enforced here rather than
            # trusted to each policy.
            if decision.at_hour < last_hour + MIN_ATTEMPT_GAP_H:
                break
            last_hour = decision.at_hour

            # Cost of acting, before knowing whether it worked.
            if decision.action is Action.NUDGE:
                res.cost += cfg.nudge_cost_inr
                res.nudges += 1
            elif decision.action in (Action.RETRY, Action.SWITCH_METHOD):
                res.cost += cfg.retry_cost_inr
                res.issuer_attempts += 1
                if is_fraud:
                    res.fraud_attempts += 1
                    res.fraud_penalty += cfg.fraud_retry_penalty_inr
                    res.cost += cfg.fraud_retry_penalty_inr
            else:
                break  # Action.STOP

            ok = sim.would_succeed(rec, decision.action, decision.at_hour, rng)
            res.attempts += 1
            bucket["attempts"] += 1
            history.append(Attempt(decision.action, decision.at_hour, decision.reason, ok))

            if collect_audit and len(audit_rows) < collect_audit:
                audit_rows.append({
                    "txn_id": rec["txn_id"],
                    "error_code": rec["error_code"],
                    "recovery_class": cls,
                    "amount": rec["amount"],
                    "attempt": len(history),
                    "action": decision.action.value,
                    "delay_h": round(decision.at_hour - rec["hour"], 2),
                    "reason": decision.reason,
                    "succeeded": ok,
                    "_true_cause": rec["_cause"],   # audit only, never shown to the policy
                })

            if ok:
                res.recovered_n += 1
                res.revenue_recovered += rec["amount"]
                res.hours_to_recovery.append(decision.at_hour - rec["hour"])
                bucket["recovered"] += 1
                break

    return res, pd.DataFrame(audit_rows)


def load() -> tuple[list[dict], PaymentSimulator, SimConfig]:
    """Load the failed payments and rebuild the simulator that produced them.

    The simulator's world (downtime windows, customer attributes) is built in
    __init__ from the seed alone, so reconstructing it here reproduces exactly
    the world the dataset came from.
    """
    cfg = SimConfig()
    sim = PaymentSimulator(cfg)
    failed = pd.read_parquet(DATA / "failed_payments.parquet")
    return failed.to_dict("records"), sim, cfg


def compare(policies, collect_audit: int = 0) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    records, sim, cfg = load()
    rows, results, audit = [], {}, pd.DataFrame()
    for p in policies:
        res, aud = evaluate(p, records, sim, cfg, collect_audit=collect_audit)
        rows.append(res.row())
        results[p.name] = res
        if collect_audit and audit.empty:
            audit = aud
    return pd.DataFrame(rows), results, audit
