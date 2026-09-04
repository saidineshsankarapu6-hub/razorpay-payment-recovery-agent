"""Synthetic payment-failure simulator.

Why this file matters more than the model
-----------------------------------------
There is no public dataset of real failed Razorpay transactions. Every metric
this project reports is therefore only as credible as this generator. So the
design goal is not "produce plausible-looking rows" -- it is "produce a world
in which a smarter retry policy earns its lift for the same reasons it would
in production, and can be scored counterfactually".

Two design decisions carry that weight.

1. LATENT CAUSE -> OBSERVED CODE.
   We sample a hidden cause first, then emit a Razorpay error code from it.
   Crucially the mapping is many-to-one and lossy: NO_FUNDS, LIMIT_EXCEEDED,
   RISK_BLOCK and ISSUER_DOWN can all surface as the opaque card_declined /
   payment_failed / payment_declined. That mirrors real issuer behaviour
   (issuers routinely mask NSF as a generic do-not-honour) and it manufactures
   precisely the inference problem the agent exists to solve: same code, four
   different correct actions.

2. A COUNTERFACTUAL RECOVERY ORACLE.
   Each failed payment carries the latent state needed to answer "if you retry
   at time T (or switch method, or nudge), does it succeed?" without having
   generated that retry in advance. This lets the evaluation harness score ANY
   policy offline on identical data -- which is what makes an agent-vs-baseline
   comparison honest rather than self-serving.

Every numeric assumption is named in SimConfig and defended in
docs/data_assumptions.md. They are deliberately editable: a panel asking
"what if UPI success is really 96%?" should be answerable by changing one line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .taxonomy import Method


class Cause(str, Enum):
    """Latent reason a payment failed. Never observable to the agent."""
    ISSUER_DOWN = "ISSUER_DOWN"
    GATEWAY_ISSUE = "GATEWAY_ISSUE"
    NO_FUNDS = "NO_FUNDS"
    AUTH_FAIL = "AUTH_FAIL"
    INSTRUMENT_STATE = "INSTRUMENT_STATE"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    RISK_BLOCK = "RISK_BLOCK"
    CUSTOMER_ABANDON = "CUSTOMER_ABANDON"


class Action(str, Enum):
    """What a recovery policy can do with a failed payment."""
    RETRY = "RETRY"                  # same method, later
    SWITCH_METHOD = "SWITCH_METHOD"  # different instrument
    NUDGE = "NUDGE"                  # ask the customer to act
    STOP = "STOP"                    # give up (or must not retry)


@dataclass
class SimConfig:
    """All assumptions live here so they can be challenged one line at a time."""

    seed: int = 7

    # Scale. n_customers is set so that each customer carries roughly 90
    # transactions of history (~60 inside the training window). The agent has
    # to ESTIMATE a customer's payday from when their payments historically
    # succeed -- it cannot read salary_day -- and that is not learnable from a
    # handful of observations. This is a repeat/subscription-commerce
    # assumption and it is stated rather than hidden: a merchant seeing each
    # customer twice could not run the per-customer half of this agent.
    n_customers: int = 1_200
    n_merchants: int = 60
    n_days: int = 90
    txns_per_day: int = 1_200

    # Temporal split. Everything the agent learns -- customer profiles and the
    # recovery model -- is fitted on failures before this hour, and it is
    # scored only on failures after it. No shuffled split: a policy that
    # learned from the future would be worthless in production and the lift
    # would be an illusion.
    train_hours: int = 60 * 24

    # Method mix. India skews heavily to UPI; card remains dominant for
    # higher-ticket and recurring. Assumption, documented for challenge.
    method_mix: dict = field(default_factory=lambda: {
        Method.UPI: 0.62,
        Method.CARD: 0.22,
        Method.NETBANKING: 0.08,
        Method.WALLET: 0.08,
    })

    # Baseline success rate per method, before any modifiers.
    base_success: dict = field(default_factory=lambda: {
        # Success GIVEN funds are available. Funds failures are modelled
        # separately below, so these must not double-count them -- they sit
        # above the commonly quoted blended rates for that reason.
        Method.UPI: 0.97,
        Method.CARD: 0.93,
        Method.NETBANKING: 0.91,
        Method.WALLET: 0.96,
    })

    # Issuer downtime: rare, bursty, correlated across all customers on that
    # issuer. Modelled as windows rather than per-transaction noise, because
    # clustering is exactly what makes "wait for the window to clear" a
    # learnable strategy instead of a coin flip.
    n_issuers: int = 12
    downtime_events_per_issuer: float = 4.0   # over the whole horizon
    downtime_hours_mean: float = 3.5

    # Salary cycle. Liquidity sawtooths: high just after payday, low before.
    # This is what makes retry timing worth learning for NO_FUNDS.
    salary_days: tuple = (1, 7, 25, 30)

    # Affordability. Rather than an abstract "strain" term, this models the
    # thing that actually happens: a customer has some spending capacity, the
    # salary cycle scales how much of it is available right now, and a payment
    # fails for funds reasons when the amount is large relative to what is
    # available. The failure probability is a logistic in that ratio.
    #
    # This is more interpretable than a fitted penalty AND it produces a funds
    # effect large enough to be inferable from a customer's own history --
    # which the previous formulation did not, at roughly 2 percentage points of
    # swing. Calibrated so overall success stays in the 88-92% band.
    capacity_median_inr: float = 3_600.0
    capacity_sigma: float = 0.30
    # How strongly a customer's ticket sizes track their own capacity. At 0 a
    # customer with a small balance is just as likely to attempt a large
    # purchase as anyone else, which produced an implausible tail of payments
    # far beyond the payer's means and swamped the salary signal. People
    # broadly spend in proportion to what they have.
    capacity_amount_coupling: float = 0.6
    nsf_cap: float = 0.90          # max funds-failure probability when broke
    nsf_steepness: float = 7.0
    nsf_midpoint: float = 1.05     # ratio at which funds trouble is 50% of cap

    # Share of customers who are strongly salary-cycle-bound.
    thin_margin_share: float = 0.30

    # Economics. A retry is not free: gateway cost per attempt, plus an
    # issuer-goodwill penalty for retrying something that should not be retried.
    retry_cost_inr: float = 2.5
    nudge_cost_inr: float = 0.9
    fraud_retry_penalty_inr: float = 250.0

    horizon_hours: int = 168  # recovery window a policy is scored over (7 days)


# Cause -> observed error code, per method. Weights are emission probabilities.
# The opaque codes appear under multiple causes on purpose: that lossiness IS
# the problem being solved.
_EMISSION = {
    Method.CARD: {
        # payment_failed is Razorpay's generic "bank declined the transaction".
        # It is emitted from several causes on purpose: if any single cause
        # owned it, the code would be perfectly decodable by lookup and the
        # inference problem would be an artefact of the generator, not of
        # payments. Same reasoning for card_declined.
        Cause.ISSUER_DOWN:      {"bank_technical_error": 0.70, "payment_failed": 0.30},
        Cause.GATEWAY_ISSUE:    {"gateway_technical_error": 0.6, "payment_timed_out": 0.4},
        Cause.NO_FUNDS:         {"insufficient_funds": 0.50, "card_declined": 0.35,
                                 "payment_failed": 0.15},
        Cause.AUTH_FAIL:        {"authentication_failed": 0.75, "incorrect_cvv": 0.15,
                                 "payment_failed": 0.10},
        Cause.INSTRUMENT_STATE: {"card_expired": 0.25, "debit_instrument_blocked": 0.2,
                                 "card_not_enrolled": 0.2,
                                 "card_disabled_for_online_payments": 0.2,
                                 "debit_instrument_inactive": 0.15},
        Cause.LIMIT_EXCEEDED:   {"transaction_limit_exceeded": 0.50, "card_declined": 0.30,
                                 "payment_failed": 0.20},
        Cause.RISK_BLOCK:       {"payment_risk_check_failed": 0.60, "card_declined": 0.25,
                                 "payment_failed": 0.15},
        Cause.CUSTOMER_ABANDON: {"payment_cancelled": 0.7, "payment_timed_out": 0.3},
    },
    Method.UPI: {
        Cause.ISSUER_DOWN:      {"bank_technical_error": 0.8, "payment_declined": 0.2},
        Cause.GATEWAY_ISSUE:    {"gateway_technical_error": 0.55, "payment_timed_out": 0.45},
        Cause.NO_FUNDS:         {"insufficient_funds": 0.6, "payment_declined": 0.4},
        Cause.AUTH_FAIL:        {"payment_declined": 0.5, "credit_failed": 0.5},
        Cause.INSTRUMENT_STATE: {"invalid_vpa": 0.4, "vpa_resolution_failed": 0.3,
                                 "customer_bank_account_mismatch": 0.3},
        Cause.LIMIT_EXCEEDED:   {"payment_declined": 0.7, "credit_failed": 0.3},
        Cause.RISK_BLOCK:       {"payment_declined": 0.6, "credit_failed": 0.4},
        Cause.CUSTOMER_ABANDON: {"payment_cancelled": 0.5,
                                 "payment_collect_request_expired": 0.5},
    },
}
# Netbanking and wallet reuse the card/UPI emission tables: Razorpay documents
# the same generic codes for them, and modelling them separately would invent
# structure we have no evidence for.
_EMISSION[Method.NETBANKING] = _EMISSION[Method.CARD]
_EMISSION[Method.WALLET] = _EMISSION[Method.UPI]


# Relative propensity of each cause, before context modifiers.
_CAUSE_PRIOR = {
    Cause.NO_FUNDS: 0.30,
    Cause.AUTH_FAIL: 0.20,
    Cause.CUSTOMER_ABANDON: 0.14,
    Cause.GATEWAY_ISSUE: 0.12,
    Cause.INSTRUMENT_STATE: 0.10,
    Cause.ISSUER_DOWN: 0.06,
    Cause.LIMIT_EXCEEDED: 0.05,
    Cause.RISK_BLOCK: 0.03,
}


class PaymentSimulator:
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._build_world()

    # -- world -------------------------------------------------------------
    def _build_world(self) -> None:
        cfg, rng = self.cfg, self.rng

        self.customers = pd.DataFrame({
            "customer_id": np.arange(cfg.n_customers),
            "issuer_id": rng.integers(0, cfg.n_issuers, cfg.n_customers),
            "salary_day": rng.choice(cfg.salary_days, cfg.n_customers),
            # Structural liquidity: some customers are simply closer to the edge.
            "balance_health": rng.beta(2.5, 2.0, cfg.n_customers),
            # What this customer can comfortably spend when fully liquid.
            "spend_capacity": rng.lognormal(
                np.log(cfg.capacity_median_inr), cfg.capacity_sigma, cfg.n_customers),
            # How much this customer's available money swings with their salary
            # cycle. Near 0 = comfortable, balance barely moves with payday.
            # Near 1 = paycheck-to-paycheck, flush on payday and empty before it.
            #
            # This heterogeneity is the point. A single population-wide salary
            # effect is both unrealistic and uninteresting: if everyone had the
            # same mild cycle, no per-customer inference would be needed or
            # possible. In reality a minority of customers are strongly
            # cycle-bound -- and they are precisely the ones generating
            # insufficient-funds failures worth timing a retry around. The
            # agent's job is to work out WHICH customers those are.
            "salary_sensitivity": np.where(
                rng.random(cfg.n_customers) < cfg.thin_margin_share,
                rng.beta(6.0, 2.0, cfg.n_customers),   # thin margin: strong cycle
                rng.beta(1.5, 6.0, cfg.n_customers),   # comfortable: weak cycle
            ),
            # Will they act on a nudge? Drives NUDGE value; independent of money.
            "engagement": rng.beta(2.0, 3.0, cfg.n_customers),
            # Does the customer have a usable second instrument?
            "has_alt_method": rng.random(cfg.n_customers) < 0.62,
        })

        self.merchants = pd.DataFrame({
            "merchant_id": np.arange(cfg.n_merchants),
            "category": rng.choice(
                ["ecommerce", "saas_subscription", "utility", "travel", "edtech"],
                cfg.n_merchants, p=[0.35, 0.2, 0.2, 0.1, 0.15]),
            "log_ticket": rng.normal(6.4, 0.75, cfg.n_merchants),  # ~INR 600 median
        })

        # Downtime windows per issuer: (issuer, start_hour, end_hour)
        windows = []
        total_hours = cfg.n_days * 24
        for issuer in range(cfg.n_issuers):
            n_events = rng.poisson(cfg.downtime_events_per_issuer)
            for _ in range(n_events):
                start = rng.uniform(0, total_hours)
                dur = max(0.5, rng.exponential(cfg.downtime_hours_mean))
                windows.append((issuer, start, start + dur))
        self.downtime = pd.DataFrame(
            windows, columns=["issuer_id", "start_h", "end_h"])

    # -- mechanics ---------------------------------------------------------
    def _liquidity(self, salary_day: np.ndarray, hour: np.ndarray) -> np.ndarray:
        """Fraction of the salary cycle's money still available.

        Rises to 1 on payday and decays until the next one. This is the signal
        that makes "retry on payday" beat "retry in 24h" for NO_FUNDS -- and it
        is why a fixed backoff leaves money on the table.
        """
        day_of_month = (hour // 24) % 30 + 1
        days_since = (day_of_month - salary_day) % 30
        return np.exp(-days_since / 11.0)

    def _p_nsf(self, amount, capacity, balance_health, eff_liquidity):
        """Probability a payment fails for funds reasons.

        Available money is the customer's capacity scaled by their structural
        balance health and by how far through the salary cycle they are. The
        more the amount exceeds it, the more likely the payment bounces.

        Generation and the recovery oracle both call this, so "would a retry
        succeed later?" is answered by the same model that decided the payment
        failed in the first place -- rather than by a second, unrelated set of
        hand-set constants.
        """
        cfg = self.cfg
        # Deliberately NOT scaled by balance_health as well. Capacity already
        # carries the cross-customer spread (lognormal), and multiplying by a
        # second poverty term double-counted it: customers low on both were
        # broke in every month, which flattened the salary cycle into a
        # constant and dragged the whole population's success rate down.
        available = np.maximum(capacity * eff_liquidity, 1.0)
        ratio = amount / available
        return cfg.nsf_cap / (1.0 + np.exp(-cfg.nsf_steepness * (ratio - cfg.nsf_midpoint)))

    def _in_downtime(self, issuer_id: np.ndarray, hour: np.ndarray) -> np.ndarray:
        out = np.zeros(len(hour), dtype=bool)
        for iss, s, e in self.downtime.itertuples(index=False):
            out |= (issuer_id == iss) & (hour >= s) & (hour < e)
        return out

    def _downtime_end(self, issuer_id: int, hour: float) -> float | None:
        """When the active downtime window for this issuer clears."""
        hits = self.downtime[(self.downtime.issuer_id == issuer_id)
                             & (self.downtime.start_h <= hour)
                             & (self.downtime.end_h > hour)]
        return float(hits.end_h.max()) if len(hits) else None

    def generate(self) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        n = cfg.n_days * cfg.txns_per_day

        cust_idx = rng.integers(0, cfg.n_customers, n)
        merch_idx = rng.integers(0, cfg.n_merchants, n)
        cust = self.customers.iloc[cust_idx].reset_index(drop=True)
        merch = self.merchants.iloc[merch_idx].reset_index(drop=True)

        # Transaction times, skewed toward waking hours.
        day = rng.integers(0, cfg.n_days, n)
        hour_of_day = np.clip(rng.normal(15, 4.5, n), 0, 23.99)
        hour = day * 24 + hour_of_day

        cust_scale = (np.log(cust.spend_capacity.values)
                      - np.log(cfg.capacity_median_inr))
        log_mu = merch.log_ticket.values + cfg.capacity_amount_coupling * cust_scale
        amount = np.exp(rng.normal(log_mu, 0.55)).round(2)
        # Sample indices, not enum members: numpy coerces objects to str, and
        # a (str, Enum) member stringifies to "Method.UPI" rather than "upi".
        methods = list(cfg.method_mix.keys())
        m_idx = rng.choice(len(methods), n, p=list(cfg.method_mix.values()))
        method = [methods[k] for k in m_idx]

        liquidity = self._liquidity(cust.salary_day.values, hour)
        # Effective liquidity blends the cycle with the customer's sensitivity:
        # a comfortable customer sits near 1.0 all month, a paycheck-to-paycheck
        # one tracks the cycle closely.
        sens = cust.salary_sensitivity.values
        eff_liquidity = (1 - sens) + sens * liquidity
        downtime = self._in_downtime(cust.issuer_id.values, hour)

        # Success probability: base, degraded by downtime, thin liquidity, and
        # amounts that are large relative to the customer's headroom.
        p = np.array([cfg.base_success[m] for m in method])
        p_nsf = self._p_nsf(amount, cust.spend_capacity.values,
                            cust.balance_health.values, eff_liquidity)
        p = p * (1 - p_nsf)
        p = np.where(downtime, p * 0.25, p)
        p = np.clip(p, 0.02, 0.995)

        success = rng.random(n) < p

        df = pd.DataFrame({
            "txn_id": np.arange(n),
            "hour": hour.round(3),
            "customer_id": cust.customer_id.values,
            "merchant_id": merch.merchant_id.values,
            "issuer_id": cust.issuer_id.values,
            "category": merch.category.values,
            "method": [m.value for m in method],
            "amount": amount,
            "success": success,
            # latent, for the oracle and for evaluation only
            "_liquidity": liquidity.round(4),
            "_eff_liquidity": eff_liquidity.round(4),
            "_salary_sensitivity": sens.round(4),
            "_spend_capacity": cust.spend_capacity.values.round(2),
            "_balance_health": cust.balance_health.values.round(4),
            "_engagement": cust.engagement.values.round(4),
            "_has_alt_method": cust.has_alt_method.values,
            "_in_downtime": downtime,
            "_salary_day": cust.salary_day.values,
        })

        failed = ~success
        causes = self._sample_causes(df[failed], rng)
        df.loc[failed, "_cause"] = [c.value for c in causes]
        df.loc[failed, "error_code"] = self._emit_codes(
            df.loc[failed, "method"].values, causes, rng)

        return df

    def _sample_causes(self, failed: pd.DataFrame, rng) -> list[Cause]:
        """Context-conditioned cause sampling.

        Priors are reweighted per transaction: downtime forces ISSUER_DOWN,
        thin liquidity pulls toward NO_FUNDS, large amounts toward LIMIT and
        RISK. Without this conditioning the cause would be independent of the
        features, and any model would be learning noise.
        """
        keys = list(_CAUSE_PRIOR.keys())
        base = np.array([_CAUSE_PRIOR[k] for k in keys])
        w = np.tile(base, (len(failed), 1))

        i = {c: k for k, c in enumerate(keys)}
        liq = failed["_eff_liquidity"].values * failed["_balance_health"].values
        big = np.clip(np.log1p(failed["amount"].values) / 9.0, 0, 1.5)

        w[:, i[Cause.NO_FUNDS]] *= np.clip(1.6 - 1.4 * liq, 0.20, 2.4)
        w[:, i[Cause.LIMIT_EXCEEDED]] *= np.clip(0.4 + 1.8 * big, 0.3, 3.0)
        w[:, i[Cause.RISK_BLOCK]] *= np.clip(0.3 + 1.6 * big, 0.2, 3.0)

        down = failed["_in_downtime"].values
        w[down] = 0.0
        w[down, i[Cause.ISSUER_DOWN]] = 1.0

        w /= w.sum(axis=1, keepdims=True)
        picks = np.array([rng.choice(len(keys), p=row) for row in w])
        return [keys[k] for k in picks]

    def _emit_codes(self, methods, causes, rng) -> list[str]:
        out = []
        for m, c in zip(methods, causes):
            table = _EMISSION[Method(m)][c]
            codes, probs = list(table.keys()), list(table.values())
            out.append(str(rng.choice(codes, p=probs)))
        return out

    # -- counterfactual oracle --------------------------------------------
    def would_succeed(self, row, action: Action, at_hour: float, rng) -> bool:
        """Ground truth: does this action taken at at_hour recover the payment?

        This is the whole point of the simulator. Any policy -- naive baseline
        or learned agent -- is scored against the same oracle on the same rows,
        so a lift cannot be an artefact of the policy having seen better data.
        """
        cause = Cause(row["_cause"])
        delay = at_hour - row["hour"]
        if delay < 0:
            return False

        if action is Action.STOP:
            return False

        if cause is Cause.RISK_BLOCK:
            return False  # never recoverable; retrying only incurs the penalty

        if action is Action.NUDGE:
            # A nudge only helps where a human decision is the blocker, and its
            # power decays as the purchase intent goes cold.
            if cause not in (Cause.CUSTOMER_ABANDON, Cause.AUTH_FAIL,
                             Cause.INSTRUMENT_STATE, Cause.NO_FUNDS):
                return False
            decay = np.exp(-delay / 48.0)
            return rng.random() < row["_engagement"] * decay * 0.8

        if action is Action.SWITCH_METHOD:
            if not row["_has_alt_method"]:
                return False
            if cause in (Cause.INSTRUMENT_STATE, Cause.LIMIT_EXCEEDED,
                         Cause.ISSUER_DOWN):
                return rng.random() < 0.80   # different rails, problem bypassed
            if cause is Cause.NO_FUNDS:
                return rng.random() < 0.25   # usually the same empty wallet
            if cause is Cause.GATEWAY_ISSUE:
                return rng.random() < 0.70
            return rng.random() < 0.30

        # Action.RETRY -- same method, later.
        if cause is Cause.ISSUER_DOWN:
            end = self._downtime_end(int(row["issuer_id"]), row["hour"])
            if end is None or at_hour >= end:
                return rng.random() < 0.88
            return rng.random() < 0.05      # still down; wasted attempt

        if cause is Cause.GATEWAY_ISSUE:
            return rng.random() < (0.30 if delay < 0.25 else 0.82)

        if cause is Cause.NO_FUNDS:
            # The signal worth learning: recovery tracks the salary cycle, not
            # elapsed time. Retrying at a fixed +24h mostly misses it.
            liq = float(self._liquidity(
                np.array([row["_salary_day"]]), np.array([at_hour]))[0])
            sens = row["_salary_sensitivity"]
            eff = (1 - sens) + sens * liq
            p_nsf = float(self._p_nsf(row["amount"], row["_spend_capacity"],
                                      row["_balance_health"], eff))
            return rng.random() < np.clip(0.95 * (1.0 - p_nsf), 0.02, 0.95)

        if cause is Cause.AUTH_FAIL:
            return rng.random() < 0.18      # silent retry rarely fixes a human
        if cause is Cause.CUSTOMER_ABANDON:
            return rng.random() < 0.10
        if cause is Cause.INSTRUMENT_STATE:
            return rng.random() < 0.02      # the instrument is genuinely broken
        if cause is Cause.LIMIT_EXCEEDED:
            return rng.random() < (0.72 if delay >= 24 else 0.06)  # limits reset daily
        return rng.random() < 0.15


def build(cfg: SimConfig | None = None) -> pd.DataFrame:
    return PaymentSimulator(cfg).generate()
