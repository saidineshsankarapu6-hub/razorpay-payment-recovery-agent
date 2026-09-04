"""Recovery policies: the baselines the agent must beat.

A policy sees a failed payment and decides, step by step, what to do about it:
retry now, retry later, switch instrument, nudge the customer, or stop.

Design notes that matter for the evaluation being honest
-------------------------------------------------------
* A policy is ADAPTIVE. It is asked for one decision at a time and is shown
  what has already been tried and failed. A policy that must commit to a fixed
  schedule up front is a special case of this, not a different interface.

* A policy sees ONLY observable fields. The harness constructs the observation
  from a whitelist, so a policy physically cannot read the latent cause,
  liquidity, or salary day that the simulator used. This is enforced in code
  rather than by convention, because "we were careful not to peek" is not an
  answer that survives scrutiny.

* Every decision carries a human-readable reason. That is not decoration: it
  becomes the audit trail, which the AI Revenue Recovery track calls for
  explicitly, and it is what lets a merchant ask "why did you charge my
  customer again at 10am on Tuesday?"

The baselines here are built deliberately COMPETENT. Beating a strawman proves
nothing. In particular RuleBased already suppresses the explicit fraud code and
already routes by recovery class -- it does everything the failure -> action
lookup table can do. The agent's advantage therefore has to come from the
places a lookup table structurally cannot reach: the 37% of failures whose code
discloses nothing, and the timing of retries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .generator import Action
from .taxonomy import RecoveryClass, classify, never_retry

# The only fields a policy may see. Anything prefixed with "_" in the dataset
# is simulator ground truth and is excluded by construction.
# Attempt budget per failed payment. Defined here so the harness and any
# policy that plans ahead agree on it.
MAX_ATTEMPTS = 4

OBSERVABLE_FIELDS = (
    "txn_id", "hour", "customer_id", "merchant_id", "issuer_id",
    "category", "method", "amount", "error_code",
)


@dataclass(frozen=True)
class Decision:
    action: Action
    at_hour: float          # absolute hour at which to act
    reason: str             # audit trail entry


@dataclass(frozen=True)
class Attempt:
    action: Action
    at_hour: float
    reason: str
    succeeded: bool


class Policy:
    """Base class. Return None to stop pursuing this payment."""

    name = "base"

    def decide(self, obs: Mapping, history: Sequence[Attempt]) -> Decision | None:
        raise NotImplementedError

    def reset(self) -> None:
        """Called once per payment, for policies that carry per-payment state."""


class NoRecovery(Policy):
    """The floor: abandon every failed payment. Establishes what is at stake."""

    name = "no_recovery"

    def decide(self, obs, history):
        return None


class FixedRetry(Policy):
    """The common production default: retry on a fixed schedule, code-blind.

    This is the honest naive baseline -- not a strawman. Retrying at +1h, +24h
    and +72h is a real pattern, and it recovers a genuine share of transient
    failures. Its weaknesses are structural rather than silly: it retries
    instruments that are broken, it retries risk declines it cannot see, and
    its timing is unrelated to when the customer will actually have money.
    """

    name = "fixed_retry"
    SCHEDULE = (1.0, 24.0, 72.0)

    def decide(self, obs, history):
        i = len(history)
        if i >= len(self.SCHEDULE):
            return None
        return Decision(
            Action.RETRY,
            obs["hour"] + self.SCHEDULE[i],
            f"fixed schedule attempt {i + 1} at +{self.SCHEDULE[i]:g}h",
        )


class RuleBased(Policy):
    """A competent rules engine: the failure -> action lookup table, done well.

    This is the baseline that matters. It routes by recovery class, does not
    retry instruments it knows are broken, nudges where a human is the blocker,
    and suppresses the explicit risk-decline code.

    Its one structural blind spot is the one it cannot fix: for an opaque code
    it has no idea which cause it is facing, so it must pick a single action
    for a bucket that mixes NO_FUNDS, LIMIT_EXCEEDED, ISSUER_DOWN and
    RISK_BLOCK. It retries -- which is the best single guess available to it,
    and which quietly retries the hidden risk declines.
    """

    name = "rule_based"

    PLAYBOOK: dict[RecoveryClass, tuple[tuple[Action, float], ...]] = {
        RecoveryClass.SOFT_TRANSIENT: (
            (Action.RETRY, 1.0), (Action.RETRY, 6.0), (Action.RETRY, 24.0),
        ),
        # Also sweep-selected. Note what the rules engine can and cannot do
        # here: it can learn that waiting longer beats retrying immediately,
        # because that is true on average. It cannot target an individual
        # customer's payday, because the schedule is fixed for everyone. That
        # residual is the agent's opportunity, and it is left on the table
        # honestly rather than by crippling the baseline.
        RecoveryClass.FUNDS: (
            (Action.NUDGE, 2.0), (Action.RETRY, 48.0), (Action.RETRY, 120.0),
        ),
        RecoveryClass.AUTH_FAILED: (
            (Action.NUDGE, 0.5), (Action.NUDGE, 24.0),
        ),
        RecoveryClass.METHOD_BROKEN: (
            (Action.SWITCH_METHOD, 0.5), (Action.NUDGE, 24.0),
        ),
        RecoveryClass.ABANDONED: (
            (Action.NUDGE, 1.0), (Action.NUDGE, 24.0),
        ),
        # The blind bucket. One fixed sequence has to cover four different
        # causes. This ordering was not hand-picked -- scripts/tune_baseline.py
        # sweeps the alternatives and this one wins on net value, so it is the
        # strongest rules engine available and therefore the fair bar for the
        # agent. Nudging first also happens to dodge some fraud penalties,
        # since a nudge never reaches the issuer.
        RecoveryClass.HARD_DECLINE: (
            (Action.NUDGE, 1.0), (Action.RETRY, 24.0), (Action.SWITCH_METHOD, 48.0),
        ),
        RecoveryClass.FRAUD: (),
    }

    def decide(self, obs, history):
        code = obs["error_code"]
        if never_retry(code):
            return None  # explicit risk decline: suppressed

        steps = self.PLAYBOOK[classify(code)]
        i = len(history)
        if i >= len(steps):
            return None

        action, delay = steps[i]
        return Decision(
            action,
            obs["hour"] + delay,
            f"rule[{classify(code).value}] step {i + 1}: {action.value} at +{delay:g}h",
        )


BASELINES: tuple[Policy, ...] = (NoRecovery(), FixedRetry(), RuleBased())
