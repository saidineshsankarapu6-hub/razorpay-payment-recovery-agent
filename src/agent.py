"""The recovery agent.

Decision procedure
------------------
For a failed payment the agent enumerates candidate (action, delay) pairs,
asks two learned models about each, converts them to an expected value, and
takes the best -- or stops if nothing is worth doing.

    EV(c)    = P(recovery) * amount - cost(c) - P(pushback) * penalty
    total(c) = EV(c) + (1 - P(recovery)) * max EV(c') over c' after c

The second line values the SEQUENCE rather than the next action alone. A
one-step-greedy version spends the whole horizon reaching its single best
moment and has nothing left when that moment fails.

What the agent is NOT allowed to do
-----------------------------------
* It never sees the latent cause. The model is trained on observed retry
  outcomes -- what a processor genuinely logs -- not on ground-truth labels
  that would not exist in production.
* It never reads the customer's payday. It uses an estimate inferred from that
  customer's own transaction history, together with a confidence score that is
  deliberately low for customers whose behaviour shows no salary cycle.

Guardrails sit OUTSIDE the model
--------------------------------
A learned expected value is the wrong place to encode "never do this". Two
rules are enforced in code, so no amount of model drift or a mispredicted
probability can override them:

1. An explicit risk decline is never retried.
2. Issuer-facing actions require P(recovery) above a floor. A retry at a 1%
   chance can still show positive EV on a large payment, but putting
   near-hopeless traffic in front of an issuer is exactly how a merchant's
   standing degrades. Nudges are exempt: they never touch the issuer.

The floor is chosen on TRAINING data (scripts/train_agent.py) and then left
alone. It is selected under an explicit constraint -- do not exceed the
incumbent rules engine's issuer friction -- rather than by maximising net
value, which would pick 0.0 and buy revenue with the merchant's bank
relationships.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_COLUMNS, FeatureContext, featurise_candidates
from .generator import Action, SimConfig
from .policies import MAX_ATTEMPTS, Attempt, Decision, Policy
from .taxonomy import is_known, never_retry

# Candidate action grid. Delays are denser early (most transient failures
# resolve fast) and stretch to the full 7-day horizon so the agent can wait for
# a salary credit when that is the better play.
RETRY_DELAYS = (0.5, 2.0, 6.0, 12.0, 24.0, 48.0, 72.0, 120.0, 168.0)
SWITCH_DELAYS = (0.5, 6.0, 24.0, 48.0)
NUDGE_DELAYS = (0.5, 2.0, 12.0, 24.0, 48.0)

CANDIDATES: tuple[tuple[Action, float], ...] = tuple(
    [(Action.RETRY, d) for d in RETRY_DELAYS]
    + [(Action.SWITCH_METHOD, d) for d in SWITCH_DELAYS]
    + [(Action.NUDGE, d) for d in NUDGE_DELAYS]
)

ISSUER_FACING = (Action.RETRY, Action.SWITCH_METHOD)

# Must match the harness rule in evaluate.py: successive attempts on one
# payment have to move forward in time.
MIN_GAP_H = 0.5

# Unknown codes are held to a higher bar before anything is sent to an issuer.
# `never_retry` can only block codes it knows about, so a NEW risk code from an
# issuer is invisible to guardrail 1 until someone triages it into the
# taxonomy. Raising the probability floor is the proportionate response: a
# genuinely recoverable new code still gets retried once the model is confident,
# while a new fraud code does not get chased on the strength of a large amount.
UNKNOWN_CODE_FLOOR_MULTIPLIER = 2.0

# Column offsets used by the lookahead to patch history counters in place.
COL_PRIOR_RETRIES = FEATURE_COLUMNS.index("n_prior_retries")
_PRIOR_COL_FOR = {
    Action.RETRY: COL_PRIOR_RETRIES,
    Action.NUDGE: FEATURE_COLUMNS.index("n_prior_nudges"),
    Action.SWITCH_METHOD: FEATURE_COLUMNS.index("n_prior_switches"),
}


class RecoveryAgent(Policy):
    name = "agent"

    def __init__(
        self,
        model,
        ctx: FeatureContext,
        cfg: SimConfig,
        p_floor: float = 0.05,
        min_ev: float = 0.0,
        risk_model=None,
    ):
        self.model = model
        self.risk_model = risk_model
        self.ctx = ctx
        self.cfg = cfg
        self.p_floor = p_floor
        self.min_ev = min_ev
        self._classes = list(model.classes_)
        self._pos = self._classes.index(True) if True in self._classes else 1
        if risk_model is not None:
            rc = list(risk_model.classes_)
            self._risk_pos = rc.index(True) if True in rc else 1

    def _cost(self, action: Action) -> float:
        return (self.cfg.nudge_cost_inr if action is Action.NUDGE
                else self.cfg.retry_cost_inr)

    def decide(self, obs, history) -> Decision | None:
        # Guardrail 1: an explicit risk decline is never pursued through the
        # issuer. Enforced before the model is consulted at all.
        if never_retry(obs["error_code"]):
            return None

        # Candidates must land after anything already tried. Delays are
        # measured from the original failure, so without this the agent
        # re-proposes the same instant on every attempt.
        horizon = self.cfg.horizon_hours
        earliest = 0.0
        if history:
            earliest = max(a.at_hour for a in history) - obs["hour"] + MIN_GAP_H
        cands = [(a, d) for a, d in CANDIDATES if earliest <= d <= horizon]
        if not cands:
            return None

        n = len(cands)
        actions = [a for a, _ in cands]
        delays = np.array([d for _, d in cands])

        amount = float(obs["amount"])
        floor = self.p_floor
        if not is_known(obs["error_code"]):
            floor = min(1.0, floor * UNKNOWN_CODE_FLOOR_MULTIPLIER)

        X = featurise_candidates(obs, history, self.ctx, actions, delays)

        # Score the immediate candidates and every lookahead row in ONE pass.
        # Each predict_proba call carries fixed overhead that dominates at
        # these batch sizes, so scoring the two stages separately -- across two
        # models -- meant four calls per decision and roughly doubled the
        # agent's runtime for no change in its choices.
        lookahead = len(history) + 1 < MAX_ATTEMPTS
        if lookahead:
            Xc, spans, cont_actions = self._continuation_rows(
                obs, history, cands, horizon)
        else:
            Xc, spans, cont_actions = None, None, []

        if Xc is not None and Xc.shape[0]:
            stacked = np.vstack([X, Xc])
            all_actions = actions + cont_actions
        else:
            stacked, all_actions = X, actions

        ev_all, p_all, pen_all = self._step_value(
            stacked, all_actions, amount, floor)
        immediate, p, p_pen = ev_all[:n], p_all[:n], pen_all[:n]

        # Value the SEQUENCE, not just the next action.
        #
        # A one-step-greedy agent picks the single highest-probability moment
        # and spends the whole 7-day window reaching it -- then has no horizon
        # left if it fails. Measured: greedy scheduling pushed median
        # time-to-recovery to 48h against the rules engine's 2h and dropped
        # easy transient failures from 0.934 to 0.871, because "retry at +1h,
        # and if that fails you still have six days" beats "retry once at
        # +168h" even when the later attempt looks better in isolation.
        #
        # So each candidate is scored as its own expected value plus, if it
        # fails, the best value still reachable afterwards:
        #
        #   total(c) = EV(c) + (1 - p(c)) * max EV(c') over c' after c
        #
        # One level of lookahead. It is an approximation -- the continuation
        # itself is scored greedily -- but it is what converts an early cheap
        # attempt from "worse" into "worth it because it keeps the option".
        total = immediate.copy()
        if spans is not None:
            cont_ev = ev_all[n:]
            cont = np.zeros(n)
            for i, (lo, hi) in enumerate(spans):
                if hi > lo:
                    # Continuing is optional, so a negative continuation is 0.
                    cont[i] = max(cont_ev[lo:hi].max(), 0.0)
            total = total + (1.0 - p) * cont

        best = int(np.argmax(total))
        if not np.isfinite(total[best]) or total[best] <= self.min_ev:
            return None

        action, delay = cands[best]
        pen_note = f" pen={p_pen[best]:.2f}" if self.risk_model is not None else ""
        return Decision(
            action,
            obs["hour"] + delay,
            f"{action.value} at +{delay:g}h | p={p[best]:.3f}{pen_note} "
            f"EV={immediate[best]:.1f} seq={total[best]:.1f} "
            f"(best of {n} candidates)",
        )

    def _step_value(self, X, actions, amount, floor=None):
        """Expected value of one action, net of cost and expected pushback."""
        p = self.model.predict_proba(X)[:, self._pos]
        costs = np.array([self._cost(a) for a in actions])
        issuer_facing = np.array([a in ISSUER_FACING for a in actions])

        # Price in the expected cost of issuer pushback. Without this the agent
        # values every retry at the gateway fee and will happily chase a large
        # unrecoverable payment forever: p * amount stays positive long after
        # the attempt has become a pure liability.
        p_pen = np.zeros(len(actions))
        if self.risk_model is not None:
            p_pen = self.risk_model.predict_proba(X)[:, self._risk_pos]
            p_pen = np.where(issuer_facing, p_pen, 0.0)

        ev = p * amount - costs - p_pen * self.cfg.fraud_retry_penalty_inr

        # Guardrail: issuer-facing actions need a minimum credible chance,
        # independent of how large the payment is.
        ev = np.where(
            issuer_facing & (p < (self.p_floor if floor is None else floor)),
            -np.inf, ev)
        return ev, p, p_pen

    def _continuation_rows(self, obs, history, cands, horizon):
        """Feature rows for "what could still be done after this fails".

        Featurised once, not once per candidate. A follow-up row depends on
        which first action preceded it only through three history counters, so
        the grid is built a single time and those columns are patched per
        group. Doing it the obvious way -- a featurise call per candidate --
        made each decision an order of magnitude more expensive, which put the
        training sweep out of reach.
        """
        grid = [(a, d) for a, d in CANDIDATES if d <= horizon]
        if not grid:
            return None, None, []

        # One featurisation, with a placeholder failed attempt so the
        # attempt-index column is already correct for the next step.
        placeholder = list(history) + [
            Attempt(Action.RETRY, obs["hour"], "assumed-failed", False)]
        base = featurise_candidates(
            obs, placeholder, self.ctx,
            [a for a, _ in grid], np.array([d for _, d in grid]))
        grid_delays = np.array([d for _, d in grid])
        grid_actions = [a for a, _ in grid]

        blocks, spans, block_actions = [], [], []
        for action, delay in cands:
            sel = np.nonzero(grid_delays >= delay + MIN_GAP_H)[0]
            if sel.size == 0:
                spans.append((0, 0))
                continue
            rows = base[sel].copy()
            # Patch the counter for whichever action was assumed to precede.
            rows[:, COL_PRIOR_RETRIES] = rows[:, COL_PRIOR_RETRIES] - 1
            col = _PRIOR_COL_FOR.get(action)
            if col is not None:
                rows[:, col] = rows[:, col] + 1
            start = sum(b.shape[0] for b in blocks)
            blocks.append(rows)
            block_actions.extend(grid_actions[k] for k in sel)
            spans.append((start, start + rows.shape[0]))

        if not blocks:
            return None, None, []
        return np.vstack(blocks), spans, block_actions
