# The recovery agent

## Results

Held-out evaluation window: **4,314 failed payments, INR 9,568,381 at risk.**
Nothing scored here was fitted here — both models, the guardrail threshold, and
the rules engine's playbooks were chosen on the earlier 60-day training window.

| Policy | Recovery | Value rate | Revenue recovered | Cost | Net value | Fraud attempts |
|---|---|---|---|---|---|---|
| do nothing | 0.0% | 0.0% | 0 | 0 | 0 | 0 |
| fixed retry | 58.3% | 51.9% | 4,968,517 | 174,015 | 4,794,502 | 603 |
| tuned rules engine | 65.4% | 60.5% | 5,789,857 | 89,542 | 5,700,314 | 302 |
| **agent** | **70.9%** | **65.2%** | **6,239,374** | **76,212** | **6,163,162** | **244** |

Against the strongest baseline: **+5.5 points of recovery, +8.1% net value,
15% lower cost, and 19% fewer issuer-facing attempts on risk declines.** It
wins on every axis rather than trading one for another.

By recovery class:

| Class | fixed retry | rules | agent |
|---|---|---|---|
| SOFT_TRANSIENT | 0.934 | 0.934 | **0.936** |
| FUNDS | 0.695 | 0.744 | **0.795** |
| HARD_DECLINE | 0.619 | **0.627** | 0.615 |
| METHOD_BROKEN | 0.166 | 0.595 | **0.695** |
| AUTH_FAILED | 0.431 | 0.402 | **0.655** |
| ABANDONED | 0.273 | 0.433 | **0.646** |
| FRAUD | 0.000 | 0.000 | 0.000 |

HARD_DECLINE is the one class where the rules engine edges ahead, and that is
the guardrail working as intended: the opaque bucket is where hidden risk
declines live, so the agent declines some marginal retries there. It gives up
1.2 points of recovery in that bucket and takes 58 fewer penalties overall.

## How it decides

For each failed payment the agent scores 18 candidate (action, delay) pairs and
takes the best, or stops:

```
EV(c)    = P(recovery) * amount - cost(action) - P(issuer pushback) * penalty
total(c) = EV(c) + (1 - P(recovery)) * max EV(c') over c' reachable after c
```

The second line is the part that matters. See "one-step myopia" below.

Two learned heads, both trained only on observed retry outcomes:

- **Recovery head** — will this action, at this delay, recover the payment?
  ROC AUC 0.791, Brier 0.143 against a base rate of 0.259. Calibration matters
  more than ranking, because EV is computed directly from these numbers; the
  top bin predicts 0.725 against an actual 0.757.
- **Pushback head** — will this attempt draw issuer pushback? ROC AUC 0.922.

## What the agent is never given

- **The latent cause.** It trains on what a processor genuinely logs:
  randomised recovery attempts and whether they worked. One exploration episode
  per training payment — not replays of the same failure under different
  actions, which would hand the model counterfactual information no real log
  contains.
- **The customer's payday.** Estimated from that customer's own outcomes.
- **Issuer downtime.** Inferred from the failure rate on that issuer in the two
  hours *before* the payment — strictly backward-looking.
- **Any evaluation-window data**, for any purpose.

## Estimating payday without being told it

Success probability rises after payday and decays across the month, so a
customer's payday is the offset whose implied liquidity curve best explains
which of their payments succeeded. Scored across all 30 candidate offsets, from
training-window history only.

| Segment | Within ±2 days | Median error |
|---|---|---|
| Random baseline | 0.167 | — |
| **Cycle-bound customers (n=361)** | **0.476** | 3 days |
| Comfortable customers (n=839) | 0.198 | 7 days |

The second row is the honest half. Customers whose balance does not track their
salary have no cycle to find, and the estimator does not pretend otherwise. Its
confidence score reflects this — the top confidence quartile is 57.7%
cycle-bound and 45.3% accurate, the bottom quartile 18.0% — so the agent knows
when to trust its own timing signal.

This matters because the value is concentrated in exactly that segment: payday
targeting is worth **2.09×** on cycle-bound customers and 1.16× on everyone
else.

## Guardrails sit outside the model

A learned expected value is the wrong place to encode "never do this":

1. **An explicit risk decline is never pursued.** Checked before the model is
   consulted at all.
2. **Issuer-facing actions require P(recovery) ≥ 0.28.** Nudges are exempt —
   they never touch the issuer.

### How that threshold was chosen

Maximising net value alone selects 0.0, because under this cost model a ₹250
penalty is cheap next to the revenue an occasional extra retry brings in. That
is the right answer to a narrow question and the wrong answer to the real one:
a recovery product that degrades a merchant's standing with its banks to book
short-term revenue is not deployable.

So issuer friction is a **constraint, not another term to trade away**. The
rule — stated before any result was inspected, and applied to training data
only — is: *the highest net value among thresholds that put no more traffic in
front of issuers than the incumbent rules engine already does.* On the training
subsample the rules engine made 198 such attempts; 0.28 was the best threshold
under that budget.

The frontier is a documented dial, measured on held-out data:

| p_floor | Recovery | Net value | Fraud attempts |
|---|---|---|---|
| 0.00 | 0.7242 | 6,296,820 | 497 |
| 0.18 | 0.7218 | 6,272,901 | 467 |
| **0.28 (selected)** | **0.7086** | **6,163,162** | **244** |
| 0.40 | 0.6808 | 5,932,996 | 74 |

A merchant that weights issuer standing more heavily moves down this table.
Every row still beats the rules engine's 65.4% recovery.

## One code, three truths

Every payment below returned `card_declined`. Razorpay documents that as *"the
bank declined the transaction"* and nothing more. Reproduce with
`python -m scripts.demo_cases`; the true cause is shown for the reader only.

**True cause: insufficient funds — INR 11,306**

| Policy | Outcome | Cost | Actions |
|---|---|---|---|
| fixed retry | recovered | 2.50 | retry@+1h |
| rules engine | recovered | 0.90 | nudge@+1h |
| agent | recovered | 0.90 | nudge@+0.5h |

**True cause: risk decline — INR 7,987**

| Policy | Outcome | Cost | Actions |
|---|---|---|---|
| fixed retry | not recovered | **757.50** | retry@+1h -> retry@+24h -> retry@+72h |
| rules engine | not recovered | **505.90** | nudge@+1h -> retry@+24h -> switch@+48h |
| **agent** | not recovered | **3.60** | nudge@+0.5h -> nudge@+2h -> nudge@+12h -> nudge@+24h |

Nobody recovers this one, because it is unrecoverable. The difference is what
each policy spends finding that out. The baselines put it in front of the
issuer three and two times respectively and pay for it; the agent works the
customer side only and spends INR 3.60. It was never told this was a risk
decline -- it inferred that retrying would draw pushback.

**True cause: daily limit exceeded — INR 6,841**

| Policy | Outcome | Cost | Actions |
|---|---|---|---|
| fixed retry | recovered | 5.00 | retry@+1h -> retry@+24h |
| rules engine | recovered | 5.90 | nudge@+1h -> retry@+24h -> switch@+48h |
| agent | **not recovered** | 3.60 | four nudges |

Included because it is a loss. Here the agent's caution costs it: the limit
resets on its own and a plain retry would have worked, but it read the case as
risky and stayed on the customer side. This is the same instinct that saved
INR 502 on the row above, and the same instinct is why HARD_DECLINE is the one
class where the rules engine still edges ahead.

## Three bugs worth recording

All three were caught by measurement rather than review.

**1. The agent was gaming the harness.** Its candidate delays were measured
from the original failure time, so after a failed attempt at +48h it would
propose +48h again — and the stochastic oracle would hand it a fresh draw.
Sequences like `RETRY+48h, RETRY+48h, RETRY+48h, RETRY+48h` were four
independent rolls of the dice at one instant, which no real processor gets.
Closing it dropped training recovery from 0.759 to 0.663, so the exploit had
been doing most of the work. The rule is now enforced in the harness for every
policy, not trusted to the agent, and covered by a regression test.

**2. One-step myopia.** With the exploit closed the agent barely beat the
baseline (+0.5pp recovery, −1.1% net value). It was choosing the single
highest-EV moment and spending the entire 7-day horizon reaching it: median
time-to-recovery 48h against the rules engine's 2h, and easy transient failures
falling to 0.871 from 0.934. Scoring the *sequence* — an action's own value
plus what remains reachable if it fails — fixed it. "Retry at +1h, and if that
fails you still have six days" beats "retry once at +168h" even when the later
attempt looks better alone.

**3. The agent was making fraud worse.** An earlier run raised issuer-facing
attempts on risk declines from 302 to 541, with 88% of its total cost being
penalties. The EV formula priced every retry at the ₹2.5 gateway fee with no
notion that some cost ₹250, so it rationally chased large payments it could
never recover. The pushback head fixed the pricing; the friction budget fixed
the threshold.

A fourth, smaller one: the first threshold sweep ran to 0.18 and 0.18 won — the
endpoint of its own range. A threshold search whose best value sits on its
boundary has not converged.
