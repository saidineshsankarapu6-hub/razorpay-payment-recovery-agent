# Baseline results

Recorded **before** the agent was built, so the comparison cannot be
reverse-engineered to flatter it. Reproduce with:

```
python -m scripts.make_dataset
python -m scripts.tune_baseline
python -m scripts.run_baselines
```

Scope: the **held-out evaluation window** — 4,314 failed payments,
**INR 9,568,381 at risk**, 7-day recovery horizon, maximum 4 attempts.
Playbooks were tuned on the earlier training window only.

## Results

| Policy | Recovery rate | Revenue recovered | Cost | Net value | Attempts/payment | Fraud attempts |
|---|---|---|---|---|---|---|
| `no_recovery` | 0.0% | 0 | 0 | 0 | 0.00 | 0 |
| `fixed_retry` | 58.3% | 4,968,517 | 174,015 | 4,794,502 | 2.16 | 603 |
| **`rule_based`** | **65.4%** | **5,789,857** | **89,542** | **5,700,314** | **1.86** | **302** |

The bar for the agent is therefore **65.4% recovery / INR 5,700,314 net**, not
the naive retry policy and certainly not zero. What the agent achieved against
it: [agent.md](agent.md).

## The baselines were strengthened, not handicapped

An agent that beats a strawman has proved nothing. Both branches where a rules
engine has real freedom were swept exhaustively and the **winning** variant was
adopted into the baseline ([scripts/tune_baseline.py](../scripts/tune_baseline.py)):

| Branch | First draft | Sweep winner | Effect (training window) |
|---|---|---|---|
| `HARD_DECLINE` | retry, retry, switch | **nudge, retry, switch** | 64.9% → 66.4% overall |
| `FUNDS` | nudge, retry@24h | **nudge, retry@48h, retry@120h** | FUNDS 65.6% → 75.1% |

Nudging first on the opaque bucket also cut fraud penalties by a third, because
a nudge never reaches the issuer. Both improvements made the agent's job
harder, and both were kept.

## Recovery rate by class

| Class | `fixed_retry` | `rule_based` |
|---|---|---|
| SOFT_TRANSIENT | 0.934 | 0.934 |
| FUNDS | 0.695 | **0.744** |
| HARD_DECLINE | 0.619 | **0.627** |
| METHOD_BROKEN | 0.166 | **0.595** |
| AUTH_FAILED | 0.431 | 0.402 |
| ABANDONED | 0.273 | **0.433** |
| FRAUD | 0.000 | 0.000 |

`FRAUD` at 0.000 for both policies is the oracle behaving correctly: risk
declines are unrecoverable by construction, so any policy retrying them is
buying nothing.

## Where the headroom is

Three gaps the rules engine cannot close, because none of them is a property
of the error code it can see:

**1. It retries risk declines it cannot identify — 302 attempts, INR 75,500
in penalties.** `rule_based` already suppresses the explicit
`payment_risk_check_failed`. But roughly two thirds of risk declines surface as
an opaque code instead, and there is no rule that separates them from the
NO_FUNDS declines sharing the same string.

**2. Its timing is the same for every customer.** The sweep found that waiting
48h beats waiting 24h on FUNDS *on average* — but the average is not where the
money is. Targeting an individual customer's payday recovers 80.1% of NO_FUNDS
failures for cycle-bound customers, against 38.4% for fixed +24h. A fixed
schedule cannot reach that, because the right moment differs per customer.

**3. One action must cover several causes.** 37% of failures carry an opaque
code.

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
