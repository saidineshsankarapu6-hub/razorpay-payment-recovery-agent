# AI Payment Recovery Agent

**Razorpay AI Buildathon — AI Revenue Recovery track**

A failed payment is not a lost sale. It is a decision problem: retry now, retry
later, switch instrument, ask the customer, or stop — where every option costs
money and the reason for the failure is frequently undisclosed.

## The problem, in Razorpay's own words

Razorpay documents `card_declined` and `payment_failed` as *"the bank declined
the transaction"*, with remediation amounting to *contact your bank, try
another card*. That vagueness is not a documentation gap — it is what the
issuer actually returned.

**37% of failed payments in this dataset carry a code that discloses nothing
about why they failed.** A rules engine must pick one fixed action for that
entire bucket. Three payments, one identical error code:

| Code | True cause | Correct action |
|---|---|---|
| `card_declined` | daily limit exceeded | retry after limits reset |
| `card_declined` | insufficient funds | retry on the customer's payday |
| `card_declined` | risk decline | **never retry** |

A lookup table gets two of these wrong. Getting the third wrong costs real
money. On one held-out `card_declined` payment of INR 7,987 that was actually a
risk decline, no policy recovers it — because it is unrecoverable — but the
cost of finding that out differs by two orders of magnitude:

| Policy | Cost | What it did |
|---|---|---|
| fixed retry | INR 757.50 | three retries, three issuer penalties |
| rules engine | INR 505.90 | nudge, retry, switch — two penalties |
| **agent** | **INR 3.60** | four nudges, never touched the issuer |

The agent was never told this was a risk decline. It inferred that retrying
would draw pushback. Reproduce with `python -m scripts.demo_cases`.

## Results

Held-out window: **4,314 failed payments, INR 9,568,381 at risk.** Nothing
scored here was fitted here.

| Policy | Recovery | Net value | Cost | Fraud attempts |
|---|---|---|---|---|
| do nothing | 0.0% | 0 | 0 | 0 |
| fixed retry (+1h/+24h/+72h) | 58.3% | 4,794,502 | 174,015 | 603 |
| tuned rules engine | 65.4% | 5,700,314 | 89,542 | 302 |
| **agent** | **70.9%** | **6,163,162** | **76,212** | **244** |

**+5.5 points of recovery, +8.1% net value, 15% lower cost, and 19% fewer
issuer-facing attempts on risk declines.** It wins on every axis rather than
trading one for another. Full detail: [docs/agent.md](docs/agent.md).

## How it works

For each failed payment the agent scores 18 candidate (action, delay) pairs:

```
EV(c)    = P(recovery) * amount - cost(action) - P(issuer pushback) * penalty
total(c) = EV(c) + (1 - P(recovery)) * max EV(c') over c' reachable after c
```

The second line values the *sequence*, not just the next action — a one-step
greedy agent spends the whole 7-day window reaching its single best moment and
has nothing left if it fails.

Two learned heads, both trained only on observed retry outcomes — never on the
latent cause, which would not exist in a production log:

- **Recovery head** — will this action, at this delay, work? AUC 0.791, and
  well calibrated (top bin predicts 0.725, actual 0.757), which matters because
  EV is computed straight from these probabilities.
- **Pushback head** — will this attempt draw issuer pushback? AUC 0.922.

Two signals the simulator knows but the agent must infer:

- **The customer's payday** — estimated from that customer's own transaction
  outcomes. 47.6% within ±2 days for cycle-bound customers against a 16.7%
  random baseline, and near-chance for customers with no salary cycle, which is
  the correct answer for them. Its confidence score says which is which.
- **Issuer downtime** — inferred from the failure rate on that issuer in the
  two hours *before* the payment. Strictly backward-looking.

Guardrails live outside the model: an explicit risk decline is never pursued,
and issuer-facing actions need P(recovery) ≥ 0.28. A learned expected value is
the wrong place to encode "never do this".

That threshold was **not** chosen by maximising net value — doing so picks 0.0
and buys revenue with issuer goodwill. Issuer friction is treated as a
constraint: the best net value among thresholds that put no more traffic in
front of issuers than the incumbent rules engine already does. Stated before
looking at results, applied to training data only.

## What makes the numbers trustworthy

This runs on a simulator, because no public dataset of real failed Razorpay
transactions exists. That makes the simulator the weakest link, so it is the
part most heavily defended:

- **Latent cause → observed code.** A hidden cause is sampled first, then an
  error code emitted from it — lossily, so several causes share one opaque
  code. The inference problem is a property of the data, not an assumption.
- **A counterfactual oracle.** Every failed payment carries the state needed to
  answer *"would this action, at this time, have worked?"*, so any policy is
  scored offline on identical transactions.
- **A temporal split.** 60 days to learn, 30 held out. Never shuffled: a policy
  that learned from the future would be worthless in production.
- **Baselines tuned to win, not to lose.** Both branches where a rules engine
  has freedom were swept and the strongest variant adopted — on training data
  only, exactly like the agent. The first draft scored 58.5%; the version the
  agent must beat scores 65.4%.
- **Policies cannot see ground truth.** Observations come from a whitelist,
  enforced in code and covered by tests.
- **Costs are counted.** Net value is revenue minus gateway cost, nudge cost,
  and the goodwill penalty for retrying a risk decline.
- **27 integrity checks**, including one asserting that opaque codes stay
  genuinely ambiguous — if that fails, the project's premise is invalid — and
  one that stops any policy collecting multiple oracle draws for one instant.
- **20 adversarial tests** ([docs/security.md](docs/security.md)) that attack
  the guardrails rather than confirm them. They found five real vulnerabilities,
  two HIGH — including fraud suppression being defeated by a capital letter, and
  NaN-scheduled attempts skipping every bounds check. All fixed.

**Full architecture:** [docs/architecture.md](docs/architecture.md).
Assumptions and limitations: [docs/data_assumptions.md](docs/data_assumptions.md).
Absolute rupee figures are illustrative; the defensible claim is the *relative*
lift of one policy over another on identical data.

## Three bugs worth reading about

All caught by measurement, and all recorded in [docs/agent.md](docs/agent.md)
rather than quietly fixed:

- **The agent was gaming the harness.** Candidate delays were measured from the
  original failure, so after failing at +48h it proposed +48h again — and the
  stochastic oracle handed it a fresh draw. Four rolls of the dice at one
  instant. Closing it dropped training recovery from 0.759 to 0.663: the
  exploit had been doing most of the work. Now enforced in the harness for
  every policy, with a regression test.
- **One-step myopia.** With the exploit closed the agent barely beat the
  baseline. It was spending the whole 7-day horizon reaching its single best
  moment — median time-to-recovery 48h vs the rules engine's 2h. Scoring the
  sequence rather than the next action fixed it.
- **The agent was making fraud worse** — 541 issuer-facing attempts on risk
  declines against the baseline's 302, with 88% of its cost being penalties.
  Its EV priced every retry at ₹2.5 with no notion that some cost ₹250.

## Layout

```
src/taxonomy.py    27 Razorpay error codes -> 7 recovery classes
src/generator.py   simulator + counterfactual recovery oracle
src/features.py    payday estimation, issuer stress, feature construction
src/policies.py    baseline policies; the Policy interface
src/agent.py       the recovery agent
src/evaluate.py    scoring harness (common random numbers, full costing)
scripts/           dataset build, baseline tuning, training, evaluation
tests/             integrity tests
docs/              architecture, taxonomy, assumptions, baselines, agent,
                   security, pitch
```

## Reproduce

```bash
pip install numpy pandas pyarrow scikit-learn

python verify.py            # everything, end to end (~25 min)
python verify.py --quick    # both test suites only (~90s)
```

Or run the stages individually:

```bash

python -m scripts.make_dataset      # seeded -> byte-identical data
python -m scripts.tune_baseline     # strongest rules playbook (training window)
python -m scripts.run_baselines     # score the baselines
python -m scripts.train_agent       # exploration log -> two models -> guardrail
python -m scripts.run_evaluation    # held-out comparison
python -m scripts.demo_cases        # side-by-side case studies
python -m tests.test_integrity      # 27 checks: the evaluation is honest
python -m tests.test_vulnerabilities  # 20 attacks on the guardrails
```
