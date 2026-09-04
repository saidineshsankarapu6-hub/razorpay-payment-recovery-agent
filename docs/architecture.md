# Architecture

**AI Payment Recovery Agent — Razorpay AI Buildathon, AI Revenue Recovery track**

## 1. The problem

A failed payment is a decision, not an outcome. Retry now, retry later, switch
instrument, ask the customer, or stop — every option costs money, and the
reason for the failure is frequently undisclosed.

Razorpay's own documentation is the evidence. `card_declined` and
`payment_failed` are documented as *"the bank declined the transaction"*, with
remediation amounting to *contact your bank, try another card*. **37% of failed
payments carry a code that discloses nothing about why they failed.**

A rules engine must therefore pick one fixed action for a bucket that mixes
insufficient funds, exceeded limits, issuer downtime and risk declines — four
causes whose correct actions are respectively *wait for payday*, *wait 24h*,
*wait for the outage*, and *never retry*.

## 2. System overview

```
                    ┌──────────────────────────────────────┐
                    │      Razorpay error codes (27)       │
                    │   cards + UPI, from public docs      │
                    └──────────────────┬───────────────────┘
                                       │
                            src/taxonomy.py
                    7 recovery classes + OPAQUE_CODES set
                                       │
   ┌───────────────────────────────────┼──────────────────────────────────┐
   │                                   │                                  │
   ▼                                   ▼                                  ▼
┌────────────────┐          ┌────────────────────┐          ┌──────────────────┐
│ src/generator  │          │   src/features     │          │  src/policies    │
│                │          │                    │          │                  │
│ latent cause   │          │ payday estimator   │          │ NoRecovery       │
│   ↓ (lossy)    │          │ issuer stress      │          │ FixedRetry       │
│ observed code  │          │ customer profile   │          │ RuleBased        │
│                │          │                    │          │  (sweep-tuned)   │
│ counterfactual │          │ TRAINING WINDOW    │          │                  │
│ recovery oracle│          │ ONLY               │          │                  │
└───────┬────────┘          └─────────┬──────────┘          └────────┬─────────┘
        │                             │                              │
        │                             ▼                              │
        │                   ┌───────────────────┐                    │
        │                   │    src/agent      │                    │
        │                   │                   │                    │
        │                   │ recovery head ────┼── P(recovery)      │
        │                   │ pushback head ────┼── P(issuer pushback)│
        │                   │ 1-step lookahead  │                    │
        │                   │ hard guardrails   │                    │
        │                   └─────────┬─────────┘                    │
        │                             │                              │
        └─────────────┬───────────────┴──────────────────────────────┘
                      ▼
            ┌─────────────────────┐
            │    src/evaluate     │   common random numbers
            │  scoring harness    │   full costing
            │                     │   observation whitelist
            └──────────┬──────────┘   forward-time enforcement
                       ▼
              held-out comparison
              + audit trail
```

## 3. Data layer — `src/generator.py`

No public dataset of real failed Razorpay transactions exists, so the project
runs on a simulator. That makes the simulator the weakest link, and it is built
to be attacked rather than trusted.

**Latent cause → observed code.** A hidden cause is sampled first, conditioned
on context (liquidity, amount, issuer downtime), then a Razorpay error code is
emitted from it — *lossily*. Several causes share one opaque code, so the
inference problem is a property of the data rather than an assumption baked
into the model. Verified, not asserted: `payment_failed` carries 2.14 bits of
cause ambiguity, `card_declined` 1.07, and a test fails if any opaque code ever
collapses to a single cause.

**Counterfactual recovery oracle.** Each failed payment carries the latent
state needed to answer *"if you took action A at time T, would it succeed?"*
without that attempt having been generated in advance. This is what lets any
policy — baseline or agent — be scored offline on identical transactions.

**One affordability model, used twice.** A customer has spending capacity; the
salary cycle scales how much is available now; funds failure is a logistic in
amount-over-available. Generation and the oracle call the same function, so
"would a retry work later?" is answered by the model that decided the payment
failed in the first place.

**Customer heterogeneity.** 30% of customers are cycle-bound (flush on payday,
empty before it); the rest barely move. Without this the salary effect is a
uniform ~2 percentage points, per-customer inference is impossible, and the
timing thesis is unsupportable.

Scale: 108,000 transactions, 1,200 customers (~90 transactions of history
each), 88.0% success, 12,945 failed payments.

## 4. Feature layer — `src/features.py`

Two signals the simulator knows and the agent must infer.

**Payday estimation.** Never read from `_salary_day`. Success probability rises
after payday and decays across the month, so the payday is the offset whose
implied liquidity curve best explains which of that customer's payments
succeeded — scored across all 30 candidate offsets, from training-window
history only.

| Segment | Within ±2 days |
|---|---|
| Random baseline | 0.167 |
| Cycle-bound customers | **0.476** |
| Comfortable customers | 0.198 |

The second number is the honest half: customers with no salary cycle have
nothing to find, and the confidence score says which is which.

**Issuer stress.** Never read from `_in_downtime`. Computed as the failure rate
on that issuer in the two hours *before* the payment — strictly
backward-looking, so it uses only information a processor holds at decision
time. When a bank is down its other customers are failing too, and that is
visible.

Feature vectors are plain numpy with integer-coded categoricals. The agent
scores ~18 candidates per decision hundreds of thousands of times, and building
a pandas frame per call dominated runtime by an order of magnitude.

## 5. The agent — `src/agent.py`

For each failed payment, 18 candidate (action, delay) pairs are scored:

```
EV(c)    = P(recovery) · amount − cost(c) − P(pushback) · penalty
total(c) = EV(c) + (1 − P(recovery)) · max EV(c′) over c′ reachable after c
```

**Two learned heads**, both trained only on observed retry outcomes — never on
the latent cause, which would not exist in a production log:

| Head | Question | Quality |
|---|---|---|
| Recovery | will this action, at this delay, work? | AUC 0.791, Brier 0.143 |
| Pushback | will this attempt draw issuer pushback? | AUC 0.922 |

Calibration matters more than ranking, because expected value is computed
directly from these probabilities. Top bin predicts 0.725 against actual 0.757.

**The second line of the formula is not decoration.** A one-step-greedy agent
picks its single best moment and spends the whole 7-day horizon reaching it —
measured, that pushed median time-to-recovery to 48h against the rules engine's
2h and dropped easy transient failures from 0.934 to 0.871. Valuing the
sequence is what makes an early cheap attempt worth taking.

**Guardrails sit outside the model.** A learned expected value is the wrong
place to encode "never do this":

1. An explicit risk decline is never pursued — checked before the model runs.
2. Issuer-facing actions require P(recovery) ≥ 0.28. Nudges are exempt; they
   never touch the issuer.

That threshold was not chosen by maximising net value, which picks 0.0 and buys
revenue with the merchant's bank relationships. Issuer friction is a
**constraint**: the best net value among thresholds that put no more traffic in
front of issuers than the incumbent rules engine already does — stated before
results were inspected, applied to training data only.

## 6. Training — `scripts/train_agent.py`

1. Temporal split: 60 days to learn, 30 held out. Never shuffled.
2. Customer profiles and issuer-stress features from the training window.
3. **Exploration** over training-window failures produces the outcome log:
   randomised (action, delay) attempts and whether they recovered. This is the
   analogue of a processor's own retry history — the only supervision that
   would genuinely exist. One episode per payment, not replays of the same
   failure under different actions, which would hand the model counterfactual
   information no real log contains.
4. Fit both heads on that log.
5. Select the guardrail threshold under the friction constraint.

## 7. Evaluation — `src/evaluate.py`

Four properties make the comparison honest:

- **Identical data** — every policy sees the same held-out failures.
- **Common random numbers** — each payment gets its own RNG seeded from its
  `txn_id`, so policies face identical draws rather than sharing a stream whose
  position depends on earlier policies' attempt counts.
- **Observation whitelist** — the policy-visible view is constructed from
  `OBSERVABLE_FIELDS`; latent columns are unreachable from policy code, and a
  test confirms the whitelist is doing real work.
- **Full costing** — net value is revenue minus gateway cost, nudge cost, and
  the goodwill penalty for retrying a risk decline. Recovery rate alone would
  reward retrying everything forever.
- **Forward-time enforcement** — successive attempts must move forward by at
  least 0.5h, so no policy can collect several oracle draws for one instant.

Every decision carries a human-readable reason, which becomes the audit trail
the track calls for: *"retry at +120h | p=0.799 pen=0.01 EV=₹456.7"*.

## 8. Results

Held-out: 4,314 failed payments, ₹95,68,381 at risk.

| Policy | Recovery | Net value | Cost | Fraud attempts |
|---|---|---|---|---|
| do nothing | 0.0% | 0 | 0 | 0 |
| fixed retry | 58.3% | 47,94,502 | 1,74,015 | 603 |
| tuned rules engine | 65.4% | 57,00,314 | 89,542 | 302 |
| **agent** | **70.9%** | **61,63,162** | **76,212** | **244** |

+5.5 points recovery, +8.1% net value, −15% cost, −19% issuer friction. It wins
on every axis rather than trading one for another.

## 9. What we would build next

Stated as roadmap, deliberately unbuilt:

- **Learning loop** — recovery outcomes feed back into both heads continuously.
- **Nudge generation** — LLM-written recovery copy, tone matched to inferred
  cause; multilingual is a genuine India differentiator. Channel selection
  (SMS / WhatsApp / email) per customer.
- **Route and PSP switching** — retry through a different acquirer, not only at
  a different time.
- **Subscription / e-mandate handling** — RBI retry rules apply and differ from
  one-off payments.
- **Per-merchant policy configuration** — the guardrail frontier in §5 is
  already a dial; merchants should own it.
- **Human escalation queue** for high-value unrecovered payments.

## 9a. Security posture

Sixteen adversarial tests (20 assertions) attack the guardrails rather than
confirm them; they
found five real vulnerabilities, two HIGH, all now fixed. Full write-up in
[security.md](security.md). The two that mattered:

- **Fraud suppression was defeated by string casing.**
  `PAYMENT_RISK_CHECK_FAILED` bypassed `never_retry()` entirely — six of six
  case and whitespace variants got through. Codes are now canonicalised before
  any lookup.
- **NaN-scheduled attempts skipped every bounds check**, because every
  comparison against NaN is False. 947 attempts and INR 14,367 of cost executed
  in a 300-payment test. Non-finite times are now rejected explicitly.

The design property worth noting is that neither a *missing* pushback model nor
one returning P=1.0 for everything can defeat the explicit fraud block — it is
enforced in code before any model is consulted. That is the whole argument for
keeping "never do this" out of a learned expected value.

Known accepted risk: `models/agent.pkl` is loaded with `pickle`, which executes
arbitrary code on load. Acceptable for a locally-produced prototype artifact,
documented rather than hidden, and listed with its remediations in
[security.md](security.md).

## 10. Known limitations

Stated plainly:

- **Absolute rupee figures are illustrative.** Oracle probabilities are
  hand-set from payments domain knowledge, since no real data exists to fit
  them. The defensible claim is the *relative* lift of one policy over another
  on identical transactions through an identical oracle.
- **Repeat-commerce assumption.** Per-customer payday inference needs ~90
  transactions of history. A merchant seeing each customer twice could not run
  that half of the agent.
- Customers are independent — no basket effects or cross-merchant history.
- Nudge response is a single scalar, not a model of channel, language, timing
  and content.
- Issuer downtimes are independent of each other; no network contagion.

## 11. Reproduce

```bash
pip install numpy pandas pyarrow scikit-learn

python -m scripts.make_dataset      # seeded -> byte-identical data
python -m scripts.tune_baseline     # strongest rules playbook (training window)
python -m scripts.run_baselines     # score the baselines
python -m scripts.train_agent       # exploration log -> two heads -> guardrail
python -m scripts.run_evaluation    # held-out comparison
python -m scripts.demo_cases        # side-by-side case studies
python -m tests.test_integrity      # 27 checks: the evaluation is honest
python -m tests.test_vulnerabilities  # 20 attacks on the guardrails
```
