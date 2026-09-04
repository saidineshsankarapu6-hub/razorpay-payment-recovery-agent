# Data assumptions

There is no public dataset of real failed Razorpay transactions, so this
project runs on a simulator. That makes the simulator — not the model — the
weakest link in every number reported downstream. This document exists so that
each assumption can be challenged individually, and so that the honest answer
to "where did these numbers come from?" is a page rather than a shrug.

Every assumption below is a single named constant in `SimConfig`
([src/generator.py](../src/generator.py)). Changing one and re-running the
evaluation is a one-line operation, by design.

## What is grounded vs. what is assumed

**Grounded in Razorpay's published documentation:**

- The 27 error-code strings, and their documented meanings and remediation
  steps ([cards](https://razorpay.com/docs/errors/payments/cards/),
  [UPI](https://razorpay.com/docs/errors/payments/upi/)).
- The existence of generic, non-disclosing decline codes. Razorpay documents
  `card_declined` and `payment_failed` as "bank declined the transaction" with
  advice amounting to *contact your bank / try another card*. That documented
  vagueness is the problem this project addresses.

**Assumed, and calibrated to published industry ranges:**

| Assumption | Value | Basis |
|---|---|---|
| Overall success rate | 88.0% | Indian online payment success is generally reported in the 85–92% band |
| UPI success | ~89% | UPI runs above cards; technical declines remain material |
| Card success | ~85% | Cards lag UPI, largely on auth friction and issuer declines |
| Method mix | UPI 62 / card 22 / netbanking 8 / wallet 8 | UPI-dominant, consistent with Indian volumes |
| Median ticket | ~INR 600 | Lognormal per merchant category |
| Cycle-bound customers | 30% | Share of customers whose balance strongly tracks their salary |

Per-method rates in `base_success` are set ABOVE these blended figures on
purpose: they represent success *given funds are available*, because funds
failures are modelled separately. Subtracting both would double-count.

**Assumed, structural — chosen because the structure is what makes the
problem interesting, not to flatter the result:**

- **Issuer downtime is bursty, not per-transaction noise.** Modelled as windows
  affecting all customers on an issuer simultaneously. If downtime were i.i.d.
  noise, "wait for the window to clear" would be a coin flip rather than a
  learnable strategy — and real bank downtime is unambiguously clustered.
- **Liquidity follows a salary sawtooth, but not for everyone.** Customers are
  assigned a payday from {1, 7, 25, 30}; available funds peak on payday and
  decay with an 11-day time constant. Crucially each customer also has a
  *salary sensitivity*: 30% are cycle-bound (flush on payday, empty before it)
  and the rest barely move. A uniform population-wide cycle would be both
  unrealistic and uninteresting — if everyone had the same mild cycle, no
  per-customer inference would be needed or possible.
- **Funds failures come from an explicit balance model**, not an abstract
  penalty: a customer has a spending capacity, the cycle scales how much is
  available now, and failure probability is a logistic in amount-over-available.
  Generation and the recovery oracle call the same function, so "would a retry
  work later?" is answered by the model that decided the payment failed.
- **Ticket sizes track capacity.** Without this, low-capacity customers
  attempted purchases far beyond their means at the same rate as everyone else,
  which swamped the salary signal and dragged population success to 66%.
- **Daily limits reset at 24h.** Drives the LIMIT_EXCEEDED recovery curve.
- **Risk declines are never recoverable.** Retrying one earns no revenue and
  incurs a goodwill penalty. This is asserted, not learned, and the agent
  enforces it as a hard guardrail rather than a model output.

## The two properties that had to be verified, not assumed

A simulator can accidentally make its own thesis true. Both of these were
checked against the generated data rather than asserted:

**1. Opaque codes must genuinely hide multiple causes.** If each opaque code
resolved to one cause, a lookup table would win and the project would have no
reason to exist.

| Opaque code | Entropy (bits) | Causes mixed |
|---|---|---|
| `payment_failed` | 2.14 | 5 |
| `payment_declined` | 2.04 | 5 |
| `credit_failed` | 1.21 | 3 |
| `card_declined` | 1.07 | 3 |

**37.1% of all failures carry an opaque code**, and `card_declined` retains a
RISK_BLOCK tail — the minority case where a naive retry policy does active
harm. This property is asserted by a test, not just measured once: if any
opaque code ever collapses to a single cause, the suite fails.

An earlier revision of the emission table gave `payment_failed` an entropy of
**0.00** — it was emitted only by ISSUER_DOWN and was therefore perfectly
decodable by lookup. That was a generator artefact, not a fact about payments,
and it was fixed by emitting the code from five causes as its documented
"generic decline" meaning implies.

**2. Retry timing must carry signal that fixed backoff cannot capture.**

Recovery rate by retry delay, by latent cause:

| Cause | +0.5h | +6h | +24h | +72h | +168h |
|---|---|---|---|---|---|
| ISSUER_DOWN | 0.655 | 0.858 | 0.880 | 0.897 | 0.901 |
| LIMIT_EXCEEDED | 0.052 | 0.061 | 0.731 | 0.713 | 0.728 |
| GATEWAY_ISSUE | 0.829 | 0.799 | 0.836 | 0.825 | 0.807 |
| AUTH_FAIL | 0.155 | 0.198 | 0.163 | 0.160 | 0.184 |
| NO_FUNDS | 0.472 | 0.501 | 0.498 | 0.490 | 0.535 |
| RISK_BLOCK | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

NO_FUNDS looks nearly flat here — and that is the point. A fixed delay lands at
a random point in each customer's salary cycle, so the effect averages away.
Targeting the customer's next payday instead:

| Policy | All NO_FUNDS | Cycle-bound customers | Comfortable customers |
|---|---|---|---|
| Fixed +24h | 0.511 | 0.384 | 0.643 |
| Fixed +72h | 0.544 | — | — |
| **Next payday** | **0.761** | **0.801** | 0.744 |

The split is the whole argument. Payday targeting is worth **2.09×** on
cycle-bound customers and only 1.16× on everyone else — so the value is
concentrated precisely in the segment the payday estimator can actually
identify, and the estimator's confidence score is what tells the agent which
segment it is looking at. A fixed schedule cannot reach any of it.

An earlier revision of the affordability model produced only ~2 percentage
points of liquidity swing, which made per-customer payday inference impossible
— the estimator scored 18.8% within ±2 days against a 16.7% random baseline.
Rather than build a timing story on an estimator that did not work, the
affordability model was rewritten around an explicit balance model. See
[agent.md](agent.md) for what the estimator achieves now, including where it
correctly fails.

## Known limitations

Stated plainly, because a panel will find them anyway:

- **Customers are independent.** No basket effects, no cross-merchant history.
- **No network-level contagion.** Issuer downtimes are independent of each other.
- **Nudge response is a single scalar** (`engagement`) rather than a model of
  channel, language, timing, and message content.
- **Recovery probabilities in the oracle are hand-set**, not fitted to
  observed data — because no such observed data is available. The *relative
  ordering* is defensible from payments domain knowledge; the absolute levels
  are not claimed to be precise.
- **Consequently, absolute rupee figures from this project are illustrative.**
  The defensible claim is the *relative* lift of one policy over another,
  measured on identical transactions through an identical oracle.
