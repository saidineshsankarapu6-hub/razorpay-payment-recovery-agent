# 5-minute pitch video — shot script

Total 5:00. Screen recording + voiceover. Every number on screen is
reproducible by a command in this repo, so record the terminal live where you
can — a judge who sees real output trusts the rest.

**Before recording:** run the full pipeline once so `results/` is populated and
nothing has to be generated on camera.

---

## 0:00–0:35 — The problem, in Razorpay's own words

*On screen:* Razorpay's [cards error codes page](https://razorpay.com/docs/errors/payments/cards/),
scrolled to `card_declined`.

> "This is Razorpay's own documentation for `card_declined`. It says: the bank
> declined the transaction. Contact your bank, try another card. That's not a
> gap in the docs — that's what the issuer actually returned.
>
> In my dataset, **37% of failed payments carry a code like this** — a code
> that tells you nothing about why it failed."

*Cut to:* the three-row table.

| Code | True cause | Correct action |
|---|---|---|
| `card_declined` | daily limit exceeded | retry after limits reset |
| `card_declined` | insufficient funds | retry on the customer's payday |
| `card_declined` | risk decline | **never retry** |

> "Same code. Three different right answers. A rules engine has to pick one."

---

## 0:35–1:15 — The money shot

*On screen:* run it live — `python -m scripts.demo_cases`

> "Here's one real held-out payment. ₹7,987. Code says `card_declined`. It was
> actually a risk decline — the issuer's fraud system said no.
>
> Nobody recovers it. It's unrecoverable. Watch what each policy *spends*
> finding that out."

| Policy | Cost | What it did |
|---|---|---|
| fixed retry | **₹757.50** | three retries, three issuer penalties |
| rules engine | **₹505.90** | nudge, retry, switch — two penalties |
| **my agent** | **₹3.60** | four nudges, never touched the issuer |

> "Two orders of magnitude. And the agent was never told this was fraud — it
> inferred that retrying would draw issuer pushback."

**Pause here.** This is the moment the judges remember.

---

## 1:15–2:15 — How it works

*On screen:* the architecture diagram from `docs/architecture.md` §2.

> "For every failed payment the agent scores 18 candidate actions — retry now,
> retry in two days, switch instrument, nudge the customer — and picks by
> expected value."

```
EV(c)    = P(recovery) · amount − cost(c) − P(pushback) · penalty
total(c) = EV(c) + (1 − P(recovery)) · best value still reachable after c
```

> "Two learned heads. One predicts whether an action recovers the payment —
> AUC 0.79, and well calibrated, which matters because I'm computing rupees
> straight from these probabilities. The second predicts whether the attempt
> draws issuer pushback — AUC 0.92.
>
> Neither is ever trained on the true cause. They're trained on what a payment
> processor actually logs: retries, and whether they worked."

*Beat on the second line of the formula.*

> "That second line values the *sequence*. I'll come back to why."

---

## 2:15–3:00 — Inferring what you're not told

> "Two things my simulator knows that the agent is not allowed to read.
>
> **When the customer gets paid.** I estimate it from the shape of their own
> transaction history — success rises after payday and decays across the month,
> so payday is the offset that best explains which of their payments went
> through."

| Segment | Within ±2 days |
|---|---|
| Random guessing | 0.167 |
| Cycle-bound customers | **0.476** |
| Everyone else | 0.198 |

> "Nearly three times better than chance for customers who live paycheck to
> paycheck — and barely better than chance for everyone else. **That's
> correct.** People whose balance doesn't track their salary have no cycle to
> find, and the model's confidence score tells the agent which case it's in.
>
> That matters because payday targeting is worth **2.1×** on cycle-bound
> customers and only 1.16× on the rest. The value is concentrated exactly where
> the estimator works.
>
> **And whether the bank is down.** Inferred from the failure rate on that
> issuer in the two hours *before* the payment. When a bank breaks, its other
> customers are failing too — that's visible without any special feed."

---

## 3:00–3:45 — Results

*On screen:* run `python -m scripts.run_evaluation` live, or show its output.

> "Held out: 4,314 failed payments the agent has never seen, ₹95.7 lakh at
> risk. Nothing here was tuned on this window."

| Policy | Recovery | Net value | Cost | Issuer hits on fraud |
|---|---|---|---|---|
| do nothing | 0.0% | 0 | 0 | 0 |
| fixed retry | 58.3% | ₹47.9L | ₹1.74L | 603 |
| tuned rules engine | 65.4% | ₹57.0L | ₹89.5K | 302 |
| **agent** | **70.9%** | **₹61.6L** | **₹76.2K** | **244** |

> "+5.5 points of recovery. +8.1% net value. 15% lower cost. And 19% *fewer*
> issuer-facing attempts on payments that were never recoverable.
>
> It wins on every axis instead of trading one for another — and that matters,
> because the easy way to win a recovery benchmark is to retry everything and
> quietly damage the merchant's bank relationships."

*Emphasise:*

> "That rules engine isn't a strawman. I swept both branches where it has real
> freedom and adopted the *winning* variant. My first version scored 58.5%. The
> one I'm comparing against scores 65.4%. I made my own job harder on purpose."

---

## 3:45–4:30 — What went wrong

**Do not skip this section.** It is what separates a demo from engineering.

> "Three things broke, and all three were caught by measurement, not by reading
> the code.
>
> **One — the agent was cheating.** Its retry delays were measured from the
> original failure, so after failing at +48 hours it would propose +48 hours
> again. My simulator rolled fresh dice each time. It had found four free
> attempts at a single instant. Closing that hole dropped training recovery
> from 0.76 to 0.66 — **the exploit had been doing most of the work.** If I'd
> shipped my first result, it would have been fiction. It's now enforced in the
> harness for every policy, with a regression test.
>
> **Two — it was short-sighted.** With the exploit closed it barely beat the
> baseline. It was picking its single best moment and burning the entire
> seven-day window to reach it — median recovery time 48 hours versus the rules
> engine's 2. That's the sequence term in the formula. 'Retry in an hour, and
> if that fails you still have six days' beats 'retry once next week.'
>
> **Three — it was making fraud worse.** At one point it recovered more than
> the baseline while making *more* issuer attempts on risk declines, and 88% of
> its total cost was penalties. Its formula priced every retry at the ₹2.50
> gateway fee, with no idea some cost ₹250. That's the pushback head."

---

## 4:30–5:00 — Honesty and close

> "Two things I want to be straight about.
>
> There's no public dataset of real failed Razorpay transactions, so this runs
> on a simulator I built. **The absolute rupee figures are illustrative** — the
> oracle's probabilities are hand-set from domain knowledge. What's defensible
> is the *relative* lift of one policy over another on identical transactions.
> Every assumption is a named constant, documented, and changeable in one line.
>
> And there's a case in my demo where the agent **loses** — a limit-exceeded
> payment the baselines recover and it misses, because it played cautious. I
> left it in.
>
> Everything is reproducible: six commands, seeded, 27 integrity tests
> including one that fails if my core premise stops holding.
>
> Thanks."

---

## Recording notes

- **Terminal, not slides**, wherever possible. Live output beats a screenshot.
- Large font. Judges may watch on a laptop.
- The 0:35 case study and the 3:45 bug section are the two moments that
  differentiate you. Give them air; cut elsewhere if you run long.
- If you must cut for time, drop 2:15–3:00 (payday inference) before touching
  the bug section.
- Don't apologise for the simulator — state it once, plainly, and move on.
  Volunteering the limitation reads as confidence; being caught on it doesn't.
