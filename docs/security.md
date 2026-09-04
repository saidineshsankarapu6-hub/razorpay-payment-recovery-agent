# Security and robustness

This is a prototype, not a deployed payments system. This page records what an
operator would need to know before it became one, and what the adversarial test
suite ([tests/test_vulnerabilities.py](../tests/test_vulnerabilities.py))
actually found.

Run it with `python -m tests.test_vulnerabilities`. Sixteen tests, 20 assertions;
a passing check means the attack failed.

## Vulnerabilities found and fixed

The suite was written to break the system, and it did. All five findings below
were real, and all are fixed.

### HIGH — fraud suppression bypassed by string casing

`never_retry()` was a dict lookup on the raw error code, so every one of these
sailed straight past the block:

```
PAYMENT_RISK_CHECK_FAILED
Payment_Risk_Check_Failed
 payment_risk_check_failed
payment_risk_check_failed\n
```

Six of six variants bypassed the single most important guardrail in the system.
A processor does not control the exact bytes an issuer or an upstream
integration puts in that field, and casing or padding differences are ordinary
in practice — so this was a whitespace character away from retrying fraud.

**Fixed:** `normalise_code()` in [src/taxonomy.py](../src/taxonomy.py)
canonicalises (strip + lowercase, bytes decoded) before any lookup. Every
taxonomy entry point routes through it.

### HIGH — NaN-scheduled attempts bypassed every bound check

Every guard in the harness is a `<` or `>` comparison, and **every comparison
against NaN is False**. A policy returning `at_hour = nan` therefore passed the
deadline check, passed the forward-time check, and executed: **947 attempts and
INR 14,367 of cost** in a 300-payment test.

**Fixed:** [src/evaluate.py](../src/evaluate.py) rejects non-finite times
explicitly, before any comparison. This is the general lesson — a NaN does not
fail a bounds check, it *skips* it.

### MEDIUM — unknown codes failed open

`classify()` routes unrecognised codes to `HARD_DECLINE` so they reach
inference rather than a blind retry. But that means a **new** risk code from an
issuer is invisible to `never_retry` until someone triages it into the
taxonomy — and the agent would send a large payment carrying an unknown code
straight to the issuer.

**Fixed:** `is_known()` plus `UNKNOWN_CODE_FLOOR_MULTIPLIER` in
[src/agent.py](../src/agent.py). Unknown codes are held to double the
probability floor before anything issuer-facing. A genuinely recoverable new
code still gets retried once the model is confident; a new fraud code is not
chased on the strength of a large amount.

### MEDIUM — non-string codes raised

`classify([...])` and `classify({...})` raised `TypeError` on an unhashable
dict key. In production that exception takes down recovery for every payment
behind it.

**Fixed:** `normalise_code()` coerces rather than raising. An unrecognisable
code is exactly what the `HARD_DECLINE` fallback exists for.

### MEDIUM — pickle deserialisation was undocumented

See below. **Fixed** by this document.

## Known risk: the model artifact executes code

`models/agent.pkl` is loaded with `pickle.load()`, which **executes arbitrary
code during deserialisation**. This is a property of pickle, not a bug in this
code, and it is not mitigated by validating the file afterwards — the code runs
first.

Anyone who can write that file can run commands as whoever runs the evaluation.

For a prototype whose artifact is produced locally by `scripts/train_agent.py`
and never fetched from a network, this is acceptable and recorded rather than
hidden. Before deployment it would need to change:

- serialise to a non-executing format (ONNX, or the model's own parameters as
  arrays plus a schema);
- or sign the artifact and verify the signature before loading;
- and treat the model store as a privileged write path, not a shared directory.

Never load an `agent.pkl` you did not produce.

## Attacks that failed

Twenty checks, fifteen repelled at first run. Worth stating because these are
the properties the design was actually meant to have:

| Attack | Result |
|---|---|
| Buy past the guardrail with a huge amount (up to INR 1e9) | blocked — the floor is a probability, not a value |
| Forge history to win extra attempts | blocked — the cap counts real attempts in the harness |
| Schedule an attempt before the failure | blocked |
| Inject extra fields into a record to reach a policy | blocked — observation whitelist |
| Feed NaN / inf / negative / 1e15 amounts | no crash, no non-finite schedule |
| Feed unknown method, category and error code together | handled via reserved unknown slots |
| Cold-start customer with no history | falls back, and reports zero payday confidence |
| Remove the pushback model entirely | explicit fraud block still holds |
| Substitute a model that returns P=1.0 for everything | explicit fraud block still holds |
| Check costs are charged regardless of outcome | exact to the rupee |
| Recover more than the value at risk | impossible |

The last four are the ones that matter most: **guardrail 1 is enforced in code
before the model is consulted**, so neither a missing model nor a maximally
over-confident one can defeat it. That is the whole argument for keeping "never
do this" out of a learned expected value.

## Not addressed

This is a prototype. Out of scope, and listed so the gaps are explicit:

- No authentication, authorisation, rate limiting or transport security — there
  is no service here, only offline scripts.
- No PII handling. Customer identifiers are synthetic integers; a real system
  would need to treat them, and any nudge content, under the applicable data
  protection rules.
- No audit-log tamper resistance. The audit trail is a CSV.
- No adversarial-ML analysis. A merchant who could influence the training log
  could in principle steer the recovery head; the hard guardrails limit the
  blast radius but do not eliminate it.
- Dependency supply chain is unpinned beyond the four libraries in the README.
