# Failure Taxonomy — grounded in Razorpay's documented error codes

Source: razorpay.com/docs/errors/payments/cards/ and /upi/ (fetched 2026-08-26)

Every code below is a real Razorpay error string. The `class` column is our
inference layer — it collapses ~27 raw codes into 7 recovery intents.

## Recovery classes

| Class | Meaning | Default posture |
|---|---|---|
| `SOFT_TRANSIENT` | Nothing wrong with the instrument; infra or timing failed | Retry, short delay |
| `FUNDS` | Instrument valid, money absent | Retry on a *predicted liquidity date* |
| `AUTH_FAILED` | Customer failed/abandoned authentication | Nudge to re-attempt, same method |
| `METHOD_BROKEN` | Instrument unusable as-is | Switch method; retry is futile |
| `HARD_DECLINE` | Issuer said no, reason opaque | Probabilistic — the interesting case |
| `FRAUD` | Risk system declined | **NEVER retry** |
| `ABANDONED` | Customer walked away | Nudge + payment link |

## Cards

| Razorpay code | Class | Retry? | Action |
|---|---|---|---|
| `gateway_technical_error` | SOFT_TRANSIENT | Yes | Retry; consider alternate terminal/route |
| `bank_technical_error` | SOFT_TRANSIENT | Yes | Wait for issuer downtime window to clear |
| `payment_timed_out` | SOFT_TRANSIENT | Yes | Retry; 10-min limit was exceeded |
| `insufficient_funds` | FUNDS | Yes, timed | Retry near predicted salary/credit date |
| `authentication_failed` | AUTH_FAILED | No (auto) | Nudge customer; OTP wrong or browser closed |
| `incorrect_cvv` | AUTH_FAILED | No (auto) | Nudge; suggest CVV-less saved-card flow |
| `payment_cancelled` | ABANDONED | No (auto) | Nudge + payment link |
| `card_expired` | METHOD_BROKEN | No | Request updated card credentials |
| `card_not_enrolled` | METHOD_BROKEN | No | Guide to enable online txns; offer alt method |
| `card_disabled_for_online_payments` | METHOD_BROKEN | No | Same as above |
| `debit_instrument_inactive` | METHOD_BROKEN | No | Guide to card-control activation |
| `debit_instrument_blocked` | METHOD_BROKEN | No | Offer alternate method |
| `transaction_limit_exceeded` | METHOD_BROKEN | No | Suggest alt method, or split amount |
| `card_declined` | HARD_DECLINE | Maybe | Inference required |
| `payment_failed` | HARD_DECLINE | Maybe | Inference required |
| `payment_risk_check_failed` | FRAUD | **NEVER** | Suppress. Retrying harms merchant standing |

## UPI

| Razorpay code | Class | Retry? | Action |
|---|---|---|---|
| `bank_technical_error` | SOFT_TRANSIENT | Yes | UPI provider downtime; retry after window |
| `gateway_technical_error` | SOFT_TRANSIENT | Yes | Retry |
| `payment_timed_out` | SOFT_TRANSIENT | Yes | Retry |
| `payment_collect_request_expired` | ABANDONED | No (auto) | Re-issue collect request / link |
| `payment_cancelled` | ABANDONED | No (auto) | Nudge |
| `insufficient_funds` | FUNDS | Yes, timed | Timed retry, or prompt alternate account |
| `payment_declined` | HARD_DECLINE | Maybe | Debit failed, reason opaque — inference |
| `credit_failed` | HARD_DECLINE | Maybe | Inference |
| `invalid_vpa` | METHOD_BROKEN | No | Customer not registered on UPI app |
| `vpa_resolution_failed` | METHOD_BROKEN | No | Escalate |
| `customer_bank_account_mismatch` | METHOD_BROKEN | No | Direct to registered account |

## Where the AI actually lives

Rows marked `Maybe` are the thesis. `card_declined`, `payment_failed`,
`payment_declined` and `credit_failed` are Razorpay's own opaque buckets —
the issuer refused without saying why. A rules engine must guess one fixed
action for all of them. The agent infers the *likely* underlying cause from
context (amount, method, merchant category, customer history, time of day,
prior attempts) and picks accordingly.

Secondary AI surfaces:
- `FUNDS` timing — *when* to retry is a learned per-customer prediction
- Cost-aware stopping — expected recovery vs. retry cost
- Nudge copy generation for AUTH_FAILED / ABANDONED
