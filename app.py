"""Recovery Console — watch the model run.

    python app.py     ->  http://localhost:8000

Failed payments arrive; the console replays each one through the agent stage by
stage: the features it derives, every option it scores, which safety rules fire,
what it chooses, and what happens when the attempt is actually made.

The replay mirrors src/evaluate.py exactly — same per-payment random seed, same
order of checks — so nothing shown here can disagree with the scored results in
scripts/run_evaluation.py.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.agent import CANDIDATES, RecoveryAgent
from src.evaluate import evaluate, load, observe
from src.explain import cause_plain, code_meaning, diagnose
from src.features import featurise_candidates
from src.generator import Action
from src.policies import Attempt, RuleBased
from src.taxonomy import classify, is_known, is_opaque, never_retry

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)

print("loading data and model ...")
RECORDS, SIM, CFG = load()
EVAL = [r for r in RECORDS if r["hour"] >= CFG.train_hours]
EVAL.sort(key=lambda r: r["hour"])          # chronological, like a real feed

with open(ROOT / "models" / "agent.pkl", "rb") as fh:
    _b = pickle.load(fh)
AGENT = RecoveryAgent(_b["model"], _b["ctx"], CFG,
                      p_floor=_b["p_floor"], risk_model=_b.get("risk_model"))
RULES = RuleBased()

EVAL_SEED = 20260826        # must match src/evaluate.py
MAX_ATTEMPTS = 4
_trace_cache: dict[int, dict] = {}
print(f"ready: {len(EVAL):,} held-out failed payments\n")


def _run(policy, rec):
    res, audit = evaluate(policy, [rec], SIM, CFG, collect_audit=10)
    steps = [] if audit.empty else [
        {"action": r.action.lower(), "delay_h": float(r.delay_h),
         "ok": bool(r.succeeded)} for r in audit.itertuples()
    ]
    return {"recovered": bool(res.recovered_n),
            "cost": round(float(res.cost), 2),
            "steps": steps,
            "issuer_hits": int(res.fraud_attempts)}


def _clean(records):
    """Replace NaN/inf with null before serialising.

    Python's json module emits bare NaN and reads it back happily, so this looks
    fine server-side. A browser's JSON.parse rejects it outright, which showed up
    as a panel stuck on "Loading" with nothing in the console to explain it.
    """
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
             for k, v in rec.items()} for rec in records]


# --- scoring and guardrails, exposed so the UI can show the working ----------
def _score_options(rec, history):
    """Every option the model scored at this point, with its numbers."""
    obs = observe(rec)
    cands = [(a, d) for a, d in CANDIDATES if d <= CFG.horizon_hours]
    if history:
        earliest = max(h.at_hour for h in history) - obs["hour"] + 0.5
        cands = [(a, d) for a, d in cands if d >= earliest]
    if not cands:
        return [], None, 0.0, AGENT.p_floor

    actions = [a for a, _ in cands]
    delays = np.array([d for _, d in cands])

    floor = AGENT.p_floor
    if not is_known(obs["error_code"]):
        floor = min(1.0, floor * 2.0)

    X = featurise_candidates(obs, history, AGENT.ctx, actions, delays)
    ev, p, pen = AGENT._step_value(X, actions, float(obs["amount"]), floor)
    decision = AGENT.decide(obs, history)

    chosen_i, p_chosen = None, 0.0
    if decision is not None:
        for k, (a, d) in enumerate(cands):
            if a is decision.action and abs(
                    decision.at_hour - obs["hour"] - d) < 1e-9:
                chosen_i, p_chosen = k, float(p[k])
                break

    rows = [{"action": a.value, "delay_h": float(d),
             "p_recover": float(pi), "p_pushback": float(pe),
             "ev": None if not np.isfinite(e) else float(e),
             "chosen": k == chosen_i}
            for k, ((a, d), pi, pe, e) in enumerate(zip(cands, p, pen, ev))]
    rows.sort(key=lambda r: (0 if r["chosen"] else 1,
                             1e18 if r["ev"] is None else -r["ev"]))
    return rows, decision, p_chosen, floor


def _guardrails(rec, decision, p_chosen, floor_used):
    """Which safety rules were checked, and what each concluded."""
    code = rec["error_code"]
    checks = [{
        "name": "Bank explicitly flagged fraud",
        "detail": f"code is “{code}”",
        "verdict": "BLOCKED" if never_retry(code) else "clear",
    }, {
        "name": "Code recognised",
        "detail": "a documented Razorpay code" if is_known(code)
                  else "unknown code — bar for involving the bank is doubled",
        "verdict": "clear" if is_known(code) else "raised",
    }]
    if decision is not None:
        issuer_facing = decision.action in (Action.RETRY, Action.SWITCH_METHOD)
        checks.append({
            "name": "Enough chance before involving the bank",
            "detail": (f"{p_chosen:.0%} chance vs {floor_used:.0%} minimum"
                       if issuer_facing else
                       "chosen action never touches the bank"),
            "verdict": "clear" if (not issuer_facing or p_chosen >= floor_used)
                       else "BLOCKED",
        })
    return checks


def trace(i: int) -> dict:
    """Replay one payment through the agent, recording every stage."""
    if i in _trace_cache:
        return _trace_cache[i]

    rec = EVAL[i % len(EVAL)]
    prof = AGENT.ctx.profiles.get(rec["customer_id"], AGENT.ctx.fallback_profile)
    stress = float(AGENT.ctx.issuer_stress.get(rec["txn_id"], 0.0))

    rng = np.random.default_rng(EVAL_SEED * 1_000_003 + int(rec["txn_id"]))
    deadline = rec["hour"] + CFG.horizon_hours
    history: list[Attempt] = []
    attempts: list[dict] = []
    last_hour = rec["hour"]
    cost = recovered = 0.0
    is_fraud = rec["_cause"] == "RISK_BLOCK"
    stop_reason = "reached the limit of 4 attempts"
    first_opts = None

    while len(history) < MAX_ATTEMPTS:
        rows, decision, p_chosen, floor = _score_options(rec, history)
        checks = _guardrails(rec, decision, p_chosen, floor)
        if first_opts is None:
            first_opts = (rows, decision, p_chosen)

        if decision is None:
            stop_reason = ("the bank already called this fraud — those are "
                           "never retried" if never_retry(rec["error_code"])
                           else "nothing left was worth what it would cost to try")
            attempts.append({"n": len(history) + 1, "options": rows[:8],
                             "checks": checks, "decision": None,
                             "outcome": None, "stop_reason": stop_reason})
            break

        if (not np.isfinite(decision.at_hour) or decision.at_hour > deadline
                or decision.at_hour < last_hour + 0.5):
            stop_reason = "ran out of room inside the 7-day window"
            attempts.append({"n": len(history) + 1, "options": rows[:8],
                             "checks": checks, "decision": None,
                             "outcome": None, "stop_reason": stop_reason})
            break

        last_hour = decision.at_hour
        step_cost = (CFG.nudge_cost_inr if decision.action is Action.NUDGE
                     else CFG.retry_cost_inr)
        penalty = (CFG.fraud_retry_penalty_inr
                   if decision.action in (Action.RETRY, Action.SWITCH_METHOD)
                   and is_fraud else 0.0)
        cost += step_cost + penalty

        ok = SIM.would_succeed(rec, decision.action, decision.at_hour, rng)
        history.append(Attempt(decision.action, decision.at_hour,
                               decision.reason, ok))
        attempts.append({
            "n": len(history), "options": rows[:8], "checks": checks,
            "decision": {"action": decision.action.value,
                         "delay_h": round(decision.at_hour - rec["hour"], 2),
                         "p_recover": round(p_chosen, 4),
                         "cost": round(step_cost, 2),
                         "penalty": round(penalty, 2)},
            "outcome": "recovered" if ok else "failed again",
            "stop_reason": None,
        })
        if ok:
            recovered = float(rec["amount"])
            stop_reason = "recovered"
            break

    d0, p0 = (first_opts[1], first_opts[2]) if first_opts else (None, 0.0)
    pen0 = (first_opts[0][0]["p_pushback"]
            if first_opts and first_opts[0] else 0.0)

    out = {
        "i": i,
        "input": {
            "txn_id": int(rec["txn_id"]), "amount": float(rec["amount"]),
            "method": rec["method"], "category": rec["category"].replace("_", " "),
            "error_code": rec["error_code"],
            "code_meaning": code_meaning(rec["error_code"]),
            "opaque": bool(is_opaque(rec["error_code"])),
        },
        "features": [
            {"k": "Payments seen from this customer",
             "v": str(int(prof["cust_n_txns"])),
             "why": "how much history there is to reason from"},
            {"k": "Their usual success rate",
             "v": f"{100 * float(prof['cust_success_rate']):.0f}%",
             "why": "how reliably this customer's payments normally go through"},
            {"k": "Estimated payday",
             "v": f"day {int(prof['payday_hat'])}",
             "why": "worked out from when their payments succeed — never told to us"},
            {"k": "Confidence in that payday",
             "v": f"{float(prof['payday_confidence']):.2f}",
             "why": "low means this customer shows no clear salary pattern"},
            {"k": "Their bank's health just before",
             "v": f"{100 * stress:.0f}% failing",
             "why": "failure rate across that bank in the previous 2 hours"},
        ],
        "attempts": attempts,
        "result": {"recovered": bool(recovered), "amount": recovered,
                   "cost": round(cost, 2), "stop_reason": stop_reason},
        "rules": _run(RULES, rec),
        "true_cause": cause_plain(rec["_cause"]),
        "diagnosis": diagnose(rec, d0, p0, pen0, prof, stress),
    }
    _trace_cache[i] = out
    return out


@app.route("/api/trace/<int:i>")
def api_trace(i: int):
    return jsonify(trace(i))


@app.route("/api/queue/<int:start>")
def api_queue(start: int):
    """A window of upcoming payments, for the sidebar."""
    items = []
    for i in range(start, start + 30):
        rec = EVAL[i % len(EVAL)]
        items.append({
            "i": i, "amount": float(rec["amount"]), "method": rec["method"],
            "error_code": rec["error_code"],
            "opaque": bool(is_opaque(rec["error_code"])),
        })
    return jsonify(items)


# Scenario picker -------------------------------------------------------------
SCENARIOS = [
    {"key": "fraud", "cause": "RISK_BLOCK", "prefer_opaque": True,
     "title": "Bank quietly blocked it",
     "blurb": "Unrecoverable — and the code does not say so."},
    {"key": "nofunds", "cause": "NO_FUNDS", "prefer_opaque": False,
     "title": "Customer had no money",
     "blurb": "Retrying today fails. Waiting for payday works."},
    {"key": "bankdown", "cause": "ISSUER_DOWN", "prefer_opaque": False,
     "title": "Their bank was down",
     "blurb": "Nothing wrong with the card at all."},
    {"key": "abandoned", "cause": "CUSTOMER_ABANDON", "prefer_opaque": False,
     "title": "Customer walked away",
     "blurb": "No retry fixes this. Only the customer can."},
    {"key": "limit", "cause": "LIMIT_EXCEEDED", "prefer_opaque": True,
     "title": "Hit their daily limit",
     "blurb": "Resets on its own — retry too early and it fails again."},
    {"key": "broken", "cause": "INSTRUMENT_STATE", "prefer_opaque": False,
     "title": "Card was unusable",
     "blurb": "Expired or blocked. Another retry cannot help."},
]
_SCEN_BY_KEY = {s["key"]: s for s in SCENARIOS}
_scen_cache: dict[str, int] = {}


def _pick_scenario(key: str) -> int:
    """Index of the most illustrative payment for this failure type."""
    if key in _scen_cache:
        return _scen_cache[key]

    spec = _SCEN_BY_KEY[key]
    pool = [(i, r) for i, r in enumerate(EVAL) if r["_cause"] == spec["cause"]]
    if spec["prefer_opaque"]:
        opaque = [(i, r) for i, r in pool if is_opaque(r["error_code"])]
        pool = opaque or pool
    pool.sort(key=lambda t: -t[1]["amount"])

    best, best_score = pool[0][0], -1e18
    for i, rec in pool[:20]:
        t = trace(i)
        r = t["rules"]
        score = (1e6 if (t["result"]["recovered"] and not r["recovered"]) else 0) \
            + (r["cost"] - t["result"]["cost"]) + t["input"]["amount"] / 1000.0
        if score > best_score:
            best, best_score = i, score

    _scen_cache[key] = best
    return best


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify([{k: s[k] for k in ("key", "title", "blurb")}
                    for s in SCENARIOS])


@app.route("/api/scenario/<key>")
def api_scenario(key: str):
    if key not in _SCEN_BY_KEY:
        return jsonify({"error": "unknown scenario"}), 404
    return jsonify(trace(_pick_scenario(key)))


@app.route("/api/results")
def api_results():
    df = pd.read_csv(ROOT / "results" / "final_evaluation.csv")
    dial = []
    f = ROOT / "results" / "guardrail_tradeoff.csv"
    if f.exists():
        dial = _clean(pd.read_csv(f).to_dict("records"))
    return jsonify({
        "policies": _clean(df.to_dict("records")),
        "n_payments": len(EVAL),
        "at_risk": float(sum(x["amount"] for x in EVAL)),
        "dial": dial,
    })


@app.route("/")
def index():
    return render_template("console.html")




# ---------------------------------------------------------------------------
# LIVE INFERENCE
# ---------------------------------------------------------------------------
# The trace endpoints replay stored payments. This one does not: it builds an
# observation from whatever the caller sends, runs the real feature pipeline and
# the real models on it, and returns the decision. Change an input, get a fresh
# inference -- the same model, the same code path, no cache anywhere.
from dataclasses import dataclass as _dc

from src.explain import code_meaning as _cm
from src.taxonomy import ALL_CODES as _ALL


@_dc
class _Ctx:
    profiles: dict
    issuer_stress: dict
    fallback_profile: dict


@app.route("/api/codes")
def api_codes():
    """Every documented Razorpay code, for the picker."""
    return jsonify(sorted(
        ({"code": c, "meaning": _cm(c),
          "opaque": bool(is_opaque(c)), "cls": classify(c).value}
         for c in _ALL),
        key=lambda r: (not r["opaque"], r["code"])))


@app.route("/api/predict")
def api_predict():
    """Score a payment the caller describes, right now.

    Nothing here is looked up from the dataset. The observation is assembled
    from the query string, the customer profile and issuer-stress features are
    synthesised from the sliders, and the two trained models are called on the
    result. This is the same code the evaluation runs.
    """
    a = request.args
    try:
        amount = max(1.0, float(a.get("amount", 2000)))
        day = int(a.get("day", 12))                      # day of month now
        payday = int(a.get("payday", 1))
        conf = float(a.get("conf", 0.35))
        stress = float(a.get("stress", 0.0))
        n_txns = int(a.get("n_txns", 80))
        succ = float(a.get("succ", 0.9))
        n_methods = int(a.get("n_methods", 2))
    except ValueError:
        return jsonify({"error": "bad parameters"}), 400

    code = a.get("code", "card_declined")
    method = a.get("method", "card")
    category = a.get("category", "ecommerce")

    # Place the payment on a day-of-month that matches the slider.
    hour = float(((day - 1) + 60) * 24 + 14)
    txn_id = -1
    cust_id = -1

    obs = {"txn_id": txn_id, "hour": hour, "customer_id": cust_id,
           "merchant_id": 0, "issuer_id": 0, "category": category,
           "method": method, "amount": amount, "error_code": code}

    profile = {"payday_hat": payday, "payday_confidence": conf,
               "cust_n_txns": n_txns, "cust_success_rate": succ,
               "cust_mean_amount": max(500.0, amount * 0.8),
               "cust_n_methods": n_methods}
    ctx = _Ctx(profiles={cust_id: profile},
               issuer_stress={txn_id: stress},
               fallback_profile=profile)

    cands = [(act, d) for act, d in CANDIDATES if d <= CFG.horizon_hours]
    actions = [x for x, _ in cands]
    delays = np.array([d for _, d in cands])

    floor = AGENT.p_floor
    if not is_known(code):
        floor = min(1.0, floor * 2.0)

    # Swap the agent onto this synthetic context for one call.
    real_ctx = AGENT.ctx
    try:
        AGENT.ctx = ctx
        X = featurise_candidates(obs, [], ctx, actions, delays)
        ev, p, pen = AGENT._step_value(X, actions, amount, floor)
        decision = AGENT.decide(obs, [])
    finally:
        AGENT.ctx = real_ctx

    chosen = None
    p_chosen = 0.0
    if decision is not None:
        for k, (act, d) in enumerate(cands):
            if act is decision.action and abs(decision.at_hour - hour - d) < 1e-9:
                chosen, p_chosen = k, float(p[k])
                break

    options = [{"action": act.value, "delay_h": float(d),
                "p_recover": float(pi), "p_pushback": float(pe),
                "ev": None if not np.isfinite(e) else float(e),
                "chosen": k == chosen}
               for k, ((act, d), pi, pe, e) in enumerate(zip(cands, p, pen, ev))]
    options.sort(key=lambda r: (0 if r["chosen"] else 1,
                                1e18 if r["ev"] is None else -r["ev"]))

    days_to_payday = (payday - day) % 30

    # Reuse the same plain-English explainer the replay view uses, so the live
    # panel can lead with a sentence instead of a probability table.
    pseudo = dict(obs)
    pseudo["_cause"] = ""
    dx = diagnose(pseudo, decision, p_chosen,
                  float(pen[chosen]) if chosen is not None else 0.0,
                  profile, stress)

    return jsonify({
        "diagnosis": dx,
        "input": {"amount": amount, "method": method, "code": code,
                  "code_meaning": _cm(code), "opaque": bool(is_opaque(code)),
                  "recovery_class": classify(code).value,
                  "known": bool(is_known(code))},
        "derived": {"days_to_payday": days_to_payday,
                    "payday_conf": conf, "issuer_stress": stress},
        "decision": None if decision is None else {
            "action": decision.action.value,
            "delay_h": round(decision.at_hour - hour, 2),
            "p_recover": round(p_chosen, 4),
            "reason": decision.reason},
        "options": options,
        "checks": _guardrails(obs, decision, p_chosen, floor),
        "floor": floor,
    })


# ---------------------------------------------------------------------------
# LIVE SIMULATION — you set the truth, the model is not told it
# ---------------------------------------------------------------------------
# The caller picks what ACTUALLY went wrong. We build a real payment record from
# that, emit a bank error code through the same lossy emission table the dataset
# uses (so an opaque code may well come out), then run the real agent and the
# real oracle over it, attempt by attempt.
#
# The model receives only the observable fields. It never sees the cause the
# caller chose — which is the entire point of letting them choose it.
from src.generator import _EMISSION, Cause, Method

SYNTH_ISSUER = 999          # reserved, so a bank outage can be staged on demand

# Real customer profiles, indexed by the payday the estimator inferred for them.
# The simulator borrows one so its feature vector sits inside the distribution
# the model was actually trained on.
_BY_PAYDAY: dict = {}
for _cid, _pr in AGENT.ctx.profiles.items():
    _BY_PAYDAY.setdefault(int(_pr["payday_hat"]), []).append(_pr)
for _k in _BY_PAYDAY:                       # most-confident first, deterministic
    _BY_PAYDAY[_k].sort(key=lambda r: -float(r["payday_confidence"]))


def _real_profile(payday: int) -> dict:
    """A genuine customer whose inferred payday matches, if one exists."""
    for offset in range(0, 16):
        for d in ((payday + offset - 1) % 30 + 1, (payday - offset - 1) % 30 + 1):
            if _BY_PAYDAY.get(d):
                p = dict(_BY_PAYDAY[d][0])
                p["payday_hat"] = d
                return p
    return dict(AGENT.ctx.fallback_profile)



def _emit_code(method, cause, rng):
    table = _EMISSION[Method(method)][cause]
    codes, probs = list(table.keys()), list(table.values())
    return str(rng.choice(codes, p=probs))


def _replay_record(rec, prof, stress):
    """Run the agent over one record, recording every stage. Never cached."""
    rng = np.random.default_rng(EVAL_SEED * 1_000_003 + int(rec["txn_id"]))
    deadline = rec["hour"] + CFG.horizon_hours
    history, attempts, last_hour = [], [], rec["hour"]
    cost = recovered = 0.0
    is_fraud = rec["_cause"] == "RISK_BLOCK"
    stop_reason = "reached the limit of 4 attempts"
    first = None

    while len(history) < MAX_ATTEMPTS:
        rows, decision, p_chosen, floor = _score_options(rec, history)
        checks = _guardrails(rec, decision, p_chosen, floor)
        if first is None:
            first = (rows, decision, p_chosen)

        bad = (decision is None or not np.isfinite(decision.at_hour)
               or decision.at_hour > deadline
               or decision.at_hour < last_hour + 0.5)
        if bad:
            stop_reason = ("the bank already called this fraud — those are never "
                           "retried" if never_retry(rec["error_code"])
                           else "nothing left was worth what it would cost to try")
            attempts.append({"n": len(history) + 1, "options": rows[:8],
                             "checks": checks, "decision": None, "outcome": None,
                             "stop_reason": stop_reason})
            break

        last_hour = decision.at_hour
        step_cost = (CFG.nudge_cost_inr if decision.action is Action.NUDGE
                     else CFG.retry_cost_inr)
        penalty = (CFG.fraud_retry_penalty_inr
                   if decision.action in (Action.RETRY, Action.SWITCH_METHOD)
                   and is_fraud else 0.0)
        cost += step_cost + penalty

        ok = SIM.would_succeed(rec, decision.action, decision.at_hour, rng)
        history.append(Attempt(decision.action, decision.at_hour, "", ok))
        attempts.append({
            "n": len(history), "options": rows[:8], "checks": checks,
            "decision": {"action": decision.action.value,
                         "delay_h": round(decision.at_hour - rec["hour"], 2),
                         "p_recover": round(p_chosen, 4),
                         "cost": round(step_cost, 2), "penalty": round(penalty, 2)},
            "outcome": "recovered" if ok else "failed again", "stop_reason": None})
        if ok:
            recovered = float(rec["amount"])
            stop_reason = "recovered"
            break

    d0, p0 = (first[1], first[2]) if first else (None, 0.0)
    pen0 = first[0][0]["p_pushback"] if first and first[0] else 0.0

    return {
        "i": -1,
        "input": {"txn_id": int(rec["txn_id"]), "amount": float(rec["amount"]),
                  "method": rec["method"], "category": rec["category"],
                  "error_code": rec["error_code"],
                  "code_meaning": code_meaning(rec["error_code"]),
                  "opaque": bool(is_opaque(rec["error_code"]))},
        "features": [
            {"k": "Payments seen from this customer", "v": str(prof["cust_n_txns"]),
             "why": "how much history there is to reason from"},
            {"k": "Their usual success rate",
             "v": "%.0f%%" % (100 * prof["cust_success_rate"]),
             "why": "how reliably their payments normally go through"},
            {"k": "Estimated payday", "v": "day %d" % prof["payday_hat"],
             "why": "worked out from their history — never told to us"},
            {"k": "Confidence in that payday",
             "v": "%.2f" % prof["payday_confidence"],
             "why": "low means no clear salary pattern"},
            {"k": "Their bank's health just before",
             "v": "%.0f%% failing" % (100 * stress),
             "why": "failure rate on that bank in the previous 2 hours"},
        ],
        "attempts": attempts,
        "result": {"recovered": bool(recovered), "amount": recovered,
                   "cost": round(cost, 2), "stop_reason": stop_reason},
        "rules": _run(RULES, rec),
        "true_cause": cause_plain(rec["_cause"]),
        "diagnosis": diagnose(rec, d0, p0, pen0, prof, stress),
    }


@app.route("/api/simulate")
def api_simulate():
    a = request.args
    try:
        amount = max(50.0, float(a.get("amount", 8000)))
        day = int(a.get("day", 25))
        payday = int(a.get("payday", 1))
        seed = int(a.get("seed", 7))
    except ValueError:
        return jsonify({"error": "bad parameters"}), 400

    method = a.get("method", "card")
    try:
        cause = Cause(a.get("cause", "NO_FUNDS"))
    except ValueError:
        return jsonify({"error": "unknown situation"}), 400

    rng = np.random.default_rng(seed * 7919 + day * 31 + int(amount))
    hour = float(((day - 1) + 60) * 24 + 14)

    # A bank outage must genuinely exist for the oracle to reason about it, so
    # stage one on a reserved issuer covering this payment.
    issuer = 0
    if cause is Cause.ISSUER_DOWN:
        issuer = SYNTH_ISSUER
        SIM.downtime = SIM.downtime[SIM.downtime.issuer_id != SYNTH_ISSUER]
        SIM.downtime = pd.concat([SIM.downtime, pd.DataFrame([{
            "issuer_id": SYNTH_ISSUER, "start_h": hour - 1.0,
            "end_h": hour + float(a.get("outage_h", 20))}])], ignore_index=True)

    code = _emit_code(method, cause, rng)

    rec = {
        "txn_id": -abs(int(hour * 7 + amount)) - 1,
        "hour": hour, "customer_id": -1, "merchant_id": 0, "issuer_id": issuer,
        "category": a.get("category", "ecommerce"), "method": method,
        "amount": amount, "error_code": code,
        "_cause": cause.value,
        "_salary_day": payday,
        "_salary_sensitivity": float(a.get("sens", 0.85)),
        "_balance_health": float(a.get("health", 0.5)),
        "_spend_capacity": float(a.get("capacity", 3600.0)),
        "_has_alt_method": a.get("alt", "1") == "1",
        "_engagement": float(a.get("engagement", 0.6)),
    }

    # Borrow a REAL customer profile rather than inventing one.
    #
    # A fabricated profile puts every feature out of the distribution the model
    # was trained on, and the model then behaves nothing like it does in the
    # evaluation: an early version of this endpoint made up a profile and the
    # agent read almost every failure as "instrument broken", switching method
    # four times and spending INR 1,010 on a fraud block it normally refuses to
    # touch at all. The demo would have contradicted the measured results.
    profile = _real_profile(payday)
    stress = float(a.get("stress", 0.55 if cause is Cause.ISSUER_DOWN else 0.0))

    ctx = _Ctx(profiles={-1: profile}, issuer_stress={rec["txn_id"]: stress},
               fallback_profile=profile)

    real_ctx = AGENT.ctx
    try:
        AGENT.ctx = ctx
        return jsonify(_replay_record(rec, profile, stress))
    finally:
        AGENT.ctx = real_ctx


@app.route("/api/situations")
def api_situations():
    """The things that actually go wrong, in plain words."""
    return jsonify([
        {"cause": "NO_FUNDS", "label": "The customer had no money",
         "hint": "Their account was empty when the payment ran."},
        {"cause": "RISK_BLOCK", "label": "The bank blocked it as fraud",
         "hint": "Nothing can recover this. The question is what it costs to find out."},
        {"cause": "ISSUER_DOWN", "label": "Their bank was down",
         "hint": "Nothing wrong with the card — the bank was simply broken."},
        {"cause": "CUSTOMER_ABANDON", "label": "The customer walked away",
         "hint": "They closed the page or cancelled."},
        {"cause": "LIMIT_EXCEEDED", "label": "They hit their daily limit",
         "hint": "It resets on its own overnight."},
        {"cause": "INSTRUMENT_STATE", "label": "Their card was unusable",
         "hint": "Expired, blocked, or not enabled for online payments."},
        {"cause": "AUTH_FAIL", "label": "They failed the OTP",
         "hint": "Wrong code, or they closed the bank page."},
        {"cause": "GATEWAY_ISSUE", "label": "A technical glitch",
         "hint": "Something broke mid-payment."},
    ])


# ---------------------------------------------------------------------------
# PRODUCTION SEAM — POST a real Razorpay webhook, get a recovery plan back
# ---------------------------------------------------------------------------
# Everything above runs on the simulator. This is the endpoint a merchant would
# actually point Razorpay at. It takes the JSON Razorpay POSTs on payment.failed,
# verifies the signature when a secret is configured, and answers with the action
# to schedule.
#
# No outcome is returned, because there is no oracle for a real payment: the
# honest answer to a live webhook is a PLAN, not a result.
import os

from src.razorpay_webhook import (WebhookError, parse_payment_failed,
                                  verify_signature)

WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")


@app.route("/api/webhook", methods=["POST"])
def api_webhook():
    raw = request.get_data()

    # Signature is enforced only when a secret is configured, so the demo runs
    # without one — but a deployment that sets the secret cannot be driven by
    # anyone who finds the URL.
    if WEBHOOK_SECRET:
        sig = request.headers.get("X-Razorpay-Signature", "")
        if not verify_signature(raw, sig, WEBHOOK_SECRET):
            return jsonify({"error": "signature verification failed"}), 401

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"payload is not valid JSON: {exc}"}), 400

    try:
        pay = parse_payment_failed(payload)
    except WebhookError as exc:
        return jsonify({"error": str(exc)}), 400

    # Build the observation. A real deployment would look the customer up; here
    # we borrow a representative profile so the features stay in distribution.
    hour = float(((14 - 1) + 60) * 24 + 14)
    profile = _real_profile(1)
    rec = {"txn_id": -7, "hour": hour, "customer_id": -1, "merchant_id": 0,
           "issuer_id": 0, "category": "ecommerce", "method": pay["method"],
           "amount": pay["amount"], "error_code": pay["error_code"]}

    ctx = _Ctx(profiles={-1: profile}, issuer_stress={-7: 0.0},
               fallback_profile=profile)
    real_ctx = AGENT.ctx
    try:
        AGENT.ctx = ctx
        rows, decision, p_chosen, floor = _score_options(rec, [])
        checks = _guardrails(rec, decision, p_chosen, floor)
        dx = diagnose(rec, decision, p_chosen,
                      rows[0]["p_pushback"] if rows else 0.0, profile, 0.0)
    finally:
        AGENT.ctx = real_ctx

    plan = None
    if decision is not None:
        plan = {
            "action": decision.action.value,
            "run_at_hours_from_now": round(decision.at_hour - hour, 2),
            "confidence": round(p_chosen, 4),
            "touches_issuer": decision.action.value in ("RETRY", "SWITCH_METHOD"),
        }

    return jsonify({
        "payment": {k: pay[k] for k in (
            "payment_id", "order_id", "amount", "amount_paise", "currency",
            "method", "error_code", "raw_error_code", "error_source",
            "error_step", "recognised", "recovery_class", "opaque")},
        "plan": plan,
        "explanation": {"read": dx["read"], "action": dx["action"],
                        "why": dx["rationale"], "confidence": dx["confidence"],
                        "evidence": dx["evidence"]},
        "guardrails": checks,
        "options_considered": len(rows),
        "signature_verified": bool(WEBHOOK_SECRET),
    })


@app.route("/api/sample-webhook")
def api_sample_webhook():
    """A payload shaped exactly like Razorpay's, for the console to POST."""
    return jsonify({
        "entity": "event",
        "account_id": "acc_BFQ7uQEaa7j2z7",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {"payment": {"entity": {
            "id": "pay_29QQoUBi66xm2f",
            "entity": "payment",
            "amount": 900000,
            "currency": "INR",
            "status": "failed",
            "order_id": "order_9A33XWu170gUtm",
            "method": "card",
            "error_code": "BAD_REQUEST_ERROR",
            "error_description": "Payment was unsuccessful as the bank declined it",
            "error_source": "bank",
            "error_step": "payment_authorization",
            "error_reason": "card_declined",
            "created_at": 1757000000,
        }}},
        "created_at": 1757000000,
    })


if __name__ == "__main__":
    print("  ->  http://localhost:8000\n")
    app.run(host="127.0.0.1", port=8000, debug=False)
