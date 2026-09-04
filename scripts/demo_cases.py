"""Side-by-side case studies: one error code, several truths.

Run:  python -m scripts.demo_cases

Picks real held-out payments that all carry the SAME opaque Razorpay error
code but were caused by different things, and replays every policy on each so
the divergence is visible rather than asserted. This is the argument for
inference over lookup, drawn from generated data.

The true cause is printed for the reader only. No policy ever sees it.
"""

import pickle
from pathlib import Path

import pandas as pd

from src.agent import RecoveryAgent
from src.evaluate import evaluate, load
from src.policies import FixedRetry, RuleBased

ROOT = Path(__file__).resolve().parents[1]
CODE = "card_declined"
CAUSES = ("NO_FUNDS", "LIMIT_EXCEEDED", "RISK_BLOCK", "ISSUER_DOWN")


def replay(policy, rec, sim, cfg):
    """Run one policy over one payment and render what it did."""
    res, audit = evaluate(policy, [rec], sim, cfg, collect_audit=10)
    if audit.empty:
        return "stopped immediately", False, res.cost
    steps = " -> ".join(
        f"{r.action.lower()}@+{r.delay_h:g}h" for r in audit.itertuples())
    return steps, bool(audit.succeeded.any()), res.cost


def main() -> None:
    records, sim, cfg = load()
    eval_records = [r for r in records if r["hour"] >= cfg.train_hours]

    with open(ROOT / "models" / "agent.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    agent = RecoveryAgent(bundle["model"], bundle["ctx"], cfg,
                          p_floor=bundle["p_floor"],
                          risk_model=bundle.get("risk_model"))
    policies = [FixedRetry(), RuleBased(), agent]

    print("=" * 100)
    print(f"CASE STUDIES -- every payment below returned `{CODE}`")
    print("=" * 100)
    print("Razorpay documents this code as 'the bank declined the transaction'.")
    print("It discloses nothing further. The true cause is shown for the reader")
    print("only -- no policy has access to it.")
    print()

    rows = []
    for cause in CAUSES:
        pool = [r for r in eval_records
                if r["error_code"] == CODE and r["_cause"] == cause]
        if not pool:
            print(f"(no held-out {CODE} payment with cause {cause})")
            continue
        # Largest amount, so the stakes of getting it wrong are legible.
        rec = max(pool, key=lambda r: r["amount"])

        print(f"--- true cause: {cause} | txn {rec['txn_id']} "
              f"| INR {rec['amount']:,.0f} | {rec['method']} ---")
        for p in policies:
            steps, ok, cost = replay(p, rec, sim, cfg)
            verdict = "RECOVERED" if ok else "not recovered"
            print(f"  {p.name:12s} {verdict:14s} cost INR {cost:7,.1f}  {steps}")
            rows.append({"cause": cause, "txn_id": rec["txn_id"],
                         "amount": rec["amount"], "policy": p.name,
                         "recovered": ok, "cost": round(cost, 2),
                         "actions": steps})
        print()

    out = ROOT / "results" / "demo_cases.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
