"""Final comparison: baselines vs the agent, on held-out data.

Run:  python -m scripts.run_evaluation

Every policy is scored on the SAME evaluation-window failures, with the same
per-payment random draws and the same cost model. Nothing scored here was
fitted here: the agent's model, its guardrail threshold, and the rules engine's
playbooks were all chosen on the training window.
"""

import pickle
from pathlib import Path

import pandas as pd

from src.agent import RecoveryAgent
from src.evaluate import evaluate, load
from src.policies import BASELINES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    records, sim, cfg = load()
    eval_records = [r for r in records if r["hour"] >= cfg.train_hours]
    at_risk = sum(r["amount"] for r in eval_records)

    with open(ROOT / "models" / "agent.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    agent = RecoveryAgent(bundle["model"], bundle["ctx"], cfg,
                          p_floor=bundle["p_floor"],
                          risk_model=bundle.get("risk_model"))

    policies = list(BASELINES) + [agent]

    print("=" * 100)
    print("HELD-OUT EVALUATION -- policies never fitted on this window")
    print("=" * 100)
    print(f"{len(eval_records):,} failed payments | INR {at_risk:,.0f} at risk "
          f"| 7-day horizon | max 4 attempts")
    print()

    rows, results, audit = [], {}, pd.DataFrame()
    for p in policies:
        collect = 600 if p.name == "agent" else 0
        res, aud = evaluate(p, eval_records, sim, cfg, collect_audit=collect)
        rows.append(res.row())
        results[p.name] = res
        if not aud.empty:
            audit = aud

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print()

    best_base = results["rule_based"]
    ag = results["agent"]
    print("-- agent vs the strongest baseline (tuned rules engine) --")
    print(f"  recovery rate : {best_base.recovery_rate:.4f} -> {ag.recovery_rate:.4f} "
          f"({ag.recovery_rate - best_base.recovery_rate:+.4f})")
    print(f"  net value     : INR {best_base.net_value:,.0f} -> INR {ag.net_value:,.0f} "
          f"({(ag.net_value / best_base.net_value - 1) * 100:+.1f}%)")
    print(f"  fraud attempts: {best_base.fraud_attempts} -> {ag.fraud_attempts} "
          f"({ag.fraud_attempts - best_base.fraud_attempts:+d})")
    print(f"  cost          : INR {best_base.cost:,.0f} -> INR {ag.cost:,.0f}")
    print()

    print("-- recovery rate by class --")
    per_class = {
        name: {cls: round(d["recovered"] / d["n"], 3)
               for cls, d in sorted(res.per_class.items())}
        for name, res in results.items() if name != "no_recovery"
    }
    print(pd.DataFrame(per_class).to_string())

    table.to_csv(RESULTS / "final_evaluation.csv", index=False)
    audit.to_csv(RESULTS / "agent_audit.csv", index=False)
    print()
    print(f"written -> {RESULTS / 'final_evaluation.csv'}")
    print(f"written -> {RESULTS / 'agent_audit.csv'}")


if __name__ == "__main__":
    main()
