"""Score the baseline policies.

Run:  python -m scripts.run_baselines

This establishes the bar the agent has to clear. It is run and recorded BEFORE
the agent exists, so the comparison cannot be reverse-engineered to flatter it.
"""

from pathlib import Path

import pandas as pd

from src.evaluate import evaluate, load
from src.policies import BASELINES

OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)

    # Scored on the held-out window, like every other policy in this project.
    records, sim, cfg = load()
    eval_records = [r for r in records if r["hour"] >= cfg.train_hours]

    rows, results, audit = [], {}, pd.DataFrame()
    for p in BASELINES:
        res, aud = evaluate(p, eval_records, sim, cfg, collect_audit=500)
        rows.append(res.row())
        results[p.name] = res
        if audit.empty and not aud.empty:
            audit = aud
    table = pd.DataFrame(rows)

    print("=" * 78)
    print("BASELINE RESULTS -- held-out window, 7-day horizon, max 4 attempts")
    print("=" * 78)
    print(table.to_string(index=False))
    print()

    floor = results["no_recovery"]
    print(f"value at risk : INR {floor.value_at_risk:,.0f} across {floor.n_payments:,} failed payments")
    print()

    print("-- recovery rate by class --")
    per_class = {}
    for name, res in results.items():
        if name == "no_recovery":
            continue
        per_class[name] = {
            cls: round(d["recovered"] / d["n"], 3)
            for cls, d in sorted(res.per_class.items())
        }
    print(pd.DataFrame(per_class).to_string())
    print()

    print("-- guardrail check: issuer-facing attempts on risk declines --")
    for name, res in results.items():
        if name == "no_recovery":
            continue
        print(f"  {name:12s} {res.fraud_attempts:5d} attempts  "
              f"INR {res.fraud_penalty:,.0f} penalty")

    table.to_csv(OUT / "baselines.csv", index=False)
    audit.to_csv(OUT / "audit_sample.csv", index=False)
    print()
    print(f"written -> {OUT / 'baselines.csv'}")
    print(f"written -> {OUT / 'audit_sample.csv'}")


if __name__ == "__main__":
    main()
