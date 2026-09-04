"""Sweep the rules engine's playbook for the opaque-code bucket.

Run:  python -m scripts.tune_baseline

Why this script exists
----------------------
The agent's claim is "it beats a good rules engine". That claim is worthless if
the rules engine was hand-written to lose. So the opaque-code playbook -- the
one branch where a rules engine has no information to go on and must guess --
is chosen by exhaustive sweep on net value rather than by authorial preference.

The winner becomes RuleBased.PLAYBOOK[HARD_DECLINE]. Re-run this whenever the
cost model or the oracle changes, so the baseline stays the strongest available
rules policy rather than a stale one.
"""

from pathlib import Path

import pandas as pd

from src.evaluate import evaluate, load
from src.generator import Action as A
from src.policies import RuleBased
from src.taxonomy import RecoveryClass

OUT = Path(__file__).resolve().parents[1] / "results"

# The two branches worth sweeping. HARD_DECLINE is where the rules engine has
# no information at all; FUNDS is where fixed timing is weakest. The other
# classes have an obvious dominant action and nothing to tune.
SWEEPS = {
    RecoveryClass.HARD_DECLINE: {
        "nudge,retry,switch":  ((A.NUDGE, 1.0), (A.RETRY, 24.0), (A.SWITCH_METHOD, 48.0)),
        "retry,retry,retry":   ((A.RETRY, 1.0), (A.RETRY, 24.0), (A.RETRY, 72.0)),
        "switch,retry,retry":  ((A.SWITCH_METHOD, 1.0), (A.RETRY, 24.0), (A.RETRY, 48.0)),
        "retry,retry,switch":  ((A.RETRY, 1.0), (A.RETRY, 24.0), (A.SWITCH_METHOD, 48.0)),
        "retry,switch,retry":  ((A.RETRY, 1.0), (A.SWITCH_METHOD, 24.0), (A.RETRY, 48.0)),
        "retry24,retry72":     ((A.RETRY, 24.0), (A.RETRY, 72.0)),
        "retry,nudge,switch":  ((A.RETRY, 1.0), (A.NUDGE, 24.0), (A.SWITCH_METHOD, 48.0)),
        "retry,retry":         ((A.RETRY, 1.0), (A.RETRY, 24.0)),
    },
    RecoveryClass.FUNDS: {
        "nudge2,retry48,retry120":  ((A.NUDGE, 2.0), (A.RETRY, 48.0), (A.RETRY, 120.0)),
        "nudge2,retry24,retry96":   ((A.NUDGE, 2.0), (A.RETRY, 24.0), (A.RETRY, 96.0)),
        "nudge2,retry24,switch72":  ((A.NUDGE, 2.0), (A.RETRY, 24.0), (A.SWITCH_METHOD, 72.0)),
        "retry24,retry72,retry144": ((A.RETRY, 24.0), (A.RETRY, 72.0), (A.RETRY, 144.0)),
        "retry1,retry24,retry72":   ((A.RETRY, 1.0), (A.RETRY, 24.0), (A.RETRY, 72.0)),
        "retry24,nudge48,retry120": ((A.RETRY, 24.0), (A.NUDGE, 48.0), (A.RETRY, 120.0)),
        "nudge2,retry24":           ((A.NUDGE, 2.0), (A.RETRY, 24.0)),
    },
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    records, sim, cfg = load()
    # Tune on the TRAINING window only. The baselines get exactly the same
    # deal as the agent: nothing may be fitted on the evaluation window, or the
    # comparison stops being a comparison.
    records = [r for r in records if r["hour"] < cfg.train_hours]
    print(f"tuning on {len(records):,} training-window failures\n")

    all_rows = []
    for branch, variants in SWEEPS.items():
        rows = []
        for name, steps in variants.items():
            policy = RuleBased()
            policy.PLAYBOOK = dict(RuleBased.PLAYBOOK)
            policy.PLAYBOOK[branch] = steps
            res, _ = evaluate(policy, records, sim, cfg)
            cls_stats = res.per_class.get(branch.value, {"n": 1, "recovered": 0})
            rows.append({
                "branch": branch.value,
                "playbook": name,
                "branch_recovery": round(cls_stats["recovered"] / cls_stats["n"], 4),
                "overall_recovery": round(res.recovery_rate, 4),
                "net_value": round(res.net_value, 0),
                "fraud_attempts": res.fraud_attempts,
            })

        table = pd.DataFrame(rows).sort_values("net_value", ascending=False)
        print(f"--- {branch.value} ---")
        print(table.to_string(index=False))
        print(f"strongest: {table.iloc[0].playbook}\n")
        all_rows.extend(table.to_dict("records"))

    pd.DataFrame(all_rows).to_csv(OUT / "baseline_tuning.csv", index=False)
    print(f"written -> {OUT / 'baseline_tuning.csv'}")


if __name__ == "__main__":
    main()
