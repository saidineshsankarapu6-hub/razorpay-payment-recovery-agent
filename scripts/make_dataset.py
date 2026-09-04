"""Generate and persist the transaction dataset.

Run:  python -m scripts.make_dataset

Writes a full 90-day transaction log plus the failed-payment subset that the
recovery policies operate on. The simulator is seeded, so this is reproducible:
a reviewer who clones the repo gets byte-identical data.
"""

from pathlib import Path

import pandas as pd

from src.generator import PaymentSimulator, SimConfig
from src.taxonomy import OPAQUE_CODES, classify

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    cfg = SimConfig()
    sim = PaymentSimulator(cfg)
    df = sim.generate()

    failed = df[~df.success].copy()
    failed["recovery_class"] = [classify(c).value for c in failed.error_code]
    failed["is_opaque"] = failed.error_code.isin(OPAQUE_CODES)

    df.to_parquet(DATA / "transactions.parquet", index=False)
    failed.to_parquet(DATA / "failed_payments.parquet", index=False)

    lost = failed.amount.sum()
    print(f"transactions      : {len(df):,}")
    print(f"success rate      : {df.success.mean():.4f}")
    print(f"failed payments   : {len(failed):,}")
    print(f"value at risk     : INR {lost:,.0f}")
    print(f"opaque-code share : {failed.is_opaque.mean():.3f}")
    print()
    print("recovery class mix:")
    print(failed.recovery_class.value_counts(normalize=True).round(3).to_string())
    print()
    print(f"written -> {DATA / 'transactions.parquet'}")
    print(f"written -> {DATA / 'failed_payments.parquet'}")


if __name__ == "__main__":
    main()
