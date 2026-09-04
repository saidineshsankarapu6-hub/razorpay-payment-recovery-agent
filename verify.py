"""Run everything, end to end, and report.

    python verify.py           # full pipeline from scratch (~25 min)
    python verify.py --quick   # tests only, reuses existing data (~4 min)

Use this to convince yourself the project works. Each stage prints whether it
passed and how long it took; the script stops at the first failure and shows
that stage's output.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FULL = [
    ("Build dataset",        "scripts.make_dataset",
     "108,000 transactions, seeded so you get identical data"),
    ("Tune baselines",       "scripts.tune_baseline",
     "sweeps the rules engine to its STRONGEST form, on training data only"),
    ("Score baselines",      "scripts.run_baselines",
     "the bar the agent has to clear"),
    ("Train agent",          "scripts.train_agent",
     "exploration log -> two models -> guardrail threshold (slowest stage)"),
    ("Evaluate",             "scripts.run_evaluation",
     "held-out comparison: the headline result"),
    ("Case studies",         "scripts.demo_cases",
     "same error code, different truths, side by side"),
    ("Integrity tests",      "tests.test_integrity",
     "27 checks that the evaluation is honest"),
    ("Vulnerability tests",  "tests.test_vulnerabilities",
     "20 attacks on the guardrails"),
]

QUICK = [s for s in FULL if s[1].startswith("tests.")]


def run(label: str, module: str, why: str) -> tuple[bool, str, float]:
    print(f"\n{'=' * 72}")
    print(f"  {label}   ({module})")
    print(f"  {why}")
    print("=" * 72, flush=True)

    t0 = time.time()
    proc = subprocess.run([sys.executable, "-u", "-m", module],
                          cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    out = proc.stdout + proc.stderr

    if proc.returncode == 0:
        # Show the last few lines so progress is visible without a wall of text.
        tail = [ln for ln in out.strip().splitlines() if ln.strip()][-12:]
        print("\n".join(tail))
        print(f"\n  PASSED in {elapsed:.0f}s")
    else:
        print(out[-4000:])
        print(f"\n  FAILED after {elapsed:.0f}s (exit {proc.returncode})")

    return proc.returncode == 0, out, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="run only the test suites, reusing existing data")
    args = ap.parse_args()

    stages = QUICK if args.quick else FULL
    if args.quick and not (ROOT / "models" / "agent.pkl").exists():
        print("No trained model found -- run without --quick first.")
        return 1

    print(f"Running {len(stages)} stages"
          f"{' (quick mode)' if args.quick else ' from scratch'}.")

    results, total = [], 0.0
    for label, module, why in stages:
        ok, out, secs = run(label, module, why)
        results.append((label, ok, secs))
        total += secs
        if not ok:
            print(f"\nStopped at: {label}")
            break

    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print("=" * 72)
    for label, ok, secs in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:22s} {secs:6.0f}s")
    print(f"\n  total {total:.0f}s")

    if all(ok for _, ok, _ in results) and len(results) == len(stages):
        print("\n  Everything passed.")
        if not args.quick:
            print("  Headline result is in the 'Evaluate' stage above,")
            print("  and saved to results/final_evaluation.csv")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
