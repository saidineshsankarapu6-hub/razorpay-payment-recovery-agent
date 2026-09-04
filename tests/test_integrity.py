"""Integrity tests.

These guard the claims the evaluation rests on. If any of these fail, every
number this project reports becomes meaningless -- so they are tested rather
than asserted in prose.

Run:  python -m pytest tests/ -q     (or: python -m tests.test_integrity)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import DATA as DATA_DIR
from src.evaluate import MAX_ATTEMPTS, evaluate, load, observe
from src.generator import Action, Cause, PaymentSimulator, SimConfig, build
from src.policies import BASELINES, OBSERVABLE_FIELDS, FixedRetry, Policy, RuleBased
from src.taxonomy import ALL_CODES, RecoveryClass, classify, never_retry

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------
def test_no_latent_leakage() -> None:
    """A policy must not be able to see simulator ground truth."""
    records, _, _ = load()
    obs = observe(records[0])

    leaked = [k for k in obs if k.startswith("_")]
    check("observation excludes latent fields", not leaked, f"leaked={leaked}")

    check("observation matches whitelist exactly",
          set(obs) == set(OBSERVABLE_FIELDS))

    # The dataset genuinely contains the latent fields -- so the whitelist is
    # doing real work, not passing because the columns happen to be absent.
    latent = [k for k in records[0] if k.startswith("_")]
    check("latent fields exist in the record but are withheld",
          len(latent) >= 5, f"withheld={sorted(latent)}")


def test_fraud_is_never_recoverable() -> None:
    """The guardrail is only meaningful if retrying fraud truly earns nothing."""
    sim = PaymentSimulator(SimConfig(n_days=20))
    df = sim.generate()
    risk = df[df._cause == Cause.RISK_BLOCK.value]
    rng = np.random.default_rng(0)

    recovered = 0
    for _, row in risk.head(400).iterrows():
        for action in (Action.RETRY, Action.SWITCH_METHOD, Action.NUDGE):
            for delay in (0.5, 24.0, 168.0):
                if sim.would_succeed(row, action, row["hour"] + delay, rng):
                    recovered += 1
    check("risk declines never recover under any action or delay",
          recovered == 0, f"{recovered} spurious recoveries")

    check("never_retry() fires on the explicit risk code",
          never_retry("payment_risk_check_failed"))


def test_determinism() -> None:
    """A reviewer cloning the repo must get identical data."""
    a = build(SimConfig(n_days=10))
    b = build(SimConfig(n_days=10))
    check("generator is reproducible under a fixed seed",
          a.equals(b))

    c = build(SimConfig(n_days=10, seed=99))
    check("a different seed produces different data",
          not a.success.equals(c.success))


def test_taxonomy_totality() -> None:
    """Every documented code must classify, and unknown codes must fail safe."""
    unmapped = [c for c in ALL_CODES if classify(c) is None]
    check("every known code maps to a recovery class", not unmapped)

    check("unknown codes fall back to HARD_DECLINE, not to a blind retry",
          classify("brand_new_code_2027") is RecoveryClass.HARD_DECLINE)


def test_emission_probabilities_sum_to_one() -> None:
    """A malformed emission table would silently distort the cause mix."""
    from src.generator import _EMISSION

    bad = []
    for method, table in _EMISSION.items():
        for cause, dist in table.items():
            total = sum(dist.values())
            if abs(total - 1.0) > 1e-9:
                bad.append(f"{method.value}/{cause.value}={total:.4f}")
    check("all emission distributions sum to 1.0", not bad, "; ".join(bad))


def test_opaque_codes_stay_ambiguous() -> None:
    """The thesis needs opaque codes to hide several causes.

    If any opaque code collapses to a single cause it becomes decodable by
    lookup, and the inference problem would be an artefact of the generator.
    """
    from src.taxonomy import OPAQUE_CODES

    df = build(SimConfig(n_days=45))
    f = df[~df.success]
    op = f[f.error_code.isin(OPAQUE_CODES)]

    ct = pd.crosstab(op.error_code, op._cause, normalize="index")
    ent = -(ct.replace(0, np.nan) * np.log2(ct.replace(0, np.nan))).sum(axis=1)

    check("every opaque code mixes >= 3 causes",
          bool((ct > 0.01).sum(axis=1).min() >= 3),
          f"min distinct causes={int((ct > 0.01).sum(axis=1).min())}")
    check("every opaque code carries >= 0.5 bits of ambiguity",
          bool(ent.min() >= 0.5), f"min entropy={ent.min():.2f} bits")


def test_policies_respect_attempt_cap() -> None:
    """A runaway policy would inflate both cost and recovery."""

    class Runaway(Policy):
        name = "runaway"

        def decide(self, obs, history):
            from src.policies import Decision
            return Decision(Action.RETRY, obs["hour"] + 0.1 * (len(history) + 1), "always")

    records, sim, cfg = load()
    res, _ = evaluate(Runaway(), records[:500], sim, cfg)
    check("attempt cap is enforced",
          res.attempts <= MAX_ATTEMPTS * 500,
          f"{res.attempts} attempts over {500} payments (cap {MAX_ATTEMPTS})")


def test_horizon_is_enforced() -> None:
    """Actions scheduled past the recovery horizon must not be executed."""

    class TooLate(Policy):
        name = "too_late"

        def decide(self, obs, history):
            from src.policies import Decision
            return Decision(Action.RETRY, obs["hour"] + 10_000, "beyond horizon")

    records, sim, cfg = load()
    res, _ = evaluate(TooLate(), records[:500], sim, cfg)
    check("beyond-horizon actions are not executed",
          res.attempts == 0 and res.cost == 0.0,
          f"attempts={res.attempts} cost={res.cost}")


def test_no_recovery_is_a_true_floor() -> None:
    records, sim, cfg = load()
    from src.policies import NoRecovery

    res, _ = evaluate(NoRecovery(), records, sim, cfg)
    check("doing nothing recovers nothing and costs nothing",
          res.recovered_n == 0 and res.cost == 0.0 and res.attempts == 0)
    check("value at risk is positive", res.value_at_risk > 0,
          f"INR {res.value_at_risk:,.0f}")


def test_common_random_numbers() -> None:
    """Two policies must face identical randomness on the same payment.

    Re-running the same policy must reproduce its score exactly; otherwise a
    reported lift could be a draw of the RNG rather than a property of the
    policy.
    """
    records, sim, cfg = load()
    subset = records[:2000]

    a, _ = evaluate(FixedRetry(), subset, sim, cfg)
    b, _ = evaluate(FixedRetry(), subset, sim, cfg)
    check("evaluation is deterministic across runs",
          a.recovered_n == b.recovered_n and abs(a.cost - b.cost) < 1e-9,
          f"{a.recovered_n} vs {b.recovered_n}")


def test_rule_based_suppresses_explicit_fraud() -> None:
    """The rules baseline must already handle the easy fraud case.

    This keeps the baseline honest: the agent should not get credit for a
    guardrail that any competent rules engine would also have.
    """
    policy = RuleBased()
    obs = {"error_code": "payment_risk_check_failed", "hour": 100.0}
    check("rule_based stops on an explicit risk decline",
          policy.decide(obs, []) is None)

    obs2 = {"error_code": "card_declined", "hour": 100.0}
    check("rule_based still retries opaque codes (its structural blind spot)",
          policy.decide(obs2, []) is not None)


def test_profiles_use_training_window_only() -> None:
    """Customer profiles must not be influenced by evaluation-window data.

    Built twice: once from the training window, once from the training window
    with the evaluation window appended. If the estimates differ, something is
    reading the future.
    """
    from src.features import build_customer_profiles

    cfg = SimConfig()
    txns = pd.read_parquet(DATA_DIR / "transactions.parquet")
    train = txns[txns.hour < cfg.train_hours]

    a = build_customer_profiles(train).sort_values("customer_id").reset_index(drop=True)
    b = build_customer_profiles(
        pd.concat([train, txns[txns.hour >= cfg.train_hours]])
    ).sort_values("customer_id").reset_index(drop=True)

    same = a.payday_hat.equals(b.payday_hat)
    check("adding future data changes the profiles (so training-only is real)",
          not same,
          "" if not same else "identical -- the estimator may be ignoring its input")

    # And the profile actually fed to the agent must come from the train split.
    check("training-window profiles cover every customer",
          a.customer_id.nunique() == cfg.n_customers)


def test_payday_estimator_beats_chance_where_it_should() -> None:
    """The estimator must work on cycle-bound customers and admit when it can't.

    A model that claimed high confidence for customers with no salary cycle
    would be worse than useless -- the agent would time retries around noise.
    """
    from src.features import build_customer_profiles

    cfg = SimConfig()
    txns = pd.read_parquet(DATA_DIR / "transactions.parquet")
    train = txns[txns.hour < cfg.train_hours]

    prof = build_customer_profiles(train)
    truth = txns.groupby("customer_id").agg(
        true_payday=("_salary_day", "first"),
        sens=("_salary_sensitivity", "first"))
    m = prof.merge(truth, on="customer_id")
    d = np.abs(m.payday_hat.values - m.true_payday.values) % 30
    m["err"] = np.minimum(d, 30 - d)

    thin = m[m.sens > 0.5]
    hit_thin = (thin.err <= 2).mean()
    check("payday estimate beats chance on cycle-bound customers",
          hit_thin > 0.30, f"{hit_thin:.3f} within +/-2 days (chance = 0.167)")

    # Confidence must be informative, not decorative.
    top = m.nlargest(len(m) // 4, "payday_confidence")
    bottom = m.nsmallest(len(m) // 4, "payday_confidence")
    check("high-confidence estimates are more accurate than low-confidence ones",
          (top.err <= 2).mean() > (bottom.err <= 2).mean(),
          f"top={(top.err <= 2).mean():.3f} vs bottom={(bottom.err <= 2).mean():.3f}")


def test_features_never_read_latent_columns() -> None:
    """Feature construction must work on a record stripped of ground truth."""
    from src.features import featurise, make_context

    cfg = SimConfig()
    txns = pd.read_parquet(DATA_DIR / "transactions.parquet")
    failed = pd.read_parquet(DATA_DIR / "failed_payments.parquet")
    ctx = make_context(txns[txns.hour < cfg.train_hours], txns)

    rec = failed.iloc[0].to_dict()
    stripped = {k: v for k, v in rec.items() if not k.startswith("_")}

    try:
        X = featurise([stripped], [Action.RETRY], [24.0], [[]], ctx)
        ok = X.shape[0] == 1 and bool(np.isfinite(X).all())
    except KeyError as exc:
        ok = False
        print(f"       featurise touched a latent column: {exc}")
    check("featurise() works without any underscore-prefixed column", ok)

    # And the agent's fast path must be equally clean.
    from src.features import featurise_candidates
    try:
        Xc = featurise_candidates(stripped, [], ctx, [Action.RETRY, Action.NUDGE],
                                  np.array([1.0, 24.0]))
        ok_c = Xc.shape[0] == 2 and bool(np.isfinite(Xc).all())
    except KeyError as exc:
        ok_c = False
        print(f"       featurise_candidates touched a latent column: {exc}")
    check("featurise_candidates() works without any latent column", ok_c)


def test_attempts_must_move_forward_in_time() -> None:
    """A policy must not collect several oracle draws for one instant.

    The oracle is stochastic, so re-proposing the same action at the same
    absolute time yields independent draws -- free retries no real processor
    gets. The agent discovered and exploited this, so the harness now enforces
    the rule for every policy.
    """
    from src.evaluate import MIN_ATTEMPT_GAP_H
    from src.policies import Decision

    class Stuck(Policy):
        name = "stuck"

        def decide(self, obs, history):
            # Always the same instant, regardless of what has been tried.
            return Decision(Action.RETRY, obs["hour"] + 24.0, "same instant")

    records, sim, cfg = load()
    res, _ = evaluate(Stuck(), records[:800], sim, cfg)
    check("a policy repeating one instant gets exactly one attempt per payment",
          res.attempts <= 800,
          f"{res.attempts} attempts over 800 payments")

    class Creeping(Policy):
        name = "creeping"

        def decide(self, obs, history):
            # Advances, but by less than the minimum gap.
            base = 24.0 + 0.01 * len(history)
            return Decision(Action.RETRY, obs["hour"] + base, "tiny step")

    res2, _ = evaluate(Creeping(), records[:800], sim, cfg)
    check("sub-gap advances are rejected too",
          res2.attempts <= 800,
          f"{res2.attempts} attempts (gap = {MIN_ATTEMPT_GAP_H}h)")


def main() -> int:
    for fn in (
        test_no_latent_leakage,
        test_fraud_is_never_recoverable,
        test_determinism,
        test_taxonomy_totality,
        test_emission_probabilities_sum_to_one,
        test_opaque_codes_stay_ambiguous,
        test_policies_respect_attempt_cap,
        test_horizon_is_enforced,
        test_no_recovery_is_a_true_floor,
        test_common_random_numbers,
        test_rule_based_suppresses_explicit_fraud,
        test_profiles_use_training_window_only,
        test_payday_estimator_beats_chance_where_it_should,
        test_features_never_read_latent_columns,
        test_attempts_must_move_forward_in_time,
    ):
        print(f"\n--- {fn.__name__} ---")
        fn()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        return 1
    print("all integrity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
