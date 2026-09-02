"""
Statistical analysis of compare_models.py's compare_log.csv.

Run after compare_models.py finishes (or partway through):

    python analyze_compare.py compare_log.csv

What this script does:

1. REWARD DISTRIBUTION section -- min/max/p1/p99 per model. A capped
   per-step no-op penalty can still sum to a large negative total over
   a full 500-step episode if the agent gets stuck repeating one bad
   action for the whole episode -- that's a BOUNDED-but-still-ugly
   number, different from an unbounded quadratic blowup. This section
   surfaces it either way so it doesn't get missed in a wall of
   per-episode console output.

2. POST-EPSILON-FLOOR COLLAPSE CHECK -- splits each model's episodes
   into "while still exploring" (eps_a above the floor threshold) vs
   "after exploration bottomed out" (eps_a at/near its floor), and
   compares success rate in each phase. A model whose success rate
   craters specifically AFTER epsilon reaches its floor is showing a
   symptom consistent with the DQN converging to a degenerate,
   repetitive greedy policy -- a real, known DQN failure mode, distinct
   from ordinary noise. Flagged explicitly with a WARNING line so it
   can't be silently averaged away by only looking at a final summary
   number.

3. Handles any number of models, all pairwise comparisons (z-test,
   Cohen's h, Mann-Whitney on steps and reward), collapse-firing rate,
   controller override rate, and an honest single-seed scope note at
   the end.
"""

import argparse
import csv
import itertools
import math
from collections import defaultdict

# epsilon_decay=0.998, epsilon_min=0.10 in dqn.py by default -- eps
# effectively reaches its floor once it's within this tolerance of it.
# If your run uses different DQN hyperparameters, override with --floor.
DEFAULT_EPS_FLOOR_THRESHOLD = 0.11


def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval for a proportion -- better-behaved than
    normal approximation when the proportion is near 0 or 1, which is
    common early/late in RL training."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def two_proportion_z(s1, n1, s2, n2):
    """Two-proportion z-test. Returns (z, p_two_tailed). None if
    undefined (e.g. zero variance)."""
    if n1 == 0 or n2 == 0:
        return None, None
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def cohens_h(p1, p2):
    """Effect size for two proportions. Rule of thumb: 0.2 small,
    0.5 medium, 0.8 large -- independent of sample size, unlike p-value."""
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return phi1 - phi2


def mann_whitney_u(a, b):
    """Mann-Whitney U test, no scipy dependency. Normal approximation
    (fine for n > ~20 per group, which any real run here will have)."""
    if not a or not b:
        return None, None
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    n1, n2 = len(a), len(b)
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mean_u = n1 * n2 / 2.0
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if std_u == 0:
        return u, 1.0
    z = (u - mean_u) / std_u
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u, p


def load_rows(path):
    by_model = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_model[row["model"]].append(row)
    return by_model


def combined_success_series(rows):
    """Each compare_models.py row logs BOTH agent A and agent B for that
    model in one row -- treat each as its own trial."""
    out = []
    for r in rows:
        out.append(int(r["success_a"]))
        out.append(int(r["success_b"]))
    return out


def has_overrides_column(rows):
    return bool(rows) and "overrides_a" in rows[0]


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(p * len(sorted_vals))))
    return sorted_vals[idx]


def analyze(path, tail_n, eps_floor):
    by_model = load_rows(path)
    tags = sorted(by_model.keys())
    print(f"=== compare_log.csv analysis: {path} ===")
    print(f"Models found: {tags}")
    for tag in tags:
        print(f"  {tag}: {len(by_model[tag])} episodes logged")
    print()

    print("--- Convergence check (is training still moving?) ---")
    print("first-100-trial vs last-100-trial success rate (both agents pooled),")
    print("plus current epsilon -- large gap or epsilon far from epsilon_min means")
    print("the run hasn't converged and a snapshot comparison compares two still-")
    print("moving targets, not two settled policies.\n")

    for tag in tags:
        rows = by_model[tag]
        s = combined_success_series(rows)
        first_rate = sum(s[:100]) / min(100, len(s)) if s else float("nan")
        last_rate = sum(s[-100:]) / min(100, len(s)) if s else float("nan")
        eps_a_last = float(rows[-1]["eps_a"]) if rows else float("nan")
        eps_b_last = float(rows[-1]["eps_b"]) if rows else float("nan")
        print(f"  {tag:20s} first-100={first_rate:5.1%}  last-100={last_rate:5.1%}  "
              f"eps_a={eps_a_last:.3f} eps_b={eps_b_last:.3f}")
    print()

    print("--- Reward distribution (all episodes) ---")
    print("A capped per-step penalty can still sum to a large negative TOTAL")
    print("over a full episode if the agent repeats one bad action the whole")
    print("time -- that's bounded, not a runaway blowup, but still worth seeing.\n")
    for tag in tags:
        rows = by_model[tag]
        for col, label in (("reward_a", "reward_a"), ("reward_b", "reward_b")):
            vals = sorted(float(r[col]) for r in rows)
            if not vals:
                continue
            print(f"  {tag:20s} {label:10s} min={vals[0]:8.1f}  p1={percentile(vals,0.01):8.1f}  "
                  f"p99={percentile(vals,0.99):7.1f}  max={vals[-1]:6.1f}")
    print()

    print(f"--- POST-EPSILON-FLOOR collapse check (floor threshold: eps <= {eps_floor}) ---")
    print("Splits episodes into 'still exploring' vs 'exploration bottomed out',")
    print("and compares success rate in each phase. A model whose rate craters")
    print("specifically AFTER hitting the floor is showing a symptom consistent")
    print("with the DQN converging to a degenerate, repetitive greedy policy --")
    print("a real DQN failure mode, not just noise.\n")
    for tag in tags:
        rows = by_model[tag]
        pre = [r for r in rows if float(r["eps_a"]) > eps_floor]
        post = [r for r in rows if float(r["eps_a"]) <= eps_floor]
        pre_s = combined_success_series(pre)
        post_s = combined_success_series(post)
        pre_rate = sum(pre_s) / len(pre_s) if pre_s else float("nan")
        post_rate = sum(post_s) / len(post_s) if post_s else float("nan")
        warn = ""
        if pre_s and post_s and post_rate < pre_rate * 0.5 and post_rate < 0.15:
            warn = "  <-- WARNING: rate roughly halved (or worse) after floor; check for degenerate policy"
        print(f"  {tag:20s} pre-floor n={len(pre_s):5d} rate={pre_rate:5.1%}   "
              f"post-floor n={len(post_s):5d} rate={post_rate:5.1%}{warn}")
    print()

    print(f"--- Success rate, last {tail_n} episodes per model (2*{tail_n} trials, A+B pooled) ---")
    tail_success = {}
    for tag in tags:
        rows = by_model[tag][-tail_n:]
        s = combined_success_series(rows)
        n = len(s)
        succ = sum(s)
        rate = succ / n if n else float("nan")
        lo, hi = wilson_ci(succ, n)
        tail_success[tag] = (succ, n, rate)
        print(f"  {tag:20s} {succ}/{n} = {rate:5.1%}   95% CI [{lo:5.1%}, {hi:5.1%}]")
    print()

    print(f"--- collapse (objective-reduction) firing rate, last {tail_n} episodes ---")
    for tag in tags:
        rows = by_model[tag][-tail_n:]
        or_events = sum(int(r["or_a"]) + int(r["or_b"]) for r in rows)
        total = 2 * len(rows)
        print(f"  {tag:20s} {or_events}/{total} = {100*or_events/total if total else 0:5.1f}% of trials")
    print()

    any_rows = next(iter(by_model.values()), [])
    if has_overrides_column(any_rows):
        print(f"--- metacognitive controller overrides per episode, last {tail_n} episodes ---")
        print("(0 is expected/correct for models with no controller -- e.g. A, B)")
        for tag in tags:
            rows = by_model[tag][-tail_n:]
            if not rows:
                continue
            ov = [int(r["overrides_a"]) + int(r["overrides_b"]) for r in rows]
            avg_ov = sum(ov) / (2 * len(rows))
            print(f"  {tag:20s} avg overrides/agent/episode = {avg_ov:6.1f}")
        print()

    if len(tags) >= 2:
        print("=" * 70)
        print(f"PAIRWISE COMPARISONS (last {tail_n} episodes each)")
        print("=" * 70)
        for tag1, tag2 in itertools.combinations(tags, 2):
            (s1, n1, p1), (s2, n2, p2) = tail_success[tag1], tail_success[tag2]
            print(f"\n--- {tag1} vs {tag2} ---")

            z, p = two_proportion_z(s1, n1, s2, n2)
            h = cohens_h(p1, p2)
            if z is not None:
                sig = "YES (p < 0.05)" if p < 0.05 else "NO (p >= 0.05)"
                print(f"  success rate:  z={z:+.3f}, p={p:.4f}  -- significant? {sig}")
            print(f"                 Cohen's h={h:+.3f}  "
                  f"({'negligible' if abs(h) < 0.2 else 'small' if abs(h) < 0.5 else 'medium' if abs(h) < 0.8 else 'large'} effect)")

            for metric_prefix, label in (("steps", "steps-to-solve"), ("reward", "reward")):
                vals = {}
                for tag in (tag1, tag2):
                    rows = by_model[tag][-tail_n:]
                    if metric_prefix == "steps":
                        vals[tag] = [int(r["steps_a"]) for r in rows if int(r["success_a"])] + \
                                    [int(r["steps_b"]) for r in rows if int(r["success_b"])]
                    else:
                        vals[tag] = [float(r["reward_a"]) for r in rows] + \
                                    [float(r["reward_b"]) for r in rows]
                a, b = vals[tag1], vals[tag2]
                u, p = mann_whitney_u(a, b)
                mean_a = sum(a) / len(a) if a else float("nan")
                mean_b = sum(b) / len(b) if b else float("nan")
                if u is not None:
                    sig = "YES" if p < 0.05 else "NO"
                    print(f"  {label:14s} {tag1}: mean={mean_a:7.2f} (n={len(a)})  "
                          f"{tag2}: mean={mean_b:7.2f} (n={len(b)})  "
                          f"U-test p={p:.4f} sig?{sig}")
        print()

    print("=" * 70)
    print("HONEST SCOPE NOTE (read before writing any claim):")
    print("This is ONE seed per model from ONE training run. A significant")
    print("result here describes THIS run, not a generalizable finding.")
    print("A real claim needs multiple seeds per model, averaged, before")
    print("'Model X outperforms Model Y' can be said without a caveat.")
    print("If p < 0.05 here, treat it as 'promising, worth replicating,'")
    print("not as a settled result.")
    print()
    print("Also remember: Model B vs Model C isolates whether an EXPLICIT")
    print("override controller does more than passively exposing the same")
    print("confidence/entropy_trend signal as observation features. Model A")
    print("vs (B or C) isolates whether collapse+metacognition together")
    print("change behavior at all, relative to neither.")
    print()
    print("If the post-epsilon-floor check above shows a WARNING for any")
    print("model, treat that model's late-training numbers with caution --")
    print("a degenerate repeated-action policy inflates negative reward and")
    print("crushes success rate independently of whatever mechanism you're")
    print("actually trying to test, and can make a comparison against that")
    print("model look better (or worse) than the mechanism itself deserves.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="compare_log.csv")
    parser.add_argument("--tail", type=int, default=500,
                         help="how many most-recent episodes per model to test (default 500)")
    parser.add_argument("--floor", type=float, default=DEFAULT_EPS_FLOOR_THRESHOLD,
                         help="epsilon value at/below which exploration is considered "
                              "'at its floor' for the collapse check (default 0.11)")
    args = parser.parse_args()
    analyze(args.csv_path, args.tail, args.floor)
