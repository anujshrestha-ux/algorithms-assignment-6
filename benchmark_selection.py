"""
benchmark_selection.py -- Empirical comparison for Assignment 6, Part 1.

Times Median of Medians (deterministic), Randomized Quickselect, and a
sort-then-index baseline (``sorted(a)[k-1]``) on several input sizes and
input distributions, verifies every answer against the sorted order, writes
CSV files, and draws two figures.

Usage:
    python benchmark_selection.py            # full run (about a minute)
    python benchmark_selection.py --quick    # smaller sizes, fewer repeats
    python benchmark_selection.py --no-plots # CSV + console table only

Outputs (in ./results):
    selection_timings.csv        median/min runtime per (distribution, n, algorithm)
    selection_scaling.csv        fitted log-log slope per (distribution, algorithm)
    selection_vs_k.csv           runtime as the requested rank k varies (random input)
    selection_runtime.png        runtime vs n
    selection_per_element.png    runtime per element vs n (flat line == linear time)
"""

import argparse
import csv
import gc
import os
import random
import statistics
import time

from selection import deterministic_select, randomized_select

SEED = 20261122
RESULTS_DIR = "results"

ALGORITHMS = ["deterministic (median of medians)", "randomized (quickselect)", "sort + index (baseline)"]


def make_input(dist: str, n: int, rng: random.Random):
    if dist == "random":
        return rng.sample(range(n), n)          # random permutation, all distinct
    if dist == "sorted":
        return list(range(n))
    if dist == "reverse_sorted":
        return list(range(n, 0, -1))
    if dist == "few_unique":
        return [rng.randrange(10) for _ in range(n)]   # heavy duplicates
    if dist == "all_equal":
        return [7] * n
    raise ValueError(dist)


DISTRIBUTIONS = ["random", "sorted", "reverse_sorted", "few_unique", "all_equal"]


def run_algorithm(name: str, data, k: int, seed: int):
    if name.startswith("deterministic"):
        return deterministic_select(data, k)
    if name.startswith("randomized"):
        return randomized_select(data, k, random.Random(seed))
    return sorted(data)[k - 1]


def time_one(name: str, data, k: int, repeats: int):
    """Return (median_seconds, min_seconds) over ``repeats`` runs."""
    times = []
    for r in range(repeats):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        run_algorithm(name, data, k, seed=SEED + r)
        times.append(time.perf_counter() - t0)
        gc.enable()
    return statistics.median(times), min(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    sizes = [1_000, 2_000, 5_000, 10_000] if args.quick else \
        [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    repeats = 3 if args.quick else 5
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = random.Random(SEED)

    rows = []
    print(f"sizes={sizes} repeats={repeats} (median of repeats, seconds)\n")
    print(f"{'distribution':<15}{'n':>8}  " + "".join(f"{a.split(' ')[0]:>15}" for a in ALGORITHMS))
    for dist in DISTRIBUTIONS:
        for n in sizes:
            data = make_input(dist, n, rng)
            k = (n + 1) // 2                      # the median
            expected = sorted(data)[k - 1]
            line = f"{dist:<15}{n:>8}  "
            for algo in ALGORITHMS:
                # correctness check on every configuration
                got = run_algorithm(algo, data, k, SEED)
                assert got == expected, f"WRONG ANSWER: {algo} {dist} n={n}"
                med, mn = time_one(algo, data, k, repeats)
                rows.append({"distribution": dist, "n": n, "algorithm": algo,
                             "median_seconds": med, "min_seconds": mn})
                line += f"{med:>15.5f}"
            print(line)

    with open(os.path.join(RESULTS_DIR, "selection_timings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- scaling exponent: slope of log(time) vs log(n); 1.0 == linear ------
    import numpy as np
    scaling = []
    for dist in DISTRIBUTIONS:
        for algo in ALGORITHMS:
            pts = [(r["n"], r["median_seconds"]) for r in rows
                   if r["distribution"] == dist and r["algorithm"] == algo]
            xs = np.log([p[0] for p in pts])
            ys = np.log([p[1] for p in pts])
            slope = float(np.polyfit(xs, ys, 1)[0])
            scaling.append({"distribution": dist, "algorithm": algo, "loglog_slope": round(slope, 3)})
    with open(os.path.join(RESULTS_DIR, "selection_scaling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scaling[0]))
        w.writeheader()
        w.writerows(scaling)
    print("\nlog-log slope of runtime vs n (1.0 = linear, 2.0 = quadratic):")
    for s in scaling:
        print(f"  {s['distribution']:<15}{s['algorithm']:<36}{s['loglog_slope']:.3f}")

    # ---- effect of k on random input ------------------------------------------
    n_k = sizes[-1]
    data = make_input("random", n_k, rng)
    k_rows = []
    print(f"\nruntime vs rank k (random input, n={n_k}):")
    for label, k in [("k=1 (min)", 1), ("k=n/4", n_k // 4), ("k=n/2 (median)", n_k // 2),
                     ("k=3n/4", 3 * n_k // 4), ("k=n (max)", n_k)]:
        line = f"  {label:<16}"
        for algo in ALGORITHMS:
            assert run_algorithm(algo, data, k, SEED) == sorted(data)[k - 1]
            med, _ = time_one(algo, data, k, repeats)
            k_rows.append({"k_label": label, "k": k, "n": n_k, "algorithm": algo, "median_seconds": med})
            line += f"{med:>12.5f}"
        print(line)
    with open(os.path.join(RESULTS_DIR, "selection_vs_k.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(k_rows[0]))
        w.writeheader()
        w.writerows(k_rows)

    if not args.no_plots:
        make_plots(rows, sizes)
        print(f"\nwrote figures to {RESULTS_DIR}/")


# --------------------------------------------------------------------------
# Plots (matplotlib).  Colors: validated categorical slots 1-3 (blue, orange,
# aqua), assigned in fixed order; marker shape differs per series so identity
# never depends on color alone.
# --------------------------------------------------------------------------
def make_plots(rows, sizes) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    style = {
        ALGORITHMS[0]: dict(color="#2a78d6", marker="o"),
        ALGORITHMS[1]: dict(color="#eb6834", marker="s"),
        ALGORITHMS[2]: dict(color="#1baf7a", marker="^"),
    }
    plt.rcParams.update({
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
        "text.color": ink, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
        "axes.edgecolor": grid, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })

    def figure(ycol_fn, ylabel, title, filename):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
        flat = axes.flatten()
        for ax, dist in zip(flat, DISTRIBUTIONS):
            for algo in ALGORITHMS:
                pts = sorted((r["n"], r["median_seconds"]) for r in rows
                             if r["distribution"] == dist and r["algorithm"] == algo)
                ax.plot([p[0] for p in pts], [ycol_fn(p[0], p[1]) for p in pts],
                        label=algo, linewidth=2, markersize=6,
                        markeredgecolor=surface, markeredgewidth=1.2, **style[algo])
            ax.set_title(dist.replace("_", " "), loc="left", color=ink, fontsize=11)
            ax.set_xlabel("input size n")
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.ticklabel_format(axis="x", style="plain")
        legend_ax = flat[5]
        legend_ax.axis("off")
        handles, labels = flat[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=11)
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=ink)
        fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
        plt.close(fig)

    figure(lambda n, t: t, "median runtime (s)",
           "Selecting the median: runtime vs input size", "selection_runtime.png")
    figure(lambda n, t: t / n * 1e6, "microseconds per element",
           "Runtime per element (a flat line means linear time)", "selection_per_element.png")


if __name__ == "__main__":
    main()
