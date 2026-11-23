"""
benchmark_structures.py -- Empirical comparison for Assignment 6, Part 2.

Compares array-based and linked-list-based structures on the operations where
their asymptotic costs differ, and measures per-element memory overhead.

Experiments
    stack      N pushes then N pops          ArrayStack  vs LinkedStack
    queue      N enqueues then N dequeues    ArrayQueue  vs LinkedQueue
    front      N insertions at the front     DynamicArray.insert(0, x) vs SinglyLinkedList.insert_front
    access     2000 random reads by index    DynamicArray[i]           vs SinglyLinkedList.get(i)
    back       remove the last element N times  DynamicArray.pop()     vs SinglyLinkedList.delete_back()

Usage:
    python benchmark_structures.py [--quick] [--no-plots]

Outputs (in ./results): structure_timings.csv, structure_comparison.png
"""

import argparse
import csv
import gc
import os
import random
import statistics
import sys
import time

from data_structures import (ArrayQueue, ArrayStack, DynamicArray, LinkedQueue,
                             LinkedStack, Node, SinglyLinkedList)

RESULTS_DIR = "results"
SEED = 20261122


def timed(fn, repeats):
    """Median wall-clock seconds of fn() over ``repeats`` runs."""
    times = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        gc.enable()
    return statistics.median(times)


# ---- workloads ---------------------------------------------------------------
def stack_workload(cls, n):
    def run():
        s = cls()
        for i in range(n):
            s.push(i)
        for _ in range(n):
            s.pop()
    return run


def queue_workload(cls, n):
    def run():
        q = cls()
        for i in range(n):
            q.enqueue(i)
        for _ in range(n):
            q.dequeue()
    return run


def front_insert_array(n):
    def run():
        a = DynamicArray()
        for i in range(n):
            a.insert(0, i)
    return run


def front_insert_list(n):
    def run():
        ll = SinglyLinkedList()
        for i in range(n):
            ll.insert_front(i)
    return run


def access_array(n, idxs):
    a = DynamicArray(range(n))

    def run():
        for i in idxs:
            a[i]
    return run


def access_list(n, idxs):
    ll = SinglyLinkedList(range(n))

    def run():
        for i in idxs:
            ll.get(i)
    return run


def back_pop_array(n):
    def run():
        a = DynamicArray(range(n))
        for _ in range(n):
            a.pop()
    return run


def back_pop_list(n):
    def run():
        ll = SinglyLinkedList(range(n))
        for _ in range(n):
            ll.delete_back()
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    repeats = 3
    big = [10_000, 50_000, 100_000, 200_000] if not args.quick else [5_000, 20_000]
    small = [1_000, 2_000, 4_000, 8_000] if not args.quick else [500, 1_000]
    rng = random.Random(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []

    def record(experiment, impl, n, seconds, ops):
        rows.append({"experiment": experiment, "implementation": impl, "n": n,
                     "total_seconds": seconds, "microseconds_per_op": seconds / ops * 1e6})

    print("experiment   implementation          n    total s   us/op")

    def show(experiment, impl, n, seconds, ops):
        record(experiment, impl, n, seconds, ops)
        print(f"{experiment:<10}  {impl:<18}{n:>8}  {seconds:>8.4f}  {seconds / ops * 1e6:>8.3f}")

    for n in big:
        show("stack", "array", n, timed(stack_workload(ArrayStack, n), repeats), 2 * n)
        show("stack", "linked", n, timed(stack_workload(LinkedStack, n), repeats), 2 * n)
    for n in big:
        show("queue", "array", n, timed(queue_workload(ArrayQueue, n), repeats), 2 * n)
        show("queue", "linked", n, timed(queue_workload(LinkedQueue, n), repeats), 2 * n)
    for n in small:
        show("front", "array", n, timed(front_insert_array(n), repeats), n)
        show("front", "linked", n, timed(front_insert_list(n), repeats), n)
    accesses = 2000
    for n in small:
        idxs = [rng.randrange(n) for _ in range(accesses)]
        show("access", "array", n, timed(access_array(n, idxs), repeats), accesses)
        show("access", "linked", n, timed(access_list(n, idxs), repeats), accesses)
    for n in small:
        show("back", "array", n, timed(back_pop_array(n), repeats), n)
        show("back", "linked", n, timed(back_pop_list(n), repeats), n)

    with open(os.path.join(RESULTS_DIR, "structure_timings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- memory per element (CPython, 64-bit) -----------------------------------
    slot = sys.getsizeof([None] * 1000) / 1000
    node = sys.getsizeof(Node(0))
    print(f"\nmemory per element: array slot = {slot:.1f} bytes, "
          f"linked-list node = {node} bytes ({node / slot:.1f}x), "
          "both excluding the stored object itself")

    if not args.no_plots:
        make_plot(rows)
        print(f"wrote {RESULTS_DIR}/structure_comparison.png")


def make_plot(rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    style = {"array": dict(color="#2a78d6", marker="o", label="array-based"),
             "linked": dict(color="#eb6834", marker="s", label="linked-list-based")}
    plt.rcParams.update({
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
        "text.color": ink, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
        "axes.edgecolor": grid, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })
    panels = [
        ("stack", "Stack: push then pop", "microseconds_per_op", "us per operation"),
        ("queue", "Queue: enqueue then dequeue", "microseconds_per_op", "us per operation"),
        ("front", "Insert at the front", "total_seconds", "total seconds"),
        ("access", "Random access by index", "microseconds_per_op", "us per access"),
        ("back", "Remove the last element", "total_seconds", "total seconds"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    flat = axes.flatten()
    for ax, (exp, title, col, ylabel) in zip(flat, panels):
        for impl in ("array", "linked"):
            pts = sorted((r["n"], r[col]) for r in rows
                         if r["experiment"] == exp and r["implementation"] == impl)
            s = style[impl]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=s["color"], marker=s["marker"],
                    label=s["label"], linewidth=2, markersize=6,
                    markeredgecolor=surface, markeredgewidth=1.2)
        ax.set_title(title, loc="left", color=ink, fontsize=11)
        ax.set_xlabel("n (number of elements)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="x", style="plain")
    flat[5].axis("off")
    h, l = flat[0].get_legend_handles_labels()
    flat[5].legend(h, l, loc="center left", frameon=False, fontsize=11)
    fig.suptitle("Array-based vs linked-list-based structures", x=0.01, ha="left", fontsize=13, color=ink)
    fig.savefig(os.path.join(RESULTS_DIR, "structure_comparison.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
