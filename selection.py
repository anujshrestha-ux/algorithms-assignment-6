"""
selection.py -- Order-statistic selection algorithms (Assignment 6, Part 1)

Two algorithms for finding the k-th smallest element of an unsorted sequence
(see Cormen, Leiserson, Rivest, Stein, *Introduction to Algorithms*, 4th ed.,
Chapter 9, "Medians and Order Statistics"):

    deterministic_select(arr, k)   Median of Medians (BFPRT), O(n) worst case
    randomized_select(arr, k)      Randomized Quickselect,   O(n) expected time

Conventions
-----------
* ``k`` is 1-based: ``k = 1`` is the minimum, ``k = len(arr)`` the maximum.
* Duplicates are counted separately.  The k-th smallest element is therefore
  exactly ``sorted(arr)[k - 1]``.
* The input sequence is never modified; each function works on a private copy.
* Both algorithms partition three ways (``< pivot``, ``== pivot``,
  ``> pivot``).  This is what makes arrays with many duplicates (even arrays
  where every element is equal) run in linear time.  A classic two-way
  Lomuto/Hoare partition degrades to O(n^2) on such inputs.

Both functions are iterative at the top level (a loop narrows the window
``a[lo..hi]`` that must contain the answer), so recursion depth is small:
the deterministic algorithm only recurses to find the median of the group
medians, which shrinks the problem by a factor of 5 each level.
"""

import random
from typing import Any, List, Optional, Sequence

GROUP_SIZE = 5  # group size used by Median of Medians (odd, >= 5 gives linear time)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _check_args(n: int, k: int) -> None:
    """Validate the input size and the rank k (1-based)."""
    if n == 0:
        raise ValueError("cannot select from an empty sequence")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= {n}, got {k}")


def _insertion_sort(a: List[Any], lo: int, hi: int) -> None:
    """Sort a[lo..hi] (inclusive) in place. Used only on <= 5 elements: O(1)."""
    for i in range(lo + 1, hi + 1):
        key = a[i]
        j = i - 1
        while j >= lo and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


def _partition3(a: List[Any], lo: int, hi: int, pivot: Any):
    """
    Three-way (Dutch national flag) partition of a[lo..hi] around ``pivot``.

    Returns (lt, gt) such that afterwards
        a[lo .. lt-1]  <  pivot
        a[lt .. gt]    == pivot
        a[gt+1 .. hi]  >  pivot
    Runs in Theta(hi - lo + 1) time and O(1) extra space.
    """
    lt, i, gt = lo, lo, hi
    while i <= gt:
        x = a[i]
        if x < pivot:
            a[lt], a[i] = a[i], a[lt]
            lt += 1
            i += 1
        elif x > pivot:
            a[i], a[gt] = a[gt], a[i]
            gt -= 1
        else:
            i += 1
    return lt, gt


# --------------------------------------------------------------------------
# Deterministic selection: Median of Medians, worst-case O(n)
# --------------------------------------------------------------------------
def _median_of_medians_pivot(a: List[Any], lo: int, hi: int) -> Any:
    """
    Return a pivot value guaranteed to have a constant fraction of the
    elements of a[lo..hi] on each side.

    1. Split a[lo..hi] into groups of 5 (the last group may be smaller).
    2. Insertion-sort each group and move its median to the front of the
       window, into positions lo, lo+1, ..., lo+m-1.
    3. Recursively select the median of those m medians.
    """
    m = 0
    for start in range(lo, hi + 1, GROUP_SIZE):
        end = min(start + GROUP_SIZE - 1, hi)
        _insertion_sort(a, start, end)
        median_pos = start + (end - start) // 2
        a[lo + m], a[median_pos] = a[median_pos], a[lo + m]
        m += 1
    # Median of the m medians stored in a[lo .. lo+m-1]
    return _mom_select(a, lo, lo + m - 1, lo + (m - 1) // 2)


def _mom_select(a: List[Any], lo: int, hi: int, k: int) -> Any:
    """Return the value that belongs at index k (absolute) once a[lo..hi] is sorted."""
    while True:
        if hi - lo + 1 <= GROUP_SIZE:
            _insertion_sort(a, lo, hi)
            return a[k]
        pivot = _median_of_medians_pivot(a, lo, hi)
        lt, gt = _partition3(a, lo, hi, pivot)
        if k < lt:
            hi = lt - 1          # answer is among the elements smaller than pivot
        elif k > gt:
            lo = gt + 1          # answer is among the elements larger than pivot
        else:
            return pivot         # k falls in the block equal to the pivot


def deterministic_select(arr: Sequence[Any], k: int) -> Any:
    """
    Return the k-th smallest element (1-based) of ``arr`` in worst-case O(n)
    time using the Median of Medians algorithm.

    >>> deterministic_select([9, 1, 8, 2, 7, 3], 3)
    3
    >>> deterministic_select([5, 5, 5, 5], 2)
    5
    """
    _check_args(len(arr), k)
    a = list(arr)  # private copy; the caller's data is untouched
    return _mom_select(a, 0, len(a) - 1, k - 1)


# --------------------------------------------------------------------------
# Randomized selection: Quickselect, expected O(n)
# --------------------------------------------------------------------------
def randomized_select(arr: Sequence[Any], k: int,
                      rng: Optional[random.Random] = None) -> Any:
    """
    Return the k-th smallest element (1-based) of ``arr`` in expected O(n)
    time using Randomized Quickselect.

    ``rng`` may be a ``random.Random`` instance for reproducible runs.

    >>> randomized_select([9, 1, 8, 2, 7, 3], 3)
    3
    >>> randomized_select([4, 4, 1, 4], 1)
    1
    """
    _check_args(len(arr), k)
    rand = rng if rng is not None else random
    a = list(arr)
    lo, hi, idx = 0, len(a) - 1, k - 1
    while lo < hi:
        pivot = a[rand.randint(lo, hi)]
        lt, gt = _partition3(a, lo, hi, pivot)
        if idx < lt:
            hi = lt - 1
        elif idx > gt:
            lo = gt + 1
        else:
            return pivot
    return a[idx]  # window shrank to one element, which must be at idx


# --------------------------------------------------------------------------
# Convenience helper
# --------------------------------------------------------------------------
def lower_median(arr: Sequence[Any]) -> Any:
    """Return the lower median, i.e. the ceil(n/2)-th smallest element."""
    return deterministic_select(arr, (len(arr) + 1) // 2)


if __name__ == "__main__":
    import doctest

    data = [38, 27, 43, 3, 9, 82, 10, 3, 27]
    print("data:", data, "sorted:", sorted(data))
    for kk in (1, 5, len(data)):
        print(f"k={kk}: deterministic={deterministic_select(data, kk)}, "
              f"randomized={randomized_select(data, kk)}")
    print("doctest:", doctest.testmod())
