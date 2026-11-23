"""Correctness tests for selection.py (run: python -m unittest discover -v)."""

import random
import unittest

from selection import deterministic_select, lower_median, randomized_select

ALGORITHMS = {
    "deterministic": deterministic_select,
    "randomized": lambda arr, k: randomized_select(arr, k, random.Random(1)),
}


class SelectionTests(unittest.TestCase):
    def check_all_ranks(self, data):
        expected = sorted(data)
        for name, fn in ALGORITHMS.items():
            for k in range(1, len(data) + 1):
                self.assertEqual(fn(data, k), expected[k - 1],
                                 f"{name} failed for k={k}, data={data}")

    # ---- small and edge cases -------------------------------------------
    def test_single_element(self):
        self.check_all_ranks([42])

    def test_two_elements(self):
        self.check_all_ranks([2, 1])
        self.check_all_ranks([1, 2])

    def test_sizes_around_group_boundary(self):
        # exercises 1-group, exact-multiple-of-5, and ragged last-group cases
        rng = random.Random(7)
        for n in list(range(1, 32)) + [49, 50, 51, 99, 100, 101]:
            data = rng.sample(range(10 * n), n)
            self.check_all_ranks(data)

    def test_all_equal(self):
        self.check_all_ranks([7] * 50)

    def test_many_duplicates(self):
        rng = random.Random(3)
        for _ in range(50):
            n = rng.randint(1, 80)
            self.check_all_ranks([rng.randint(0, 3) for _ in range(n)])

    def test_negative_numbers_and_floats(self):
        self.check_all_ranks([-5, 3.5, 0, -1.25, 9, -5, 2])

    def test_strings(self):
        self.check_all_ranks(["pear", "apple", "fig", "apple", "kiwi", "date"])

    def test_sorted_reverse_sorted(self):
        self.check_all_ranks(list(range(100)))
        self.check_all_ranks(list(range(100, 0, -1)))

    def test_organ_pipe(self):
        data = list(range(50)) + list(range(50, 0, -1))
        self.check_all_ranks(data)

    # ---- randomized stress ------------------------------------------------
    def test_random_stress(self):
        rng = random.Random(2024)
        for _ in range(300):
            n = rng.randint(1, 200)
            hi = rng.choice([2, 10, 1000])
            data = [rng.randint(0, hi) for _ in range(n)]
            k = rng.randint(1, n)
            expected = sorted(data)[k - 1]
            self.assertEqual(deterministic_select(data, k), expected)
            self.assertEqual(randomized_select(data, k, random.Random(rng.random())), expected)

    def test_large_inputs(self):
        rng = random.Random(5)
        data = [rng.randint(0, 10**6) for _ in range(20000)]
        expected = sorted(data)
        for k in (1, 10000, 20000):
            self.assertEqual(deterministic_select(data, k), expected[k - 1])
            self.assertEqual(randomized_select(data, k), expected[k - 1])

    def test_large_all_equal_is_fast_enough(self):
        # With a two-way partition this would be quadratic; three-way is linear.
        data = [1] * 50000
        self.assertEqual(deterministic_select(data, 25000), 1)
        self.assertEqual(randomized_select(data, 25000), 1)

    # ---- API contract -------------------------------------------------------
    def test_input_not_modified(self):
        data = [5, 3, 9, 1, 7, 3, 8, 2]
        snapshot = list(data)
        deterministic_select(data, 4)
        randomized_select(data, 4)
        self.assertEqual(data, snapshot)

    def test_accepts_tuple(self):
        self.assertEqual(deterministic_select((4, 2, 9), 2), 4)
        self.assertEqual(randomized_select((4, 2, 9), 2), 4)

    def test_invalid_k_and_empty(self):
        for fn in (deterministic_select, randomized_select):
            with self.assertRaises(ValueError):
                fn([], 1)
            with self.assertRaises(ValueError):
                fn([1, 2, 3], 0)
            with self.assertRaises(ValueError):
                fn([1, 2, 3], 4)
            with self.assertRaises(TypeError):
                fn([1, 2, 3], 1.5)
            with self.assertRaises(TypeError):
                fn([1, 2, 3], True)

    def test_lower_median(self):
        self.assertEqual(lower_median([3, 1, 2]), 2)
        self.assertEqual(lower_median([4, 1, 3, 2]), 2)

    def test_randomized_is_reproducible_with_seed(self):
        data = list(range(1000))
        random.Random(0).shuffle(data)
        a = randomized_select(data, 500, random.Random(99))
        b = randomized_select(data, 500, random.Random(99))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
