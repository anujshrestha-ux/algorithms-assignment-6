"""Correctness tests for data_structures.py (run: python -m unittest discover -v)."""

import random
import unittest

from data_structures import (ArrayQueue, ArrayStack, DynamicArray, LinkedQueue,
                             LinkedStack, Matrix, RootedTree, SinglyLinkedList)


class DynamicArrayTests(unittest.TestCase):
    def test_append_access_and_growth(self):
        a = DynamicArray()
        for i in range(100):
            a.append(i)
        self.assertEqual(len(a), 100)
        self.assertEqual([a[i] for i in range(100)], list(range(100)))
        self.assertGreaterEqual(a.capacity(), 100)

    def test_negative_index_and_set(self):
        a = DynamicArray([1, 2, 3])
        self.assertEqual(a[-1], 3)
        a[-1] = 30
        self.assertEqual(list(a), [1, 2, 30])

    def test_insert_delete_match_python_list(self):
        rng = random.Random(11)
        a, ref = DynamicArray(), []
        for _ in range(2000):
            if ref and rng.random() < 0.45:
                i = rng.randrange(len(ref))
                self.assertEqual(a.delete(i), ref.pop(i))
            else:
                i = rng.randint(0, len(ref))
                v = rng.randint(0, 99)
                a.insert(i, v)
                ref.insert(i, v)
            self.assertEqual(list(a), ref)

    def test_shrinks_after_many_deletes(self):
        a = DynamicArray(range(1000))
        big = a.capacity()
        while len(a) > 10:
            a.pop()
        self.assertLess(a.capacity(), big)
        self.assertEqual(list(a), list(range(10)))

    def test_errors_and_index(self):
        a = DynamicArray([5, 6, 7])
        self.assertEqual(a.index(7), 2)
        with self.assertRaises(ValueError):
            a.index(99)
        with self.assertRaises(IndexError):
            a[3]
        with self.assertRaises(IndexError):
            a.insert(5, 0)
        with self.assertRaises(IndexError):
            DynamicArray().pop()


class MatrixTests(unittest.TestCase):
    def test_get_set_shape(self):
        m = Matrix(2, 3, fill=0)
        m.set(1, 2, 9)
        self.assertEqual(m.get(1, 2), 9)
        self.assertEqual(m[0, 0], 0)
        self.assertEqual(m.shape, (2, 3))
        with self.assertRaises(IndexError):
            m.get(2, 0)

    def test_from_lists_and_transpose(self):
        m = Matrix.from_lists([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.transpose().to_lists(), [[1, 4], [2, 5], [3, 6]])

    def test_insert_and_delete_row(self):
        m = Matrix.from_lists([[1, 2], [3, 4]])
        m.insert_row(1, [9, 9])
        self.assertEqual(m.to_lists(), [[1, 2], [9, 9], [3, 4]])
        self.assertEqual(m.delete_row(0), [1, 2])
        self.assertEqual(m.to_lists(), [[9, 9], [3, 4]])
        with self.assertRaises(ValueError):
            m.insert_row(0, [1])

    def test_ragged_rejected(self):
        with self.assertRaises(ValueError):
            Matrix.from_lists([[1, 2], [3]])


class StackQueueTests(unittest.TestCase):
    def test_stack_lifo(self):
        for cls in (ArrayStack, LinkedStack):
            s = cls()
            self.assertTrue(s.is_empty())
            for i in range(50):
                s.push(i)
            self.assertEqual(s.peek(), 49)
            self.assertEqual([s.pop() for _ in range(50)], list(range(49, -1, -1)))
            with self.assertRaises(IndexError):
                s.pop()
            with self.assertRaises(IndexError):
                s.peek()

    def test_queue_fifo(self):
        for cls in (ArrayQueue, LinkedQueue):
            q = cls()
            for i in range(50):
                q.enqueue(i)
            self.assertEqual(q.front(), 0)
            self.assertEqual([q.dequeue() for _ in range(50)], list(range(50)))
            with self.assertRaises(IndexError):
                q.dequeue()
            with self.assertRaises(IndexError):
                q.front()

    def test_circular_queue_wraparound_and_resize(self):
        q = ArrayQueue()
        ref = []
        rng = random.Random(4)
        for step in range(5000):
            if ref and rng.random() < 0.5:
                self.assertEqual(q.dequeue(), ref.pop(0))
            else:
                q.enqueue(step)
                ref.append(step)
            self.assertEqual(len(q), len(ref))
            if ref:
                self.assertEqual(q.front(), ref[0])

    def test_stack_matches_python_list(self):
        rng = random.Random(8)
        s, ref = ArrayStack(), []
        for step in range(5000):
            if ref and rng.random() < 0.5:
                self.assertEqual(s.pop(), ref.pop())
            else:
                s.push(step)
                ref.append(step)
            self.assertEqual(len(s), len(ref))


class LinkedListTests(unittest.TestCase):
    def test_insertions(self):
        ll = SinglyLinkedList()
        ll.insert_back(2)
        ll.insert_front(1)
        ll.insert_back(4)
        ll.insert_at(2, 3)
        ll.insert_at(0, 0)
        ll.insert_at(5, 5)
        self.assertEqual(list(ll), [0, 1, 2, 3, 4, 5])
        self.assertEqual(ll.tail.value, 5)
        self.assertEqual(len(ll), 6)

    def test_insert_after_updates_tail(self):
        ll = SinglyLinkedList([1, 2])
        ll.insert_after(ll.tail, 3)
        self.assertEqual(ll.tail.value, 3)
        self.assertEqual(list(ll), [1, 2, 3])

    def test_deletions(self):
        ll = SinglyLinkedList([1, 2, 3, 4, 5])
        self.assertEqual(ll.delete_front(), 1)
        self.assertEqual(ll.delete_back(), 5)
        self.assertEqual(ll.delete_at(1), 3)
        self.assertTrue(ll.delete_value(4))
        self.assertFalse(ll.delete_value(99))
        self.assertEqual(list(ll), [2])
        self.assertEqual(ll.delete_back(), 2)
        self.assertEqual(len(ll), 0)
        self.assertIsNone(ll.head)
        self.assertIsNone(ll.tail)
        with self.assertRaises(IndexError):
            ll.delete_front()
        with self.assertRaises(IndexError):
            ll.delete_back()

    def test_tail_stays_correct_after_deleting_last(self):
        ll = SinglyLinkedList([1, 2, 3])
        ll.delete_value(3)
        ll.insert_back(9)
        self.assertEqual(list(ll), [1, 2, 9])
        ll.delete_at(2)
        ll.insert_back(8)
        self.assertEqual(list(ll), [1, 2, 8])

    def test_search_get_reverse(self):
        ll = SinglyLinkedList([10, 20, 30])
        self.assertEqual(ll.search(20).value, 20)
        self.assertIsNone(ll.search(99))
        self.assertEqual(ll.get(2), 30)
        with self.assertRaises(IndexError):
            ll.get(3)
        ll.reverse()
        self.assertEqual(list(ll), [30, 20, 10])
        self.assertEqual(ll.tail.value, 10)
        ll.insert_back(0)
        self.assertEqual(list(ll), [30, 20, 10, 0])

    def test_random_ops_match_python_list(self):
        rng = random.Random(21)
        ll, ref = SinglyLinkedList(), []
        for _ in range(3000):
            op = rng.random()
            if ref and op < 0.35:
                i = rng.randrange(len(ref))
                self.assertEqual(ll.delete_at(i), ref.pop(i))
            elif op < 0.7:
                i = rng.randint(0, len(ref))
                v = rng.randint(0, 99)
                ll.insert_at(i, v)
                ref.insert(i, v)
            elif op < 0.85:
                v = rng.randint(0, 99)
                ll.insert_back(v)
                ref.append(v)
            else:
                v = rng.randint(0, 99)
                ll.insert_front(v)
                ref.insert(0, v)
            self.assertEqual(list(ll), ref)
            self.assertEqual(len(ll), len(ref))
            if ref:
                self.assertEqual(ll.tail.value, ref[-1])


class RootedTreeTests(unittest.TestCase):
    def build(self):
        t = RootedTree("A")
        b = t.add_child(t.root, "B")
        c = t.add_child(t.root, "C")
        d = t.add_child(b, "D")
        t.add_child(b, "E")
        t.add_child(d, "F")
        t.add_child(c, "G")
        return t

    def test_traversals_and_height(self):
        t = self.build()
        self.assertEqual(t.preorder(), ["A", "B", "D", "F", "E", "C", "G"])
        self.assertEqual(t.level_order(), ["A", "B", "C", "D", "E", "G", "F"])
        self.assertEqual(t.height(), 3)
        self.assertEqual(len(t), 7)

    def test_find_and_children(self):
        t = self.build()
        self.assertEqual([n.key for n in t.find("B").children()], ["D", "E"])
        self.assertIsNone(t.find("Z"))

    def test_remove_subtree(self):
        t = self.build()
        self.assertEqual(t.remove_subtree(t.find("B")), 4)  # B, D, F, E
        self.assertEqual(t.preorder(), ["A", "C", "G"])
        self.assertEqual(len(t), 3)
        self.assertEqual(t.remove_subtree(t.find("G")), 1)
        self.assertEqual(t.preorder(), ["A", "C"])

    def test_remove_middle_sibling_and_root(self):
        t = RootedTree(0)
        kids = [t.add_child(t.root, i) for i in (1, 2, 3)]
        t.remove_subtree(kids[1])
        self.assertEqual(t.preorder(), [0, 1, 3])
        self.assertEqual(t.remove_subtree(t.root), 3)
        self.assertEqual(t.preorder(), [])
        self.assertEqual(t.height(), -1)

    def test_deep_tree_does_not_overflow_stack(self):
        t = RootedTree(0)
        node = t.root
        for i in range(1, 5000):
            node = t.add_child(node, i)
        self.assertEqual(t.height(), 4999)
        self.assertEqual(len(t.preorder()), 5000)


if __name__ == "__main__":
    unittest.main()
