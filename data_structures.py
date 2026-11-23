"""
data_structures.py -- Elementary data structures (Assignment 6, Part 2)

Everything here is built from scratch.  The only Python container used is a
plain fixed-size ``list`` as a *raw block of memory* (``[None] * capacity``);
none of ``list.append``, ``list.insert``, ``list.pop`` or ``collections.deque``
is used to do the real work.  Growing and shrinking is done by hand.

Contents (see Cormen et al., *Introduction to Algorithms*, 4th ed., Ch. 10)
    DynamicArray     array with insert / delete / access, amortized O(1) append
    Matrix           2-D array with access, row insert/delete, transpose
    ArrayStack       LIFO stack on an array (push / pop / peek)
    ArrayQueue       FIFO queue on a circular array (enqueue / dequeue)
    SinglyLinkedList singly linked list with head and tail pointers
    LinkedStack      stack on a linked list      (for array-vs-list comparison)
    LinkedQueue      queue on a linked list      (for array-vs-list comparison)
    RootedTree       rooted tree, left-child / right-sibling representation
                     (optional part of the assignment)

Big-O summary (n = current number of elements)
    ----------------------------------------------------------------------
    DynamicArray   access O(1) | append O(1) amortized | insert/delete O(n)
                   search O(n)
    Matrix r x c   access O(1) | insert/delete row O(r) | transpose O(r*c)
    ArrayStack     push O(1) amortized | pop O(1) amortized | peek O(1)
    ArrayQueue     enqueue O(1) amortized | dequeue O(1) amortized
    LinkedList     insert/delete front O(1) | insert back O(1) (tail pointer)
                   delete back O(n) | access/search/insert-at/delete-at O(n)
    LinkedStack    push O(1) worst case | pop O(1) worst case
    LinkedQueue    enqueue O(1) worst case | dequeue O(1) worst case
    RootedTree     add child O(d) | traversal O(n) | remove subtree O(d + s)
    ----------------------------------------------------------------------
"""

from typing import Any, Iterable, Iterator, List, Optional

_MIN_CAPACITY = 4


def _new_block(capacity: int) -> List[Any]:
    """Allocate a fixed-size block of memory (all slots empty)."""
    return [None] * capacity


# ==========================================================================
# Arrays and matrices
# ==========================================================================
class DynamicArray:
    """
    Resizable array.  Capacity doubles when full and halves when the array is
    a quarter full, so a sequence of n appends/pops costs O(n) in total
    (amortized O(1) per operation).
    """

    def __init__(self, items: Iterable[Any] = ()):
        self._data = _new_block(_MIN_CAPACITY)
        self._size = 0
        for x in items:
            self.append(x)

    # ---- basic protocol ------------------------------------------------
    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        for i in range(self._size):
            yield self._data[i]

    def __repr__(self) -> str:
        return "DynamicArray([" + ", ".join(repr(x) for x in self) + "])"

    def capacity(self) -> int:
        return len(self._data)

    def _normalize(self, i: int, allow_end: bool = False) -> int:
        """Translate negative indexes and bounds-check. O(1)."""
        n = self._size
        if i < 0:
            i += n
        upper = n if allow_end else n - 1
        if not 0 <= i <= upper:
            raise IndexError("index out of range")
        return i

    def _resize(self, new_capacity: int) -> None:
        """Copy into a new block. O(n)."""
        new_data = _new_block(new_capacity)
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data

    # ---- access --------------------------------------------------------
    def __getitem__(self, i: int) -> Any:          # O(1)
        return self._data[self._normalize(i)]

    def __setitem__(self, i: int, value: Any) -> None:  # O(1)
        self._data[self._normalize(i)] = value

    # ---- insertion -----------------------------------------------------
    def append(self, value: Any) -> None:
        """Add at the end. O(1) amortized (O(n) only when a resize happens)."""
        if self._size == len(self._data):
            self._resize(2 * len(self._data))
        self._data[self._size] = value
        self._size += 1

    def insert(self, index: int, value: Any) -> None:
        """Insert before ``index`` (index == len appends). O(n - index) shifts."""
        index = self._normalize(index, allow_end=True)
        if self._size == len(self._data):
            self._resize(2 * len(self._data))
        for j in range(self._size, index, -1):   # shift right
            self._data[j] = self._data[j - 1]
        self._data[index] = value
        self._size += 1

    # ---- deletion ------------------------------------------------------
    def delete(self, index: int) -> Any:
        """Remove and return the element at ``index``. O(n - index) shifts."""
        index = self._normalize(index)
        removed = self._data[index]
        for j in range(index, self._size - 1):   # shift left
            self._data[j] = self._data[j + 1]
        self._size -= 1
        self._data[self._size] = None            # drop stale reference
        cap = len(self._data)
        if cap > _MIN_CAPACITY and self._size <= cap // 4:
            self._resize(cap // 2)
        return removed

    def pop(self) -> Any:
        """Remove and return the last element. O(1) amortized."""
        if self._size == 0:
            raise IndexError("pop from empty array")
        return self.delete(self._size - 1)

    # ---- search --------------------------------------------------------
    def index(self, value: Any) -> int:
        """Return the first index holding ``value`` (linear search, O(n))."""
        for i in range(self._size):
            if self._data[i] == value:
                return i
        raise ValueError(f"{value!r} is not in array")


class Matrix:
    """
    rows x cols matrix.  Each row is a fixed-size block and the rows are held
    in a DynamicArray, so element access is O(1) and whole rows can be
    inserted/deleted by moving row references instead of every element.
    """

    def __init__(self, rows: int, cols: int, fill: Any = 0):
        if rows < 0 or cols < 0:
            raise ValueError("dimensions must be non-negative")
        self._cols = cols
        self._rows = DynamicArray()
        for _ in range(rows):
            row = _new_block(cols)
            for c in range(cols):
                row[c] = fill
            self._rows.append(row)

    @classmethod
    def from_lists(cls, data: List[List[Any]]) -> "Matrix":
        """Build a matrix from a rectangular list of lists. O(r*c)."""
        rows = len(data)
        cols = len(data[0]) if rows else 0
        m = cls(rows, cols)
        for r in range(rows):
            if len(data[r]) != cols:
                raise ValueError("all rows must have the same length")
            for c in range(cols):
                m.set(r, c, data[r][c])
        return m

    @property
    def shape(self):
        return (len(self._rows), self._cols)

    def _check(self, r: int, c: int) -> None:
        if not (0 <= r < len(self._rows) and 0 <= c < self._cols):
            raise IndexError("matrix index out of range")

    def get(self, r: int, c: int) -> Any:                 # O(1)
        self._check(r, c)
        return self._rows[r][c]

    def set(self, r: int, c: int, value: Any) -> None:    # O(1)
        self._check(r, c)
        self._rows[r][c] = value

    def __getitem__(self, pos):
        return self.get(*pos)

    def __setitem__(self, pos, value) -> None:
        self.set(*pos, value)

    def insert_row(self, index: int, values: Iterable[Any]) -> None:
        """Insert a new row before ``index``. O(cols) to build + O(rows - index) to shift."""
        values = list(values)
        if len(values) != self._cols:
            raise ValueError(f"row must have {self._cols} values")
        row = _new_block(self._cols)
        for c in range(self._cols):
            row[c] = values[c]
        self._rows.insert(index, row)

    def delete_row(self, index: int) -> List[Any]:
        """Remove row ``index`` and return its values. O(cols + rows - index)."""
        row = self._rows.delete(index)
        return [row[c] for c in range(self._cols)]

    def transpose(self) -> "Matrix":
        """Return the transpose as a new matrix. O(rows * cols)."""
        r, c = self.shape
        t = Matrix(c, r)
        for i in range(r):
            for j in range(c):
                t.set(j, i, self.get(i, j))
        return t

    def to_lists(self) -> List[List[Any]]:
        r, c = self.shape
        return [[self.get(i, j) for j in range(c)] for i in range(r)]

    def __repr__(self) -> str:
        return f"Matrix({self.to_lists()})"


# ==========================================================================
# Stacks and queues on arrays
# ==========================================================================
class ArrayStack:
    """LIFO stack stored in an array with a ``top`` index (CLRS 10.1)."""

    def __init__(self, capacity: int = _MIN_CAPACITY):
        self._data = _new_block(max(capacity, _MIN_CAPACITY))
        self._top = 0                      # number of elements == next free slot

    def __len__(self) -> int:
        return self._top

    def is_empty(self) -> bool:
        return self._top == 0

    def _resize(self, new_capacity: int) -> None:
        new_data = _new_block(new_capacity)
        for i in range(self._top):
            new_data[i] = self._data[i]
        self._data = new_data

    def push(self, value: Any) -> None:
        """O(1) amortized: doubles the array when full."""
        if self._top == len(self._data):
            self._resize(2 * len(self._data))
        self._data[self._top] = value
        self._top += 1

    def pop(self) -> Any:
        """O(1) amortized: halves the array when it is a quarter full."""
        if self._top == 0:
            raise IndexError("pop from empty stack")
        self._top -= 1
        value = self._data[self._top]
        self._data[self._top] = None
        cap = len(self._data)
        if cap > _MIN_CAPACITY and self._top <= cap // 4:
            self._resize(cap // 2)
        return value

    def peek(self) -> Any:
        """O(1) worst case."""
        if self._top == 0:
            raise IndexError("peek at empty stack")
        return self._data[self._top - 1]


class ArrayQueue:
    """
    FIFO queue stored in a circular array (CLRS 10.1).  ``head`` is the index
    of the oldest element; the tail wraps around with modular arithmetic, so
    neither end ever needs to shift elements.
    """

    def __init__(self, capacity: int = _MIN_CAPACITY):
        self._data = _new_block(max(capacity, _MIN_CAPACITY))
        self._head = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def _resize(self, new_capacity: int) -> None:
        """Copy elements in queue order to index 0.. of a new block. O(n)."""
        cap = len(self._data)
        new_data = _new_block(new_capacity)
        for i in range(self._size):
            new_data[i] = self._data[(self._head + i) % cap]
        self._data = new_data
        self._head = 0

    def enqueue(self, value: Any) -> None:
        """O(1) amortized."""
        if self._size == len(self._data):
            self._resize(2 * len(self._data))
        self._data[(self._head + self._size) % len(self._data)] = value
        self._size += 1

    def dequeue(self) -> Any:
        """O(1) amortized."""
        if self._size == 0:
            raise IndexError("dequeue from empty queue")
        value = self._data[self._head]
        self._data[self._head] = None
        self._head = (self._head + 1) % len(self._data)
        self._size -= 1
        cap = len(self._data)
        if cap > _MIN_CAPACITY and self._size <= cap // 4:
            self._resize(cap // 2)
        return value

    def front(self) -> Any:
        """O(1) worst case."""
        if self._size == 0:
            raise IndexError("front of empty queue")
        return self._data[self._head]


# ==========================================================================
# Singly linked list (+ stack and queue built on it)
# ==========================================================================
class Node:
    """A node of a singly linked list."""

    __slots__ = ("value", "next")

    def __init__(self, value: Any, next: Optional["Node"] = None):
        self.value = value
        self.next = next


class SinglyLinkedList:
    """Singly linked list that keeps both a head and a tail pointer."""

    def __init__(self, items: Iterable[Any] = ()):
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self._size = 0
        for x in items:
            self.insert_back(x)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        """Traversal: O(n)."""
        node = self.head
        while node is not None:
            yield node.value
            node = node.next

    def __repr__(self) -> str:
        return "SinglyLinkedList(" + " -> ".join(repr(x) for x in self) + ")"

    # ---- insertion -----------------------------------------------------
    def insert_front(self, value: Any) -> Node:
        """O(1)."""
        node = Node(value, self.head)
        self.head = node
        if self.tail is None:
            self.tail = node
        self._size += 1
        return node

    def insert_back(self, value: Any) -> Node:
        """O(1) thanks to the tail pointer."""
        node = Node(value)
        if self.tail is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self._size += 1
        return node

    def insert_after(self, node: Node, value: Any) -> Node:
        """Insert after a node you already hold. O(1)."""
        new = Node(value, node.next)
        node.next = new
        if node is self.tail:
            self.tail = new
        self._size += 1
        return new

    def insert_at(self, index: int, value: Any) -> Node:
        """Insert so the new node ends up at ``index``. O(index)."""
        if not 0 <= index <= self._size:
            raise IndexError("index out of range")
        if index == 0:
            return self.insert_front(value)
        return self.insert_after(self._node_at(index - 1), value)

    # ---- access / search -----------------------------------------------
    def _node_at(self, index: int) -> Node:
        """Walk from the head. O(index)."""
        if not 0 <= index < self._size:
            raise IndexError("index out of range")
        node = self.head
        for _ in range(index):
            node = node.next
        return node

    def get(self, index: int) -> Any:
        """O(index) -- unlike an array, there is no random access."""
        return self._node_at(index).value

    def search(self, value: Any) -> Optional[Node]:
        """Return the first node holding ``value`` or None. O(n)."""
        node = self.head
        while node is not None:
            if node.value == value:
                return node
            node = node.next
        return None

    # ---- deletion ------------------------------------------------------
    def delete_front(self) -> Any:
        """O(1)."""
        if self.head is None:
            raise IndexError("delete from empty list")
        node = self.head
        self.head = node.next
        if self.head is None:
            self.tail = None
        self._size -= 1
        return node.value

    def delete_back(self) -> Any:
        """O(n): a singly linked list must walk to the node before the tail."""
        if self.head is None:
            raise IndexError("delete from empty list")
        if self.head is self.tail:
            return self.delete_front()
        prev = self.head
        while prev.next is not self.tail:
            prev = prev.next
        value = self.tail.value
        prev.next = None
        self.tail = prev
        self._size -= 1
        return value

    def delete_at(self, index: int) -> Any:
        """O(index)."""
        if not 0 <= index < self._size:
            raise IndexError("index out of range")
        if index == 0:
            return self.delete_front()
        prev = self._node_at(index - 1)
        node = prev.next
        prev.next = node.next
        if node is self.tail:
            self.tail = prev
        self._size -= 1
        return node.value

    def delete_value(self, value: Any) -> bool:
        """Delete the first node holding ``value``. O(n). Returns success."""
        prev, node = None, self.head
        while node is not None:
            if node.value == value:
                if prev is None:
                    self.delete_front()
                else:
                    prev.next = node.next
                    if node is self.tail:
                        self.tail = prev
                    self._size -= 1
                return True
            prev, node = node, node.next
        return False

    def reverse(self) -> None:
        """Reverse in place by re-pointing every ``next``. O(n), O(1) space."""
        prev, node = None, self.head
        self.tail = self.head
        while node is not None:
            node.next, prev, node = prev, node, node.next
        self.head = prev


class LinkedStack:
    """Stack on a singly linked list: push/pop at the head, O(1) worst case."""

    def __init__(self):
        self._list = SinglyLinkedList()

    def __len__(self) -> int:
        return len(self._list)

    def is_empty(self) -> bool:
        return len(self._list) == 0

    def push(self, value: Any) -> None:
        self._list.insert_front(value)

    def pop(self) -> Any:
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._list.delete_front()

    def peek(self) -> Any:
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._list.head.value


class LinkedQueue:
    """Queue on a singly linked list: enqueue at the tail, dequeue at the head."""

    def __init__(self):
        self._list = SinglyLinkedList()

    def __len__(self) -> int:
        return len(self._list)

    def is_empty(self) -> bool:
        return len(self._list) == 0

    def enqueue(self, value: Any) -> None:
        self._list.insert_back(value)

    def dequeue(self) -> Any:
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._list.delete_front()

    def front(self) -> Any:
        if self.is_empty():
            raise IndexError("front of empty queue")
        return self._list.head.value


# ==========================================================================
# Rooted trees (optional): left-child / right-sibling representation
# ==========================================================================
class TreeNode:
    """
    Node of a rooted tree with an unbounded number of children.  Each node
    stores only three pointers regardless of how many children it has:
    ``parent``, ``left_child`` (its first child) and ``right_sibling`` (the
    next child of the same parent).  The children of a node are therefore a
    singly linked list.
    """

    __slots__ = ("key", "parent", "left_child", "right_sibling")

    def __init__(self, key: Any):
        self.key = key
        self.parent: Optional["TreeNode"] = None
        self.left_child: Optional["TreeNode"] = None
        self.right_sibling: Optional["TreeNode"] = None

    def children(self) -> List["TreeNode"]:
        """List the children left to right. O(d), d = number of children."""
        out = []
        child = self.left_child
        while child is not None:
            out.append(child)
            child = child.right_sibling
        return out


class RootedTree:
    """Rooted tree built from linked TreeNodes (CLRS 10.3)."""

    def __init__(self, root_key: Any):
        self.root: Optional[TreeNode] = TreeNode(root_key)
        self._size = 1

    def __len__(self) -> int:
        return self._size

    def add_child(self, parent: TreeNode, key: Any) -> TreeNode:
        """Append a new last child to ``parent``. O(d) to find the last sibling."""
        node = TreeNode(key)
        node.parent = parent
        if parent.left_child is None:
            parent.left_child = node
        else:
            last = parent.left_child
            while last.right_sibling is not None:
                last = last.right_sibling
            last.right_sibling = node
        self._size += 1
        return node

    def remove_subtree(self, node: TreeNode) -> int:
        """
        Detach ``node`` and all its descendants.  Returns how many nodes were
        removed.  O(d + s): d to unlink from the sibling list, s to count the
        subtree.
        """
        removed = sum(1 for _ in self._preorder_from(node))
        if node.parent is None:                       # removing the root
            self.root = None
        else:
            parent = node.parent
            if parent.left_child is node:
                parent.left_child = node.right_sibling
            else:
                prev = parent.left_child
                while prev.right_sibling is not node:
                    prev = prev.right_sibling
                prev.right_sibling = node.right_sibling
        node.parent = None
        node.right_sibling = None
        self._size -= removed
        return removed

    def _preorder_from(self, start: TreeNode) -> Iterator[TreeNode]:
        stack = ArrayStack()
        stack.push(start)
        while not stack.is_empty():
            node = stack.pop()
            yield node
            for child in reversed(node.children()):   # leftmost child pops first
                stack.push(child)

    def preorder(self) -> List[Any]:
        """Keys in preorder (iterative, so deep trees cannot overflow the call stack). O(n)."""
        if self.root is None:
            return []
        return [n.key for n in self._preorder_from(self.root)]

    def level_order(self) -> List[Any]:
        """Keys level by level using a queue. O(n)."""
        if self.root is None:
            return []
        out, queue = [], ArrayQueue()
        queue.enqueue(self.root)
        while not queue.is_empty():
            node = queue.dequeue()
            out.append(node.key)
            for child in node.children():
                queue.enqueue(child)
        return out

    def height(self) -> int:
        """Number of edges on the longest root-to-leaf path (-1 if empty). O(n)."""
        if self.root is None:
            return -1
        height = -1
        level = [self.root]
        while level:
            height += 1
            nxt = []
            for node in level:
                nxt.extend(node.children())
            level = nxt
        return height

    def find(self, key: Any) -> Optional[TreeNode]:
        """First node (in preorder) with the given key, or None. O(n)."""
        if self.root is None:
            return None
        for node in self._preorder_from(self.root):
            if node.key == key:
                return node
        return None


if __name__ == "__main__":
    arr = DynamicArray([10, 20, 30])
    arr.insert(1, 15)
    print("after insert(1, 15):", arr)
    removed = arr.delete(0)
    print("after delete(0) ->", removed, ":", arr)

    s, q = ArrayStack(), ArrayQueue()
    for i in range(1, 4):
        s.push(i)
        q.enqueue(i)
    print("stack pops:", [s.pop() for _ in range(3)], "queue dequeues:", [q.dequeue() for _ in range(3)])

    ll = SinglyLinkedList([1, 2, 3])
    ll.insert_front(0)
    ll.delete_value(2)
    print(ll)

    t = RootedTree("A")
    b = t.add_child(t.root, "B")
    t.add_child(t.root, "C")
    t.add_child(b, "D")
    print("preorder:", t.preorder(), "level order:", t.level_order(), "height:", t.height())
