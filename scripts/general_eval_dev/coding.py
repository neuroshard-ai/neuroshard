'''Development coding items: a Python function from a specification, scored by hidden tests in the sandbox.

Every item carries a reference solution that the builder runs against its own hidden tests in the same
sandbox; an item whose reference does not pass is rejected. Tests use ``same`` (exact type and value) and
reach the solution only through the ``solution`` namespace.
'''

ASK = ('\n\nReply with the complete Python 3 code in one ```python block. Use only the standard library; do not '
       'read input or print.')


def code(entry, reference, tests, timeout=10):
    return {'kind': 'code', 'entry_point': entry, 'reference': reference.strip() + '\n',
            'tests': f"f = solution['{entry}']\n" + tests.strip() + '\n', 'timeout': timeout}


CODING = [
    ('strings', 'Write `run_length(s: str) -> str` that run-length encodes a string: each maximal run of one '
                'character becomes the character followed by the run length. "aaabcc" becomes "a3b1c2"; the empty '
                'string stays empty.',
     code('run_length', '''
def run_length(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)''', '''
assert same(f("aaabcc"), "a3b1c2")
assert same(f(""), "")
assert same(f("z"), "z1")
assert same(f("aaaaaaaaaaaab"), "a12b1")
assert same(f("abAB"), "a1b1A1B1")''')),
    ('parsing', 'Write `balanced(s: str) -> bool` that returns True when the brackets (), [] and {} in `s` are '
                'balanced and properly nested. Other characters are ignored.',
     code('balanced', '''
def balanced(s):
    pairs, stack = {")": "(", "]": "[", "}": "{"}, []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack''', '''
assert f("") is True and f("a(b)c") is True and f("{[()()]}") is True
assert f("(]") is False and f("((") is False and f(")(") is False and f("[(])") is False''')),
    ('intervals', 'Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping '
                  'or touching closed intervals and returns them sorted by start. The input may be unsorted; do not '
                  'modify it.',
     code('merge_intervals', '''
def merge_intervals(intervals):
    out = []
    for a, b in sorted(map(list, intervals)):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out''', '''
data = [[8, 10], [1, 3], [2, 6], [15, 18]]
assert same(f(data), [[1, 6], [8, 10], [15, 18]])
assert same(data, [[8, 10], [1, 3], [2, 6], [15, 18]])
assert same(f([[1, 4], [4, 5]]), [[1, 5]])
assert same(f([]), [])
assert same(f([[5, 7], [1, 10]]), [[1, 10]])''')),
    ('text', 'Write `word_frequencies(text: str) -> dict[str, int]` that counts words case-insensitively. A word is a '
             'maximal run of letters and apostrophes; everything else separates words.',
     code('word_frequencies', '''
import re
def word_frequencies(text):
    counts = {}
    for w in re.findall(r"[A-Za-z']+", text):
        counts[w.lower()] = counts.get(w.lower(), 0) + 1
    return counts''', '''
assert same(f("The cat and the hat."), {"the": 2, "cat": 1, "and": 1, "hat": 1})
assert same(f("Don't stop, don't!"), {"don't": 2, "stop": 1})
assert same(f("123 -- !!"), {})''')),
    ('matrices', 'Write `rotate(matrix: list[list[int]]) -> list[list[int]]` that returns the matrix rotated 90 '
                 'degrees clockwise as a new list of lists. The matrix may be rectangular.',
     code('rotate', '''
def rotate(matrix):
    return [list(row) for row in zip(*matrix[::-1])]''', '''
assert same(f([[1, 2], [3, 4]]), [[3, 1], [4, 2]])
assert same(f([[1, 2, 3]]), [[1], [2], [3]])
assert same(f([[1], [2], [3]]), [[3, 2, 1]])''')),
    ('lists', 'Write `second_largest(nums: list[int]) -> int | None` returning the second largest distinct value, '
              'or None if there are fewer than two distinct values.',
     code('second_largest', '''
def second_largest(nums):
    values = sorted(set(nums))
    return values[-2] if len(values) >= 2 else None''', '''
assert same(f([3, 1, 4, 4, 2]), 3)
assert f([7, 7]) is None and f([]) is None
assert same(f([-5, -1, -3]), -3)''')),
    ('parsing', 'Write `roman_to_int(s: str) -> int` converting a valid Roman numeral (I, V, X, L, C, D, M, with '
                'subtractive forms such as IV and CM) between 1 and 3999 to an integer.',
     code('roman_to_int', '''
def roman_to_int(s):
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        if i + 1 < len(s) and v[ch] < v[s[i + 1]]:
            total -= v[ch]
        else:
            total += v[ch]
    return total''', '''
assert same(f("III"), 3) and same(f("IV"), 4) and same(f("IX"), 9)
assert same(f("MCMXCIV"), 1994) and same(f("MMMCMXCIX"), 3999) and same(f("XLII"), 42)''')),
    ('recursion', 'Write `flatten(nested: list) -> list` that flattens arbitrarily nested lists into one flat list, '
                  'keeping order. Only lists are flattened; tuples and strings are kept as single elements.',
     code('flatten', '''
def flatten(nested):
    out = []
    for x in nested:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out''', '''
assert same(f([1, [2, [3, [4]], 5]]), [1, 2, 3, 4, 5])
assert same(f([]), []) and same(f([[[]]]), [])
assert same(f(["ab", (1, 2), [3]]), ["ab", (1, 2), 3])''')),
    ('lists', 'Write `chunk(items: list, size: int) -> list[list]` splitting a list into consecutive chunks of '
              '`size` (the last may be shorter). Raise ValueError if size < 1.',
     code('chunk', '''
def chunk(items, size):
    if size < 1:
        raise ValueError("size must be positive")
    return [items[i:i + size] for i in range(0, len(items), size)]''', '''
assert same(f([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])
assert same(f([], 3), [])
try:
    f([1], 0)
    raise AssertionError("expected ValueError")
except ValueError:
    pass''')),
    ('dates', 'Write `days_between(a: str, b: str) -> int` returning the absolute number of days between two ISO '
              'dates (YYYY-MM-DD).',
     code('days_between', '''
from datetime import date
def days_between(a, b):
    return abs((date.fromisoformat(b) - date.fromisoformat(a)).days)''', '''
assert same(f("2024-02-28", "2024-03-01"), 2)
assert same(f("2023-03-01", "2023-02-28"), 1)
assert same(f("2000-01-01", "2000-01-01"), 0)
assert same(f("1999-12-31", "2026-10-30"), 9800)''')),
    ('counting', 'Write `top_k(nums: list[int], k: int) -> list[int]` returning the k most frequent values, most '
                 'frequent first; ties go to the smaller value first.',
     code('top_k', '''
from collections import Counter
def top_k(nums, k):
    return [v for v, _ in sorted(Counter(nums).items(), key=lambda p: (-p[1], p[0]))[:k]]''', '''
assert same(f([1, 1, 1, 2, 2, 3], 2), [1, 2])
assert same(f([4, 4, 5, 5, 6], 2), [4, 5])
assert same(f([9], 3), [9])''')),
    ('strings', 'Write `caesar(text: str, shift: int) -> str` that shifts ASCII letters by `shift` places in the '
                'alphabet (wrapping, negative shifts allowed), keeps their case, and leaves every other character '
                'unchanged.',
     code('caesar', '''
def caesar(text, shift):
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)''', '''
assert same(f("Hello, World!", 3), "Khoor, Zruog!")
assert same(f("abc", -1), "zab")
assert same(f("xyz", 29), "abc")
assert same(f("é 1", 5), "é 1")''')),
    ('parsing', 'Write `parse_duration(text: str) -> int` that converts strings like "1h30m", "45s" or "2h5s" into '
                'a number of seconds. Units are h, m and s, each at most once, in that order. Raise ValueError on '
                'anything else, including the empty string.',
     code('parse_duration', '''
import re
def parse_duration(text):
    m = re.fullmatch(r"(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", text)
    if not text or not m:
        raise ValueError(text)
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s''', '''
assert same(f("1h30m"), 5400) and same(f("45s"), 45) and same(f("2h5s"), 7205) and same(f("0m"), 0)
for bad in ["", "5x", "30m1h", "h"]:
    try:
        f(bad)
        raise AssertionError("accepted " + bad)
    except ValueError:
        pass''')),
    ('strings', 'Write `common_prefix(words: list[str]) -> str` returning the longest common prefix of all words '
                '("" for an empty list).',
     code('common_prefix', '''
import os
def common_prefix(words):
    return os.path.commonprefix(words) if words else ""''', '''
assert same(f(["flower", "flow", "flight"]), "fl")
assert same(f(["dog", "racecar"]), "")
assert same(f([]), "") and same(f(["solo"]), "solo")''')),
    ('search', 'Write `leftmost(sorted_list: list[int], x: int) -> int` returning the index of the first occurrence '
               'of x in a sorted list, or -1. It must run in O(log n): it will be called on a list of ten million '
               'items.',
     code('leftmost', '''
import bisect
def leftmost(sorted_list, x):
    i = bisect.bisect_left(sorted_list, x)
    return i if i < len(sorted_list) and sorted_list[i] == x else -1''', '''
big = list(range(0, 20_000_000, 2))
for _ in range(20000):
    assert same(f(big, 19_999_998), 9_999_999)
assert same(f(big, 3), -1)
assert same(f([1, 2, 2, 2, 3], 2), 1) and same(f([], 1), -1)''', timeout=20)),
    ('performance', 'Write `count_primes(n: int) -> int` returning how many primes are strictly less than n. It must '
                    'handle n up to 5,000,000 within a few seconds.',
     code('count_primes', '''
def count_primes(n):
    if n < 3:
        return 0
    sieve = bytearray([1]) * n
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, n, i)))
    return sum(sieve)''', '''
assert same(f(0), 0) and same(f(2), 0) and same(f(3), 1) and same(f(10), 4) and same(f(100), 25)
assert same(f(5_000_000), 348513)''')),
    ('matrices', 'Write `spiral(matrix: list[list[int]]) -> list[int]` returning the elements in clockwise spiral '
                 'order starting at the top-left corner.',
     code('spiral', '''
def spiral(matrix):
    out, m = [], [list(r) for r in matrix]
    while m:
        out += m.pop(0)
        m = [list(r) for r in zip(*m)][::-1]
    return out''', '''
assert same(f([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [1, 2, 3, 6, 9, 8, 7, 4, 5])
assert same(f([[1, 2, 3, 4], [5, 6, 7, 8]]), [1, 2, 3, 4, 8, 7, 6, 5])
assert same(f([]), []) and same(f([[1], [2]]), [1, 2])''')),
    ('validation', 'Write `normalize_phone(s: str) -> str | None`. Keep only the digits of `s`. Ten digits are a US '
                   'number: return "+1" followed by them. Eleven digits starting with 1 are also a US number: return '
                   '"+" followed by them. Anything else returns None.',
     code('normalize_phone', '''
def normalize_phone(s):
    d = "".join(c for c in s if c.isdigit() and c.isascii())
    if len(d) == 10:
        return "+1" + d
    if len(d) == 11 and d[0] == "1":
        return "+" + d
    return None''', '''
assert same(f("(415) 555-0132"), "+14155550132")
assert same(f("1-415-555-0132"), "+14155550132")
assert f("555-0132") is None and f("2-415-555-0132") is None and f("") is None''')),
    ('classes', 'Write a class `LRUCache` with `__init__(self, capacity: int)`, `get(self, key) -> int` (returns -1 '
                'when missing) and `put(self, key, value) -> None`. When full, `put` of a new key evicts the least '
                'recently used key; both `get` and `put` count as use.',
     code('LRUCache', '''
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.capacity, self.data = capacity, OrderedDict()
    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]
    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)''', '''
c = f(2)
c.put(1, 1); c.put(2, 2)
assert same(c.get(1), 1)
c.put(3, 3)
assert same(c.get(2), -1)
c.put(1, 10); c.put(4, 4)
assert same(c.get(3), -1) and same(c.get(1), 10) and same(c.get(4), 4)''')),
    ('debugging', 'This function should return the median of a list of numbers without changing the list, but it '
                  'is wrong for even-length lists and it reorders the caller\'s list. Fix it.\n\n```python\ndef '
                  'median(xs):\n    xs.sort()\n    return xs[len(xs) // 2]\n```\n\nFor an even number of items the '
                  'median is the mean of the two middle values. Raise ValueError for an empty list.',
     code('median', '''
def median(xs):
    if not xs:
        raise ValueError("empty")
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2''', '''
data = [5, 1, 4, 2]
assert same(f(data), 3.0) and same(data, [5, 1, 4, 2])
assert same(f([3, 1, 2]), 2) and same(f([7]), 7)
try:
    f([])
    raise AssertionError("expected ValueError")
except ValueError:
    pass''')),
]
