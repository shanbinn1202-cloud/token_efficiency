"""Vectorized, bit-identical pieces of Selector V2 for long histories (CPU only).

The reference selection (``query_selection.QuerySelectionAuthority`` through the
upstream ``PrefillPageEmbeddingSelectorV2``) scores every candidate Page with
``dram_kv.page_vectors.compact_cosine`` in pure Python: about 10 us per Page,
twice per turn, plus Page records, vector references and sorts. At P = 16 and
128-256K tokens (8-16K Pages) that is 0.3-0.5 s on the engine thread per
admission. This module computes the same floats with numpy:

* ``cosines``: ``compact_cosine(query, page)`` for every Page at once. The Page
  unit vectors are the upstream ``CompactVector.unit_values`` themselves
  (computed once per Page, kept in a growing column matrix); the dot product
  reproduces CPython 3.12's ``sum()`` of floats step by step (Neumaier's
  compensated summation over the 128 products in order, then the final
  compensation, then the [-1, 1] clamp). Every operation is one IEEE-754 double
  add, subtract, multiply, compare or absolute value, elementwise, so each Page
  gets exactly the float the reference computes (no reassociation, no FMA).
* ``fill_order``: ``selection_contract.fill_order`` with the SplitMix64 outputs
  drawn as uint64 vectors (wrapping arithmetic is exact modulo 2^64); a draw
  that the reference would reject (about one in 2^64 / m) falls back to the
  reference loop, so the permutation is always the reference one.

``available()`` runs a startup self-check of both against the reference
implementations on fixed inputs (ties, zeros, extremes, tiny and large scales)
and enables the fast path only if every float and every permutation is
identical; otherwise the callers keep the reference path. Nothing here hashes
anything and nothing runs per request beyond the vector arithmetic.
"""
from __future__ import annotations

import math

D_MAX = 65536
_STATE = {}


def numpy():
    try:
        import numpy as np
    except ImportError:          # pragma: no cover - numpy is a vLLM dependency
        return None
    return np


def cosines(np, columns, query_unit):
    """``compact_cosine(query, page)`` per column of ``columns`` (D x n float64 unit vectors).

    ``query_unit`` is the query's ``unit_values()`` (D floats). Returns n float64.
    """
    d, n = columns.shape
    q = np.asarray(query_unit, dtype=np.float64)
    if q.shape != (d,):
        raise ValueError('compact vector dimensionality changed')
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    # sum(l * r for l, r in zip(query, page)): start 0 (int) + first product.
    f = columns[0] * q[0]
    f += 0.0
    c = np.zeros(n, dtype=np.float64)
    x = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    big = np.empty(n, dtype=np.bool_)
    fa = np.empty(n, dtype=np.float64)
    xa = np.empty(n, dtype=np.float64)
    d1 = np.empty(n, dtype=np.float64)
    d2 = np.empty(n, dtype=np.float64)
    for i in range(1, d):
        np.multiply(columns[i], q[i], out=x)
        np.add(f, x, out=t)
        # if fabs(f) >= fabs(x): c += (f - t) + x  else: c += (x - t) + f
        np.abs(f, out=fa)
        np.abs(x, out=xa)
        np.greater_equal(fa, xa, out=big)
        np.subtract(f, t, out=d1)
        d1 += x
        np.subtract(x, t, out=d2)
        d2 += f
        np.copyto(d2, d1, where=big)
        c += d2
        f, t = t, f
    # if (c && isfinite(c)) f += c
    keep = (c != 0.0) & np.isfinite(c)
    np.add(f, c, out=f, where=keep)
    # max(-1.0, min(1.0, s))
    np.copyto(f, 1.0, where=~(f < 1.0))
    np.copyto(f, -1.0, where=~(f > -1.0))
    return f


class PageColumns:
    """One session's Page unit vectors as a growing D x capacity float64 matrix.

    Column p holds ``CompactVector.unit_values(vector of Page p)`` (the upstream
    method, so the values are the reference ones by construction). ``objects``
    keeps the vector each column was computed from: a Page whose catalog vector
    is another object is compared by value (sealed Pages are immutable; a
    different value is refused like the upstream index refuses an overwrite).
    """

    def __init__(self, np, key, dimension, unit_values):
        self.np, self.key, self.dimension, self.unit_values = np, key, dimension, unit_values
        self.objects = []
        self.matrix = np.empty((dimension, 0), dtype=np.float64)

    def view(self, vectors, n):
        """Columns [0, n) for ``vectors`` (a mapping Page -> CompactVector covering [0, n))."""
        np, objects = self.np, self.objects
        k = min(len(objects), n)
        for page in range(k):
            vector = vectors[page]
            if vector is not objects[page]:
                if vector != objects[page]:
                    raise ValueError('a sealed physical Page vector cannot be overwritten')
                objects[page] = vector
        if n > len(objects):
            fresh = [vectors[page] for page in range(len(objects), n)]
            for vector in fresh:
                if vector.dimension != self.dimension:
                    raise ValueError('compact vector dimensionality changed')
            rows = np.array([self.unit_values(vector) for vector in fresh], dtype=np.float64)
            start = len(objects)
            if self.matrix.shape[1] < n:
                grown = np.empty((self.dimension, max(n, 2 * self.matrix.shape[1], 256)), dtype=np.float64)
                grown[:, :start] = self.matrix[:, :start]
                self.matrix = grown
            self.matrix[:, start:n] = rows.T
            objects.extend(fresh)
        return self.matrix[:, :n]


_MASK64 = (1 << 64) - 1
_GOLDEN64 = 0x9E3779B97F4A7C15


def splitmix64_outputs(np, key, count):
    """The first ``count`` outputs of ``selection_contract.splitmix64`` from state ``key``
    (state k = key + k * golden mod 2^64), as uint64."""
    u = np.uint64
    with np.errstate(over='ignore'):
        state = np.arange(1, count + 1, dtype=np.uint64) * u(_GOLDEN64) + u(key)
        z = (state ^ (state >> u(30))) * u(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> u(27))) * u(0x94D049BB133111EB)
        return z ^ (z >> u(31))


def rejected(np, z, bound):
    """Where the reference draws again: z >= 2^64 - (2^64 mod bound) (never when bound | 2^64)."""
    u = np.uint64
    with np.errstate(over='ignore'):
        rest = (u(0) - bound) % bound                         # 2^64 mod bound
        return (rest != 0) & (z >= (u(0) - rest))


def fill_order(np, eligible, seed, session_index, turn, reference):
    """``reference(eligible, seed, session_index, turn)`` (selection_contract.fill_order),
    with the SplitMix64 draws vectorized; any draw the reference would reject
    makes this return the reference result itself."""
    order = list(eligible)
    m = len(order)
    if m < 2:
        return reference(eligible, seed, session_index, turn)
    from .selection_contract import fill_stream_key
    z = splitmix64_outputs(np, fill_stream_key(seed, session_index, turn), m - 1)
    bound = np.arange(m, 1, -1, dtype=np.uint64)              # m - i for i = 0 .. m-2
    if bool(rejected(np, z, bound).any()):
        return reference(eligible, seed, session_index, turn)
    jumps = (z % bound).tolist()
    for i, step in enumerate(jumps):
        j = i + step
        order[i], order[j] = order[j], order[i]
    return order


def _lcg(state):
    state = (state * 6364136223846793005 + 1442695040888963407) & _MASK64
    return state, state >> 33


def _check_vectors(np, vectors_module):
    """Fast cosines against compact_cosine on fixed vectors: every float identical."""
    CompactVector, compact_cosine = vectors_module.CompactVector, vectors_module.compact_cosine
    unit = vectors_module.CompactVector.unit_values
    state, vectors = 20261007, []
    for k in range(96):
        values = []
        for i in range(128):
            state, r = _lcg(state)
            if k % 7 == 0:
                v = (r % 3) - 1                       # many zeros, ties
            elif k % 7 == 1:
                v = 127 if r % 2 else -127            # extremes
            else:
                v = int(r % 255) - 127
            values.append(v)
        if not any(values):
            values[0] = 1
        state, r = _lcg(state)
        scale = (1 + r % 1000) / 127000.0 if k % 5 else (r % 97 + 1) * 1e-7
        vectors.append(CompactVector(tuple(values), 'int8', float(scale)))
    vectors.append(CompactVector((1,) + (0,) * 127, 'int8', 1.0))
    vectors.append(CompactVector((-1,) + (0,) * 127, 'int8', 0.5))
    vectors.append(vectors[3])
    vectors.append(CompactVector(vectors[3].values, 'int8', vectors[3].scale * 2))
    columns = np.array([unit(v) for v in vectors], dtype=np.float64).T.copy()
    for query in vectors[:12] + vectors[-4:]:
        got = cosines(np, columns, unit(query)).tolist()
        want = [compact_cosine(query, v) for v in vectors]
        if any(a != b or math.copysign(1.0, a) != math.copysign(1.0, b) for a, b in zip(got, want)):
            return False
    return True


def _check_fill(np):
    from .selection_contract import fill_order as reference
    for m, seed, session, turn in ((2, 0, 0, 0), (3, 42, 1, 7), (17, 42, 0, 3), (300, 7, 2, 255),
                                   (1000, (1 << 32) - 1, (1 << 16) - 1, (1 << 16) - 1)):
        eligible = list(range(5, 5 + m))
        if fill_order(np, eligible, seed, session, turn, reference) != reference(eligible, seed, session, turn):
            return False
    return True


def available(vectors_module):
    """Startup only: numpy present and both fast paths reproduce the reference exactly."""
    key = id(vectors_module)
    if key not in _STATE:
        np = numpy()
        ok = False
        if np is not None:
            try:
                ok = _check_vectors(np, vectors_module) and _check_fill(np)
            except Exception:
                ok = False
        _STATE[key] = np if ok else None
    return _STATE[key]
