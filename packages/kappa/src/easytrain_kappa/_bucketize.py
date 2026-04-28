"""Score-to-bucket bucketization for kappa.

A bin specification is a sorted sequence of edges. With edges
``[e0, e1, ..., en]`` you get ``n`` buckets:

* bucket 0 = ``[e0, e1)``
* bucket 1 = ``[e1, e2)``
* ...
* bucket n-1 = ``[e_{n-1}, e_n]``  (right-closed for the final bin only)

The asymmetry on the final bin matches numpy's ``digitize(right=False)``
convention and means a perfect 1.0 score lands in the top bucket rather
than overflowing.

Out-of-range scores raise :class:`ValueError`. Graders are typed to
return values in [0, 1] so any out-of-range value indicates a bug
upstream — silent clamping would mask it.
"""

from __future__ import annotations

from itertools import pairwise

DEFAULT_BIN_EDGES: list[float] = [0.0, 0.5, 1.0]
"""Pass/fail default: 2 buckets split at 0.5."""


def validate_bin_edges(edges: list[float]) -> None:
    """Raise if ``edges`` is not a strictly-increasing list of >= 2 floats.

    Strictly increasing matters because equal edges would create empty
    buckets that confuse the kappa expected-agreement formula.
    """
    if len(edges) < 2:
        raise ValueError(f"need at least 2 bin edges to form 1 bucket, got {len(edges)}")
    for prev, curr in pairwise(edges):
        if curr <= prev:
            raise ValueError(
                f"bin edges must be strictly increasing; got {prev} >= {curr}"
            )


def n_buckets(edges: list[float]) -> int:
    return len(edges) - 1


def bucketize(score: float, edges: list[float]) -> int:
    """Return the bucket index for ``score`` given ``edges``.

    Half-open intervals on every bucket except the last (which is closed
    on both sides). Out-of-range values raise ``ValueError`` — see module
    docstring.
    """
    if score < edges[0] or score > edges[-1]:
        raise ValueError(
            f"score {score!r} is outside bin range [{edges[0]}, {edges[-1]}]"
        )
    # Last bucket is right-closed: a score exactly at the top edge belongs
    # to the final bucket, not bucket n (which doesn't exist).
    if score == edges[-1]:
        return n_buckets(edges) - 1
    # Linear scan — bin counts are tiny (typically 2-5) so this is faster
    # than bisect for the realistic case and avoids the import.
    for i, hi in enumerate(edges[1:]):
        if score < hi:
            return i
    # Unreachable given the range check above, but Python needs it.
    raise AssertionError(  # pragma: no cover
        f"bucketize fell through for score={score!r} edges={edges!r}"
    )


__all__ = ["DEFAULT_BIN_EDGES", "bucketize", "n_buckets", "validate_bin_edges"]
