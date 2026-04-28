"""Tests for the bucketize helper."""

from __future__ import annotations

import pytest
from easytrain_kappa import DEFAULT_BIN_EDGES, bucketize
from easytrain_kappa._bucketize import n_buckets, validate_bin_edges


class TestValidateEdges:
    def test_too_few_edges(self) -> None:
        with pytest.raises(ValueError, match="at least 2 bin edges"):
            validate_bin_edges([0.5])

    def test_non_increasing(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_bin_edges([0.0, 0.5, 0.5, 1.0])

    def test_decreasing(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_bin_edges([0.0, 0.7, 0.3, 1.0])

    def test_valid_passes(self) -> None:
        validate_bin_edges([0.0, 0.5, 1.0])
        validate_bin_edges([0.0, 0.25, 0.5, 0.75, 1.0])


class TestBucketize:
    def test_default_edges_pass_fail(self) -> None:
        assert bucketize(0.0, DEFAULT_BIN_EDGES) == 0
        assert bucketize(0.49, DEFAULT_BIN_EDGES) == 0
        assert bucketize(0.5, DEFAULT_BIN_EDGES) == 1  # half-open: 0.5 -> upper bucket
        assert bucketize(0.99, DEFAULT_BIN_EDGES) == 1
        assert bucketize(1.0, DEFAULT_BIN_EDGES) == 1  # right-closed final bin

    def test_ternary_edges(self) -> None:
        edges = [0.0, 0.34, 0.67, 1.0]
        assert bucketize(0.0, edges) == 0
        assert bucketize(0.33, edges) == 0
        assert bucketize(0.34, edges) == 1
        assert bucketize(0.5, edges) == 1
        assert bucketize(0.67, edges) == 2
        assert bucketize(1.0, edges) == 2

    def test_below_range_raises(self) -> None:
        with pytest.raises(ValueError, match="outside bin range"):
            bucketize(-0.01, DEFAULT_BIN_EDGES)

    def test_above_range_raises(self) -> None:
        with pytest.raises(ValueError, match="outside bin range"):
            bucketize(1.01, DEFAULT_BIN_EDGES)

    def test_n_buckets(self) -> None:
        assert n_buckets([0.0, 1.0]) == 1
        assert n_buckets([0.0, 0.5, 1.0]) == 2
        assert n_buckets([0.0, 0.25, 0.5, 0.75, 1.0]) == 4
