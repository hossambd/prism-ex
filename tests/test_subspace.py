"""Marker subspace selection (the join between 2.1 and 2.2)."""

from __future__ import annotations

import numpy as np
import pytest

from prism_ex.errors import AmbiguousMarker, ConfigurationError, UnknownMarker
from prism_ex.fcs.reader import read_fcs_bytes
from prism_ex.fcs.writer import build_fcs_bytes
from prism_ex.subspace import select_subspace


@pytest.fixture
def fcs():
    rng = np.random.default_rng(3)
    matrix = rng.uniform(1, 10_000, size=(50, 4))
    return read_fcs_bytes(
        build_fcs_bytes(
            matrix, ["FL1-A", "FL2-A", "FL3-A", "FSC-A"], stains=["CD3", "CD4", None, None]
        )
    )


def test_markers_resolve_by_name_and_by_stain(fcs):
    subspace = select_subspace(fcs, ["CD3", "FL3-A"])
    assert subspace.markers == ("FL1-A", "FL3-A")
    assert subspace.requested == ("CD3", "FL3-A")


def test_column_order_follows_the_request_not_the_file(fcs):
    assert select_subspace(fcs, ["FL3-A", "FL1-A"]).markers == ("FL3-A", "FL1-A")


def test_unknown_marker_names_the_alternatives(fcs):
    with pytest.raises(UnknownMarker, match="FL1-A"):
        select_subspace(fcs, ["CD8"])


def test_repeating_a_marker_is_an_error_not_a_silent_deduplication(fcs):
    """A repeated marker would double that marker's weight in every distance."""
    with pytest.raises(AmbiguousMarker):
        select_subspace(fcs, ["CD3", "FL1-A"])


def test_empty_request_is_rejected(fcs):
    with pytest.raises(ConfigurationError):
        select_subspace(fcs, [])


def test_zscore_scaling_standardises_each_column(fcs):
    matrix = select_subspace(fcs, ["FL1-A", "FL2-A"], scaling="zscore").matrix
    np.testing.assert_allclose(matrix.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(matrix.std(axis=0), 1, atol=1e-12)


def test_constant_channel_does_not_produce_nan():
    """Zero variance must not become a division by zero half way down the pipeline."""
    matrix = np.column_stack([np.ones(20), np.arange(20.0)])
    fcs = read_fcs_bytes(build_fcs_bytes(matrix, ["flat", "ramp"]))
    subspace = select_subspace(fcs, ["flat", "ramp"], scaling="zscore")
    assert np.isfinite(subspace.matrix).all()


def test_asinh_is_monotone_and_compresses_the_top_decade(fcs):
    raw = select_subspace(fcs, ["FL1-A"], transform="none", scaling="none").matrix
    transformed = select_subspace(fcs, ["FL1-A"], transform="asinh", scaling="none").matrix
    assert np.array_equal(np.argsort(raw, axis=0), np.argsort(transformed, axis=0))
    assert np.ptp(transformed) < np.ptp(raw)


def test_subset_reuses_the_scaling(fcs):
    """Re-standardising a subset would make resampled distances incomparable."""
    subspace = select_subspace(fcs, ["FL1-A", "FL2-A"])
    rows = np.arange(0, 50, 2)
    np.testing.assert_array_equal(subspace.subset(rows).matrix, subspace.matrix[rows])
