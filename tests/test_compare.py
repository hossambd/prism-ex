"""Section 2.3: comparing two communities.

The tests that matter here are the negative ones. Any comparison function will
report a difference between two populations that differ; the question is whether it
reports one when they do not, and whether it declines to compute a p-value in the
one situation where a p-value would be indefensible.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism_ex.compare import ComparisonResult, cliffs_delta, compare_communities, energy_distance
from prism_ex.errors import CommunityNotFound, ConfigurationError
from prism_ex.fcs.reader import read_fcs
from prism_ex.pipeline import find_communities


@pytest.fixture(scope="module")
def result(demo_file_module, config_module):
    return find_communities(read_fcs(demo_file_module[0]), config_module)


@pytest.fixture(scope="module")
def demo_file_module(tmp_path_factory):
    from prism_ex.synth import write_demo_file

    path = tmp_path_factory.mktemp("compare") / "demo.fcs"
    return write_demo_file(path, 900, seed=7)


@pytest.fixture(scope="module")
def config_module():
    from prism_ex.pipeline import CommunityConfig
    from prism_ex.synth import CLUSTERING_MARKERS

    return CommunityConfig(markers=CLUSTERING_MARKERS, k=15, resolution=0.6, min_size=5)


def test_cliffs_delta_is_plus_one_for_disjoint_samples():
    assert cliffs_delta(np.array([4.0, 5, 6]), np.array([1.0, 2, 3])) == 1.0
    assert cliffs_delta(np.array([1.0, 2, 3]), np.array([4.0, 5, 6])) == -1.0


def test_cliffs_delta_is_zero_for_identical_samples():
    values = np.arange(10.0)
    assert cliffs_delta(values, values) == 0.0


def test_cliffs_delta_is_invariant_under_monotone_transform():
    """The reason for choosing a rank statistic: the asinh cofactor cannot move it."""
    rng = np.random.default_rng(0)
    a, b = rng.lognormal(0, 1, 200), rng.lognormal(0.5, 1, 200)
    assert cliffs_delta(a, b) == pytest.approx(
        cliffs_delta(np.arcsinh(a / 250), np.arcsinh(b / 250))
    )


def test_energy_distance_is_zero_for_the_same_cloud():
    rng = np.random.default_rng(1)
    cloud = rng.normal(size=(300, 3))
    assert energy_distance(cloud, cloud.copy(), rng=rng) < 0.05


def test_energy_distance_separates_clouds_with_a_common_centre():
    """A distance between centroids would call these two identical."""
    rng = np.random.default_rng(2)
    tight = rng.normal(0, 0.2, size=(400, 2))
    wide = rng.normal(0, 3.0, size=(400, 2))
    assert energy_distance(tight, wide, rng=rng) > 1.0


def test_comparison_ranks_the_separating_markers_first(result):
    comparison = compare_communities(result, 0, 1, n_bootstrap=50)
    ranked = comparison.ranked()
    assert abs(ranked[0].delta) >= abs(ranked[-1].delta)
    assert all(-1.0 <= entry.delta <= 1.0 for entry in comparison.per_marker)


def test_bootstrap_interval_contains_the_estimate(result):
    comparison = compare_communities(result, 0, 1, n_bootstrap=200, seed=3)
    for entry in comparison.per_marker:
        assert entry.ci_low <= entry.delta <= entry.ci_high


def test_markers_used_for_clustering_are_flagged(result):
    comparison = compare_communities(result, 0, 1, markers=["CD3", "Viability"], n_bootstrap=20)
    flags = {entry.marker: entry.used_for_clustering for entry in comparison.per_marker}
    assert flags["CD3"] is True
    assert flags["Viability"] is False


def test_no_p_values_are_offered_for_the_clustering_markers(result):
    """The refusal is the feature: see the module docstring of prism_ex.compare."""
    comparison = compare_communities(
        result, 0, 1, markers=["CD3", "CD4"], inference="split", n_bootstrap=20
    )
    assert all(entry.p_value is None for entry in comparison.per_marker)
    assert any("no null hypothesis is testable" in note for note in comparison.notes)


def test_a_marker_with_no_real_difference_is_not_significant(result):
    """Negative control: Viability is generated independently of population."""
    comparison = compare_communities(
        result,
        0,
        1,
        markers=["Viability"],
        inference="split",
        n_bootstrap=50,
        n_permutations=199,
        seed=11,
    )
    entry = comparison.per_marker[0]
    if entry.p_value is not None:  # None means the split could not be made honestly
        assert entry.p_value > 0.01
    assert entry.magnitude == "negligible"


def test_comparing_a_community_with_itself_is_refused(result):
    with pytest.raises(ConfigurationError):
        compare_communities(result, 0, 0)


def test_absent_community_is_refused(result):
    with pytest.raises(CommunityNotFound):
        compare_communities(result, 0, 999)


def test_result_serialises_and_renders(result):
    comparison = compare_communities(result, 0, 1, n_bootstrap=20)
    assert isinstance(comparison, ComparisonResult)
    payload = comparison.to_dict()
    assert payload["n_a"] > 0 and len(payload["markers"]) == len(comparison.markers)
    assert "Cliff" in comparison.to_markdown()
