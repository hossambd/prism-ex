"""Section 2.2: the neighbourhood graph and the communities found in it."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from prism_ex.communities import UNASSIGNED, detect_communities
from prism_ex.errors import CommunityNotFound, ConfigurationError
from prism_ex.fcs.reader import read_fcs
from prism_ex.graph import build_graph
from prism_ex.pipeline import find_communities
from prism_ex.subspace import select_subspace


@pytest.fixture(scope="module")
def separated_subspace():
    """Three populations a human could separate with a ruler."""
    rng = np.random.default_rng(5)
    blocks = [rng.normal(centre, 0.25, size=(120, 2)) for centre in ([0, 0], [6, 0], [0, 6])]
    matrix = np.vstack(blocks)
    truth = np.repeat([0, 1, 2], 120)

    from prism_ex.fcs.reader import read_fcs_bytes
    from prism_ex.fcs.writer import build_fcs_bytes

    fcs = read_fcs_bytes(build_fcs_bytes(matrix + 10.0, ["X", "Y"]))
    return select_subspace(fcs, ["X", "Y"], transform="none", scaling="zscore"), truth


def test_graph_is_symmetric_and_has_no_self_loops(separated_subspace):
    graph = build_graph(separated_subspace[0], k=10)
    difference = graph.adjacency - graph.adjacency.T
    assert abs(difference).max() == 0
    assert graph.adjacency.diagonal().sum() == 0


def test_jaccard_weights_are_bounded(separated_subspace):
    graph = build_graph(separated_subspace[0], k=10)
    assert graph.adjacency.data.min() > 0
    assert graph.adjacency.data.max() <= 1.0


@pytest.mark.parametrize("weighting", ["jaccard", "uniform", "distance"])
def test_every_weighting_produces_a_usable_graph(separated_subspace, weighting):
    graph = build_graph(separated_subspace[0], k=10, weighting=weighting)
    assert graph.n_edges > 0
    assert sp.issparse(graph.adjacency)


def test_k_must_be_smaller_than_the_number_of_events(separated_subspace):
    with pytest.raises(ConfigurationError, match="k="):
        build_graph(separated_subspace[0], k=10_000)


def test_graph_construction_is_deterministic(separated_subspace):
    first = build_graph(separated_subspace[0], k=10)
    second = build_graph(separated_subspace[0], k=10)
    assert (first.adjacency != second.adjacency).nnz == 0


def test_duplicate_events_do_not_break_the_self_exclusion():
    """Identical rows put another event at distance zero, ahead of the event itself."""
    from prism_ex.fcs.reader import read_fcs_bytes
    from prism_ex.fcs.writer import build_fcs_bytes

    matrix = np.repeat(np.arange(1, 21, dtype=float).reshape(-1, 1), 2, axis=1)
    matrix = np.vstack([matrix, matrix])  # every event has an exact twin
    fcs = read_fcs_bytes(build_fcs_bytes(matrix, ["X", "Y"]))
    graph = build_graph(select_subspace(fcs, ["X", "Y"], transform="none"), k=5)
    assert graph.adjacency.diagonal().sum() == 0


def test_communities_recover_well_separated_populations(separated_subspace):
    subspace, truth = separated_subspace
    # Resolution 0.2 rather than the package default: modularity over-splits small
    # uniform blobs, and 360 events in three clouds is exactly that regime. The
    # behaviour is documented in the report; the point of this test is that when the
    # answer is unambiguous, some setting finds it exactly.
    partition = detect_communities(build_graph(subspace, k=15), resolution=0.2, seed=0)
    assert partition.n_communities == 3
    assert adjusted_rand_score(truth, partition.labels) > 0.99


def test_communities_are_labelled_in_descending_size(demo_file, config):
    result = find_communities(read_fcs(demo_file[0]), config)
    sizes = [size for community, size in result.sizes.items() if community != UNASSIGNED]
    assert sizes == sorted(sizes, reverse=True)


def test_min_size_moves_small_communities_to_unassigned(separated_subspace):
    subspace, _ = separated_subspace
    graph = build_graph(subspace, k=15)
    partition = detect_communities(graph, resolution=0.2, seed=0, min_size=200)
    assert partition.n_communities == 0
    assert partition.n_unassigned == subspace.n_events


def test_members_of_an_absent_community_is_an_error(separated_subspace):
    partition = detect_communities(build_graph(separated_subspace[0], k=15), seed=0)
    with pytest.raises(CommunityNotFound):
        partition.members(99)


def test_sizes_sum_to_the_number_of_events(demo_file, config):
    result = find_communities(read_fcs(demo_file[0]), config)
    assert sum(result.sizes.values()) == result.fcs.n_events


@pytest.mark.parametrize("resolution", [0.2, 0.6, 1.2])
def test_higher_resolution_never_finds_fewer_communities(separated_subspace, resolution):
    """The resolution parameter has to behave like a dial, or nothing downstream reads."""
    subspace, _ = separated_subspace
    graph = build_graph(subspace, k=15)
    coarse = detect_communities(graph, resolution=0.2, seed=0).n_communities
    finer = detect_communities(graph, resolution=resolution, seed=0).n_communities
    assert finer >= coarse
