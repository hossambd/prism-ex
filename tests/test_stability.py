"""Section 2.4: the instrument that produces the evidence for the claim.

The deliverable of 2.4 is the claim in the report, not this module. What is tested
here is that the measurements behave the way a stability measurement must: that a
partition of obvious populations comes out reliable, that an over-split partition
does not, and that the numbers are reproducible so the claim can be re-derived.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism_ex.errors import ConfigurationError
from prism_ex.fcs.reader import read_fcs_bytes
from prism_ex.fcs.writer import build_fcs_bytes
from prism_ex.pipeline import CommunityConfig, find_communities
from prism_ex.stability import CORE_THRESHOLD, assess_stability, jaccard


@pytest.fixture(scope="module")
def obvious():
    """Three clouds no reasonable method could disagree about."""
    rng = np.random.default_rng(4)
    matrix = np.vstack([rng.normal(c, 0.3, size=(150, 2)) for c in ([0, 0], [8, 0], [0, 8])])
    fcs = read_fcs_bytes(build_fcs_bytes(matrix + 20.0, ["X", "Y"]))
    config = CommunityConfig(markers=("X", "Y"), transform="none", k=15, resolution=0.2, min_size=5)
    return find_communities(fcs, config)


def test_jaccard_edge_cases():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1, 2}, {1, 2}) == 1.0
    assert jaccard({1, 2}, {3}) == 0.0
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_obvious_populations_are_reported_reliable(obvious):
    evidence = assess_stability(
        obvious, n_resamples=8, n_seeds=3, k_grid=(15,), resolution_grid=(0.2,)
    )
    assert all(entry.verdict == "reliable" for entry in evidence.per_community)
    assert min(evidence.seed_ari) > 0.99
    assert evidence.boundary_fraction < 0.05


def test_an_over_split_partition_is_not_reported_reliable(obvious):
    """Splitting three clouds into many pieces must show up as instability."""
    over_split = find_communities(obvious.fcs, obvious.config.replace(resolution=1.5))
    evidence = assess_stability(
        over_split, n_resamples=8, n_seeds=3, k_grid=(15,), resolution_grid=(1.5,)
    )
    assert over_split.partition.n_communities > 3
    assert any(entry.verdict != "reliable" for entry in evidence.per_community)


def test_core_scores_are_probabilities_or_minus_one(obvious):
    evidence = assess_stability(
        obvious, n_resamples=6, n_seeds=2, k_grid=(15,), resolution_grid=(0.2,)
    )
    scores = evidence.event_core_score
    assert scores.shape == (obvious.subspace.n_events,)
    assert np.all((scores == -1.0) | ((scores >= 0.0) & (scores <= 1.0)))
    assert np.mean(scores >= CORE_THRESHOLD) > 0.9


def test_sweep_covers_the_requested_grid(obvious):
    evidence = assess_stability(
        obvious, n_resamples=4, n_seeds=2, k_grid=(10, 20), resolution_grid=(0.2, 0.5)
    )
    assert {(point.k, point.resolution) for point in evidence.sweep} == {
        (10, 0.2),
        (10, 0.5),
        (20, 0.2),
        (20, 0.5),
    }
    assert all(-1.0 <= point.ari_vs_reference <= 1.0 for point in evidence.sweep)


def test_evidence_is_reproducible(obvious):
    kwargs = {
        "n_resamples": 5,
        "n_seeds": 2,
        "k_grid": (15,),
        "resolution_grid": (0.2,),
        "seed": 42,
    }
    first = assess_stability(obvious, **kwargs)
    second = assess_stability(obvious, **kwargs)
    assert first.resample_ari == second.resample_ari
    np.testing.assert_array_equal(first.event_core_score, second.event_core_score)


def test_serialisation_and_rendering(obvious):
    evidence = assess_stability(
        obvious, n_resamples=4, n_seeds=2, k_grid=(15,), resolution_grid=(0.2,)
    )
    payload = evidence.to_dict()
    assert len(payload["per_community"]) == obvious.partition.n_communities
    assert "verdict" in evidence.to_markdown()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"subsample_fraction": 1.0}, "subsample_fraction"),
        ({"n_resamples": 1}, "n_resamples"),
    ],
)
def test_invalid_settings_are_refused(obvious, kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        assess_stability(obvious, **kwargs)


@pytest.mark.slow
def test_the_deliberately_ambiguous_pair_is_flagged_on_the_demo_data(demo_file, config):
    """The finding the synthetic generator was built to produce.

    Two of its six populations sit one within-population sigma apart. Whatever the
    partition does with them, the stability analysis must not report the result as
    solid while the well-separated populations are equally solid -- if it does, the
    instrument cannot tell the two situations apart and no claim can rest on it.
    """
    from prism_ex.fcs.reader import read_fcs

    result = find_communities(read_fcs(demo_file[0]), config)
    evidence = assess_stability(result, n_resamples=10, n_seeds=3)
    jaccards = sorted(entry.mean_jaccard for entry in evidence.per_community)
    assert jaccards[0] < jaccards[-1] - 0.15
