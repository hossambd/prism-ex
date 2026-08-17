"""Section 3.2: same input, same configuration, same package version, same answer."""

from __future__ import annotations

import numpy as np

from prism_ex.fcs.reader import read_fcs
from prism_ex.pipeline import find_communities


def test_repeated_runs_give_identical_labels(demo_file, config):
    fcs = read_fcs(demo_file[0])
    first = find_communities(fcs, config)
    second = find_communities(fcs, config)
    np.testing.assert_array_equal(first.partition.labels, second.partition.labels)
    assert first.partition.quality == second.partition.quality


def test_reading_the_file_twice_gives_identical_events(demo_file):
    path = demo_file[0]
    np.testing.assert_array_equal(read_fcs(path).events, read_fcs(path).events)


def test_configuration_is_recorded_in_the_provenance(demo_file, config):
    result = find_communities(read_fcs(demo_file[0]), config)
    recorded = result.partition.provenance.config
    assert recorded["k"] == config.k
    assert recorded["resolution"] == config.resolution
    assert recorded["markers"] == list(config.markers)
    assert result.partition.provenance.source_sha256 is not None


def test_configuration_id_is_order_independent(demo_file, config):
    """Two spellings of the same configuration must not look like two experiments."""
    from prism_ex.provenance import config_hash

    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_a_different_seed_is_visible_in_the_provenance(demo_file, config):
    fcs = read_fcs(demo_file[0])
    first = find_communities(fcs, config)
    second = find_communities(fcs, config.replace(seed=99))
    assert first.partition.provenance.config_id != second.partition.provenance.config_id
