"""Shared fixtures.

Fixtures are generated in code rather than committed, so the suite has no
dependency on files present only on one machine, and each corruption test can point
at the exact byte it broke.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism_ex.fcs.writer import build_fcs_bytes
from prism_ex.pipeline import CommunityConfig
from prism_ex.synth import CLUSTERING_MARKERS, make_dataset, write_demo_file

SMALL_EVENTS = 900
"""Small enough that the whole suite runs in well under a minute, large enough that
the rare population still has ~13 events and the clustering is a real test."""


@pytest.fixture(scope="session")
def dataset():
    return make_dataset(SMALL_EVENTS, seed=7)


@pytest.fixture(scope="session")
def demo_file(tmp_path_factory):
    """A valid synthetic FCS 3.1 file, written once for the session."""
    path = tmp_path_factory.mktemp("data") / "demo.fcs"
    written, truth = write_demo_file(path, SMALL_EVENTS, seed=7)
    return written, truth


@pytest.fixture(scope="session")
def config():
    return CommunityConfig(markers=CLUSTERING_MARKERS, k=15, resolution=0.6, min_size=5)


@pytest.fixture
def tiny_matrix():
    """Four events, three channels: small enough to reason about by hand."""
    return np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ]
    )


@pytest.fixture
def tiny_bytes(tiny_matrix):
    return build_fcs_bytes(tiny_matrix, ["FSC-A", "CD3", "CD19"])
