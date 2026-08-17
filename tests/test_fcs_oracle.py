"""Differential test against an independent FCS implementation.

The reader in this package is hand-written, which buys the strict rejection
semantics section 2.1 asks for and costs the confidence that comes from using
something with users. This test buys that confidence back: for files that are
unambiguously well-formed, an independent implementation must extract the same
numbers. A bug in the offset arithmetic or the byte order would show up here even
though every round-trip test in this suite would still pass, because the round-trip
tests share their assumptions with the writer.

``flowio`` is a development dependency only; the package does not import it.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism_ex.fcs.reader import read_fcs
from prism_ex.fcs.writer import write_fcs

flowio = pytest.importorskip("flowio", reason="differential oracle not installed")


@pytest.mark.parametrize("datatype", ["F", "D", "I"])
def test_independent_reader_agrees_on_the_event_matrix(tmp_path, datatype):
    rng = np.random.default_rng(11)
    matrix = rng.uniform(0, 5000, size=(200, 4))
    path = write_fcs(
        tmp_path / f"oracle-{datatype}.fcs",
        matrix,
        ["FSC-A", "SSC-A", "CD3", "CD19"],
        datatype=datatype,
    )

    ours = read_fcs(path)
    theirs = flowio.FlowData(str(path))
    reference = np.reshape(np.asarray(theirs.events, dtype=np.float64), (-1, theirs.channel_count))

    assert (theirs.event_count, theirs.channel_count) == (ours.n_events, ours.n_channels)
    np.testing.assert_allclose(ours.events, reference, rtol=1e-6)


def test_independent_reader_agrees_on_channel_names(tmp_path):
    matrix = np.zeros((10, 3))
    path = write_fcs(
        tmp_path / "names.fcs", matrix, ["FSC-A", "CD3", "CD19"], stains=[None, "BV421", None]
    )

    ours = read_fcs(path)
    theirs = flowio.FlowData(str(path))
    their_names = tuple(
        theirs.channels[index]["pnn"] for index in range(1, theirs.channel_count + 1)
    )
    assert ours.channel_names == their_names
