"""The four results, chained.

The brief asks for "a single coherent piece of software rather than four unrelated
exercises in one repository". The chaining is this module: one configuration object
describes the whole path from a file to a partition, one function walks it, and the
comparison (2.3), the stability analysis (2.4) and the HTTP endpoint (2.5) all take
that same configuration rather than re-specifying its parts. When section 2.4 asks
how stable the partition of 2.2 is, "the partition of 2.2" has to be a thing that
can be re-run under perturbation with everything else held fixed -- so it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prism_ex.communities import Partition, detect_communities
from prism_ex.fcs.model import FCSFile, Keywords
from prism_ex.fcs.reader import read_fcs
from prism_ex.graph import NeighbourhoodGraph, build_graph
from prism_ex.subspace import Subspace, select_subspace

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

__all__ = [
    "CommunityConfig",
    "CommunityResult",
    "find_communities",
    "find_communities_in_file",
    "partition_subspace",
    "subset_events",
]

DEFAULT_K = 30
"""Neighbours per event. See the report: chosen on the synthetic generator, where
smaller values fragment the large populations and larger ones only cost time."""

DEFAULT_RESOLUTION = 0.6
"""Modularity resolution. Deliberately below the customary 1.0: on this graph
construction 1.0 splits homogeneous populations. This default is the single most
consequential choice in the package and the one the stability analysis exists to
qualify."""


@dataclass(frozen=True, slots=True)
class CommunityConfig:
    """Every choice that affects the partition, in one place.

    Collected into a single object so that a result can carry the configuration
    that produced it, a resampling run can vary one field and hold the rest, and
    two runs can be compared by comparing their configurations rather than by
    reading the calling code.
    """

    markers: tuple[str, ...]
    transform: str = "asinh"
    cofactor: float = 250.0
    scaling: str = "zscore"
    k: int = DEFAULT_K
    weighting: str = "jaccard"
    metric: str = "euclidean"
    prune_below: float = 1.0 / 15.0
    resolution: float = DEFAULT_RESOLUTION
    objective: str = "modularity"
    seed: int = 0
    min_size: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "markers", tuple(self.markers))

    def replace(self, **changes: Any) -> CommunityConfig:
        """Return a copy with ``changes`` applied. Used by the stability sweeps."""
        return CommunityConfig(**{**asdict(self), **changes})

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "markers": list(self.markers)}


@dataclass(frozen=True, slots=True)
class CommunityResult:
    """A partition together with everything it was built from.

    The intermediate objects are kept rather than discarded because section 2.3
    needs the events, section 2.4 needs to rebuild the graph under resampling, and
    a caller who wants to inspect the graph should not have to recompute it.
    """

    fcs: FCSFile
    subspace: Subspace
    graph: NeighbourhoodGraph
    partition: Partition
    config: CommunityConfig = field(repr=False)

    @property
    def sizes(self) -> dict[int, int]:
        """Community sizes: the result section 2.2 asks for."""
        return self.partition.sizes

    def to_dict(self) -> dict[str, Any]:
        payload = self.partition.to_dict()
        payload["config"] = self.config.to_dict()
        payload["graph"] = {
            "n_nodes": self.graph.n_nodes,
            "n_edges": self.graph.n_edges,
            "density": self.graph.density(),
        }
        return payload


def build_subspace(fcs: FCSFile, config: CommunityConfig) -> Subspace:
    """Apply the subspace half of a configuration."""
    return select_subspace(
        fcs,
        config.markers,
        transform=config.transform,
        cofactor=config.cofactor,
        scaling=config.scaling,
    )


def partition_subspace(
    subspace: Subspace, config: CommunityConfig
) -> tuple[NeighbourhoodGraph, Partition]:
    """Apply the graph and clustering half of a configuration to a subspace.

    Exposed separately because the stability analysis and the split-sample
    inference both need to re-cluster a *subset of an already-built subspace*.
    Re-selecting the subspace from a subset of events would re-estimate the
    scaling on that subset, and the resulting partitions would then differ for two
    reasons at once -- different events and different coordinates -- which is
    exactly the confound a stability analysis must not have.
    """
    graph = build_graph(
        subspace,
        k=config.k,
        weighting=config.weighting,
        metric=config.metric,
        prune_below=config.prune_below,
    )
    partition = detect_communities(
        graph,
        resolution=config.resolution,
        seed=config.seed,
        objective=config.objective,
        min_size=config.min_size,
    )
    return graph, partition


def find_communities(fcs: FCSFile, config: CommunityConfig) -> CommunityResult:
    """Run subspace selection, graph construction and community detection."""
    subspace = build_subspace(fcs, config)
    graph, partition = partition_subspace(subspace, config)
    return CommunityResult(
        fcs=fcs, subspace=subspace, graph=graph, partition=partition, config=config
    )


def subset_events(fcs: FCSFile, rows: np.ndarray) -> FCSFile:
    """Return a view of ``fcs`` containing only ``rows``, keywords carried over.

    ``$TOT`` is rewritten so the object stays self-consistent: a file that says it
    has 6000 events while holding 4800 is precisely the kind of internal
    contradiction the reader refuses to produce, and it would be odd to
    manufacture one here.
    """
    import numpy as np

    rows = np.asarray(rows, dtype=np.int64)
    events = np.array(fcs.events[rows], dtype=np.float64)
    events.setflags(write=False)
    keywords = dict(fcs.keywords)
    for key in list(keywords):
        if key.upper() == "$TOT":
            keywords[key] = str(int(rows.size))
    return FCSFile(
        keywords=Keywords(keywords),
        channels=fcs.channels,
        events=events,
        provenance=fcs.provenance.derive(subset_size=int(rows.size)),
    )


def find_communities_in_file(
    path: str | Path, markers: Sequence[str], **overrides: Any
) -> CommunityResult:
    """Read a file and find its communities: the one-call form used by the README."""
    config = CommunityConfig(markers=tuple(markers), **overrides)
    return find_communities(read_fcs(path), config)
