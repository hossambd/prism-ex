"""Community detection (second half of section 2.2).

Leiden is used rather than Louvain. The reason is not fashion: Louvain can return
communities that are internally disconnected -- a node can be left in a community
none of whose members it is adjacent to -- and a disconnected community is not a
thing one can then make a claim about in section 2.4. Leiden's guarantee that all
communities are internally connected is exactly the property that makes the
stability question well posed.

RBConfiguration (modularity with a resolution parameter) is the default quality
function. Modularity has a known resolution limit -- it cannot see communities
below a size set by the total edge weight -- which is why ``resolution`` is exposed
rather than fixed, and why the rare population in the synthetic data is a fair test
rather than a trick.

Determinism: Leiden is a randomised algorithm, so the seed is a parameter with a
default rather than an implicit clock read. Communities are relabelled by
descending size before returning, so that "community 0" means the largest one in
every run and across runs -- without that, two identical partitions can disagree on
every label, and every downstream comparison becomes a puzzle about labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import leidenalg
import numpy as np

from prism_ex.errors import ConfigurationError
from prism_ex.graph import NeighbourhoodGraph
from prism_ex.provenance import Provenance

__all__ = ["Partition", "detect_communities"]

UNASSIGNED = -1
"""Label given to events in communities smaller than ``min_size``."""

OBJECTIVES = ("modularity", "cpm")


@dataclass(frozen=True, slots=True)
class Partition:
    """A community assignment over the events of a graph."""

    labels: np.ndarray
    """``(n_events,)`` int32. ``-1`` marks events below the size threshold."""

    quality: float
    """Value of the quality function achieved (higher is better)."""

    resolution: float
    seed: int
    objective: str
    provenance: Provenance
    _sizes: dict[int, int] = field(default_factory=dict, repr=False)

    @property
    def sizes(self) -> dict[int, int]:
        """Community id -> number of events, largest first. This is what 2.2 reports."""
        return dict(self._sizes)

    @property
    def ids(self) -> tuple[int, ...]:
        """Community ids in descending size order, excluding :data:`UNASSIGNED`."""
        return tuple(cid for cid in self._sizes if cid != UNASSIGNED)

    @property
    def n_communities(self) -> int:
        return len(self.ids)

    @property
    def n_unassigned(self) -> int:
        return int(self._sizes.get(UNASSIGNED, 0))

    def members(self, community: int) -> np.ndarray:
        """Row indices of the events in ``community``."""
        from prism_ex.errors import CommunityNotFound

        if community not in self._sizes:
            raise CommunityNotFound(
                f"community {community} is not in this partition; ids are {list(self.ids)}"
            )
        return np.flatnonzero(self.labels == community)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary: sizes plus how they were produced."""
        return {
            "n_events": int(self.labels.size),
            "n_communities": self.n_communities,
            "sizes": {str(cid): size for cid, size in self._sizes.items()},
            "quality": self.quality,
            "objective": self.objective,
            "resolution": self.resolution,
            "seed": self.seed,
            "provenance": self.provenance.to_dict(),
        }


def detect_communities(
    graph: NeighbourhoodGraph,
    *,
    resolution: float = 1.0,
    seed: int = 0,
    objective: str = "modularity",
    n_iterations: int = 5,
    min_size: int = 10,
) -> Partition:
    """Find communities in a neighbourhood graph and report their sizes.

    Parameters
    ----------
    graph:
        From :func:`prism_ex.graph.build_graph`.
    resolution:
        Higher values split more. Under ``"modularity"`` this scales the null model;
        under ``"cpm"`` it is directly the minimum within-community edge density,
        which makes it comparable across graphs of different sizes.
    seed:
        Leiden's random seed. Fixed by default: see module docstring.
    objective:
        ``"modularity"`` (RBConfiguration) or ``"cpm"`` (constant Potts model).
    n_iterations:
        Leiden iterations. The algorithm is run to convergence in practice by 5;
        more costs time and changes nothing.
    min_size:
        Communities smaller than this are marked :data:`UNASSIGNED` rather than
        reported. Every graph of this kind produces a tail of two- and three-node
        communities, and reporting them as findings would be dishonest. Set to 1 to
        keep everything.

    Returns
    -------
    Partition
        Labels and sizes, with the configuration that produced them attached.
    """
    if objective not in OBJECTIVES:
        raise ConfigurationError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if resolution <= 0:
        raise ConfigurationError(f"resolution must be positive, got {resolution}")
    if min_size < 1:
        raise ConfigurationError(f"min_size must be at least 1, got {min_size}")

    igraph_graph, weights = _to_igraph(graph)
    partition_class = (
        leidenalg.RBConfigurationVertexPartition
        if objective == "modularity"
        else leidenalg.CPMVertexPartition
    )
    found = leidenalg.find_partition(
        igraph_graph,
        partition_class,
        weights=weights,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )

    labels = _relabel_by_size(np.asarray(found.membership, dtype=np.int32), min_size)
    sizes = _size_table(labels)

    return Partition(
        labels=labels,
        quality=float(found.quality()),
        resolution=resolution,
        seed=seed,
        objective=objective,
        provenance=graph.provenance.derive(
            resolution=resolution, seed=seed, objective=objective, min_size=min_size
        ),
        _sizes=sizes,
    )


def _to_igraph(graph: NeighbourhoodGraph) -> tuple[ig.Graph, list[float]]:
    """Convert the upper triangle of the adjacency to an undirected igraph."""
    upper = graph.adjacency.tocoo()
    mask = upper.row < upper.col
    edges = list(zip(upper.row[mask].tolist(), upper.col[mask].tolist(), strict=True))
    weights = upper.data[mask].astype(float).tolist()
    converted = ig.Graph(n=graph.n_nodes, edges=edges, directed=False)
    return converted, weights


def _relabel_by_size(membership: np.ndarray, min_size: int) -> np.ndarray:
    """Relabel communities 0, 1, 2, ... in descending size, with ties by first event.

    Ties are broken by the index of the first member so that two communities of
    equal size still get a reproducible order.
    """
    unique, counts = np.unique(membership, return_counts=True)
    first_seen = {int(value): int(np.argmax(membership == value)) for value in unique}
    order = sorted(
        zip(unique.tolist(), counts.tolist(), strict=True),
        key=lambda pair: (-pair[1], first_seen[pair[0]]),
    )

    mapping: dict[int, int] = {}
    next_label = 0
    for original, count in order:
        if count < min_size:
            mapping[original] = UNASSIGNED
        else:
            mapping[original] = next_label
            next_label += 1

    relabelled = np.empty_like(membership)
    for original, new in mapping.items():
        relabelled[membership == original] = new
    return relabelled.astype(np.int32)


def _size_table(labels: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    table = {int(value): int(count) for value, count in zip(unique, counts, strict=True)}
    ordered = {cid: table[cid] for cid in sorted(c for c in table if c != UNASSIGNED)}
    if UNASSIGNED in table:
        ordered[UNASSIGNED] = table[UNASSIGNED]
    return ordered
