"""Neighbourhood graph construction (first half of section 2.2).

Three decisions are made here and defended in the report.

**Exact k-nearest neighbours, not approximate.** Approximate neighbour search is
what makes million-event datasets tractable, and it is also non-deterministic in
most implementations. Section 3.2 asks that the same input and configuration return
the same answer, and at the scale this exercise runs at, exactness is affordable.
The seam is left open: ``metric`` and ``algorithm`` are parameters, and an
approximate backend would slot in behind the same call.

**Ties broken by index.** Two events at identical distance are ordered by row index
rather than by whatever order the search returned. Without this, determinism holds
only for data with no duplicate distances -- and integer-valued cytometry data has
plenty.

**Shared-nearest-neighbour weights.** Edges are weighted by the Jaccard overlap of
the two endpoints' neighbourhoods rather than by distance. In a space where local
density varies by orders of magnitude -- which is the normal condition for
cytometry, where one population may hold 40% of events and another 1% -- a fixed
distance means different things in different regions, while shared neighbours are
a local, scale-free statement. This is the same construction Seurat and scanpy use,
for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from prism_ex.errors import ConfigurationError
from prism_ex.provenance import Provenance
from prism_ex.subspace import Subspace

__all__ = ["NeighbourhoodGraph", "build_graph"]

WEIGHTING = ("jaccard", "uniform", "distance")


@dataclass(frozen=True, slots=True)
class NeighbourhoodGraph:
    """A symmetric weighted graph over events."""

    adjacency: sp.csr_matrix
    """``(n_events, n_events)`` upper- and lower-triangular symmetric weights."""

    k: int
    weighting: str
    metric: str
    provenance: Provenance

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of undirected edges."""
        return int(self.adjacency.nnz // 2)

    def density(self) -> float:
        """Fraction of possible undirected edges present."""
        n = self.n_nodes
        return 0.0 if n < 2 else 2.0 * self.n_edges / (n * (n - 1))


def build_graph(
    subspace: Subspace,
    *,
    k: int = 20,
    weighting: str = "jaccard",
    metric: str = "euclidean",
    prune_below: float = 1.0 / 15.0,
    algorithm: str = "brute",
) -> NeighbourhoodGraph:
    """Build a symmetric k-nearest-neighbour graph over the events of ``subspace``.

    Parameters
    ----------
    subspace:
        The marker subspace, from :func:`prism_ex.subspace.select_subspace`.
    k:
        Number of neighbours per event, excluding the event itself. Larger k
        smooths the graph and merges small populations; smaller k fragments them.
        This is the parameter the stability analysis of section 2.4 varies first.
    weighting:
        ``"jaccard"`` (shared-neighbour overlap, the default), ``"uniform"`` (all
        edges weight 1), or ``"distance"`` (Gaussian kernel on the distance,
        bandwidth set per event by its k-th neighbour).
    metric:
        Any metric accepted by scikit-learn's :class:`~sklearn.neighbors.NearestNeighbors`.
    prune_below:
        Jaccard weights at or below this are dropped. Pruning removes the long
        edges that connect a population to its neighbour by way of one shared
        outlier, which are the edges that make communities bleed into each other.
        Ignored unless ``weighting="jaccard"``.
    algorithm:
        Neighbour search backend. ``"brute"`` is exact and order-independent.

    Returns
    -------
    NeighbourhoodGraph

    Raises
    ------
    ConfigurationError
        If ``k`` is not smaller than the number of events, or an option is unknown.
    """
    if weighting not in WEIGHTING:
        raise ConfigurationError(f"weighting must be one of {WEIGHTING}, got {weighting!r}")
    n_events = subspace.n_events
    if k < 1:
        raise ConfigurationError(f"k must be at least 1, got {k}")
    if k >= n_events:
        raise ConfigurationError(
            f"k={k} needs at least {k + 1} events; the subspace has {n_events}"
        )

    distances, neighbours = _knn(subspace.matrix, k=k, metric=metric, algorithm=algorithm)

    if weighting == "jaccard":
        adjacency = _jaccard_weights(neighbours, k=k, prune_below=prune_below)
    elif weighting == "uniform":
        adjacency = _symmetrise(_binary(neighbours), how="max")
    else:
        adjacency = _distance_weights(distances, neighbours)

    adjacency.sort_indices()
    return NeighbourhoodGraph(
        adjacency=adjacency,
        k=k,
        weighting=weighting,
        metric=metric,
        provenance=subspace.provenance.derive(
            k=k, weighting=weighting, metric=metric, prune_below=prune_below
        ),
    )


def _knn(
    matrix: np.ndarray, *, k: int, metric: str, algorithm: str
) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-NN with ties broken by row index, self excluded."""
    search = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm=algorithm)
    search.fit(matrix)
    distances, indices = search.kneighbors(matrix)

    # Re-sort each row by (distance, index) so that equidistant neighbours are
    # ordered reproducibly rather than in backend order.
    order = np.lexsort((indices, distances), axis=1)
    rows = np.arange(matrix.shape[0])[:, None]
    distances = distances[rows, order]
    indices = indices[rows, order]

    # Drop self. It is normally the first column, but a duplicated row can put
    # another event at distance zero ahead of it, so it is removed by identity.
    keep = indices != rows
    counts = keep.sum(axis=1)
    # Every row keeps exactly k neighbours: rows that contained self drop it, rows
    # that did not (possible only with duplicate points) drop their last neighbour.
    trimmed_indices = np.empty((matrix.shape[0], k), dtype=np.int64)
    trimmed_distances = np.empty((matrix.shape[0], k), dtype=np.float64)
    for row in range(matrix.shape[0]):
        selected = indices[row][keep[row]] if counts[row] >= k else indices[row][1:]
        selected_d = distances[row][keep[row]] if counts[row] >= k else distances[row][1:]
        trimmed_indices[row] = selected[:k]
        trimmed_distances[row] = selected_d[:k]
    return trimmed_distances, trimmed_indices


def _binary(neighbours: np.ndarray) -> sp.csr_matrix:
    n, k = neighbours.shape
    rows = np.repeat(np.arange(n), k)
    data = np.ones(n * k, dtype=np.float64)
    return sp.csr_matrix((data, (rows, neighbours.ravel())), shape=(n, n))


def _symmetrise(matrix: sp.csr_matrix, *, how: str = "max") -> sp.csr_matrix:
    """Make a directed k-NN matrix undirected.

    ``max`` keeps an edge when either endpoint listed the other. The alternative,
    keeping only mutual pairs, isolates events in sparse regions -- which is
    precisely where the rare populations live.
    """
    transposed = matrix.T.tocsr()
    combined = matrix.maximum(transposed) if how == "max" else matrix.minimum(transposed)
    combined.setdiag(0)
    combined.eliminate_zeros()
    return combined.tocsr()


def _jaccard_weights(neighbours: np.ndarray, *, k: int, prune_below: float) -> sp.csr_matrix:
    """Weight each candidate edge by the Jaccard overlap of the two neighbourhoods."""
    binary = _binary(neighbours)
    # Include each event in its own neighbourhood: two events that are each other's
    # only neighbour should not score zero overlap.
    binary = binary + sp.identity(binary.shape[0], format="csr")
    shared = (binary @ binary.T).tocoo()

    candidate = _symmetrise(_binary(neighbours), how="max").tocoo()
    lookup = sp.csr_matrix((shared.data, (shared.row, shared.col)), shape=binary.shape)
    overlap = np.asarray(lookup[candidate.row, candidate.col]).ravel()
    union = 2.0 * (k + 1) - overlap
    weights = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)

    keep = weights > prune_below
    graph = sp.csr_matrix(
        (weights[keep], (candidate.row[keep], candidate.col[keep])), shape=binary.shape
    )
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def _distance_weights(distances: np.ndarray, neighbours: np.ndarray) -> sp.csr_matrix:
    """Gaussian kernel with a per-event bandwidth: the adaptive-sigma construction."""
    sigma = distances[:, -1:].copy()
    sigma[sigma <= 0] = 1.0
    weights = np.exp(-((distances / sigma) ** 2))
    n, k = neighbours.shape
    rows = np.repeat(np.arange(n), k)
    graph = sp.csr_matrix((weights.ravel(), (rows, neighbours.ravel())), shape=(n, n))
    return _symmetrise(graph, how="max")
