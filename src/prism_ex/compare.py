"""Quantitative comparison of two communities.

Three choices govern what is computed.

The effect size is Cliff's delta: the probability that an event from A exceeds one
from B, rescaled to [-1, 1]. Fluorescence is heavy-tailed and frequently bimodal, so
a difference of means is a poor summary. Being rank-based, the statistic is
invariant under the transform cofactor, which is an analysis parameter that appears
in no result table.

Distributions are compared rather than centroids. Medians and MADs are reported for
readability but nothing is computed from them; the multivariate energy distance is
reported alongside the per-marker table because it is zero only when the two
distributions coincide.

P-values are computed only where they can be valid. The communities were derived
from the event matrix, so testing them on the same events is circular. Splitting the
data makes the community definitions independent of the test set, but held-out
events must still be assigned, and the assignment uses their marker values: for a
marker inside the clustering subspace the assignment rule guarantees a difference.
P-values are therefore restricted to markers outside the clustering subspace, and
the restriction is stated in the output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from prism_ex.communities import UNASSIGNED, Partition
from prism_ex.errors import CommunityNotFound, ConfigurationError, InsufficientData
from prism_ex.pipeline import (
    CommunityConfig,
    CommunityResult,
    build_subspace,
    partition_subspace,
)
from prism_ex.provenance import Provenance
from prism_ex.subspace import select_subspace

__all__ = ["ComparisonResult", "MarkerComparison", "compare_communities"]

INFERENCE_MODES = ("descriptive", "split")


@dataclass(frozen=True, slots=True)
class MarkerComparison:
    """The comparison of two communities on one marker."""

    marker: str
    median_a: float
    median_b: float
    mad_a: float
    mad_b: float
    delta: float
    """Cliff's delta in [-1, 1]. Positive means community A tends to be higher."""

    ci_low: float
    ci_high: float
    """Percentile bootstrap interval for :attr:`delta`, over events."""

    used_for_clustering: bool
    """True when this marker was in the subspace that produced the communities."""

    p_value: float | None = None
    """Split-sample permutation p-value; ``None`` when it cannot honestly be one."""

    q_value: float | None = None
    """Benjamini-Hochberg adjusted :attr:`p_value` across the tested markers."""

    @property
    def magnitude(self) -> str:
        """Conventional reading of |delta|: negligible / small / medium / large.

        Thresholds from Romano et al. (2006). Reported as words because a number
        without a scale invites the reader to invent one.
        """
        size = abs(self.delta)
        if size < 0.147:
            return "negligible"
        if size < 0.33:
            return "small"
        if size < 0.474:
            return "medium"
        return "large"


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The comparison of two communities across a marker subspace."""

    community_a: int
    community_b: int
    n_a: int
    n_b: int
    markers: tuple[str, ...]
    per_marker: tuple[MarkerComparison, ...]
    energy_distance: float
    """Joint distance between the two clouds; 0 iff the distributions coincide."""

    inference: str
    notes: tuple[str, ...]
    provenance: Provenance

    def ranked(self) -> tuple[MarkerComparison, ...]:
        """Per-marker results sorted by descending |delta|: the separating markers first."""
        return tuple(sorted(self.per_marker, key=lambda m: -abs(m.delta)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "community_a": self.community_a,
            "community_b": self.community_b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "energy_distance": self.energy_distance,
            "inference": self.inference,
            "markers": [
                {
                    "marker": m.marker,
                    "median_a": m.median_a,
                    "median_b": m.median_b,
                    "mad_a": m.mad_a,
                    "mad_b": m.mad_b,
                    "cliffs_delta": m.delta,
                    "ci_low": m.ci_low,
                    "ci_high": m.ci_high,
                    "magnitude": m.magnitude,
                    "used_for_clustering": m.used_for_clustering,
                    "p_value": m.p_value,
                    "q_value": m.q_value,
                }
                for m in self.ranked()
            ],
            "notes": list(self.notes),
            "provenance": self.provenance.to_dict(),
        }

    def to_markdown(self) -> str:
        """A table a human can read, which is what the report needs."""
        head = (
            f"Community {self.community_a} (n={self.n_a}) vs "
            f"community {self.community_b} (n={self.n_b}) - "
            f"energy distance {self.energy_distance:.3f}\n\n"
            "| marker | median A | median B | Cliff's d | 95% CI | magnitude | q |\n"
            "| --- | ---: | ---: | ---: | :---: | --- | ---: |\n"
        )
        rows = []
        for m in self.ranked():
            q = "n/a" if m.q_value is None else f"{m.q_value:.3g}"
            star = "*" if m.used_for_clustering else ""
            rows.append(
                f"| {m.marker}{star} | {m.median_a:.2f} | {m.median_b:.2f} | "
                f"{m.delta:+.3f} | [{m.ci_low:+.2f}, {m.ci_high:+.2f}] | "
                f"{m.magnitude} | {q} |"
            )
        tail = "\n\n" + "\n".join(f"- {note}" for note in self.notes) if self.notes else ""
        return head + "\n".join(rows) + tail


def compare_communities(
    result: CommunityResult,
    community_a: int,
    community_b: int,
    *,
    markers: Sequence[str] | None = None,
    inference: str = "descriptive",
    n_bootstrap: int = 500,
    n_permutations: int = 999,
    seed: int = 0,
    max_pairs: int = 2000,
) -> ComparisonResult:
    """Compare the marker profiles of two communities.

    Parameters
    ----------
    result:
        The output of :func:`prism_ex.pipeline.find_communities`.
    community_a, community_b:
        Community ids from ``result.partition``.
    markers:
        The comparison subspace. Defaults to the clustering subspace; passing
        markers that were *not* used for clustering is the more informative
        comparison, and is the only case in which p-values are available.
    inference:
        ``"descriptive"`` (default): effect sizes and bootstrap intervals only.
        ``"split"``: additionally re-derives the communities on half the events and
        tests the other half, for markers outside the clustering subspace.
    n_bootstrap:
        Bootstrap resamples for the confidence intervals.
    n_permutations:
        Label permutations for the split-sample test.
    seed:
        Seed for both resampling procedures.
    max_pairs:
        Cap on the events per community used for the energy distance, which is
        quadratic. Sampling is seeded, so the result is still reproducible.

    Returns
    -------
    ComparisonResult
    """
    if inference not in INFERENCE_MODES:
        raise ConfigurationError(f"inference must be one of {INFERENCE_MODES}")
    partition = result.partition
    for community in (community_a, community_b):
        if community not in partition.sizes or community == UNASSIGNED:
            raise CommunityNotFound(
                f"community {community} is not in this partition; ids are {list(partition.ids)}"
            )
    if community_a == community_b:
        raise ConfigurationError("a community cannot be compared with itself")

    marker_names = tuple(markers) if markers is not None else result.config.markers
    comparison_space = select_subspace(
        result.fcs,
        marker_names,
        transform=result.config.transform,
        cofactor=result.config.cofactor,
        scaling="none",  # medians should be readable on the transformed scale
    )
    rows_a = partition.members(community_a)
    rows_b = partition.members(community_b)
    if min(rows_a.size, rows_b.size) < 3:
        raise InsufficientData(
            f"communities of size {rows_a.size} and {rows_b.size} are too small to compare"
        )

    matrix = comparison_space.matrix
    rng = np.random.default_rng(seed)
    clustering_markers = {m.casefold() for m in result.config.markers}

    p_values: dict[str, float] = {}
    split_reason = ""
    if inference == "split":
        p_values, split_reason = _split_p_values(
            result,
            community_a,
            community_b,
            comparison_space.markers,
            clustering_markers,
            n_permutations=n_permutations,
            seed=seed,
        )

    per_marker: list[MarkerComparison] = []
    for column, marker in enumerate(comparison_space.markers):
        values_a = matrix[rows_a, column]
        values_b = matrix[rows_b, column]
        delta = cliffs_delta(values_a, values_b)
        low, high = _bootstrap_delta_ci(values_a, values_b, n_bootstrap, rng)
        per_marker.append(
            MarkerComparison(
                marker=marker,
                median_a=float(np.median(values_a)),
                median_b=float(np.median(values_b)),
                mad_a=float(stats.median_abs_deviation(values_a)),
                mad_b=float(stats.median_abs_deviation(values_b)),
                delta=delta,
                ci_low=low,
                ci_high=high,
                used_for_clustering=marker.casefold() in clustering_markers,
                p_value=p_values.get(marker),
            )
        )

    per_marker = _add_q_values(per_marker)
    energy = energy_distance(matrix[rows_a], matrix[rows_b], rng=rng, max_pairs=max_pairs)

    notes = [
        "Cliff's delta is the probability that an event from A exceeds one from B, "
        "rescaled to [-1, 1]; the interval is a percentile bootstrap over events.",
        "Markers marked * were used to define the communities: for these the "
        "comparison is descriptive by construction and no p-value is meaningful.",
    ]
    if inference == "split":
        tested = [m.marker for m in per_marker if m.p_value is not None]
        if tested:
            notes.append(
                "Split-sample inference: communities re-derived on a random half, "
                f"held-out events assigned by {result.config.k}-NN vote, tested on "
                f"the other half. Markers tested: {', '.join(tested)}."
            )
        else:
            notes.append(f"Split-sample inference produced no p-values: {split_reason}")
    else:
        notes.append(
            "No p-values requested. The events were clustered before being "
            "compared, so a test on the same events is circular; pass "
            "inference='split' with markers outside the clustering subspace."
        )
    if not any(not m.used_for_clustering for m in per_marker):
        notes.append(
            "Every marker compared was also used for clustering. Consider comparing "
            "on markers held out of the subspace, where the difference is a finding "
            "rather than a restatement of the clustering."
        )

    return ComparisonResult(
        community_a=community_a,
        community_b=community_b,
        n_a=int(rows_a.size),
        n_b=int(rows_b.size),
        markers=comparison_space.markers,
        per_marker=tuple(per_marker),
        energy_distance=energy,
        inference=inference,
        notes=tuple(notes),
        provenance=partition.provenance.derive(
            comparison_markers=list(comparison_space.markers),
            inference=inference,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            compare_seed=seed,
        ),
    )


# ------------------------------------------------------------------- statistics


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: ``P(a > b) - P(a < b)``, computed from ranks in O(n log n)."""
    n_a, n_b = a.size, b.size
    if n_a == 0 or n_b == 0:
        raise InsufficientData("Cliff's delta needs a non-empty sample on both sides")
    ranks = stats.rankdata(np.concatenate([a, b]))
    rank_sum_a = float(ranks[:n_a].sum())
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    return float(2.0 * u_a / (n_a * n_b) - 1.0)


def _bootstrap_delta_ci(
    a: np.ndarray, b: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float]:
    if n_bootstrap <= 0:
        return (float("nan"), float("nan"))
    deltas = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample_a = a[rng.integers(0, a.size, a.size)]
        sample_b = b[rng.integers(0, b.size, b.size)]
        deltas[index] = cliffs_delta(sample_a, sample_b)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def energy_distance(
    a: np.ndarray, b: np.ndarray, *, rng: np.random.Generator, max_pairs: int = 2000
) -> float:
    """Energy distance between two point clouds.

    ``2 E|A-B| - E|A-A'| - E|B-B'|``: zero if and only if the two distributions are
    identical, so unlike a distance between centroids it does not report two
    differently-shaped clouds with a common centre as the same. Quadratic in the
    sample size, hence the cap.
    """
    sample_a = _subsample(a, max_pairs, rng)
    sample_b = _subsample(b, max_pairs, rng)
    from scipy.spatial.distance import cdist, pdist

    between = float(cdist(sample_a, sample_b).mean())
    within_a = float(pdist(sample_a).mean()) if sample_a.shape[0] > 1 else 0.0
    within_b = float(pdist(sample_b).mean()) if sample_b.shape[0] > 1 else 0.0
    return max(0.0, 2 * between - within_a - within_b)


def _subsample(matrix: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if matrix.shape[0] <= cap:
        return matrix
    return matrix[rng.choice(matrix.shape[0], cap, replace=False)]


def _add_q_values(entries: list[MarkerComparison]) -> list[MarkerComparison]:
    """Benjamini-Hochberg across the markers that carry a p-value."""
    tested = [entry for entry in entries if entry.p_value is not None]
    if not tested:
        return entries
    order = sorted(range(len(tested)), key=lambda i: tested[i].p_value or 1.0)
    m = len(tested)
    running = 1.0
    q_by_marker: dict[str, float] = {}
    for rank, position in enumerate(reversed(order), start=1):
        entry = tested[position]
        raw = (entry.p_value or 1.0) * m / (m - rank + 1)
        running = min(running, raw)
        q_by_marker[entry.marker] = min(1.0, running)
    return [
        MarkerComparison(
            **{
                **{field: getattr(entry, field) for field in entry.__slots__},
                "q_value": q_by_marker.get(entry.marker),
            }
        )
        if entry.p_value is not None
        else entry
        for entry in entries
    ]


# --------------------------------------------------------- split-sample testing


def _split_p_values(
    result: CommunityResult,
    community_a: int,
    community_b: int,
    markers: tuple[str, ...],
    clustering_markers: set[str],
    *,
    n_permutations: int,
    seed: int,
    min_match: float = 0.5,
) -> tuple[dict[str, float], str]:
    """Permutation p-values on held-out events, for markers outside the subspace.

    Returns an empty mapping and a reason -- rather than raising -- when the
    communities cannot be recovered on the training half. "This community is not
    stable enough to test" is itself the result; a p-value computed against a
    community that half the data does not contain would be worse than none.
    """
    testable = [m for m in markers if m.casefold() not in clustering_markers]
    if not testable:
        return {}, (
            "every marker compared was used for clustering, so no null hypothesis is testable"
        )

    config: CommunityConfig = result.config
    n_events = result.fcs.n_events
    rng = np.random.default_rng(seed + 977)
    shuffled = rng.permutation(n_events)
    train_rows = np.sort(shuffled[: n_events // 2])
    test_rows = np.sort(shuffled[n_events // 2 :])

    train_space = result.subspace.subset(train_rows)
    if train_space.n_events <= config.k:
        return {}, "too few events to cluster half of them"
    _, train_partition = partition_subspace(train_space, config)

    matched = {
        community: _best_match(result.partition, train_partition, train_rows, community)
        for community in (community_a, community_b)
    }
    weak = [
        f"community {community} (best Jaccard {0.0 if match is None else match[1]:.2f})"
        for community, match in matched.items()
        if match is None or match[1] < min_match
    ]
    if weak:
        return {}, (
            "the communities could not be recovered from half the events: "
            + ", ".join(weak)
            + f"; a match of at least {min_match:.2f} is required"
        )

    predicted = _assign_by_vote(
        reference=result.subspace.matrix,
        train_rows=train_rows,
        train_labels=train_partition.labels,
        test_rows=test_rows,
        k=config.k,
    )
    group_a = test_rows[predicted == matched[community_a][0]]
    group_b = test_rows[predicted == matched[community_b][0]]
    if min(group_a.size, group_b.size) < 20:
        return {}, (
            f"only {group_a.size} and {group_b.size} held-out events were assigned "
            "to the two communities"
        )

    comparison_space = build_subspace(result.fcs, config.replace(markers=tuple(testable)))
    p_values: dict[str, float] = {}
    permutation_rng = np.random.default_rng(seed + 4159)
    for column, marker in enumerate(comparison_space.markers):
        values_a = comparison_space.matrix[group_a, column]
        values_b = comparison_space.matrix[group_b, column]
        p_values[marker] = _permutation_p(values_a, values_b, n_permutations, permutation_rng)
    return p_values, (f"tested on {group_a.size} and {group_b.size} held-out events")


def _best_match(
    reference: Partition, candidate: Partition, rows: np.ndarray, community: int
) -> tuple[int, float] | None:
    """Return the candidate community with the highest Jaccard against ``community``."""
    reference_members = set(np.flatnonzero(reference.labels[rows] == community).tolist())
    if not reference_members:
        return None
    best: tuple[int, float] | None = None
    for other in candidate.ids:
        other_members = set(np.flatnonzero(candidate.labels == other).tolist())
        union = len(reference_members | other_members)
        score = len(reference_members & other_members) / union if union else 0.0
        if best is None or score > best[1]:
            best = (other, score)
    return best


def _assign_by_vote(
    *,
    reference: np.ndarray,
    train_rows: np.ndarray,
    train_labels: np.ndarray,
    test_rows: np.ndarray,
    k: int,
) -> np.ndarray:
    """Assign held-out events by majority vote of their k nearest training events."""
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(n_neighbors=min(k, train_rows.size), algorithm="brute")
    classifier.fit(reference[train_rows], train_labels)
    return classifier.predict(reference[test_rows])


def _permutation_p(
    a: np.ndarray, b: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> float:
    """Two-sided permutation p-value for Cliff's delta, with the +1 correction."""
    observed = abs(cliffs_delta(a, b))
    pooled = np.concatenate([a, b])
    n_a = a.size
    exceed = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        if abs(cliffs_delta(shuffled[:n_a], shuffled[n_a:])) >= observed:
            exceed += 1
    return (exceed + 1) / (n_permutations + 1)
