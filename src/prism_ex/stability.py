"""Partition stability measurements.

Four independent sources of variation are measured separately, because they have
different remedies:

1. The algorithm. Leiden is randomised; re-running on the same graph with a
   different seed should not change the result materially.
2. The sample. Repeated re-clustering of a random subsample, with each reference
   community matched back by maximum Jaccard overlap -- the construction of Hennig
   (2007). Sub-sampling is used instead of the bootstrap he proposes: a bootstrap
   duplicates events, and a duplicated event is its own nearest neighbour at
   distance zero, which perturbs the graph under measurement.
3. The parameters. A grid over k and resolution.
4. The individual event. Per-event frequency of returning to its own community,
   separating population cores from boundaries.

A single global index would report a partition as adequate while concealing which of
the four failed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from prism_ex.communities import UNASSIGNED, Partition
from prism_ex.errors import ConfigurationError
from prism_ex.pipeline import CommunityConfig, CommunityResult, partition_subspace

__all__ = [
    "CommunityStability",
    "StabilityEvidence",
    "SweepPoint",
    "assess_stability",
    "jaccard",
]

CORE_THRESHOLD = 0.8
"""An event is 'core' when it returns to its own community in >= 80% of resamples."""

STABLE_JACCARD = 0.75
"""Hennig's rule of thumb: below ~0.75 a community should not be taken at face value;
below 0.5 it is dissolved by resampling."""


def jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard index of two sets of event indices."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


@dataclass(frozen=True, slots=True)
class CommunityStability:
    """Per-community resampling evidence."""

    community: int
    size: int
    mean_jaccard: float
    p05_jaccard: float
    """5th percentile across resamples: the bad-day value, which is the honest one."""

    recovery_rate: float
    """Fraction of resamples in which the best match reached :data:`STABLE_JACCARD`."""

    core_fraction: float
    """Fraction of this community's events that are core (see :data:`CORE_THRESHOLD`)."""

    @property
    def verdict(self) -> str:
        """A word, so the table can be read without a legend."""
        if self.mean_jaccard >= STABLE_JACCARD and self.core_fraction >= 0.8:
            return "reliable"
        if self.mean_jaccard >= 0.5:
            return "qualified"
        return "unreliable"


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One point of the parameter sweep."""

    k: int
    resolution: float
    n_communities: int
    ari_vs_reference: float


@dataclass(frozen=True, slots=True)
class StabilityEvidence:
    """The complete set of stability measurements for one partition."""

    reference_sizes: dict[int, int]
    per_community: tuple[CommunityStability, ...]
    event_core_score: np.ndarray
    """Per-event fraction of resamples in which the event returned to its community."""

    seed_ari: tuple[float, ...]
    """ARI of the reference partition against re-runs that differ only in Leiden's seed."""

    resample_ari: tuple[float, ...]
    """ARI of each 80% resample against the reference, on the shared events."""

    resample_ami: tuple[float, ...]
    sweep: tuple[SweepPoint, ...]
    n_resamples: int
    subsample_fraction: float
    config: CommunityConfig

    @property
    def boundary_fraction(self) -> float:
        """Fraction of assigned events that are not core: the size of the grey zone."""
        assigned = self.event_core_score >= 0
        if not assigned.any():
            return 0.0
        return float(np.mean(self.event_core_score[assigned] < CORE_THRESHOLD))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_sizes": {str(k): v for k, v in self.reference_sizes.items()},
            "n_resamples": self.n_resamples,
            "subsample_fraction": self.subsample_fraction,
            "seed_ari": {
                "min": float(np.min(self.seed_ari)) if self.seed_ari else None,
                "values": list(self.seed_ari),
            },
            "resample_ari": {
                "mean": float(np.mean(self.resample_ari)) if self.resample_ari else None,
                "min": float(np.min(self.resample_ari)) if self.resample_ari else None,
            },
            "resample_ami": {
                "mean": float(np.mean(self.resample_ami)) if self.resample_ami else None,
            },
            "boundary_fraction": self.boundary_fraction,
            "per_community": [
                {
                    "community": c.community,
                    "size": c.size,
                    "mean_jaccard": c.mean_jaccard,
                    "p05_jaccard": c.p05_jaccard,
                    "recovery_rate": c.recovery_rate,
                    "core_fraction": c.core_fraction,
                    "verdict": c.verdict,
                }
                for c in self.per_community
            ],
            "sweep": [
                {
                    "k": s.k,
                    "resolution": s.resolution,
                    "n_communities": s.n_communities,
                    "ari_vs_reference": s.ari_vs_reference,
                }
                for s in self.sweep
            ],
            "config": self.config.to_dict(),
        }

    def to_markdown(self) -> str:
        """The tables that go into the report."""
        lines = [
            f"Stability of a {len(self.reference_sizes)}-community partition "
            f"({self.n_resamples} resamples at {self.subsample_fraction:.0%})",
            "",
            "| community | size | mean J | 5th pct J | recovery | core | verdict |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for entry in self.per_community:
            lines.append(
                f"| {entry.community} | {entry.size} | {entry.mean_jaccard:.2f} | "
                f"{entry.p05_jaccard:.2f} | {entry.recovery_rate:.0%} | "
                f"{entry.core_fraction:.0%} | {entry.verdict} |"
            )
        lines += [
            "",
            f"Seed-only ARI: min {min(self.seed_ari):.3f} over {len(self.seed_ari)} seeds."
            if self.seed_ari
            else "",
            f"Resample ARI: mean {np.mean(self.resample_ari):.3f}, "
            f"min {np.min(self.resample_ari):.3f}."
            if self.resample_ari
            else "",
            f"Boundary events (core score < {CORE_THRESHOLD:.0%}): "
            f"{self.boundary_fraction:.1%} of assigned events.",
            "",
            "| k | resolution | communities | ARI vs reference |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for point in self.sweep:
            lines.append(
                f"| {point.k} | {point.resolution:g} | {point.n_communities} | "
                f"{point.ari_vs_reference:.3f} |"
            )
        return "\n".join(lines)


def assess_stability(
    result: CommunityResult,
    *,
    n_resamples: int = 40,
    subsample_fraction: float = 0.8,
    n_seeds: int = 5,
    k_grid: Sequence[int] | None = None,
    resolution_grid: Sequence[float] | None = None,
    seed: int = 0,
) -> StabilityEvidence:
    """Gather the evidence described in the module docstring.

    Parameters
    ----------
    result:
        The reference partition, from :func:`prism_ex.pipeline.find_communities`.
    n_resamples:
        Number of sub-sampled re-clusterings. Cost is roughly this times the cost
        of the reference run; 40 is enough for a 5th percentile to mean anything.
    subsample_fraction:
        Fraction of events drawn without replacement for each resample.
    n_seeds:
        Number of alternative Leiden seeds for the algorithm-only check.
    k_grid, resolution_grid:
        Parameter sweep. Default to a grid centred on the reference configuration.
    seed:
        Seed for the resampling, independent of the clustering seed.

    Returns
    -------
    StabilityEvidence
    """
    if not 0.1 <= subsample_fraction < 1.0:
        raise ConfigurationError("subsample_fraction must lie in [0.1, 1.0)")
    if n_resamples < 2:
        raise ConfigurationError("n_resamples must be at least 2")

    config = result.config
    reference = result.partition
    n_events = result.subspace.n_events
    rng = np.random.default_rng(seed)

    reference_members = {
        community: set(reference.members(community).tolist()) for community in reference.ids
    }

    match_counts = np.zeros(n_events, dtype=np.int64)
    appearance_counts = np.zeros(n_events, dtype=np.int64)
    jaccards: dict[int, list[float]] = {community: [] for community in reference.ids}
    resample_ari: list[float] = []
    resample_ami: list[float] = []

    take = max(config.k + 1, round(subsample_fraction * n_events))
    for _ in range(n_resamples):
        rows = np.sort(rng.choice(n_events, size=take, replace=False))
        _, resampled = partition_subspace(result.subspace.subset(rows), config)

        reference_here = reference.labels[rows]
        resample_ari.append(float(adjusted_rand_score(reference_here, resampled.labels)))
        resample_ami.append(float(adjusted_mutual_info_score(reference_here, resampled.labels)))

        appearance_counts[rows] += 1
        row_set = set(rows.tolist())
        by_resampled_community = {
            community: set(rows[resampled.labels == community].tolist())
            for community in resampled.ids
        }

        for community, members in reference_members.items():
            present = members & row_set
            if not present:
                jaccards[community].append(0.0)
                continue
            best_score, best_members = 0.0, set()
            for candidate_members in by_resampled_community.values():
                score = jaccard(present, candidate_members)
                if score > best_score:
                    best_score, best_members = score, candidate_members
            jaccards[community].append(best_score)
            if best_members:
                returned = np.fromiter(present & best_members, dtype=np.int64, count=-1)
                if returned.size:
                    match_counts[returned] += 1

    core_score = np.where(
        appearance_counts > 0, match_counts / np.maximum(appearance_counts, 1), -1.0
    )
    core_score[reference.labels == UNASSIGNED] = -1.0

    per_community = tuple(
        CommunityStability(
            community=community,
            size=len(members),
            mean_jaccard=float(np.mean(jaccards[community])),
            p05_jaccard=float(np.percentile(jaccards[community], 5)),
            recovery_rate=float(np.mean(np.asarray(jaccards[community]) >= STABLE_JACCARD)),
            core_fraction=float(
                np.mean(core_score[list(members)] >= CORE_THRESHOLD) if members else 0.0
            ),
        )
        for community, members in reference_members.items()
    )

    seed_ari = tuple(
        float(
            adjusted_rand_score(
                reference.labels,
                partition_subspace(result.subspace, config.replace(seed=alt))[1].labels,
            )
        )
        for alt in range(config.seed + 1, config.seed + 1 + n_seeds)
    )

    sweep = _sweep(result, config, reference, k_grid, resolution_grid)

    return StabilityEvidence(
        reference_sizes=reference.sizes,
        per_community=per_community,
        event_core_score=core_score,
        seed_ari=seed_ari,
        resample_ari=tuple(resample_ari),
        resample_ami=tuple(resample_ami),
        sweep=sweep,
        n_resamples=n_resamples,
        subsample_fraction=subsample_fraction,
        config=config,
    )


def _sweep(
    result: CommunityResult,
    config: CommunityConfig,
    reference: Partition,
    k_grid: Sequence[int] | None,
    resolution_grid: Sequence[float] | None,
) -> tuple[SweepPoint, ...]:
    """Re-cluster across a parameter grid and compare each result to the reference."""
    k_values = list(k_grid) if k_grid is not None else _around_k(config.k)
    resolutions = (
        list(resolution_grid)
        if resolution_grid is not None
        else [round(config.resolution * factor, 3) for factor in (0.5, 0.75, 1.0, 1.5, 2.0)]
    )
    points: list[SweepPoint] = []
    for k in k_values:
        if k >= result.subspace.n_events:
            continue
        for resolution in resolutions:
            _, partition = partition_subspace(
                result.subspace, config.replace(k=k, resolution=resolution)
            )
            points.append(
                SweepPoint(
                    k=k,
                    resolution=resolution,
                    n_communities=partition.n_communities,
                    ari_vs_reference=float(adjusted_rand_score(reference.labels, partition.labels)),
                )
            )
    return tuple(points)


def _around_k(k: int) -> list[int]:
    values = sorted({max(2, k // 2), max(3, int(k * 0.75)), k, int(k * 1.5), k * 2})
    return values


def stability_of_pair(evidence: StabilityEvidence, a: int, b: int) -> str:
    """Describe whether the split between two communities survives resampling.

    A convenience for the report: the interesting failure is rarely "community 3 is
    unstable" but "3 and 5 are one population at some settings and two at others",
    and that is visible as two communities that are individually mediocre.
    """
    lookup = {entry.community: entry for entry in evidence.per_community}
    if a not in lookup or b not in lookup:
        return f"communities {a} and {b} are not both in this partition"
    first, second = lookup[a], lookup[b]
    counts = {point.n_communities for point in evidence.sweep}
    return (
        f"community {a}: mean J {first.mean_jaccard:.2f} ({first.verdict}); "
        f"community {b}: mean J {second.mean_jaccard:.2f} ({second.verdict}); "
        f"community count across the sweep ranges over {sorted(counts)}"
    )
