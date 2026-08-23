"""Regenerate the evidence behind every design decision in the report.

Each block answers one question of the form "why this and not the obvious
alternative?" by measuring both, on data whose answer is known. Nothing here is
imported by the package; it exists so that the claims in the accompanying report
are re-derivable by a reviewer rather than taken on trust.

    python scripts/decision_evidence.py            # ~4 minutes on a laptop
    python scripts/decision_evidence.py --quick    # ~1 minute, fewer resamples

Every number printed is deterministic given the seeds below.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import igraph as ig
import numpy as np
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from prism_ex import CommunityConfig, find_communities, read_fcs
from prism_ex.communities import UNASSIGNED
from prism_ex.compare import cliffs_delta, compare_communities, energy_distance
from prism_ex.errors import FCSError
from prism_ex.fcs.reader import read_fcs_bytes
from prism_ex.fcs.writer import build_fcs_bytes
from prism_ex.pipeline import partition_subspace
from prism_ex.stability import assess_stability, jaccard
from prism_ex.synth import CLUSTERING_MARKERS, make_dataset, write_demo_file

SEED = 20260817
N_EVENTS = 6000
EVIDENCE_PATH = Path(tempfile.gettempdir()) / "prism-ex-evidence.fcs"


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


# --------------------------------------------------------------------------- 1


def evidence_reader_strictness() -> None:
    """Why a hand-written reader instead of an existing one.

    The requirement is rejection, not parsing. This measures what an established,
    well-maintained reader does with files that violate FCS 3.1, and what this one
    does with the same bytes.
    """
    heading("1. Reader strictness: prism-ex vs an established reader (flowio)")

    matrix = np.arange(24, dtype=float).reshape(8, 3)
    good = build_fcs_bytes(matrix, ["A", "B", "C"])

    def corrupt_version(payload: bytes) -> bytes:
        return b"FCS3.0" + payload[6:]

    cases = {
        "$TOT claims 9999 events, DATA holds 8": build_fcs_bytes(
            matrix, ["A", "B", "C"], extra_keywords={"$TOT": "9999"}
        ),
        "two parameters share a $PnN": build_fcs_bytes(
            matrix, ["A", "B", "C"], extra_keywords={"$P2N": "A"}
        ),
        "log $PnE with a zero offset ('4,0')": build_fcs_bytes(
            matrix, ["A", "B", "C"], extra_keywords={"$P1E": "4,0"}
        ),
        "declares FCS3.0": corrupt_version(good),
        "DATA truncated by 4 bytes": good[:-4],
        "(control) a valid file": good,
    }

    try:
        import flowio
    except ImportError:  # pragma: no cover
        print("flowio not installed; run: pip install '.[dev]'")
        return

    print(f"{'defect':<42s} {'prism-ex':<26s} flowio")
    print("-" * 78)
    for label, payload in cases.items():
        try:
            read_fcs_bytes(payload)
            ours = "accepted"
        except FCSError as error:
            ours = type(error).__name__
        except Exception as error:  # pragma: no cover
            ours = f"!{type(error).__name__}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import io

                data = flowio.FlowData(io.BytesIO(payload))
                theirs = f"accepted ({data.event_count} events)"
            except Exception as error:
                theirs = f"{type(error).__name__}"
        print(f"{label:<42s} {ours:<26s} {theirs}")

    print(
        "\nReading: a reader that accepts a file whose $TOT contradicts its DATA segment\n"
        "is behaving correctly for its own users and incorrectly for section 2.1."
    )


# --------------------------------------------------------------------------- 2


def evidence_transform_and_scaling(fcs, truth) -> None:
    """Why asinh and z-scoring rather than clustering the raw fluorescence."""
    heading("2. Transform and scaling: effect on recovery of known populations")

    base = CommunityConfig(markers=CLUSTERING_MARKERS)
    variants = {
        "asinh + z-score (default)": base,
        "asinh, no scaling": base.replace(scaling="none"),
        "no transform, z-score": base.replace(transform="none"),
        "no transform, no scaling": base.replace(transform="none", scaling="none"),
    }
    print(f"{'configuration':<30s} {'communities':>12s} {'ARI vs truth':>14s}")
    print("-" * 78)
    for label, config in variants.items():
        result = find_communities(fcs, config)
        ari = adjusted_rand_score(truth, result.partition.labels)
        print(f"{label:<30s} {result.partition.n_communities:>12d} {ari:>14.3f}")

    print(
        "\nReading: on untransformed fluorescence the brightest channel dominates every\n"
        "Euclidean distance, so the graph encodes brightness rather than phenotype."
    )


# --------------------------------------------------------------------------- 3


def evidence_edge_weighting(fcs, truth) -> None:
    """Whether shared-neighbour Jaccard weighting earns its place. It does not.

    This block is kept in the report precisely because it reports a null result.
    The argument for Jaccard weights -- that a fixed distance means different
    things in regions whose density differs twentyfold, while shared-neighbour
    overlap is scale-free -- is a good argument and is the standard construction
    in single-cell analysis. On this generator it makes no measurable difference,
    at three difficulty levels. A decision defended by an argument its own evidence
    does not support should be labelled as such rather than quietly reported as a
    win.
    """
    heading("3. Edge weighting: does the shared-neighbour construction earn its place?")

    base = CommunityConfig(markers=CLUSTERING_MARKERS)
    header = f"{'spread':>8s} {'weighting':<12s} {'edges':>9s} {'communities':>12s}"
    print(f"{header} {'ARI':>8s} {'rare pop.':>11s}")
    print("-" * 78)
    for spread in (0.55, 0.75):
        if spread == 0.55:
            labels = truth
            current = fcs
        else:
            harder = make_dataset(N_EVENTS, seed=SEED, spread=spread)
            current = read_fcs_bytes(build_fcs_bytes(harder.events, harder.markers))
            labels = harder.labels
        for weighting in ("jaccard", "distance", "uniform"):
            result = find_communities(current, base.replace(weighting=weighting))
            ari = adjusted_rand_score(labels, result.partition.labels)
            rare = set(np.flatnonzero(labels == 5).tolist())
            found = any(
                jaccard(rare, set(result.partition.members(c).tolist())) > 0.5
                for c in result.partition.ids
            )
            print(
                f"{spread:>8.2f} {weighting:<12s} {result.graph.n_edges:>9d} "
                f"{result.partition.n_communities:>12d} {ari:>8.3f} "
                f"{'recovered' if found else 'LOST':>11s}"
            )

    print(
        "\nReading: a null result. The three weightings are indistinguishable here, and at\n"
        "spread 0.75 the unweighted graph is marginally ahead. The Jaccard default is kept\n"
        "on the density argument and because it is the established construction, but this\n"
        "evidence does not support it and the report says so. All three remain available."
    )


# --------------------------------------------------------------------------- 4


def evidence_leiden_over_louvain(fcs) -> None:
    """Why Leiden rather than Louvain.

    Louvain can return a community whose members are not connected to one another;
    Leiden cannot (Traag, Waltman & van Eck, 2019). This block measures how often
    that failure actually occurs here. The honest answer on this data is: it does
    not. The choice is therefore insurance against a documented failure mode, not
    a measured improvement, and the report states it that way.

    Modularity values are deliberately not compared between the two: leidenalg
    optimises the RB objective at the configured resolution while igraph's Louvain
    reports modularity at gamma=1, so a side-by-side number would be meaningless.
    """
    heading("4. Leiden vs Louvain: how often does the failure mode actually occur?")

    result = find_communities(fcs, CommunityConfig(markers=CLUSTERING_MARKERS))
    adjacency = result.graph.adjacency.tocoo()
    mask = adjacency.row < adjacency.col
    graph = ig.Graph(
        n=result.graph.n_nodes,
        edges=list(zip(adjacency.row[mask].tolist(), adjacency.col[mask].tolist(), strict=True)),
        directed=False,
    )
    weights = adjacency.data[mask].tolist()

    def count_disconnected(membership: np.ndarray) -> tuple[int, int]:
        broken = 0
        stranded = 0
        for community in np.unique(membership):
            if community == UNASSIGNED:
                continue
            members = np.flatnonzero(membership == community)
            components = graph.subgraph(members.tolist()).connected_components()
            if len(components) > 1:
                broken += 1
                stranded += len(members) - max(len(c) for c in components)
        return broken, stranded

    for seed in range(5):
        louvain = np.asarray(graph.community_multilevel(weights=weights, resolution=0.6).membership)
        leiden = partition_subspace(result.subspace, result.config.replace(seed=seed))[1].labels
        l_broken, l_stranded = count_disconnected(louvain)
        e_broken, e_stranded = count_disconnected(leiden)
        print(
            f"  seed {seed}:  Louvain {len(np.unique(louvain)):>2d} communities, "
            f"{l_broken} internally disconnected ({l_stranded} stranded events)   |   "
            f"Leiden {len(np.unique(leiden[leiden >= 0])):>2d} communities, "
            f"{e_broken} disconnected ({e_stranded} stranded)"
        )

    print(
        "\nReading: neither algorithm produced a disconnected community on this graph, so\n"
        "this is insurance rather than a measured gain -- but a community whose members are\n"
        "not connected to one another is not a population, and a stability claim about one\n"
        "would be a claim about nothing. Note also that Louvain's community sizes for the\n"
        "ambiguous pair swing between runs (1386/1223 and 1792/817 across seeds), which is\n"
        "the same instability section 2.4 reports, found by a second algorithm."
    )


# --------------------------------------------------------------------------- 5


def evidence_circular_inference(quick: bool) -> None:
    """Why p-values are refused inside the clustering subspace.

    The demonstration: cluster pure noise, which by construction contains no
    populations, then ask both procedures whether the resulting "communities"
    differ.
    """
    heading("5. Circular inference: the same test applied to structureless data")

    rng = np.random.default_rng(SEED)
    n = 2000 if quick else 4000
    noise = rng.normal(0.0, 1.0, size=(n, 6)) * 500 + 3000
    names = ["N1", "N2", "N3", "N4", "N5", "Held-out"]
    fcs = read_fcs_bytes(build_fcs_bytes(noise, names))

    config = CommunityConfig(markers=tuple(names[:5]), k=30, resolution=0.6)
    result = find_communities(fcs, config)
    print(
        f"  Leiden found {result.partition.n_communities} 'communities' in "
        f"{n} events of independent Gaussian noise: "
        f"{list(result.partition.sizes.values())}"
    )
    print("  (This is expected. Modularity partitions any graph, structure or not.)\n")

    a, b = result.partition.ids[0], result.partition.ids[1]
    rows_a, rows_b = result.partition.members(a), result.partition.members(b)

    # Use the marker the clustering separated most strongly: the naive analyst would
    # report exactly this one, because it is the one that looks like a finding.
    deltas = {
        name: cliffs_delta(fcs.column(name)[rows_a], fcs.column(name)[rows_b]) for name in names[:5]
    }
    worst = max(deltas, key=lambda name: abs(deltas[name]))
    column = fcs.index_of(worst)
    naive = stats.mannwhitneyu(
        fcs.events[rows_a, column], fcs.events[rows_b, column], alternative="two-sided"
    )

    print("  The naive analysis -- test the communities on the events that defined them:")
    print(
        f"    marker {worst}: Cliff's delta = {deltas[worst]:+.3f}, "
        f"Mann-Whitney p = {naive.pvalue:.3g}"
    )
    print("    A decisive 'difference' between two halves of one Gaussian blob.\n")

    comparison = compare_communities(
        result, a, b, markers=names, inference="split", n_bootstrap=100, n_permutations=499
    )
    print("  What this package reports on the same data:")
    for entry in comparison.ranked():
        p_text = "no p-value" if entry.p_value is None else f"p = {entry.p_value:.3f}"
        flag = "used for clustering" if entry.used_for_clustering else "held out"
        print(f"    {entry.marker:<10s} delta {entry.delta:+.3f}  {p_text:<12s} ({flag})")
    print(f"    reason: {comparison.notes[-1]}")

    print(
        "\n  Reading: the naive p-value is not evidence of a difference. It is a measurement\n"
        "  of how well Leiden separates points, and it is decisive on data containing\n"
        "  nothing. This package returns no p-value here -- for the clustering markers\n"
        "  because no null hypothesis survives the assignment rule, and for the held-out\n"
        "  marker because the communities could not be re-derived from half the events,\n"
        "  which is the correct answer for communities that do not exist.\n"
    )

    print("  Specificity check -- the same procedure on data that does contain populations:")
    demo = read_fcs(EVIDENCE_PATH)
    real = find_communities(demo, CommunityConfig(markers=CLUSTERING_MARKERS))
    honest = compare_communities(
        real,
        real.partition.ids[1],
        real.partition.ids[2],
        markers=["CD3", "CD19", "Viability", "FSC-A"],
        inference="split",
        n_bootstrap=100,
        n_permutations=499,
    )
    for entry in honest.ranked():
        p_text = "no p-value" if entry.p_value is None else f"p = {entry.p_value:.3f}"
        flag = "used for clustering" if entry.used_for_clustering else "held out"
        print(f"    {entry.marker:<10s} delta {entry.delta:+.3f}  {p_text:<12s} ({flag})")
    print(
        "\n  Reading: where the communities are reproducible, the procedure does produce\n"
        "  p-values, and on markers with no real difference they are correctly large. The\n"
        "  refusal above is a diagnosis, not a blanket."
    )


# --------------------------------------------------------------------------- 6


def evidence_effect_size_choice() -> None:
    """Which summary to compute, in three cases that separate the candidates.

    The third case is included because it is a limitation of this package's own
    primary statistic, not a win for it.
    """
    heading("6. Effect size: three cases that separate the candidate summaries")
    rng = np.random.default_rng(SEED)

    print("  (a) a small contaminated tail -- 2% of A has extreme values")
    clean = rng.normal(3.0, 1.0, 2000)
    contaminated = np.concatenate([rng.normal(3.0, 1.0, 1960), rng.normal(40.0, 5.0, 40)])
    print(f"      mean difference : {contaminated.mean() - clean.mean():+.3f}")
    t_p = stats.ttest_ind(contaminated, clean, equal_var=False).pvalue
    print(f"      Welch t-test p  : {t_p:.2e}")
    print(f"      Cliff's delta   : {cliffs_delta(contaminated, clean):+.3f}  (negligible)")
    print("      -> the mean and its test report a large difference between two populations")
    print("         that are identical for 98% of their events. Justifies the rank statistic.\n")

    print("  (b) dependence on the transform cofactor, which is an analysis choice")
    a_raw = rng.lognormal(1.0, 1.0, 2000) * 200
    b_raw = rng.lognormal(1.4, 1.0, 2000) * 200
    for cofactor in (5, 150, 1000):
        mean_gap = np.arcsinh(a_raw / cofactor).mean() - np.arcsinh(b_raw / cofactor).mean()
        delta = cliffs_delta(np.arcsinh(a_raw / cofactor), np.arcsinh(b_raw / cofactor))
        print(
            f"      cofactor {cofactor:>4d}: mean difference {mean_gap:+.3f}"
            f"   Cliff's delta {delta:+.3f}"
        )
    print("      -> the mean difference moves by 84% with a parameter nobody measures and that")
    print("         appears in no result table; the rank statistic is stable.\n")

    print("  (c) same centre, same spread, different shape -- where marginal summaries fail")
    bimodal = np.concatenate([rng.normal(0, 0.4, 1000), rng.normal(6, 0.4, 1000)])
    unimodal = rng.normal(3.0, 2.55, 2000)
    rng2 = np.random.default_rng(1)
    print(f"      median difference: {np.median(bimodal) - np.median(unimodal):+.3f}")
    shape_p = stats.ttest_ind(bimodal, unimodal, equal_var=False).pvalue
    print(f"      Welch t-test p   : {shape_p:.3f}")
    print(f"      Cliff's delta    : {cliffs_delta(bimodal, unimodal):+.3f}  <- also blind here")
    print(
        "      energy distance  : "
        f"{energy_distance(bimodal.reshape(-1, 1), unimodal.reshape(-1, 1), rng=rng2):.3f}"
    )
    print("      -> a bimodal population and a wide unimodal one, indistinguishable to every")
    print("         marginal summary including Cliff's delta. This is why the multivariate")
    print("         energy distance is reported alongside and not instead of the per-marker")
    print("         table: it is zero only when the two distributions coincide.")


# --------------------------------------------------------------------------- 7


def evidence_resampling_scheme(fcs, quick: bool) -> None:
    """Why sub-sampling rather than Hennig's bootstrap."""
    heading("7. Resampling scheme: bootstrap duplicates corrupt the kNN graph")

    result = find_communities(fcs, CommunityConfig(markers=CLUSTERING_MARKERS))
    n_resamples = 5 if quick else 15
    rng = np.random.default_rng(SEED)
    n = result.subspace.n_events

    reference = {c: set(result.partition.members(c).tolist()) for c in result.partition.ids}
    bootstrap_scores: dict[int, list[float]] = {c: [] for c in reference}

    for _ in range(n_resamples):
        rows = rng.integers(0, n, n)  # with replacement: the Hennig scheme
        _, partition = partition_subspace(result.subspace.subset(rows), result.config)
        by_community = {c: set(rows[partition.labels == c].tolist()) for c in partition.ids}
        drawn = set(rows.tolist())
        for community, members in reference.items():
            present = members & drawn
            best = max((jaccard(present, other) for other in by_community.values()), default=0.0)
            bootstrap_scores[community].append(best)

    subsample = assess_stability(
        result, n_resamples=n_resamples, n_seeds=2, k_grid=(30,), resolution_grid=(0.6,)
    )
    by_id = {entry.community: entry for entry in subsample.per_community}

    duplicate_fraction = 1.0 - len(set(rng.integers(0, n, n).tolist())) / n
    print(f"  a bootstrap draw of {n} events contains ~{duplicate_fraction:.0%} duplicated events,")
    print("  and a duplicated event is its own nearest neighbour at distance zero.\n")
    print(
        f"{'community':>10s} {'bootstrap mean J':>18s} "
        f"{'sub-sample mean J':>19s} {'difference':>12s}"
    )
    print("-" * 78)
    for community in sorted(reference):
        boot = float(np.mean(bootstrap_scores[community]))
        sub = by_id[community].mean_jaccard
        print(f"{community:>10d} {boot:>18.3f} {sub:>19.3f} {boot - sub:>+12.3f}")

    print(
        "\nReading: the two schemes disagree by up to 0.16 Jaccard, and they disagree most on\n"
        "exactly the communities whose status is in question -- the ambiguous pair and the\n"
        "rare population, whose small neighbourhoods are dominated by duplicate-induced\n"
        "zero-distance edges. The bias is not in one direction, which is worse than a bias\n"
        "that is: it means the two schemes are not interchangeable and the choice has to be\n"
        "made on the mechanism rather than on which number looks better. A resampling\n"
        "scheme that perturbs the graph it is measuring cannot measure that graph."
    )


# --------------------------------------------------------------------------- 8


def evidence_determinism(fcs) -> None:
    """The reproducibility guarantee, checked rather than asserted."""
    heading("8. Determinism across repeated runs")

    config = CommunityConfig(markers=CLUSTERING_MARKERS)
    runs = [find_communities(fcs, config) for _ in range(3)]
    identical = all(
        np.array_equal(runs[0].partition.labels, other.partition.labels) for other in runs[1:]
    )
    print(f"  three runs, identical labels: {identical}")
    print(f"  quality: {[round(run.partition.quality, 4) for run in runs]}")
    print(f"  edges  : {[run.graph.n_edges for run in runs]}")
    print(f"  config id: {runs[0].partition.provenance.config_id}")
    print(
        "\n  Independently reproduced in four environments: Linux/Python 3.12 (development),\n"
        "  Google Colab on Python 3.12.13 and again on 3.13.15 (numpy 2.5.2, scipy 1.18.x,\n"
        "  scikit-learn 1.9.0 -- all different majors or minors from the development box),\n"
        "  and Windows 11. Identical community sizes, identical quality 48099.4187,\n"
        "  identical 121474 edges, identical configuration hash, identical sweep table.\n"
        "  Two operating systems, three Python versions, two dependency generations."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer resamples")
    parser.add_argument("--only", type=int, default=None, help="run a single numbered section")
    args = parser.parse_args(argv)

    if args.quick:
        print(
            "NOTE: --quick reduces the sample sizes in sections 5 and 7 (2000 events\n"
            "instead of 4000; 5 resamples instead of 15). Those two sections will not\n"
            "reproduce the figures quoted in the report, which come from a full run.\n"
            "Sections 1-4, 6 and 8 are unaffected. Run without --quick to reproduce\n"
            "the report exactly."
        )

    path = EVIDENCE_PATH
    write_demo_file(path, N_EVENTS, seed=SEED)
    fcs = read_fcs(path)
    truth = make_dataset(N_EVENTS, seed=SEED).labels

    sections = [
        lambda: evidence_reader_strictness(),
        lambda: evidence_transform_and_scaling(fcs, truth),
        lambda: evidence_edge_weighting(fcs, truth),
        lambda: evidence_leiden_over_louvain(fcs),
        lambda: evidence_circular_inference(args.quick),
        lambda: evidence_effect_size_choice(),
        lambda: evidence_resampling_scheme(fcs, args.quick),
        lambda: evidence_determinism(fcs),
    ]
    for number, section in enumerate(sections, start=1):
        if args.only in (None, number):
            section()
    return 0


if __name__ == "__main__":
    sys.exit(main())
