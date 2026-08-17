"""Command line interface.

Written with :mod:`argparse` rather than a CLI framework: the package has five
subcommands and no need for a dependency that a reviewer then has to install to
run the worked example in the README.

Every subcommand can emit JSON (``--json``), because the point of a CLI in a
scientific package is that its output can be a file in a pipeline rather than
something a human retypes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from prism_ex import __version__
from prism_ex.communities import UNASSIGNED
from prism_ex.compare import compare_communities
from prism_ex.errors import PrismExError
from prism_ex.fcs.reader import read_fcs
from prism_ex.pipeline import CommunityConfig, find_communities
from prism_ex.stability import assess_stability
from prism_ex.synth import CLUSTERING_MARKERS, write_demo_file

DEFAULT_MARKERS = ",".join(CLUSTERING_MARKERS)

_DEFAULTS = CommunityConfig(markers=("placeholder",))
"""An instance, not the class: on a slots dataclass the class attributes are slot
descriptors rather than the default values."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prism-ex",
        description="FCS 3.1 ingestion, communities, comparison and stability.",
    )
    parser.add_argument("--version", action="version", version=f"prism-ex {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="generate a synthetic FCS 3.1 file with known populations"
    )
    demo.add_argument("path", type=Path, help="where to write the file")
    demo.add_argument("--events", type=int, default=6000)
    demo.add_argument("--seed", type=int, default=20260817)
    demo.set_defaults(handler=_demo)

    info = subparsers.add_parser("info", help="read a file and describe it")
    info.add_argument("path", type=Path)
    info.add_argument("--json", action="store_true")
    info.set_defaults(handler=_info)

    communities = subparsers.add_parser(
        "communities", help="find communities and report their sizes (section 2.2)"
    )
    _add_clustering_arguments(communities)
    communities.add_argument("--json", action="store_true")
    communities.add_argument(
        "--labels", type=Path, help="write per-event community labels to this .npy file"
    )
    communities.set_defaults(handler=_communities)

    compare = subparsers.add_parser(
        "compare", help="compare two communities' marker profiles (section 2.3)"
    )
    _add_clustering_arguments(compare)
    compare.add_argument("--a", type=int, required=True, help="first community id")
    compare.add_argument("--b", type=int, required=True, help="second community id")
    compare.add_argument(
        "--compare-markers",
        default=None,
        help="comma-separated comparison subspace; defaults to the clustering markers",
    )
    compare.add_argument("--inference", choices=("descriptive", "split"), default="descriptive")
    compare.add_argument("--json", action="store_true")
    compare.set_defaults(handler=_compare)

    stability = subparsers.add_parser(
        "stability", help="gather the evidence for the section 2.4 claim"
    )
    _add_clustering_arguments(stability)
    stability.add_argument("--resamples", type=int, default=40)
    stability.add_argument("--fraction", type=float, default=0.8)
    stability.add_argument("--seeds", type=int, default=5)
    stability.add_argument("--json", action="store_true")
    stability.set_defaults(handler=_stability)

    serve = subparsers.add_parser(
        "serve", help="run the optional HTTP endpoint (section 2.5, needs [api])"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_serve)

    return parser


def _add_clustering_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path, help="an FCS 3.1 file")
    parser.add_argument(
        "--markers",
        default=DEFAULT_MARKERS,
        help=f"comma-separated marker subspace (default: {DEFAULT_MARKERS})",
    )
    parser.add_argument("--k", type=int, default=_DEFAULTS.k)
    parser.add_argument("--resolution", type=float, default=_DEFAULTS.resolution)
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument("--min-size", type=int, default=_DEFAULTS.min_size)
    parser.add_argument("--transform", choices=("asinh", "none"), default=_DEFAULTS.transform)
    parser.add_argument("--cofactor", type=float, default=_DEFAULTS.cofactor)


def _config(args: argparse.Namespace) -> CommunityConfig:
    return CommunityConfig(
        markers=tuple(_split_markers(args.markers)),
        k=args.k,
        resolution=args.resolution,
        seed=args.seed,
        min_size=args.min_size,
        transform=args.transform,
        cofactor=args.cofactor,
    )


def _split_markers(raw: str) -> list[str]:
    markers = [item.strip() for item in raw.split(",") if item.strip()]
    if not markers:
        raise PrismExError("--markers must name at least one channel")
    return markers


# ------------------------------------------------------------------- handlers


def _demo(args: argparse.Namespace) -> int:
    path, dataset = write_demo_file(args.path, args.events, seed=args.seed)
    print(f"wrote {path} ({dataset.n_events} events, {len(dataset.markers)} channels)")
    print("ground-truth populations:")
    for name, size in dataset.sizes().items():
        print(f"  {name:<18s} {size:>6d}")
    return 0


def _info(args: argparse.Namespace) -> int:
    fcs = read_fcs(args.path)
    if args.json:
        payload = {
            "n_events": fcs.n_events,
            "channels": [
                {
                    "index": c.index,
                    "name": c.name,
                    "stain": c.stain,
                    "bits": c.bits,
                    "log": c.is_log,
                    "range": c.range_,
                }
                for c in fcs.channels
            ],
            "keywords": dict(fcs.keywords),
            "provenance": fcs.provenance.to_dict(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(fcs.summary())
    return 0


def _communities(args: argparse.Namespace) -> int:
    result = find_communities(read_fcs(args.path), _config(args))
    if args.labels:
        np.save(args.labels, result.partition.labels)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    print(
        f"{result.partition.n_communities} communities over {result.fcs.n_events} events "
        f"({result.graph.n_edges} edges, quality {result.partition.quality:.4f})"
    )
    for community, size in result.sizes.items():
        label = "unassigned" if community == UNASSIGNED else f"community {community}"
        share = size / result.fcs.n_events
        print(f"  {label:<14s} {size:>7d}  {share:>6.1%}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    result = find_communities(read_fcs(args.path), _config(args))
    markers = _split_markers(args.compare_markers) if args.compare_markers else None
    comparison = compare_communities(
        result, args.a, args.b, markers=markers, inference=args.inference
    )
    print(json.dumps(comparison.to_dict(), indent=2) if args.json else comparison.to_markdown())
    return 0


def _stability(args: argparse.Namespace) -> int:
    result = find_communities(read_fcs(args.path), _config(args))
    evidence = assess_stability(
        result,
        n_resamples=args.resamples,
        subsample_fraction=args.fraction,
        n_seeds=args.seeds,
        seed=args.seed,
    )
    print(json.dumps(evidence.to_dict(), indent=2) if args.json else evidence.to_markdown())
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "the HTTP endpoint needs the optional extra: pip install 'prism-ex[api]'",
            file=sys.stderr,
        )
        return 2
    uvicorn.run("prism_ex.api:app", host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code; errors are messages, not tracebacks."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except BrokenPipeError:
        # `prism-ex info big.fcs | head` closes the pipe early. Python would print a
        # traceback at shutdown; this is the documented way to exit quietly.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except PrismExError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    except FileNotFoundError as error:
        print(f"{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
