"""Optional HTTP endpoint.

One endpoint returning community sizes and the provenance record. No persistence, no
job queue, no authentication.

The endpoint receives bytes rather than a path, which is why
:func:`prism_ex.fcs.read_fcs_bytes` exists alongside :func:`prism_ex.fcs.read_fcs`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from prism_ex import __version__
from prism_ex.errors import FCSError, PrismExError
from prism_ex.fcs.reader import read_fcs_bytes
from prism_ex.pipeline import CommunityConfig, find_communities

app = FastAPI(
    title="prism-ex",
    version=__version__,
    summary="Community sizes for an uploaded FCS 3.1 file.",
)

MAX_UPLOAD_BYTES = 256 * 1024 * 1024
"""Refuse uploads above this size rather than exhausting memory on a bad request."""


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe, so that "is it up" does not require uploading a file."""
    return {"status": "ok", "version": __version__}


@app.post("/communities/sizes")
async def community_sizes(
    file: Annotated[UploadFile, File(description="An FCS 3.1 file")],
    markers: Annotated[str, Form(description="Comma-separated marker subspace")],
    k: Annotated[int, Form()] = CommunityConfig(markers=("x",)).k,
    resolution: Annotated[float, Form()] = CommunityConfig(markers=("x",)).resolution,
    seed: Annotated[int, Form()] = 0,
) -> dict:
    """Return the community sizes for an uploaded file.

    The response carries the provenance record as well as the sizes: a client that
    gets numbers back from a service has no other way to know which version and
    configuration produced them.
    """
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes")

    selected = tuple(item.strip() for item in markers.split(",") if item.strip())
    if not selected:
        raise HTTPException(status_code=422, detail="markers must name at least one channel")

    try:
        fcs = read_fcs_bytes(payload, source=file.filename or "<upload>")
        result = find_communities(
            fcs, CommunityConfig(markers=selected, k=k, resolution=resolution, seed=seed)
        )
    except FCSError as error:
        # The file was the client's mistake: say which kind, since the reader knows.
        raise HTTPException(status_code=422, detail=f"{type(error).__name__}: {error}") from error
    except PrismExError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "sizes": {str(community): size for community, size in result.sizes.items()},
        "n_events": fcs.n_events,
        "n_communities": result.partition.n_communities,
        "provenance": result.partition.provenance.to_dict(),
    }
