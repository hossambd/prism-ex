"""The interfaces a reviewer actually touches.

The README's worked example is a sequence of CLI calls; if those break, the
submission is unusable no matter what the library does. These tests run the same
calls through ``main`` in-process.
"""

from __future__ import annotations

import json

import pytest

from prism_ex.cli import main


def test_demo_writes_a_readable_file(tmp_path, capsys):
    path = tmp_path / "demo.fcs"
    assert main(["demo", str(path), "--events", "400"]) == 0
    assert path.is_file()
    assert "ground-truth populations" in capsys.readouterr().out


def test_info_describes_the_file(tmp_path, capsys):
    path = tmp_path / "demo.fcs"
    main(["demo", str(path), "--events", "300"])
    capsys.readouterr()
    assert main(["info", str(path)]) == 0
    assert "FCS 3.1" in capsys.readouterr().out


def test_info_json_is_valid_json(tmp_path, capsys):
    path = tmp_path / "demo.fcs"
    main(["demo", str(path), "--events", "300"])
    capsys.readouterr()
    main(["info", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_events"] == 300


def test_communities_reports_sizes(tmp_path, capsys):
    path = tmp_path / "demo.fcs"
    main(["demo", str(path), "--events", "600"])
    capsys.readouterr()
    assert main(["communities", str(path), "--k", "15", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert sum(payload["sizes"].values()) == 600
    assert payload["config"]["k"] == 15


def test_compare_runs_end_to_end(tmp_path, capsys):
    path = tmp_path / "demo.fcs"
    main(["demo", str(path), "--events", "600"])
    capsys.readouterr()
    assert main(["compare", str(path), "--k", "15", "--a", "0", "--b", "1"]) == 0
    assert "Cliff" in capsys.readouterr().out


def test_a_bad_file_is_a_message_not_a_traceback(tmp_path, capsys):
    broken = tmp_path / "broken.fcs"
    broken.write_bytes(b"FCS2.0    " + b"0" * 100)
    assert main(["info", str(broken)]) == 1
    assert "UnsupportedFCSVersion" in capsys.readouterr().err


def test_unknown_marker_is_a_message_not_a_traceback(tmp_path, capsys):
    path = tmp_path / "demo.fcs"
    main(["demo", str(path), "--events", "300"])
    capsys.readouterr()
    assert main(["communities", str(path), "--markers", "CD999", "--k", "10"]) == 1
    assert "UnknownMarker" in capsys.readouterr().err


# ------------------------------------------------------------------ HTTP endpoint

fastapi = pytest.importorskip("fastapi", reason="optional [api] extra not installed")
pytest.importorskip("multipart", reason="python-multipart not installed")

from fastapi.testclient import TestClient  # noqa: E402

from prism_ex.api import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_sizes_endpoint_returns_community_sizes(client, tmp_path_factory):
    from prism_ex.synth import write_demo_file

    path, _ = write_demo_file(tmp_path_factory.mktemp("api") / "demo.fcs", 500, seed=7)
    with open(path, "rb") as handle:
        response = client.post(
            "/communities/sizes",
            files={"file": ("demo.fcs", handle, "application/octet-stream")},
            data={"markers": "CD3,CD4,CD8,CD19,CD56", "k": 15, "resolution": 0.6},
        )
    payload = response.json()
    assert response.status_code == 200
    assert sum(payload["sizes"].values()) == 500
    assert payload["provenance"]["package_version"]


def test_a_malformed_upload_is_a_422_naming_the_defect(client):
    response = client.post(
        "/communities/sizes",
        files={"file": ("bad.fcs", b"FCS2.0    " + b"0" * 100, "application/octet-stream")},
        data={"markers": "CD3"},
    )
    assert response.status_code == 422
    assert "UnsupportedFCSVersion" in response.json()["detail"]


def test_an_empty_upload_is_a_400(client):
    response = client.post(
        "/communities/sizes",
        files={"file": ("empty.fcs", b"", "application/octet-stream")},
        data={"markers": "CD3"},
    )
    assert response.status_code == 400
