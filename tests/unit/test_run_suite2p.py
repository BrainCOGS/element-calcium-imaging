"""_run_suite2p on a job folder that already holds suite2p output (a rerun).

The pipeline hands an existing folder to suite2p as it always has: suite2p reuses
complete plane folders (keeping their binaries and db.npy) and overwrites the
results, or re-converts into incomplete ones. Nothing is refused or deleted by the
pipeline itself.
"""

import types

import numpy as np
import pytest

pytest.importorskip("suite2p")

from element_calcium_imaging import imaging_preprocess  # noqa: E402

KEY = {"recording_id": 1, "tiff_split": 0}


class _FakeScanInfo:
    def __and__(self, key):
        return self

    def fetch1(self, *attrs):
        return 10.0, 2, 1  # fps, ndepths, nchannels


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = []

    def fake_run_s2p(db, settings):
        calls.append(db)
        save = __import__("pathlib").Path(db["save_path0"]) / db["save_folder"]
        for i in range(db["nplanes"]):
            plane = save / f"plane{i}"
            plane.mkdir(parents=True, exist_ok=True)
            for f in ("ops", "iscell", "F"):
                np.save(plane / f"{f}.npy", np.zeros(1))
            np.save(plane / "settings.npy", settings, allow_pickle=True)

    monkeypatch.setattr(
        imaging_preprocess, "scan", types.SimpleNamespace(ScanInfo=_FakeScanInfo())
    )
    monkeypatch.setattr("suite2p.run_s2p", fake_run_s2p)
    return calls


def _previous_output(out, planes=("plane0", "plane1"), with_combined=True):
    for name in planes + (("combined",) if with_combined else ()):
        d = out / "suite2p" / name
        d.mkdir(parents=True)
        np.save(d / "stat.npy", np.zeros(3))
        (d / "data.bin").write_bytes(b"old")
    return out


@pytest.mark.parametrize("with_combined", [True, False])
def test_rerun_into_existing_output_runs_suite2p(
    tmp_path, fake_pipeline, with_combined
):
    out = _previous_output(tmp_path / "suite2p_output", with_combined=with_combined)
    tif = tmp_path / "raw" / "a.tif"
    tif.parent.mkdir()
    tif.write_bytes(b"")

    imaging_preprocess._run_suite2p({"nplanes": 2}, KEY, [tif], out)

    assert len(fake_pipeline) == 1
    assert fake_pipeline[0]["save_path0"] == out.as_posix()
    assert fake_pipeline[0]["save_folder"] == "suite2p"
    # nothing the previous attempt left was removed by the pipeline
    assert (out / "suite2p" / "plane0" / "data.bin").read_bytes() == b"old"


def test_fresh_folder_runs_suite2p(tmp_path, fake_pipeline):
    out = tmp_path / "suite2p_output"
    tif = tmp_path / "a.tif"
    tif.write_bytes(b"")
    imaging_preprocess._run_suite2p({"nplanes": 2}, KEY, [tif], out)
    assert len(fake_pipeline) == 1


def test_rerun_still_checks_outputs(tmp_path, monkeypatch, fake_pipeline):
    """The post-run check stays: a rerun that leaves a plane missing is reported."""
    out = _previous_output(tmp_path / "suite2p_output", planes=("plane0",))
    tif = tmp_path / "a.tif"
    tif.write_bytes(b"")
    monkeypatch.setattr("suite2p.run_s2p", lambda db, settings: None)  # writes nothing
    with pytest.raises(RuntimeError, match="plane"):
        imaging_preprocess._run_suite2p({"nplanes": 2}, KEY, [tif], out)
