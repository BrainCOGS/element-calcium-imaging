"""run_suite2p / check_torch_device: running suite2p without the database (slurm jobs)."""

import copy

import numpy as np
import pytest

suite2p = pytest.importorskip("suite2p")
torch = pytest.importorskip("torch")

from element_calcium_imaging import suite2p_settings  # noqa: E402
from element_calcium_imaging.suite2p_settings import (  # noqa: E402
    check_torch_device,
    expected_plane_folders,
    run_suite2p,
)

SCAN_INFO = {"fs": 30.0, "nplanes": 2, "nchannels": 1}


@pytest.fixture
def fake_s2p(monkeypatch):
    """Replace suite2p.run_s2p with a stand-in that writes the output verify_outputs checks."""
    calls = []

    def run_s2p(db, settings):
        calls.append((copy.deepcopy(db), copy.deepcopy(settings)))
        for name in expected_plane_folders(db):
            plane = suite2p_settings.suite2p_save_dir(db) / name
            plane.mkdir(parents=True, exist_ok=True)
            for f in ("ops", "iscell", "F"):
                np.save(plane / f"{f}.npy", np.zeros(1))
            np.save(plane / "settings.npy", settings, allow_pickle=True)
        # suite2p edits the dicts it is given
        db.clear()
        settings.clear()

    monkeypatch.setattr(suite2p, "run_s2p", run_s2p)
    return calls


def run(tmp_path, params=None, **kw):
    kw.setdefault("image_files", [tmp_path / "raw" / "a.tif", tmp_path / "raw" / "b.tif"])
    kw.setdefault("output_dir", tmp_path / "out")
    kw.setdefault("scan_info", SCAN_INFO)
    return run_suite2p({} if params is None else params, **kw)


# ---- run_suite2p ----


def test_runs_with_pipeline_inputs(tmp_path, fake_s2p):
    db, settings = run(tmp_path)
    (called_db, called_settings), = fake_s2p
    assert called_db["file_list"] == [(tmp_path / "raw" / f).as_posix() for f in ("a.tif", "b.tif")]
    assert called_db["data_path"] == [(tmp_path / "raw").as_posix()]
    assert called_db["save_path0"] == (tmp_path / "out").as_posix()
    assert called_db["input_format"] == "tif"
    assert called_db["nplanes"] == 2 and called_settings["fs"] == 30.0
    # returns the dicts it asked for, not the ones suite2p edited
    assert db == called_db and settings == called_settings


def test_accepts_str_paths(tmp_path, fake_s2p):
    run(tmp_path, image_files=[str(tmp_path / "a.TIF")], output_dir=str(tmp_path / "out"))
    (db, _), = fake_s2p
    assert db["input_format"] == "tif"


def test_no_image_files(tmp_path, fake_s2p):
    with pytest.raises(FileNotFoundError, match="No input image files"):
        run(tmp_path, image_files=[])
    assert fake_s2p == []


def test_scan_info_optional(tmp_path, fake_s2p):
    run(tmp_path, params={"nplanes": 1, "fs": 15.0}, scan_info=None)
    (db, settings), = fake_s2p
    assert db["nplanes"] == 1 and settings["fs"] == 15.0


def test_rerun_into_existing_output(tmp_path, fake_s2p):
    """A requeued job writes into the folder its first attempt left behind."""
    old = tmp_path / "out" / "suite2p" / "plane0"
    old.mkdir(parents=True)
    (old / "data.bin").write_bytes(b"old")
    run(tmp_path)
    assert len(fake_s2p) == 1
    assert (old / "data.bin").read_bytes() == b"old"


def test_params_not_mutated(tmp_path, fake_s2p):
    params = {"nplanes": 2, "torch_device": "cpu", "registration": {"batch_size": 100}}
    before = copy.deepcopy(params)
    run(tmp_path, params=params, torch_device="cpu")
    assert params == before


def test_missing_output_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(suite2p, "run_s2p", lambda db, settings: None)
    with pytest.raises(RuntimeError, match="plane0"):
        run(tmp_path)


# ---- device selection ----


def test_device_defaults_to_stored_then_cpu(tmp_path, fake_s2p):
    _, settings = run(tmp_path)
    assert settings["torch_device"] == "cpu"


def test_explicit_device_overrides_stored(tmp_path, fake_s2p, monkeypatch):
    checked = []
    monkeypatch.setattr(suite2p_settings, "check_torch_device", lambda d: checked.append(d))
    _, settings = run(tmp_path, params={"torch_device": "cpu"}, torch_device="cuda")
    assert settings["torch_device"] == "cuda"
    assert fake_s2p[0][1]["torch_device"] == "cuda"
    assert checked == ["cuda"]


def test_stored_device_is_checked(tmp_path, fake_s2p, monkeypatch):
    checked = []
    monkeypatch.setattr(suite2p_settings, "check_torch_device", lambda d: checked.append(d))
    run(tmp_path, params={"torch_device": "cuda"})
    assert checked == ["cuda"]


def test_unusable_device_fails_before_running(tmp_path, fake_s2p, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        run(tmp_path, torch_device="cuda")
    assert fake_s2p == []
    assert not (tmp_path / "out" / "suite2p").exists()


# ---- check_torch_device ----


def test_check_cpu():
    check_torch_device("cpu")


@pytest.mark.parametrize("device", [None, "", 0])
def test_check_rejects_non_device(device):
    with pytest.raises(ValueError):
        check_torch_device(device)


def test_check_rejects_unknown_device_string():
    with pytest.raises(ValueError, match="gpu0"):
        check_torch_device("gpu0")


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:3"])
def test_check_cuda_unavailable(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        check_torch_device(device)


def test_check_cuda_kernel_failure(monkeypatch):
    """A GPU the installed torch has no kernels for (e.g. Pascal on a CUDA 13 build)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def no_kernel(*a, **k):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    monkeypatch.setattr(torch, "ones", no_kernel)
    with pytest.raises(RuntimeError, match="cannot run on cuda.*no kernel image"):
        check_torch_device("cuda")


def test_check_cuda_index_out_of_range():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    with pytest.raises(RuntimeError, match="cannot run on cuda"):
        check_torch_device(f"cuda:{torch.cuda.device_count()}")


def test_check_cuda_real_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    check_torch_device("cuda")


# ---- suite2p 1.1.0: io.save_mat writes into the settings of later planes ----
#
# run_plane passes save_mat ops = {**db, **settings, ..., **plane_times}, a shallow
# merge, so each nested group (registration, detection, ...) is the run's own dict
# unless plane_times has an entry of the same name, which it has only for the stages
# that ran on that plane. save_mat replaces every None in the groups it can reach with
# np.array([]). Job 1405 (a rerun): plane0 was already registered, so registration was
# skipped there, and plane1 then registered with upsample_meanImg=array([]):
# "The truth value of an empty array is ambiguous".

STAGES = ("registration", "detection", "classification")

# (path, the stage whose skipping exposes it to save_mat)
NONE_DEFAULTS = [
    (("registration", "upsample_meanImg"), "registration"),
    (("detection", "bin_size"), "detection"),
    (("detection", "cellpose_settings", "params"), "detection"),
    (("detection", "cellpose_settings", "params_chan2"), "detection"),
    (("classification", "classifier_path"), "classification"),
]


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def _save_mat_like_run_s2p(db, settings, plane, skipped):
    """What suite2p 1.1.0's run_plane does after a plane: save Fall.mat with real io.save_mat."""
    plane_settings = {**suite2p.default_settings(), **settings}
    plane_times = {stage: 1.0 for stage in STAGES if stage not in skipped}
    ops = {**db, **plane_settings, "save_path": plane.as_posix(), **plane_times}
    n = 3
    suite2p.io.save_mat(ops, np.zeros(0, dtype=object), np.zeros((0, n)), np.zeros((0, n)),
                        np.zeros((0, n)), np.zeros((0, 2)), None)


def _fake_s2p_saving_mat(monkeypatch, skipped_on_plane0):
    """run_s2p stand-in that saves Fall.mat after each plane and records each plane's settings.

    Plane 0 skips the ``skipped_on_plane0`` stages (e.g. already registered on a rerun);
    later planes run every stage.
    """
    seen = []

    def run_s2p(db, settings):
        for i, name in enumerate(expected_plane_folders(db)):
            seen.append(copy.deepcopy(settings))
            plane = suite2p_settings.suite2p_save_dir(db) / name
            plane.mkdir(parents=True, exist_ok=True)
            for f in ("ops", "iscell", "F"):
                np.save(plane / f"{f}.npy", np.zeros(1))
            np.save(plane / "settings.npy", settings, allow_pickle=True)
            _save_mat_like_run_s2p(db, settings, plane, skipped_on_plane0 if i == 0 else ())

    monkeypatch.setattr(suite2p, "run_s2p", run_s2p)
    return seen


@pytest.mark.parametrize("path,stage", NONE_DEFAULTS, ids=lambda p: ".".join(p) if isinstance(p, tuple) else None)
def test_save_mat_does_not_change_later_planes(tmp_path, monkeypatch, path, stage):
    seen = _fake_s2p_saving_mat(monkeypatch, skipped_on_plane0=(stage,))
    run(tmp_path)
    assert len(seen) == SCAN_INFO["nplanes"]
    for settings in seen:
        assert _get(settings, path) is None


def test_save_mat_rerun_with_plane0_registered(tmp_path, monkeypatch):
    """Job 1405: every stage but registration ran on plane0."""
    seen = _fake_s2p_saving_mat(monkeypatch, skipped_on_plane0=("registration",))
    run(tmp_path, params={"save_mat": True})
    assert _get(seen[1], ("registration", "upsample_meanImg")) is None


@pytest.mark.parametrize("skipped", [(), STAGES], ids=["all-ran", "none-ran"])
def test_save_mat_still_writes_fall_mat(tmp_path, monkeypatch, skipped):
    scipy_io = pytest.importorskip("scipy.io")
    _fake_s2p_saving_mat(monkeypatch, skipped_on_plane0=skipped)
    run(tmp_path)
    for name in ("plane0", "plane1"):
        mat = scipy_io.loadmat(tmp_path / "out" / "suite2p" / name / "Fall.mat")
        assert {"ops", "F", "iscell"} <= set(mat)


def test_save_mat_single_plane(tmp_path, monkeypatch):
    seen = _fake_s2p_saving_mat(monkeypatch, skipped_on_plane0=STAGES)
    run(tmp_path, scan_info={**SCAN_INFO, "nplanes": 1})
    assert len(seen) == 1
    assert _get(seen[0], ("registration", "upsample_meanImg")) is None


def test_copy_dicts_copies_dicts_only():
    arr = np.zeros(3)
    ops = {"registration": {"upsample_meanImg": None, "nested": {"x": None}}, "meanImg": arr, "fs": 10.0}
    copied = suite2p_settings._copy_dicts(ops)
    assert copied.keys() == ops.keys() and copied["fs"] == 10.0
    assert copied["registration"] == {"upsample_meanImg": None, "nested": {"x": None}}
    assert copied is not ops
    assert copied["registration"] is not ops["registration"]
    assert copied["registration"]["nested"] is not ops["registration"]["nested"]
    assert copied["meanImg"] is arr  # arrays are shared, not copied
    copied["registration"]["nested"]["x"] = 1
    assert ops["registration"]["nested"]["x"] is None


@pytest.mark.parametrize("value", [None, 0, [], {}, np.array([])], ids=repr)
def test_copy_dicts_non_dict_and_empty(value):
    copied = suite2p_settings._copy_dicts(value)
    if isinstance(value, dict):
        assert copied == {} and copied is not value
    else:
        assert copied is value


@pytest.mark.parametrize("fails", [False, True])
def test_save_mat_restored_after_run(tmp_path, monkeypatch, fails):
    original = suite2p.io.save_mat

    def run_s2p(db, settings):
        assert suite2p.io.save_mat is not original  # guarded while suite2p runs
        if fails:
            raise RuntimeError("suite2p failed")
        for name in expected_plane_folders(db):
            plane = suite2p_settings.suite2p_save_dir(db) / name
            plane.mkdir(parents=True, exist_ok=True)
            for f in ("ops", "iscell", "F"):
                np.save(plane / f"{f}.npy", np.zeros(1))
            np.save(plane / "settings.npy", settings, allow_pickle=True)

    monkeypatch.setattr(suite2p, "run_s2p", run_s2p)
    if fails:
        with pytest.raises(RuntimeError, match="suite2p failed"):
            run(tmp_path)
    else:
        run(tmp_path)
    assert suite2p.io.save_mat is original
