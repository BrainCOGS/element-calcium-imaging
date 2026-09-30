import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("suite2p")

from element_calcium_imaging.suite2p_settings import (  # noqa: E402
    build_suite2p_inputs,
    expected_plane_folders,
    to_native,
    verify_outputs,
)

LEGACY_OPS = pathlib.Path(__file__).parent / "data" / "ops_0.10.1.json"


def build(params, tmp_path=pathlib.Path("/out"), **kw):
    kw.setdefault("data_path", "/data")
    kw.setdefault("file_list", ["/data/a.tif", "/data/b.tif"])
    kw.setdefault("save_path0", tmp_path)
    kw.setdefault("input_format", "tif")
    return build_suite2p_inputs(params, **kw)


@pytest.fixture
def legacy():
    return json.loads(LEGACY_OPS.read_text())


# ---- the full legacy config (values from the 0.10.1 -> 1.1.0 migration report) ----


def test_legacy_config_converts(legacy):
    db, s, _ = build(legacy)
    reg, det, ext = s["registration"], s["detection"], s["extraction"]
    assert det["algorithm"] == "sourcery"  # sparse_mode=False
    assert reg["spatial_taper"] == pytest.approx(3.45)  # not the unused 40
    assert reg["batch_size"] == ext["batch_size"] == db["batch_size"] == 500
    assert ext["neuropil_coefficient"] == 0.7
    assert det["nbins"] == 5000
    assert det["highpass_time"] == 100
    assert det["sparsery_settings"]["highpass_neuropil"] == 25
    assert reg["align_by_chan2"] is False
    assert reg["block_size"] == [64, 64]
    assert reg["nonrigid"] is True
    assert s["classification"]["classifier_path"] is None
    assert s["run"]["do_detection"] is True and s["run"]["do_deconvolution"] is True
    assert s["io"]["save_mat"] is True and s["io"]["combined"] is True
    assert det["sourcery_settings"]["max_iterations"] == 20
    assert db["nplanes"] == 3 and db["nchannels"] == 1
    for k in ("fast_disk", "subfolders", "ignore_flyback", "lines", "dy", "dx"):
        assert db[k] is None, k


# Paramsets stored in the lab database (0.10.1 flat, one 1.x nested, one with
# whitespace in its keys).
STORED = json.loads((pathlib.Path(__file__).parent / "data" / "stored_paramsets.json").read_text())


@pytest.mark.parametrize("idx", [2, 3, 4, 6, 7])
def test_stored_paramsets_convert(idx):
    p = STORED[idx]
    db, s, _ = build(p)
    flat = {k.replace(" ", ""): v for k, v in p.items()}
    assert db["nplanes"] == flat["nplanes"]
    assert s["fs"] == flat["fs"] and s["tau"] == flat["tau"]
    assert s["detection"]["algorithm"] == "sourcery"
    assert s["extraction"]["neuropil_coefficient"] == 0.7
    assert s["extraction"]["neuropil_extract"] is flat["neuropil_extract"]
    assert s["registration"]["block_size"] == flat["block_size"]
    assert s["registration"]["spatial_taper"] == pytest.approx(3.45)
    assert s["registration"]["nimg_init"] == flat["nimg_init"]
    assert s["detection"]["threshold_scaling"] == flat["threshold_scaling"]


@pytest.mark.parametrize("idx", [0, 1])
def test_stored_paramsets_zero_diameter(idx):
    db, s, _ = build(STORED[idx])
    assert s["diameter"] == [0.0, 0.0]
    assert s["detection"]["algorithm"] == "sourcery"
    assert db["nplanes"] == STORED[idx]["nplanes"]


def test_stored_1x_paramset_nested_wins():
    _, s, notes = build(STORED[5])
    assert s["extraction"]["neuropil_extract"] is True
    assert any("neuropil_extract" in n and "disagree" in n for n in notes)


def test_stored_1x_paramset_once_resolved():
    p = {k: v for k, v in STORED[5].items() if k != "neuropil_extract"}
    db, s, _ = build(p)
    assert s["registration"]["block_size"] == [128, 128]
    assert s["detection"]["nbins"] == 2000
    assert s["torch_device"] == "cpu" and s["tau"] == 1.5 and s["fs"] == 50
    assert s["io"]["combined"] is False
    assert s["registration"]["upsample_meanImg"] is None  # absent key -> 1.x default


def test_input_not_mutated(legacy):
    before = json.dumps(legacy, sort_keys=True)
    build(legacy)
    assert json.dumps(legacy, sort_keys=True) == before


def test_calls_do_not_share_state():
    """convert_settings_orig's default dicts leak values between calls."""
    db1, s1, _ = build({"nplanes": 5, "nonrigid": False})
    db2, s2, _ = build({})
    assert db1 is not db2 and s1 is not s2
    assert db2["nplanes"] == 1
    assert s2["registration"]["nonrigid"] is True


def test_empty_params_gives_defaults():
    from suite2p.parameters import default_settings

    db, s, _ = build({})
    expected = default_settings()
    expected["torch_device"] = "cpu"
    for group in ("registration", "detection"):  # tuples normalized to int lists
        expected[group]["block_size"] = list(expected[group]["block_size"])
    assert s == expected
    assert db["nplanes"] == 1 and db["input_format"] == "tif"


# ---- renamed / re-interpreted keys ----


@pytest.mark.parametrize("chan, expected", [(1, False), (2, True), (np.int64(2), True)])
def test_align_by_chan(chan, expected):
    _, s, _ = build({"align_by_chan": chan})
    assert s["registration"]["align_by_chan2"] is expected


@pytest.mark.parametrize(
    "params, algorithm, img",
    [
        ({"sparse_mode": False}, "sourcery", None),
        ({"sparse_mode": True}, "sparsery", None),
        ({"sparse_mode": 0}, "sourcery", None),
        ({"anatomical_only": 0}, "sparsery", None),  # 0.x default sparse_mode=True
        ({"anatomical_only": None, "sparse_mode": False}, "sourcery", None),
        ({"anatomical_only": 1, "sparse_mode": False}, "cellpose", "max_proj / meanImg"),
        ({"anatomical_only": 2}, "cellpose", "meanImg"),
        ({"anatomical_only": 4}, "cellpose", "max_proj"),
        ({}, "sparsery", None),
    ],
)
def test_detection_algorithm(params, algorithm, img):
    _, s, _ = build(params)
    assert s["detection"]["algorithm"] == algorithm
    if img:
        assert s["detection"]["cellpose_settings"]["img"] == img


def test_anatomical_only_3_falls_back_to_meanimg():
    _, s, notes = build({"anatomical_only": 3})
    assert s["detection"]["algorithm"] == "cellpose"
    assert s["detection"]["cellpose_settings"]["img"] == "meanImg"
    assert any("anatomical_only=3" in n for n in notes)


@pytest.mark.parametrize(
    "params, taper",
    [
        ({"spatial_taper": 40}, 3.45),
        ({"spatial_taper": 40, "smooth_sigma": 2.0}, 6.0),
        ({"smooth_sigma": 0}, 0.0),
        ({}, 3.45),  # untouched 1.x default
    ],
)
def test_spatial_taper_uses_0x_effective_value(params, taper):
    _, s, _ = build(params)
    assert s["registration"]["spatial_taper"] == pytest.approx(taper)


def test_batch_size_copied_everywhere():
    db, s, _ = build({"batch_size": 250})
    assert db["batch_size"] == s["registration"]["batch_size"] == s["extraction"]["batch_size"] == 250


def test_batch_size_absent_keeps_1x_defaults():
    db, s, _ = build({})
    assert (db["batch_size"], s["registration"]["batch_size"]) == (500, 100)


@pytest.mark.parametrize("value", [0, "", [], False, None])
def test_classifier_path_unset(value):
    _, s, _ = build({"classifier_path": value})
    assert s["classification"]["classifier_path"] is None


def test_classifier_path_kept():
    _, s, _ = build({"classifier_path": "/x/classifier.npy"})
    assert s["classification"]["classifier_path"] == "/x/classifier.npy"


@pytest.mark.parametrize("diameter, expected", [(12, [12.0, 12.0]), ([10, 14], [10, 14]),
                                                (np.float64(8), [8.0, 8.0])])
def test_diameter(diameter, expected):
    _, s, _ = build({"diameter": diameter})
    assert s["diameter"] == expected


@pytest.mark.parametrize("params", [
    {"anatomical_only": 1, "diameter": 0},
    {"sparse_mode": True, "diameter": 0},
    {"sparse_mode": False, "diameter": 0},  # values are not policed here
])
def test_zero_diameter_passed_through(params):
    _, s, _ = build(params)
    assert s["diameter"] == [0.0, 0.0]


@pytest.mark.parametrize("group", ["registration", "detection"])
def test_float_block_size_coerced_to_int(group):
    _, s, _ = build({group: {"block_size": [128.0, 64.0]}})
    assert s[group]["block_size"] == [128, 64]
    assert all(type(b) is int for b in s[group]["block_size"])


# ---- key clean-up ----


def test_whitespace_in_keys_removed():
    db, s, notes = build({" nplanes": 3, " do_ bidiphase": True, "registration": {" nonrigid": False}})
    assert db["nplanes"] == 3
    assert s["registration"]["do_bidiphase"] is True
    assert s["registration"]["nonrigid"] is False
    assert any("' do_ bidiphase'" in n for n in notes)


def test_whitespace_duplicate_keys_last_wins():
    _, _, notes = build({"nplanes": 3, " nplanes": 3})
    assert not any("more than once" in n for n in notes)
    db, _, notes = build({"nplanes": 3, " nplanes": 4})
    assert db["nplanes"] == 4
    assert any("more than once" in n for n in notes)


@pytest.mark.parametrize("key", ["neucoeff", " neucoeff", "Neucoeff"])
def test_neucoeff_spellings(key):
    _, s, _ = build({key: 0.5})
    assert s["extraction"]["neuropil_coefficient"] == 0.5


def test_neucoeff_both_spellings_last_wins():
    _, s, notes = build({"neucoeff": 0.7, "Neucoeff": 0.5})
    assert s["extraction"]["neuropil_coefficient"] == 0.5
    assert any("more than once" in n for n in notes)


def test_bruker_flag_sets_input_format():
    db, _, _ = build({"bruker": True})
    assert db["input_format"] == "bruker"


# ---- nested 1.x groups ----


def test_nested_group_is_deep_merged():
    _, s, _ = build({"nonrigid": False, "registration": {"batch_size": 64, "spatial_taper": 10}})
    reg = s["registration"]
    assert reg["batch_size"] == 64
    assert reg["spatial_taper"] == 10
    assert reg["nonrigid"] is False  # flat value survives a partial group
    assert reg["maxregshift"] == 0.1  # rest of the group keeps defaults


@pytest.mark.parametrize("params, path, value", [
    ({"neuropil_extract": False, "extraction": {"neuropil_extract": True}},
     ("extraction", "neuropil_extract"), True),
    ({"batch_size": 500, "registration": {"batch_size": 64}}, ("registration", "batch_size"), 64),
    ({"Neucoeff": 0.5, "extraction": {"neuropil_coefficient": 0.7}},
     ("extraction", "neuropil_coefficient"), 0.7),
    ({"sparse_mode": False, "detection": {"algorithm": "sparsery"}},
     ("detection", "algorithm"), "sparsery"),
    ({"roidetect": False, "run": {"do_detection": True}}, ("run", "do_detection"), True),
])
def test_flat_and_nested_conflict_nested_wins(params, path, value):
    _, s, notes = build(params)
    assert s[path[0]][path[1]] == value
    assert any("disagree" in n for n in notes)


@pytest.mark.parametrize("params", [
    {"neuropil_extract": False, "extraction": {"neuropil_extract": False}},
    {"batch_size": 64, "registration": {"batch_size": 64}},
    {"sparse_mode": False, "detection": {"algorithm": "sourcery"}},
    {"block_size": [64, 64], "registration": {"block_size": [64.0, 64.0]}},
])
def test_flat_and_nested_agreeing_ok(params):
    _, _, notes = build(params)
    assert not any("disagree" in n for n in notes)


def test_flat_spatial_taper_fixup_does_not_conflict_with_nested():
    """smooth_sigma (flat) derives spatial_taper; an explicit nested taper is not a clash."""
    _, s, _ = build({"smooth_sigma": 1.15, "registration": {"spatial_taper": 10}})
    assert s["registration"]["spatial_taper"] == 10


def test_nested_subgroup():
    _, s, _ = build({"detection": {"sparsery_settings": {"max_ROIs": 10}}})
    sp = s["detection"]["sparsery_settings"]
    assert sp["max_ROIs"] == 10 and "highpass_neuropil" in sp


def test_nested_unknown_key_dropped():
    _, s, notes = build({"registration": {"nonrgid": False}})
    assert s["registration"]["nonrigid"] is True
    assert "nonrgid" not in s["registration"]
    assert any("'nonrgid'" in n and "'nonrigid'" in n for n in notes)


def test_nested_group_not_a_dict_dropped():
    from suite2p.parameters import default_settings

    _, s, notes = build({"registration": False})
    assert s["registration"]["nonrigid"] == default_settings()["registration"]["nonrigid"]
    assert any("expected a dict" in n for n in notes)


def test_flat_1x_names_accepted():
    _, s, _ = build({"neuropil_coefficient": 0.5, "algorithm": "cellpose", "torch_device": "cuda"})
    assert s["extraction"]["neuropil_coefficient"] == 0.5
    assert s["detection"]["algorithm"] == "cellpose"
    assert s["torch_device"] == "cuda"


def test_1x_version_key_ignored():
    _, s, _ = build({"version": "1.0.0"})
    assert s["version"] == "1.1.0"


# ---- rejected / dropped keys ----


def test_unknown_key_dropped_with_suggestion():
    _, s, notes = build({"nonrgid": False})
    assert s["registration"]["nonrigid"] is True
    assert any("nonrgid" in n and "'nonrigid'" in n for n in notes)


def test_1preg_on_keeps_spatial_taper():
    _, s, notes = build({"1Preg": True, "spatial_taper": 40})
    assert s["registration"]["spatial_taper"] == 40
    assert any("1Preg" in n for n in notes)


@pytest.mark.parametrize("value", [False, 0, None])
def test_1preg_off_is_dropped(value):
    build({"1Preg": value})


@pytest.mark.parametrize("value", [-1, 0, None, []])
def test_frames_include_unset_ok(value):
    build({"frames_include": value})


@pytest.mark.parametrize("params", [{"frames_include": 1000}, {"h5py": ["/x.h5"]}])
def test_removed_inputs_dropped_with_warning(params):
    db, _, notes = build(params)
    assert any(f"dropped {next(iter(params))}" in n for n in notes)
    assert db["file_list"] == ["/data/a.tif", "/data/b.tif"]


def test_saved_db_output_keys_dropped_quietly():
    db, _, notes = build({"first_files": [True, False], "Ly": 512, "nframes": 100, "iplane": 0})
    assert "first_files" not in db and "Ly" not in db
    assert not any("first_files" in n or "Ly" in n for n in notes)


def test_long_values_truncated_in_notes():
    _, _, notes = build({"file_list": [f"/very/long/path/file_{i:05d}.tif" for i in range(50)]})
    assert all(len(n) < 200 for n in notes)


def test_output_only_keys_dropped():
    build({"xrange": np.array([0, 0]), "yrange": np.array([0, 0])})


def test_wrong_type_passed_through_with_warning():
    _, s, notes = build({"registration": {"nonrigid": "yes"}})
    assert s["registration"]["nonrigid"] == "yes"
    assert any("nonrigid" in n and "expects bool" in n for n in notes)


# ---- numpy values from DataJoint blobs ----


def test_numpy_values_become_native():
    db, s, _ = build({
        "nplanes": np.int64(2), "fs": np.float64(30.0), "nonrigid": np.bool_(False),
        "block_size": np.array([64, 64]), "fast_disk": np.array([]),
        "Neucoeff": np.array(0.6),
    })
    assert type(db["nplanes"]) is int and type(s["fs"]) is float
    assert s["registration"]["nonrigid"] is False
    assert s["registration"]["block_size"] == [64, 64]
    assert db["fast_disk"] is None
    assert s["extraction"]["neuropil_coefficient"] == 0.6


def test_to_native_nested():
    assert to_native({"a": [np.int32(1), {"b": np.array([])}], "c": (np.float32(0.5),)}) == {
        "a": [1, {"b": []}], "c": (0.5,)}


# ---- pipeline-owned paths ----


def test_pipeline_dictates_paths():
    db, _, notes = build({"save_path0": "/elsewhere", "save_folder": "custom",
                          "data_path": ["/x"], "look_one_level_down": True},
                         save_path0="/proc/out")
    assert db["save_path0"] == "/proc/out"
    assert db["save_folder"] == "suite2p"
    assert db["data_path"] == ["/data"]
    assert db["file_list"] == ["/data/a.tif", "/data/b.tif"]
    assert db["look_one_level_down"] is False
    assert any("save_path0" in n for n in notes)


def test_fast_disk_is_the_users():
    db, _, _ = build({"fast_disk": "/scratch"})
    assert db["fast_disk"] == "/scratch"


def test_stored_input_format_wins():
    db, _, _ = build({"input_format": "h5"}, input_format="tif")
    assert db["input_format"] == "h5"


# ---- ScanInfo vs stored values ----


SCAN = {"fs": 30.0, "nplanes": 4, "nchannels": 2}


def test_scan_info_fills_missing():
    db, s, _ = build({}, scan_info=SCAN)
    assert (s["fs"], db["nplanes"], db["nchannels"]) == (30.0, 4, 2)


def test_stored_values_win_over_scan_info():
    db, s, notes = build({"fs": 10, "nplanes": 1, "nchannels": 1}, scan_info=SCAN)
    assert (s["fs"], db["nplanes"], db["nchannels"]) == (10, 1, 1)
    assert sum("differs from ScanInfo" in n for n in notes) == 3


def test_scan_info_agreeing_values_not_noted():
    _, _, notes = build({"fs": 30.00001, "nplanes": 4}, scan_info=SCAN)
    assert not any("differs" in n for n in notes)


def test_scan_info_none_values_ignored():
    db, s, _ = build({"fs": 10}, scan_info={"fs": None})
    assert s["fs"] == 10


def test_scan_info_numpy_values():
    db, s, _ = build({}, scan_info={"fs": np.float64(7.5), "nplanes": np.int64(2)})
    assert type(s["fs"]) is float and type(db["nplanes"]) is int


# ---- torch device ----


def test_torch_device_defaults_to_cpu():
    _, s, _ = build({})
    assert s["torch_device"] == "cpu"


def test_torch_device_default_overridable():
    _, s, _ = build({}, default_torch_device="cuda")
    assert s["torch_device"] == "cuda"


# ---- multi-ROI ----


ROIS = {"lines": [list(range(0, 128)), list(range(144, 272))], "dy": [0, 0], "dx": [0, 128]}


def test_mroi_passthrough():
    db, _, _ = build({**ROIS, "nplanes": 3})
    assert db["lines"] == ROIS["lines"] and db["dx"] == [0, 128] and db["nrois"] == 2


def test_nrois_follows_lines():
    db, _, notes = build({**ROIS, "nrois": 3})
    assert db["nrois"] == 2
    assert any("replaced by len(lines)" in n for n in notes)


@pytest.mark.parametrize("nrois", [1, None])
def test_nrois_1_or_unset_with_lines(nrois):
    db, _, notes = build({**ROIS, "nrois": nrois})
    assert db["nrois"] == 2 and not any("nrois" in n for n in notes)


def test_nrois_without_lines_warns():
    db, _, notes = build({"nrois": 2})
    assert any("no effect without lines" in n for n in notes)


@pytest.mark.parametrize("value", [[], None])
def test_mroi_empty_lines_is_single_roi(value):
    db, _, _ = build({"lines": value, "dy": value, "dx": value, "mesoscan": False})
    assert db["lines"] is None and db["nrois"] == 1


def test_expected_plane_folders_mroi_ordering():
    db = {"nplanes": 3, "lines": ROIS["lines"], "ignore_flyback": None}
    assert expected_plane_folders(db) == [f"plane{i}" for i in range(6)]


def test_expected_plane_folders_flyback():
    db = {"nplanes": 4, "lines": None, "ignore_flyback": [3]}
    assert expected_plane_folders(db) == ["plane0", "plane1", "plane2"]


def test_expected_plane_folders_zero_planes():
    assert expected_plane_folders({"nplanes": 0, "lines": None}) == []


# ---- output folder guard / verification ----


def _fake_run(tmp_path, db, settings, planes):
    for name in planes:
        p = tmp_path / "suite2p" / name
        p.mkdir(parents=True)
        for f in ("ops", "iscell", "F"):
            np.save(p / f"{f}.npy", np.zeros(1))
        np.save(p / "settings.npy", settings, allow_pickle=True)


def test_verify_outputs_ok(tmp_path):
    db, s, _ = build({"nplanes": 2}, tmp_path=tmp_path)
    saved = {**s, "diameter": np.array(s["diameter"])}  # suite2p may store arrays
    _fake_run(tmp_path, db, saved, ["plane0", "plane1"])
    verify_outputs(db, s)


def test_verify_outputs_missing_plane(tmp_path):
    db, s, _ = build({"nplanes": 2}, tmp_path=tmp_path)
    _fake_run(tmp_path, db, s, ["plane0"])
    with pytest.raises(RuntimeError, match="plane1"):
        verify_outputs(db, s)


def test_verify_outputs_setting_mismatch(tmp_path):
    db, s, _ = build({}, tmp_path=tmp_path)
    used = json.loads(json.dumps(s))
    used["registration"]["batch_size"] = 100
    _fake_run(tmp_path, db, used, ["plane0"])
    with pytest.raises(RuntimeError, match=r"batch_size.*requested 500.*used 100"):
        verify_outputs(db, {**s, "registration": {**s["registration"], "batch_size": 500}})


# ---- dicts as fetched from ProcessingParamSet.params (DataJoint longblob) ----


@pytest.mark.parametrize("idx", range(len(STORED)))
def test_stored_paramsets_survive_datajoint_blob(idx):
    """fetch1("params") returns the blob-decoded dict; it must convert like the original."""
    from datajoint.blob import pack, unpack

    fetched = unpack(pack(STORED[idx]))
    want, got = build(STORED[idx]), build(fetched)
    assert want[0] == got[0] and want[1] == got[1]


def test_blob_with_numpy_values():
    """Blobs written from numpy (e.g. an ops/db.npy stored as-is) decode to arrays."""
    from datajoint.blob import pack, unpack

    fetched = unpack(pack({"nplanes": np.int64(2), "block_size": np.array([64, 64]),
                           "fast_disk": np.array([]), "Neucoeff": np.float32(0.5),
                           "registration": {"nonrigid": np.bool_(False)}}))
    db, s, _ = build(fetched)
    assert db["nplanes"] == 2 and type(db["nplanes"]) is int
    assert s["registration"]["block_size"] == [64, 64]
    assert db["fast_disk"] is None
    assert s["extraction"]["neuropil_coefficient"] == pytest.approx(0.5)
    assert s["registration"]["nonrigid"] is False
