import types

import numpy as np
import pytest

from element_calcium_imaging.imaging_preprocess import _s2p_package_version


def dataset(*ops):
    planes = {i: types.SimpleNamespace(ops=o) for i, o in enumerate(ops)}
    return types.SimpleNamespace(planes=planes)


@pytest.mark.parametrize(
    "ops, expected",
    [
        ({"version": "1.1.0"}, "1.1.0"),  # suite2p 1.x: settings["version"] in ops.npy
        ({"suite2p_version": "0.10.1"}, "0.10.1"),  # suite2p 0.x
        ({"version": "1.1.0", "suite2p_version": "0.10.1"}, "1.1.0"),
        ({"version": np.str_("1.1.0")}, "1.1.0"),
        ({}, ""),  # unknown: keep the column default
        ({"version": None}, ""),
        ({"version": ""}, ""),
        ({"version": "1.1.0.dev123+gabcdef0123456"}, "1.1.0.dev123+gab"),  # varchar(16)
    ],
)
def test_s2p_package_version(ops, expected):
    assert _s2p_package_version(dataset(ops)) == expected


def test_s2p_package_version_uses_first_plane():
    assert _s2p_package_version(dataset({"version": "1.1.0"}, {"version": "9.9.9"})) == "1.1.0"


def test_s2p_package_version_no_planes():
    assert _s2p_package_version(dataset()) == ""
