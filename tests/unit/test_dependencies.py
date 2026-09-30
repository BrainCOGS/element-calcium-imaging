"""The package declares what it imports; nothing may rely on another package's deps."""

import ast
import pathlib
import sys
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

ROOT = pathlib.Path(__file__).parents[2]
PACKAGE = ROOT / "element_calcium_imaging"
NWB_EXPORT = PACKAGE / "export" / "nwb"

# import name -> distribution name, where they differ
DIST = {
    "yaml": "pyyaml",
    "skimage": "scikit-image",
    "dash_extensions": "dash-extensions",
    "element_interface": "element-interface",
}


def _third_party(node):
    if isinstance(node, ast.Import):
        return [a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module.split(".")[0]]
    return []


def _imports(paths, top_level):
    found = set()
    for path in paths:
        tree = ast.parse(path.read_text())
        nodes = tree.body if top_level else ast.walk(tree)
        for node in nodes:
            for mod in _third_party(node):
                if mod not in sys.stdlib_module_names and mod != PACKAGE.name:
                    found.add(DIST.get(mod, mod).replace("_", "-").lower())
    return found


@pytest.fixture(scope="module")
def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def _names(reqs):
    return {Requirement(r).name.lower() for r in reqs}


def test_load_time_imports_are_declared(project):
    """Anything imported when a module loads must be a direct dependency."""
    paths = [p for p in PACKAGE.rglob("*.py") if NWB_EXPORT not in p.parents]
    missing = _imports(paths, top_level=True) - _names(project["dependencies"])
    assert not missing, f"imported at load time but not declared: {sorted(missing)}"


def test_nwb_export_imports_are_in_nwb_extra(project):
    """The optional NWB export (docs: `pip install element-calcium-imaging[nwb]`)."""
    paths = list(NWB_EXPORT.rglob("*.py"))
    declared = _names(project["dependencies"]) | _names(
        project["optional-dependencies"]["nwb"]
    )
    missing = _imports(paths, top_level=False) - declared
    assert not missing, f"NWB export imports not declared: {sorted(missing)}"


def test_datajoint_below_2(project):
    """datajoint 2 renamed dj.schema to dj.Schema; scan.py fails at import on 2.x."""
    (req,) = [Requirement(r) for r in project["dependencies"] if r.startswith("datajoint")]
    assert "2.0.0" not in req.specifier
    assert not SpecifierSet(str(req.specifier)).contains("2.3.3")
    assert req.specifier.contains("0.14.9")  # what U19 runs today
