"""CaImAn is disabled: every CaImAn entry point raises NotImplementedError.

DataJoint's CaImAn fork (the only CaImAn element_interface.run_caiman works with)
does not run with numpy 2, and upstream CaImAn's fit_file no longer does what
run_caiman expects. See element_calcium_imaging.caiman_support.
"""

import ast
import pathlib
import tomllib

import pytest

from element_calcium_imaging.caiman_support import (
    CAIMAN_UNSUPPORTED,
    caiman_unsupported,
)

ROOT = pathlib.Path(__file__).parents[2]
PACKAGE = ROOT / "element_calcium_imaging"
TABLE_MODULES = ["imaging.py", "imaging_no_curation.py", "imaging_preprocess.py"]


def test_caiman_unsupported_raises():
    with pytest.raises(NotImplementedError, match="CaImAn"):
        caiman_unsupported()


def test_message_explains_why_and_what_to_do():
    assert "numpy" in CAIMAN_UNSUPPORTED and "suite2p" in CAIMAN_UNSUPPORTED


def _is_guard(stmt):
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and getattr(stmt.value.func, "id", None) == "caiman_unsupported"
    )


def _is_caiman_import(stmt):
    if isinstance(stmt, ast.ImportFrom) and stmt.level == 0:
        module, names = stmt.module or "", [a.name for a in stmt.names]
    elif isinstance(stmt, ast.Import):
        module, names = "", [a.name for a in stmt.names]
    else:
        return False
    return (
        module.startswith("caiman")
        or "caiman_loader" in module
        or "run_caiman" in module
        or any(n.split(".")[0] == "caiman" for n in names)
        or any(n in ("caiman_loader", "run_caiman") for n in names)
    )


def _unguarded_caiman_imports(tree):
    """CaImAn imports not preceded by caiman_unsupported() in the same block.

    Legacy CaImAn code is kept after the guard for reference; it is unreachable.
    """
    bad = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            guarded = False
            for stmt in block:
                guarded = guarded or _is_guard(stmt)
                if _is_caiman_import(stmt) and not guarded:
                    bad.append(stmt.lineno)
    return bad


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: p.name)
def test_caiman_imports_only_after_guard(path):
    """Every CaImAn import sits after caiman_unsupported(), i.e. is unreachable."""
    assert _unguarded_caiman_imports(ast.parse(path.read_text())) == []


@pytest.mark.parametrize("name", TABLE_MODULES)
def test_every_caiman_entry_point_is_guarded(name):
    """Task generation, triggering and loading each call caiman_unsupported()."""
    tree = ast.parse((PACKAGE / name).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "caiman_unsupported"
    ]
    assert len(calls) == 3


@pytest.mark.parametrize("name", TABLE_MODULES)
def test_legacy_caiman_code_kept_for_reference(name):
    """The pre-disable implementation stays readable after each guard."""
    text = (PACKAGE / name).read_text()
    assert "run_caiman(" in text
    assert text.count("caiman_loader.CaImAn(") == 2


# ---- packaging ----


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_setup_py_removed():
    assert not (ROOT / "setup.py").exists()


def test_no_caiman_extras(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    assert not [e for e in extras if "caiman" in e.lower()]
    every = pyproject["project"]["dependencies"] + [
        d for ds in extras.values() for d in ds
    ]
    assert not [d for d in every if "caiman" in d.lower()]


def test_version_read_from_version_py(pyproject):
    assert "version" in pyproject["project"]["dynamic"]
    attr = pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "element_calcium_imaging.version.__version__"


def test_build_needs_no_network(pyproject):
    """setup.py fetched CaImAn's pyproject.toml at build time; nothing may do that now."""
    assert pyproject["build-system"]["requires"] == ["setuptools>=77"]  # SPDX `license`
