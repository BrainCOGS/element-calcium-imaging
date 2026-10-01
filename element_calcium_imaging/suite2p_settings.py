"""Build suite2p >= 1.0 ``db`` and ``settings`` dicts from one stored settings dict.

``ProcessingParamSet.params`` holds a single dict (one JSON) of suite2p settings,
normally in the flat pre-1.0 ``ops`` style. suite2p 1.x instead takes two dicts:

- ``db``: flat; files, planes, channels, ScanImage multi-ROI geometry
  (``lines``/``dy``/``dx``), output location.
- ``settings``: nested into ``run``, ``io``, ``registration``, ``detection``,
  ``classification``, ``extraction`` and ``dcnv_preprocess``.

suite2p ships ``suite2p.parameters.convert_settings_orig`` for this, but on its own
it is not safe: its default dicts are shared between calls, it silently drops
renamed keys (``Neucoeff``, ``nbinned``, ``sparse_mode`` ...), and it copies a few
keys whose meaning changed (``spatial_taper``, ``batch_size``). This module wraps it
with explicit handling for each of those. Its job is placement: every stored value
ends up where suite2p 1.x reads it. It does not judge whether values are sensible;
anything it cannot place (unknown or removed keys, contradictions) is logged as a
warning and returned in ``notes``, and the run proceeds.

Accepted input (one dict):

- Flat keys are read with their **suite2p 0.x** meaning (e.g. ``spatial_taper`` is
  ignored unless ``1Preg``, as 0.x did). Flat 1.x key names (``neuropil_coefficient``,
  ``algorithm``, ``lines``, ``torch_device`` ...) are also accepted.
- Nested 1.x groups (``{"registration": {...}, "detection": {...}}``) are applied
  last, with their **1.x** meaning, and override anything set by flat keys.

The pipeline owns the input and output locations (``data_path``, ``file_list``,
``save_path0``, ``save_folder``); values for these in the stored dict are replaced.
Planes, channels, ROIs (``lines``/``dy``/``dx``) and ``fs`` are the user's choice:
stored values win, and ``ScanInfo`` only fills in what the stored dict leaves out.

Mapping verified against suite2p 1.1.0 (see ``SUITE2P_VERSION``).
"""

import copy
import difflib
import logging
import pathlib
import re

import numpy as np

logger = logging.getLogger(__name__)

SUITE2P_VERSION = "1.1.0"

# Pipeline-owned db keys: always set from the processing task, never from params.
PIPELINE_DB_KEYS = ("data_path", "file_list", "save_path0", "save_folder")
SAVE_FOLDER = "suite2p"

# Flat 0.x key -> (path in settings, transform). convert_settings_orig drops these.
_RENAMED = {
    "Neucoeff": (("extraction", "neuropil_coefficient"), None),
    "nbinned": (("detection", "nbins"), None),
    "high_pass": (("detection", "highpass_time"), None),
    "spatial_hp_detect": (("detection", "sparsery_settings", "highpass_neuropil"), None),
    # 0.x: 1-based channel number; 1.x: bool "use channel 2"
    "align_by_chan": (("registration", "align_by_chan2"), lambda v: int(v) == 2),
}

# Misspellings found in stored paramsets. 0.x ignored lowercase "neucoeff" (it read
# "Neucoeff"), but every stored use of it is clearly meant as the neuropil coefficient.
_ALIASES = {"neucoeff": "Neucoeff"}

# 0.x anatomical_only -> 1.x detection.cellpose_settings.img. 3 (enhanced mean
# image) has no 1.x equivalent and is rejected.
_ANATOMICAL_IMG = {1: "max_proj / meanImg", 2: "meanImg", 4: "max_proj"}

# 0.x keys with no 1.x equivalent whose default/typical values change nothing.
# Values that *would* change processing are rejected in _check_removed_keys.
_DROPPED = {
    "suite2p_version": "replaced by settings['version'], set by suite2p",
    "aspect": "GUI display option only",
    "bidi_corrected": "output flag written by suite2p, not an input",
    "force_refImg": "removed; the reference image is always computed",
    "pad_fft": "removed from registration",
    "spatial_hp": "removed (legacy alias)",
    "spatial_hp_reg": "removed with 1P registration (1Preg)",
    "pre_smooth": "removed with 1P registration (1Preg)",
    "1Preg": "removed; 1P registration no longer exists",
    "frames_include": "removed; no frame-limit option in 1.x",
    "mesoscan": "removed; multi-ROI is enabled by lines/dy/dx",
    "h5py": "removed; the pipeline supplies the input files",
    "bruker": "replaced by input_format='bruker'",
    "xrange": "registration output, not an input",
    "yrange": "registration output, not an input",
    "chan2_thres": "meaning changed: 1.x detection.chan2_threshold is an IoU threshold "
    "for cellpose_chan2, not a red-cell probability; left at the 1.x default",
}

# Keys suite2p 1.x writes into a saved db.npy; present when a used db is stored back.
_DB_OUTPUTS = (
    "first_files", "Lx", "Ly", "db_path", "frames_per_file", "frames_per_folder", "iplane",
    "iroi", "meanImg", "nframes", "reg_file", "raw_file", "reg_file_chan2", "raw_file_chan2",
    "save_path", "settings_path", "ops_path",
)
_DROPPED.update({k: "written by suite2p, not an input" for k in _DB_OUTPUTS})

# db keys where 0.x used [] (or "") for "unset" and 1.x uses None.
_EMPTY_TO_NONE = ("fast_disk", "subfolders", "ignore_flyback", "lines", "dy", "dx")


def to_native(value):
    """Recursively turn numpy scalars/arrays (e.g. from a DataJoint blob) into Python types.

    suite2p tests values like ``if upsample_meanImg:``, which raises on numpy arrays.
    Empty arrays become empty lists; 0-d arrays become scalars.
    """
    if isinstance(value, dict):
        return {k: to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_native(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else to_native(value.tolist())
    return value


def _short(value, n=80):
    text = repr(value)
    return text if len(text) <= n else text[: n - 3] + "..."


def _is_empty(value):
    return value is None or (isinstance(value, (list, tuple, str)) and len(value) == 0)


def _set_path(d, path, value):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def _deep_update(target, updates, spec, where, warn):
    """Recursively apply ``updates`` onto ``target``; keys not in ``spec`` are dropped."""
    for k, v in updates.items():
        if k not in spec:
            warn(f"dropped unknown suite2p setting {where}[{k!r}]{_suggest(k, spec)}")
        elif isinstance(spec[k], dict) and "description" not in spec[k]:
            if isinstance(v, dict):
                _deep_update(target[k], v, spec[k], f"{where}[{k!r}]", warn)
            else:
                warn(f"dropped {where}[{k!r}]={v!r}: expected a dict of settings")
        else:
            target[k] = v


def _suggest(key, candidates):
    match = difflib.get_close_matches(key, list(candidates), n=1, cutoff=0.6)
    lower = {c.lower(): c for c in candidates}
    if key.lower() in lower:
        match = [lower[key.lower()]]
    return f" (did you mean {match[0]!r}?)" if match else ""


def _validate(values, spec, where, warnings):
    """Type/range check against suite2p's parameter spec (DB / SETTINGS)."""
    for k, s in spec.items():
        if "description" not in s:
            _validate(values.get(k, {}), s, f"{where}[{k!r}]", warnings)
            continue
        v = values.get(k)
        if v is None:
            continue
        typ = s["type"]
        if typ in (list, tuple):
            ok = isinstance(v, (list, tuple))
        elif typ is float:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        elif typ in (int, bool):
            ok = isinstance(v, (int, bool))
        else:
            ok = isinstance(v, typ)
        if not ok:
            warnings.append(f"{where}[{k!r}] = {v!r}: suite2p expects {typ.__name__}")
        elif s["min"] is not None and typ in (int, float) and not s["min"] <= v <= s["max"]:
            warnings.append(f"{where}[{k!r}] = {v!r}: outside [{s['min']}, {s['max']}]")


def _set_nrois(db, warn):
    """suite2p splits ScanImage ROIs only from ``lines`` (with ``dy``/``dx``)."""
    if db.get("lines"):
        if db.get("nrois") not in (None, 1, len(db["lines"])):
            warn(f"nrois={db['nrois']} replaced by len(lines)={len(db['lines'])}")
        db["nrois"] = len(db["lines"])
    elif db.get("nrois", 1) != 1:
        warn(f"nrois={db['nrois']} has no effect without lines/dy/dx")


def build_suite2p_inputs(
    params,
    *,
    data_path,
    file_list,
    save_path0,
    input_format,
    scan_info=None,
    default_torch_device="cpu",
):
    """Convert one stored suite2p settings dict into suite2p 1.x ``(db, settings)``.

    Args:
        params (dict): Stored settings (``ProcessingParamSet.params``). Not modified.
        data_path (str): Folder holding the input files.
        file_list (list[str]): Input files, in order.
        save_path0 (str): Output folder; suite2p writes to ``save_path0/suite2p``.
        input_format (str): suite2p input format, e.g. ``"tif"``. A stored
            ``input_format`` (or legacy ``bruker=True``) takes precedence.
        scan_info (dict, optional): ``fs``, ``nplanes``, ``nchannels`` from ScanInfo.
            Used only for keys the stored dict leaves out; disagreements are logged.
        default_torch_device (str): ``torch_device`` when the stored dict has none.

    Returns:
        tuple: ``(db, settings, notes)`` where ``notes`` lists what was renamed,
        dropped or overridden (also logged).

    Nothing about the values is enforced: settings that cannot be placed are
    dropped with a warning (see ``notes``).
    """
    from suite2p.parameters import (
        DB,
        SETTINGS,
        convert_settings_orig,
        default_db,
        default_settings,
    )

    if isinstance(params, (str, pathlib.Path)):
        raise TypeError("params must be a dict (load the JSON first)")
    notes = []

    def note(msg, level=logging.INFO):
        notes.append(msg)
        logger.log(level, msg)

    def warn(msg):
        note(msg, logging.WARNING)

    params = _normalize_keys(to_native(copy.deepcopy(dict(params))), warn)

    # Nested 1.x groups are applied last, with 1.x meaning.
    groups = {k for k, v in SETTINGS.items() if "description" not in v}
    nested = {k: params.pop(k) for k in list(params) if k in groups}
    if "version" in params:  # a 1.x settings dict carries its version; not an input
        params.pop("version")
    flat = params

    # Fresh dicts: convert_settings_orig's defaults are created once at import
    # and mutated, so they would leak values between calls.
    db, settings, left = convert_settings_orig(
        copy.deepcopy(flat), db=default_db(), settings=default_settings()
    )
    reg, det, ext = settings["registration"], settings["detection"], settings["extraction"]

    for old, (path, fn) in _RENAMED.items():
        if old in left:
            value = left.pop(old)
            value = fn(value) if fn else value
            _set_path(settings, path, value)
            note(f"{old} -> {'.'.join(path)} = {value!r}")

    # sparse_mode / anatomical_only -> detection.algorithm. Only when given: the
    # 0.x default (sparse_mode=True) is the 1.x default (sparsery).
    if "sparse_mode" in left or "anatomical_only" in left:
        anat = int(left.pop("anatomical_only", 0) or 0)
        sparse = bool(left.pop("sparse_mode", True))
        if anat > 0:
            det["algorithm"] = "cellpose"
            if anat not in _ANATOMICAL_IMG:
                warn(f"anatomical_only={anat} has no 1.x image; using 'meanImg'")
            det["cellpose_settings"]["img"] = _ANATOMICAL_IMG.get(anat, "meanImg")
        else:
            det["algorithm"] = "sparsery" if sparse else "sourcery"
        note(
            f"sparse_mode={sparse}, anatomical_only={anat} -> "
            f"detection.algorithm = {det['algorithm']!r}"
        )

    # 0.x had one batch_size; 1.x has it in db, registration and extraction and
    # convert_settings_orig only fills db (registration would fall back to 100).
    if "batch_size" in flat:
        reg["batch_size"] = ext["batch_size"] = db["batch_size"]
        note(f"batch_size = {db['batch_size']} -> db, registration and extraction")

    # 0.x used spatial_taper only with 1Preg=True; otherwise it
    # tapered by 3 * smooth_sigma. 1.x always uses spatial_taper.
    if flat.get("1Preg"):
        warn("1Preg=True: 1P registration was removed in 1.x; spatial_taper kept as given")
    elif "spatial_taper" in flat or "smooth_sigma" in flat:
        taper = round(3 * float(reg["smooth_sigma"]), 6)
        if "spatial_taper" in flat and flat["spatial_taper"] != taper:
            note(
                f"spatial_taper = {flat['spatial_taper']!r} was unused in 0.x (1Preg=False); "
                f"registration.spatial_taper = 3 * smooth_sigma = {taper:g}",
                logging.WARNING,
            )
        reg["spatial_taper"] = taper

    if settings["classification"]["classifier_path"] in (0, "", [], False):
        settings["classification"]["classifier_path"] = None

    if left.pop("bruker", False) and "input_format" not in flat:
        db["input_format"] = "bruker"

    for k in list(left):
        value = left.pop(k)
        if k not in _DROPPED:
            warn(f"dropped unknown suite2p setting {k}={_short(value)}"
                 f"{_suggest(k, _known_flat_keys(DB, SETTINGS))}")
        elif k not in _DB_OUTPUTS and not _is_empty(value) and value not in (False, 0, -1):
            warn(f"dropped {k}={_short(value)}: {_DROPPED[k]}")

    # Nested 1.x groups are applied last and win over top-level keys.
    flat_paths = _flat_target_paths(flat, DB, SETTINGS)
    for path, value in _leaves(nested):
        for key, fpath in flat_paths:
            if fpath == path and _diff(value, _get_path(settings, path), ""):
                warn(f"{'.'.join(path)}: top-level {key}={flat[key]!r} and nested "
                     f"{value!r} disagree; using the nested value")
    _deep_update(settings, nested, SETTINGS, "settings", warn)
    if nested:
        note(f"applied nested settings groups: {sorted(nested)}")

    # 0.x used [] for "unset"; 1.x uses None.
    for k in _EMPTY_TO_NONE:
        if _is_empty(db.get(k)):
            db[k] = None
    if not isinstance(settings["diameter"], (list, tuple)):
        settings["diameter"] = [float(settings["diameter"])] * 2
    for group in (reg, det):  # stored as floats by some tools; used as array sizes
        if group["block_size"] is not None:
            group["block_size"] = [int(round(b)) for b in group["block_size"]]
    if "torch_device" not in flat:
        settings["torch_device"] = default_torch_device

    # Inputs and outputs are the pipeline's.
    dictated = {
        "data_path": [str(data_path)],
        "file_list": [str(f) for f in file_list],
        "save_path0": str(save_path0),
        "save_folder": SAVE_FOLDER,
        "look_one_level_down": False,
        "subfolders": None,
    }
    for k, v in dictated.items():
        if k in flat and not _is_empty(flat[k]) and flat[k] != v:
            note(f"{k}={_short(flat[k])} ignored: set by the pipeline")
        db[k] = v
    if "input_format" not in flat and db["input_format"] != "bruker":
        db["input_format"] = input_format

    # Planes/channels/fs are the user's call; ScanInfo only fills gaps.
    for k, v in (scan_info or {}).items():
        target = settings if k == "fs" else db
        if k not in flat:
            target[k] = to_native(v)
        elif v is not None and not np.isclose(float(target[k]), float(v), rtol=1e-3):
            note(
                f"stored {k}={target[k]!r} differs from ScanInfo {k}={to_native(v)!r}; "
                "using the stored value",
                logging.WARNING,
            )

    _set_nrois(db, warn)
    warnings = []
    _validate(db, DB, "db", warnings)
    _validate(settings, SETTINGS, "settings", warnings)
    if det["algorithm"] != "cellpose":  # cellpose settings are unused otherwise
        warnings = [w for w in warnings if "cellpose_settings" not in w]
    for w in warnings:
        warn(w)

    return db, settings, notes


def _normalize_keys(params, warn):
    """Remove whitespace inside keys and resolve legacy aliases.

    Hand-edited JSON has produced keys like ``" fast_disk"`` and ``" do_ bidiphase"``;
    suite2p would not recognise them. Nested groups are normalized too.
    """
    out = {}
    for key, value in params.items():
        clean = re.sub(r"\s+", "", str(key))
        if isinstance(value, dict):
            value = _normalize_keys(value, warn)
        clean = _ALIASES.get(clean, clean)
        if clean in out and _diff(out[clean], value, ""):
            warn(f"{clean!r} given more than once ({out[clean]!r}, {value!r}); using {value!r}")
        if clean != key:
            warn(f"key {key!r} read as {clean!r}")
        out[clean] = value
    return out


def _leaves(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def _get_path(d, path):
    for k in path:
        d = d[k]
    return d


def _flat_target_paths(flat, db_spec, settings_spec):
    """Settings path each flat key is written to, mirroring convert_settings_orig.

    db keys win first (so ``batch_size`` goes to db), then top-level settings, then
    the first nested group holding a key of that name (depth-first, spec order).
    """
    first = {}
    for path, _ in _leaves(_spec_tree(settings_spec)):
        first.setdefault(path[-1], path)
    pairs = []
    for key in flat:
        if key in _RENAMED:
            pairs.append((key, _RENAMED[key][0]))
        elif key in ("sparse_mode", "anatomical_only"):
            pairs.append((key, ("detection", "algorithm")))
        elif key == "roidetect":
            pairs.append((key, ("run", "do_detection")))
        elif key == "spikedetect":
            pairs.append((key, ("run", "do_deconvolution")))
        elif key == "batch_size":  # copied into registration and extraction
            pairs += [(key, ("registration", "batch_size")), (key, ("extraction", "batch_size"))]
        elif key not in db_spec and key in first:
            pairs.append((key, first[key]))
    return pairs


def _spec_tree(spec):
    return {k: (None if "description" in v else _spec_tree(v)) for k, v in spec.items()}


def _known_flat_keys(db_spec, settings_spec):
    keys = set(db_spec) | set(_RENAMED) | set(_DROPPED) | {"sparse_mode", "anatomical_only"}
    keys |= {"roidetect", "spikedetect"}

    def walk(spec):
        for k, v in spec.items():
            keys.add(k)
            if "description" not in v:
                walk(v)

    walk(settings_spec)
    return keys


def suite2p_save_dir(db):
    """Folder suite2p writes plane folders to for this ``db``."""
    return pathlib.Path(db["save_path0"]) / db["save_folder"]


def expected_plane_folders(db):
    """Plane folder names suite2p 1.x writes: ``plane{iplane * nrois + iroi}``."""
    nrois = len(db["lines"]) if db.get("lines") else 1
    skip = set(db.get("ignore_flyback") or [])
    return [
        f"plane{p * nrois + r}"
        for p in range(db["nplanes"])
        if p not in skip
        for r in range(nrois)
    ]


def verify_outputs(db, settings):
    """Check the run used the intended settings and produced every plane folder.

    Compares the ``settings.npy`` suite2p saved in each plane folder with
    ``settings`` and checks the expected ``planeN`` folders exist with their
    ``ops.npy``/``iscell.npy``/``F.npy``.

    Raises:
        RuntimeError: on any missing folder or setting that differs.
    """
    save_dir = suite2p_save_dir(db)
    problems = []
    intended = to_native(settings)
    for name in expected_plane_folders(db):
        plane = save_dir / name
        missing = [f for f in ("ops.npy", "iscell.npy", "F.npy", "settings.npy")
                   if not (plane / f).exists()]
        if missing:
            problems.append(f"{plane}: missing {', '.join(missing)}")
            continue
        used = to_native(np.load(plane / "settings.npy", allow_pickle=True).item())
        problems += [f"{name}: {d}" for d in _diff(intended, used, "settings")]
    if problems:
        raise RuntimeError("suite2p output does not match the requested run:\n  "
                           + "\n  ".join(problems))


def check_torch_device(device):
    """Fail early if torch cannot compute on ``device``.

    ``torch.cuda.is_available()`` alone is not enough: it is True on a GPU the
    installed torch has no kernels for (e.g. a Pascal card with a CUDA 13 build),
    and suite2p would then fail partway through. A small FFT on the device catches
    that before any work starts.

    Raises:
        ValueError: ``device`` is not a torch device string.
        RuntimeError: the device is unavailable or cannot run torch kernels.
    """
    import torch

    if not isinstance(device, str) or not device:
        raise ValueError(f"torch_device must be a device string such as 'cpu' or 'cuda', got {device!r}")
    try:
        dev = torch.device(device)
    except RuntimeError as e:
        raise ValueError(f"Unknown torch_device {device!r}: {e}") from e
    build = f"torch {torch.__version__} (CUDA {torch.version.cuda})"
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"torch_device={device!r} requested but CUDA is not available to {build}")
    try:
        torch.fft.fft2(torch.ones(64, 64, device=dev)).abs().sum().item()
    except Exception as e:
        raise RuntimeError(f"{build} cannot run on {device}: {e}") from e
    if dev.type == "cuda":
        logger.info("suite2p torch_device=%s: %s", device, torch.cuda.get_device_name(dev))


def run_suite2p(params, *, image_files, output_dir, scan_info=None, torch_device=None):
    """Run suite2p 1.x on ``image_files`` with the stored settings dict ``params``.

    Needs no database, so it also runs where the pipeline database is out of reach
    (e.g. a slurm job); ``Processing`` then ingests the output with ``task_mode="load"``.
    Checks the device before starting and each plane's saved settings afterwards.

    A rerun writes into the existing output folder: suite2p reuses complete plane
    folders (their binaries and db.npy; file list, nplanes and ROI geometry are not
    re-read) and overwrites the results, or re-converts the inputs into incomplete ones.

    Args:
        params (dict): Stored settings (``ProcessingParamSet.params``). Not modified.
        image_files (list): Input files, in order; all in one folder.
        output_dir (str | Path): suite2p writes to ``output_dir/suite2p``.
        scan_info (dict, optional): ``fs``, ``nplanes``, ``nchannels`` (see
            ``build_suite2p_inputs``).
        torch_device (str, optional): Device to compute on, e.g. ``"cuda"``. Overrides
            the stored ``torch_device``; None keeps the stored value (default ``"cpu"``).

    Returns:
        tuple: the ``(db, settings)`` suite2p was run with.
    """
    import suite2p

    image_files = [pathlib.Path(f) for f in image_files]
    if not image_files:
        raise FileNotFoundError("No input image files for suite2p processing")
    db, settings, _ = build_suite2p_inputs(
        params,
        data_path=image_files[0].parent.as_posix(),
        file_list=[f.as_posix() for f in image_files],
        save_path0=pathlib.Path(output_dir).as_posix(),
        input_format=image_files[0].suffix.lstrip(".").lower(),
        scan_info=scan_info,
    )
    if torch_device is not None:
        if settings["torch_device"] != torch_device:
            logger.info("torch_device=%r replaced by %r", settings["torch_device"], torch_device)
        settings["torch_device"] = torch_device
    check_torch_device(settings["torch_device"])
    # run_s2p edits the dicts it is given; keep ours for the check afterwards.
    suite2p.run_s2p(db=copy.deepcopy(db), settings=copy.deepcopy(settings))
    verify_outputs(db, settings)
    return db, settings


def _diff(want, got, where):
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return [f"{where}: expected a dict, got {got!r}"]
        return [d for k in want for d in _diff(want[k], got.get(k), f"{where}[{k!r}]")]
    if isinstance(want, (list, tuple)) and isinstance(got, (list, tuple)):
        if len(want) == len(got) and all(not _diff(a, b, "") for a, b in zip(want, got)):
            return []
    elif isinstance(want, float) or isinstance(got, float):
        if want is not None and got is not None and np.isclose(want, got):
            return []
    elif want == got:
        return []
    return [f"{where}: requested {want!r}, suite2p used {got!r}"]
