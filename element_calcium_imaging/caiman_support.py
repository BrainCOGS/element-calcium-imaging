"""CaImAn processing is disabled in this fork.

element_interface's CaImAn support (``run_caiman``, ``caiman_loader``) is written
against DataJoint's CaImAn fork: ``CNMF.fit_file(output_dir=..., return_mc=True)``
and a ``fit_file`` that also re-fits, evaluates components, computes dF/F and saves
the ``.hdf5``. That fork no longer runs in a current environment (it uses names
numpy 2 removed, e.g. ``np.Inf``, ``np.trapz``, ``np.string_``). Upstream CaImAn runs
on numpy 2, but since 2025 its ``fit_file`` only motion-corrects and fits.

Every CaImAn entry point in the table modules calls ``caiman_unsupported()`` first;
the legacy implementation is kept, unreachable, right after each call so the next
person can see what it did. Re-enabling CaImAn means porting ``run_caiman`` to
upstream CaImAn's API, pinning a tagged upstream release, and removing the guards.
"""

CAIMAN_UNSUPPORTED = (
    "CaImAn processing is disabled: element_interface's CaImAn support needs "
    "DataJoint's CaImAn fork, which does not run with numpy 2, and upstream CaImAn's "
    "API has changed. Use suite2p, or port element_interface.run_caiman to upstream "
    "CaImAn (see element_calcium_imaging.caiman_support)."
)


def caiman_unsupported():
    """Raise for any CaImAn task: generation, triggering or loading of results."""
    raise NotImplementedError(CAIMAN_UNSUPPORTED)
