#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import Callable, Optional

log = logging.getLogger(__name__)

ProgressCb = Callable[[str, int], None]
DoneCb = Callable[[bool, Optional[str], Optional[str]], None]


def run(
    m2m_dir: str,
    out_dir: str,
    coil: str,
    didt: float,
    matsimnibs: list,
    progress_cb: ProgressCb,
    done_cb: DoneCb,
) -> None:
    """Run a TMS e-field simulation using the SimNIBS Python API.

    Called in a daemon thread by the message handler.

    Parameters
    ----------
    m2m_dir:
        Path to the m2m_subjectID folder produced by CHARM.
    out_dir:
        Directory where simulation results will be written.
    coil:
        Path to a SimNIBS coil file (.tcd or .ccd).
    didt:
        Rate of change of coil current (A/s), e.g. 1e6.
    matsimnibs:
        4×4 coil-to-MRI transformation matrix as a nested list.
    progress_cb:
        Called with (message, percent) as the run progresses.
    done_cb:
        Called with (success, error_msg, result_msh_path) when finished.
    """
    try:
        import simnibs
        from simnibs import sim_struct
    except ImportError as exc:
        done_cb(False, f"Could not import simnibs: {exc}", None)
        return

    os.makedirs(out_dir, exist_ok=True)
    progress_cb("Building simulation session…", 5)

    try:
        import numpy as np

        s = sim_struct.SESSION()
        s.subpath = m2m_dir
        s.pathfem = out_dir

        tms = s.add_tmslist()
        tms.fnamecoil = coil

        pos = tms.add_position()
        pos.matsimnibs = np.array(matsimnibs)
        pos.didt = didt

        progress_cb("Running SimNIBS simulation…", 10)
        simnibs.run_simnibs(s)

    except Exception as exc:
        done_cb(False, str(exc), None)
        return

    # Locate the output mesh (SimNIBS writes subject_TMS_*.msh)
    result_msh = _find_result_msh(out_dir)
    if result_msh is None:
        done_cb(False, f"Simulation finished but no result .msh found in {out_dir}", None)
        return

    # InVesalius cannot read gmsh .msh files, so convert the E-field to a VTK
    # surface it can render directly. On failure, fall back to the raw mesh.
    progress_cb("Preparing E-field surface…", 95)
    result_path = result_msh
    try:
        from src.simnibs_server.processing import msh_to_surface

        surface_path, vmin, vmax = msh_to_surface.convert(result_msh)
        log.info("E-field surface written: %s (range %.3g..%.3g)", surface_path, vmin, vmax)
        result_path = surface_path
    except Exception as exc:
        log.exception("Could not convert result mesh to a VTK surface: %s", exc)

    progress_cb("Simulation complete.", 100)
    done_cb(True, None, result_path)


def _find_result_msh(out_dir: str) -> Optional[str]:
    """Return the first TMS result mesh found in out_dir, or None."""
    for name in os.listdir(out_dir):
        if name.endswith(".msh") and "_TMS_" in name:
            return os.path.join(out_dir, name)
    return None
