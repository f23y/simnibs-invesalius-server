#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.simnibs_server.core.socket_client import SocketClient
    from src.simnibs_server.core.message_emit import MessageEmit

log = logging.getLogger(__name__)


class MessageHandler:
    def __init__(self, socket_client: "SocketClient", message_emit: "MessageEmit") -> None:
        self._client = socket_client
        self._emit = message_emit
        self._charm_runner = None

    def process_messages(self) -> None:
        """Drain the socket buffer and dispatch each message."""
        for msg in self._client.get_buffer():
            topic = msg.get("topic", "")
            data = msg.get("data", {})
            self._dispatch(topic, data)

    def _dispatch(self, topic: str, data) -> None:
        match topic:
            case "From Neuronavigation: Send coil pose":
                self._on_coil_pose(data)
            case "SimNIBS: Run charm":
                self._on_run_charm(data)
            case "SimNIBS: Cancel charm":
                self._on_cancel_charm()
            case "SimNIBS: Run simulation":
                self._on_run_simulation(data)
            case "SimNIBS: Cancel simulation":
                self._on_cancel_simulation()

    def _on_coil_pose(self, data: dict) -> None:
        # InVesalius owns the coil-pose -> matsimnibs
        log.debug("Coil pose received from navigation (unused server-side): %s", data.get("coord"))

    def _on_run_charm(self, data: dict) -> None:
        from src.simnibs_server.processing.charm_runner import CharmRunner

        subject_dir = data.get("subject_dir", "")
        mri_files = data.get("mri_files", [])
        forcerun = bool(data.get("forcerun", False))
        force_qform = bool(data.get("force_qform", False))
        force_sform = bool(data.get("force_sform", False))

        if not subject_dir or not mri_files:
            self._emit.error("SimNIBS: Run charm — missing subject_dir or mri_files")
            return

        self._emit.progress("Starting CHARM…", 0)
        self._charm_runner = CharmRunner()
        self._charm_runner.start(
            subject_dir=subject_dir,
            mri_files=mri_files,
            forcerun=forcerun,
            force_qform=force_qform,
            force_sform=force_sform,
            progress_cb=lambda msg, pct: self._emit.progress(msg, pct),
            done_cb=self._on_charm_done,
        )

    def _on_cancel_charm(self) -> None:
        log.info("Cancel charm requested")
        self._emit.progress("Cancellation requested — charm is running and cannot be "
                            "interrupted cleanly via the Python API.", 0)

    def _on_charm_done(self, success: bool, error: Optional[str], subject_dir: Optional[str]) -> None:
        self._charm_runner = None
        if success:
            self._emit.charm_done(subject_dir)
            self._open_m2m_nifti(subject_dir)
        else:
            self._emit.error(error or "charm failed")

    def _on_run_simulation(self, data: dict) -> None:
        from src.simnibs_server.processing import sim_runner

        m2m_dir = data.get("m2m_dir", "")
        out_dir = data.get("output_dir", "")
        coil = data.get("coil", "")
        didt = float(data.get("didt", 1_000_000.0))
        matsimnibs = data.get("matsimnibs")

        if not m2m_dir or not out_dir or not coil:
            self._emit.error(
                "SimNIBS: Run simulation — missing m2m_dir, output_dir, or coil"
            )
            return

        if matsimnibs is None:
            self._emit.error(
                "SimNIBS: Run simulation — no matsimnibs in request. InVesalius must "
                "send the coil pose matrix (a coil pose from navigation is required)."
            )
            return

        self._emit.progress("Starting simulation…", 0)
        threading.Thread(
            target=sim_runner.run,
            args=(m2m_dir, out_dir, coil, didt, matsimnibs,
                  lambda msg, pct: self._emit.progress(msg, pct),
                  self._on_sim_done),
            daemon=True,
            name="sim-runner",
        ).start()

    def _on_cancel_simulation(self) -> None:
        log.info("Cancel simulation requested")
        self._emit.progress("Cancellation requested — simulation cannot be interrupted "
                            "cleanly via the Python API.", 0)

    def _on_sim_done(self, success: bool, error: Optional[str], result_msh: Optional[str]) -> None:
        if success:
            self._emit.simulation_done(result_msh)
        else:
            self._emit.error(error or "simulation failed")

    def _open_m2m_nifti(self, subject_dir: Optional[str]) -> None:
        """After charm completes, tell InVesalius to open the tissue-label NIfTI."""
        if not subject_dir:
            return
        for name in ("final_tissues.nii.gz", "final_tissues.nii"):
            nifti_path = os.path.join(subject_dir, name)
            if os.path.isfile(nifti_path):
                self._emit.open_nifti(nifti_path)
                return
        log.warning("charm done but no tissue NIfTI found in %s", subject_dir)
