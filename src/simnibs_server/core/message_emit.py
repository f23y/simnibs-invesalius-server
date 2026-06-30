#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simnibs_server.core.socket_client import SocketClient

log = logging.getLogger(__name__)


class MessageEmit:
    """Sends typed messages back to InVesalius and the dashboard via the relay."""

    def __init__(self, socket_client: "SocketClient") -> None:
        self._client = socket_client

    # --- dashboard notifications -------------------------------------------

    def progress(self, message: str, percent: int) -> None:
        self._client.emit("SimNIBS progress", {"message": message, "percent": percent})

    def charm_done(self, m2m_dir: str) -> None:
        self._client.emit("Charm done", {"m2m_dir": m2m_dir})

    def simulation_done(self, result_msh: str) -> None:
        self._client.emit("SimNIBS efield loaded", {"result_msh": result_msh})

    def error(self, message: str) -> None:
        log.error(message)
        self._client.emit("SimNIBS error", {"message": message})

    # --- InVesalius viewer commands ----------------------------------------

    def open_nifti(self, filepath: str) -> None:
        self._client.emit("Open other files", {"filepath": filepath})

    def import_nifti_mask(self, filepath: str, mask_name: str = "") -> None:
        self._client.emit("Import Nifti mask", {"filepath": filepath, "mask_name": mask_name})

    def create_surface(self, surface_parameters: dict) -> None:
        self._client.emit("Create surface from index", {"surface_parameters": surface_parameters})
