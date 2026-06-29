#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simnibs_server.core.socket_client import SocketClient

log = logging.getLogger(__name__)


class MessageEmit:
    """Sends typed messages back to InVesalius via the relay."""

    def __init__(self, socket_client: "SocketClient") -> None:
        self._client = socket_client

    def progress(self, message: str, percent: int) -> None:
        self._client.emit("SimNIBS progress", {"message": message, "percent": percent})

    def charm_done(self, m2m_dir: str) -> None:
        self._client.emit("Charm done", {"m2m_dir": m2m_dir})

    def simulation_done(self, result_msh: str) -> None:
        self._client.emit("SimNIBS efield loaded", {"result_msh": result_msh})

    def error(self, message: str) -> None:
        log.error(message)
        self._client.emit("SimNIBS error", {"message": message})
