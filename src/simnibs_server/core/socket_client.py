#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import threading
import time
from queue import Queue
from typing import Optional

import socketio

logging.getLogger("socketio").setLevel(logging.WARNING)
logging.getLogger("engineio").setLevel(logging.WARNING)

log = logging.getLogger(__name__)


class SocketClient:
    """
    Thread-safe Socket.IO client for the SimNIBS processing server.

    Parameters
    ----------
    remote_host:
        URL of the relay server, e.g. ``"http://127.0.0.1:5000"``.
    """

    def __init__(self, remote_host: str):
        self._remote_host = remote_host
        self._buffer: Queue = Queue()
        self._connected = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sio: Optional[socketio.Client] = None

    def connect(self) -> None:
        """Start the background thread and begin connecting to the relay."""
        if self._thread and self._thread.is_alive():
            log.warning("SocketClient: already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="SimNIBS-SocketClient",
        )
        self._thread.start()
        log.info("SocketClient: background thread started")

    def disconnect(self) -> None:
        """Stop the background thread and close the connection."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def emit(self, topic: str, data: dict) -> bool:
        """
        Send a message to InVesalius via the relay.

        The message is wrapped in the InVesalius pubsub envelope
        ``{"topic": topic, "data": data}`` and emitted on ``from_simnibs``.

        Parameters
        ----------
        topic:
            InVesalius pubsub topic string, e.g. ``"Charm done"``.
        data:
            Payload dict, e.g. ``{"m2m_dir": "/data/m2m_ernie"}``.

        Returns
        -------
        bool
            True if the emit succeeded, False if not connected.
        """
        if not self._sio or not self._connected:
            log.warning("SocketClient: cannot emit '%s' — not connected", topic)
            return False
        try:
            self._sio.emit("from_simnibs", {"topic": topic, "data": data})
            return True
        except Exception as exc:
            log.error("SocketClient: emit failed for '%s': %s", topic, exc)
            return False

    def get_buffer(self) -> list[dict]:
        """
        Drain and return all buffered messages received since the last call.

        Each message is the raw dict received on ``to_simnibs``, expected to
        have the shape ``{"topic": str, "data": dict}``.

        Returns
        -------
        list[dict]
            May be empty if no messages have arrived.
        """
        messages = []
        while not self._buffer.empty():
            try:
                messages.append(self._buffer.get_nowait())
            except Exception:
                break
        return messages

    def clear_buffer(self) -> None:
        """Discard all buffered messages."""
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except Exception:
                break

    @property
    def is_connected(self) -> bool:
        return self._connected

  
    def _run(self) -> None:
        self._sio = socketio.Client(
            logger=False,
            engineio_logger=False,
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=1,
            reconnection_delay_max=10,
        )

        @self._sio.event
        def connect():
            self._connected = True
            log.info("SocketClient: connected to %s", self._remote_host)

        @self._sio.event
        def disconnect():
            self._connected = False
            log.warning("SocketClient: disconnected (will auto-reconnect)")

        @self._sio.event
        def connect_error(data):
            self._connected = False
            log.warning("SocketClient: connection error: %s", data)

        @self._sio.on("to_simnibs")
        def on_to_simnibs(msg):
            """
            Receives messages forwarded by the relay from InVesalius.
            Expected shape: {"topic": str, "data": dict}
            """
            if isinstance(msg, dict) and "topic" in msg:
                self._buffer.put(msg)
            else:
                log.warning("SocketClient: unexpected message shape: %s", msg)

        while not self._stop_event.is_set():
            try:
                if not self._sio.connected:
                    log.info("SocketClient: connecting to %s…", self._remote_host)
                    self._sio.connect(
                        self._remote_host,
                        wait_timeout=5,
                        transports=["websocket", "polling"],
                    )
                while self._sio.connected and not self._stop_event.is_set():
                    time.sleep(0.5)
            except Exception as exc:
                if not self._stop_event.is_set():
                    log.warning("SocketClient: %s — retrying in 2 s", exc)
                    time.sleep(2)

        if self._sio and self._sio.connected:
            self._sio.disconnect()
        log.info("SocketClient: stopped")
