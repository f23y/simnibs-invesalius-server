#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  python server.py [host] port

  Default host: 127.0.0.1
  Port:         must match the relay_server.py port (e.g. 5000)

InVesalius must be started with:
  python app.py --remote-host http://localhost:5000
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("simnibs-server")

_DEFAULT_HOST = "127.0.0.1"

if len(sys.argv) == 3:
    host = sys.argv[1]
    port = int(sys.argv[2])
elif len(sys.argv) == 2:
    host = _DEFAULT_HOST
    port = int(sys.argv[1])
else:
    print(f"Usage: python {sys.argv[0]} [host] port")
    sys.exit(1)

relay_url = f"http://{host}:{port}"

from src.simnibs_server.core.socket_client import SocketClient
from src.simnibs_server.core.message_emit import MessageEmit
from src.simnibs_server.core.message_handler import MessageHandler

socket_client = SocketClient(relay_url)
message_emit = MessageEmit(socket_client)
message_handler = MessageHandler(socket_client, message_emit)

if __name__ == "__main__":
    log.info("Connecting to relay at %s", relay_url)
    socket_client.connect()

    try:
        while True:
            message_handler.process_messages()
            time.sleep(0.05)
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        socket_client.disconnect()
