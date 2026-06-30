#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import sys

import nest_asyncio
import socketio
import uvicorn

nest_asyncio.apply()

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

sio = socketio.AsyncServer(
    async_mode="asgi",
    max_http_buffer_size=500_000_000,
    cors_allowed_origins="*",
)
app = socketio.ASGIApp(sio)


@sio.event
async def from_neuronavigation(sid, msg):
    await sio.emit("to_simnibs", msg)


@sio.event
async def from_simnibs(sid, msg):
    await sio.emit("to_neuronavigation", msg)


@sio.event
async def from_robot(sid, msg):
    await sio.emit("to_neuronavigation", msg)


@sio.event
async def connect(sid, environ):
    print(f"[relay] client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"[relay] client disconnected: {sid}")


if __name__ == "__main__":
    print(f"[relay] starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port, loop="asyncio")
