"""Realtime alert broadcasting over WebSocket connections.

A single module-level :class:`ConnectionManager` fan-outs every
``/events/ingest`` payload to all currently connected ``/ws/alerts``
clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Track WebSocket clients and broadcast JSON messages to all of them."""

    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def active_connections(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and remember a new client."""
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("Alerts WebSocket client connected (%d active)",
                    len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Forget a client (idempotent)."""
        if websocket in self._connections:
            self._connections.discard(websocket)
            logger.info("Alerts WebSocket client disconnected (%d active)",
                        len(self._connections))

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Send ``message`` to every connected client; returns recipients count.

        A failure to send (client went away mid-broadcast) removes and skips
        that client rather than breaking the whole fan-out.
        """
        payload = json.dumps(message, default=str)
        async with self._lock:
            targets: List[WebSocket] = list(self._connections)
        dead: List[WebSocket] = []
        for websocket in targets:
            try:
                await websocket.send_text(payload)
            except Exception:
                logger.debug("dropping dead alerts WebSocket", exc_info=True)
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)
        return len(targets) - len(dead)


#: shared broadcaster used by the alerts router
broadcaster = ConnectionManager()