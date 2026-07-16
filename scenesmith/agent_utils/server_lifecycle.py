"""Reliable lifecycle wrapper for Flask apps embedded in experiment processes."""

from __future__ import annotations

import threading
from typing import Any

from werkzeug.serving import BaseWSGIServer, make_server


class ManagedWSGIServer:
    """Own a Werkzeug server object so shutdown does not depend on an HTTP hook."""

    def __init__(self, app: Any, host: str, port: int, *, threaded: bool = True):
        self._server: BaseWSGIServer = make_server(
            host,
            port,
            app,
            threaded=threaded,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"wsgi-{host}-{port}",
            daemon=False,
        )
        self._started = False
        self._closed = False

    @property
    def thread(self) -> threading.Thread:
        return self._thread

    def start(self) -> threading.Thread:
        if self._closed:
            raise RuntimeError("Cannot restart a closed WSGI server")
        if not self._started:
            self._thread.start()
            self._started = True
        return self._thread

    def stop(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        try:
            if self._started:
                self._server.shutdown()
                if self._thread.is_alive():
                    self._thread.join(timeout=timeout)
        finally:
            self._server.server_close()
            self._started = False
            self._closed = True
