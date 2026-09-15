"""Small, credential-safe diagnostics for direct model service connections."""

from __future__ import annotations

import errno
import os
import re

from typing import Any
from urllib.parse import urlsplit, urlunsplit


def safe_endpoint(url: str) -> str:
    """Keep routing information but never credentials or query parameters."""
    parts = urlsplit(str(url))
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port is not None else ""
    return urlunsplit((parts.scheme, host + port, parts.path, "", ""))


def safe_error_text(text: str) -> str:
    """Redact known keys and common URL/header secrets before saving errors."""
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "AZURE_OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[redacted]")
    text = re.sub(r"(?i)(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", text)
    text = re.sub(r"(?i)(bearer\s+)[\w.\-]+", r"\1[redacted]", text)
    text = re.sub(
        r"(?i)(api[_-]?key|token|password|secret)=([^&\s]+)", r"\1=[redacted]", text
    )
    return text[:600]


def connection_diagnostics(exc: BaseException) -> dict[str, Any]:
    """Expose bounded exception causes instead of only 'Connection error'."""
    chain, seen = [], set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(chain) < 6:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": safe_error_text(str(current)),
                "errno": getattr(current, "errno", None),
            }
        )
        current = current.__cause__ or current.__context__
    text = " ".join(r["type"] + " " + r["message"] for r in chain).lower()
    codes = {r["errno"] for r in chain}
    if errno.ECONNREFUSED in codes or 10061 in codes or "connection refused" in text:
        kind = "connection_refused"
    elif "proxy" in text:
        kind = "proxy_error"
    elif any(
        t in text
        for t in (
            "gaierror",
            "name or service not known",
            "name resolution",
            "getaddrinfo failed",
        )
    ):
        kind = "dns_error"
    elif any(t in text for t in ("ssl", "certificate", "tls")):
        kind = "tls_error"
    elif "timeout" in text or "timed out" in text:
        kind = "timeout"
    elif "connection reset" in text or "remoteprotocolerror" in text:
        kind = "connection_interrupted"
    else:
        kind = (
            "unresolved_transport_error"
            if any(term in text for term in ("connection", "connecterror", "transport"))
            else "non_transport_error"
        )
    return {"kind": kind, "exception_chain": chain}
