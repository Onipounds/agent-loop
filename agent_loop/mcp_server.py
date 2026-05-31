"""Expose agent-loop's tools as an MCP server.

This lets any MCP-capable client (e.g. an IDE assistant) call the same tools
the agent uses. It exposes the *tools*, not the whole agent, on purpose:
small, composable, and easy to reason about.

Run:  python -m agent_loop.mcp_server
Requires the optional 'mcp' dependency (pip install 'agent-loop[mcp]').
The import is lazy so the rest of the package works without it.
"""
from __future__ import annotations

from .tools import calculate, query_telemetry, check_threshold


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - exercised only with extra installed
        raise SystemExit(
            "The 'mcp' package is required. Install with: pip install 'agent-loop[mcp]'"
        ) from e

    server = FastMCP("agent-loop-tools")

    @server.tool()
    def calc(expression: str) -> dict:
        """Evaluate an arithmetic/statistics expression."""
        return calculate(expression)

    @server.tool()
    def telemetry(device: str, metric: str, window: int = 60) -> dict:
        """Fetch recent sensor readings for a device and metric."""
        return query_telemetry(device=device, metric=metric, window=window)

    @server.tool()
    def threshold(value: float, limit: float, kind: str = "max") -> dict:
        """Check whether a value is within a limit."""
        return check_threshold(value=value, limit=limit, kind=kind)

    return server


def main() -> None:  # pragma: no cover - needs the mcp runtime
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
