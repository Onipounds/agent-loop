"""Tool registry and the concrete tools the agent can call.

A tool is a plain Python function plus a JSON schema describing its inputs.
The registry maps a name to both, executes calls, and produces the schema
list the model needs. Adding a tool is one ``register(...)`` call.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    schema: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, func: Callable[..., Any],
                 properties: dict[str, Any], required: list[str]) -> None:
        schema = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        self._tools[name] = Tool(name, description, func, schema)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    def execute(self, name: str, tool_input: dict[str, Any]) -> Any:
        if name not in self._tools:
            return {"error": f"unknown tool: {name}"}
        try:
            return self._tools[name].func(**tool_input)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # tools must never crash the loop
            return {"error": f"{type(e).__name__}: {e}"}


# --- Tool implementations ---------------------------------------------------

def _varargs(fn):
    """Allow fn(1,2,3) as well as fn([1,2,3])."""
    def wrapper(*args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        return fn(args)
    return wrapper


_SAFE_FUNCS = {
    "mean": _varargs(statistics.fmean),
    "max": _varargs(max),
    "min": _varargs(min),
    "sum": _varargs(sum),
    "len": _varargs(len),
    "stdev": _varargs(lambda xs: statistics.pstdev(xs) if len(xs) > 1 else 0.0),
    "round": round, "abs": abs, "sqrt": math.sqrt,
}


def calculate(expression: str) -> dict[str, Any]:
    """Evaluate a small arithmetic/stat expression safely (no builtins).

    Returns an ``{"error": ...}`` dict on any failure rather than raising, so
    a bad expression feeds back into the agent loop instead of crashing it.
    """
    try:
        allowed = dict(_SAFE_FUNCS)
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307 - locked-down namespace
        return {"result": result}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _load_telemetry() -> dict[str, Any]:
    path = DATA_DIR / "telemetry.json"
    with open(path) as f:
        return json.load(f)


def query_telemetry(device: str, metric: str, window: int = 60) -> dict[str, Any]:
    """Return up to ``window`` recent readings for a device+metric from the bundled dataset."""
    data = _load_telemetry()
    key = f"{device}:{metric}"
    if key not in data:
        return {"error": f"no data for {key}", "available": list(data)}
    readings = data[key][-window:]
    return {"device": device, "metric": metric, "count": len(readings), "values": readings}


def check_threshold(value: float, limit: float, kind: str = "max") -> dict[str, Any]:
    """Check a value against a limit. kind='max' => ok if value <= limit; 'min' => ok if value >= limit."""
    if kind == "max":
        ok = value <= limit
    elif kind == "min":
        ok = value >= limit
    else:
        return {"error": f"unknown kind: {kind} (use 'max' or 'min')"}
    return {"value": value, "limit": limit, "kind": kind, "within_limit": ok}


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "calculate",
        "Evaluate an arithmetic or statistics expression. "
        "Supports mean, max, min, sum, stdev, sqrt, round, abs, e.g. 'mean(70,72,71)'.",
        calculate,
        {"expression": {"type": "string", "description": "expression to evaluate"}},
        ["expression"],
    )
    reg.register(
        "query_telemetry",
        "Fetch recent sensor readings for a device and metric.",
        query_telemetry,
        {
            "device": {"type": "string", "description": "device name, e.g. 'pump-3'"},
            "metric": {"type": "string", "description": "metric name, e.g. 'temperature'"},
            "window": {"type": "integer", "description": "max number of recent readings"},
        },
        ["device", "metric"],
    )
    reg.register(
        "check_threshold",
        "Check whether a value is within a limit.",
        check_threshold,
        {
            "value": {"type": "number"},
            "limit": {"type": "number"},
            "kind": {"type": "string", "description": "'max' or 'min'"},
        },
        ["value", "limit"],
    )
    return reg
